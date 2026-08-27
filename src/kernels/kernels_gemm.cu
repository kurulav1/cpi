// Batched fp16 GEMM: the sequence-mode counterpart of the row-major GEMV.
//
//   y[t, m] = sum_k w[m, k] * x[t, k]
//
// Same weight layout as launch_rowmajor_half_gemv_f16 ([out_features, in_features],
// row-major), so a plan can bind one weight and run it either per-token or over a
// whole sequence.
//
// Why this exists: text prefill sidesteps fp16 GEMM entirely by quantising the
// activations and using the int8 dp4a path. The vision tower runs fp16 weights over
// a few hundred patches at once, so it needs a real fp16 GEMM.
//
// NOTE ON NUMERICS: a tiled GEMM sums K in a different order than the GEMV, so the
// two do not agree bit-for-bit. The executor therefore keeps dispatching single-token
// work to the GEMV; sequence mode must not perturb decode output.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "runtime/kernels.cuh"

namespace kernels {
namespace {

__device__ __forceinline__ float warp_sum_f(float v) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

constexpr int kTile = 32;

__global__ void rowmajor_half_gemm_f16_kernel(const half* __restrict__ w,
                                              const half* __restrict__ x, half* __restrict__ y,
                                              int out_features, int in_features, int tokens,
                                              float in_min, float in_max, float out_min,
                                              float out_max) {
  // +1 column of padding: the compute step reads Ws down a column, which would
  // otherwise hit the same shared-memory bank for every thread in a warp.
  __shared__ half Xs[kTile][kTile + 1];
  __shared__ half Ws[kTile][kTile + 1];

  const int tx = threadIdx.x;  // output feature within the tile
  const int ty = threadIdx.y;  // token within the tile
  const int m = blockIdx.x * kTile + tx;
  const int t = blockIdx.y * kTile + ty;

  float acc = 0.0f;
  for (int k0 = 0; k0 < in_features; k0 += kTile) {
    const int kx = k0 + tx;
    // Each thread stages one element of X and one of W. Out-of-range lanes stage
    // zeros so the multiply-accumulate below needs no bounds check.
    const int xt = blockIdx.y * kTile + ty;
    // Clipped projections clamp the ACTIVATION as it is read. Defaults are +-inf, so
    // this is a no-op for every model that does not use them.
    Xs[ty][tx] =
        (xt < tokens && kx < in_features)
            ? __float2half(fminf(
                  fmaxf(__half2float(x[static_cast<std::size_t>(xt) * in_features + kx]), in_min),
                  in_max))
            : __float2half(0.0f);
    const int wm = blockIdx.x * kTile + ty;
    Ws[ty][tx] = (wm < out_features && kx < in_features)
                     ? w[static_cast<std::size_t>(wm) * in_features + kx]
                     : __float2half(0.0f);
    __syncthreads();

    for (int kk = 0; kk < kTile; ++kk) {
      acc += __half2float(Xs[ty][kk]) * __half2float(Ws[tx][kk]);
    }
    __syncthreads();
  }

  if (t < tokens && m < out_features) {
    y[static_cast<std::size_t>(t) * out_features + m] =
        __float2half(fminf(fmaxf(acc, out_min), out_max));
  }
}

}  // namespace

void launch_rowmajor_half_gemm_f16(const half* w, const half* x, half* y, int out_features,
                                   int in_features, int tokens, cudaStream_t stream, float in_min,
                                   float in_max, float out_min, float out_max) {
  if (out_features <= 0 || in_features <= 0 || tokens <= 0) {
    return;
  }
  const dim3 threads(kTile, kTile);
  const dim3 grid(static_cast<unsigned>((out_features + kTile - 1) / kTile),
                  static_cast<unsigned>((tokens + kTile - 1) / kTile));
  rowmajor_half_gemm_f16_kernel<<<grid, threads, 0, stream>>>(
      w, x, y, out_features, in_features, tokens, in_min, in_max, out_min, out_max);
}

// out[t][i] = gelu(a[t][i]) * b[t*b_stride + i]
//
// The per-layer-input gate multiplies by a slice of a WIDER per-token tensor (the PLE
// table is [tokens][num_layers * ple], and layer L wants its own ple-wide window), so b
// advances by a different stride than a and out. At one token this is just the flat
// kernel; over a sequence it is not, and a flat kernel silently reads the wrong rows.
__global__ void gelu_mul_strided_kernel(const half* a, const half* b, half* out, int n, int tokens,
                                        int b_stride) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y;
  if (i >= n || t >= tokens) {
    return;
  }
  const float g = __half2float(a[static_cast<std::size_t>(t) * n + i]);
  const float u = __half2float(b[static_cast<std::size_t>(t) * b_stride + i]);
  const float x3 = g * g * g;
  const float gelu = 0.5f * g * (1.0f + tanhf(0.7978845608028654f * (g + 0.044715f * x3)));
  out[static_cast<std::size_t>(t) * n + i] = __float2half(gelu * u);
}

void launch_gelu_mul_strided(const half* a, const half* b, half* out, int n, int tokens,
                             int b_stride, cudaStream_t stream) {
  constexpr int kThreads = 256;
  const dim3 grid(static_cast<unsigned>((n + kThreads - 1) / kThreads),
                  static_cast<unsigned>(tokens));
  gelu_mul_strided_kernel<<<grid, kThreads, 0, stream>>>(a, b, out, n, tokens, b_stride);
}

// Fused SwiGLU projection: gate-GEMV + up-GEMV + silu_mul in one kernel.
//
//   out[i] = silu( dot(w_gate[i], x) ) * dot(w_up[i], x)
//
// Why: at batch 1 each kernel costs a fixed ~2.7 us of scheduling no matter how little
// it does, and a 0.5B model spends over half its per-token budget on that tax rather
// than on memory traffic. gate/up/silu were three kernels touching the same x; now they
// are one. No weight relayout; the kernel simply takes both weight pointers.
//
// byte-identical by construction: g and u are rounded to fp16 before silu(g)*u, exactly
// as the three-kernel version does when it stores them. Accumulating in fp32 straight
// through would be more accurate and would silently change every model's output.
template <int WarpsPerBlock, int TilePairs>
__global__ void swiglu_gemv_f16_kernel(const half* __restrict__ w_gate,
                                       const half* __restrict__ w_up, const half* __restrict__ x,
                                       half* __restrict__ out, int inter, int in_features) {
  __shared__ half2 x_tile[TilePairs];
  const int warp_id = threadIdx.x / warpSize;
  const int lane = threadIdx.x & (warpSize - 1);
  const int row = blockIdx.x * WarpsPerBlock + warp_id;

  const bool active = row < inter;
  const half* g_row = active ? w_gate + static_cast<std::size_t>(row) * in_features : nullptr;
  const half* u_row = active ? w_up + static_cast<std::size_t>(row) * in_features : nullptr;

  float gacc = 0.0f, uacc = 0.0f;
  if ((in_features & 1) == 0) {
    const int pairs = in_features / 2;
    const half2* x2 = reinterpret_cast<const half2*>(x);
    const half2* g2 = active ? reinterpret_cast<const half2*>(g_row) : nullptr;
    const half2* u2 = active ? reinterpret_cast<const half2*>(u_row) : nullptr;
    for (int tile_base = 0; tile_base < pairs; tile_base += TilePairs) {
      const int tile_count = min(TilePairs, pairs - tile_base);
      for (int idx = threadIdx.x; idx < tile_count; idx += blockDim.x) {
        x_tile[idx] = x2[tile_base + idx];
      }
      __syncthreads();
      if (active) {
        for (int idx = lane; idx < tile_count; idx += warpSize) {
          const float2 xv = __half22float2(x_tile[idx]);
          const float2 gv = __half22float2(g2[tile_base + idx]);
          const float2 uv = __half22float2(u2[tile_base + idx]);
          gacc += gv.x * xv.x + gv.y * xv.y;
          uacc += uv.x * xv.x + uv.y * xv.y;
        }
      }
      __syncthreads();
    }
  } else if (active) {
    for (int col = lane; col < in_features; col += warpSize) {
      const float xv = __half2float(x[col]);
      gacc += __half2float(g_row[col]) * xv;
      uacc += __half2float(u_row[col]) * xv;
    }
  }

  gacc = warp_sum_f(gacc);
  uacc = warp_sum_f(uacc);
  if (active && lane == 0) {
    // Round to fp16 first; this is what the unfused path stores between kernels.
    const float g = __half2float(__float2half(gacc));
    const float u = __half2float(__float2half(uacc));
    const float silu = g / (1.0f + __expf(-g));
    out[row] = __float2half(silu * u);
  }
}

void launch_swiglu_gemv_f16(const half* w_gate, const half* w_up, const half* x, half* out,
                            int inter, int in_features, cudaStream_t stream) {
  constexpr int kWarps = 4;
  constexpr int kTile = 128;
  const int blocks = (inter + kWarps - 1) / kWarps;
  swiglu_gemv_f16_kernel<kWarps, kTile>
      <<<blocks, kWarps * 32, 0, stream>>>(w_gate, w_up, x, out, inter, in_features);
}

// Split-K GEMV: one BLOCK per output row, W warps each covering a slice of the input,
// reduced in shared memory.
//
// The existing GEMV gives one WARP per output row, so its parallelism is set by the row
// count. That is fine for a 151936-row LM head (4.8M threads) and useless for a 896-row
// o_proj (29k threads on a GPU that runs ~348k); the machine sits 90% idle and memory
// latency is never hidden, which is why those GEMVs measured 6-8x off peak while the tall
// ones sat at peak.
//
// Splitting K instead makes parallelism = rows x W warps, which fills the GPU for exactly
// the short-and-wide shapes that were starving.
//
// NOTE ON NUMERICS: partial sums are combined across warps, so the reduction order differs
// from the one-warp kernel; results agree to fp32 rounding but are not bit-identical.
// That is why this is opt-in per call site, not a silent replacement.
template <int Warps>
__global__ void gemv_splitk_f16_kernel(const half* __restrict__ w, const half* __restrict__ x,
                                       half* __restrict__ y, int out_features, int in_features,
                                       half* __restrict__ residual) {
  __shared__ float partial[Warps];
  const int row = blockIdx.x;
  if (row >= out_features) {
    return;
  }
  const int warp = threadIdx.x / warpSize;
  const int lane = threadIdx.x & (warpSize - 1);
  const half* wrow = w + static_cast<std::size_t>(row) * in_features;

  float acc = 0.0f;
  if ((in_features & 1) == 0) {
    const int pairs = in_features / 2;
    const half2* w2 = reinterpret_cast<const half2*>(wrow);
    const half2* x2 = reinterpret_cast<const half2*>(x);
    // Warp `warp` strides the pairs; every thread keeps several loads in flight, which is
    // the whole point; latency hiding, not extra bandwidth.
    for (int i = warp * warpSize + lane; i < pairs; i += Warps * warpSize) {
      const float2 wv = __half22float2(w2[i]);
      const float2 xv = __half22float2(x2[i]);
      acc += wv.x * xv.x + wv.y * xv.y;
    }
  } else {
    for (int i = warp * warpSize + lane; i < in_features; i += Warps * warpSize) {
      acc += __half2float(wrow[i]) * __half2float(x[i]);
    }
  }

  acc = warp_sum_f(acc);
  if (lane == 0) {
    partial[warp] = acc;
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < Warps; ++i) {
      total += partial[i];
    }
    if (residual != nullptr) {
      residual[row] = __hadd(residual[row], __float2half(total));
    } else {
      y[row] = __float2half(total);
    }
  }
}

void launch_gemv_splitk_f16(const half* w, const half* x, half* y, int out_features,
                            int in_features, cudaStream_t stream, half* residual) {
  constexpr int kWarps = 8;
  gemv_splitk_f16_kernel<kWarps>
      <<<out_features, kWarps * 32, 0, stream>>>(w, x, y, out_features, in_features, residual);
}

}  // namespace kernels
