// Mixture-of-Experts feed-forward kernels.
//
// The experts are stored as one contiguous matrix each, not as N separate
// tensors: gate_up is [num_experts * 2 * inter, hidden] and down is
// [num_experts * hidden, inter], both row-major. Expert e's rows simply start at
// e * rows_per_expert, so an expert selection is a row offset; which means the
// existing quantiser and its group-wise scales apply unchanged, and the expert
// weights (which are ~90% of an MoE model) get quantised like any other matrix.
//
// The selected expert indices live on the device (the router wrote them there).
// Nothing is read back to the host between the router and the experts: a
// host round-trip would cost a sync per layer per token and would make the decode
// graph uncapturable. Each block reads its own expert index from device memory.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

#include "runtime/kernels.cuh"

namespace kernels {
namespace {

__device__ __forceinline__ float warp_sum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ float gelu_tanh(float x) {
  const float x3 = x * x * x;
  const float inner = 0.7978845608028654f * (x + 0.044715f * x3);
  return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ int load_signed_int4(const std::int8_t* row, int col) {
  const std::uint8_t byte = static_cast<std::uint8_t>(row[col >> 1]);
  const std::uint8_t nib = (col & 1) ? (byte >> 4) : (byte & 0x0f);
  return (nib & 0x8) ? static_cast<int>(nib) - 16 : static_cast<int>(nib);
}

// One weight element, dequantised. BITS: 0 = fp16, 8 = int8, 4 = packed int4.
//
// Per-row scales are expressed as group_shift = 31 / n_groups = 1, which makes
// (col >> 31) == 0 for every real column; so the per-row and group-wise cases
// use the same indexing with no branch in the inner loop.
template <int BITS>
struct Weight {
  const void* w;
  const float* scales;
  int in_features;
  int group_shift;
  int n_groups;

  __device__ __forceinline__ float at(std::size_t row, int col) const {
    if constexpr (BITS == 0) {
      const half* p = static_cast<const half*>(w);
      return __half2float(p[row * static_cast<std::size_t>(in_features) + col]);
    } else {
      const float s = scales[row * static_cast<std::size_t>(n_groups) + (col >> group_shift)];
      if constexpr (BITS == 8) {
        const std::int8_t* p = static_cast<const std::int8_t*>(w);
        return static_cast<float>(p[row * static_cast<std::size_t>(in_features) + col]) * s;
      } else {
        const std::int8_t* p = static_cast<const std::int8_t*>(w);
        const std::size_t stride = static_cast<std::size_t>((in_features + 1) / 2);
        return static_cast<float>(load_signed_int4(p + row * stride, col)) * s;
      }
    }
  }
};

// inter[k, r] = act(gate) * up, where gate and up are rows r and (inter + r) of the selected
// expert's fused gate_up matrix. one warp per output element (grid.x groups warps_per_block rows;
// grid.y = top_k); far more output rows in flight per SM than the old block-per-row form, and no
// shared-memory block reduction (just a warp shuffle). Memory-bound int4/int8 matvec, so occupancy
// is what matters.
template <int BITS>
__global__ void moe_gate_up_geglu_kernel(Weight<BITS> w, const half* x, const int* topk_idx,
                                         half* inter_out, int inter, int hidden, int top_k,
                                         bool use_gelu) {
  const int warps_per_block = blockDim.x / warpSize;
  const int r = blockIdx.x * warps_per_block + (threadIdx.x / warpSize);
  const int k = blockIdx.y;
  if (r >= inter || k >= top_k) return;
  const int lane = threadIdx.x & (warpSize - 1);

  const std::size_t base =
      static_cast<std::size_t>(topk_idx[k]) * static_cast<std::size_t>(2 * inter);
  const std::size_t gate_row = base + static_cast<std::size_t>(r);
  const std::size_t up_row = base + static_cast<std::size_t>(inter + r);

  float g = 0.0f, u = 0.0f;
  for (int c = lane; c < hidden; c += warpSize) {
    const float xv = __half2float(x[c]);
    g += w.at(gate_row, c) * xv;
    u += w.at(up_row, c) * xv;
  }
  g = warp_sum(g);
  u = warp_sum(u);
  if (lane == 0) {
    const float act = use_gelu ? gelu_tanh(g) : (g / (1.0f + __expf(-g)));  // GELU or SiLU
    inter_out[static_cast<std::size_t>(k) * inter + r] = __float2half(act * u);
  }
}

// y[r] = sum_k topk_weight[k] * dot(down[expert_k].row(r), inter[k]). one warp per output row: the
// warp walks all top_k experts (weighted sum, no atomics), reducing each expert's dot with a warp
// shuffle no shared memory, no __syncthreads. grid.x groups warps_per_block output rows.
template <int BITS>
__global__ void moe_down_accum_kernel(Weight<BITS> w, const half* inter_in, const int* topk_idx,
                                      const float* topk_weight, half* y, int hidden, int inter,
                                      int top_k) {
  const int warps_per_block = blockDim.x / warpSize;
  const int r = blockIdx.x * warps_per_block + (threadIdx.x / warpSize);
  if (r >= hidden) return;
  const int lane = threadIdx.x & (warpSize - 1);

  float acc = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    const std::size_t row =
        static_cast<std::size_t>(topk_idx[k]) * static_cast<std::size_t>(hidden) +
        static_cast<std::size_t>(r);
    const half* xk = inter_in + static_cast<std::size_t>(k) * inter;
    float d = 0.0f;
    for (int c = lane; c < inter; c += warpSize) {
      d += w.at(row, c) * __half2float(xk[c]);
    }
    d = warp_sum(d);  // full dot for expert k (valid on lane 0)
    acc += topk_weight[k] * d;
  }
  if (lane == 0) y[r] = __float2half(acc);
}

int threads_for(int n) {
  if (n >= 1024) return 256;
  if (n >= 256) return 128;
  return 64;
}

// Pack the encoding into the shape the kernels index with. Per-row scales become
// group_shift=31 / n_groups=1 so the inner loop is identical either way.
template <int BITS>
Weight<BITS> make_weight(const void* w, const float* scales, int in_features, int group) {
  Weight<BITS> out{};
  out.w = w;
  out.scales = scales;
  out.in_features = in_features;
  if (group > 0) {
    int shift = 0;
    while ((1 << shift) < group) ++shift;
    out.group_shift = shift;
    out.n_groups = quant_group_count(in_features, group);
  } else {
    out.group_shift = 31;
    out.n_groups = 1;
  }
  return out;
}

}  // namespace

void launch_moe_gate_up_geglu(const void* w, const float* scales, int qbits, int group,
                              const half* x, const int* topk_idx, half* inter_out, int inter,
                              int hidden, int top_k, cudaStream_t stream, bool use_gelu) {
  // Warp-per-output-row: 8 warps/block, grid.x groups rows, grid.y = top_k. No shared memory.
  constexpr int threads = 256;
  constexpr int wpb = threads / 32;
  const dim3 grid(static_cast<unsigned>((inter + wpb - 1) / wpb), static_cast<unsigned>(top_k));
  if (qbits == 4) {
    moe_gate_up_geglu_kernel<4>
        <<<grid, threads, 0, stream>>>(make_weight<4>(w, scales, hidden, group), x, topk_idx,
                                       inter_out, inter, hidden, top_k, use_gelu);
  } else if (qbits == 8) {
    moe_gate_up_geglu_kernel<8>
        <<<grid, threads, 0, stream>>>(make_weight<8>(w, scales, hidden, group), x, topk_idx,
                                       inter_out, inter, hidden, top_k, use_gelu);
  } else {
    moe_gate_up_geglu_kernel<0><<<grid, threads, 0, stream>>>(make_weight<0>(w, nullptr, hidden, 0),
                                                              x, topk_idx, inter_out, inter, hidden,
                                                              top_k, use_gelu);
  }
}

// dp4a MoE gate_up: warp per output row r (expert k = blockIdx.y). The activation xq is the
// perm8-int8 quantised XNorm (shared across all experts, quantised once by the caller); int4 expert
// weights are unpacked to int8 and dotted with dp4a. inter[k,r] = act(gate.xq) * (up.xq). Same
// weight packing + perm8 activation layout as the general int4 dp4a matvec (verified path); we just
// index the expert row.
__global__ void moe_gate_up_geglu_dp4a_kernel(const std::int8_t* __restrict__ wg,
                                              const float* __restrict__ sg,
                                              const std::int8_t* __restrict__ xq,
                                              const float* __restrict__ x_scale,
                                              const int* __restrict__ topk_idx, half* inter_out,
                                              int inter, int hidden, int top_k, int group_shift,
                                              int n_groups, bool use_gelu) {
  const int warps_per_block = blockDim.x / warpSize;
  const int r = blockIdx.x * warps_per_block + (threadIdx.x / warpSize);
  const int k = blockIdx.y;
  if (r >= inter || k >= top_k) return;
  const int lane = threadIdx.x & (warpSize - 1);
  const int packed_cols = hidden / 2;
  const int chunks = packed_cols / 16;  // uint4 = 16 bytes = 32 int4 weights
  const std::size_t base =
      static_cast<std::size_t>(topk_idx[k]) * static_cast<std::size_t>(2 * inter);
  const uint4* wgrow =
      reinterpret_cast<const uint4*>(wg + (base + r) * static_cast<std::size_t>(packed_cols));
  const uint4* wurow = reinterpret_cast<const uint4*>(
      wg + (base + inter + r) * static_cast<std::size_t>(packed_cols));
  const float* sgrow = sg + (base + r) * static_cast<std::size_t>(n_groups);
  const float* surow = sg + (base + inter + r) * static_cast<std::size_t>(n_groups);

  float gf = 0.0f, uf = 0.0f;
  for (int c = lane; c < chunks; c += warpSize) {
    const uint4 wgv = wgrow[c];
    const uint4 wuv = wurow[c];
    const uint4 xa = reinterpret_cast<const uint4*>(xq)[2 * c];
    const uint4 xb = reinterpret_cast<const uint4*>(xq)[2 * c + 1];
    const int xv[8] = {static_cast<int>(xa.x), static_cast<int>(xa.y), static_cast<int>(xa.z),
                       static_cast<int>(xa.w), static_cast<int>(xb.x), static_cast<int>(xb.y),
                       static_cast<int>(xb.z), static_cast<int>(xb.w)};
    const float xs = x_scale[c];
    const float gs = sgrow[(c * 32) >> group_shift] * xs;
    const float us = surow[(c * 32) >> group_shift] * xs;
    const unsigned* wgu = reinterpret_cast<const unsigned*>(&wgv);
    const unsigned* wuu = reinterpret_cast<const unsigned*>(&wuv);
    int g = 0, u = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const unsigned gw = wgu[i], uw = wuu[i];
      const int gwlo = __vsubss4(static_cast<int>((gw & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
      const int gwhi =
          __vsubss4(static_cast<int>(((gw >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
      const int uwlo = __vsubss4(static_cast<int>((uw & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
      const int uwhi =
          __vsubss4(static_cast<int>(((uw >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
      g = __dp4a(gwlo, xv[2 * i], g);
      g = __dp4a(gwhi, xv[2 * i + 1], g);
      u = __dp4a(uwlo, xv[2 * i], u);
      u = __dp4a(uwhi, xv[2 * i + 1], u);
    }
    gf += static_cast<float>(g) * gs;
    uf += static_cast<float>(u) * us;
  }
  gf = warp_sum(gf);
  uf = warp_sum(uf);
  if (lane == 0) {
    const float act = use_gelu ? gelu_tanh(gf) : (gf / (1.0f + __expf(-gf)));
    inter_out[static_cast<std::size_t>(k) * inter + r] = __float2half(act * uf);
  }
}

// group_shift such that (col >> shift) is the weight-group index; group must be a power of two.
static inline int moe_group_shift(int group) {
  int s = 0;
  while ((1 << s) < group) ++s;
  return s;
}

void launch_moe_gate_up_geglu_dp4a(const void* wg, const float* sg, const std::int8_t* xq,
                                   const float* x_scale, const int* topk_idx, half* inter_out,
                                   int inter, int hidden, int top_k, int group, cudaStream_t stream,
                                   bool use_gelu) {
  constexpr int threads = 256;
  constexpr int wpb = threads / 32;
  const dim3 grid(static_cast<unsigned>((inter + wpb - 1) / wpb), static_cast<unsigned>(top_k));
  moe_gate_up_geglu_dp4a_kernel<<<grid, threads, 0, stream>>>(
      static_cast<const std::int8_t*>(wg), sg, xq, x_scale, topk_idx, inter_out, inter, hidden,
      top_k, moe_group_shift(group), quant_group_count(hidden, group), use_gelu);
}

// dp4a MoE down: warp per output row r; walk the top_k experts, dp4a(down[expert_k].row(r),
// inter_int8[k]). The activation is per-expert inter[k], quantised by the caller (mt perm8: xq_all
// = [top_k, inter] int8, xs_all = [top_k, inter/32] scales). y[r] = sum_k topk_weight[k] * dot.
__global__ void moe_down_accum_dp4a_kernel(
    const std::int8_t* __restrict__ wd, const float* __restrict__ sd,
    const std::int8_t* __restrict__ xq_all, const float* __restrict__ xs_all,
    const int* __restrict__ topk_idx, const float* __restrict__ topk_weight, half* y, int hidden,
    int inter, int top_k, int group_shift, int n_groups) {
  const int warps_per_block = blockDim.x / warpSize;
  const int r = blockIdx.x * warps_per_block + (threadIdx.x / warpSize);
  if (r >= hidden) return;
  const int lane = threadIdx.x & (warpSize - 1);
  const int packed_cols = inter / 2;
  const int chunks = packed_cols / 16;
  const int xs_stride = inter / 32;

  float acc = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    const std::size_t down_row =
        static_cast<std::size_t>(topk_idx[k]) * static_cast<std::size_t>(hidden) +
        static_cast<std::size_t>(r);
    const uint4* wrow =
        reinterpret_cast<const uint4*>(wd + down_row * static_cast<std::size_t>(packed_cols));
    const float* srow = sd + down_row * static_cast<std::size_t>(n_groups);
    const std::int8_t* xq = xq_all + static_cast<std::size_t>(k) * inter;
    const float* xs = xs_all + static_cast<std::size_t>(k) * xs_stride;
    float d = 0.0f;
    for (int c = lane; c < chunks; c += warpSize) {
      const uint4 wv = wrow[c];
      const uint4 xa = reinterpret_cast<const uint4*>(xq)[2 * c];
      const uint4 xb = reinterpret_cast<const uint4*>(xq)[2 * c + 1];
      const int xv[8] = {static_cast<int>(xa.x), static_cast<int>(xa.y), static_cast<int>(xa.z),
                         static_cast<int>(xa.w), static_cast<int>(xb.x), static_cast<int>(xb.y),
                         static_cast<int>(xb.z), static_cast<int>(xb.w)};
      const float s = srow[(c * 32) >> group_shift] * xs[c];
      const unsigned* wu = reinterpret_cast<const unsigned*>(&wv);
      int dd = 0;
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const unsigned word = wu[i];
        const int wlo = __vsubss4(static_cast<int>((word & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
        const int whi =
            __vsubss4(static_cast<int>(((word >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
        dd = __dp4a(wlo, xv[2 * i], dd);
        dd = __dp4a(whi, xv[2 * i + 1], dd);
      }
      d += static_cast<float>(dd) * s;
    }
    d = warp_sum(d);
    acc += topk_weight[k] * d;
  }
  if (lane == 0) y[r] = __float2half(acc);
}

void launch_moe_down_accum_dp4a(const void* wd, const float* sd, const std::int8_t* xq_all,
                                const float* xs_all, const int* topk_idx, const float* topk_weight,
                                half* y, int hidden, int inter, int top_k, int group,
                                cudaStream_t stream) {
  constexpr int threads = 256;
  constexpr int wpb = threads / 32;
  const unsigned blocks = static_cast<unsigned>((hidden + wpb - 1) / wpb);
  moe_down_accum_dp4a_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const std::int8_t*>(wd), sd, xq_all, xs_all, topk_idx, topk_weight, y, hidden,
      inter, top_k, moe_group_shift(group), quant_group_count(inter, group));
}

void launch_moe_down_accum(const void* w, const float* scales, int qbits, int group,
                           const half* inter_in, const int* topk_idx, const float* topk_weight,
                           half* y, int hidden, int inter, int top_k, cudaStream_t stream) {
  // Warp-per-output-row: 8 warps/block, grid.x groups rows. No shared memory.
  constexpr int threads = 256;
  constexpr int wpb = threads / 32;
  const unsigned blocks = static_cast<unsigned>((hidden + wpb - 1) / wpb);
  if (qbits == 4) {
    moe_down_accum_kernel<4>
        <<<blocks, threads, 0, stream>>>(make_weight<4>(w, scales, inter, group), inter_in,
                                         topk_idx, topk_weight, y, hidden, inter, top_k);
  } else if (qbits == 8) {
    moe_down_accum_kernel<8>
        <<<blocks, threads, 0, stream>>>(make_weight<8>(w, scales, inter, group), inter_in,
                                         topk_idx, topk_weight, y, hidden, inter, top_k);
  } else {
    moe_down_accum_kernel<0><<<blocks, threads, 0, stream>>>(make_weight<0>(w, nullptr, inter, 0),
                                                             inter_in, topk_idx, topk_weight, y,
                                                             hidden, inter, top_k);
  }
}

// out[i] = in[i] * vec[i] * scale. The router's pre-projection scaling
// (hidden * router.scale * hidden^-0.5) in one pass.
__global__ void mul_vec_kernel(const half* in, const half* vec, half* out, int n, float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  out[i] = __float2half(__half2float(in[i]) * __half2float(vec[i]) * scale);
}

void launch_mul_vec(const half* in, const half* vec, half* out, int n, float scale,
                    cudaStream_t stream) {
  constexpr int kThreads = 256;
  const int blocks = (n + kThreads - 1) / kThreads;
  mul_vec_kernel<<<blocks, kThreads, 0, stream>>>(in, vec, out, n, scale);
}

// int4-direct grouped MoE GEMM on int8 tensor cores (m16n8k32.s8). Reads the int4 expert weights
// directly (nibble^8-8, per-128-group scales) against int8 activations (natural per-32-group
// scales), no dequant-to-fp16, no fp16 re-read. This is the MMQ-style path that skips the ~28ms
// dequant-all + the 4x fp16 weight read that dominate short-prompt DeepSeek MoE prefill (the
// fp16-dequant + cuBLAS grouped path still wins at long prompts, where that fixed cost amortizes).
// grid.z = expert e; each expert e's weight rows start at e*N, its gathered tokens at off[e]. The
// fused gate_up output is split at column `split` into out_lo (gate) / out_hi (up); down passes
// split=N, out_hi=nullptr. Fragment layout empirically verified (scratchpad/mma_probe) +
// cross-checked vs llama.cpp mma.cuh + PTX ISA.
namespace {
constexpr int kMoeMmaBM = 64, kMoeMmaBN = 64;
}  // namespace

__global__ void moe_int4_grouped_mma_kernel(const std::int8_t* __restrict__ xq,
                                            const float* __restrict__ as,
                                            const std::int8_t* __restrict__ wpacked,
                                            const float* __restrict__ ws,
                                            const int* __restrict__ off, half* __restrict__ out_lo,
                                            half* __restrict__ out_hi, int N, int K, int wsg_stride,
                                            int wratio, int split, int lo_width, int hi_width) {
// mma.m16n8k32.s8 needs sm_80+. For older targets in a fatbin (sm_75) compile
// an empty body so ptxas can assemble the arch; the engine never launches this
// kernel there (moe_use_int4_direct checks compute capability at runtime).
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
  (void)xq;
  (void)as;
  (void)wpacked;
  (void)ws;
  (void)off;
  (void)out_lo;
  (void)out_hi;
  (void)N;
  (void)K;
  (void)wsg_stride;
  (void)wratio;
  (void)split;
  (void)lo_width;
  (void)hi_width;
#else
  const int e = blockIdx.z;
  const int m_start = off[e], n_e = off[e + 1] - off[e];
  const int by = blockIdx.y;
  if (by * kMoeMmaBM >= n_e) return;
  __shared__ __align__(16) std::int8_t As[kMoeMmaBM][32];
  __shared__ __align__(16) std::int8_t Bs[kMoeMmaBN][32];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3, mt = warp & 3, nhalf = warp >> 2;
  const int blockM = m_start + by * kMoeMmaBM;
  const int wrowbase = e * N;
  const int pstride = (K + 1) / 2, Kg = K / 32;
  float facc[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) facc[i][j] = 0.0f;
  const int blockNrow = blockIdx.x * kMoeMmaBN;

  for (int kg = 0; kg < Kg; ++kg) {
    const int k0 = kg * 32;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int idx = tid + i * 256, r = idx >> 5, c = idx & 31, col = k0 + c;
      const int tok = blockM + r;
      As[r][c] = (by * kMoeMmaBM + r < n_e) ? xq[static_cast<std::size_t>(tok) * K + col] : 0;
      const int wrow = wrowbase + blockNrow + r;
      std::int8_t wv = 0;
      if (blockNrow + r < N) {
        const std::uint8_t byte = static_cast<std::uint8_t>(
            wpacked[static_cast<std::size_t>(wrow) * pstride + (col >> 1)]);
        const std::uint8_t nib = (col & 1) ? (byte >> 4) : (byte & 0x0F);
        wv = static_cast<std::int8_t>(static_cast<int>(nib ^ 0x8) - 8);
      }
      Bs[r][c] = wv;
    }
    __syncthreads();
    const int arow = 16 * mt;
    const bool m0 = (by * kMoeMmaBM + arow + g) < n_e, m8 = (by * kMoeMmaBM + arow + g + 8) < n_e;
    const int a0 = *reinterpret_cast<int*>(&As[arow + g][t * 4]);
    const int a1 = *reinterpret_cast<int*>(&As[arow + g + 8][t * 4]);
    const int a2 = *reinterpret_cast<int*>(&As[arow + g][16 + t * 4]);
    const int a3 = *reinterpret_cast<int*>(&As[arow + g + 8][16 + t * 4]);
    const float as_g = m0 ? as[static_cast<std::size_t>(blockM + arow + g) * Kg + kg] : 0.0f;
    const float as_g8 = m8 ? as[static_cast<std::size_t>(blockM + arow + g + 8) * Kg + kg] : 0.0f;
    const int wsg = kg / wratio;
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const int nbase = nhalf * 32 + nt * 8;
      const int b0 = *reinterpret_cast<int*>(&Bs[nbase + g][t * 4]);
      const int b1 = *reinterpret_cast<int*>(&Bs[nbase + g][16 + t * 4]);
      int c0 = 0, c1 = 0, c2 = 0, c3 = 0;
      asm volatile(
          "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
          "{%0,%1,%2,%3};\n"
          : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
          : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
      const int nra = blockNrow + nbase + 2 * t, nrb = blockNrow + nbase + 2 * t + 1;
      const float ws_a =
          (nra < N) ? ws[static_cast<std::size_t>(wrowbase + nra) * wsg_stride + wsg] : 0.0f;
      const float ws_b =
          (nrb < N) ? ws[static_cast<std::size_t>(wrowbase + nrb) * wsg_stride + wsg] : 0.0f;
      facc[nt][0] += static_cast<float>(c0) * as_g * ws_a;
      facc[nt][1] += static_cast<float>(c1) * as_g * ws_b;
      facc[nt][2] += static_cast<float>(c2) * as_g8 * ws_a;
      facc[nt][3] += static_cast<float>(c3) * as_g8 * ws_b;
    }
    __syncthreads();
  }
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    const int nbase = nhalf * 32 + nt * 8;
    const int rows[4] = {g, g, g + 8, g + 8}, cols[4] = {2 * t, 2 * t + 1, 2 * t, 2 * t + 1};
#pragma unroll
    for (int en = 0; en < 4; ++en) {
      const int lm = by * kMoeMmaBM + 16 * mt + rows[en], gn = blockNrow + nbase + cols[en];
      if (lm < n_e && gn < N) {
        const std::size_t grow = static_cast<std::size_t>(m_start + lm);
        if (gn < split)
          out_lo[grow * lo_width + gn] = __float2half(facc[nt][en]);
        else if (out_hi)
          out_hi[grow * hi_width + (gn - split)] = __float2half(facc[nt][en]);
      }
    }
  }
#endif  // __CUDA_ARCH__ >= 800
}

// off[] is device-resident (grid.z indexes it). max_ne = max tokens routed to any one expert
// (bounds grid.y). group = weight quant group (128); wratio = group/32. split/out_hi implement the
// fused gate_up split (down: split=N, out_hi=nullptr).
void launch_moe_int4_grouped_mma(const std::int8_t* xq, const float* as, const std::int8_t* wpacked,
                                 const float* ws, const int* off, half* out_lo, half* out_hi, int E,
                                 int N, int K, int max_ne, int group, int split, int lo_width,
                                 int hi_width, cudaStream_t stream) {
  if (max_ne <= 0) return;
  const int wsg_stride = (K + 127) / 128;
  const int wratio = group / 32;
  dim3 grid((N + kMoeMmaBN - 1) / kMoeMmaBN, (max_ne + kMoeMmaBM - 1) / kMoeMmaBM, E);
  moe_int4_grouped_mma_kernel<<<grid, 256, 0, stream>>>(xq, as, wpacked, ws, off, out_lo, out_hi, N,
                                                        K, wsg_stride, wratio, split, lo_width,
                                                        hi_width);
}

}  // namespace kernels
