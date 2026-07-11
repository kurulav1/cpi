// Mixture-of-Experts feed-forward kernels.
//
// The experts are stored as one contiguous matrix each, not as N separate
// tensors: gate_up is [num_experts * 2 * inter, hidden] and down is
// [num_experts * hidden, inter], both row-major. Expert e's rows simply start at
// e * rows_per_expert, so an expert selection is a row offset -- which means the
// existing quantiser and its group-wise scales apply unchanged, and the expert
// weights (which are ~90% of an MoE model) get quantised like any other matrix.
//
// The selected expert indices live on the DEVICE (the router wrote them there).
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
// (col >> 31) == 0 for every real column -- so the per-row and group-wise cases
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

// inter[k, r] = gelu(gate) * up, where gate and up are rows r and (inter + r) of
// the selected expert's fused gate_up matrix. Grid is (inter, top_k): one block
// per output element per selected expert.
template <int BITS>
__global__ void moe_gate_up_geglu_kernel(Weight<BITS> w, const half* x, const int* topk_idx,
                                         half* inter_out, int inter, int hidden, int top_k) {
  const int r = blockIdx.x;
  const int k = blockIdx.y;
  if (r >= inter || k >= top_k) {
    return;
  }
  extern __shared__ float smem[];
  const int tid = threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int warps = blockDim.x / warpSize;

  const std::size_t base =
      static_cast<std::size_t>(topk_idx[k]) * static_cast<std::size_t>(2 * inter);
  const std::size_t gate_row = base + static_cast<std::size_t>(r);
  const std::size_t up_row = base + static_cast<std::size_t>(inter + r);

  float g = 0.0f, u = 0.0f;
  for (int c = tid; c < hidden; c += blockDim.x) {
    const float xv = __half2float(x[c]);
    g += w.at(gate_row, c) * xv;
    u += w.at(up_row, c) * xv;
  }
  g = warp_sum(g);
  u = warp_sum(u);
  if (lane == 0) {
    smem[warp] = g;
    smem[warps + warp] = u;
  }
  __syncthreads();

  if (tid == 0) {
    float gs = 0.0f, us = 0.0f;
    for (int i = 0; i < warps; ++i) {
      gs += smem[i];
      us += smem[warps + i];
    }
    inter_out[static_cast<std::size_t>(k) * inter + r] = __float2half(gelu_tanh(gs) * us);
  }
}

// y[r] = sum_k topk_weight[k] * dot(down[expert_k].row(r), inter[k]).
// One block per output row; the block walks all top_k experts so the weighted sum
// needs no atomics and no second pass.
template <int BITS>
__global__ void moe_down_accum_kernel(Weight<BITS> w, const half* inter_in, const int* topk_idx,
                                      const float* topk_weight, half* y, int hidden, int inter,
                                      int top_k) {
  const int r = blockIdx.x;
  if (r >= hidden) {
    return;
  }
  extern __shared__ float smem[];
  const int tid = threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int warps = blockDim.x / warpSize;

  float acc = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    const std::size_t row = static_cast<std::size_t>(topk_idx[k]) *
                                static_cast<std::size_t>(hidden) +
                            static_cast<std::size_t>(r);
    const half* xk = inter_in + static_cast<std::size_t>(k) * inter;

    float d = 0.0f;
    for (int c = tid; c < inter; c += blockDim.x) {
      d += w.at(row, c) * __half2float(xk[c]);
    }
    d = warp_sum(d);
    if (lane == 0) {
      smem[warp] = d;
    }
    __syncthreads();
    if (tid == 0) {
      float t = 0.0f;
      for (int i = 0; i < warps; ++i) {
        t += smem[i];
      }
      smem[warps] = t;  // stash the full dot for this expert
    }
    __syncthreads();
    acc += topk_weight[k] * smem[warps];
    __syncthreads();  // smem is reused by the next expert
  }

  if (tid == 0) {
    y[r] = __float2half(acc);
  }
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
                              int hidden, int top_k, cudaStream_t stream) {
  const int threads = threads_for(hidden);
  const dim3 grid(static_cast<unsigned>(inter), static_cast<unsigned>(top_k));
  const std::size_t smem = static_cast<std::size_t>(2 * (threads / 32)) * sizeof(float);
  if (qbits == 4) {
    moe_gate_up_geglu_kernel<4><<<grid, threads, smem, stream>>>(
        make_weight<4>(w, scales, hidden, group), x, topk_idx, inter_out, inter, hidden, top_k);
  } else if (qbits == 8) {
    moe_gate_up_geglu_kernel<8><<<grid, threads, smem, stream>>>(
        make_weight<8>(w, scales, hidden, group), x, topk_idx, inter_out, inter, hidden, top_k);
  } else {
    moe_gate_up_geglu_kernel<0><<<grid, threads, smem, stream>>>(
        make_weight<0>(w, nullptr, hidden, 0), x, topk_idx, inter_out, inter, hidden, top_k);
  }
}

void launch_moe_down_accum(const void* w, const float* scales, int qbits, int group,
                           const half* inter_in, const int* topk_idx, const float* topk_weight,
                           half* y, int hidden, int inter, int top_k, cudaStream_t stream) {
  const int threads = threads_for(inter);
  const std::size_t smem = static_cast<std::size_t>(threads / 32 + 1) * sizeof(float);
  if (qbits == 4) {
    moe_down_accum_kernel<4><<<hidden, threads, smem, stream>>>(
        make_weight<4>(w, scales, inter, group), inter_in, topk_idx, topk_weight, y, hidden, inter,
        top_k);
  } else if (qbits == 8) {
    moe_down_accum_kernel<8><<<hidden, threads, smem, stream>>>(
        make_weight<8>(w, scales, inter, group), inter_in, topk_idx, topk_weight, y, hidden, inter,
        top_k);
  } else {
    moe_down_accum_kernel<0><<<hidden, threads, smem, stream>>>(
        make_weight<0>(w, nullptr, inter, 0), inter_in, topk_idx, topk_weight, y, hidden, inter,
        top_k);
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

}  // namespace kernels
