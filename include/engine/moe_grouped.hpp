#pragma once

// Grouped-GEMM MoE for DeepSeek sequence prefill. The naive seq path loops the batch-1 expert
// matvec per token, so every expert weight is re-read once per token routed to it (~59x for a
// 634-token prompt over 64 experts). Grouped-GEMM instead buckets all tokens routed to an expert,
// runs ONE real GEMM per expert (the weight is streamed once, reused across its tokens) and
// scatters the weighted results back. These are the two glue kernels; the routing is host-computed
// (T*top_k is tiny) and the per-expert GEMMs reuse the verified dequant + cuBLAS path.
//
//   gather:  gathered[p] = src[perm[p]]                    (collect an expert's token rows)
//   scatter: out_f32[perm[p]] += rweight[p] * grouped[p]   (weighted, atomic; tokens repeat top_k
//   x)

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace engine {

// gathered[p, :] = src[perm[p], :]  for p in [0, P), width H. One thread per element.
__global__ inline void moe_gather_rows_kernel(const __half* src, const int* perm, __half* gathered,
                                              int H, int P) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(P) * H;
  if (idx >= total) return;
  const int p = static_cast<int>(idx / H);
  const int h = static_cast<int>(idx % H);
  gathered[idx] = src[static_cast<std::size_t>(perm[p]) * H + h];
}

inline void launch_moe_gather_rows(const __half* src, const int* perm, __half* gathered, int H,
                                   int P, cudaStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(P) * H;
  const int block = 256;
  const std::size_t grid = (total + block - 1) / block;
  moe_gather_rows_kernel<<<static_cast<unsigned>(grid), block, 0, stream>>>(src, perm, gathered, H,
                                                                            P);
}

// out_f32[perm[p], :] += rweight[p] * grouped[p, :]  (fp32 accumulate; a token appears top_k times
// across different experts' blocks, so the adds race -> atomicAdd). Zero out_f32 first.
__global__ inline void moe_scatter_accum_kernel(const __half* grouped, const int* perm,
                                                const float* rweight, float* out_f32, int H,
                                                int P) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t total = static_cast<std::size_t>(P) * H;
  if (idx >= total) return;
  const int p = static_cast<int>(idx / H);
  const int h = static_cast<int>(idx % H);
  const float v = __half2float(grouped[idx]) * rweight[p];
  atomicAdd(&out_f32[static_cast<std::size_t>(perm[p]) * H + h], v);
}

inline void launch_moe_scatter_accum(const __half* grouped, const int* perm, const float* rweight,
                                     float* out_f32, int H, int P, cudaStream_t stream) {
  const std::size_t total = static_cast<std::size_t>(P) * H;
  const int block = 256;
  const std::size_t grid = (total + block - 1) / block;
  moe_scatter_accum_kernel<<<static_cast<unsigned>(grid), block, 0, stream>>>(
      grouped, perm, rweight, out_f32, H, P);
}

// moeout[t, :] = out_f32[t, :]  (fp32 accumulator -> fp16 output). One thread per element.
__global__ inline void moe_f32_to_f16_kernel(const float* in, __half* out, std::size_t n) {
  const std::size_t idx = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  out[idx] = __float2half(in[idx]);
}

inline void launch_moe_f32_to_f16(const float* in, __half* out, std::size_t n,
                                  cudaStream_t stream) {
  const int block = 256;
  const std::size_t grid = (n + block - 1) / block;
  moe_f32_to_f16_kernel<<<static_cast<unsigned>(grid), block, 0, stream>>>(in, out, n);
}

}  // namespace engine
