// kernels_turboquant.cu
//
// CUDA kernels and host launch wrappers for TurboQuant/TQ3 helpers.

#include "runtime/kernels.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>

namespace kernels {
namespace {

__device__ __forceinline__ float warp_sum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__global__ void hadamard_rotate_fp16_kernel(half* __restrict__ x,
                                            const int8_t* __restrict__ signs,
                                            int block_size) {
  extern __shared__ float smem[];

  // Each CUDA block handles one WHT sub-block.
  const int base       = blockIdx.x * block_size;
  half*         xb     = x     + base;
  const int8_t* sb     = signs + base;

  // Load and apply random sign diagonal D for this sub-block.
  for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
    smem[i] = __half2float(xb[i]) * static_cast<float>(sb[i]);
  }
  __syncthreads();

  // In-place Walsh-Hadamard butterfly.  log2(block_size) stages.
  for (int stride = 1; stride < block_size; stride <<= 1) {
    for (int i = threadIdx.x; i < block_size / 2; i += blockDim.x) {
      const int j = (i / stride) * (stride * 2) + (i % stride);
      const int k = j + stride;
      const float a = smem[j];
      const float b = smem[k];
      smem[j] = a + b;
      smem[k] = a - b;
    }
    __syncthreads();
  }

  // Normalise and write back as fp16.
  const float inv_sqrt_bs = rsqrtf(static_cast<float>(block_size));
  for (int i = threadIdx.x; i < block_size; i += blockDim.x) {
    xb[i] = __float2half(smem[i] * inv_sqrt_bs);
  }
}

// Weight-only 3-bit TQ3 GEMV.  Each warp handles one output row.
//
// Template params:
//   WarpsPerBlock – warps per block; one warp per output row.
//
// Shared memory: in_features half values (x loaded once per block) plus
//   8 floats for the codebook.
template <int WarpsPerBlock>
__global__ void tq3_gemv_f16_kernel(const uint32_t* __restrict__ w_packed,
                                    const half*     __restrict__ codebook,
                                    const half*     __restrict__ scales,
                                    const half*     __restrict__ x,
                                    half*           __restrict__ y,
                                    int out_features,
                                    int in_features,
                                    int words_per_row) {
  // Shared memory: x only.  Codebook is held in per-lane registers and looked
  // up with __shfl_sync (2-cycle warp op) rather than shared memory, which
  // would have 4-way bank conflicts (8 entries × 32 threads → 4 threads/bank).
  extern __shared__ half x_sh[];

  // Load x into shared memory collaboratively.
  for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
    x_sh[i] = x[i];
  }

  const int warp_id = threadIdx.x / warpSize;
  const int lane    = threadIdx.x & (warpSize - 1);

  // Lane i (0..7) holds codebook[i]; lanes 8..31 hold 0 (unused as source).
  // All threads issue the same 8 global loads → L2 broadcast, no bank issue.
  const float my_cb = (lane < 8) ? __half2float(codebook[lane]) : 0.0f;

  __syncthreads();

  const int row = blockIdx.x * WarpsPerBlock + warp_id;
  if (row >= out_features) {
    return;
  }

  const uint32_t* row_w = w_packed + static_cast<std::size_t>(row) * words_per_row;
  float acc = 0.0f;

  // Pad words_per_row to the next multiple of warpSize so every lane executes
  // the same number of outer-loop iterations.  This keeps all lanes in lockstep
  // for __shfl_sync, which requires all mask bits (0xFFFFFFFF) to execute the
  // instruction together — a requirement enforced by Independent Thread
  // Scheduling on Ampere/Ada (SM 8.0+). Without padding, lanes 26-31 exit the
  // loop one iteration earlier than lane 25 and proceed to warp_sum while lane
  // 25 is still calling __shfl_sync, creating a deadlock.
  const int words_padded = ((words_per_row + warpSize - 1) / warpSize) * warpSize;

  // Each lane processes a strided subset of the packed words.
  for (int wi = lane; wi < words_padded; wi += warpSize) {
    // For dummy (padding) iterations beyond words_per_row, use 0 so all 3-bit
    // indices are 0 — the __shfl_sync still executes but j >= in_features
    // guards prevent any accumulation.
    const uint32_t packed  = (wi < words_per_row) ? row_w[wi] : 0u;
    const int      base_j  = wi * 10;
    // Unroll the inner 10-element loop; the compiler can hoist this fully.
    #pragma unroll
    for (int k = 0; k < 10; ++k) {
      const int j = base_j + k;
      const int   idx  = (packed >> (k * 3)) & 0x7;
      // __shfl_sync MUST be called unconditionally by all warp lanes so the
      // 0xFFFFFFFF convergence requirement is satisfied.  The j < in_features
      // guard below prevents accumulating out-of-range elements.
      const float cb_val = __shfl_sync(0xFFFFFFFF, my_cb, idx);
      if (j < in_features) {
        acc += cb_val * __half2float(x_sh[j]);
      }
    }
  }

  // Warp reduction.
  acc = warp_sum(acc);
  if (lane == 0) {
    y[row] = __float2half(acc * __half2float(scales[row]));
  }
}

__global__ void tq_qjl_pack_sign_bits_kernel(const half* __restrict__ x,
                                             const int32_t* __restrict__ indices,
                                             const int8_t* __restrict__ signs,
                                             uint32_t* __restrict__ out_bits,
                                             int qjl_dim) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= qjl_dim) {
    return;
  }
  const int idx = indices[j];
  float v = __half2float(x[idx]);
  if (signs[j] < 0) {
    v = -v;
  }
  if (v >= 0.0f) {
    atomicOr(&out_bits[j >> 5], (1u << (j & 31)));
  }
}

template <int WarpsPerBlock>
__global__ void tq_qjl_residual_add_f16_kernel(const uint32_t* __restrict__ row_bits,
                                               const half* __restrict__ scales,
                                               const uint32_t* __restrict__ x_bits,
                                               half* __restrict__ y,
                                               int out_features,
                                               int qjl_dim,
                                               int words_per_row) {
  const int warp_id = threadIdx.x / warpSize;
  const int lane = threadIdx.x & (warpSize - 1);
  const int row = blockIdx.x * WarpsPerBlock + warp_id;
  if (row >= out_features || qjl_dim <= 0) {
    return;
  }

  const uint32_t* row_ptr = row_bits + static_cast<std::size_t>(row) * words_per_row;
  int mismatch = 0;
  for (int wi = lane; wi < words_per_row; wi += warpSize) {
    uint32_t diff = row_ptr[wi] ^ x_bits[wi];
    if (wi == words_per_row - 1 && (qjl_dim & 31) != 0) {
      const uint32_t mask = (1u << (qjl_dim & 31)) - 1u;
      diff &= mask;
    }
    mismatch += __popc(diff);
  }

  for (int offset = 16; offset > 0; offset >>= 1) {
    mismatch += __shfl_down_sync(0xffffffffu, mismatch, offset);
  }

  if (lane == 0) {
    const int signed_agreement = qjl_dim - 2 * mismatch;
    const float corr = static_cast<float>(signed_agreement) / static_cast<float>(qjl_dim);
    const float updated = __half2float(y[row]) + corr * __half2float(scales[row]);
    y[row] = __float2half(updated);
  }
}

}  // namespace

// block_size: largest power-of-2 factor of n (== n when n is a power of 2).
// n / block_size CUDA blocks are launched, each handling one WHT sub-block.
void launch_hadamard_rotate_fp16(half* x, const int8_t* signs, int n, int block_size, cudaStream_t stream) {
  const int cuda_blocks = n / block_size;
  const int threads = 512;
  const std::size_t shmem = static_cast<std::size_t>(block_size) * sizeof(float);
  hadamard_rotate_fp16_kernel<<<cuda_blocks, threads, shmem, stream>>>(x, signs, block_size);
}

void launch_tq3_gemv_f16(const uint32_t* w_packed,
                          const half*     codebook,
                          const half*     scales,
                          const half*     x,
                          half*           y,
                          int             out_features,
                          int             in_features,
                          cudaStream_t    stream) {
  constexpr int kWarps   = 8;
  constexpr int kThreads = kWarps * 32;
  const int words_per_row = (in_features + 9) / 10;
  const int blocks = (out_features + kWarps - 1) / kWarps;
  // Shared: in_features halves (codebook is in registers, not shmem).
  const std::size_t shmem = static_cast<std::size_t>(in_features) * sizeof(half);
  tq3_gemv_f16_kernel<kWarps><<<blocks, kThreads, shmem, stream>>>(
      w_packed, codebook, scales, x, y, out_features, in_features, words_per_row);
}

void launch_tq_qjl_pack_sign_bits(const half*     x,
                                  const int32_t*  indices,
                                  const int8_t*   signs,
                                  uint32_t*       out_bits,
                                  int             qjl_dim,
                                  cudaStream_t    stream) {
  if (qjl_dim <= 0) {
    return;
  }
  const int words = (qjl_dim + 31) / 32;
  cudaMemsetAsync(out_bits, 0, static_cast<std::size_t>(words) * sizeof(uint32_t), stream);
  constexpr int kThreads = 256;
  const int blocks = (qjl_dim + kThreads - 1) / kThreads;
  tq_qjl_pack_sign_bits_kernel<<<blocks, kThreads, 0, stream>>>(x, indices, signs, out_bits, qjl_dim);
}

void launch_tq_qjl_residual_add_f16(const uint32_t* row_bits,
                                    const half*     scales,
                                    const uint32_t* x_bits,
                                    half*           y,
                                    int             out_features,
                                    int             qjl_dim,
                                    cudaStream_t    stream) {
  if (out_features <= 0 || qjl_dim <= 0) {
    return;
  }
  constexpr int kWarps = 8;
  constexpr int kThreads = kWarps * 32;
  const int blocks = (out_features + kWarps - 1) / kWarps;
  const int words = (qjl_dim + 31) / 32;
  tq_qjl_residual_add_f16_kernel<kWarps><<<blocks, kThreads, 0, stream>>>(
      row_bits, scales, x_bits, y, out_features, qjl_dim, words);
}

}  // namespace kernels
