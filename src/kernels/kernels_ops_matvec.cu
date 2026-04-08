// kernels_ops_matvec.cu
//
// CUDA kernels and host launch wrappers for pointwise ops, quant/dequant,
// weight-only matvec, projection GEMV, and argmax helpers.

#include "runtime/kernels.cuh"

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <cuda_fp16.h>
#include <sm_61_intrinsics.h>

namespace kernels {
namespace {

inline int choose_reduction_threads(int cols) {
  return (cols <= 1024) ? 128 : 256;
}

__device__ __forceinline__ half bf16_bits_to_half(std::uint16_t bits) {
  const unsigned int u = static_cast<unsigned int>(bits) << 16;
  const float f = __uint_as_float(u);
  return __float2half(f);
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
  }
  return v;
}

__device__ __forceinline__ void warp_argmax(float& value, int& index) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    const float other_value = __shfl_down_sync(0xffffffffu, value, offset);
    const int other_index = __shfl_down_sync(0xffffffffu, index, offset);
    if (other_value > value) {
      value = other_value;
      index = other_index;
    }
  }
}

// Pointwise math, quantization, and dequantization kernels.
__global__ void add_inplace_kernel(half* x, const half* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  x[i] = __hadd(x[i], y[i]);
}

__global__ void add_inplace_half2_kernel(half2* x, const half2* y, int n2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n2) {
    return;
  }
  x[i] = __hadd2(x[i], y[i]);
}

__global__ void silu_mul_kernel(const half* gate, const half* up, half* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float g = __half2float(gate[i]);
  const float u = __half2float(up[i]);
  const float s = g / (1.0f + expf(-g));
  out[i] = __float2half(s * u);
}

__global__ void silu_mul_half2_kernel(const half2* gate, const half2* up, half2* out, int n2) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n2) {
    return;
  }

  const half2 g2 = gate[i];
  const half2 u2 = up[i];
  const float g0 = __half2float(__low2half(g2));
  const float g1 = __half2float(__high2half(g2));
  const float u0 = __half2float(__low2half(u2));
  const float u1 = __half2float(__high2half(u2));
  const float s0 = g0 / (1.0f + expf(-g0));
  const float s1 = g1 / (1.0f + expf(-g1));
  out[i] = __halves2half2(__float2half(s0 * u0), __float2half(s1 * u1));
}

__global__ void scale_copy_kernel(half* dst, const half* src, int n, float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  dst[i] = __float2half(__half2float(src[i]) * scale);
}

__global__ void scale_add_inplace_kernel(half* dst, const half* src, int n, float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  const float acc = __half2float(dst[i]) + __half2float(src[i]) * scale;
  dst[i] = __float2half(acc);
}

__global__ void moe_router_topk_softmax_kernel(const half* logits,
                                               int experts,
                                               int top_k,
                                               int* topk_idx,
                                               float* topk_prob) {
  extern __shared__ float probs[];
  if (threadIdx.x != 0) {
    return;
  }

  if (experts <= 0 || top_k <= 0) {
    return;
  }

  float max_logit = __half2float(logits[0]);
  for (int e = 1; e < experts; ++e) {
    max_logit = fmaxf(max_logit, __half2float(logits[e]));
  }

  float sum = 0.0f;
  for (int e = 0; e < experts; ++e) {
    const float p = expf(__half2float(logits[e]) - max_logit);
    probs[e] = p;
    sum += p;
  }
  const float inv_sum = 1.0f / fmaxf(sum, 1.0e-8f);
  for (int e = 0; e < experts; ++e) {
    probs[e] *= inv_sum;
  }

  constexpr int kMaxTopK = 8;
  int picked[kMaxTopK];
  float picked_prob[kMaxTopK];
  const int capped_topk = top_k > kMaxTopK ? kMaxTopK : top_k;
  for (int k = 0; k < capped_topk; ++k) {
    picked[k] = -1;
    picked_prob[k] = 0.0f;
  }

  for (int k = 0; k < capped_topk; ++k) {
    int best_e = -1;
    float best_p = -1.0f;
    for (int e = 0; e < experts; ++e) {
      bool used = false;
      for (int prev = 0; prev < k; ++prev) {
        if (picked[prev] == e) {
          used = true;
          break;
        }
      }
      if (used) {
        continue;
      }
      const float p = probs[e];
      if (p > best_p) {
        best_p = p;
        best_e = e;
      }
    }
    picked[k] = best_e;
    picked_prob[k] = best_p > 0.0f ? best_p : 0.0f;
  }

  float picked_sum = 0.0f;
  for (int k = 0; k < capped_topk; ++k) {
    picked_sum += picked_prob[k];
  }
  const float inv_pick_sum = 1.0f / fmaxf(picked_sum, 1.0e-8f);

  for (int k = 0; k < top_k; ++k) {
    if (k < capped_topk) {
      topk_idx[k] = picked[k] >= 0 ? picked[k] : 0;
      topk_prob[k] = picked_prob[k] * inv_pick_sum;
    } else {
      topk_idx[k] = 0;
      topk_prob[k] = 0.0f;
    }
  }
}

__global__ void convert_bf16_to_fp16_kernel(const std::uint16_t* src, half* dst, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) {
    return;
  }
  dst[idx] = bf16_bits_to_half(src[idx]);
}

__global__ void dequant_int8_to_fp16_kernel(const int8_t* src, half* dst, int n, float scale) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) {
    return;
  }
  dst[i] = __float2half(static_cast<float>(src[i]) * scale);
}

__global__ void dequant_rowwise_int8_to_fp16_kernel(const int8_t* src,
                                                    const float* scales,
                                                    half* dst,
                                                    int rows,
                                                    int cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int n = rows * cols;
  if (idx >= n) {
    return;
  }
  const int row = idx / cols;
  dst[idx] = __float2half(static_cast<float>(src[idx]) * scales[row]);
}

__global__ void quantize_rowwise_fp16_to_int8_kernel(const half* src,
                                                     int8_t* dst,
                                                     float* scales,
                                                     int cols,
                                                     int max_q) {
  extern __shared__ float smax[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
  const int base = row * cols;

  float local_max = 0.0f;
  if ((cols & 1) == 0) {
    const half2* src2 = reinterpret_cast<const half2*>(src + base);
    const int cols2 = cols / 2;
    for (int col2 = tid; col2 < cols2; col2 += blockDim.x) {
      const half2 v2 = src2[col2];
      local_max = fmaxf(local_max, fabsf(__half2float(__low2half(v2))));
      local_max = fmaxf(local_max, fabsf(__half2float(__high2half(v2))));
    }
  } else {
    for (int col = tid; col < cols; col += blockDim.x) {
      local_max = fmaxf(local_max, fabsf(__half2float(src[base + col])));
    }
  }

  local_max = warp_max(local_max);
  if (lane == 0) {
    smax[warp] = local_max;
  }
  __syncthreads();

  if (warp == 0) {
    float block_max = (lane < warp_count) ? smax[lane] : 0.0f;
    block_max = warp_max(block_max);
    if (lane == 0) {
      const float max_q_f = static_cast<float>(max_q);
      float scale = block_max / max_q_f;
      if (scale < 1.0e-8f) {
        scale = 1.0e-8f;
      }
      scales[row] = scale;
      smax[0] = 1.0f / scale;
    }
  }
  __syncthreads();

  const float inv_scale = smax[0];
  if ((cols & 1) == 0) {
    const half2* src2 = reinterpret_cast<const half2*>(src + base);
    char2* dst2 = reinterpret_cast<char2*>(dst + base);
    const int cols2 = cols / 2;
    for (int col2 = tid; col2 < cols2; col2 += blockDim.x) {
      const half2 v2 = src2[col2];
      int q0 = __float2int_rn(__half2float(__low2half(v2)) * inv_scale);
      int q1 = __float2int_rn(__half2float(__high2half(v2)) * inv_scale);
      q0 = max(-max_q, min(max_q, q0));
      q1 = max(-max_q, min(max_q, q1));
      dst2[col2] = make_char2(static_cast<signed char>(q0), static_cast<signed char>(q1));
    }
  } else {
    for (int col = tid; col < cols; col += blockDim.x) {
      const float v = __half2float(src[base + col]) * inv_scale;
      int q = __float2int_rn(v);
      q = max(-max_q, min(max_q, q));
      dst[base + col] = static_cast<int8_t>(q);
    }
  }
}

__global__ void pack_rowwise_int8_to_int4_kernel(const int8_t* src,
                                                 int8_t* dst,
                                                 int rows,
                                                 int cols) {
  const int row = blockIdx.x;
  const int col2 = blockIdx.y * blockDim.x + threadIdx.x;
  const int packed_cols = (cols + 1) / 2;
  if (row >= rows || col2 >= packed_cols) {
    return;
  }

  const int col = col2 * 2;
  const std::size_t src_base = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  auto clamp_i4 = [](int v) -> std::uint8_t {
    const int q = max(-8, min(7, v));
    return static_cast<std::uint8_t>(q < 0 ? q + 16 : q);
  };

  const std::uint8_t lo = clamp_i4(static_cast<int>(src[src_base + static_cast<std::size_t>(col)]));
  std::uint8_t hi = 0;
  if (col + 1 < cols) {
    hi = clamp_i4(static_cast<int>(src[src_base + static_cast<std::size_t>(col + 1)]));
  }

  const std::size_t dst_base = static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
  dst[dst_base + static_cast<std::size_t>(col2)] = static_cast<int8_t>(lo | (hi << 4));
}

// Weight-only int8 GEMV kernels.

// Weight-only int8/int4 matvec kernels moved to kernels_weight_only_matvec.cu.

template <typename OutT>
__device__ __forceinline__ void store_projection_value(OutT* y, int row, float value);

template <>
__device__ __forceinline__ void store_projection_value<half>(half* y, int row, float value) {
  y[row] = __float2half(value);
}

template <>
__device__ __forceinline__ void store_projection_value<float>(float* y, int row, float value) {
  y[row] = value;
}

// Shared-memory half2 GEMV used by resident projection layers and the LM head.
template <int WarpsPerBlock, int TilePairs, int RowsPerWarp, typename OutT>
__global__ void rowmajor_half_gemv_kernel(const half* w,
                                          const half* x,
                                          OutT* y,
                                          int out_features,
                                          int in_features) {
  static_assert(RowsPerWarp >= 1, "RowsPerWarp must be >= 1");
  __shared__ half2 x_tile[TilePairs];
  const int warp_id = threadIdx.x / warpSize;
  const int lane = threadIdx.x & (warpSize - 1);
  const int row_base = (blockIdx.x * WarpsPerBlock + warp_id) * RowsPerWarp;
  if (row_base >= out_features) {
    return;
  }

  float local[RowsPerWarp] = {0.0f};
  const half* row_ptrs[RowsPerWarp];
  #pragma unroll
  for (int r = 0; r < RowsPerWarp; ++r) {
    const int row = row_base + r;
    row_ptrs[r] = (row < out_features)
                      ? (w + static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features))
                      : nullptr;
  }
  if ((in_features & 1) == 0) {
    const int pairs = in_features / 2;
    const half2* x2 = reinterpret_cast<const half2*>(x);
    const half2* row_ptrs2[RowsPerWarp];
    #pragma unroll
    for (int r = 0; r < RowsPerWarp; ++r) {
      row_ptrs2[r] = row_ptrs[r] ? reinterpret_cast<const half2*>(row_ptrs[r]) : nullptr;
    }
    for (int tile_base = 0; tile_base < pairs; tile_base += TilePairs) {
      const int tile_count = min(TilePairs, pairs - tile_base);
      // Use min(blockDim.x, TilePairs) as stride so all threads participate
      // even when blockDim.x > TilePairs (e.g. warps=8, tile=128).
      for (int idx = threadIdx.x; idx < tile_count; idx += blockDim.x) {
        x_tile[idx] = x2[tile_base + idx];
      }
      __syncthreads();

      for (int idx = lane; idx < tile_count; idx += warpSize) {
        const float2 xv = __half22float2(x_tile[idx]);
        #pragma unroll
        for (int r = 0; r < RowsPerWarp; ++r) {
          if (row_ptrs2[r]) {
            const float2 wv = __half22float2(row_ptrs2[r][tile_base + idx]);
            local[r] += wv.x * xv.x + wv.y * xv.y;
          }
        }
      }
      __syncthreads();
    }
  } else {
    for (int col = lane; col < in_features; col += warpSize) {
      const float xv = __half2float(x[col]);
      #pragma unroll
      for (int r = 0; r < RowsPerWarp; ++r) {
        if (row_ptrs[r]) {
          local[r] += __half2float(row_ptrs[r][col]) * xv;
        }
      }
    }
  }

  #pragma unroll
  for (int r = 0; r < RowsPerWarp; ++r) {
    local[r] = warp_sum(local[r]);
    if (lane == 0) {
      const int row = row_base + r;
      if (row < out_features) {
        store_projection_value<OutT>(y, row, local[r]);
      }
    }
  }
}

// Single-block argmax for the final logits vector.
__global__ void argmax_float_kernel(const float* logits, int n, int* out_index) {
  const int tid = threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
  __shared__ float warp_max[32];
  __shared__ int warp_idx[32];

  float local_max = -3.402823466e+38F;
  int local_idx = 0;
  for (int i = tid; i < n; i += blockDim.x) {
    const float v = logits[i];
    if (v > local_max) {
      local_max = v;
      local_idx = i;
    }
  }

  warp_argmax(local_max, local_idx);
  if (lane == 0) {
    warp_max[warp] = local_max;
    warp_idx[warp] = local_idx;
  }
  __syncthreads();

  if (warp == 0) {
    float block_max = (lane < warp_count) ? warp_max[lane] : -3.402823466e+38F;
    int block_idx = (lane < warp_count) ? warp_idx[lane] : 0;
    warp_argmax(block_max, block_idx);
    if (lane == 0) {
      *out_index = block_idx;
    }
  }
}
}  // namespace

// Host launch wrappers for pointwise/quant/matvec/projection kernels.
__global__ void add_bias_broadcast_kernel(half* out, const half* bias, int rows, int cols) {
  const int col = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.y);
  if (col < cols) {
    out[row * cols + col] = __hadd(out[row * cols + col], bias[col]);
  }
}

__global__ void add_bias_inplace_float_from_half_kernel(float* out, const half* bias, int n) {
  const int i = static_cast<int>(blockIdx.x) * blockDim.x + static_cast<int>(threadIdx.x);
  if (i < n) {
    out[i] += __half2float(bias[i]);
  }
}

void launch_add_bias_broadcast(half* out, const half* bias, int rows, int cols, cudaStream_t stream) {
  if (rows <= 0 || cols <= 0) {
    return;
  }
  constexpr int threads = 256;
  const dim3 grid(static_cast<unsigned>((cols + threads - 1) / threads), static_cast<unsigned>(rows));
  add_bias_broadcast_kernel<<<grid, threads, 0, stream>>>(out, bias, rows, cols);
}

void launch_add_bias_inplace_float_from_half(float* out,
                                             const half* bias,
                                             int n,
                                             cudaStream_t stream) {
  if (n <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  add_bias_inplace_float_from_half_kernel<<<blocks, threads, 0, stream>>>(out, bias, n);
}

void launch_add_inplace(half* x, const half* y, int n, cudaStream_t stream) {
  constexpr int threads = 256;
  const bool aligned = ((reinterpret_cast<std::uintptr_t>(x) | reinterpret_cast<std::uintptr_t>(y)) &
                        (alignof(half2) - 1)) == 0;
  if ((n & 1) == 0 && aligned) {
    const int n2 = n / 2;
    const int blocks = (n2 + threads - 1) / threads;
    add_inplace_half2_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<half2*>(x), reinterpret_cast<const half2*>(y), n2);
    return;
  }

  const int blocks = (n + threads - 1) / threads;
  add_inplace_kernel<<<blocks, threads, 0, stream>>>(x, y, n);
}

void launch_silu_mul(const half* gate,
                     const half* up,
                     half* out,
                     int n,
                     cudaStream_t stream) {
  constexpr int threads = 256;
  const bool aligned = ((reinterpret_cast<std::uintptr_t>(gate) | reinterpret_cast<std::uintptr_t>(up) |
                         reinterpret_cast<std::uintptr_t>(out)) &
                        (alignof(half2) - 1)) == 0;
  if ((n & 1) == 0 && aligned) {
    const int n2 = n / 2;
    const int blocks = (n2 + threads - 1) / threads;
    silu_mul_half2_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const half2*>(gate),
                                                          reinterpret_cast<const half2*>(up),
                                                          reinterpret_cast<half2*>(out),
                                                          n2);
    return;
  }

  const int blocks = (n + threads - 1) / threads;
  silu_mul_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, n);
}

void launch_scale_copy(half* dst,
                       const half* src,
                       int n,
                       float scale,
                       cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  scale_copy_kernel<<<blocks, threads, 0, stream>>>(dst, src, n, scale);
}

void launch_scale_add_inplace(half* dst,
                              const half* src,
                              int n,
                              float scale,
                              cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  scale_add_inplace_kernel<<<blocks, threads, 0, stream>>>(dst, src, n, scale);
}

void launch_moe_router_topk_softmax(const half* logits,
                                    int experts,
                                    int top_k,
                                    int* topk_idx,
                                    float* topk_prob,
                                    cudaStream_t stream) {
  if (experts <= 0 || top_k <= 0) {
    return;
  }
  const std::size_t smem = static_cast<std::size_t>(experts) * sizeof(float);
  moe_router_topk_softmax_kernel<<<1, 32, smem, stream>>>(logits, experts, top_k, topk_idx, topk_prob);
}

void launch_dequant_int8_to_fp16(const int8_t* src,
                                 half* dst,
                                 int n,
                                 float scale,
                                 cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  dequant_int8_to_fp16_kernel<<<blocks, threads, 0, stream>>>(src, dst, n, scale);
}

void launch_dequant_rowwise_int8_to_fp16(const int8_t* src,
                                         const float* scales,
                                         half* dst,
                                         int rows,
                                         int cols,
                                         cudaStream_t stream) {
  constexpr int threads = 256;
  const int n = rows * cols;
  const int blocks = (n + threads - 1) / threads;
  dequant_rowwise_int8_to_fp16_kernel<<<blocks, threads, 0, stream>>>(src, scales, dst, rows, cols);
}

void launch_quantize_rowwise_fp16_to_int8(const half* src,
                                          int8_t* dst,
                                          float* scales,
                                          int rows,
                                          int cols,
                                          cudaStream_t stream,
                                          int max_q) {
  if (max_q <= 0) {
    max_q = 127;
  }
  const int threads = choose_reduction_threads(cols);
  constexpr int kWarpSize = 32;
  const int warp_count = (threads + kWarpSize - 1) / kWarpSize;
  quantize_rowwise_fp16_to_int8_kernel<<<rows, threads, static_cast<std::size_t>(warp_count) * sizeof(float), stream>>>(
      src, dst, scales, cols, max_q);
}

void launch_pack_rowwise_int8_to_int4(const int8_t* src,
                                      int8_t* dst,
                                      int rows,
                                      int cols,
                                      cudaStream_t stream) {
  constexpr int threads = 256;
  const int packed_cols = (cols + 1) / 2;
  const dim3 grid(static_cast<unsigned int>(rows),
                  static_cast<unsigned int>((packed_cols + threads - 1) / threads));
  pack_rowwise_int8_to_int4_kernel<<<grid, threads, 0, stream>>>(src, dst, rows, cols);
}


template <typename OutT>
static void launch_rowmajor_half_gemv(const half* w,
                                      const half* x,
                                      OutT* y,
                                      int out_features,
                                      int in_features,
                                      cudaStream_t stream,
                                      int warps_per_block,
                                      int tile_pairs,
                                      int rows_per_warp) {
  const int warps = (warps_per_block > 0) ? warps_per_block : ((out_features >= 8192) ? 8 : 4);
  const int tile = (tile_pairs > 0) ? tile_pairs : 128;
  const int rows = (rows_per_warp > 0) ? rows_per_warp : 1;
  auto launch = [&](auto warps_tag, auto tile_tag, auto rows_tag) {
    constexpr int kWarps = decltype(warps_tag)::value;
    constexpr int kTile = decltype(tile_tag)::value;
    constexpr int kRows = decltype(rows_tag)::value;
    constexpr int kThreads = kWarps * 32;
    const int blocks = (out_features + kWarps * kRows - 1) / (kWarps * kRows);
    rowmajor_half_gemv_kernel<kWarps, kTile, kRows><<<blocks, kThreads, 0, stream>>>(w, x, y, out_features, in_features);
  };
  auto launch_rows = [&](auto warps_tag, auto tile_tag) {
    if (rows >= 2) {
      launch(warps_tag, tile_tag, std::integral_constant<int, 2>{});
    } else {
      launch(warps_tag, tile_tag, std::integral_constant<int, 1>{});
    }
  };

  if (warps >= 16) {
    if (tile >= 512) {
      launch_rows(std::integral_constant<int, 16>{}, std::integral_constant<int, 512>{});
    } else if (tile >= 256) {
      launch_rows(std::integral_constant<int, 16>{}, std::integral_constant<int, 256>{});
    } else {
      launch_rows(std::integral_constant<int, 16>{}, std::integral_constant<int, 128>{});
    }
    return;
  }

  if (warps >= 8) {
    if (tile >= 512) {
      launch_rows(std::integral_constant<int, 8>{}, std::integral_constant<int, 512>{});
    } else if (tile >= 256) {
      launch_rows(std::integral_constant<int, 8>{}, std::integral_constant<int, 256>{});
    } else {
      launch_rows(std::integral_constant<int, 8>{}, std::integral_constant<int, 128>{});
    }
    return;
  }

  if (tile >= 512) {
    launch_rows(std::integral_constant<int, 4>{}, std::integral_constant<int, 512>{});
  } else if (tile >= 256) {
    launch_rows(std::integral_constant<int, 4>{}, std::integral_constant<int, 256>{});
  } else {
    launch_rows(std::integral_constant<int, 4>{}, std::integral_constant<int, 128>{});
  }
}

void launch_rowmajor_half_gemv_f16(const half* w, const half* x, half* y,
                                   int out_features, int in_features,
                                   cudaStream_t stream, int warps_per_block,
                                   int tile_pairs, int rows_per_warp) {
  launch_rowmajor_half_gemv(w, x, y, out_features, in_features, stream,
                             warps_per_block, tile_pairs, rows_per_warp);
}

void launch_rowmajor_half_gemv_f32(const half* w, const half* x, float* y,
                                   int out_features, int in_features,
                                   cudaStream_t stream, int warps_per_block,
                                   int tile_pairs, int rows_per_warp) {
  launch_rowmajor_half_gemv(w, x, y, out_features, in_features, stream,
                             warps_per_block, tile_pairs, rows_per_warp);
}

void launch_argmax_float(const float* logits, int n, int* out_index, cudaStream_t stream) {
  constexpr int threads = 256;
  argmax_float_kernel<<<1, threads, 0, stream>>>(logits, n, out_index);
}

void launch_convert_bf16_to_fp16(const std::uint16_t* src,
                                 half* dst,
                                 int n,
                                 cudaStream_t stream) {
  constexpr int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  convert_bf16_to_fp16_kernel<<<blocks, threads, 0, stream>>>(src, dst, n);
}

}  // namespace kernels
