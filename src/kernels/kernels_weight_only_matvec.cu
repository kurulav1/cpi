// kernels_weight_only_matvec.cu
//
// CUDA kernels and host launch wrappers for weight-only int8/int4 matvec paths.

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

__device__ __forceinline__ float warp_sum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ int warp_sum_int(int v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ int decode_signed_int4(std::uint8_t nibble) {
  return (static_cast<int>(nibble) ^ 0x8) - 0x8;
}

__device__ __forceinline__ int8_t load_signed_int4(const int8_t* row_packed, int col) {
  const std::uint8_t byte = static_cast<std::uint8_t>(row_packed[col >> 1]);
  const std::uint8_t nibble = (col & 1) == 0 ? (byte & 0x0Fu) : ((byte >> 4) & 0x0Fu);
  return static_cast<int8_t>(decode_signed_int4(nibble));
}

// Decodes 4 consecutive signed int4 values into one packed int32 suitable for dp4a.
__device__ __forceinline__ int load_packed_int4x4(const int8_t* row_packed, int packed4_index) {
  const int byte_index = packed4_index * 2;
  const std::uint8_t b0 = static_cast<std::uint8_t>(row_packed[byte_index + 0]);
  const std::uint8_t b1 = static_cast<std::uint8_t>(row_packed[byte_index + 1]);
  const std::uint8_t q0 = static_cast<std::uint8_t>(static_cast<int8_t>(decode_signed_int4(b0 & 0x0Fu)));
  const std::uint8_t q1 = static_cast<std::uint8_t>(static_cast<int8_t>(decode_signed_int4((b0 >> 4) & 0x0Fu)));
  const std::uint8_t q2 = static_cast<std::uint8_t>(static_cast<int8_t>(decode_signed_int4(b1 & 0x0Fu)));
  const std::uint8_t q3 = static_cast<std::uint8_t>(static_cast<int8_t>(decode_signed_int4((b1 >> 4) & 0x0Fu)));
  return static_cast<int>(q0) |
         (static_cast<int>(q1) << 8) |
         (static_cast<int>(q2) << 16) |
         (static_cast<int>(q3) << 24);
}

__global__ void weight_only_int8_matvec_kernel(const int8_t* w,
                                               const float* scales,
                                               const half* x,
                                               half* y,
                                               int out_features,
                                               int in_features) {
  extern __shared__ float ssum[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= out_features) {
    return;
  }

  float local = 0.0f;
  const int base = row * in_features;
  for (int col = tid; col < in_features; col += blockDim.x) {
    local += static_cast<float>(w[base + col]) * __half2float(x[col]);
  }

  {
    const int lane = tid & (warpSize - 1);
    const int warp_id = tid / warpSize;
    float dot = warp_sum(local);
    if (lane == 0) {
      ssum[warp_id] = dot;
    }
  }
  __syncthreads();

  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < blockDim.x / warpSize; ++w) {
      total += ssum[w];
    }
    y[row] = __float2half(total * scales[row]);
  }
}

__global__ void weight_only_int8_matvec_batched_kernel(const int8_t* w,
                                                       const float* scales,
                                                       const half* x,
                                                       half* y,
                                                       int batch_size,
                                                       int out_features,
                                                       int in_features) {
  extern __shared__ float ssum[];
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= out_features || batch >= batch_size) {
    return;
  }

  float local = 0.0f;
  const int w_base = row * in_features;
  const int x_base = batch * in_features;
  for (int col = tid; col < in_features; col += blockDim.x) {
    local += static_cast<float>(w[w_base + col]) * __half2float(x[x_base + col]);
  }

  ssum[tid] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      ssum[tid] += ssum[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    y[batch * out_features + row] = __float2half(ssum[0] * scales[row]);
  }
}

// The dp4a path consumes 16-byte chunks first, then packed int32 groups, and
// finally a scalar tail so one kernel covers arbitrary input widths.
__global__ void weight_only_int8_matvec_batched_dp4a_kernel(const int8_t* w,
                                                            const float* w_scales,
                                                            const int8_t* x,
                                                            const float* x_scales,
                                                            half* y,
                                                            int batch_size,
                                                            int out_features,
                                                            int in_features) {
  __shared__ int warp_sums[32];
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= out_features || batch >= batch_size) {
    return;
  }

  const int w_base = row * in_features;
  const int x_base = batch * in_features;
  int local = 0;
  int consumed = 0;

  if ((in_features & 15) == 0) {
    const int packed16 = in_features / 16;
    const int4* w16 = reinterpret_cast<const int4*>(w + w_base);
    const int4* x16 = reinterpret_cast<const int4*>(x + x_base);
    for (int idx = tid; idx < packed16; idx += blockDim.x) {
      const int4 wv = w16[idx];
      const int4 xv = x16[idx];
      local = __dp4a(wv.x, xv.x, local);
      local = __dp4a(wv.y, xv.y, local);
      local = __dp4a(wv.z, xv.z, local);
      local = __dp4a(wv.w, xv.w, local);
    }
    consumed = packed16 * 16;
  }

  const int packed4 = (in_features - consumed) / 4;
  const int* w4 = reinterpret_cast<const int*>(w + w_base + consumed);
  const int* x4 = reinterpret_cast<const int*>(x + x_base + consumed);
  for (int idx = tid; idx < packed4; idx += blockDim.x) {
    local = __dp4a(w4[idx], x4[idx], local);
  }

  consumed += packed4 * 4;
  for (int col = consumed + tid; col < in_features; col += blockDim.x) {
    local += static_cast<int>(w[w_base + col]) * static_cast<int>(x[x_base + col]);
  }

  local = warp_sum_int(local);
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
  if (lane == 0) {
    warp_sums[warp] = local;
  }
  __syncthreads();

  if (warp == 0) {
    int block_sum = (lane < warp_count) ? warp_sums[lane] : 0;
    block_sum = warp_sum_int(block_sum);
    if (lane == 0) {
      const float scale = w_scales[row] * x_scales[batch];
      y[batch * out_features + row] = __float2half(static_cast<float>(block_sum) * scale);
    }
  }
}

// Tiled dp4a GEMV stages the activation tile once per block and can split the
// K dimension across multiple warps that cooperate on the same output row.
template <int TotalWarps, int TilePacked4, int WarpsPerRow>
__global__ void weight_only_int8_matvec_dp4a_tiled_kernel(const int8_t* w,
                                                          const float* w_scales,
                                                          const int8_t* x,
                                                          const float* x_scale,
                                                          half* y,
                                                          int out_features,
                                                          int in_features) {
  static_assert(TotalWarps % WarpsPerRow == 0, "TotalWarps must be divisible by WarpsPerRow");
  constexpr int RowsPerBlock = TotalWarps / WarpsPerRow;
  __shared__ int x_tile[TilePacked4];
  __shared__ int partial_sums[RowsPerBlock][WarpsPerRow];
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane = tid & (warpSize - 1);
  const int row_group = warp_id / WarpsPerRow;
  const int split_warp = warp_id % WarpsPerRow;
  const int row = blockIdx.x * RowsPerBlock + row_group;
  if (row >= out_features) {
    return;
  }

  const int packed4_total = in_features / 4;
  const int* w4 = reinterpret_cast<const int*>(w + static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features));
  int local = 0;

  for (int tile_base = 0; tile_base < packed4_total; tile_base += TilePacked4) {
    const int tile_count = min(TilePacked4, packed4_total - tile_base);
    for (int idx = tid; idx < tile_count; idx += blockDim.x) {
      x_tile[idx] = reinterpret_cast<const int*>(x)[tile_base + idx];
    }
    __syncthreads();

    // Split the K work across several warps that collaborate on the same row
    // while still processing two packed fragments per loop step when possible.
    for (int idx = split_warp * warpSize + lane; idx < tile_count; idx += WarpsPerRow * warpSize * 2) {
      local = __dp4a(w4[tile_base + idx], x_tile[idx], local);
      const int idx_next = idx + WarpsPerRow * warpSize;
      if (idx_next < tile_count) {
        local = __dp4a(w4[tile_base + idx_next], x_tile[idx_next], local);
      }
    }
    __syncthreads();
  }

  const int consumed = packed4_total * 4;
  for (int col = consumed + lane; col < in_features; col += warpSize) {
    local += static_cast<int>(w[static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features) + static_cast<std::size_t>(col)]) *
             static_cast<int>(x[col]);
  }

  local = warp_sum_int(local);
  if (lane == 0) {
    partial_sums[row_group][split_warp] = local;
  }
  __syncthreads();

  if (split_warp == 0) {
    int block_sum = (lane < WarpsPerRow) ? partial_sums[row_group][lane] : 0;
    block_sum = warp_sum_int(block_sum);
    if (lane == 0) {
      const float scale = w_scales[row] * x_scale[0];
      y[row] = __float2half(static_cast<float>(block_sum) * scale);
    }
  }
}

// Dual-output variant that reuses the staged activation tile for two
// independent weight matrices.
template <int TotalWarps, int TilePacked4, int WarpsPerRow>
__global__ void weight_only_int8_matvec_dual_dp4a_tiled_kernel(const int8_t* w_a,
                                                               const float* w_scales_a,
                                                               const int8_t* w_b,
                                                               const float* w_scales_b,
                                                               const int8_t* x,
                                                               const float* x_scale,
                                                               half* y_a,
                                                               half* y_b,
                                                               int out_features,
                                                               int in_features) {
  static_assert(TotalWarps % WarpsPerRow == 0, "TotalWarps must be divisible by WarpsPerRow");
  constexpr int RowsPerBlock = TotalWarps / WarpsPerRow;
  __shared__ int x_tile[TilePacked4];
  __shared__ int partial_sums_a[RowsPerBlock][WarpsPerRow];
  __shared__ int partial_sums_b[RowsPerBlock][WarpsPerRow];
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane = tid & (warpSize - 1);
  const int row_group = warp_id / WarpsPerRow;
  const int split_warp = warp_id % WarpsPerRow;
  const int row = blockIdx.x * RowsPerBlock + row_group;
  if (row >= out_features) {
    return;
  }

  const int packed4_total = in_features / 4;
  const int* wa4 = reinterpret_cast<const int*>(w_a + static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features));
  const int* wb4 = reinterpret_cast<const int*>(w_b + static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features));
  int local_a = 0;
  int local_b = 0;

  for (int tile_base = 0; tile_base < packed4_total; tile_base += TilePacked4) {
    const int tile_count = min(TilePacked4, packed4_total - tile_base);
    for (int idx = tid; idx < tile_count; idx += blockDim.x) {
      x_tile[idx] = reinterpret_cast<const int*>(x)[tile_base + idx];
    }
    __syncthreads();

    // Reuse each staged x tile across both outputs and split the K work across
    // several warps when configured to do so.
    for (int idx = split_warp * warpSize + lane; idx < tile_count; idx += WarpsPerRow * warpSize * 2) {
      const int xv0 = x_tile[idx];
      local_a = __dp4a(wa4[tile_base + idx], xv0, local_a);
      local_b = __dp4a(wb4[tile_base + idx], xv0, local_b);
      const int idx_next = idx + WarpsPerRow * warpSize;
      if (idx_next < tile_count) {
        const int xv1 = x_tile[idx_next];
        local_a = __dp4a(wa4[tile_base + idx_next], xv1, local_a);
        local_b = __dp4a(wb4[tile_base + idx_next], xv1, local_b);
      }
    }
    __syncthreads();
  }

  const int consumed = packed4_total * 4;
  for (int col = consumed + lane; col < in_features; col += warpSize) {
    const int xv = static_cast<int>(x[col]);
    const std::size_t offset = static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features) + static_cast<std::size_t>(col);
    local_a += static_cast<int>(w_a[offset]) * xv;
    local_b += static_cast<int>(w_b[offset]) * xv;
  }

  local_a = warp_sum_int(local_a);
  local_b = warp_sum_int(local_b);
  if (lane == 0) {
    partial_sums_a[row_group][split_warp] = local_a;
    partial_sums_b[row_group][split_warp] = local_b;
  }
  __syncthreads();

  if (split_warp == 0) {
    int block_sum_a = (lane < WarpsPerRow) ? partial_sums_a[row_group][lane] : 0;
    int block_sum_b = (lane < WarpsPerRow) ? partial_sums_b[row_group][lane] : 0;
    block_sum_a = warp_sum_int(block_sum_a);
    block_sum_b = warp_sum_int(block_sum_b);
    if (lane == 0) {
      const float scale = x_scale[0];
      y_a[row] = __float2half(static_cast<float>(block_sum_a) * w_scales_a[row] * scale);
      y_b[row] = __float2half(static_cast<float>(block_sum_b) * w_scales_b[row] * scale);
    }
  }
}

__global__ void weight_only_int4_matvec_kernel(const int8_t* w_packed,
                                               const float* scales,
                                               const half* x,
                                               half* y,
                                               int out_features,
                                               int in_features) {
  extern __shared__ float ssum[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= out_features) {
    return;
  }

  const int packed_cols = (in_features + 1) / 2;
  const int8_t* row_w = w_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);

  float local = 0.0f;
  for (int col = tid; col < in_features; col += blockDim.x) {
    local += static_cast<float>(load_signed_int4(row_w, col)) * __half2float(x[col]);
  }

  {
    const int lane = tid & (warpSize - 1);
    const int warp_id = tid / warpSize;
    float dot = warp_sum(local);
    if (lane == 0) {
      ssum[warp_id] = dot;
    }
  }
  __syncthreads();

  if (tid == 0) {
    float total = 0.0f;
    for (int w = 0; w < blockDim.x / warpSize; ++w) {
      total += ssum[w];
    }
    y[row] = __float2half(total * scales[row]);
  }
}

__global__ void weight_only_int4_matvec_batched_kernel(const int8_t* w_packed,
                                                       const float* scales,
                                                       const half* x,
                                                       half* y,
                                                       int batch_size,
                                                       int out_features,
                                                       int in_features) {
  extern __shared__ float ssum[];
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= out_features || batch >= batch_size) {
    return;
  }

  const int packed_cols = (in_features + 1) / 2;
  const int8_t* row_w = w_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
  const int x_base = batch * in_features;

  float local = 0.0f;
  for (int col = tid; col < in_features; col += blockDim.x) {
    local += static_cast<float>(load_signed_int4(row_w, col)) * __half2float(x[x_base + col]);
  }

  ssum[tid] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      ssum[tid] += ssum[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    y[batch * out_features + row] = __float2half(ssum[0] * scales[row]);
  }
}

__global__ void weight_only_int4_matvec_batched_dp4a_kernel(const int8_t* w_packed,
                                                            const float* w_scales,
                                                            const int8_t* x,
                                                            const float* x_scales,
                                                            half* y,
                                                            int batch_size,
                                                            int out_features,
                                                            int in_features) {
  __shared__ int warp_sums[32];
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= out_features || batch >= batch_size) {
    return;
  }

  const int packed_cols = (in_features + 1) / 2;
  const int8_t* row_w = w_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
  const int x_base = batch * in_features;
  const int* x4 = reinterpret_cast<const int*>(x + x_base);
  const int packed4_total = in_features / 4;

  int local = 0;
  for (int idx = tid; idx < packed4_total; idx += blockDim.x) {
    const int w4 = load_packed_int4x4(row_w, idx);
    local = __dp4a(w4, x4[idx], local);
  }

  const int consumed = packed4_total * 4;
  for (int col = consumed + tid; col < in_features; col += blockDim.x) {
    local += static_cast<int>(load_signed_int4(row_w, col)) * static_cast<int>(x[x_base + col]);
  }

  local = warp_sum_int(local);
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
  if (lane == 0) {
    warp_sums[warp] = local;
  }
  __syncthreads();

  if (warp == 0) {
    int block_sum = (lane < warp_count) ? warp_sums[lane] : 0;
    block_sum = warp_sum_int(block_sum);
    if (lane == 0) {
      const float scale = w_scales[row] * x_scales[batch];
      y[batch * out_features + row] = __float2half(static_cast<float>(block_sum) * scale);
    }
  }
}

template <int TotalWarps, int TilePacked4, int WarpsPerRow>
__global__ void weight_only_int4_matvec_dp4a_tiled_kernel(const int8_t* w_packed,
                                                          const float* w_scales,
                                                          const int8_t* x,
                                                          const float* x_scale,
                                                          half* y,
                                                          int out_features,
                                                          int in_features) {
  static_assert(TotalWarps % WarpsPerRow == 0, "TotalWarps must be divisible by WarpsPerRow");
  constexpr int RowsPerBlock = TotalWarps / WarpsPerRow;
  __shared__ int x_tile[TilePacked4];
  __shared__ int partial_sums[RowsPerBlock][WarpsPerRow];
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane = tid & (warpSize - 1);
  const int row_group = warp_id / WarpsPerRow;
  const int split_warp = warp_id % WarpsPerRow;
  const int row = blockIdx.x * RowsPerBlock + row_group;
  if (row >= out_features) {
    return;
  }

  const int packed_cols = (in_features + 1) / 2;
  const int8_t* row_w = w_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
  const int packed4_total = in_features / 4;
  int local = 0;

  for (int tile_base = 0; tile_base < packed4_total; tile_base += TilePacked4) {
    const int tile_count = min(TilePacked4, packed4_total - tile_base);
    for (int idx = tid; idx < tile_count; idx += blockDim.x) {
      x_tile[idx] = reinterpret_cast<const int*>(x)[tile_base + idx];
    }
    __syncthreads();

    for (int idx = split_warp * warpSize + lane; idx < tile_count; idx += WarpsPerRow * warpSize * 2) {
      local = __dp4a(load_packed_int4x4(row_w, tile_base + idx), x_tile[idx], local);
      const int idx_next = idx + WarpsPerRow * warpSize;
      if (idx_next < tile_count) {
        local = __dp4a(load_packed_int4x4(row_w, tile_base + idx_next), x_tile[idx_next], local);
      }
    }
    __syncthreads();
  }

  const int consumed = packed4_total * 4;
  for (int col = consumed + lane; col < in_features; col += warpSize) {
    local += static_cast<int>(load_signed_int4(row_w, col)) * static_cast<int>(x[col]);
  }

  local = warp_sum_int(local);
  if (lane == 0) {
    partial_sums[row_group][split_warp] = local;
  }
  __syncthreads();

  if (split_warp == 0) {
    int block_sum = (lane < WarpsPerRow) ? partial_sums[row_group][lane] : 0;
    block_sum = warp_sum_int(block_sum);
    if (lane == 0) {
      const float scale = w_scales[row] * x_scale[0];
      y[row] = __float2half(static_cast<float>(block_sum) * scale);
    }
  }
}

template <int TotalWarps, int TilePacked4, int WarpsPerRow>
__global__ void weight_only_int4_matvec_dual_dp4a_tiled_kernel(const int8_t* w_a_packed,
                                                               const float* w_scales_a,
                                                               const int8_t* w_b_packed,
                                                               const float* w_scales_b,
                                                               const int8_t* x,
                                                               const float* x_scale,
                                                               half* y_a,
                                                               half* y_b,
                                                               int out_features,
                                                               int in_features) {
  static_assert(TotalWarps % WarpsPerRow == 0, "TotalWarps must be divisible by WarpsPerRow");
  constexpr int RowsPerBlock = TotalWarps / WarpsPerRow;
  __shared__ int x_tile[TilePacked4];
  __shared__ int partial_sums_a[RowsPerBlock][WarpsPerRow];
  __shared__ int partial_sums_b[RowsPerBlock][WarpsPerRow];
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane = tid & (warpSize - 1);
  const int row_group = warp_id / WarpsPerRow;
  const int split_warp = warp_id % WarpsPerRow;
  const int row = blockIdx.x * RowsPerBlock + row_group;
  if (row >= out_features) {
    return;
  }

  const int packed_cols = (in_features + 1) / 2;
  const int8_t* row_wa = w_a_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
  const int8_t* row_wb = w_b_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
  const int packed4_total = in_features / 4;
  int local_a = 0;
  int local_b = 0;

  for (int tile_base = 0; tile_base < packed4_total; tile_base += TilePacked4) {
    const int tile_count = min(TilePacked4, packed4_total - tile_base);
    for (int idx = tid; idx < tile_count; idx += blockDim.x) {
      x_tile[idx] = reinterpret_cast<const int*>(x)[tile_base + idx];
    }
    __syncthreads();

    for (int idx = split_warp * warpSize + lane; idx < tile_count; idx += WarpsPerRow * warpSize * 2) {
      const int xv0 = x_tile[idx];
      local_a = __dp4a(load_packed_int4x4(row_wa, tile_base + idx), xv0, local_a);
      local_b = __dp4a(load_packed_int4x4(row_wb, tile_base + idx), xv0, local_b);
      const int idx_next = idx + WarpsPerRow * warpSize;
      if (idx_next < tile_count) {
        const int xv1 = x_tile[idx_next];
        local_a = __dp4a(load_packed_int4x4(row_wa, tile_base + idx_next), xv1, local_a);
        local_b = __dp4a(load_packed_int4x4(row_wb, tile_base + idx_next), xv1, local_b);
      }
    }
    __syncthreads();
  }

  const int consumed = packed4_total * 4;
  for (int col = consumed + lane; col < in_features; col += warpSize) {
    const int xv = static_cast<int>(x[col]);
    local_a += static_cast<int>(load_signed_int4(row_wa, col)) * xv;
    local_b += static_cast<int>(load_signed_int4(row_wb, col)) * xv;
  }

  local_a = warp_sum_int(local_a);
  local_b = warp_sum_int(local_b);
  if (lane == 0) {
    partial_sums_a[row_group][split_warp] = local_a;
    partial_sums_b[row_group][split_warp] = local_b;
  }
  __syncthreads();

  if (split_warp == 0) {
    int block_sum_a = (lane < WarpsPerRow) ? partial_sums_a[row_group][lane] : 0;
    int block_sum_b = (lane < WarpsPerRow) ? partial_sums_b[row_group][lane] : 0;
    block_sum_a = warp_sum_int(block_sum_a);
    block_sum_b = warp_sum_int(block_sum_b);
    if (lane == 0) {
      const float scale = x_scale[0];
      y_a[row] = __float2half(static_cast<float>(block_sum_a) * w_scales_a[row] * scale);
      y_b[row] = __float2half(static_cast<float>(block_sum_b) * w_scales_b[row] * scale);
    }
  }
}


}  // namespace

// Host launch wrappers for weight-only int8/int4 matvec kernels.
void launch_weight_only_int8_matvec(const int8_t* w,
                                    const float* scales,
                                    const half* x,
                                    half* y,
                                    int out_features,
                                    int in_features,
                                    cudaStream_t stream) {
  const int threads = choose_reduction_threads(in_features);
  weight_only_int8_matvec_kernel<<<out_features, threads, static_cast<std::size_t>(threads) * sizeof(float), stream>>>(
      w, scales, x, y, out_features, in_features);
}

void launch_weight_only_int8_matvec_batched(const int8_t* w,
                                            const float* scales,
                                            const half* x,
                                            half* y,
                                            int batch_size,
                                            int out_features,
                                            int in_features,
                                            cudaStream_t stream) {
  const int threads = choose_reduction_threads(in_features);
  const dim3 grid(out_features, batch_size);
  weight_only_int8_matvec_batched_kernel<<<grid, threads, static_cast<std::size_t>(threads) * sizeof(float), stream>>>(
      w, scales, x, y, batch_size, out_features, in_features);
}

void launch_weight_only_int8_matvec_batched_dp4a(const int8_t* w,
                                                 const float* w_scales,
                                                 const int8_t* x,
                                                 const float* x_scales,
                                                 half* y,
                                                 int batch_size,
                                                 int out_features,
                                                 int in_features,
                                                 cudaStream_t stream) {
  const int threads = choose_reduction_threads(in_features);
  const dim3 grid(out_features, batch_size);
  weight_only_int8_matvec_batched_dp4a_kernel<<<grid, threads, 0, stream>>>(
      w, w_scales, x, x_scales, y, batch_size, out_features, in_features);
}

void launch_weight_only_int8_matvec_dp4a(const int8_t* w,
                                         const float* w_scales,
                                         const int8_t* x,
                                         const float* x_scale,
                                         half* y,
                                         int out_features,
                                         int in_features,
                                         cudaStream_t stream,
                                         int warps_per_block,
                                         int tile_packed4,
                                         int warps_per_row) {
  const int warps = (warps_per_block > 0) ? warps_per_block : 4;
  const int tile = (tile_packed4 > 0) ? tile_packed4 : 128;
  int split = (warps_per_row > 0) ? warps_per_row : 1;
  if (split > warps || (warps % split) != 0) {
    split = 1;
  }
  auto launch = [&](auto warps_tag, auto tile_tag, auto split_tag) {
    constexpr int kWarps = decltype(warps_tag)::value;
    constexpr int kTile = decltype(tile_tag)::value;
    constexpr int kSplit = decltype(split_tag)::value;
    constexpr int kThreads = kWarps * 32;
    constexpr int kRowsPerBlock = kWarps / kSplit;
    const int blocks = (out_features + kRowsPerBlock - 1) / kRowsPerBlock;
    weight_only_int8_matvec_dp4a_tiled_kernel<kWarps, kTile, kSplit><<<blocks, kThreads, 0, stream>>>(
        w, w_scales, x, x_scale, y, out_features, in_features);
  };

  auto launch_split = [&](auto warps_tag, auto tile_tag) {
    if (split >= 4 && (decltype(warps_tag)::value % 4) == 0) {
      launch(warps_tag, tile_tag, std::integral_constant<int, 4>{});
    } else if (split >= 2 && (decltype(warps_tag)::value % 2) == 0) {
      launch(warps_tag, tile_tag, std::integral_constant<int, 2>{});
    } else {
      launch(warps_tag, tile_tag, std::integral_constant<int, 1>{});
    }
  };

  if (warps == 8) {
    if (tile >= 512) {
      launch_split(std::integral_constant<int, 8>{}, std::integral_constant<int, 512>{});
    } else if (tile >= 256) {
      launch_split(std::integral_constant<int, 8>{}, std::integral_constant<int, 256>{});
    } else {
      launch_split(std::integral_constant<int, 8>{}, std::integral_constant<int, 128>{});
    }
    return;
  }

  if (warps == 16) {
    if (tile >= 512) {
      launch_split(std::integral_constant<int, 16>{}, std::integral_constant<int, 512>{});
    } else if (tile >= 256) {
      launch_split(std::integral_constant<int, 16>{}, std::integral_constant<int, 256>{});
    } else {
      launch_split(std::integral_constant<int, 16>{}, std::integral_constant<int, 128>{});
    }
    return;
  }

  if (tile >= 512) {
    launch_split(std::integral_constant<int, 4>{}, std::integral_constant<int, 512>{});
  } else if (tile >= 256) {
    launch_split(std::integral_constant<int, 4>{}, std::integral_constant<int, 256>{});
  } else {
    launch_split(std::integral_constant<int, 4>{}, std::integral_constant<int, 128>{});
  }
}

void launch_weight_only_int8_matvec_dual_dp4a(const int8_t* w_a,
                                              const float* w_scales_a,
                                              const int8_t* w_b,
                                              const float* w_scales_b,
                                              const int8_t* x,
                                              const float* x_scale,
                                              half* y_a,
                                              half* y_b,
                                              int out_features,
                                              int in_features,
                                              cudaStream_t stream,
                                              int warps_per_block,
                                              int tile_packed4,
                                              int warps_per_row) {
  const int warps = (warps_per_block > 0) ? warps_per_block : 4;
  const int tile = (tile_packed4 > 0) ? tile_packed4 : 128;
  int split = (warps_per_row > 0) ? warps_per_row : 1;
  if (split > warps || (warps % split) != 0) {
    split = 1;
  }
  auto launch = [&](auto warps_tag, auto tile_tag, auto split_tag) {
    constexpr int kWarps = decltype(warps_tag)::value;
    constexpr int kTile = decltype(tile_tag)::value;
    constexpr int kSplit = decltype(split_tag)::value;
    constexpr int kThreads = kWarps * 32;
    constexpr int kRowsPerBlock = kWarps / kSplit;
    const int blocks = (out_features + kRowsPerBlock - 1) / kRowsPerBlock;
    weight_only_int8_matvec_dual_dp4a_tiled_kernel<kWarps, kTile, kSplit><<<blocks, kThreads, 0, stream>>>(
        w_a, w_scales_a, w_b, w_scales_b, x, x_scale, y_a, y_b, out_features, in_features);
  };

  auto launch_split = [&](auto warps_tag, auto tile_tag) {
    if (split >= 4 && (decltype(warps_tag)::value % 4) == 0) {
      launch(warps_tag, tile_tag, std::integral_constant<int, 4>{});
    } else if (split >= 2 && (decltype(warps_tag)::value % 2) == 0) {
      launch(warps_tag, tile_tag, std::integral_constant<int, 2>{});
    } else {
      launch(warps_tag, tile_tag, std::integral_constant<int, 1>{});
    }
  };

  if (warps == 8) {
    if (tile >= 512) {
      launch_split(std::integral_constant<int, 8>{}, std::integral_constant<int, 512>{});
    } else if (tile >= 256) {
      launch_split(std::integral_constant<int, 8>{}, std::integral_constant<int, 256>{});
    } else {
      launch_split(std::integral_constant<int, 8>{}, std::integral_constant<int, 128>{});
    }
    return;
  }

  if (warps == 16) {
    if (tile >= 512) {
      launch_split(std::integral_constant<int, 16>{}, std::integral_constant<int, 512>{});
    } else if (tile >= 256) {
      launch_split(std::integral_constant<int, 16>{}, std::integral_constant<int, 256>{});
    } else {
      launch_split(std::integral_constant<int, 16>{}, std::integral_constant<int, 128>{});
    }
    return;
  }

  if (tile >= 512) {
    launch_split(std::integral_constant<int, 4>{}, std::integral_constant<int, 512>{});
  } else if (tile >= 256) {
    launch_split(std::integral_constant<int, 4>{}, std::integral_constant<int, 256>{});
  } else {
    launch_split(std::integral_constant<int, 4>{}, std::integral_constant<int, 128>{});
  }
}

void launch_weight_only_int4_matvec(const int8_t* w_packed,
                                    const float* scales,
                                    const half* x,
                                    half* y,
                                    int out_features,
                                    int in_features,
                                    cudaStream_t stream) {
  const int threads = choose_reduction_threads(in_features);
  weight_only_int4_matvec_kernel<<<out_features, threads, static_cast<std::size_t>(threads) * sizeof(float), stream>>>(
      w_packed, scales, x, y, out_features, in_features);
}

void launch_weight_only_int4_matvec_batched(const int8_t* w_packed,
                                            const float* scales,
                                            const half* x,
                                            half* y,
                                            int batch_size,
                                            int out_features,
                                            int in_features,
                                            cudaStream_t stream) {
  const int threads = choose_reduction_threads(in_features);
  const dim3 grid(out_features, batch_size);
  weight_only_int4_matvec_batched_kernel<<<grid, threads, static_cast<std::size_t>(threads) * sizeof(float), stream>>>(
      w_packed, scales, x, y, batch_size, out_features, in_features);
}

void launch_weight_only_int4_matvec_batched_dp4a(const int8_t* w_packed,
                                                 const float* w_scales,
                                                 const int8_t* x,
                                                 const float* x_scales,
                                                 half* y,
                                                 int batch_size,
                                                 int out_features,
                                                 int in_features,
                                                 cudaStream_t stream) {
  const int threads = choose_reduction_threads(in_features);
  const dim3 grid(out_features, batch_size);
  weight_only_int4_matvec_batched_dp4a_kernel<<<grid, threads, 0, stream>>>(
      w_packed, w_scales, x, x_scales, y, batch_size, out_features, in_features);
}

void launch_weight_only_int4_matvec_dp4a(const int8_t* w_packed,
                                         const float* w_scales,
                                         const int8_t* x,
                                         const float* x_scale,
                                         half* y,
                                         int out_features,
                                         int in_features,
                                         cudaStream_t stream,
                                         int warps_per_block,
                                         int tile_packed4,
                                         int warps_per_row) {
  const int warps = (warps_per_block > 0) ? warps_per_block : 4;
  const int tile = (tile_packed4 > 0) ? tile_packed4 : 128;
  int split = (warps_per_row > 0) ? warps_per_row : 1;
  if (split > warps || (warps % split) != 0) {
    split = 1;
  }
  auto launch = [&](auto warps_tag, auto tile_tag, auto split_tag) {
    constexpr int kWarps = decltype(warps_tag)::value;
    constexpr int kTile = decltype(tile_tag)::value;
    constexpr int kSplit = decltype(split_tag)::value;
    constexpr int kThreads = kWarps * 32;
    constexpr int kRowsPerBlock = kWarps / kSplit;
    const int blocks = (out_features + kRowsPerBlock - 1) / kRowsPerBlock;
    weight_only_int4_matvec_dp4a_tiled_kernel<kWarps, kTile, kSplit><<<blocks, kThreads, 0, stream>>>(
        w_packed, w_scales, x, x_scale, y, out_features, in_features);
  };

  auto launch_split = [&](auto warps_tag, auto tile_tag) {
    if (split >= 4 && (decltype(warps_tag)::value % 4) == 0) {
      launch(warps_tag, tile_tag, std::integral_constant<int, 4>{});
    } else if (split >= 2 && (decltype(warps_tag)::value % 2) == 0) {
      launch(warps_tag, tile_tag, std::integral_constant<int, 2>{});
    } else {
      launch(warps_tag, tile_tag, std::integral_constant<int, 1>{});
    }
  };

  if (warps == 8) {
    if (tile >= 512) {
      launch_split(std::integral_constant<int, 8>{}, std::integral_constant<int, 512>{});
    } else if (tile >= 256) {
      launch_split(std::integral_constant<int, 8>{}, std::integral_constant<int, 256>{});
    } else {
      launch_split(std::integral_constant<int, 8>{}, std::integral_constant<int, 128>{});
    }
    return;
  }

  if (warps == 16) {
    if (tile >= 512) {
      launch_split(std::integral_constant<int, 16>{}, std::integral_constant<int, 512>{});
    } else if (tile >= 256) {
      launch_split(std::integral_constant<int, 16>{}, std::integral_constant<int, 256>{});
    } else {
      launch_split(std::integral_constant<int, 16>{}, std::integral_constant<int, 128>{});
    }
    return;
  }

  if (tile >= 512) {
    launch_split(std::integral_constant<int, 4>{}, std::integral_constant<int, 512>{});
  } else if (tile >= 256) {
    launch_split(std::integral_constant<int, 4>{}, std::integral_constant<int, 256>{});
  } else {
    launch_split(std::integral_constant<int, 4>{}, std::integral_constant<int, 128>{});
  }
}

void launch_weight_only_int4_matvec_dual_dp4a(const int8_t* w_a_packed,
                                              const float* w_scales_a,
                                              const int8_t* w_b_packed,
                                              const float* w_scales_b,
                                              const int8_t* x,
                                              const float* x_scale,
                                              half* y_a,
                                              half* y_b,
                                              int out_features,
                                              int in_features,
                                              cudaStream_t stream,
                                              int warps_per_block,
                                              int tile_packed4,
                                              int warps_per_row) {
  const int warps = (warps_per_block > 0) ? warps_per_block : 4;
  const int tile = (tile_packed4 > 0) ? tile_packed4 : 128;
  int split = (warps_per_row > 0) ? warps_per_row : 1;
  if (split > warps || (warps % split) != 0) {
    split = 1;
  }
  auto launch = [&](auto warps_tag, auto tile_tag, auto split_tag) {
    constexpr int kWarps = decltype(warps_tag)::value;
    constexpr int kTile = decltype(tile_tag)::value;
    constexpr int kSplit = decltype(split_tag)::value;
    constexpr int kThreads = kWarps * 32;
    constexpr int kRowsPerBlock = kWarps / kSplit;
    const int blocks = (out_features + kRowsPerBlock - 1) / kRowsPerBlock;
    weight_only_int4_matvec_dual_dp4a_tiled_kernel<kWarps, kTile, kSplit><<<blocks, kThreads, 0, stream>>>(
        w_a_packed, w_scales_a, w_b_packed, w_scales_b, x, x_scale, y_a, y_b, out_features, in_features);
  };

  auto launch_split = [&](auto warps_tag, auto tile_tag) {
    if (split >= 4 && (decltype(warps_tag)::value % 4) == 0) {
      launch(warps_tag, tile_tag, std::integral_constant<int, 4>{});
    } else if (split >= 2 && (decltype(warps_tag)::value % 2) == 0) {
      launch(warps_tag, tile_tag, std::integral_constant<int, 2>{});
    } else {
      launch(warps_tag, tile_tag, std::integral_constant<int, 1>{});
    }
  };

  if (warps == 8) {
    if (tile >= 512) {
      launch_split(std::integral_constant<int, 8>{}, std::integral_constant<int, 512>{});
    } else if (tile >= 256) {
      launch_split(std::integral_constant<int, 8>{}, std::integral_constant<int, 256>{});
    } else {
      launch_split(std::integral_constant<int, 8>{}, std::integral_constant<int, 128>{});
    }
    return;
  }

  if (warps == 16) {
    if (tile >= 512) {
      launch_split(std::integral_constant<int, 16>{}, std::integral_constant<int, 512>{});
    } else if (tile >= 256) {
      launch_split(std::integral_constant<int, 16>{}, std::integral_constant<int, 256>{});
    } else {
      launch_split(std::integral_constant<int, 16>{}, std::integral_constant<int, 128>{});
    }
    return;
  }

  if (tile >= 512) {
    launch_split(std::integral_constant<int, 4>{}, std::integral_constant<int, 512>{});
  } else if (tile >= 256) {
    launch_split(std::integral_constant<int, 4>{}, std::integral_constant<int, 256>{});
  } else {
    launch_split(std::integral_constant<int, 4>{}, std::integral_constant<int, 128>{});
  }
}


}  // namespace kernels
