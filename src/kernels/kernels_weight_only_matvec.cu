// kernels_weight_only_matvec.cu
//
// CUDA kernels and host launch wrappers for weight-only int8/int4 matvec paths.

#include <cuda_fp16.h>
#include <sm_61_intrinsics.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "runtime/kernels.cuh"

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

// Decodes the two packed bytes of a dp4a group into one int32 of four int8 lanes.
__device__ __forceinline__ int unpack_int4x4(std::uint8_t b0, std::uint8_t b1) {
  const std::uint8_t q0 =
      static_cast<std::uint8_t>(static_cast<int8_t>(decode_signed_int4(b0 & 0x0Fu)));
  const std::uint8_t q1 =
      static_cast<std::uint8_t>(static_cast<int8_t>(decode_signed_int4((b0 >> 4) & 0x0Fu)));
  const std::uint8_t q2 =
      static_cast<std::uint8_t>(static_cast<int8_t>(decode_signed_int4(b1 & 0x0Fu)));
  const std::uint8_t q3 =
      static_cast<std::uint8_t>(static_cast<int8_t>(decode_signed_int4((b1 >> 4) & 0x0Fu)));
  return static_cast<int>(q0) | (static_cast<int>(q1) << 8) | (static_cast<int>(q2) << 16) |
         (static_cast<int>(q3) << 24);
}

// Decodes 4 consecutive signed int4 values into one packed int32 suitable for dp4a.
//
// Fetches the two packed bytes as a single 16-bit load rather than two 8-bit loads. They are
// adjacent and byte_index is always even, so the access is aligned whenever the row itself is
// 2-byte aligned (packed_cols even, which holds for every real in_features). The odd-row case
// falls back to byte-wise reads.
//
// Same bytes, same nibble decode, same dp4a: only the access width differs.
__device__ __forceinline__ int load_packed_int4x4(const int8_t* row_packed, int packed4_index) {
  const int byte_index = packed4_index * 2;
  if ((reinterpret_cast<std::uintptr_t>(row_packed) & 1u) == 0u) {
    const std::uint16_t pair =
        reinterpret_cast<const std::uint16_t*>(row_packed)[packed4_index];
    return unpack_int4x4(static_cast<std::uint8_t>(pair & 0xFFu),
                         static_cast<std::uint8_t>((pair >> 8) & 0xFFu));
  }
  return unpack_int4x4(static_cast<std::uint8_t>(row_packed[byte_index + 0]),
                       static_cast<std::uint8_t>(row_packed[byte_index + 1]));
}

// int8 weight-only GEMV with fp16 activations and fp32 output: the LM head.
//
// Separate from the dp4a matvecs for two reasons. The LM head must emit fp32 logits (the argmax
// and the top-k sampler read floats) and every dp4a kernel writes half; and it quantizes the
// WEIGHTS ONLY -- x stays fp16 and the dot accumulates in fp32. The dp4a path would also
// quantize x, which is the wrong trade in the layer that selects the output token: the
// activation is a few KB against a weight matrix measured in GB.
//
// 128-bit loads on both operands: one int4 holds 16 int8 weights or 8 halves of x, so each
// weight vector pairs with two x vectors.
template <int Warps>
__global__ void weight_only_int8_gemv_f32_kernel(const int8_t* __restrict__ w,
                                                 const float* __restrict__ scales,
                                                 const half* __restrict__ x, float* __restrict__ y,
                                                 int out_features, int in_features) {
  const int row = blockIdx.x * Warps + (threadIdx.x / warpSize);
  if (row >= out_features) {
    return;
  }
  const int lane = threadIdx.x & (warpSize - 1);
  const int8_t* wrow = w + static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features);
  const int vecs = in_features / 16;  // 16 int8 weights per 128-bit load

  float acc = 0.0f;
  for (int v = lane; v < vecs; v += warpSize) {
    const int4 wp = reinterpret_cast<const int4*>(wrow)[v];
    const int8_t* wb = reinterpret_cast<const int8_t*>(&wp);
    const int4 xa = reinterpret_cast<const int4*>(x)[v * 2];
    const int4 xb = reinterpret_cast<const int4*>(x)[v * 2 + 1];
    const half2* xh0 = reinterpret_cast<const half2*>(&xa);
    const half2* xh1 = reinterpret_cast<const half2*>(&xb);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 a = __half22float2(xh0[j]);
      acc += static_cast<float>(wb[j * 2 + 0]) * a.x + static_cast<float>(wb[j * 2 + 1]) * a.y;
      const float2 b = __half22float2(xh1[j]);
      acc += static_cast<float>(wb[8 + j * 2 + 0]) * b.x + static_cast<float>(wb[8 + j * 2 + 1]) * b.y;
    }
  }
  for (int c = vecs * 16 + lane; c < in_features; c += warpSize) {
    acc += static_cast<float>(wrow[c]) * __half2float(x[c]);
  }

  acc = warp_sum(acc);
  if (lane == 0) {
    y[row] = acc * scales[row];
  }
}

__global__ void weight_only_int8_matvec_kernel(const int8_t* w, const float* scales, const half* x,
                                               half* y, int out_features, int in_features) {
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

__global__ void weight_only_int8_matvec_batched_kernel(const int8_t* w, const float* scales,
                                                       const half* x, half* y, int batch_size,
                                                       int out_features, int in_features) {
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
__global__ void weight_only_int8_matvec_batched_dp4a_kernel(const int8_t* w, const float* w_scales,
                                                            const int8_t* x, const float* x_scales,
                                                            half* y, int batch_size,
                                                            int out_features, int in_features) {
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
__global__ void weight_only_int8_matvec_dp4a_tiled_kernel(const int8_t* w, const float* w_scales,
                                                          const int8_t* x, const float* x_scale,
                                                          half* y, int out_features,
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
  const int* w4 = reinterpret_cast<const int*>(w + static_cast<std::size_t>(row) *
                                                       static_cast<std::size_t>(in_features));
  int local = 0;

  for (int tile_base = 0; tile_base < packed4_total; tile_base += TilePacked4) {
    const int tile_count = min(TilePacked4, packed4_total - tile_base);
    for (int idx = tid; idx < tile_count; idx += blockDim.x) {
      x_tile[idx] = reinterpret_cast<const int*>(x)[tile_base + idx];
    }
    __syncthreads();

    // Split the K work across several warps that collaborate on the same row
    // while still processing two packed fragments per loop step when possible.
    // Fetch weights 128 bits at a time: one uint4 = 4 dp4a groups = 16 int8 weights, so
    // consecutive lanes read consecutive 16-byte chunks. Reading one int (32 bits) per dp4a
    // leaves too few bytes in flight. Bit-identical: __dp4a accumulates in int32.
    const int* tile_w4 = w4 + tile_base;
    const bool aligned16 = (reinterpret_cast<std::uintptr_t>(tile_w4) & 15u) == 0u;
    const int vec_groups = aligned16 ? (tile_count >> 2) : 0;  // groups of 4 dp4a units
    for (int v = split_warp * warpSize + lane; v < vec_groups; v += WarpsPerRow * warpSize) {
      const uint4 packed = reinterpret_cast<const uint4*>(tile_w4)[v];
      const int g0 = v << 2;
      local = __dp4a(static_cast<int>(packed.x), x_tile[g0 + 0], local);
      local = __dp4a(static_cast<int>(packed.y), x_tile[g0 + 1], local);
      local = __dp4a(static_cast<int>(packed.z), x_tile[g0 + 2], local);
      local = __dp4a(static_cast<int>(packed.w), x_tile[g0 + 3], local);
    }
    for (int idx = (vec_groups << 2) + split_warp * warpSize + lane; idx < tile_count;
         idx += WarpsPerRow * warpSize) {
      local = __dp4a(w4[tile_base + idx], x_tile[idx], local);
    }
    __syncthreads();
  }

  const int consumed = packed4_total * 4;
  for (int col = consumed + lane; col < in_features; col += warpSize) {
    local +=
        static_cast<int>(w[static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features) +
                           static_cast<std::size_t>(col)]) *
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
__global__ void weight_only_int8_matvec_dual_dp4a_tiled_kernel(
    const int8_t* w_a, const float* w_scales_a, const int8_t* w_b, const float* w_scales_b,
    const int8_t* x, const float* x_scale, half* y_a, half* y_b, int out_features,
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
  const int* wa4 = reinterpret_cast<const int*>(w_a + static_cast<std::size_t>(row) *
                                                          static_cast<std::size_t>(in_features));
  const int* wb4 = reinterpret_cast<const int*>(w_b + static_cast<std::size_t>(row) *
                                                          static_cast<std::size_t>(in_features));
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
    for (int idx = split_warp * warpSize + lane; idx < tile_count;
         idx += WarpsPerRow * warpSize * 2) {
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
    const std::size_t offset =
        static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features) +
        static_cast<std::size_t>(col);
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

__global__ void weight_only_int4_matvec_kernel(const int8_t* w_packed, const float* scales,
                                               const half* x, half* y, int out_features,
                                               int in_features) {
  extern __shared__ float ssum[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= out_features) {
    return;
  }

  const int packed_cols = (in_features + 1) / 2;
  const int8_t* row_w =
      w_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);

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

// Wide warp-per-row decode kernels. The block-per-row kernels above read the packed row a
// BYTE at a time -- 196 GB/s effective on the 5090, which handed the whole bandwidth win of
// int4 back. These mirror gemv_wide_kernel's shape (Warps warps per block, one row per warp,
// 16-byte vector loads, kUnroll prefetch) on the packed data. A 16-byte chunk holds 32 int4
// or 16 int8 weights; group scales hoist per chunk when group % 32 == 0 (the default 128).
// The launchers route here when the shape allows and fall back to the naive kernels else.
template <int Warps>
__global__ void weight_only_int4_matvec_grouped_wide_kernel(const int8_t* __restrict__ w_packed,
                                                            const float* __restrict__ scales,
                                                            const half* __restrict__ x,
                                                            half* __restrict__ y, int out_features,
                                                            int in_features, int group_shift,
                                                            int n_groups) {
  const int row = blockIdx.x * Warps + (threadIdx.x / warpSize);
  if (row >= out_features) {
    return;
  }
  const int lane = threadIdx.x & (warpSize - 1);
  const int packed_cols = in_features / 2;
  const int4* wrow = reinterpret_cast<const int4*>(
      w_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols));
  const float* row_s = scales + static_cast<std::size_t>(row) * static_cast<std::size_t>(n_groups);
  const half2* x2 = reinterpret_cast<const half2*>(x);

  const int chunks = packed_cols / 16;  // 32 weights per 16-byte chunk
  constexpr int kUnroll = 4;
  float acc = 0.0f;
  int4 wbuf[kUnroll];
  for (int base = lane; base < chunks; base += warpSize * kUnroll) {
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int c = base + u * warpSize;
      if (c < chunks) {
        wbuf[u] = wrow[c];
      }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int c = base + u * warpSize;
      if (c < chunks) {
        const float s = row_s[(c * 32) >> group_shift];
        float chunk = 0.0f;
        const unsigned* wu = reinterpret_cast<const unsigned*>(&wbuf[u]);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const unsigned word = wu[i];  // 4 bytes = 8 consecutive weights
#pragma unroll
          for (int b = 0; b < 4; ++b) {
            const unsigned byte = (word >> (8 * b)) & 0xFFu;
            const int lo = static_cast<int>((byte & 0x0Fu) ^ 0x8u) - 0x8;
            const int hi = static_cast<int>((byte >> 4) ^ 0x8u) - 0x8;
            const float2 xv = __half22float2(x2[c * 16 + i * 4 + b]);
            chunk += static_cast<float>(lo) * xv.x + static_cast<float>(hi) * xv.y;
          }
        }
        acc += s * chunk;
      }
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    y[row] = __float2half(acc);
  }
}

// Grouped int4 x int8-activation dp4a matvec. The activation vector must be quantized with
// launch_quantize_fp16_to_int8_perm8 (even/odd bytes deinterleaved per 8-column window):
// `word & 0x0F0F0F0F` extracts a byte-word's four EVEN columns and `(word >> 4) & ...` the
// four ODD columns, each pairing with one aligned 32-bit load of the permuted x. Nibbles
// are re-signed with a single __vsubss4. Per 32-weight chunk the dot is EXACT int32, then
// scaled by the chunk's group scale; the activation scale applies once at the end.
template <int Warps>
__global__ void weight_only_int4_matvec_grouped_dp4a_wide_kernel(
    const int8_t* __restrict__ w_packed, const float* __restrict__ scales,
    const int8_t* __restrict__ xq, const float* __restrict__ x_scale, half* __restrict__ y,
    int out_features, int in_features, int group_shift, int n_groups) {
  const int row = blockIdx.x * Warps + (threadIdx.x / warpSize);
  if (row >= out_features) {
    return;
  }
  const int lane = threadIdx.x & (warpSize - 1);
  const int packed_cols = in_features / 2;
  const uint4* wrow = reinterpret_cast<const uint4*>(
      w_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols));
  const float* row_s = scales + static_cast<std::size_t>(row) * static_cast<std::size_t>(n_groups);
  const int* xw = reinterpret_cast<const int*>(xq);

  const int chunks = packed_cols / 16;  // 32 weights per 16-byte chunk
  constexpr int kUnroll = 4;
  float acc = 0.0f;
  uint4 wbuf[kUnroll];
  float sbuf[kUnroll];
  for (int base = lane; base < chunks; base += warpSize * kUnroll) {
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int c = base + u * warpSize;
      if (c < chunks) {
        wbuf[u] = wrow[c];
        sbuf[u] = row_s[(c * 32) >> group_shift];
      }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int c = base + u * warpSize;
      if (c < chunks) {
        const unsigned* wu = reinterpret_cast<const unsigned*>(&wbuf[u]);
        // x for the chunk as two 128-bit loads (8 words) instead of eight 32-bit loads:
        // the kernel was 78% long-scoreboard stalled, largely on x-read L1TEX slots.
        const uint4 xa = reinterpret_cast<const uint4*>(xq)[2 * c];
        const uint4 xb = reinterpret_cast<const uint4*>(xq)[2 * c + 1];
        const int xv[8] = {static_cast<int>(xa.x), static_cast<int>(xa.y),
                           static_cast<int>(xa.z), static_cast<int>(xa.w),
                           static_cast<int>(xb.x), static_cast<int>(xb.y),
                           static_cast<int>(xb.z), static_cast<int>(xb.w)};
        int gacc = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const unsigned word = wu[i];  // window 4c+i: 4 bytes = 8 weights
          // Nibbles are TWO'S COMPLEMENT (pack stores q & 0xF), so the decode is
          // (n ^ 8) - 8 (sign extension), NOT n - 8 -- xor first, then subtract.
          const int wlo =
              __vsubss4(static_cast<int>((word & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
          const int whi =
              __vsubss4(static_cast<int>(((word >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
          gacc = __dp4a(wlo, xv[2 * i], gacc);
          gacc = __dp4a(whi, xv[2 * i + 1], gacc);
        }
        acc += static_cast<float>(gacc) * sbuf[u];
      }
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    y[row] = __float2half(acc * x_scale[0]);
  }
}

// Segment-routed twin of the grouped dp4a kernel: up to three int4 projections sharing one
// perm8-quantized activation run as ONE launch (q|k|v, gate|up). Same per-row arithmetic;
// only the grid packing differs (see gemv_wide_cat_kernel for the pattern).
template <int Warps>
__global__ void weight_only_int4_matvec_grouped_dp4a_cat_kernel(
    const int8_t* __restrict__ w0, const float* __restrict__ s0, half* __restrict__ y0, int n0,
    const int8_t* __restrict__ w1, const float* __restrict__ s1, half* __restrict__ y1, int n1,
    const int8_t* __restrict__ w2, const float* __restrict__ s2, half* __restrict__ y2, int n2,
    const int8_t* __restrict__ xq, const float* __restrict__ x_scale, int in_features,
    int group_shift, int n_groups) {
  const int row = blockIdx.x * Warps + (threadIdx.x / warpSize);
  const int8_t* w = nullptr;
  const float* s = nullptr;
  half* y = nullptr;
  int r = row;
  if (row < n0) {
    w = w0;
    s = s0;
    y = y0;
  } else if (row < n0 + n1) {
    r = row - n0;
    w = w1;
    s = s1;
    y = y1;
  } else if (row < n0 + n1 + n2) {
    r = row - n0 - n1;
    w = w2;
    s = s2;
    y = y2;
  } else {
    return;
  }
  const int lane = threadIdx.x & (warpSize - 1);
  const int packed_cols = in_features / 2;
  const uint4* wrow = reinterpret_cast<const uint4*>(
      w + static_cast<std::size_t>(r) * static_cast<std::size_t>(packed_cols));
  const float* row_s = s + static_cast<std::size_t>(r) * static_cast<std::size_t>(n_groups);
  const int* xw = reinterpret_cast<const int*>(xq);

  const int chunks = packed_cols / 16;
  constexpr int kUnroll = 4;
  float acc = 0.0f;
  uint4 wbuf[kUnroll];
  float sbuf[kUnroll];
  for (int base = lane; base < chunks; base += warpSize * kUnroll) {
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int c = base + u * warpSize;
      if (c < chunks) {
        wbuf[u] = wrow[c];
        sbuf[u] = row_s[(c * 32) >> group_shift];
      }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int c = base + u * warpSize;
      if (c < chunks) {
        const unsigned* wu = reinterpret_cast<const unsigned*>(&wbuf[u]);
        // x for the chunk as two 128-bit loads (8 words) instead of eight 32-bit loads:
        // the kernel was 78% long-scoreboard stalled, largely on x-read L1TEX slots.
        const uint4 xa = reinterpret_cast<const uint4*>(xq)[2 * c];
        const uint4 xb = reinterpret_cast<const uint4*>(xq)[2 * c + 1];
        const int xv[8] = {static_cast<int>(xa.x), static_cast<int>(xa.y),
                           static_cast<int>(xa.z), static_cast<int>(xa.w),
                           static_cast<int>(xb.x), static_cast<int>(xb.y),
                           static_cast<int>(xb.z), static_cast<int>(xb.w)};
        int gacc = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const unsigned word = wu[i];
          const int wlo =
              __vsubss4(static_cast<int>((word & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
          const int whi =
              __vsubss4(static_cast<int>(((word >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
          gacc = __dp4a(wlo, xv[2 * i], gacc);
          gacc = __dp4a(whi, xv[2 * i + 1], gacc);
        }
        acc += static_cast<float>(gacc) * sbuf[u];
      }
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    y[r] = __float2half(acc * x_scale[0]);
  }
}

__device__ __forceinline__ float glu_gelu_tanh_f32(float x) {
  const float k = 0.7978845608028654f;  // sqrt(2/pi), matches gelu_tanh_f32
  return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

// Fused GeGLU dp4a matvec (the llama.cpp gated-activation fusion): one warp computes row r
// of BOTH the gate and up projections against the shared perm8 activation and writes
// out[r] = gelu(gate_r) * up_r directly -- no Gate/Up round-trip, no separate GeluMul
// launch. Numerics match the unfused path: both dots are rounded to fp16 first (that is
// what the standalone GeluMul read back), then gelu in fp32.
template <int Warps>
__global__ void weight_only_int4_matvec_grouped_dp4a_glu_kernel(
    const int8_t* __restrict__ wg, const float* __restrict__ sg, const int8_t* __restrict__ wu,
    const float* __restrict__ su, const int8_t* __restrict__ xq,
    const float* __restrict__ x_scale, half* __restrict__ y, int out_features, int in_features,
    int group_shift, int n_groups) {
  const int row = blockIdx.x * Warps + (threadIdx.x / warpSize);
  if (row >= out_features) {
    return;
  }
  const int lane = threadIdx.x & (warpSize - 1);
  const int packed_cols = in_features / 2;
  const uint4* wrow_g = reinterpret_cast<const uint4*>(
      wg + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols));
  const uint4* wrow_u = reinterpret_cast<const uint4*>(
      wu + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols));
  const float* row_sg = sg + static_cast<std::size_t>(row) * static_cast<std::size_t>(n_groups);
  const float* row_su = su + static_cast<std::size_t>(row) * static_cast<std::size_t>(n_groups);

  const int chunks = packed_cols / 16;
  constexpr int kUnroll = 2;
  float acc_g = 0.0f, acc_u = 0.0f;
  uint4 gbuf[kUnroll], ubuf[kUnroll];
  float sgbuf[kUnroll], subuf[kUnroll];
  for (int base = lane; base < chunks; base += warpSize * kUnroll) {
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int c = base + u * warpSize;
      if (c < chunks) {
        gbuf[u] = wrow_g[c];
        ubuf[u] = wrow_u[c];
        sgbuf[u] = row_sg[(c * 32) >> group_shift];
        subuf[u] = row_su[(c * 32) >> group_shift];
      }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int c = base + u * warpSize;
      if (c < chunks) {
        const uint4 xa = reinterpret_cast<const uint4*>(xq)[2 * c];
        const uint4 xb = reinterpret_cast<const uint4*>(xq)[2 * c + 1];
        const int xv[8] = {static_cast<int>(xa.x), static_cast<int>(xa.y),
                           static_cast<int>(xa.z), static_cast<int>(xa.w),
                           static_cast<int>(xb.x), static_cast<int>(xb.y),
                           static_cast<int>(xb.z), static_cast<int>(xb.w)};
        const unsigned* gu = reinterpret_cast<const unsigned*>(&gbuf[u]);
        const unsigned* uu = reinterpret_cast<const unsigned*>(&ubuf[u]);
        int gg = 0, gu_acc = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const unsigned wgw = gu[i];
          const int g_lo =
              __vsubss4(static_cast<int>((wgw & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
          const int g_hi =
              __vsubss4(static_cast<int>(((wgw >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
          gg = __dp4a(g_lo, xv[2 * i], gg);
          gg = __dp4a(g_hi, xv[2 * i + 1], gg);
          const unsigned wuw = uu[i];
          const int u_lo =
              __vsubss4(static_cast<int>((wuw & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
          const int u_hi =
              __vsubss4(static_cast<int>(((wuw >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), 0x08080808);
          gu_acc = __dp4a(u_lo, xv[2 * i], gu_acc);
          gu_acc = __dp4a(u_hi, xv[2 * i + 1], gu_acc);
        }
        acc_g += static_cast<float>(gg) * sgbuf[u];
        acc_u += static_cast<float>(gu_acc) * subuf[u];
      }
    }
  }
  acc_g = warp_sum(acc_g);
  acc_u = warp_sum(acc_u);
  if (lane == 0) {
    const float xs = x_scale[0];
    const float gate_h = __half2float(__float2half(acc_g * xs));
    const float up_h = __half2float(__float2half(acc_u * xs));
    y[row] = __float2half(glu_gelu_tanh_f32(gate_h) * up_h);
  }
}

template <int Warps>
__global__ void weight_only_int8_matvec_wide_kernel(const int8_t* __restrict__ w,
                                                    const float* __restrict__ scales,
                                                    const half* __restrict__ x,
                                                    half* __restrict__ y, int out_features,
                                                    int in_features) {
  const int row = blockIdx.x * Warps + (threadIdx.x / warpSize);
  if (row >= out_features) {
    return;
  }
  const int lane = threadIdx.x & (warpSize - 1);
  const int4* wrow = reinterpret_cast<const int4*>(
      w + static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features));
  const half2* x2 = reinterpret_cast<const half2*>(x);

  const int chunks = in_features / 16;  // 16 weights per 16-byte chunk
  constexpr int kUnroll = 4;
  float acc = 0.0f;
  int4 wbuf[kUnroll];
  for (int base = lane; base < chunks; base += warpSize * kUnroll) {
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int c = base + u * warpSize;
      if (c < chunks) {
        wbuf[u] = wrow[c];
      }
    }
#pragma unroll
    for (int u = 0; u < kUnroll; ++u) {
      const int c = base + u * warpSize;
      if (c < chunks) {
        const int8_t* wb = reinterpret_cast<const int8_t*>(&wbuf[u]);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const float2 xv = __half22float2(x2[c * 8 + i]);
          acc += static_cast<float>(wb[2 * i]) * xv.x + static_cast<float>(wb[2 * i + 1]) * xv.y;
        }
      }
    }
  }
  acc = warp_sum(acc);
  if (lane == 0) {
    y[row] = __float2half(acc * scales[row]);
  }
}

// Group-wise scales: the scale varies along the row, so it cannot be hoisted out
// of the dot product the way the per-row kernels do. Each weight is dequantised
// with its own group's scale before it is accumulated. `group_shift` is
// log2(group), so the group index is a shift rather than an integer divide.
__global__ void weight_only_int4_matvec_grouped_kernel(const int8_t* w_packed, const float* scales,
                                                       const half* x, half* y, int out_features,
                                                       int in_features, int group_shift,
                                                       int n_groups) {
  extern __shared__ float ssum[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= out_features) {
    return;
  }

  const int packed_cols = (in_features + 1) / 2;
  const int8_t* row_w =
      w_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
  const float* row_s = scales + static_cast<std::size_t>(row) * static_cast<std::size_t>(n_groups);

  float local = 0.0f;
  for (int col = tid; col < in_features; col += blockDim.x) {
    const float w = static_cast<float>(load_signed_int4(row_w, col)) * row_s[col >> group_shift];
    local += w * __half2float(x[col]);
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
    y[row] = __float2half(total);
  }
}

__global__ void weight_only_int8_matvec_grouped_kernel(const int8_t* w, const float* scales,
                                                       const half* x, half* y, int out_features,
                                                       int in_features, int group_shift,
                                                       int n_groups) {
  extern __shared__ float ssum[];
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  if (row >= out_features) {
    return;
  }

  const int8_t* row_w = w + static_cast<std::size_t>(row) * static_cast<std::size_t>(in_features);
  const float* row_s = scales + static_cast<std::size_t>(row) * static_cast<std::size_t>(n_groups);

  float local = 0.0f;
  for (int col = tid; col < in_features; col += blockDim.x) {
    const float wq = static_cast<float>(row_w[col]) * row_s[col >> group_shift];
    local += wq * __half2float(x[col]);
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
    y[row] = __float2half(total);
  }
}

__global__ void weight_only_int4_matvec_batched_kernel(const int8_t* w_packed, const float* scales,
                                                       const half* x, half* y, int batch_size,
                                                       int out_features, int in_features) {
  extern __shared__ float ssum[];
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= out_features || batch >= batch_size) {
    return;
  }

  const int packed_cols = (in_features + 1) / 2;
  const int8_t* row_w =
      w_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
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
                                                            const float* w_scales, const int8_t* x,
                                                            const float* x_scales, half* y,
                                                            int batch_size, int out_features,
                                                            int in_features) {
  __shared__ int warp_sums[32];
  const int row = blockIdx.x;
  const int batch = blockIdx.y;
  const int tid = threadIdx.x;
  if (row >= out_features || batch >= batch_size) {
    return;
  }

  const int packed_cols = (in_features + 1) / 2;
  const int8_t* row_w =
      w_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
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
                                                          const float* w_scales, const int8_t* x,
                                                          const float* x_scale, half* y,
                                                          int out_features, int in_features) {
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
  const int8_t* row_w =
      w_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
  const int packed4_total = in_features / 4;
  int local = 0;

  for (int tile_base = 0; tile_base < packed4_total; tile_base += TilePacked4) {
    const int tile_count = min(TilePacked4, packed4_total - tile_base);
    for (int idx = tid; idx < tile_count; idx += blockDim.x) {
      x_tile[idx] = reinterpret_cast<const int*>(x)[tile_base + idx];
    }
    __syncthreads();

    // Fetch the packed weights 128 bits at a time: one uint4 = 8 dp4a groups = 32 int4 weights.
    // Consecutive lanes read consecutive 16-byte chunks, so a warp pulls 512 contiguous bytes.
    // Narrower requests leave the warps stalled on memory latency even at full occupancy.
    //
    // Reordering the accumulation is safe here (unlike in the fp16 GEMVs): __dp4a accumulates
    // in int32, so the sum is exact and integer addition is associative -- the result is
    // bit-identical whatever the order.
    const int8_t* tile_w = row_w + static_cast<std::size_t>(tile_base) * 2;
    const bool aligned16 = (reinterpret_cast<std::uintptr_t>(tile_w) & 15u) == 0u;
    const int vec_groups = aligned16 ? (tile_count >> 3) : 0;  // groups of 8 dp4a units
    for (int v = split_warp * warpSize + lane; v < vec_groups; v += WarpsPerRow * warpSize) {
      const uint4 packed = reinterpret_cast<const uint4*>(tile_w)[v];
      const unsigned words[4] = {packed.x, packed.y, packed.z, packed.w};
      const int g0 = v << 3;
#pragma unroll
      for (int wi = 0; wi < 4; ++wi) {
        const unsigned wd = words[wi];
        local = __dp4a(unpack_int4x4(static_cast<std::uint8_t>(wd & 0xFFu),
                                     static_cast<std::uint8_t>((wd >> 8) & 0xFFu)),
                       x_tile[g0 + wi * 2], local);
        local = __dp4a(unpack_int4x4(static_cast<std::uint8_t>((wd >> 16) & 0xFFu),
                                     static_cast<std::uint8_t>((wd >> 24) & 0xFFu)),
                       x_tile[g0 + wi * 2 + 1], local);
      }
    }
    // Tail: whatever the 8-group stride could not cover (and the unaligned-row case).
    for (int idx = (vec_groups << 3) + split_warp * warpSize + lane; idx < tile_count;
         idx += WarpsPerRow * warpSize) {
      local = __dp4a(load_packed_int4x4(row_w, tile_base + idx), x_tile[idx], local);
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
__global__ void weight_only_int4_matvec_dual_dp4a_tiled_kernel(
    const int8_t* w_a_packed, const float* w_scales_a, const int8_t* w_b_packed,
    const float* w_scales_b, const int8_t* x, const float* x_scale, half* y_a, half* y_b,
    int out_features, int in_features) {
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
  const int8_t* row_wa =
      w_a_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
  const int8_t* row_wb =
      w_b_packed + static_cast<std::size_t>(row) * static_cast<std::size_t>(packed_cols);
  const int packed4_total = in_features / 4;
  int local_a = 0;
  int local_b = 0;

  for (int tile_base = 0; tile_base < packed4_total; tile_base += TilePacked4) {
    const int tile_count = min(TilePacked4, packed4_total - tile_base);
    for (int idx = tid; idx < tile_count; idx += blockDim.x) {
      x_tile[idx] = reinterpret_cast<const int*>(x)[tile_base + idx];
    }
    __syncthreads();

    for (int idx = split_warp * warpSize + lane; idx < tile_count;
         idx += WarpsPerRow * warpSize * 2) {
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
void launch_weight_only_int8_gemv_f32(const int8_t* w, const float* scales, const half* x,
                                      float* y, int out_features, int in_features,
                                      cudaStream_t stream) {
  constexpr int kWarps = 4;
  const int blocks = (out_features + kWarps - 1) / kWarps;
  weight_only_int8_gemv_f32_kernel<kWarps>
      <<<blocks, kWarps * 32, 0, stream>>>(w, scales, x, y, out_features, in_features);
}

void launch_weight_only_int8_matvec(const int8_t* w, const float* scales, const half* x, half* y,
                                    int out_features, int in_features, cudaStream_t stream) {
  if ((in_features % 16) == 0) {
    constexpr int kWarps = 4;
    const int blocks = (out_features + kWarps - 1) / kWarps;
    weight_only_int8_matvec_wide_kernel<kWarps>
        <<<blocks, kWarps * 32, 0, stream>>>(w, scales, x, y, out_features, in_features);
    return;
  }
  const int threads = choose_reduction_threads(in_features);
  weight_only_int8_matvec_kernel<<<out_features, threads,
                                   static_cast<std::size_t>(threads) * sizeof(float), stream>>>(
      w, scales, x, y, out_features, in_features);
}

namespace {

// log2 of a power-of-two group. Returns -1 for anything else, which the grouped
// launchers reject rather than silently mis-indexing the scales.
int group_shift_of(int group) {
  if (group <= 0 || (group & (group - 1)) != 0) {
    return -1;
  }
  int shift = 0;
  while ((1 << shift) < group) {
    ++shift;
  }
  return shift;
}

}  // namespace

void launch_weight_only_int4_matvec_grouped(const int8_t* w_packed, const float* scales,
                                            const half* x, half* y, int out_features,
                                            int in_features, int group, cudaStream_t stream) {
  const int shift = group_shift_of(group);
  if (shift < 0) {
    return;  // caller bug: group must be a positive power of two
  }
  const int n_groups = quant_group_count(in_features, group);
  if ((in_features % 32) == 0 && (group % 32) == 0) {
    constexpr int kWarps = 4;
    const int blocks = (out_features + kWarps - 1) / kWarps;
    weight_only_int4_matvec_grouped_wide_kernel<kWarps><<<blocks, kWarps * 32, 0, stream>>>(
        w_packed, scales, x, y, out_features, in_features, shift, n_groups);
    return;
  }
  const int threads = choose_reduction_threads(in_features);
  weight_only_int4_matvec_grouped_kernel<<<
      out_features, threads, static_cast<std::size_t>(threads) * sizeof(float), stream>>>(
      w_packed, scales, x, y, out_features, in_features, shift, n_groups);
}

void launch_weight_only_int4_matvec_grouped_dp4a(const int8_t* w_packed, const float* scales,
                                                 const std::int8_t* xq, const float* x_scale,
                                                 half* y, int out_features, int in_features,
                                                 int group, cudaStream_t stream) {
  const int shift = group_shift_of(group);
  if (shift < 0 || (group % 32) != 0 || (in_features % 32) != 0) {
    return;  // caller must gate: permuted-x dp4a needs 32-col chunks inside one group
  }
  const int n_groups = quant_group_count(in_features, group);
  constexpr int kWarps = 4;
  const int blocks = (out_features + kWarps - 1) / kWarps;
  weight_only_int4_matvec_grouped_dp4a_wide_kernel<kWarps><<<blocks, kWarps * 32, 0, stream>>>(
      w_packed, scales, xq, x_scale, y, out_features, in_features, shift, n_groups);
}

void launch_weight_only_int4_matvec_grouped_dp4a_cat(
    const int8_t* w0, const float* s0, half* y0, int n0, const int8_t* w1, const float* s1,
    half* y1, int n1, const int8_t* w2, const float* s2, half* y2, int n2, const std::int8_t* xq,
    const float* x_scale, int in_features, int group, cudaStream_t stream) {
  const int shift = group_shift_of(group);
  if (shift < 0 || (group % 32) != 0 || (in_features % 32) != 0) {
    return;  // caller gates
  }
  const int n_groups = quant_group_count(in_features, group);
  constexpr int kWarps = 4;
  const int total = n0 + n1 + n2;
  const int blocks = (total + kWarps - 1) / kWarps;
  weight_only_int4_matvec_grouped_dp4a_cat_kernel<kWarps><<<blocks, kWarps * 32, 0, stream>>>(
      w0, s0, y0, n0, w1, s1, y1, n1, w2, s2, y2, n2, xq, x_scale, in_features, shift, n_groups);
}

void launch_weight_only_int4_matvec_grouped_dp4a_glu(
    const std::int8_t* wg, const float* sg, const std::int8_t* wu, const float* su,
    const std::int8_t* xq, const float* x_scale, half* y, int out_features, int in_features,
    int group, cudaStream_t stream) {
  const int shift = group_shift_of(group);
  if (shift < 0 || (group % 32) != 0 || (in_features % 32) != 0) {
    return;  // caller gates
  }
  const int n_groups = quant_group_count(in_features, group);
  constexpr int kWarps = 4;
  const int blocks = (out_features + kWarps - 1) / kWarps;
  weight_only_int4_matvec_grouped_dp4a_glu_kernel<kWarps><<<blocks, kWarps * 32, 0, stream>>>(
      wg, sg, wu, su, xq, x_scale, y, out_features, in_features, shift, n_groups);
}

void launch_weight_only_int8_matvec_grouped(const int8_t* w, const float* scales, const half* x,
                                            half* y, int out_features, int in_features, int group,
                                            cudaStream_t stream) {
  const int shift = group_shift_of(group);
  if (shift < 0) {
    return;
  }
  const int n_groups = quant_group_count(in_features, group);
  const int threads = choose_reduction_threads(in_features);
  weight_only_int8_matvec_grouped_kernel<<<
      out_features, threads, static_cast<std::size_t>(threads) * sizeof(float), stream>>>(
      w, scales, x, y, out_features, in_features, shift, n_groups);
}

void launch_weight_only_int8_matvec_batched(const int8_t* w, const float* scales, const half* x,
                                            half* y, int batch_size, int out_features,
                                            int in_features, cudaStream_t stream) {
  const int threads = choose_reduction_threads(in_features);
  const dim3 grid(out_features, batch_size);
  weight_only_int8_matvec_batched_kernel<<<
      grid, threads, static_cast<std::size_t>(threads) * sizeof(float), stream>>>(
      w, scales, x, y, batch_size, out_features, in_features);
}

void launch_weight_only_int8_matvec_batched_dp4a(const int8_t* w, const float* w_scales,
                                                 const int8_t* x, const float* x_scales, half* y,
                                                 int batch_size, int out_features, int in_features,
                                                 cudaStream_t stream) {
  const int threads = choose_reduction_threads(in_features);
  const dim3 grid(out_features, batch_size);
  weight_only_int8_matvec_batched_dp4a_kernel<<<grid, threads, 0, stream>>>(
      w, w_scales, x, x_scales, y, batch_size, out_features, in_features);
}

void launch_weight_only_int8_matvec_dp4a(const int8_t* w, const float* w_scales, const int8_t* x,
                                         const float* x_scale, half* y, int out_features,
                                         int in_features, cudaStream_t stream, int warps_per_block,
                                         int tile_packed4, int warps_per_row) {
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
    weight_only_int8_matvec_dp4a_tiled_kernel<kWarps, kTile, kSplit>
        <<<blocks, kThreads, 0, stream>>>(w, w_scales, x, x_scale, y, out_features, in_features);
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

void launch_weight_only_int8_matvec_dual_dp4a(const int8_t* w_a, const float* w_scales_a,
                                              const int8_t* w_b, const float* w_scales_b,
                                              const int8_t* x, const float* x_scale, half* y_a,
                                              half* y_b, int out_features, int in_features,
                                              cudaStream_t stream, int warps_per_block,
                                              int tile_packed4, int warps_per_row) {
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
    weight_only_int8_matvec_dual_dp4a_tiled_kernel<kWarps, kTile, kSplit>
        <<<blocks, kThreads, 0, stream>>>(w_a, w_scales_a, w_b, w_scales_b, x, x_scale, y_a, y_b,
                                          out_features, in_features);
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

void launch_weight_only_int4_matvec(const int8_t* w_packed, const float* scales, const half* x,
                                    half* y, int out_features, int in_features,
                                    cudaStream_t stream) {
  const int threads = choose_reduction_threads(in_features);
  weight_only_int4_matvec_kernel<<<out_features, threads,
                                   static_cast<std::size_t>(threads) * sizeof(float), stream>>>(
      w_packed, scales, x, y, out_features, in_features);
}

void launch_weight_only_int4_matvec_batched(const int8_t* w_packed, const float* scales,
                                            const half* x, half* y, int batch_size,
                                            int out_features, int in_features,
                                            cudaStream_t stream) {
  const int threads = choose_reduction_threads(in_features);
  const dim3 grid(out_features, batch_size);
  weight_only_int4_matvec_batched_kernel<<<
      grid, threads, static_cast<std::size_t>(threads) * sizeof(float), stream>>>(
      w_packed, scales, x, y, batch_size, out_features, in_features);
}

void launch_weight_only_int4_matvec_batched_dp4a(const int8_t* w_packed, const float* w_scales,
                                                 const int8_t* x, const float* x_scales, half* y,
                                                 int batch_size, int out_features, int in_features,
                                                 cudaStream_t stream) {
  const int threads = choose_reduction_threads(in_features);
  const dim3 grid(out_features, batch_size);
  weight_only_int4_matvec_batched_dp4a_kernel<<<grid, threads, 0, stream>>>(
      w_packed, w_scales, x, x_scales, y, batch_size, out_features, in_features);
}

void launch_weight_only_int4_matvec_dp4a(const int8_t* w_packed, const float* w_scales,
                                         const int8_t* x, const float* x_scale, half* y,
                                         int out_features, int in_features, cudaStream_t stream,
                                         int warps_per_block, int tile_packed4, int warps_per_row) {
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
    weight_only_int4_matvec_dp4a_tiled_kernel<kWarps, kTile, kSplit>
        <<<blocks, kThreads, 0, stream>>>(w_packed, w_scales, x, x_scale, y, out_features,
                                          in_features);
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

void launch_weight_only_int4_matvec_dual_dp4a(const int8_t* w_a_packed, const float* w_scales_a,
                                              const int8_t* w_b_packed, const float* w_scales_b,
                                              const int8_t* x, const float* x_scale, half* y_a,
                                              half* y_b, int out_features, int in_features,
                                              cudaStream_t stream, int warps_per_block,
                                              int tile_packed4, int warps_per_row) {
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
    weight_only_int4_matvec_dual_dp4a_tiled_kernel<kWarps, kTile, kSplit>
        <<<blocks, kThreads, 0, stream>>>(w_a_packed, w_scales_a, w_b_packed, w_scales_b, x,
                                          x_scale, y_a, y_b, out_features, in_features);
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
