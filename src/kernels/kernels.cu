// kernels.cu
//
// CUDA kernel implementations and host launch wrappers for the inference
// runtime. `runtime/kernels.cuh` documents the public API; this file focuses on
// the performance-critical implementation details used by RMSNorm, embedding,
// RoPE, and prefill-attention kernels.

#include <cstdlib>
#include <cuda_fp16.h>

#include <cstdint>
#include <vector>

#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"

namespace kernels {
namespace {

// Small launch-configuration helpers keep the host wrappers concise and
// centralize the size heuristics used by related kernels.
inline int choose_rmsnorm_threads(int cols) {
  return (cols <= 2048) ? 128 : 256;
}

inline int choose_copy_threads(int cols) {
  return (cols <= 2048) ? 128 : 256;
}

// Warp-wide reductions are used throughout the file to avoid shared-memory
// tree reductions when a single warp can carry the partial result.
__device__ __forceinline__ float warp_sum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__device__ __forceinline__ float warp_max_f(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, offset));
  }
  return v;
}

// Basic normalization, embedding, and RoPE kernels.
__global__ void rmsnorm_kernel(const half* x, const half* w, half* y, int cols, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
  const half* x_row = x + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  half* y_row = y + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  __shared__ float warp_sums[32];
  __shared__ float inv_shared;

  float local = 0.0f;
  if ((cols & 1) == 0) {
    const int cols2 = cols / 2;
    const half2* x_row2 = reinterpret_cast<const half2*>(x_row);
    for (int col2 = tid; col2 < cols2; col2 += blockDim.x) {
      const float2 v = __half22float2(x_row2[col2]);
      local += v.x * v.x + v.y * v.y;
    }
  } else {
    for (int col = tid; col < cols; col += blockDim.x) {
      const float v = __half2float(x_row[col]);
      local += v * v;
    }
  }

  local = warp_sum(local);
  if (lane == 0) {
    warp_sums[warp] = local;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = (lane < warp_count) ? warp_sums[lane] : 0.0f;
    block_sum = warp_sum(block_sum);
    if (lane == 0) {
      inv_shared = rsqrtf(block_sum / static_cast<float>(cols) + eps);
    }
  }
  __syncthreads();

  const float inv = inv_shared;
  if ((cols & 1) == 0) {
    const int cols2 = cols / 2;
    const half2* x_row2 = reinterpret_cast<const half2*>(x_row);
    const half2* w2 = reinterpret_cast<const half2*>(w);
    half2* y_row2 = reinterpret_cast<half2*>(y_row);
    for (int col2 = tid; col2 < cols2; col2 += blockDim.x) {
      const float2 xv = __half22float2(x_row2[col2]);
      const float2 wv = __half22float2(w2[col2]);
      y_row2[col2] =
          __halves2half2(__float2half(xv.x * inv * wv.x), __float2half(xv.y * inv * wv.y));
    }
  } else {
    for (int col = tid; col < cols; col += blockDim.x) {
      const float xv = __half2float(x_row[col]);
      const float ww = __half2float(w[col]);
      y_row[col] = __float2half(xv * inv * ww);
    }
  }
}

__global__ void rmsnorm_kernel_simple(const half* x, const half* w, half* y, int cols, float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const half* x_row = x + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  half* y_row = y + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  __shared__ float sum_sq[256];
  __shared__ float inv_shared;

  float local = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float v = __half2float(x_row[col]);
    local += v * v;
  }
  sum_sq[tid] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum_sq[tid] += sum_sq[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    inv_shared = rsqrtf(sum_sq[0] / static_cast<float>(cols) + eps);
  }
  __syncthreads();

  const float inv = inv_shared;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float xv = __half2float(x_row[col]);
    const float ww = __half2float(w[col]);
    y_row[col] = __float2half(xv * inv * ww);
  }
}

__global__ void rmsnorm_offset_kernel_simple(const half* x, const half* w, half* y, int cols,
                                             float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const half* x_row = x + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  half* y_row = y + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  __shared__ float sum_sq[256];
  __shared__ float inv_shared;

  float local = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float v = __half2float(x_row[col]);
    local += v * v;
  }
  sum_sq[tid] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sum_sq[tid] += sum_sq[tid + stride];
    }
    __syncthreads();
  }

  if (tid == 0) {
    inv_shared = rsqrtf(sum_sq[0] / static_cast<float>(cols) + eps);
  }
  __syncthreads();

  const float inv = inv_shared;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float xv = __half2float(x_row[col]);
    const float ww = __half2float(w[col]);
    y_row[col] = __float2half(xv * inv * (1.0f + ww));
  }
}

__global__ void layernorm_kernel(const half* x, const half* w, const half* b, half* y, int cols,
                                 float eps) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
  const half* x_row = x + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  half* y_row = y + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
  __shared__ float warp_sum_x[32];
  __shared__ float warp_sum_x2[32];
  __shared__ float mean_shared;
  __shared__ float inv_shared;

  float local_sum = 0.0f;
  float local_sq = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float xv = __half2float(x_row[col]);
    local_sum += xv;
    local_sq += xv * xv;
  }

  local_sum = warp_sum(local_sum);
  local_sq = warp_sum(local_sq);
  if (lane == 0) {
    warp_sum_x[warp] = local_sum;
    warp_sum_x2[warp] = local_sq;
  }
  __syncthreads();

  if (warp == 0) {
    float block_sum = (lane < warp_count) ? warp_sum_x[lane] : 0.0f;
    float block_sq = (lane < warp_count) ? warp_sum_x2[lane] : 0.0f;
    block_sum = warp_sum(block_sum);
    block_sq = warp_sum(block_sq);
    if (lane == 0) {
      const float mean = block_sum / static_cast<float>(cols);
      const float var = fmaxf(0.0f, block_sq / static_cast<float>(cols) - mean * mean);
      mean_shared = mean;
      inv_shared = rsqrtf(var + eps);
    }
  }
  __syncthreads();

  const float mean = mean_shared;
  const float inv = inv_shared;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float xv = __half2float(x_row[col]);
    const float ww = __half2float(w[col]);
    const float bb = b ? __half2float(b[col]) : 0.0f;
    y_row[col] = __float2half((xv - mean) * inv * ww + bb);
  }
}

__global__ void embedding_lookup_kernel(const half* embedding, const int* token_ids, half* out,
                                        int hidden) {
  const int token_idx = blockIdx.x;
  const int token = token_ids[token_idx];
  const half* src = embedding + static_cast<std::size_t>(token) * static_cast<std::size_t>(hidden);
  half* dst = out + static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(hidden);

  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    dst[col] = src[col];
  }
}

__global__ void rope_inplace_kernel(half* q, half* k, int num_heads_q, int num_heads_k,
                                    int head_dim, int position, float rope_theta) {
  const int head = blockIdx.x;
  const int pair = threadIdx.x;
  const int half_dim = head_dim / 2;
  if (pair >= half_dim) {
    return;
  }

  const float theta =
      powf(rope_theta, -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
  const float angle = static_cast<float>(position) * theta;
  const float c = cosf(angle);
  const float s = sinf(angle);

  if (head < num_heads_q) {
    const int i0 = head * head_dim + pair;
    const int i1 = head * head_dim + pair + half_dim;
    const float q0 = __half2float(q[i0]);
    const float q1 = __half2float(q[i1]);
    q[i0] = __float2half(q0 * c - q1 * s);
    q[i1] = __float2half(q1 * c + q0 * s);
  }
  if (head < num_heads_k) {
    const int i0 = head * head_dim + pair;
    const int i1 = head * head_dim + pair + half_dim;
    const float k0 = __half2float(k[i0]);
    const float k1 = __half2float(k[i1]);
    k[i0] = __float2half(k0 * c - k1 * s);
    k[i1] = __float2half(k1 * c + k0 * s);
  }
}

__global__ void rope_inplace_table_kernel(half* q, half* k, int num_heads_q, int num_heads_k,
                                          int head_dim, int position, const float* cos_table,
                                          const float* sin_table) {
  const int head = blockIdx.x;
  const int pair = threadIdx.x;
  const int half_dim = head_dim / 2;
  if (pair >= half_dim) {
    return;
  }

  const int table_idx = position * half_dim + pair;
  const float c = cos_table[table_idx];
  const float s = sin_table[table_idx];

  if (head < num_heads_q) {
    const int i0 = head * head_dim + pair;
    const int i1 = head * head_dim + pair + half_dim;
    const float q0 = __half2float(q[i0]);
    const float q1 = __half2float(q[i1]);
    q[i0] = __float2half(q0 * c - q1 * s);
    q[i1] = __float2half(q1 * c + q0 * s);
  }
  if (head < num_heads_k) {
    const int i0 = head * head_dim + pair;
    const int i1 = head * head_dim + pair + half_dim;
    const float k0 = __half2float(k[i0]);
    const float k1 = __half2float(k[i1]);
    k[i0] = __float2half(k0 * c - k1 * s);
    k[i1] = __float2half(k1 * c + k0 * s);
  }
}

__global__ void rope_inplace_partial_table_kernel(half* q, half* k, int num_heads_q,
                                                  int num_heads_k, int head_dim, int rotary_dim,
                                                  int position, const float* cos_table,
                                                  const float* sin_table) {
  const int head = blockIdx.x;
  const int pair = threadIdx.x;
  const int half_dim = rotary_dim / 2;
  if (pair >= half_dim) {
    return;
  }

  const int table_idx = position * half_dim + pair;
  const float c = cos_table[table_idx];
  const float s = sin_table[table_idx];

  if (head < num_heads_q) {
    const int i0 = head * head_dim + pair;
    const int i1 = head * head_dim + pair + half_dim;
    const float q0 = __half2float(q[i0]);
    const float q1 = __half2float(q[i1]);
    q[i0] = __float2half(q0 * c - q1 * s);
    q[i1] = __float2half(q1 * c + q0 * s);
  }
  if (head < num_heads_k) {
    const int i0 = head * head_dim + pair;
    const int i1 = head * head_dim + pair + half_dim;
    const float k0 = __half2float(k[i0]);
    const float k1 = __half2float(k[i1]);
    k[i0] = __float2half(k0 * c - k1 * s);
    k[i1] = __float2half(k1 * c + k0 * s);
  }
}

// The device-position twin of rope_inplace_partial_table_kernel, byte-for-byte the same
// arithmetic: only the position source differs. Keep the two in lockstep -- the graph gate
// (graphs-on vs CPI_PLAN_NO_GRAPH streams must be token-identical) is what catches
// drift between them.
// ---------------------------------------------------------------------------
// Decode fusions (the kernel-count tax). At T=1 on this GPU every kernel costs ~11-14 us
// REGARDLESS of size -- a 3 KB residual add prices the same as a 1 MB GEMV -- so Gemma 4's
// ~870-op token spent more time entering kernels than running them. These fuse the two
// highest-count patterns; the executor's peepholes decide when they apply.

// x = (x + rmsnorm_w(tmp)) * alpha, one row. Fuses the sandwich-norm tail
// [RmsNorm(Tmp) -> AddInplace(X += Tmp) -> optional ScaleCopy(X *= layer_scalar)]
// into one kernel. Rounding mirrors the eager sequence: the normalised value is rounded to
// half before the add, the sum is rounded before the scale, so drift against the unfused
// path is one reduction-order difference in the sum of squares, not a formula change.
__global__ void rmsnorm_add_scale_kernel(half* x, const half* tmp, const half* w, int cols,
                                         float eps, float alpha) {
  const int tid = threadIdx.x;
  __shared__ float sum_sq[256];
  __shared__ float inv_shared;

  float local = 0.0f;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float v = __half2float(tmp[col]);
    local += v * v;
  }
  sum_sq[tid] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) sum_sq[tid] += sum_sq[tid + stride];
    __syncthreads();
  }
  if (tid == 0) inv_shared = rsqrtf(sum_sq[0] / static_cast<float>(cols) + eps);
  __syncthreads();

  const float inv = inv_shared;
  for (int col = tid; col < cols; col += blockDim.x) {
    const float nv = __half2float(tmp[col]) * inv * __half2float(w[col]);
    const half nh = __float2half(nv);  // the fp16 the unfused RmsNorm would have written
    float acc = __half2float(x[col]) + __half2float(nh);
    if (alpha != 1.0f) {
      acc = __half2float(__float2half(acc)) * alpha;  // ScaleCopy read a rounded sum
    }
    x[col] = __float2half(acc);
  }
}

// Per-head rmsnorm followed by table RoPE, in place: fuses the [RmsNorm(rows=heads) -> Rope]
// pair on Q and K. The table encodes any partial rotary (identity rotations past the rotary
// span), exactly as the unfused rope kernel reads it. `pos_ptr` wins over `position` when
// non-null, so the same kernel serves eager decode and graph capture.
__global__ void rmsnorm_rope_kernel(half* s, const half* w, int head_dim, int position,
                                    const int* pos_ptr, const float* cos_table,
                                    const float* sin_table, float eps) {
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int half_dim = head_dim / 2;
  half* row = s + static_cast<std::size_t>(head) * head_dim;
  __shared__ float sum_sq[256];
  __shared__ float inv_shared;

  float local = 0.0f;
  for (int col = tid; col < head_dim; col += blockDim.x) {
    const float v = __half2float(row[col]);
    local += v * v;
  }
  sum_sq[tid] = local;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) sum_sq[tid] += sum_sq[tid + stride];
    __syncthreads();
  }
  if (tid == 0) inv_shared = rsqrtf(sum_sq[0] / static_cast<float>(head_dim) + eps);
  __syncthreads();

  const float inv = inv_shared;
  const int pos = (pos_ptr != nullptr) ? pos_ptr[0] : position;
  for (int pair = tid; pair < half_dim; pair += blockDim.x) {
    const int i0 = pair;
    const int i1 = pair + half_dim;
    // Normalise both lanes of the pair, rounded to half exactly as the unfused norm wrote them.
    const float n0 =
        __half2float(__float2half(__half2float(row[i0]) * inv * __half2float(w[i0])));
    const float n1 =
        __half2float(__float2half(__half2float(row[i1]) * inv * __half2float(w[i1])));
    const int table_idx = pos * half_dim + pair;
    const float c = cos_table[table_idx];
    const float sn = sin_table[table_idx];
    row[i0] = __float2half(n0 * c - n1 * sn);
    row[i1] = __float2half(n1 * c + n0 * sn);
  }
}

__global__ void rope_inplace_partial_table_device_pos_kernel(half* q, half* k, int num_heads_q,
                                                             int num_heads_k, int head_dim,
                                                             int rotary_dim,
                                                             const int* position_ptr,
                                                             const float* cos_table,
                                                             const float* sin_table) {
  const int position = position_ptr[0];
  const int head = blockIdx.x;
  const int pair = threadIdx.x;
  const int half_dim = rotary_dim / 2;
  if (pair >= half_dim) {
    return;
  }

  const int table_idx = position * half_dim + pair;
  const float c = cos_table[table_idx];
  const float s = sin_table[table_idx];

  if (head < num_heads_q) {
    const int i0 = head * head_dim + pair;
    const int i1 = head * head_dim + pair + half_dim;
    const float q0 = __half2float(q[i0]);
    const float q1 = __half2float(q[i1]);
    q[i0] = __float2half(q0 * c - q1 * s);
    q[i1] = __float2half(q1 * c + q0 * s);
  }
  if (head < num_heads_k) {
    const int i0 = head * head_dim + pair;
    const int i1 = head * head_dim + pair + half_dim;
    const float k0 = __half2float(k[i0]);
    const float k1 = __half2float(k[i1]);
    k[i0] = __float2half(k0 * c - k1 * s);
    k[i1] = __float2half(k1 * c + k0 * s);
  }
}

__global__ void rope_inplace_device_pos_kernel(half* q, half* k, int num_heads_q, int num_heads_k,
                                               int head_dim, const int* position_ptr,
                                               const float* cos_table, const float* sin_table) {
  const int position = position_ptr[0];
  const int head = blockIdx.x;
  const int pair = threadIdx.x;
  const int half_dim = head_dim / 2;
  if (pair >= half_dim) {
    return;
  }

  const int table_idx = position * half_dim + pair;
  const float c = cos_table[table_idx];
  const float s = sin_table[table_idx];

  if (head < num_heads_q) {
    const int i0 = head * head_dim + pair;
    const int i1 = head * head_dim + pair + half_dim;
    const float q0 = __half2float(q[i0]);
    const float q1 = __half2float(q[i1]);
    q[i0] = __float2half(q0 * c - q1 * s);
    q[i1] = __float2half(q1 * c + q0 * s);
  }
  if (head < num_heads_k) {
    const int i0 = head * head_dim + pair;
    const int i1 = head * head_dim + pair + half_dim;
    const float k0 = __half2float(k[i0]);
    const float k1 = __half2float(k[i1]);
    k[i0] = __float2half(k0 * c - k1 * s);
    k[i1] = __float2half(k1 * c + k0 * s);
  }
}

// Per-position RoPE (P2): like rope_inplace_batched_kernel but each of the
// num_tokens rows is rotated at its OWN position positions[token] (one token per
// sequence in a batched decode step), rather than a contiguous start_position+token.
__global__ void rope_inplace_perpos_kernel(half* q, half* k, int num_tokens, int num_heads_q,
                                           int num_heads_k, int head_dim,
                                           const int* __restrict__ positions,
                                           const float* cos_table, const float* sin_table) {
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int pair = threadIdx.x;
  const int half_dim = head_dim / 2;
  if (token >= num_tokens || pair >= half_dim) return;
  const int table_idx = positions[token] * half_dim + pair;
  const float c = cos_table[table_idx];
  const float s = sin_table[table_idx];
  if (head < num_heads_q) {
    const int base = token * num_heads_q * head_dim + head * head_dim;
    const float q0 = __half2float(q[base + pair]);
    const float q1 = __half2float(q[base + pair + half_dim]);
    q[base + pair] = __float2half(q0 * c - q1 * s);
    q[base + pair + half_dim] = __float2half(q1 * c + q0 * s);
  }
  if (head < num_heads_k) {
    const int base = token * num_heads_k * head_dim + head * head_dim;
    const float k0 = __half2float(k[base + pair]);
    const float k1 = __half2float(k[base + pair + half_dim]);
    k[base + pair] = __float2half(k0 * c - k1 * s);
    k[base + pair + half_dim] = __float2half(k1 * c + k0 * s);
  }
}

// q_row_stride / k_row_stride let this rotate Q and K IN PLACE inside the fused QKV buffer,
// rather than requiring them to be copied out into contiguous [tokens, heads*head_dim] buffers
// first. Prefill is host-bound and those copies were 3 of the 7 cudaMemcpy2DAsync per layer.
// Passing the natural strides (num_heads * head_dim) reproduces the old behaviour EXACTLY.
__global__ void rope_inplace_batched_kernel(half* q, half* k, int num_tokens, int num_heads_q,
                                            int num_heads_k, int head_dim, int start_position,
                                            const float* cos_table, const float* sin_table,
                                            int q_row_stride, int k_row_stride) {
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int pair = threadIdx.x;
  const int half_dim = head_dim / 2;
  if (token >= num_tokens || pair >= half_dim) {
    return;
  }

  const int table_idx = (start_position + token) * half_dim + pair;
  const float c = cos_table[table_idx];
  const float s = sin_table[table_idx];

  if (head < num_heads_q) {
    const int base = token * q_row_stride + head * head_dim;
    const float q0 = __half2float(q[base + pair]);
    const float q1 = __half2float(q[base + pair + half_dim]);
    q[base + pair] = __float2half(q0 * c - q1 * s);
    q[base + pair + half_dim] = __float2half(q1 * c + q0 * s);
  }
  if (head < num_heads_k) {
    const int base = token * k_row_stride + head * head_dim;
    const float k0 = __half2float(k[base + pair]);
    const float k1 = __half2float(k[base + pair + half_dim]);
    k[base + pair] = __float2half(k0 * c - k1 * s);
    k[base + pair + half_dim] = __float2half(k1 * c + k0 * s);
  }
}

// Decode-time attention helpers and kernels.
// Flatten [time, head, dim] coordinates into the packed KV-cache layout.
__device__ __forceinline__ int cache_index(int t, int head, int d, int num_heads, int head_dim) {
  return (t * num_heads + head) * head_dim + d;
}

// Prefill path: each block handles one [head, token] pair and attends over
// the cached prefix plus its own in-chunk prefix.
__global__ void attention_prefill_kernel_fallback(const half* q, const half* k_cache,
                                                  const half* v_cache, half* out, int num_tokens,
                                                  int start_position, int num_heads,
                                                  int num_kv_heads, int head_dim, int causal,
                                                  const int* limits, int window) {
  extern __shared__ unsigned char smem_bytes[];
  half* q_shared = reinterpret_cast<half*>(smem_bytes);
  float* red = reinterpret_cast<float*>(q_shared + head_dim);
  float* alpha_shared = red + blockDim.x;
  float* beta_shared = alpha_shared + 1;
  float* inv_l_shared = beta_shared + 1;
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int tid = threadIdx.x;
  if (token >= num_tokens) {
    return;
  }
  const int hidden = num_heads * head_dim;
  const int q_base = token * hidden + head * head_dim;
  const int out_base = q_base;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head =
      ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  // Each thread owns head_dim/blockDim OUTPUT channels. A SCALAR accumulator silently
  // drops every channel past blockDim -- with Gemma's head_dim=512 full layers on 128
  // threads that is 3/4 of the head, computed and written as if it did not exist. The
  // decode kernel was fixed for exactly this; the prefill fallback never was, because
  // Llama's head_dim (128) always fit in one thread's slot.

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[q_base + d];
  }
  __syncthreads();

  float running_m = -1.0e30f;
  float running_l = 0.0f;
  constexpr int kOutPer = kAccPerThread;  // covers head_dim up to blockDim * kAccPerThread
  float acc[kOutPer];
#pragma unroll
  for (int j = 0; j < kOutPer; ++j) acc[j] = 0.0f;
  // How far this token may see:
  //   limits != null : per-token key limit -- this is what makes an IMAGE SPAN
  //                    bidirectional (every token in the span sees the whole span)
  //                    while the surrounding text stays causal.
  //   causal         : its own prefix.
  //   otherwise      : the whole sequence (a vision encoder).
  const int limit = limits ? limits[token]
                           : (causal ? (start_position + token + 1)
                                     : (start_position + num_tokens));
  // Sliding-window layers only see the last `window` keys. The decode path has always
  // done this; the prefill kernel never did (Llama has no windows), so a Gemma prompt
  // longer than the window would have been silently wrong.
  const int k_start = (window > 0 && limit > window) ? (limit - window) : 0;
  for (int t = k_start; t < limit; ++t) {
    float partial_dot = 0.0f;
    const int base = cache_index(t, kv_head, 0, num_kv_heads, head_dim);
    for (int d = tid; d < head_dim; d += blockDim.x) {
      partial_dot += __half2float(q_shared[d]) * __half2float(k_cache[base + d]);
    }
    {
      const int lane_id = tid & (warpSize - 1);
      const int warp_id = tid / warpSize;
      float dot = warp_sum(partial_dot);
      if (lane_id == 0) {
        red[warp_id] = dot;
      }
    }
    __syncthreads();

    if (tid == 0) {
      float total = 0.0f;
      for (int w = 0; w < blockDim.x / warpSize; ++w) {
        total += red[w];
      }
      const float score = total * scale;
      const float new_m = fmaxf(running_m, score);
      const float alpha = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float beta = expf(score - new_m);
      running_l = running_l * alpha + beta;
      running_m = new_m;
      alpha_shared[0] = alpha;
      beta_shared[0] = beta;
      inv_l_shared[0] = 1.0f / fmaxf(running_l, 1e-8f);
    }
    __syncthreads();

#pragma unroll
    for (int j = 0; j < kOutPer; ++j) {
      const int d = tid + j * static_cast<int>(blockDim.x);
      if (d < head_dim) {
        acc[j] = acc[j] * alpha_shared[0] + beta_shared[0] * __half2float(v_cache[base + d]);
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int j = 0; j < kOutPer; ++j) {
    const int d = tid + j * static_cast<int>(blockDim.x);
    if (d < head_dim) {
      out[out_base + d] = __float2half(acc[j] * inv_l_shared[0]);
    }
  }
}

// Tiled prefill keeps the causal limit per token while still vectorizing the
// K dot Q work inside each tile.
template <int WarpsPerBlock>
__global__ void attention_prefill_kernel_tiled(const half* q, const half* k_cache,
                                               const half* v_cache, half* out, int num_tokens,
                                               int start_position, int num_heads, int num_kv_heads,
                                               int head_dim, int causal, const int* limits,
                                               int window) {
  extern __shared__ unsigned char smem_bytes[];
  half* q_shared = reinterpret_cast<half*>(smem_bytes);
  float* score_shared = reinterpret_cast<float*>(q_shared + head_dim);
  float* alpha_shared = score_shared + WarpsPerBlock;
  float* beta_shared = alpha_shared + WarpsPerBlock;
  float* stats_shared = beta_shared + WarpsPerBlock;

  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane = tid % warpSize;
  if (token >= num_tokens) {
    return;
  }

  const int hidden = num_heads * head_dim;
  const int q_base = token * hidden + head * head_dim;
  const int out_base = q_base;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head =
      ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const int head_pairs = head_dim / 2;
  // How far this token may see:
  //   limits != null : per-token key limit -- this is what makes an IMAGE SPAN
  //                    bidirectional (every token in the span sees the whole span)
  //                    while the surrounding text stays causal.
  //   causal         : its own prefix.
  //   otherwise      : the whole sequence (a vision encoder).
  const int limit = limits ? limits[token]
                           : (causal ? (start_position + token + 1)
                                     : (start_position + num_tokens));
  // Sliding-window layers only see the last `window` keys. The decode path has always
  // done this; the prefill kernel never did (Llama has no windows), so a Gemma prompt
  // longer than the window would have been silently wrong.
  const int k_start = (window > 0 && limit > window) ? (limit - window) : 0;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[q_base + d];
  }
  if (tid == 0) {
    stats_shared[0] = -1.0e30f;
    stats_shared[1] = 0.0f;
  }
  __syncthreads();

  // Each thread owns head_dim/blockDim output channels (2 at head_dim=256 with
  // 128 threads), so the accumulator MUST be an array — a scalar conflates the
  // strided output dims into one, corrupting head_dim > blockDim (256).
  constexpr int kOutPerThread = (256 + WarpsPerBlock * 32 - 1) / (WarpsPerBlock * 32);
  float acc[kOutPerThread];
#pragma unroll
  for (int j = 0; j < kOutPerThread; ++j) acc[j] = 0.0f;
  for (int tile_base = k_start; tile_base < limit; tile_base += WarpsPerBlock) {
    const int t = tile_base + warp_id;
    float score = -1.0e30f;
    if (warp_id < WarpsPerBlock && t < limit) {
      const int base = cache_index(t, kv_head, 0, num_kv_heads, head_dim);
      const half2* q2 = reinterpret_cast<const half2*>(q_shared);
      const half2* k2 = reinterpret_cast<const half2*>(k_cache + base);
      float partial = 0.0f;
      for (int pair = lane; pair < head_pairs; pair += warpSize) {
        const float2 qv = __half22float2(q2[pair]);
        const float2 kv = __half22float2(k2[pair]);
        partial += qv.x * kv.x + qv.y * kv.y;
      }
      score = warp_sum(partial) * scale;
    }
    if (lane == 0 && warp_id < WarpsPerBlock) {
      score_shared[warp_id] = score;
    }
    __syncthreads();

    if (tid == 0) {
      float running_m = stats_shared[0];
      float running_l = stats_shared[1];
      const int tile_tokens = min(WarpsPerBlock, limit - tile_base);
      for (int i = 0; i < tile_tokens; ++i) {
        const float token_score = score_shared[i];
        const float new_m = fmaxf(running_m, token_score);
        const float alpha = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
        const float beta = expf(token_score - new_m);
        running_l = running_l * alpha + beta;
        running_m = new_m;
        alpha_shared[i] = alpha;
        beta_shared[i] = beta;
      }
      stats_shared[0] = running_m;
      stats_shared[1] = running_l;
    }
    __syncthreads();

    int j = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
      float acc_local = acc[j];
      const int tile_tokens = min(WarpsPerBlock, limit - tile_base);
      for (int i = 0; i < tile_tokens; ++i) {
        const int base = cache_index(tile_base + i, kv_head, 0, num_kv_heads, head_dim);
        acc_local = acc_local * alpha_shared[i] + beta_shared[i] * __half2float(v_cache[base + d]);
      }
      acc[j] = acc_local;
    }
    __syncthreads();
  }

  const float inv_l = 1.0f / fmaxf(stats_shared[1], 1e-8f);
  int j = 0;
  for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
    out[out_base + d] = __float2half(acc[j] * inv_l);
  }
}

// Paged prefill attention (P3 phase 2d): identical to attention_prefill_kernel_tiled
// but each key position t is read through the block table (block_size tokens per
// block) — phys(t) = block_table[t/block_size]*block_size + t%block_size — into a
// block pool, so KV need not be contiguous. Same causal online-softmax math.
template <int WarpsPerBlock>
__global__ void attention_prefill_kernel_tiled_paged(const half* q, const half* k_pool,
                                                     const half* v_pool,
                                                     const int* __restrict__ block_table, half* out,
                                                     int num_tokens, int start_position,
                                                     int num_heads, int num_kv_heads, int head_dim,
                                                     int block_size) {
  extern __shared__ unsigned char smem_bytes[];
  half* q_shared = reinterpret_cast<half*>(smem_bytes);
  float* score_shared = reinterpret_cast<float*>(q_shared + head_dim);
  float* alpha_shared = score_shared + WarpsPerBlock;
  float* beta_shared = alpha_shared + WarpsPerBlock;
  float* stats_shared = beta_shared + WarpsPerBlock;

  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane = tid % warpSize;
  if (token >= num_tokens) {
    return;
  }

  const int hidden = num_heads * head_dim;
  const int q_base = token * hidden + head * head_dim;
  const int out_base = q_base;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head =
      ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const int head_pairs = head_dim / 2;
  const int limit = start_position + token + 1;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[q_base + d];
  }
  if (tid == 0) {
    stats_shared[0] = -1.0e30f;
    stats_shared[1] = 0.0f;
  }
  __syncthreads();

  // Per-thread output array (2 channels at head_dim=256 with 128 threads); a
  // scalar would conflate the strided output dims — see the non-paged variant.
  constexpr int kOutPerThread = (256 + WarpsPerBlock * 32 - 1) / (WarpsPerBlock * 32);
  float acc[kOutPerThread];
#pragma unroll
  for (int j = 0; j < kOutPerThread; ++j) acc[j] = 0.0f;
  for (int tile_base = 0; tile_base < limit; tile_base += WarpsPerBlock) {
    const int t = tile_base + warp_id;
    float score = -1.0e30f;
    if (warp_id < WarpsPerBlock && t < limit) {
      const int phys = block_table[t / block_size] * block_size + (t % block_size);
      const int base = cache_index(phys, kv_head, 0, num_kv_heads, head_dim);
      const half2* q2 = reinterpret_cast<const half2*>(q_shared);
      const half2* k2 = reinterpret_cast<const half2*>(k_pool + base);
      float partial = 0.0f;
      for (int pair = lane; pair < head_pairs; pair += warpSize) {
        const float2 qv = __half22float2(q2[pair]);
        const float2 kv = __half22float2(k2[pair]);
        partial += qv.x * kv.x + qv.y * kv.y;
      }
      score = warp_sum(partial) * scale;
    }
    if (lane == 0 && warp_id < WarpsPerBlock) {
      score_shared[warp_id] = score;
    }
    __syncthreads();

    if (tid == 0) {
      float running_m = stats_shared[0];
      float running_l = stats_shared[1];
      const int tile_tokens = min(WarpsPerBlock, limit - tile_base);
      for (int i = 0; i < tile_tokens; ++i) {
        const float token_score = score_shared[i];
        const float new_m = fmaxf(running_m, token_score);
        const float alpha = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
        const float beta = expf(token_score - new_m);
        running_l = running_l * alpha + beta;
        running_m = new_m;
        alpha_shared[i] = alpha;
        beta_shared[i] = beta;
      }
      stats_shared[0] = running_m;
      stats_shared[1] = running_l;
    }
    __syncthreads();

    int j = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
      float acc_local = acc[j];
      const int tile_tokens = min(WarpsPerBlock, limit - tile_base);
      for (int i = 0; i < tile_tokens; ++i) {
        const int phys =
            block_table[(tile_base + i) / block_size] * block_size + ((tile_base + i) % block_size);
        const int base = cache_index(phys, kv_head, 0, num_kv_heads, head_dim);
        acc_local = acc_local * alpha_shared[i] + beta_shared[i] * __half2float(v_pool[base + d]);
      }
      acc[j] = acc_local;
    }
    __syncthreads();
  }

  const float inv_l = 1.0f / fmaxf(stats_shared[1], 1e-8f);
  int j = 0;
  for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
    out[out_base + d] = __float2half(acc[j] * inv_l);
  }
}

// Scatter `rows` freshly-projected KV rows (contiguous src) into the block pool
// at paged positions base_pos..base_pos+rows-1 (P3 phase 2d prefill KV write).
__global__ void store_kv_paged_kernel(half* k_pool, half* v_pool, const half* k_src,
                                      const half* v_src, const int* __restrict__ block_table,
                                      int base_pos, int rows, int kv_hidden, int block_size) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int pos = base_pos + row;
  const int phys = block_table[pos / block_size] * block_size + (pos % block_size);
  half* kd = k_pool + static_cast<std::size_t>(phys) * kv_hidden;
  half* vd = v_pool + static_cast<std::size_t>(phys) * kv_hidden;
  const half* ks = k_src + static_cast<std::size_t>(row) * kv_hidden;
  const half* vs = v_src + static_cast<std::size_t>(row) * kv_hidden;
  for (int d = threadIdx.x; d < kv_hidden; d += blockDim.x) {
    kd[d] = ks[d];
    vd[d] = vs[d];
  }
}

// Batched decode KV scatter (P2): one token per sequence, each to its own block
// table at its own position. row = sequence; positions[row] + block_tables[row].
__global__ void store_kv_batched_paged_kernel(half* k_pool, half* v_pool, const half* k_src,
                                              const half* v_src,
                                              const int* __restrict__ block_tables,
                                              const int* __restrict__ positions, int max_blocks,
                                              int batch, int kv_hidden, int block_size) {
  const int b = blockIdx.x;
  if (b >= batch) return;
  const int pos = positions[b];
  const int* bt = block_tables + static_cast<std::size_t>(b) * max_blocks;
  const int phys = bt[pos / block_size] * block_size + (pos % block_size);
  half* kd = k_pool + static_cast<std::size_t>(phys) * kv_hidden;
  half* vd = v_pool + static_cast<std::size_t>(phys) * kv_hidden;
  const half* ks = k_src + static_cast<std::size_t>(b) * kv_hidden;
  const half* vs = v_src + static_cast<std::size_t>(b) * kv_hidden;
  for (int d = threadIdx.x; d < kv_hidden; d += blockDim.x) {
    kd[d] = ks[d];
    vd[d] = vs[d];
  }
}

// Decode-shaped RMSNorm: loads x once with 128-bit loads, keeps it in registers across both
// passes (rmsnorm_kernel_simple re-reads it from global for the normalise step), and reduces
// with warp shuffles instead of a shared-memory tree.
//
// Not bit-identical to rmsnorm_kernel_simple: the sum of squares accumulates in a different
// order, so it differs in the last fp32 bits.
template <int Threads>
__global__ void rmsnorm_fast_kernel(const half* __restrict__ x, const half* __restrict__ w,
                                    half* __restrict__ y, int cols, float eps) {
  constexpr int kMaxVecs = 8;  // int4 chunks per thread => cols <= Threads * 8 * 8
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  const int vecs = cols / 8;
  const int4* x4 =
      reinterpret_cast<const int4*>(x + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols));

  int4 buf[kMaxVecs];
  float acc = 0.0f;
  int n = 0;
  for (int i = tid; i < vecs; i += Threads, ++n) {
    buf[n] = x4[i];
    const half2* h = reinterpret_cast<const half2*>(&buf[n]);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 v = __half22float2(h[j]);
      acc += v.x * v.x + v.y * v.y;
    }
  }

#pragma unroll
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  __shared__ float partial[Threads / 32];
  __shared__ float inv_shared;
  const int warp = tid / warpSize;
  const int lane = tid & (warpSize - 1);
  if (lane == 0) {
    partial[warp] = acc;
  }
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < Threads / 32; ++i) {
      total += partial[i];
    }
    inv_shared = rsqrtf(total / static_cast<float>(cols) + eps);
  }
  __syncthreads();

  const float inv = inv_shared;
  const int4* w4 = reinterpret_cast<const int4*>(w);
  int4* y4 =
      reinterpret_cast<int4*>(y + static_cast<std::size_t>(row) * static_cast<std::size_t>(cols));
  n = 0;
  for (int i = tid; i < vecs; i += Threads, ++n) {
    const int4 wpack = w4[i];
    const half2* xh = reinterpret_cast<const half2*>(&buf[n]);  // still in registers
    const half2* wh = reinterpret_cast<const half2*>(&wpack);
    int4 out;
    half2* oh = reinterpret_cast<half2*>(&out);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 xv = __half22float2(xh[j]);
      const float2 wv = __half22float2(wh[j]);
      oh[j] = __floats2half2_rn(xv.x * inv * wv.x, xv.y * inv * wv.y);
    }
    y4[i] = out;
  }
}

// rmsnorm_fast + perm8 int8 activation quantization in ONE kernel, for the rows=1 norms
// whose output feeds dp4a int4 projections (XNorm sites). The normed row never leaves
// registers: fp16 y is written as usual, then the fp16-ROUNDED values are quantized --
// exactly what the separate quantize kernel would read back -- so the fused path is
// bit-identical to [rmsnorm_fast; quantize_fp16_to_int8_perm8]. cols <= Threads * 8.
template <int Threads>
__global__ void rmsnorm_quant_perm8_kernel(const half* __restrict__ x, const half* __restrict__ w,
                                           half* __restrict__ y, int8_t* __restrict__ xq,
                                           float* __restrict__ xscale, int cols, float eps,
                                           int max_q) {
  const int tid = threadIdx.x;
  const int vecs = cols / 8;
  const int4* x4 = reinterpret_cast<const int4*>(x);

  int4 xbuf;
  int4 obuf;
  float acc = 0.0f;
  const bool active = tid < vecs;
  if (active) {
    xbuf = x4[tid];
    const half2* h = reinterpret_cast<const half2*>(&xbuf);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 v = __half22float2(h[j]);
      acc += v.x * v.x + v.y * v.y;
    }
  }
#pragma unroll
  for (int off = warpSize / 2; off > 0; off >>= 1) {
    acc += __shfl_down_sync(0xffffffffu, acc, off);
  }
  __shared__ float partial[Threads / 32];
  __shared__ float shared_vals[2];  // [inv_rms, inv_scale]
  const int warp = tid / warpSize;
  const int lane = tid & (warpSize - 1);
  if (lane == 0) {
    partial[warp] = acc;
  }
  __syncthreads();
  if (tid == 0) {
    float total = 0.0f;
#pragma unroll
    for (int i = 0; i < Threads / 32; ++i) {
      total += partial[i];
    }
    shared_vals[0] = rsqrtf(total / static_cast<float>(cols) + eps);
  }
  __syncthreads();

  const float inv = shared_vals[0];
  float local_max = 0.0f;
  if (active) {
    const int4 wpack = reinterpret_cast<const int4*>(w)[tid];
    const half2* xh = reinterpret_cast<const half2*>(&xbuf);
    const half2* wh = reinterpret_cast<const half2*>(&wpack);
    half2* oh = reinterpret_cast<half2*>(&obuf);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 xv = __half22float2(xh[j]);
      const float2 wv = __half22float2(wh[j]);
      oh[j] = __floats2half2_rn(xv.x * inv * wv.x, xv.y * inv * wv.y);
      const float2 ov = __half22float2(oh[j]);
      local_max = fmaxf(local_max, fmaxf(fabsf(ov.x), fabsf(ov.y)));
    }
    reinterpret_cast<int4*>(y)[tid] = obuf;
  }
  // Group-32 scales (q8_1 style): each thread owns one 8-column window; a 4-lane
  // butterfly maxes the 32-column group. No block-wide reduction, no barrier.
  local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 1));
  local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, 2));
  float g_scale = local_max / static_cast<float>(max_q);
  if (g_scale < 1.0e-8f) {
    g_scale = 1.0e-8f;
  }
  if (active) {
    if ((tid & 3) == 0) {
      xscale[tid / 4] = g_scale;
    }
    const float inv_scale = 1.0f / g_scale;
    const half2* oh = reinterpret_cast<const half2*>(&obuf);
    int q[8];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float2 v = __half22float2(oh[j]);
      int a = __float2int_rn(v.x * inv_scale);
      int b = __float2int_rn(v.y * inv_scale);
      q[2 * j] = max(-max_q, min(max_q, a));
      q[2 * j + 1] = max(-max_q, min(max_q, b));
    }
    char4* dst4 = reinterpret_cast<char4*>(xq);
    dst4[tid * 2] = make_char4(static_cast<signed char>(q[0]), static_cast<signed char>(q[2]),
                               static_cast<signed char>(q[4]), static_cast<signed char>(q[6]));
    dst4[tid * 2 + 1] = make_char4(static_cast<signed char>(q[1]), static_cast<signed char>(q[3]),
                                   static_cast<signed char>(q[5]), static_cast<signed char>(q[7]));
  }
}

// Causal row-softmax over an attention score matrix, in place.
//
// Used by the tensor-core prefill attention: Q.K^T comes out of a cuBLAS batched GEMM as a
// [heads][chunk][keys] score matrix, this masks and normalises each query's row, and the
// result feeds straight back into a second GEMM as P.
//
// Row (h, i) is query token (q_start + i), so it may attend to keys j <= q_start + i. Masked
// entries are written as ZERO, not -inf: they are consumed by a GEMM, not another softmax, so
// a zero weight is what drops them from the P.V product.
__global__ void softmax_causal_rows_kernel(half* s, int chunk_stride, int keys, int q_start,
                                           int window) {
  const int i = blockIdx.x;   // query within the chunk
  const int h = blockIdx.y;   // head
  // chunk_stride is the ALLOCATED rows per head (kChunk), not the rows in flight: the GEMM
  // writes its per-head slice at h * kChunk * keys, so the stride must match that even when
  // the final chunk is short. grid.x carries the rows actually written.
  half* row = s + (static_cast<std::size_t>(h) * chunk_stride + i) * keys;

  const int valid = min(keys, q_start + i + 1);  // causal bound
  // Sliding window: keys below (q_abs - window + 1) are masked out entirely -- excluded
  // from max and sum, written as zero (they feed a GEMM, so zero drops them from P.V).
  const int lo = (window > 0) ? max(0, q_start + i + 1 - window) : 0;
  const int tid = threadIdx.x;

  __shared__ float sh_max;
  __shared__ float sh_sum;

  float m = -3.402823466e+38F;
  for (int j = lo + tid; j < valid; j += blockDim.x) {
    m = fmaxf(m, __half2float(row[j]));
  }
  m = warp_max_f(m);
  __shared__ float warp_m[32];
  const int lane = tid & 31, warp = tid / 32;
  const int warps = (blockDim.x + 31) / 32;
  if (lane == 0) warp_m[warp] = m;
  __syncthreads();
  if (tid == 0) {
    float mm = -3.402823466e+38F;
    for (int w = 0; w < warps; ++w) mm = fmaxf(mm, warp_m[w]);
    sh_max = mm;
  }
  __syncthreads();
  const float mx = sh_max;

  // Sum WITHOUT writing: the scores stay in fp16-from-the-GEMM until the single final store.
  //
  // The first version wrote exp() back into the row here and then multiplied by 1/sum in a
  // second pass -- rounding every probability to fp16 TWICE. The old kernel keeps them in fp32
  // throughout, so that doubled the precision loss for no reason. One rounding is unavoidable
  // (the second GEMM consumes P as fp16); two is just sloppy.
  float sum = 0.0f;
  for (int j = lo + tid; j < valid; j += blockDim.x) {
    sum += __expf(__half2float(row[j]) - mx);
  }
  sum = warp_sum(sum);
  __shared__ float warp_s[32];
  if (lane == 0) warp_s[warp] = sum;
  __syncthreads();
  if (tid == 0) {
    float ss = 0.0f;
    for (int w = 0; w < warps; ++w) ss += warp_s[w];
    sh_sum = (ss > 0.0f) ? ss : 1.0f;
  }
  __syncthreads();
  const float inv = 1.0f / sh_sum;

  for (int j = lo + tid; j < valid; j += blockDim.x) {
    row[j] = __float2half(__expf(__half2float(row[j]) - mx) * inv);
  }
  // Zero the past-the-window and the future: neither may contribute to P.V.
  for (int j = tid; j < lo; j += blockDim.x) {
    row[j] = __float2half(0.0f);
  }
  for (int j = valid + tid; j < keys; j += blockDim.x) {
    row[j] = __float2half(0.0f);
  }
}

// Builds the pointer arrays for the tensor-core prefill attention's batched GEMMs, ON DEVICE.
//
// These used to be built on the host into a pinned buffer and cudaMemcpyAsync'd. That is a
// RACE: prefill_attention_tensorcore runs once per LAYER, so layer L+1 overwrote the staging
// buffer while layer L's async copy was still in flight, and layer L's GEMMs could pick up
// layer L+1's K/V pointers. It produced DIFFERENT LOGITS ON EVERY RUN.
//
// (It hid at first because the staging buffer was a pageable std::vector, and a pageable
// cudaMemcpyAsync is effectively synchronous. "Optimising" it to pinned memory made the copy
// genuinely async and exposed the bug that had been there all along.)
//
// Building them in a kernel on the compute stream makes the ordering structural: the pointers
// cannot be written late or early relative to the GEMMs that consume them.
__global__ void build_attention_ptrs_kernel(const half* k_layer, const half* v_layer,
                                            const half* q, half* scores, half* out, void** ptrs,
                                            int num_heads, int group, int head_dim, int kchunk,
                                            int keys, int q_stride, int out_stride, int c0) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= num_heads) {
    return;
  }
  const int kvh = h / group;
  half* srow = scores + static_cast<std::size_t>(h) * kchunk * keys;

  ptrs[h] = const_cast<half*>(k_layer + static_cast<std::size_t>(kvh) * head_dim);
  ptrs[num_heads + h] = const_cast<half*>(
      q + static_cast<std::size_t>(c0) * q_stride + static_cast<std::size_t>(h) * head_dim);
  ptrs[2 * num_heads + h] = srow;
  ptrs[3 * num_heads + h] = const_cast<half*>(v_layer + static_cast<std::size_t>(kvh) * head_dim);
  ptrs[4 * num_heads + h] = srow;
  ptrs[5 * num_heads + h] =
      out + static_cast<std::size_t>(c0) * out_stride + static_cast<std::size_t>(h) * head_dim;
}

}  // namespace

// Host launch wrappers keep kernel-selection policy out of the runtime call
// sites.
void launch_rmsnorm_quant_perm8(const half* x, const half* w, half* y, std::int8_t* xq,
                                float* xscale, int cols, float eps, cudaStream_t stream) {
  constexpr int kThreads = 256;
  rmsnorm_quant_perm8_kernel<kThreads>
      <<<1, kThreads, 0, stream>>>(x, w, y, xq, xscale, cols, eps, 127);
}

void launch_rmsnorm(const half* x, const half* weight, half* y, int rows, int cols, float eps,
                    cudaStream_t stream) {
  // Fast path needs 128-bit alignment (cols % 8) and the row to fit in registers.
  constexpr int kThreads = 256;
  const bool legacy = [] {
    const char* e = std::getenv("CPI_LEGACY_RMSNORM");
    return e && *e == '1';
  }();
  if (!legacy && (cols % 8) == 0 && (cols / 8) <= kThreads * 8) {
    rmsnorm_fast_kernel<kThreads><<<rows, kThreads, 0, stream>>>(x, weight, y, cols, eps);
    return;
  }
  const int threads = choose_rmsnorm_threads(cols);
  rmsnorm_kernel_simple<<<rows, threads, 0, stream>>>(x, weight, y, cols, eps);
}

void launch_rmsnorm_offset(const half* x, const half* weight, half* y, int rows, int cols,
                           float eps, cudaStream_t stream) {
  const int threads = choose_rmsnorm_threads(cols);
  rmsnorm_offset_kernel_simple<<<rows, threads, 0, stream>>>(x, weight, y, cols, eps);
}

void launch_layernorm(const half* x, const half* weight, const half* bias, half* y, int rows,
                      int cols, float eps, cudaStream_t stream) {
  const int threads = choose_rmsnorm_threads(cols);
  layernorm_kernel<<<rows, threads, 0, stream>>>(x, weight, bias, y, cols, eps);
}

void launch_embedding_lookup(const half* embedding, const int* token_ids, half* out, int num_tokens,
                             int hidden, cudaStream_t stream) {
  if (num_tokens <= 0 || hidden <= 0) {
    return;
  }
  constexpr int threads = 256;
  embedding_lookup_kernel<<<num_tokens, threads, 0, stream>>>(embedding, token_ids, out, hidden);
}

void launch_rope_inplace(half* q, half* k, int num_heads_q, int num_heads_k, int head_dim,
                         int position, float rope_theta, cudaStream_t stream) {
  const int threads = head_dim / 2;
  const int blocks = (num_heads_q > num_heads_k) ? num_heads_q : num_heads_k;
  rope_inplace_kernel<<<blocks, threads, 0, stream>>>(q, k, num_heads_q, num_heads_k, head_dim,
                                                      position, rope_theta);
}

void launch_rope_inplace_table(half* q, half* k, int num_heads_q, int num_heads_k, int head_dim,
                               int position, const float* cos_table, const float* sin_table,
                               cudaStream_t stream) {
  const int threads = head_dim / 2;
  const int blocks = (num_heads_q > num_heads_k) ? num_heads_q : num_heads_k;
  rope_inplace_table_kernel<<<blocks, threads, 0, stream>>>(
      q, k, num_heads_q, num_heads_k, head_dim, position, cos_table, sin_table);
}

void launch_rope_inplace_partial_table(half* q, half* k, int num_heads_q, int num_heads_k,
                                       int head_dim, int rotary_dim, int position,
                                       const float* cos_table, const float* sin_table,
                                       cudaStream_t stream) {
  if (rotary_dim <= 0 || (rotary_dim & 1) != 0) {
    return;
  }
  const int threads = rotary_dim / 2;
  const int blocks = (num_heads_q > num_heads_k) ? num_heads_q : num_heads_k;
  rope_inplace_partial_table_kernel<<<blocks, threads, 0, stream>>>(
      q, k, num_heads_q, num_heads_k, head_dim, rotary_dim, position, cos_table, sin_table);
}

void launch_rope_inplace_partial_table_device_pos(half* q, half* k, int num_heads_q,
                                                  int num_heads_k, int head_dim, int rotary_dim,
                                                  const int* position, const float* cos_table,
                                                  const float* sin_table, cudaStream_t stream) {
  if (rotary_dim <= 0 || (rotary_dim & 1) != 0) {
    return;
  }
  const int threads = rotary_dim / 2;
  const int blocks = (num_heads_q > num_heads_k) ? num_heads_q : num_heads_k;
  rope_inplace_partial_table_device_pos_kernel<<<blocks, threads, 0, stream>>>(
      q, k, num_heads_q, num_heads_k, head_dim, rotary_dim, position, cos_table, sin_table);
}

void launch_rmsnorm_add_scale(half* x, const half* tmp, const half* w, int cols, float eps,
                              float alpha, cudaStream_t stream) {
  rmsnorm_add_scale_kernel<<<1, 256, 0, stream>>>(x, tmp, w, cols, eps, alpha);
}

void launch_rmsnorm_rope(half* s, const half* w, int heads, int head_dim, int position,
                         const int* pos_ptr, const float* cos_table, const float* sin_table,
                         float eps, cudaStream_t stream) {
  rmsnorm_rope_kernel<<<heads, 256, 0, stream>>>(s, w, head_dim, position, pos_ptr, cos_table,
                                                 sin_table, eps);
}

void launch_rope_inplace_device_pos(half* q, half* k, int num_heads_q, int num_heads_k,
                                    int head_dim, const int* position, const float* cos_table,
                                    const float* sin_table, cudaStream_t stream) {
  const int threads = head_dim / 2;
  const int blocks = (num_heads_q > num_heads_k) ? num_heads_q : num_heads_k;
  rope_inplace_device_pos_kernel<<<blocks, threads, 0, stream>>>(
      q, k, num_heads_q, num_heads_k, head_dim, position, cos_table, sin_table);
}

void launch_rope_inplace_batched_strided(half* q, half* k, int num_tokens, int num_heads_q,
                                        int num_heads_k, int head_dim, int start_position,
                                        const float* cos_table, const float* sin_table,
                                        int q_row_stride, int k_row_stride, cudaStream_t stream) {
  const int threads = head_dim / 2;
  const int blocks = (num_heads_q > num_heads_k) ? num_heads_q : num_heads_k;
  const dim3 grid(blocks, num_tokens);
  rope_inplace_batched_kernel<<<grid, threads, 0, stream>>>(
      q, k, num_tokens, num_heads_q, num_heads_k, head_dim, start_position, cos_table, sin_table,
      q_row_stride, k_row_stride);
}

void launch_rope_inplace_batched(half* q, half* k, int num_tokens, int num_heads_q, int num_heads_k,
                                 int head_dim, int start_position, const float* cos_table,
                                 const float* sin_table, cudaStream_t stream) {
  // Natural (contiguous) strides -> identical to before the stride parameters existed.
  launch_rope_inplace_batched_strided(q, k, num_tokens, num_heads_q, num_heads_k, head_dim,
                                      start_position, cos_table, sin_table,
                                      num_heads_q * head_dim, num_heads_k * head_dim, stream);
}

// Per-position RoPE (P2 batched decode): each of num_tokens rows uses its own
// positions[row] instead of a contiguous start_position.
void launch_rope_inplace_perpos(half* q, half* k, int num_tokens, int num_heads_q, int num_heads_k,
                                int head_dim, const int* positions, const float* cos_table,
                                const float* sin_table, cudaStream_t stream) {
  const int threads = head_dim / 2;
  const int blocks = (num_heads_q > num_heads_k) ? num_heads_q : num_heads_k;
  const dim3 grid(blocks, num_tokens);
  rope_inplace_perpos_kernel<<<grid, threads, 0, stream>>>(
      q, k, num_tokens, num_heads_q, num_heads_k, head_dim, positions, cos_table, sin_table);
}
void launch_build_attention_ptrs(const half* k_layer, const half* v_layer, const half* q,
                                 half* scores, half* out, void** ptrs, int num_heads, int group,
                                 int head_dim, int kchunk, int keys, int q_stride, int out_stride,
                                 int c0, cudaStream_t stream) {
  const int threads = 64;
  const int blocks = (num_heads + threads - 1) / threads;
  build_attention_ptrs_kernel<<<blocks, threads, 0, stream>>>(k_layer, v_layer, q, scores, out,
                                                              ptrs, num_heads, group, head_dim,
                                                              kchunk, keys, q_stride, out_stride,
                                                              c0);
}

void launch_softmax_causal_rows(half* scores, int heads, int chunk_stride, int rows, int keys,
                                int q_start, cudaStream_t stream, int window) {
  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(heads));
  softmax_causal_rows_kernel<<<grid, 256, 0, stream>>>(scores, chunk_stride, keys, q_start,
                                                       window);
}

void launch_attention_prefill(const half* q, const half* k_cache, const half* v_cache, half* out,
                              int num_tokens, int start_position, int num_heads, int num_kv_heads,
                              int head_dim, cudaStream_t stream, bool causal, const int* limits,
                              int window) {
  const int causal_i = causal ? 1 : 0;
  const dim3 grid(num_heads, num_tokens);
  // Fallback kernel: only reached when the engine's tensor-core prefill path is unavailable
  // (unsupported geometry, or a per-token `limits` / sliding `window` mask, which only this
  // kernel implements). CPI_PREFILL_WARPS overrides the block width.
  static const int env_warps = [] {
    const char* e = std::getenv("CPI_PREFILL_WARPS");
    return e ? atoi(e) : 0;
  }();
  if (head_dim > 0 && (head_dim % 2) == 0 && head_dim <= 256) {
    if (head_dim <= 64) {
      // Do not change the default width: the warp count sets the fp32 accumulation order in
      // the softmax/dot reduction, so a different value shifts logits and can flip a greedy
      // argmax. Wider blocks are not worth that here -- this kernel is bound by scalar-FMA
      // throughput, not occupancy.
      const int w = (env_warps > 0) ? env_warps : 2;
      const std::size_t smem64 = static_cast<std::size_t>(head_dim) * sizeof(half) +
                                 static_cast<std::size_t>(3 * w + 2) * sizeof(float);
      if (w >= 8) {
        attention_prefill_kernel_tiled<8><<<grid, 8 * 32, smem64, stream>>>(
            q, k_cache, v_cache, out, num_tokens, start_position, num_heads, num_kv_heads, head_dim,
            causal_i, limits, window);
        return;
      }
      if (w >= 4) {
        attention_prefill_kernel_tiled<4><<<grid, 4 * 32, smem64, stream>>>(
            q, k_cache, v_cache, out, num_tokens, start_position, num_heads, num_kv_heads, head_dim,
            causal_i, limits, window);
        return;
      }
      constexpr int warps = 2;
      constexpr int threads = warps * 32;
      const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                               static_cast<std::size_t>(3 * warps + 2) * sizeof(float);
      attention_prefill_kernel_tiled<warps><<<grid, threads, smem, stream>>>(
          q, k_cache, v_cache, out, num_tokens, start_position, num_heads, num_kv_heads, head_dim,
          causal_i, limits, window);
      return;
    }

    constexpr int warps = 4;
    constexpr int threads = warps * 32;
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                             static_cast<std::size_t>(3 * warps + 2) * sizeof(float);
    attention_prefill_kernel_tiled<warps><<<grid, threads, smem, stream>>>(
        q, k_cache, v_cache, out, num_tokens, start_position, num_heads, num_kv_heads, head_dim,
        causal_i, limits, window);
    return;
  }

  constexpr int threads = 128;
  const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                           static_cast<std::size_t>(threads + 3) * sizeof(float);
  attention_prefill_kernel_fallback<<<grid, threads, smem, stream>>>(
      q, k_cache, v_cache, out, num_tokens, start_position, num_heads, num_kv_heads, head_dim,
      causal_i, limits, window);
}

// Paged prefill attention (P3 phase 2d). K/V gathered via block_table from a
// block pool. Targets head_dim<=256 even (128 on Qwen/Llama); the split-K decode
// path handles the same family, so no fallback is needed here.
void launch_attention_prefill_paged(const half* q, const half* k_pool, const half* v_pool,
                                    const int* block_table, half* out, int num_tokens,
                                    int start_position, int num_heads, int num_kv_heads,
                                    int head_dim, int block_size, cudaStream_t stream) {
  const dim3 grid(num_heads, num_tokens);
  const int warps = (head_dim <= 64) ? 2 : 4;
  const int threads = warps * 32;
  const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                           static_cast<std::size_t>(3 * warps + 2) * sizeof(float);
  if (warps == 2) {
    attention_prefill_kernel_tiled_paged<2><<<grid, threads, smem, stream>>>(
        q, k_pool, v_pool, block_table, out, num_tokens, start_position, num_heads, num_kv_heads,
        head_dim, block_size);
  } else {
    attention_prefill_kernel_tiled_paged<4><<<grid, threads, smem, stream>>>(
        q, k_pool, v_pool, block_table, out, num_tokens, start_position, num_heads, num_kv_heads,
        head_dim, block_size);
  }
}

// Scatter freshly-projected prefill KV rows into the block pool at paged positions.
void launch_store_kv_paged(half* k_pool, half* v_pool, const half* k_src, const half* v_src,
                           const int* block_table, int base_pos, int rows, int kv_hidden,
                           int block_size, cudaStream_t stream) {
  if (rows <= 0) return;
  store_kv_paged_kernel<<<rows, 128, 0, stream>>>(k_pool, v_pool, k_src, v_src, block_table,
                                                  base_pos, rows, kv_hidden, block_size);
}

// Batched decode KV scatter (P2): one token per sequence to its own block table
// at its own position.
void launch_store_kv_batched_paged(half* k_pool, half* v_pool, const half* k_src, const half* v_src,
                                   const int* block_tables, const int* positions, int max_blocks,
                                   int batch, int kv_hidden, int block_size, cudaStream_t stream) {
  if (batch <= 0) return;
  store_kv_batched_paged_kernel<<<batch, 128, 0, stream>>>(k_pool, v_pool, k_src, v_src,
                                                           block_tables, positions, max_blocks,
                                                           batch, kv_hidden, block_size);
}

}  // namespace kernels
