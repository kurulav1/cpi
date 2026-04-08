// kernels.cu
//
// CUDA kernel implementations and host launch wrappers for the inference
// runtime. `runtime/kernels.cuh` documents the public API; this file focuses on
// the performance-critical implementation details used by RMSNorm, embedding,
// RoPE, and prefill-attention kernels.

#include "runtime/kernels.cuh"

#include <cstdint>
#include <cuda_fp16.h>

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

// Basic normalization, embedding, and RoPE kernels.
__global__ void rmsnorm_kernel(const half* x,
                               const half* w,
                               half* y,
                               int cols,
                               float eps) {
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
      y_row2[col2] = __halves2half2(__float2half(xv.x * inv * wv.x), __float2half(xv.y * inv * wv.y));
    }
  } else {
    for (int col = tid; col < cols; col += blockDim.x) {
      const float xv = __half2float(x_row[col]);
      const float ww = __half2float(w[col]);
      y_row[col] = __float2half(xv * inv * ww);
    }
  }
}

__global__ void layernorm_kernel(const half* x,
                                 const half* w,
                                 const half* b,
                                 half* y,
                                 int cols,
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

__global__ void embedding_lookup_kernel(const half* embedding,
                                        const int* token_ids,
                                        half* out,
                                        int hidden) {
  const int token_idx = blockIdx.x;
  const int token = token_ids[token_idx];
  const half* src = embedding + static_cast<std::size_t>(token) * static_cast<std::size_t>(hidden);
  half* dst = out + static_cast<std::size_t>(token_idx) * static_cast<std::size_t>(hidden);

  if ((hidden & 7) == 0) {
    const int vec_count = hidden / 8;
    const int4* src4 = reinterpret_cast<const int4*>(src);
    int4* dst4 = reinterpret_cast<int4*>(dst);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      dst4[i] = src4[i];
    }
    return;
  }

  if ((hidden & 1) == 0) {
    const int vec_count = hidden / 2;
    const half2* src2 = reinterpret_cast<const half2*>(src);
    half2* dst2 = reinterpret_cast<half2*>(dst);
    for (int i = threadIdx.x; i < vec_count; i += blockDim.x) {
      dst2[i] = src2[i];
    }
    return;
  }

  for (int col = threadIdx.x; col < hidden; col += blockDim.x) {
    dst[col] = src[col];
  }
}

__global__ void rope_inplace_kernel(half* q,
                                    half* k,
                                    int num_heads_q,
                                    int num_heads_k,
                                    int head_dim,
                                    int position,
                                    float rope_theta) {
  const int head = blockIdx.x;
  const int pair = threadIdx.x;
  const int half_dim = head_dim / 2;
  if (pair >= half_dim) {
    return;
  }

  const float theta = powf(rope_theta, -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
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

__global__ void rope_inplace_table_kernel(half* q,
                                          half* k,
                                          int num_heads_q,
                                          int num_heads_k,
                                          int head_dim,
                                          int position,
                                          const float* cos_table,
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

__global__ void rope_inplace_device_pos_kernel(half* q,
                                               half* k,
                                               int num_heads_q,
                                               int num_heads_k,
                                               int head_dim,
                                               const int* position_ptr,
                                               const float* cos_table,
                                               const float* sin_table) {
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

__global__ void rope_inplace_batched_kernel(half* q,
                                            half* k,
                                            int num_tokens,
                                            int num_heads_q,
                                            int num_heads_k,
                                            int head_dim,
                                            int start_position,
                                            const float* cos_table,
                                            const float* sin_table) {
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

// Decode-time attention helpers and kernels.
// Flatten [time, head, dim] coordinates into the packed KV-cache layout.
__device__ __forceinline__ int cache_index(int t, int head, int d, int num_heads, int head_dim) {
  return (t * num_heads + head) * head_dim + d;
}

// Prefill path: each block handles one [head, token] pair and attends over
// the cached prefix plus its own in-chunk prefix.
__global__ void attention_prefill_kernel_fallback(const half* q,
                                                  const half* k_cache,
                                                  const half* v_cache,
                                                  half* out,
                                                  int num_tokens,
                                                  int start_position,
                                                  int num_heads,
                                                  int num_kv_heads,
                                                  int head_dim) {
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
  const int kv_head = ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const bool active_dim = tid < head_dim;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[q_base + d];
  }
  __syncthreads();

  float running_m = -1.0e30f;
  float running_l = 0.0f;
  float acc = 0.0f;
  const int limit = start_position + token + 1;
  for (int t = 0; t < limit; ++t) {
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

    if (active_dim) {
      acc = acc * alpha_shared[0] + beta_shared[0] * __half2float(v_cache[base + tid]);
    }
    __syncthreads();
  }

  if (active_dim) {
    out[out_base + tid] = __float2half(acc * inv_l_shared[0]);
  }
}

// Tiled prefill keeps the causal limit per token while still vectorizing the
// K dot Q work inside each tile.
template <int WarpsPerBlock>
__global__ void attention_prefill_kernel_tiled(const half* q,
                                               const half* k_cache,
                                               const half* v_cache,
                                               half* out,
                                               int num_tokens,
                                               int start_position,
                                               int num_heads,
                                               int num_kv_heads,
                                               int head_dim) {
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
  const int kv_head = ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
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

  float acc = 0.0f;
  for (int tile_base = 0; tile_base < limit; tile_base += WarpsPerBlock) {
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

    for (int d = tid; d < head_dim; d += blockDim.x) {
      float acc_local = acc;
      const int tile_tokens = min(WarpsPerBlock, limit - tile_base);
      for (int i = 0; i < tile_tokens; ++i) {
        const int base = cache_index(tile_base + i, kv_head, 0, num_kv_heads, head_dim);
        acc_local = acc_local * alpha_shared[i] + beta_shared[i] * __half2float(v_cache[base + d]);
      }
      acc = acc_local;
    }
    __syncthreads();
  }

  const float inv_l = 1.0f / fmaxf(stats_shared[1], 1e-8f);
  for (int d = tid; d < head_dim; d += blockDim.x) {
    out[out_base + d] = __float2half(acc * inv_l);
  }
}

}  // namespace

// Host launch wrappers keep kernel-selection policy out of the runtime call
// sites.
void launch_rmsnorm(const half* x,
                    const half* weight,
                    half* y,
                    int rows,
                    int cols,
                    float eps,
                    cudaStream_t stream) {
  const int threads = choose_rmsnorm_threads(cols);
  rmsnorm_kernel<<<rows, threads, 0, stream>>>(x, weight, y, cols, eps);
}

void launch_layernorm(const half* x,
                      const half* weight,
                      const half* bias,
                      half* y,
                      int rows,
                      int cols,
                      float eps,
                      cudaStream_t stream) {
  const int threads = choose_rmsnorm_threads(cols);
  layernorm_kernel<<<rows, threads, 0, stream>>>(x, weight, bias, y, cols, eps);
}

void launch_embedding_lookup(const half* embedding,
                             const int* token_ids,
                             half* out,
                             int num_tokens,
                             int hidden,
                             cudaStream_t stream) {
  const int threads = choose_copy_threads(hidden);
  embedding_lookup_kernel<<<num_tokens, threads, 0, stream>>>(embedding, token_ids, out, hidden);
}

void launch_rope_inplace(half* q,
                         half* k,
                         int num_heads_q,
                         int num_heads_k,
                         int head_dim,
                         int position,
                         float rope_theta,
                         cudaStream_t stream) {
  const int threads = head_dim / 2;
  const int blocks = (num_heads_q > num_heads_k) ? num_heads_q : num_heads_k;
  rope_inplace_kernel<<<blocks, threads, 0, stream>>>(q, k, num_heads_q, num_heads_k, head_dim, position, rope_theta);
}

void launch_rope_inplace_table(half* q,
                               half* k,
                               int num_heads_q,
                               int num_heads_k,
                               int head_dim,
                               int position,
                               const float* cos_table,
                               const float* sin_table,
                               cudaStream_t stream) {
  const int threads = head_dim / 2;
  const int blocks = (num_heads_q > num_heads_k) ? num_heads_q : num_heads_k;
  rope_inplace_table_kernel<<<blocks, threads, 0, stream>>>(
      q, k, num_heads_q, num_heads_k, head_dim, position, cos_table, sin_table);
}

void launch_rope_inplace_device_pos(half* q,
                                    half* k,
                                    int num_heads_q,
                                    int num_heads_k,
                                    int head_dim,
                                    const int* position,
                                    const float* cos_table,
                                    const float* sin_table,
                                    cudaStream_t stream) {
  const int threads = head_dim / 2;
  const int blocks = (num_heads_q > num_heads_k) ? num_heads_q : num_heads_k;
  rope_inplace_device_pos_kernel<<<blocks, threads, 0, stream>>>(
      q, k, num_heads_q, num_heads_k, head_dim, position, cos_table, sin_table);
}

void launch_rope_inplace_batched(half* q,
                                 half* k,
                                 int num_tokens,
                                 int num_heads_q,
                                 int num_heads_k,
                                 int head_dim,
                                 int start_position,
                                 const float* cos_table,
                                 const float* sin_table,
                                 cudaStream_t stream) {
  const int threads = head_dim / 2;
  const int blocks = (num_heads_q > num_heads_k) ? num_heads_q : num_heads_k;
  const dim3 grid(blocks, num_tokens);
  rope_inplace_batched_kernel<<<grid, threads, 0, stream>>>(
      q, k, num_tokens, num_heads_q, num_heads_k, head_dim, start_position, cos_table, sin_table);
}
void launch_attention_prefill(const half* q,
                              const half* k_cache,
                              const half* v_cache,
                              half* out,
                              int num_tokens,
                              int start_position,
                              int num_heads,
                              int num_kv_heads,
                              int head_dim,
                              cudaStream_t stream) {
  const dim3 grid(num_heads, num_tokens);
  if (head_dim > 0 && (head_dim % 2) == 0 && head_dim <= 256) {
    if (head_dim <= 64) {
      constexpr int warps = 2;
      constexpr int threads = warps * 32;
      const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                               static_cast<std::size_t>(3 * warps + 2) * sizeof(float);
      attention_prefill_kernel_tiled<warps><<<grid, threads, smem, stream>>>(
          q, k_cache, v_cache, out, num_tokens, start_position, num_heads, num_kv_heads, head_dim);
      return;
    }

    constexpr int warps = 4;
    constexpr int threads = warps * 32;
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                             static_cast<std::size_t>(3 * warps + 2) * sizeof(float);
    attention_prefill_kernel_tiled<warps><<<grid, threads, smem, stream>>>(
        q, k_cache, v_cache, out, num_tokens, start_position, num_heads, num_kv_heads, head_dim);
    return;
  }

  constexpr int threads = 128;
  const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                           static_cast<std::size_t>(threads + 3) * sizeof(float);
  attention_prefill_kernel_fallback<<<grid, threads, smem, stream>>>(
      q, k_cache, v_cache, out, num_tokens, start_position, num_heads, num_kv_heads, head_dim);
}


}  // namespace kernels
