// kernels_attention_decode.cu
//
// CUDA kernels and host launch wrappers for FP16 decode-time attention paths
// plus device-position KV-cache updates.

#include "runtime/kernels.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <sm_61_intrinsics.h>

namespace kernels {
namespace {

inline int choose_copy_threads(int cols) {
  return (cols <= 2048) ? 128 : 256;
}

__device__ __forceinline__ float warp_sum(float v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

template <typename T>
__device__ __forceinline__ T neg_inf();

template <>
__device__ __forceinline__ float neg_inf<float>() {
  return -3.402823466e+38F;
}

__device__ __forceinline__ int cache_index(int t, int head, int d, int num_heads, int head_dim) {
  return (t * num_heads + head) * head_dim + d;
}

__global__ void attention_step_kernel_fallback(const half* q,
                                               const half* k_cache,
                                               const half* v_cache,
                                               half* out,
                                               int seq_len,
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
  const int tid = threadIdx.x;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head = ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const bool active_dim = tid < head_dim;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[head * head_dim + d];
  }
  __syncthreads();

  float running_m = -1.0e30f;
  float running_l = 0.0f;
  float acc = 0.0f;

  for (int t = 0; t < seq_len; ++t) {
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
    out[head * head_dim + tid] = __float2half(acc * inv_l_shared[0]);
  }
}

__global__ void attention_step_kernel_fallback_device_pos(const half* q,
                                                          const half* k_cache,
                                                          const half* v_cache,
                                                          half* out,
                                                          const int* position_ptr,
                                                          int num_heads,
                                                          int num_kv_heads,
                                                          int head_dim) {
  const int seq_len = position_ptr[0] + 1;
  extern __shared__ unsigned char smem_bytes[];
  half* q_shared = reinterpret_cast<half*>(smem_bytes);
  float* red = reinterpret_cast<float*>(q_shared + head_dim);
  float* alpha_shared = red + blockDim.x;
  float* beta_shared = alpha_shared + 1;
  float* inv_l_shared = beta_shared + 1;
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head = ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const bool active_dim = tid < head_dim;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[head * head_dim + d];
  }
  __syncthreads();

  float running_m = -1.0e30f;
  float running_l = 0.0f;
  float acc = 0.0f;

  for (int t = 0; t < seq_len; ++t) {
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
    out[head * head_dim + tid] = __float2half(acc * inv_l_shared[0]);
  }
}

// Tiled decode attention keeps the query resident in shared memory, stages one
// value tile at a time, and merges tile-local softmax statistics so the block
// avoids a fully serial per-token expf chain.
template <int WarpsPerBlock>
__global__ void attention_step_kernel_tiled(const half* q,
                                            const half* k_cache,
                                            const half* v_cache,
                                            half* out,
                                            int seq_len,
                                            int num_heads,
                                            int num_kv_heads,
                                            int head_dim) {
  // Shared-memory layout:
  //   half  q_shared[head_dim]
  //   float score_shared[WarpsPerBlock]
  //   float beta_shared[WarpsPerBlock]   // exp(score[i] - tile_m)
  //   float stats_shared[4]              // [running_m, running_l, tile_m, tile_l]
  //   half  v_tile[WarpsPerBlock * head_dim]
  extern __shared__ unsigned char smem_bytes[];
  half*  q_shared    = reinterpret_cast<half*>(smem_bytes);
  float* score_shared = reinterpret_cast<float*>(q_shared + head_dim);
  float* beta_shared  = score_shared + WarpsPerBlock;
  float* stats_shared = beta_shared + WarpsPerBlock;  // [running_m, running_l, tile_m, tile_l]
  half*  v_tile       = reinterpret_cast<half*>(stats_shared + 4);

  const int head    = blockIdx.x;
  const int tid     = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane    = tid % warpSize;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head = ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const int head_pairs = head_dim / 2;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[head * head_dim + d];
  }
  if (tid == 0) {
    stats_shared[0] = -1.0e30f;  // running_m
    stats_shared[1] = 0.0f;       // running_l
  }
  __syncthreads();

  float acc = 0.0f;
  for (int tile_base = 0; tile_base < seq_len; tile_base += WarpsPerBlock) {
    const int tile_tokens = min(WarpsPerBlock, seq_len - tile_base);

    // Phase 1a: each warp computes the K dot Q score for one token in the tile.
    {
      const int t = tile_base + warp_id;
      float score = -1.0e30f;
      if (warp_id < tile_tokens) {
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
      if (lane == 0 && warp_id < tile_tokens) {
        score_shared[warp_id] = score;
      }
    }

    // Phase 1b: all threads cooperatively stage the V tile into shared memory
    {
      for (int i = 0; i < tile_tokens; ++i) {
        const int base = cache_index(tile_base + i, kv_head, 0, num_kv_heads, head_dim);
        for (int d = tid; d < head_dim; d += blockDim.x) {
          v_tile[i * head_dim + d] = v_cache[base + d];
        }
      }
    }
    __syncthreads();

    // Phase 2: tid==0 computes tile_m and per-token softmax weights.
    // These are independent of running_m/running_l, so the serial work is
    // minimal: tile_tokens max comparisons + tile_tokens expf calls (no
    // chained dependency on running state).
    if (tid == 0) {
      float tile_m = -1.0e30f;
      for (int i = 0; i < tile_tokens; ++i) {
        tile_m = fmaxf(tile_m, score_shared[i]);
      }
      float tile_l = 0.0f;
      for (int i = 0; i < tile_tokens; ++i) {
        const float b = expf(score_shared[i] - tile_m);
        beta_shared[i] = b;
        tile_l += b;
      }
      stats_shared[2] = tile_m;
      stats_shared[3] = tile_l;
    }
    __syncthreads();

    // Phase 3: all threads accumulate tile_o[d] from the staged V values and
    // merge it into acc. Only two expf calls (c_prev, c_tile) are needed per
    // tile, and the accumulation stays fully parallel across the block.
    {
      const float tile_m    = stats_shared[2];
      const float tile_l    = stats_shared[3];
      const float running_m = stats_shared[0];
      const float running_l = stats_shared[1];
      const float new_m  = fmaxf(running_m, tile_m);
      const float c_prev = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float c_tile = expf(tile_m - new_m);

      for (int d = tid; d < head_dim; d += blockDim.x) {
        float tile_o = 0.0f;
        for (int i = 0; i < tile_tokens; ++i) {
          tile_o += beta_shared[i] * __half2float(v_tile[i * head_dim + d]);
        }
        acc = acc * c_prev + tile_o * c_tile;
      }
      if (tid == 0) {
        stats_shared[0] = new_m;
        stats_shared[1] = running_l * c_prev + tile_l * c_tile;
      }
    }
    __syncthreads();
  }

  const float inv_l = 1.0f / fmaxf(stats_shared[1], 1e-8f);
  for (int d = tid; d < head_dim; d += blockDim.x) {
    out[head * head_dim + d] = __float2half(acc * inv_l);
  }
}

// Device-position variant of the tiled decode path above. The sequence length
// is read from device memory so the kernel can be captured in a CUDA Graph.
template <int WarpsPerBlock>
__global__ void attention_step_kernel_tiled_device_pos(const half* q,
                                                       const half* k_cache,
                                                       const half* v_cache,
                                                       half* out,
                                                       const int* position_ptr,
                                                       int num_heads,
                                                       int num_kv_heads,
                                                       int head_dim) {
  const int seq_len = position_ptr[0] + 1;
  extern __shared__ unsigned char smem_bytes[];
  half*  q_shared    = reinterpret_cast<half*>(smem_bytes);
  float* score_shared = reinterpret_cast<float*>(q_shared + head_dim);
  float* beta_shared  = score_shared + WarpsPerBlock;
  float* stats_shared = beta_shared + WarpsPerBlock;  // [running_m, running_l, tile_m, tile_l]
  half*  v_tile       = reinterpret_cast<half*>(stats_shared + 4);

  const int head    = blockIdx.x;
  const int tid     = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane    = tid % warpSize;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head = ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const int head_pairs = head_dim / 2;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[head * head_dim + d];
  }
  if (tid == 0) {
    stats_shared[0] = -1.0e30f;
    stats_shared[1] = 0.0f;
  }
  __syncthreads();

  float acc = 0.0f;
  for (int tile_base = 0; tile_base < seq_len; tile_base += WarpsPerBlock) {
    const int tile_tokens = min(WarpsPerBlock, seq_len - tile_base);

    // Phase 1a: each warp computes the K dot Q score for one token in the tile.
    {
      const int t = tile_base + warp_id;
      float score = -1.0e30f;
      if (warp_id < tile_tokens) {
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
      if (lane == 0 && warp_id < tile_tokens) {
        score_shared[warp_id] = score;
      }
    }

    // Phase 1b: stage V tile into shared memory (all threads participate)
    {
      for (int i = 0; i < tile_tokens; ++i) {
        const int base = cache_index(tile_base + i, kv_head, 0, num_kv_heads, head_dim);
        for (int d = tid; d < head_dim; d += blockDim.x) {
          v_tile[i * head_dim + d] = v_cache[base + d];
        }
      }
    }
    __syncthreads();

    // Phase 2: tile-local softmax weights with no dependency on running state.
    if (tid == 0) {
      float tile_m = -1.0e30f;
      for (int i = 0; i < tile_tokens; ++i) {
        tile_m = fmaxf(tile_m, score_shared[i]);
      }
      float tile_l = 0.0f;
      for (int i = 0; i < tile_tokens; ++i) {
        const float b = expf(score_shared[i] - tile_m);
        beta_shared[i] = b;
        tile_l += b;
      }
      stats_shared[2] = tile_m;
      stats_shared[3] = tile_l;
    }
    __syncthreads();

    // Phase 3: parallel tile accumulation + tile-level stats merge (2 expf/tile)
    {
      const float tile_m    = stats_shared[2];
      const float tile_l    = stats_shared[3];
      const float running_m = stats_shared[0];
      const float running_l = stats_shared[1];
      const float new_m  = fmaxf(running_m, tile_m);
      const float c_prev = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float c_tile = expf(tile_m - new_m);

      for (int d = tid; d < head_dim; d += blockDim.x) {
        float tile_o = 0.0f;
        for (int i = 0; i < tile_tokens; ++i) {
          tile_o += beta_shared[i] * __half2float(v_tile[i * head_dim + d]);
        }
        acc = acc * c_prev + tile_o * c_tile;
      }
      if (tid == 0) {
        stats_shared[0] = new_m;
        stats_shared[1] = running_l * c_prev + tile_l * c_tile;
      }
    }
    __syncthreads();
  }

  const float inv_l = 1.0f / fmaxf(stats_shared[1], 1e-8f);
  for (int d = tid; d < head_dim; d += blockDim.x) {
    out[head * head_dim + d] = __float2half(acc * inv_l);
  }
}

// Split-K decode, pass 1: each block computes softmax statistics and an
// unnormalized partial output for one [head, chunk] pair.
template <int WarpsPerBlock>
__global__ void attention_step_chunk_stats_kernel(const half* q,
                                                  const half* k_cache,
                                                  const half* v_cache,
                                                  float* chunk_m,
                                                  float* chunk_l,
                                                  float* chunk_o,
                                                  int seq_len,
                                                  int num_heads,
                                                  int num_kv_heads,
                                                  int head_dim,
                                                  int chunk_size,
                                                  int scratch_chunks) {
  extern __shared__ unsigned char smem_bytes[];
  half*  q_shared    = reinterpret_cast<half*>(smem_bytes);
  float* score_shared = reinterpret_cast<float*>(q_shared + head_dim);
  float* beta_shared  = score_shared + WarpsPerBlock;
  float* stats_shared = beta_shared + WarpsPerBlock;  // [running_m, running_l, tile_m, tile_l]
  half*  v_tile       = reinterpret_cast<half*>(stats_shared + 4);

  const int head = blockIdx.x;
  const int chunk = blockIdx.y;
  const int chunk_start = chunk * chunk_size;
  if (chunk_start >= seq_len) {
    return;
  }
  const int chunk_end = min(chunk_start + chunk_size, seq_len);
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane = tid % warpSize;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head = ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const int head_pairs = head_dim / 2;
  const int chunk_index = head * scratch_chunks + chunk;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[head * head_dim + d];
  }
  if (tid == 0) {
    stats_shared[0] = neg_inf<float>();
    stats_shared[1] = 0.0f;
  }
  __syncthreads();

  float acc = 0.0f;
  for (int tile_base = chunk_start; tile_base < chunk_end; tile_base += WarpsPerBlock) {
    const int tile_tokens = min(WarpsPerBlock, chunk_end - tile_base);

    {
      const int t = tile_base + warp_id;
      float score = neg_inf<float>();
      if (warp_id < tile_tokens) {
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
      if (lane == 0 && warp_id < tile_tokens) {
        score_shared[warp_id] = score;
      }
    }

    {
      for (int i = 0; i < tile_tokens; ++i) {
        const int base = cache_index(tile_base + i, kv_head, 0, num_kv_heads, head_dim);
        for (int d = tid; d < head_dim; d += blockDim.x) {
          v_tile[i * head_dim + d] = v_cache[base + d];
        }
      }
    }
    __syncthreads();

    if (tid == 0) {
      float tile_m = neg_inf<float>();
      for (int i = 0; i < tile_tokens; ++i) {
        tile_m = fmaxf(tile_m, score_shared[i]);
      }
      float tile_l = 0.0f;
      for (int i = 0; i < tile_tokens; ++i) {
        const float b = expf(score_shared[i] - tile_m);
        beta_shared[i] = b;
        tile_l += b;
      }
      stats_shared[2] = tile_m;
      stats_shared[3] = tile_l;
    }
    __syncthreads();

    {
      const float tile_m    = stats_shared[2];
      const float tile_l    = stats_shared[3];
      const float running_m = stats_shared[0];
      const float running_l = stats_shared[1];
      const float new_m  = fmaxf(running_m, tile_m);
      const float c_prev = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float c_tile = expf(tile_m - new_m);

      for (int d = tid; d < head_dim; d += blockDim.x) {
        float tile_o = 0.0f;
        for (int i = 0; i < tile_tokens; ++i) {
          tile_o += beta_shared[i] * __half2float(v_tile[i * head_dim + d]);
        }
        acc = acc * c_prev + tile_o * c_tile;
      }
      if (tid == 0) {
        stats_shared[0] = new_m;
        stats_shared[1] = running_l * c_prev + tile_l * c_tile;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    chunk_m[chunk_index] = stats_shared[0];
    chunk_l[chunk_index] = stats_shared[1];
  }
  for (int d = tid; d < head_dim; d += blockDim.x) {
    chunk_o[static_cast<std::size_t>(chunk_index) * static_cast<std::size_t>(head_dim) + static_cast<std::size_t>(d)] = acc;
  }
}

// Device-position variant of the first split-K decode pass.
template <int WarpsPerBlock>
__global__ void attention_step_chunk_stats_device_pos_kernel(const half* q,
                                                             const half* k_cache,
                                                             const half* v_cache,
                                                             float* chunk_m,
                                                             float* chunk_l,
                                                             float* chunk_o,
                                                             const int* position_ptr,
                                                             int num_heads,
                                                             int num_kv_heads,
                                                             int head_dim,
                                                             int chunk_size,
                                                             int scratch_chunks) {
  const int seq_len = position_ptr[0] + 1;
  extern __shared__ unsigned char smem_bytes[];
  half*  q_shared    = reinterpret_cast<half*>(smem_bytes);
  float* score_shared = reinterpret_cast<float*>(q_shared + head_dim);
  float* beta_shared  = score_shared + WarpsPerBlock;
  float* stats_shared = beta_shared + WarpsPerBlock;  // [running_m, running_l, tile_m, tile_l]
  half*  v_tile       = reinterpret_cast<half*>(stats_shared + 4);

  const int head = blockIdx.x;
  const int chunk = blockIdx.y;
  const int chunk_start = chunk * chunk_size;
  if (chunk_start >= seq_len) {
    return;
  }
  const int chunk_end = min(chunk_start + chunk_size, seq_len);
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane = tid % warpSize;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head = ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const int head_pairs = head_dim / 2;
  const int chunk_index = head * scratch_chunks + chunk;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[head * head_dim + d];
  }
  if (tid == 0) {
    stats_shared[0] = neg_inf<float>();
    stats_shared[1] = 0.0f;
  }
  __syncthreads();

  float acc = 0.0f;
  for (int tile_base = chunk_start; tile_base < chunk_end; tile_base += WarpsPerBlock) {
    const int tile_tokens = min(WarpsPerBlock, chunk_end - tile_base);

    {
      const int t = tile_base + warp_id;
      float score = neg_inf<float>();
      if (warp_id < tile_tokens) {
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
      if (lane == 0 && warp_id < tile_tokens) {
        score_shared[warp_id] = score;
      }
    }

    {
      for (int i = 0; i < tile_tokens; ++i) {
        const int base = cache_index(tile_base + i, kv_head, 0, num_kv_heads, head_dim);
        for (int d = tid; d < head_dim; d += blockDim.x) {
          v_tile[i * head_dim + d] = v_cache[base + d];
        }
      }
    }
    __syncthreads();

    if (tid == 0) {
      float tile_m = neg_inf<float>();
      for (int i = 0; i < tile_tokens; ++i) {
        tile_m = fmaxf(tile_m, score_shared[i]);
      }
      float tile_l = 0.0f;
      for (int i = 0; i < tile_tokens; ++i) {
        const float b = expf(score_shared[i] - tile_m);
        beta_shared[i] = b;
        tile_l += b;
      }
      stats_shared[2] = tile_m;
      stats_shared[3] = tile_l;
    }
    __syncthreads();

    {
      const float tile_m    = stats_shared[2];
      const float tile_l    = stats_shared[3];
      const float running_m = stats_shared[0];
      const float running_l = stats_shared[1];
      const float new_m  = fmaxf(running_m, tile_m);
      const float c_prev = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float c_tile = expf(tile_m - new_m);

      for (int d = tid; d < head_dim; d += blockDim.x) {
        float tile_o = 0.0f;
        for (int i = 0; i < tile_tokens; ++i) {
          tile_o += beta_shared[i] * __half2float(v_tile[i * head_dim + d]);
        }
        acc = acc * c_prev + tile_o * c_tile;
      }
      if (tid == 0) {
        stats_shared[0] = new_m;
        stats_shared[1] = running_l * c_prev + tile_l * c_tile;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    chunk_m[chunk_index] = stats_shared[0];
    chunk_l[chunk_index] = stats_shared[1];
  }
  for (int d = tid; d < head_dim; d += blockDim.x) {
    chunk_o[static_cast<std::size_t>(chunk_index) * static_cast<std::size_t>(head_dim) + static_cast<std::size_t>(d)] = acc;
  }
}

// Split-K decode, pass 2: merge the chunk-local softmax statistics and
// partial outputs into the final normalized attention result.
__global__ void attention_step_chunk_reduce_kernel(const float* chunk_m,
                                                   const float* chunk_l,
                                                   const float* chunk_o,
                                                   half* out,
                                                   int seq_len,
                                                   int num_heads,
                                                   int head_dim,
                                                   int chunk_size,
                                                   int scratch_chunks) {
  __shared__ float scale_shared[3];
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int chunk_count = (seq_len + chunk_size - 1) / chunk_size;
  float acc = 0.0f;
  float running_m = neg_inf<float>();
  float running_l = 0.0f;

  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    if (tid == 0) {
      const int idx = head * scratch_chunks + chunk;
      const float chunk_m_value = chunk_m[idx];
      const float chunk_l_value = chunk_l[idx];
      const float new_m = fmaxf(running_m, chunk_m_value);
      const float alpha = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float beta = (chunk_l_value == 0.0f) ? 0.0f : expf(chunk_m_value - new_m);
      running_l = running_l * alpha + chunk_l_value * beta;
      running_m = new_m;
      scale_shared[0] = alpha;
      scale_shared[1] = beta;
      scale_shared[2] = running_l;
    }
    __syncthreads();

    const float alpha = scale_shared[0];
    const float beta = scale_shared[1];
    const std::size_t base = (static_cast<std::size_t>(head) * static_cast<std::size_t>(scratch_chunks) +
                              static_cast<std::size_t>(chunk)) *
                             static_cast<std::size_t>(head_dim);
    for (int d = tid; d < head_dim; d += blockDim.x) {
      acc = acc * alpha + chunk_o[base + static_cast<std::size_t>(d)] * beta;
    }
    __syncthreads();
  }

  const float inv_l = 1.0f / fmaxf(scale_shared[2], 1e-8f);
  for (int d = tid; d < head_dim; d += blockDim.x) {
    out[head * head_dim + d] = __float2half(acc * inv_l);
  }
}

__global__ void attention_step_chunk_reduce_device_pos_kernel(const float* chunk_m,
                                                              const float* chunk_l,
                                                              const float* chunk_o,
                                                              half* out,
                                                              const int* position_ptr,
                                                              int num_heads,
                                                              int head_dim,
                                                              int chunk_size,
                                                              int scratch_chunks) {
  const int seq_len = position_ptr[0] + 1;
  __shared__ float scale_shared[3];
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int chunk_count = (seq_len + chunk_size - 1) / chunk_size;
  float acc = 0.0f;
  float running_m = neg_inf<float>();
  float running_l = 0.0f;

  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    if (tid == 0) {
      const int idx = head * scratch_chunks + chunk;
      const float chunk_m_value = chunk_m[idx];
      const float chunk_l_value = chunk_l[idx];
      const float new_m = fmaxf(running_m, chunk_m_value);
      const float alpha = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float beta = (chunk_l_value == 0.0f) ? 0.0f : expf(chunk_m_value - new_m);
      running_l = running_l * alpha + chunk_l_value * beta;
      running_m = new_m;
      scale_shared[0] = alpha;
      scale_shared[1] = beta;
      scale_shared[2] = running_l;
    }
    __syncthreads();

    const float alpha = scale_shared[0];
    const float beta = scale_shared[1];
    const std::size_t base = (static_cast<std::size_t>(head) * static_cast<std::size_t>(scratch_chunks) +
                              static_cast<std::size_t>(chunk)) *
                             static_cast<std::size_t>(head_dim);
    for (int d = tid; d < head_dim; d += blockDim.x) {
      acc = acc * alpha + chunk_o[base + static_cast<std::size_t>(d)] * beta;
    }
    __syncthreads();
  }

  const float inv_l = 1.0f / fmaxf(scale_shared[2], 1e-8f);
  for (int d = tid; d < head_dim; d += blockDim.x) {
    out[head * head_dim + d] = __float2half(acc * inv_l);
  }
}

// CUDA Graph-friendly helper kernels for KV-cache writes and device-side
// counters.
__global__ void store_kv_device_pos_kernel(const half* k,
                                           const half* v,
                                           half* k_cache,
                                           half* v_cache,
                                           const int* position_ptr,
                                           int kv_hidden,
                                           int max_context) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= kv_hidden) {
    return;
  }
  const int position = position_ptr[0];
  if (position < 0 || position >= max_context) {
    return;
  }
  const int offset = position * kv_hidden + idx;
  k_cache[offset] = k[idx];
  v_cache[offset] = v[idx];
}

// int4 moves eight fp16 values at a time, so this path keeps KV cache stores
// wide when all pointers are 16-byte aligned.
__global__ void store_kv_device_pos_vec8_kernel(const int4* k,
                                                const int4* v,
                                                int4* k_cache,
                                                int4* v_cache,
                                                const int* position_ptr,
                                                int kv_hidden_vec8,
                                                int max_context) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= kv_hidden_vec8) {
    return;
  }
  const int position = position_ptr[0];
  if (position < 0 || position >= max_context) {
    return;
  }
  const int offset = position * kv_hidden_vec8 + idx;
  k_cache[offset] = k[idx];
  v_cache[offset] = v[idx];
}

__global__ void copy_int_kernel(const int* src, int* dst) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    dst[0] = src[0];
  }
}

__global__ void increment_int_kernel(int* value) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    value[0] += 1;
  }
}

// GQA-fused decode kernel.
template <int HeadDim>
__global__ void gqa_decode_kernel(const half* __restrict__ q,
                                  const half* __restrict__ k_cache,
                                  const half* __restrict__ v_cache,
                                  half* __restrict__ out,
                                  int seq_len,
                                  int num_heads,
                                  int num_kv_heads,
                                  int group_size) {
  extern __shared__ unsigned char smem_bytes[];
  half*  q_sh    = reinterpret_cast<half*>(smem_bytes);
  float* score_sh = reinterpret_cast<float*>(q_sh + group_size * HeadDim);
  half*  kv_sh   = reinterpret_cast<half*>(score_sh + group_size);

  const int kv_head = blockIdx.x;
  const int tid     = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane    = tid & 31;
  const float scale = rsqrtf(static_cast<float>(HeadDim));

  // Each warp loads its own Q head slice; no cross-warp dependency on q_sh.
  const int q_head = kv_head * group_size + warp_id;
  for (int d = lane; d < HeadDim; d += 32) {
    q_sh[warp_id * HeadDim + d] = q[q_head * HeadDim + d];
  }

  float acc[HeadDim / 32];
  #pragma unroll
  for (int i = 0; i < HeadDim / 32; ++i) acc[i] = 0.0f;
  float running_m = -1.0e30f;
  float running_l = 0.0f;

  for (int t = 0; t < seq_len; ++t) {
    const int base = cache_index(t, kv_head, 0, num_kv_heads, HeadDim);

    // Phase 1: load K[t] into kv_sh (all threads cooperate).
    for (int d = tid; d < HeadDim; d += blockDim.x) {
      kv_sh[d] = k_cache[base + d];
    }
    __syncthreads();  // K ready; intra-warp Q reads guaranteed by program order

    // Phase 2: each warp computes Q[g] · K dot product → score_sh[warp_id].
    {
      const half2* q2 = reinterpret_cast<const half2*>(q_sh + warp_id * HeadDim);
      const half2* k2 = reinterpret_cast<const half2*>(kv_sh);
      float partial = 0.0f;
      for (int pair = lane; pair < HeadDim / 2; pair += 32) {
        const float2 qv = __half22float2(q2[pair]);
        const float2 kv = __half22float2(k2[pair]);
        partial += qv.x * kv.x + qv.y * kv.y;
      }
      const float score = warp_sum(partial) * scale;
      if (lane == 0) score_sh[warp_id] = score;
    }
    __syncthreads();  // scores written; K reads done — safe to overwrite kv_sh

    // Phase 3: load V[t] into kv_sh (reuse buffer).
    for (int d = tid; d < HeadDim; d += blockDim.x) {
      kv_sh[d] = v_cache[base + d];
    }
    __syncthreads();  // V ready; scores still valid in score_sh

    // Phase 4: online-softmax update and V accumulation (per warp, in regs).
    {
      const float score  = score_sh[warp_id];
      const float new_m  = fmaxf(running_m, score);
      const float c_prev = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float weight = expf(score - new_m);
      #pragma unroll
      for (int i = 0; i < HeadDim / 32; ++i) {
        acc[i] = acc[i] * c_prev + weight * __half2float(kv_sh[lane + i * 32]);
      }
      running_m = new_m;
      running_l = running_l * c_prev + weight;
    }
    __syncthreads();  // V reads done; kv_sh safe to reuse for next K
  }

  const float inv_l = 1.0f / fmaxf(running_l, 1.0e-8f);
  #pragma unroll
  for (int i = 0; i < HeadDim / 32; ++i) {
    out[q_head * HeadDim + lane + i * 32] = __float2half(acc[i] * inv_l);
  }
}

// Device-position variant for CUDA graph capture: seq_len read from device ptr.
template <int HeadDim>
__global__ void gqa_decode_kernel_device_pos(const half* __restrict__ q,
                                             const half* __restrict__ k_cache,
                                             const half* __restrict__ v_cache,
                                             half* __restrict__ out,
                                             const int* position_ptr,
                                             int num_heads,
                                             int num_kv_heads,
                                             int group_size) {
  const int seq_len = position_ptr[0] + 1;
  extern __shared__ unsigned char smem_bytes[];
  half*  q_sh    = reinterpret_cast<half*>(smem_bytes);
  float* score_sh = reinterpret_cast<float*>(q_sh + group_size * HeadDim);
  half*  kv_sh   = reinterpret_cast<half*>(score_sh + group_size);

  const int kv_head = blockIdx.x;
  const int tid     = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane    = tid & 31;
  const float scale = rsqrtf(static_cast<float>(HeadDim));

  const int q_head = kv_head * group_size + warp_id;
  for (int d = lane; d < HeadDim; d += 32) {
    q_sh[warp_id * HeadDim + d] = q[q_head * HeadDim + d];
  }

  float acc[HeadDim / 32];
  #pragma unroll
  for (int i = 0; i < HeadDim / 32; ++i) acc[i] = 0.0f;
  float running_m = -1.0e30f;
  float running_l = 0.0f;

  for (int t = 0; t < seq_len; ++t) {
    const int base = cache_index(t, kv_head, 0, num_kv_heads, HeadDim);

    for (int d = tid; d < HeadDim; d += blockDim.x) {
      kv_sh[d] = k_cache[base + d];
    }
    __syncthreads();

    {
      const half2* q2 = reinterpret_cast<const half2*>(q_sh + warp_id * HeadDim);
      const half2* k2 = reinterpret_cast<const half2*>(kv_sh);
      float partial = 0.0f;
      for (int pair = lane; pair < HeadDim / 2; pair += 32) {
        const float2 qv = __half22float2(q2[pair]);
        const float2 kv = __half22float2(k2[pair]);
        partial += qv.x * kv.x + qv.y * kv.y;
      }
      const float score = warp_sum(partial) * scale;
      if (lane == 0) score_sh[warp_id] = score;
    }
    __syncthreads();

    for (int d = tid; d < HeadDim; d += blockDim.x) {
      kv_sh[d] = v_cache[base + d];
    }
    __syncthreads();

    {
      const float score  = score_sh[warp_id];
      const float new_m  = fmaxf(running_m, score);
      const float c_prev = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float weight = expf(score - new_m);
      #pragma unroll
      for (int i = 0; i < HeadDim / 32; ++i) {
        acc[i] = acc[i] * c_prev + weight * __half2float(kv_sh[lane + i * 32]);
      }
      running_m = new_m;
      running_l = running_l * c_prev + weight;
    }
    __syncthreads();
  }

  const float inv_l = 1.0f / fmaxf(running_l, 1.0e-8f);
  #pragma unroll
  for (int i = 0; i < HeadDim / 32; ++i) {
    out[q_head * HeadDim + lane + i * 32] = __float2half(acc[i] * inv_l);
  }
}


}  // namespace

void launch_attention_step(const half* q,
                           const half* k_cache,
                           const half* v_cache,
                           half* out,
                           int seq_len,
                           int num_heads,
                           int num_kv_heads,
                           int head_dim,
                           cudaStream_t stream,
                           float* scratch_m,
                           float* scratch_l,
                           float* scratch_o,
                           int scratch_chunks,
                           bool allow_split) {
  // GQA fused: one block per KV head, group_size warps share K/V loads.
  // Gives group_size× KV-bandwidth reduction vs. per-Q-head kernels.
  // Dispatched before split-K so GQA models always take this path.
  if (num_kv_heads > 0 && num_heads > num_kv_heads &&
      (num_heads % num_kv_heads) == 0 && head_dim == 128) {
    const int group_size_val = num_heads / num_kv_heads;
    if (group_size_val <= 32) {
      const int threads_gqa = group_size_val * 32;
      const std::size_t smem_gqa =
          static_cast<std::size_t>(group_size_val + 1) * static_cast<std::size_t>(head_dim) * sizeof(half) +
          static_cast<std::size_t>(group_size_val) * sizeof(float);
      gqa_decode_kernel<128><<<num_kv_heads, threads_gqa, smem_gqa, stream>>>(
          q, k_cache, v_cache, out, seq_len, num_heads, num_kv_heads, group_size_val);
      return;
    }
  }

  constexpr int split_chunk_size = 32;
  if (allow_split && scratch_m && scratch_l && scratch_o && scratch_chunks > 0 && head_dim == 128 && seq_len >= 64) {
    constexpr int warps = 4;
    constexpr int threads = warps * 32;
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                             static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                             static_cast<std::size_t>(warps * head_dim) * sizeof(half);
    const int chunk_count = min(scratch_chunks, (seq_len + split_chunk_size - 1) / split_chunk_size);
    const dim3 grid(num_heads, chunk_count);
    attention_step_chunk_stats_kernel<warps><<<grid, threads, smem, stream>>>(
        q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, seq_len, num_heads, num_kv_heads, head_dim, split_chunk_size, scratch_chunks);
    attention_step_chunk_reduce_kernel<<<num_heads, threads, 0, stream>>>(
        scratch_m, scratch_l, scratch_o, out, seq_len, num_heads, head_dim, split_chunk_size, scratch_chunks);
    return;
  }

  if (head_dim > 0 && (head_dim % 2) == 0 && head_dim <= 256) {
    if (head_dim <= 64) {
      constexpr int warps = 2;
      constexpr int threads = warps * 32;
      const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                               static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                               static_cast<std::size_t>(warps * head_dim) * sizeof(half);
      attention_step_kernel_tiled<warps><<<num_heads, threads, smem, stream>>>(
          q,
          k_cache,
          v_cache,
          out,
          seq_len,
          num_heads,
          num_kv_heads,
          head_dim);
      return;
    }

    constexpr int warps = 4;
    constexpr int threads = warps * 32;
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                             static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                             static_cast<std::size_t>(warps * head_dim) * sizeof(half);
    attention_step_kernel_tiled<warps><<<num_heads, threads, smem, stream>>>(
        q,
        k_cache,
        v_cache,
        out,
        seq_len,
        num_heads,
        num_kv_heads,
        head_dim);
    return;
  }

  constexpr int threads = 128;
  const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                           static_cast<std::size_t>(threads + 3) * sizeof(float);
  attention_step_kernel_fallback<<<num_heads, threads, smem, stream>>>(
      q,
      k_cache,
      v_cache,
      out,
      seq_len,
      num_heads,
      num_kv_heads,
      head_dim);
}

void launch_attention_step_device_pos(const half* q,
                                      const half* k_cache,
                                      const half* v_cache,
                                      half* out,
                                      const int* position,
                                      int num_heads,
                                      int num_kv_heads,
                                      int head_dim,
                                      cudaStream_t stream,
                                      float* scratch_m,
                                      float* scratch_l,
                                      float* scratch_o,
                                      int scratch_chunks,
                                      bool allow_split) {
  // GQA fused dispatch (device-position variant for CUDA graph capture).
  if (num_kv_heads > 0 && num_heads > num_kv_heads &&
      (num_heads % num_kv_heads) == 0 && head_dim == 128) {
    const int group_size_val = num_heads / num_kv_heads;
    if (group_size_val <= 32) {
      const int threads_gqa = group_size_val * 32;
      const std::size_t smem_gqa =
          static_cast<std::size_t>(group_size_val + 1) * static_cast<std::size_t>(head_dim) * sizeof(half) +
          static_cast<std::size_t>(group_size_val) * sizeof(float);
      gqa_decode_kernel_device_pos<128><<<num_kv_heads, threads_gqa, smem_gqa, stream>>>(
          q, k_cache, v_cache, out, position, num_heads, num_kv_heads, group_size_val);
      return;
    }
  }

  constexpr int split_chunk_size = 32;
  if (allow_split && scratch_m && scratch_l && scratch_o && scratch_chunks > 0 && head_dim == 128) {
    constexpr int warps = 4;
    constexpr int threads = warps * 32;
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                             static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                             static_cast<std::size_t>(warps * head_dim) * sizeof(half);
    const dim3 grid(num_heads, scratch_chunks);
    attention_step_chunk_stats_device_pos_kernel<warps><<<grid, threads, smem, stream>>>(
        q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, position, num_heads, num_kv_heads, head_dim, split_chunk_size, scratch_chunks);
    attention_step_chunk_reduce_device_pos_kernel<<<num_heads, threads, 0, stream>>>(
        scratch_m, scratch_l, scratch_o, out, position, num_heads, head_dim, split_chunk_size, scratch_chunks);
    return;
  }

  if (head_dim > 0 && (head_dim % 2) == 0 && head_dim <= 256) {
    if (head_dim <= 64) {
      constexpr int warps = 2;
      constexpr int threads = warps * 32;
      const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                               static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                               static_cast<std::size_t>(warps * head_dim) * sizeof(half);
      attention_step_kernel_tiled_device_pos<warps><<<num_heads, threads, smem, stream>>>(
          q, k_cache, v_cache, out, position, num_heads, num_kv_heads, head_dim);
      return;
    }

    constexpr int warps = 4;
    constexpr int threads = warps * 32;
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                             static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                             static_cast<std::size_t>(warps * head_dim) * sizeof(half);
    attention_step_kernel_tiled_device_pos<warps><<<num_heads, threads, smem, stream>>>(
        q, k_cache, v_cache, out, position, num_heads, num_kv_heads, head_dim);
    return;
  }

  constexpr int threads = 128;
  const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                           static_cast<std::size_t>(threads + 3) * sizeof(float);
  attention_step_kernel_fallback_device_pos<<<num_heads, threads, smem, stream>>>(
      q, k_cache, v_cache, out, position, num_heads, num_kv_heads, head_dim);
}

void launch_store_kv_device_pos(const half* k,
                                const half* v,
                                half* k_cache,
                                half* v_cache,
                                const int* position,
                                int kv_hidden,
                                int max_context,
                                cudaStream_t stream) {
  const bool aligned = ((reinterpret_cast<std::uintptr_t>(k) | reinterpret_cast<std::uintptr_t>(v) |
                         reinterpret_cast<std::uintptr_t>(k_cache) | reinterpret_cast<std::uintptr_t>(v_cache)) &
                        (alignof(int4) - 1)) == 0;
  if ((kv_hidden & 7) == 0 && aligned) {
    const int kv_hidden_vec8 = kv_hidden / 8;
    const int threads = choose_copy_threads(kv_hidden);
    const int blocks = (kv_hidden_vec8 + threads - 1) / threads;
    store_kv_device_pos_vec8_kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const int4*>(k),
                                                                    reinterpret_cast<const int4*>(v),
                                                                    reinterpret_cast<int4*>(k_cache),
                                                                    reinterpret_cast<int4*>(v_cache),
                                                                    position,
                                                                    kv_hidden_vec8,
                                                                    max_context);
    return;
  }

  constexpr int threads = 256;
  const int blocks = (kv_hidden + threads - 1) / threads;
  store_kv_device_pos_kernel<<<blocks, threads, 0, stream>>>(k, v, k_cache, v_cache, position, kv_hidden, max_context);
}

void launch_copy_int(const int* src, int* dst, cudaStream_t stream) {
  copy_int_kernel<<<1, 1, 0, stream>>>(src, dst);
}

void launch_increment_int(int* value, cudaStream_t stream) {
  increment_int_kernel<<<1, 1, 0, stream>>>(value);
}


}  // namespace kernels
