// kernels_attention_decode.cu
//
// CUDA kernels and host launch wrappers for FP16 decode-time attention paths
// plus device-position KV-cache updates.

#include <cuda_fp16.h>
#include <sm_61_intrinsics.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>

#include "runtime/kernels.cuh"

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
// kTiledMaxHeadDim / kAccPerThread are shared across decode-attention TUs; see
// include/runtime/kernels.cuh.

// Shared serial per-token online-softmax fallback core (one thread per head_dim
// element). seq_len is resolved by the caller so the host and CUDA-graph
// device-position kernels share this one body.
__device__ __forceinline__ void attention_step_fallback_core(const half* q, const half* k_cache,
                                                             const half* v_cache, half* out,
                                                             int seq_len, int num_heads,
                                                             int num_kv_heads, int head_dim) {
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
  const int kv_head =
      ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
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

// Host fallback kernel: seq_len is a runtime argument.
__global__ void attention_step_kernel_fallback(const half* q, const half* k_cache,
                                               const half* v_cache, half* out, int seq_len,
                                               int num_heads, int num_kv_heads, int head_dim) {
  attention_step_fallback_core(q, k_cache, v_cache, out, seq_len, num_heads, num_kv_heads,
                               head_dim);
}

// Device-position variant for CUDA-graph capture: seq_len is read from device
// memory (position_ptr[0]+1). The body is shared via attention_step_fallback_core.
__global__ void attention_step_kernel_fallback_device_pos(const half* q, const half* k_cache,
                                                          const half* v_cache, half* out,
                                                          const int* position_ptr, int num_heads,
                                                          int num_kv_heads, int head_dim) {
  attention_step_fallback_core(q, k_cache, v_cache, out, position_ptr[0] + 1, num_heads,
                               num_kv_heads, head_dim);
}

// Shared online-softmax tiled decode-attention core. Keeps the query resident in
// shared memory, stages one value tile at a time, and merges tile-local softmax
// statistics so the block avoids a fully serial per-token expf chain. seq_len is
// resolved by the caller, so the host kernel and the CUDA-graph device-position
// kernel share this one body.
template <int WarpsPerBlock>
__device__ __forceinline__ void attention_step_tiled_core(const half* q, const half* k_cache,
                                                          const half* v_cache, half* out,
                                                          int seq_len, int num_heads,
                                                          int num_kv_heads, int head_dim) {
  // Shared-memory layout:
  //   half  q_shared[head_dim]
  //   float score_shared[WarpsPerBlock]
  //   float beta_shared[WarpsPerBlock]   // exp(score[i] - tile_m)
  //   float stats_shared[4]              // [running_m, running_l, tile_m, tile_l]
  //   half  v_tile[WarpsPerBlock * head_dim]
  extern __shared__ unsigned char smem_bytes[];
  half* q_shared = reinterpret_cast<half*>(smem_bytes);
  float* score_shared = reinterpret_cast<float*>(q_shared + head_dim);
  float* beta_shared = score_shared + WarpsPerBlock;
  float* stats_shared = beta_shared + WarpsPerBlock;  // [running_m, running_l, tile_m, tile_l]
  half* v_tile = reinterpret_cast<half*>(stats_shared + 4);

  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane = tid % warpSize;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head =
      ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const int head_pairs = head_dim / 2;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[head * head_dim + d];
  }
  if (tid == 0) {
    stats_shared[0] = -1.0e30f;  // running_m
    stats_shared[1] = 0.0f;      // running_l
  }
  __syncthreads();

  // One output accumulator per head_dim element this thread owns. head_dim can
  // exceed blockDim (e.g. head_dim=256 with 128 threads → 2 elements/thread), so
  // a scalar acc would conflate the strided output dims. The static_assert ties
  // the accumulator size to the dispatch cap so raising kTiledMaxHeadDim without
  // growing kAccPerThread fails the build instead of silently corrupting output.
  static_assert(WarpsPerBlock * 32 * kAccPerThread >= kTiledMaxHeadDim,
                "tiled decode accumulator too small for kTiledMaxHeadDim");
  float acc[kAccPerThread];
#pragma unroll
  for (int j = 0; j < kAccPerThread; ++j) acc[j] = 0.0f;
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
      const float tile_m = stats_shared[2];
      const float tile_l = stats_shared[3];
      const float running_m = stats_shared[0];
      const float running_l = stats_shared[1];
      const float new_m = fmaxf(running_m, tile_m);
      const float c_prev = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float c_tile = expf(tile_m - new_m);

      int j = 0;
      for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
        float tile_o = 0.0f;
        for (int i = 0; i < tile_tokens; ++i) {
          tile_o += beta_shared[i] * __half2float(v_tile[i * head_dim + d]);
        }
        acc[j] = acc[j] * c_prev + tile_o * c_tile;
      }
      if (tid == 0) {
        stats_shared[0] = new_m;
        stats_shared[1] = running_l * c_prev + tile_l * c_tile;
      }
    }
    __syncthreads();
  }

  const float inv_l = 1.0f / fmaxf(stats_shared[1], 1e-8f);
  int j = 0;
  for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
    out[head * head_dim + d] = __float2half(acc[j] * inv_l);
  }
}

// Host tiled decode kernel: seq_len is a runtime argument.
template <int WarpsPerBlock>
__global__ void attention_step_kernel_tiled(const half* q, const half* k_cache, const half* v_cache,
                                            half* out, int seq_len, int num_heads, int num_kv_heads,
                                            int head_dim) {
  attention_step_tiled_core<WarpsPerBlock>(q, k_cache, v_cache, out, seq_len, num_heads,
                                           num_kv_heads, head_dim);
}

// Device-position variant for CUDA-graph capture: seq_len is read from device
// memory (position_ptr[0]+1) so the launch stays graph-safe as the sequence
// grows. The body is shared via attention_step_tiled_core.
template <int WarpsPerBlock>
__global__ void attention_step_kernel_tiled_device_pos(const half* q, const half* k_cache,
                                                       const half* v_cache, half* out,
                                                       const int* position_ptr, int num_heads,
                                                       int num_kv_heads, int head_dim) {
  attention_step_tiled_core<WarpsPerBlock>(q, k_cache, v_cache, out, position_ptr[0] + 1, num_heads,
                                           num_kv_heads, head_dim);
}

// Shared split-K decode pass-1 core: each block computes softmax statistics and
// an unnormalized partial output for one [head, chunk] pair. seq_len is resolved
// by the caller, so the host and CUDA-graph device-position kernels share this
// one body.
template <int WarpsPerBlock>
__device__ __forceinline__ void attention_step_chunk_stats_core(
    const half* q, const half* k_cache, const half* v_cache, float* chunk_m, float* chunk_l,
    float* chunk_o, int seq_len, int num_heads, int num_kv_heads, int head_dim, int chunk_size,
    int scratch_chunks) {
  extern __shared__ unsigned char smem_bytes[];
  half* q_shared = reinterpret_cast<half*>(smem_bytes);
  float* score_shared = reinterpret_cast<float*>(q_shared + head_dim);
  float* beta_shared = score_shared + WarpsPerBlock;
  float* stats_shared = beta_shared + WarpsPerBlock;  // [running_m, running_l, tile_m, tile_l]
  half* v_tile = reinterpret_cast<half*>(stats_shared + 4);

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
  const int kv_head =
      ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
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

  float acc[kAccPerThread];
#pragma unroll
  for (int jj = 0; jj < kAccPerThread; ++jj) acc[jj] = 0.0f;
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
      const float tile_m = stats_shared[2];
      const float tile_l = stats_shared[3];
      const float running_m = stats_shared[0];
      const float running_l = stats_shared[1];
      const float new_m = fmaxf(running_m, tile_m);
      const float c_prev = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float c_tile = expf(tile_m - new_m);

      int j = 0;
      for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
        float tile_o = 0.0f;
        for (int i = 0; i < tile_tokens; ++i) {
          tile_o += beta_shared[i] * __half2float(v_tile[i * head_dim + d]);
        }
        acc[j] = acc[j] * c_prev + tile_o * c_tile;
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
  int jo = 0;
  for (int d = tid; d < head_dim; d += blockDim.x, ++jo) {
    chunk_o[static_cast<std::size_t>(chunk_index) * static_cast<std::size_t>(head_dim) +
            static_cast<std::size_t>(d)] = acc[jo];
  }
}

// Host split-K pass-1 kernel: seq_len is a runtime argument.
template <int WarpsPerBlock>
__global__ void attention_step_chunk_stats_kernel(const half* q, const half* k_cache,
                                                  const half* v_cache, float* chunk_m,
                                                  float* chunk_l, float* chunk_o, int seq_len,
                                                  int num_heads, int num_kv_heads, int head_dim,
                                                  int chunk_size, int scratch_chunks) {
  attention_step_chunk_stats_core<WarpsPerBlock>(q, k_cache, v_cache, chunk_m, chunk_l, chunk_o,
                                                 seq_len, num_heads, num_kv_heads, head_dim,
                                                 chunk_size, scratch_chunks);
}

// Device-position variant for CUDA-graph capture: seq_len is read from device
// memory (position_ptr[0]+1). The body is shared via attention_step_chunk_stats_core.
template <int WarpsPerBlock>
__global__ void attention_step_chunk_stats_device_pos_kernel(const half* q, const half* k_cache,
                                                             const half* v_cache, float* chunk_m,
                                                             float* chunk_l, float* chunk_o,
                                                             const int* position_ptr, int num_heads,
                                                             int num_kv_heads, int head_dim,
                                                             int chunk_size, int scratch_chunks) {
  attention_step_chunk_stats_core<WarpsPerBlock>(q, k_cache, v_cache, chunk_m, chunk_l, chunk_o,
                                                 position_ptr[0] + 1, num_heads, num_kv_heads,
                                                 head_dim, chunk_size, scratch_chunks);
}

// Split-K decode, pass 2: merge the chunk-local softmax statistics and
// partial outputs into the final normalized attention result.
__global__ void attention_step_chunk_reduce_kernel(const float* chunk_m, const float* chunk_l,
                                                   const float* chunk_o, half* out, int seq_len,
                                                   int num_heads, int head_dim, int chunk_size,
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
    const std::size_t base =
        (static_cast<std::size_t>(head) * static_cast<std::size_t>(scratch_chunks) +
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

// Split-K decode pass-1, PAGED (P3 phase 2b): identical math to
// attention_step_chunk_stats_kernel, but K/V for logical chunk `chunk` live in
// physical block `block_table[chunk]` of a block pool (block_size == chunk_size).
// Only the token->address mapping changes: token t reads pool token
// (block_table[chunk]*chunk_size + (t - chunk_start)). The pool is laid out like
// a flat cache of (num_blocks*chunk_size) tokens, so cache_index() is reused on
// the remapped token index. Non-contiguous physical blocks work; the pass-2
// reduce kernel is unchanged (it only touches the scratch stats).
template <int WarpsPerBlock>
__global__ void attention_step_chunk_stats_paged_kernel(
    const half* q, const half* k_pool, const half* v_pool, const int* __restrict__ block_table,
    float* chunk_m, float* chunk_l, float* chunk_o, int seq_len, int num_heads, int num_kv_heads,
    int head_dim, int chunk_size, int scratch_chunks) {
  extern __shared__ unsigned char smem_bytes[];
  half* q_shared = reinterpret_cast<half*>(smem_bytes);
  float* score_shared = reinterpret_cast<float*>(q_shared + head_dim);
  float* beta_shared = score_shared + WarpsPerBlock;
  float* stats_shared = beta_shared + WarpsPerBlock;
  half* v_tile = reinterpret_cast<half*>(stats_shared + 4);

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
  const int kv_head =
      ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const int head_pairs = head_dim / 2;
  const int chunk_index = head * scratch_chunks + chunk;
  // Map a logical token in this chunk to its physical pool token. Every token in
  // the chunk lives in physical block block_table[chunk]; phys_token =
  // block_table[chunk]*chunk_size + (t - chunk_start) = (t + phys_base).
  const int phys_base = block_table[chunk] * chunk_size - chunk_start;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[head * head_dim + d];
  }
  if (tid == 0) {
    stats_shared[0] = neg_inf<float>();
    stats_shared[1] = 0.0f;
  }
  __syncthreads();

  float acc[kAccPerThread];
#pragma unroll
  for (int jj = 0; jj < kAccPerThread; ++jj) acc[jj] = 0.0f;
  for (int tile_base = chunk_start; tile_base < chunk_end; tile_base += WarpsPerBlock) {
    const int tile_tokens = min(WarpsPerBlock, chunk_end - tile_base);

    {
      const int t = tile_base + warp_id;
      float score = neg_inf<float>();
      if (warp_id < tile_tokens) {
        const int base = cache_index(phys_base + t, kv_head, 0, num_kv_heads, head_dim);
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
      if (lane == 0 && warp_id < tile_tokens) {
        score_shared[warp_id] = score;
      }
    }

    {
      for (int i = 0; i < tile_tokens; ++i) {
        const int base = cache_index(phys_base + tile_base + i, kv_head, 0, num_kv_heads, head_dim);
        for (int d = tid; d < head_dim; d += blockDim.x) {
          v_tile[i * head_dim + d] = v_pool[base + d];
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
      const float tile_m = stats_shared[2];
      const float tile_l = stats_shared[3];
      const float running_m = stats_shared[0];
      const float running_l = stats_shared[1];
      const float new_m = fmaxf(running_m, tile_m);
      const float c_prev = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float c_tile = expf(tile_m - new_m);

      int j = 0;
      for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
        float tile_o = 0.0f;
        for (int i = 0; i < tile_tokens; ++i) {
          tile_o += beta_shared[i] * __half2float(v_tile[i * head_dim + d]);
        }
        acc[j] = acc[j] * c_prev + tile_o * c_tile;
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
  int jo = 0;
  for (int d = tid; d < head_dim; d += blockDim.x, ++jo) {
    chunk_o[static_cast<std::size_t>(chunk_index) * static_cast<std::size_t>(head_dim) +
            static_cast<std::size_t>(d)] = acc[jo];
  }
}

__global__ void attention_step_chunk_reduce_device_pos_kernel(
    const float* chunk_m, const float* chunk_l, const float* chunk_o, half* out,
    const int* position_ptr, int num_heads, int head_dim, int chunk_size, int scratch_chunks) {
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
    const std::size_t base =
        (static_cast<std::size_t>(head) * static_cast<std::size_t>(scratch_chunks) +
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
__global__ void store_kv_device_pos_kernel(const half* k, const half* v, half* k_cache,
                                           half* v_cache, const int* position_ptr, int kv_hidden,
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
__global__ void store_kv_device_pos_vec8_kernel(const int4* k, const int4* v, int4* k_cache,
                                                int4* v_cache, const int* position_ptr,
                                                int kv_hidden_vec8, int max_context) {
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
__global__ void gqa_decode_kernel(const half* __restrict__ q, const half* __restrict__ k_cache,
                                  const half* __restrict__ v_cache, half* __restrict__ out,
                                  int seq_len, int num_heads, int num_kv_heads, int group_size) {
  extern __shared__ unsigned char smem_bytes[];
  half* q_sh = reinterpret_cast<half*>(smem_bytes);
  float* score_sh = reinterpret_cast<float*>(q_sh + group_size * HeadDim);
  half* kv_sh = reinterpret_cast<half*>(score_sh + group_size);

  const int kv_head = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid & 31;
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
      const float score = score_sh[warp_id];
      const float new_m = fmaxf(running_m, score);
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
                                             half* __restrict__ out, const int* position_ptr,
                                             int num_heads, int num_kv_heads, int group_size) {
  const int seq_len = position_ptr[0] + 1;
  extern __shared__ unsigned char smem_bytes[];
  half* q_sh = reinterpret_cast<half*>(smem_bytes);
  float* score_sh = reinterpret_cast<float*>(q_sh + group_size * HeadDim);
  half* kv_sh = reinterpret_cast<half*>(score_sh + group_size);

  const int kv_head = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid & 31;
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
      const float score = score_sh[warp_id];
      const float new_m = fmaxf(running_m, score);
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

// Sequence length at/below which the GQA-fused decode kernel (num_kv_heads
// blocks, serial over the sequence) is used. Above it, decode attention routes
// to the split-K path (parallel over KV chunks) — see launch_attention_step.
constexpr int kGqaFusedMaxSeq = 256;

// ── GQA + split-K decode attention (best of both) ────────────────────────────
// The per-Q-head split-K kernel reads each K/V token once per QUERY head, so a
// GQA group of `group_size` query heads re-reads the same KV `group_size` times
// from HBM. The GQA-fused kernel avoids that but launches only num_kv_heads
// blocks (poor occupancy at long context). This kernel combines both: grid.x =
// num_kv_heads (one block per KV head, `group_size` warps = one per query head),
// grid.y = KV chunk. Each K/V token is loaded to shared ONCE per group per chunk
// and reused across the group's warps, while num_kv_heads×chunks blocks keep the
// GPU busy. It writes per-Q-head chunk partials in the exact layout the existing
// attention_step_chunk_reduce* kernels consume, so pass 2 is unchanged.
template <int HeadDim>
__device__ __forceinline__ void gqa_split_chunk_stats_core(const half* q, const half* k_cache,
                                                           const half* v_cache, float* chunk_m,
                                                           float* chunk_l, float* chunk_o,
                                                           int seq_len, int num_kv_heads,
                                                           int group_size, int chunk_size,
                                                           int scratch_chunks) {
  // chunk_size is the split step (<= warpSize=32), so lane `t` owns token `t`.
  const int kv_head = blockIdx.x;
  const int chunk = blockIdx.y;
  const int chunk_start = chunk * chunk_size;
  if (chunk_start >= seq_len) return;
  const int chunk_end = min(chunk_start + chunk_size, seq_len);
  const int tile_tokens = chunk_end - chunk_start;

  extern __shared__ unsigned char smem_bytes[];
  half* q_sh = reinterpret_cast<half*>(smem_bytes);                       // group_size*HeadDim
  half* k_tile = q_sh + group_size * HeadDim;                             // chunk_size*HeadDim
  half* v_tile = k_tile + chunk_size * HeadDim;                           // chunk_size*HeadDim
  float* w_sh = reinterpret_cast<float*>(v_tile + chunk_size * HeadDim);  // group_size*chunk_size

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid & 31;
  const float scale = rsqrtf(static_cast<float>(HeadDim));
  const int q_head = kv_head * group_size + warp_id;

  // Load this group's query heads and the chunk's K/V tile once (coalesced),
  // reused across all group_size warps.
  for (int d = lane; d < HeadDim; d += 32) {
    q_sh[warp_id * HeadDim + d] = q[q_head * HeadDim + d];
  }
  for (int idx = tid; idx < tile_tokens * HeadDim; idx += blockDim.x) {
    const int t = idx / HeadDim;
    const int d = idx - t * HeadDim;
    const int base = cache_index(chunk_start + t, kv_head, 0, num_kv_heads, HeadDim);
    k_tile[idx] = k_cache[base + d];
    v_tile[idx] = v_cache[base + d];
  }
  __syncthreads();

  // Each warp handles one query head; lane t scores token t against shared K.
  float score = neg_inf<float>();
  if (lane < tile_tokens) {
    const half2* qh = reinterpret_cast<const half2*>(q_sh + warp_id * HeadDim);
    const half2* kt = reinterpret_cast<const half2*>(k_tile + lane * HeadDim);
    float dot = 0.0f;
#pragma unroll
    for (int p = 0; p < HeadDim / 2; ++p) {
      const float2 a = __half22float2(qh[p]);
      const float2 b = __half22float2(kt[p]);
      dot += a.x * b.x + a.y * b.y;
    }
    score = dot * scale;
  }
  float tile_m = score;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) tile_m = fmaxf(tile_m, __shfl_xor_sync(0xffffffffu, tile_m, o));
  const float weight = (lane < tile_tokens) ? expf(score - tile_m) : 0.0f;
  float tile_l = weight;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) tile_l += __shfl_xor_sync(0xffffffffu, tile_l, o);
  w_sh[warp_id * chunk_size + lane] = weight;
  __syncwarp();

  // Weighted V sum: each lane owns HeadDim/32 output channels.
  const int chunk_index = q_head * scratch_chunks + chunk;
#pragma unroll
  for (int i = 0; i < HeadDim / 32; ++i) {
    const int d = lane + i * 32;
    float o = 0.0f;
    for (int t = 0; t < tile_tokens; ++t) {
      o += w_sh[warp_id * chunk_size + t] * __half2float(v_tile[t * HeadDim + d]);
    }
    chunk_o[static_cast<std::size_t>(chunk_index) * HeadDim + d] = o;
  }
  if (lane == 0) {
    chunk_m[chunk_index] = tile_m;
    chunk_l[chunk_index] = tile_l;
  }
}

template <int HeadDim>
__global__ void attention_step_chunk_stats_gqa_split_kernel(const half* q, const half* k_cache,
                                                            const half* v_cache, float* chunk_m,
                                                            float* chunk_l, float* chunk_o,
                                                            int seq_len, int num_kv_heads,
                                                            int group_size, int chunk_size,
                                                            int scratch_chunks) {
  gqa_split_chunk_stats_core<HeadDim>(q, k_cache, v_cache, chunk_m, chunk_l, chunk_o, seq_len,
                                      num_kv_heads, group_size, chunk_size, scratch_chunks);
}

template <int HeadDim>
__global__ void attention_step_chunk_stats_gqa_split_device_pos_kernel(
    const half* q, const half* k_cache, const half* v_cache, float* chunk_m, float* chunk_l,
    float* chunk_o, const int* position_ptr, int num_kv_heads, int group_size, int chunk_size,
    int scratch_chunks) {
  gqa_split_chunk_stats_core<HeadDim>(q, k_cache, v_cache, chunk_m, chunk_l, chunk_o,
                                      position_ptr[0] + 1, num_kv_heads, group_size, chunk_size,
                                      scratch_chunks);
}

void launch_attention_step(const half* q, const half* k_cache, const half* v_cache, half* out,
                           int seq_len, int num_heads, int num_kv_heads, int head_dim,
                           cudaStream_t stream, float* scratch_m, float* scratch_l,
                           float* scratch_o, int scratch_chunks, bool allow_split) {
  const bool force_fallback = std::getenv("LLAMA_INFER_FORCE_FALLBACK_DECODE_ATTENTION") != nullptr;
  if (force_fallback) {
    constexpr int threads = 128;
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                             static_cast<std::size_t>(threads + 3) * sizeof(float);
    attention_step_kernel_fallback<<<num_heads, threads, smem, stream>>>(
        q, k_cache, v_cache, out, seq_len, num_heads, num_kv_heads, head_dim);
    return;
  }
  // GQA fused: one block per KV head, group_size warps share K/V loads.
  // Gives group_size× KV-bandwidth reduction vs. per-Q-head kernels — but it
  // launches only num_kv_heads blocks and scans the sequence serially per block,
  // so it is bandwidth-optimal at SHORT context yet badly under-parallelized at
  // long context (e.g. 4 blocks on a 170-SM GPU at 2k+ tokens dominated decode).
  // Above this length, fall through to the split-K path (parallel over KV
  // chunks, ~num_heads×ceil(seq/32) blocks).
  if (num_kv_heads > 0 && num_heads > num_kv_heads && (num_heads % num_kv_heads) == 0 &&
      head_dim == 128 && seq_len <= kGqaFusedMaxSeq) {
    const int group_size_val = num_heads / num_kv_heads;
    if (group_size_val <= 32) {
      const int threads_gqa = group_size_val * 32;
      const std::size_t smem_gqa = static_cast<std::size_t>(group_size_val + 1) *
                                       static_cast<std::size_t>(head_dim) * sizeof(half) +
                                   static_cast<std::size_t>(group_size_val) * sizeof(float);
      gqa_decode_kernel<128><<<num_kv_heads, threads_gqa, smem_gqa, stream>>>(
          q, k_cache, v_cache, out, seq_len, num_heads, num_kv_heads, group_size_val);
      return;
    }
  }

  constexpr int split_chunk_size = 32;
  if (allow_split && scratch_m && scratch_l && scratch_o && scratch_chunks > 0 && head_dim == 128 &&
      seq_len >= 64) {
    constexpr int warps = 4;
    constexpr int threads = warps * 32;
    const int chunk_count =
        min(scratch_chunks, (seq_len + split_chunk_size - 1) / split_chunk_size);
    // GQA models: share each K/V read across the query-head group (no per-head
    // redundancy) while keeping num_kv_heads×chunks blocks for occupancy.
    const bool gqa = num_kv_heads > 0 && num_heads > num_kv_heads &&
                     (num_heads % num_kv_heads) == 0 && (num_heads / num_kv_heads) * 32 <= 1024;
    if (gqa) {
      const int group_size_val = num_heads / num_kv_heads;
      const std::size_t smem_gqa =
          static_cast<std::size_t>(group_size_val + 2 * split_chunk_size) * head_dim *
              sizeof(half) +
          static_cast<std::size_t>(group_size_val) * split_chunk_size * sizeof(float);
      const dim3 grid(num_kv_heads, chunk_count);
      attention_step_chunk_stats_gqa_split_kernel<128>
          <<<grid, group_size_val * 32, smem_gqa, stream>>>(
              q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, seq_len, num_kv_heads,
              group_size_val, split_chunk_size, scratch_chunks);
    } else {
      const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                               static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                               static_cast<std::size_t>(warps * head_dim) * sizeof(half);
      const dim3 grid(num_heads, chunk_count);
      attention_step_chunk_stats_kernel<warps><<<grid, threads, smem, stream>>>(
          q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, seq_len, num_heads, num_kv_heads,
          head_dim, split_chunk_size, scratch_chunks);
    }
    attention_step_chunk_reduce_kernel<<<num_heads, threads, 0, stream>>>(
        scratch_m, scratch_l, scratch_o, out, seq_len, num_heads, head_dim, split_chunk_size,
        scratch_chunks);
    return;
  }

  if (head_dim > 0 && (head_dim % 2) == 0 && head_dim <= kTiledMaxHeadDim) {
    if (head_dim <= 64) {
      constexpr int warps = 2;
      constexpr int threads = warps * 32;
      const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                               static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                               static_cast<std::size_t>(warps * head_dim) * sizeof(half);
      attention_step_kernel_tiled<warps><<<num_heads, threads, smem, stream>>>(
          q, k_cache, v_cache, out, seq_len, num_heads, num_kv_heads, head_dim);
      return;
    }

    constexpr int warps = 4;
    constexpr int threads = warps * 32;
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                             static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                             static_cast<std::size_t>(warps * head_dim) * sizeof(half);
    attention_step_kernel_tiled<warps><<<num_heads, threads, smem, stream>>>(
        q, k_cache, v_cache, out, seq_len, num_heads, num_kv_heads, head_dim);
    return;
  }

  constexpr int threads = 128;
  const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                           static_cast<std::size_t>(threads + 3) * sizeof(float);
  attention_step_kernel_fallback<<<num_heads, threads, smem, stream>>>(
      q, k_cache, v_cache, out, seq_len, num_heads, num_kv_heads, head_dim);
}

// Paged split-K decode attention (P3 phase 2b). K/V live in a block pool; the
// per-sequence block_table (device, one physical block id per logical chunk,
// chunk_size == block_size) selects each chunk's physical block. Same online-
// softmax math as launch_attention_step's split-K path, reusing its pass-2
// reduce (scratch-only). Requires head_dim==128 + valid split-K scratch, which
// is the paged decode target on the Qwen/Llama family.
void launch_attention_step_paged(const half* q, const half* k_pool, const half* v_pool,
                                 const int* block_table, half* out, int seq_len, int num_heads,
                                 int num_kv_heads, int head_dim, int block_size,
                                 cudaStream_t stream, float* scratch_m, float* scratch_l,
                                 float* scratch_o, int scratch_chunks) {
  constexpr int warps = 4;
  constexpr int threads = warps * 32;
  const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                           static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                           static_cast<std::size_t>(warps * head_dim) * sizeof(half);
  const int chunk_count = min(scratch_chunks, (seq_len + block_size - 1) / block_size);
  const dim3 grid(num_heads, chunk_count);
  attention_step_chunk_stats_paged_kernel<warps><<<grid, threads, smem, stream>>>(
      q, k_pool, v_pool, block_table, scratch_m, scratch_l, scratch_o, seq_len, num_heads,
      num_kv_heads, head_dim, block_size, scratch_chunks);
  attention_step_chunk_reduce_kernel<<<num_heads, threads, 0, stream>>>(
      scratch_m, scratch_l, scratch_o, out, seq_len, num_heads, head_dim, block_size,
      scratch_chunks);
}

// ── Batched paged decode attention (P2 primitive) ────────────────────────────
// One decode step for a BATCH of sequences in a single launch. Sequence b (=
// blockIdx.z) has its own block table (block_tables + b*max_blocks), its own
// length (seq_lens[b]), and its own q/out/scratch slice. Same paged online-
// softmax math as the single-sequence kernel; this is the compute primitive for
// continuous batching. q: [batch][num_heads][head_dim]; scratch/out likewise
// batched. All sequences share the one KV block pool.
template <int WarpsPerBlock>
__global__ void attention_step_chunk_stats_batched_kernel(
    const half* q, const half* k_pool, const half* v_pool, const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens, int max_blocks, float* chunk_m, float* chunk_l,
    float* chunk_o, int num_heads, int num_kv_heads, int head_dim, int chunk_size,
    int scratch_chunks) {
  const int b = blockIdx.z;
  const int seq_len = seq_lens[b];
  const int chunk = blockIdx.y;
  const int chunk_start = chunk * chunk_size;
  if (chunk_start >= seq_len) return;

  extern __shared__ unsigned char smem_bytes[];
  half* q_shared = reinterpret_cast<half*>(smem_bytes);
  float* score_shared = reinterpret_cast<float*>(q_shared + head_dim);
  float* beta_shared = score_shared + WarpsPerBlock;
  float* stats_shared = beta_shared + WarpsPerBlock;
  half* v_tile = reinterpret_cast<half*>(stats_shared + 4);

  const int head = blockIdx.x;
  const int chunk_end = min(chunk_start + chunk_size, seq_len);
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane = tid % warpSize;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head =
      ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const int head_pairs = head_dim / 2;
  // Per-sequence slices.
  const int* block_table = block_tables + static_cast<std::size_t>(b) * max_blocks;
  const half* q_seq = q + (static_cast<std::size_t>(b) * num_heads + head) * head_dim;
  const int chunk_index = (b * num_heads + head) * scratch_chunks + chunk;
  const int phys_base = block_table[chunk] * chunk_size - chunk_start;

  for (int d = tid; d < head_dim; d += blockDim.x) q_shared[d] = q_seq[d];
  if (tid == 0) {
    stats_shared[0] = neg_inf<float>();
    stats_shared[1] = 0.0f;
  }
  __syncthreads();

  float acc[kAccPerThread];
#pragma unroll
  for (int jj = 0; jj < kAccPerThread; ++jj) acc[jj] = 0.0f;
  for (int tile_base = chunk_start; tile_base < chunk_end; tile_base += WarpsPerBlock) {
    const int tile_tokens = min(WarpsPerBlock, chunk_end - tile_base);
    {
      const int t = tile_base + warp_id;
      float score = neg_inf<float>();
      if (warp_id < tile_tokens) {
        const int base = cache_index(phys_base + t, kv_head, 0, num_kv_heads, head_dim);
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
      if (lane == 0 && warp_id < tile_tokens) score_shared[warp_id] = score;
    }
    {
      for (int i = 0; i < tile_tokens; ++i) {
        const int base = cache_index(phys_base + tile_base + i, kv_head, 0, num_kv_heads, head_dim);
        for (int d = tid; d < head_dim; d += blockDim.x)
          v_tile[i * head_dim + d] = v_pool[base + d];
      }
    }
    __syncthreads();
    if (tid == 0) {
      float tile_m = neg_inf<float>();
      for (int i = 0; i < tile_tokens; ++i) tile_m = fmaxf(tile_m, score_shared[i]);
      float tile_l = 0.0f;
      for (int i = 0; i < tile_tokens; ++i) {
        const float bb = expf(score_shared[i] - tile_m);
        beta_shared[i] = bb;
        tile_l += bb;
      }
      stats_shared[2] = tile_m;
      stats_shared[3] = tile_l;
    }
    __syncthreads();
    {
      const float tile_m = stats_shared[2], tile_l = stats_shared[3];
      const float running_m = stats_shared[0], running_l = stats_shared[1];
      const float new_m = fmaxf(running_m, tile_m);
      const float c_prev = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float c_tile = expf(tile_m - new_m);
      int j = 0;
      for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
        float tile_o = 0.0f;
        for (int i = 0; i < tile_tokens; ++i)
          tile_o += beta_shared[i] * __half2float(v_tile[i * head_dim + d]);
        acc[j] = acc[j] * c_prev + tile_o * c_tile;
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
  int jo = 0;
  for (int d = tid; d < head_dim; d += blockDim.x, ++jo)
    chunk_o[static_cast<std::size_t>(chunk_index) * head_dim + d] = acc[jo];
}

__global__ void attention_step_chunk_reduce_batched_kernel(const float* chunk_m,
                                                           const float* chunk_l,
                                                           const float* chunk_o, half* out,
                                                           const int* __restrict__ seq_lens,
                                                           int num_heads, int head_dim,
                                                           int chunk_size, int scratch_chunks) {
  __shared__ float scale_shared[3];
  const int b = blockIdx.y;
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int chunk_count = (seq_lens[b] + chunk_size - 1) / chunk_size;
  // One accumulator per head_dim element this thread owns: head_dim can exceed
  // blockDim (256 with 128 threads), so a scalar acc would conflate the strided
  // output dims. Unlike the non-batched split-K reduce (head_dim==128 gated),
  // this kernel runs at any head_dim, so the array is required for correctness.
  float acc[kAccPerThread];
#pragma unroll
  for (int i = 0; i < kAccPerThread; ++i) acc[i] = 0.0f;
  float running_m = neg_inf<float>(), running_l = 0.0f;
  for (int chunk = 0; chunk < chunk_count; ++chunk) {
    if (tid == 0) {
      const int idx = (b * num_heads + head) * scratch_chunks + chunk;
      const float cm = chunk_m[idx], cl = chunk_l[idx];
      const float new_m = fmaxf(running_m, cm);
      const float alpha = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float beta = (cl == 0.0f) ? 0.0f : expf(cm - new_m);
      running_l = running_l * alpha + cl * beta;
      running_m = new_m;
      scale_shared[0] = alpha;
      scale_shared[1] = beta;
      scale_shared[2] = running_l;
    }
    __syncthreads();
    const float alpha = scale_shared[0], beta = scale_shared[1];
    const std::size_t base =
        (static_cast<std::size_t>(b * num_heads + head) * scratch_chunks + chunk) * head_dim;
    int j = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++j)
      acc[j] = acc[j] * alpha + chunk_o[base + d] * beta;
    __syncthreads();
  }
  const float inv_l = 1.0f / fmaxf(scale_shared[2], 1e-8f);
  half* out_seq = out + (static_cast<std::size_t>(b) * num_heads + head) * head_dim;
  int j = 0;
  for (int d = tid; d < head_dim; d += blockDim.x, ++j) out_seq[d] = __float2half(acc[j] * inv_l);
}

// Batched + paged GQA-shared variant of split-K pass 1. grid = (num_kv_heads,
// chunk, batch); blockDim = group_size*32 (one warp per query head in the KV
// group). Each block loads one KV chunk ONCE into shared and reuses it across the
// group's query heads, so KV traffic drops by group_size vs the per-query-head
// batched kernel (which re-reads each KV token once per head). chunk_size ==
// block_size (<= 32): chunk c of sequence b maps to physical block
// block_tables[b][c] and lane t owns token t within it. Writes per-query-head
// partials in the exact layout attention_step_chunk_reduce_batched_kernel
// consumes, so pass 2 is unchanged.
template <int HeadDim>
__device__ __forceinline__ void gqa_split_chunk_stats_batched_core(
    const half* q, const half* k_pool, const half* v_pool, const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens, int max_blocks, float* chunk_m, float* chunk_l,
    float* chunk_o, int num_heads, int num_kv_heads, int group_size, int block_size,
    int scratch_chunks) {
  const int b = blockIdx.z;
  const int seq_len = seq_lens[b];
  const int kv_head = blockIdx.x;
  const int chunk = blockIdx.y;
  const int chunk_start = chunk * block_size;
  if (chunk_start >= seq_len) return;
  const int chunk_end = min(chunk_start + block_size, seq_len);
  const int tile_tokens = chunk_end - chunk_start;

  // K and V are used in disjoint phases (K for scores, V for the output sum), so
  // they share ONE tile buffer: stage K, score, then overwrite with V. Peak smem
  // is a single block_size*HeadDim tile instead of two, which roughly doubles
  // occupancy and makes the shared path a win at HeadDim=256 (where two tiles =
  // ~32 KB crushed it).
  extern __shared__ unsigned char smem_bytes[];
  half* q_sh = reinterpret_cast<half*>(smem_bytes);                        // group_size*HeadDim
  half* kv_tile = q_sh + group_size * HeadDim;                             // block_size*HeadDim
  float* w_sh = reinterpret_cast<float*>(kv_tile + block_size * HeadDim);  // group_size*block_size

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid & 31;
  const float scale = rsqrtf(static_cast<float>(HeadDim));
  const int q_head = kv_head * group_size + warp_id;

  const int* block_table = block_tables + static_cast<std::size_t>(b) * max_blocks;
  const int phys_row0 = block_table[chunk] * block_size;  // physical row of token 0 in this chunk
  const half* q_seq = q + (static_cast<std::size_t>(b) * num_heads + q_head) * HeadDim;

  // Phase 1: stage this group's query heads + the chunk's paged K tile (once,
  // reused across all group_size warps). Vectorized int4 (8-half) loads: each
  // token's HeadDim slice is contiguous and 16-byte aligned (HeadDim % 8 == 0),
  // so 8 halves move per instruction — full-width HBM transactions vs scalar.
  constexpr int kHd8 = HeadDim / 8;  // int4 chunks per token
  int4* kv4 = reinterpret_cast<int4*>(kv_tile);
  for (int d = lane; d < HeadDim; d += 32) q_sh[warp_id * HeadDim + d] = q_seq[d];
  for (int i8 = tid; i8 < tile_tokens * kHd8; i8 += blockDim.x) {
    const int t = i8 / kHd8;
    const int c = i8 - t * kHd8;
    const int base = cache_index(phys_row0 + t, kv_head, 0, num_kv_heads, HeadDim);
    kv4[i8] = reinterpret_cast<const int4*>(k_pool + base)[c];
  }
  __syncthreads();

  // Each warp handles one query head; lane t scores token t against shared K.
  float score = neg_inf<float>();
  if (lane < tile_tokens) {
    const half2* qh = reinterpret_cast<const half2*>(q_sh + warp_id * HeadDim);
    const half2* kt = reinterpret_cast<const half2*>(kv_tile + lane * HeadDim);
    float dot = 0.0f;
#pragma unroll
    for (int p = 0; p < HeadDim / 2; ++p) {
      const float2 a = __half22float2(qh[p]);
      const float2 bb = __half22float2(kt[p]);
      dot += a.x * bb.x + a.y * bb.y;
    }
    score = dot * scale;
  }
  float tile_m = score;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) tile_m = fmaxf(tile_m, __shfl_xor_sync(0xffffffffu, tile_m, o));
  const float weight = (lane < tile_tokens) ? expf(score - tile_m) : 0.0f;
  float tile_l = weight;
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) tile_l += __shfl_xor_sync(0xffffffffu, tile_l, o);
  w_sh[warp_id * block_size + lane] = weight;
  __syncthreads();  // all warps done reading K; safe to overwrite the tile with V

  // Phase 2: stage the paged V tile into the SAME buffer (vectorized int4), then
  // weighted V sum (each lane owns HeadDim/32 output channels).
  for (int i8 = tid; i8 < tile_tokens * kHd8; i8 += blockDim.x) {
    const int t = i8 / kHd8;
    const int c = i8 - t * kHd8;
    const int base = cache_index(phys_row0 + t, kv_head, 0, num_kv_heads, HeadDim);
    kv4[i8] = reinterpret_cast<const int4*>(v_pool + base)[c];
  }
  __syncthreads();

  const int chunk_index = (b * num_heads + q_head) * scratch_chunks + chunk;
#pragma unroll
  for (int i = 0; i < HeadDim / 32; ++i) {
    const int d = lane + i * 32;
    float o = 0.0f;
    for (int t = 0; t < tile_tokens; ++t)
      o += w_sh[warp_id * block_size + t] * __half2float(kv_tile[t * HeadDim + d]);
    chunk_o[static_cast<std::size_t>(chunk_index) * HeadDim + d] = o;
  }
  if (lane == 0) {
    chunk_m[chunk_index] = tile_m;
    chunk_l[chunk_index] = tile_l;
  }
}

template <int HeadDim>
__global__ void attention_step_chunk_stats_gqa_split_batched_kernel(
    const half* q, const half* k_pool, const half* v_pool, const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens, int max_blocks, float* chunk_m, float* chunk_l,
    float* chunk_o, int num_heads, int num_kv_heads, int group_size, int block_size,
    int scratch_chunks) {
  gqa_split_chunk_stats_batched_core<HeadDim>(q, k_pool, v_pool, block_tables, seq_lens, max_blocks,
                                              chunk_m, chunk_l, chunk_o, num_heads, num_kv_heads,
                                              group_size, block_size, scratch_chunks);
}

void launch_attention_step_batched_paged(const half* q, const half* k_pool, const half* v_pool,
                                         const int* block_tables, const int* seq_lens,
                                         int max_blocks, int max_seq_len, half* out, int batch,
                                         int num_heads, int num_kv_heads, int head_dim,
                                         int block_size, cudaStream_t stream, float* scratch_m,
                                         float* scratch_l, float* scratch_o, int scratch_chunks) {
  constexpr int warps = 4;
  constexpr int threads = warps * 32;
  const int chunk_count = min(scratch_chunks, (max_seq_len + block_size - 1) / block_size);

  // GQA-shared path: one block per (KV head, chunk, sequence) with group_size
  // warps sharing each KV tile, cutting KV traffic by group_size. Requires a real
  // GQA group, block_size <= 32 (lane owns one token), and group_size <= 32 (warps
  // per block). K and V share one shared-memory tile (staged in disjoint phases),
  // so peak smem is a single block_size*HeadDim tile — small enough that both
  // head_dim 128 and 256 keep good occupancy.
  const int kv_hs = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = num_heads / kv_hs;
  const bool gqa_shared = num_kv_heads > 0 && group_size > 1 && (num_heads % kv_hs) == 0 &&
                          block_size <= 32 && group_size <= 32 &&
                          (head_dim == 128 || head_dim == 256);
  if (gqa_shared) {
    const int threads_g = group_size * 32;
    const std::size_t smem_g = (static_cast<std::size_t>(group_size) * head_dim +
                                static_cast<std::size_t>(block_size) * head_dim) *
                                   sizeof(half) +
                               static_cast<std::size_t>(group_size) * block_size * sizeof(float);
    const dim3 grid_g(num_kv_heads, chunk_count, batch);
    if (head_dim == 128) {
      attention_step_chunk_stats_gqa_split_batched_kernel<128>
          <<<grid_g, threads_g, smem_g, stream>>>(
              q, k_pool, v_pool, block_tables, seq_lens, max_blocks, scratch_m, scratch_l,
              scratch_o, num_heads, num_kv_heads, group_size, block_size, scratch_chunks);
    } else {
      attention_step_chunk_stats_gqa_split_batched_kernel<256>
          <<<grid_g, threads_g, smem_g, stream>>>(
              q, k_pool, v_pool, block_tables, seq_lens, max_blocks, scratch_m, scratch_l,
              scratch_o, num_heads, num_kv_heads, group_size, block_size, scratch_chunks);
    }
  } else {
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                             static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                             static_cast<std::size_t>(warps * head_dim) * sizeof(half);
    const dim3 grid(num_heads, chunk_count, batch);
    attention_step_chunk_stats_batched_kernel<warps><<<grid, threads, smem, stream>>>(
        q, k_pool, v_pool, block_tables, seq_lens, max_blocks, scratch_m, scratch_l, scratch_o,
        num_heads, num_kv_heads, head_dim, block_size, scratch_chunks);
  }

  const dim3 rgrid(num_heads, batch);
  attention_step_chunk_reduce_batched_kernel<<<rgrid, threads, 0, stream>>>(
      scratch_m, scratch_l, scratch_o, out, seq_lens, num_heads, head_dim, block_size,
      scratch_chunks);
}

void launch_attention_step_device_pos(const half* q, const half* k_cache, const half* v_cache,
                                      half* out, const int* position, int num_heads,
                                      int num_kv_heads, int head_dim, cudaStream_t stream,
                                      float* scratch_m, float* scratch_l, float* scratch_o,
                                      int scratch_chunks, bool allow_split) {
  const bool force_fallback = std::getenv("LLAMA_INFER_FORCE_FALLBACK_DECODE_ATTENTION") != nullptr;
  if (force_fallback) {
    constexpr int threads = 128;
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                             static_cast<std::size_t>(threads + 3) * sizeof(float);
    attention_step_kernel_fallback_device_pos<<<num_heads, threads, smem, stream>>>(
        q, k_cache, v_cache, out, position, num_heads, num_kv_heads, head_dim);
    return;
  }
  // GQA fused dispatch (device-position variant for CUDA graph capture). The
  // graphed decode path can't gate on runtime seq_len (position is device-side),
  // so when the split-K path is available it takes priority for GQA models:
  // split-K parallelizes over KV chunks (fixed num_heads×scratch_chunks grid
  // with a per-chunk seq guard, so it stays graph-safe as the sequence grows)
  // and avoids the GQA-fused kernel's num_kv_heads-block serial scan that
  // dominated long-context decode. GQA-fused stays as the fallback when split is
  // disabled (--no-split-attention).
  const bool split_available =
      allow_split && scratch_m && scratch_l && scratch_o && scratch_chunks > 0 && head_dim == 128;
  if (!split_available && num_kv_heads > 0 && num_heads > num_kv_heads &&
      (num_heads % num_kv_heads) == 0 && head_dim == 128) {
    const int group_size_val = num_heads / num_kv_heads;
    if (group_size_val <= 32) {
      const int threads_gqa = group_size_val * 32;
      const std::size_t smem_gqa = static_cast<std::size_t>(group_size_val + 1) *
                                       static_cast<std::size_t>(head_dim) * sizeof(half) +
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
    // GQA models: share each K/V read across the query-head group (grid.x =
    // num_kv_heads, group_size warps) while split-K over KV chunks (grid.y)
    // keeps occupancy. Graph-safe: fixed grid, per-chunk seq guard.
    const bool gqa = num_kv_heads > 0 && num_heads > num_kv_heads &&
                     (num_heads % num_kv_heads) == 0 && (num_heads / num_kv_heads) * 32 <= 1024;
    if (gqa) {
      const int group_size_val = num_heads / num_kv_heads;
      const std::size_t smem_gqa =
          static_cast<std::size_t>(group_size_val + 2 * split_chunk_size) * head_dim *
              sizeof(half) +
          static_cast<std::size_t>(group_size_val) * split_chunk_size * sizeof(float);
      const dim3 grid(num_kv_heads, scratch_chunks);
      attention_step_chunk_stats_gqa_split_device_pos_kernel<128>
          <<<grid, group_size_val * 32, smem_gqa, stream>>>(
              q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, position, num_kv_heads,
              group_size_val, split_chunk_size, scratch_chunks);
    } else {
      const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                               static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                               static_cast<std::size_t>(warps * head_dim) * sizeof(half);
      const dim3 grid(num_heads, scratch_chunks);
      attention_step_chunk_stats_device_pos_kernel<warps><<<grid, threads, smem, stream>>>(
          q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, position, num_heads, num_kv_heads,
          head_dim, split_chunk_size, scratch_chunks);
    }
    attention_step_chunk_reduce_device_pos_kernel<<<num_heads, threads, 0, stream>>>(
        scratch_m, scratch_l, scratch_o, out, position, num_heads, head_dim, split_chunk_size,
        scratch_chunks);
    return;
  }

  if (head_dim > 0 && (head_dim % 2) == 0 && head_dim <= kTiledMaxHeadDim) {
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

void launch_store_kv_device_pos(const half* k, const half* v, half* k_cache, half* v_cache,
                                const int* position, int kv_hidden, int max_context,
                                cudaStream_t stream) {
  const bool aligned =
      ((reinterpret_cast<std::uintptr_t>(k) | reinterpret_cast<std::uintptr_t>(v) |
        reinterpret_cast<std::uintptr_t>(k_cache) | reinterpret_cast<std::uintptr_t>(v_cache)) &
       (alignof(int4) - 1)) == 0;
  if ((kv_hidden & 7) == 0 && aligned) {
    const int kv_hidden_vec8 = kv_hidden / 8;
    const int threads = choose_copy_threads(kv_hidden);
    const int blocks = (kv_hidden_vec8 + threads - 1) / threads;
    store_kv_device_pos_vec8_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const int4*>(k), reinterpret_cast<const int4*>(v),
        reinterpret_cast<int4*>(k_cache), reinterpret_cast<int4*>(v_cache), position,
        kv_hidden_vec8, max_context);
    return;
  }

  constexpr int threads = 256;
  const int blocks = (kv_hidden + threads - 1) / threads;
  store_kv_device_pos_kernel<<<blocks, threads, 0, stream>>>(k, v, k_cache, v_cache, position,
                                                             kv_hidden, max_context);
}

void launch_copy_int(const int* src, int* dst, cudaStream_t stream) {
  copy_int_kernel<<<1, 1, 0, stream>>>(src, dst);
}

void launch_increment_int(int* value, cudaStream_t stream) {
  increment_int_kernel<<<1, 1, 0, stream>>>(value);
}

}  // namespace kernels
