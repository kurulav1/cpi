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
                                                             int num_kv_heads, int head_dim,
                                                             int k_start = 0) {
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

  for (int t = k_start; t < seq_len; ++t) {
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
                                                          int num_kv_heads, int head_dim,
                                                          int window) {
  const int seq_len = position_ptr[0] + 1;
  const int k_start = (window > 0 && seq_len > window) ? seq_len - window : 0;
  attention_step_fallback_core(q, k_cache, v_cache, out, seq_len, num_heads, num_kv_heads, head_dim,
                               k_start);
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
                                                          int num_kv_heads, int head_dim,
                                                          int k_start = 0) {
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
  for (int tile_base = k_start; tile_base < seq_len; tile_base += WarpsPerBlock) {
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
                                                       int num_kv_heads, int head_dim, int window) {
  const int seq_len = position_ptr[0] + 1;
  const int k_start = (window > 0 && seq_len > window) ? seq_len - window : 0;
  attention_step_tiled_core<WarpsPerBlock>(q, k_cache, v_cache, out, seq_len, num_heads,
                                           num_kv_heads, head_dim, k_start);
}

// Shared split-K decode pass-1 core: each block computes softmax statistics and
// an unnormalized partial output for one [head, chunk] pair. seq_len is resolved
// by the caller, so the host and CUDA-graph device-position kernels share this
// one body.
template <int WarpsPerBlock>
__device__ __forceinline__ void attention_step_chunk_stats_core(
    const half* q, const half* k_cache, const half* v_cache, float* chunk_m, float* chunk_l,
    float* chunk_o, int seq_len, int num_heads, int num_kv_heads, int head_dim, int chunk_size,
    int scratch_chunks, int k_start = 0) {
  // Same score per key as the tile-loop original (same warp mapping, lane stride and
  // warp_sum) and the same tile-merge arithmetic in the same order; what changed is who
  // computes it. All keys are scored up front (one barrier), then every thread runs the
  // tile-merge recurrence redundantly in registers instead of round-tripping running
  // stats through shared memory with a single-thread serial section per tile. V is read
  // straight from cache (the old v_tile staging bought nothing: reads were already
  // coalesced and hot). smem layout: half q_shared[head_dim]; float score_shared[chunk_size].
  extern __shared__ unsigned char smem_bytes[];
  half* q_shared = reinterpret_cast<half*>(smem_bytes);
  float* score_shared = reinterpret_cast<float*>(q_shared + head_dim);

  const int head = blockIdx.x;
  // blockIdx.y maps to the first live chunk, so a windowed layer's grid only needs to span
  // the window: launchers may size grid.y to the live chunk count instead of the whole
  // scratch range. Scratch stays absolutely indexed; with k_start == 0 this is the old
  // mapping, and any grid larger than the live range still exits on the guards below.
  const int chunk = blockIdx.y + k_start / chunk_size;
  const int chunk_start = max(chunk * chunk_size, k_start);
  if (chunk_start >= seq_len || (chunk + 1) * chunk_size <= k_start) {
    return;
  }
  const int chunk_end = min(chunk * chunk_size + chunk_size, seq_len);
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
  __syncthreads();

  // Score every key in the chunk. Key (chunk_start + i) belongs to warp (i % WarpsPerBlock)
  // exactly as the old tile loop assigned it, so each dot is the same fp sequence.
  const int score_warps =
      blockDim.x / warpSize;  // launchers may raise threads for more parallel dots
  for (int t = chunk_start + warp_id; t < chunk_end; t += score_warps) {
    const int base = cache_index(t, kv_head, 0, num_kv_heads, head_dim);
    const half2* q2 = reinterpret_cast<const half2*>(q_shared);
    const half2* k2 = reinterpret_cast<const half2*>(k_cache + base);
    float partial = 0.0f;
#pragma unroll 4
    for (int pair = lane; pair < head_pairs; pair += warpSize) {
      const float2 qv = __half22float2(q2[pair]);
      const float2 kv = __half22float2(k2[pair]);
      partial += qv.x * kv.x + qv.y * kv.y;
    }
    const float score = warp_sum(partial) * scale;
    if (lane == 0) {
      score_shared[t - chunk_start] = score;
    }
  }
  __syncthreads();

  // Every thread runs the tile-merge recurrence redundantly in registers: identical inputs
  // in identical order produce identical values on every thread, so the shared-memory
  // stats round-trips and their barriers vanish without moving a single rounding.
  float running_m = neg_inf<float>();
  float running_l = 0.0f;
  float acc[kAccPerThread];
#pragma unroll
  for (int jj = 0; jj < kAccPerThread; ++jj) acc[jj] = 0.0f;

  for (int tile_base = chunk_start; tile_base < chunk_end; tile_base += WarpsPerBlock) {
    const int tile_tokens = min(WarpsPerBlock, chunk_end - tile_base);
    // Compile-time trip counts with predication: the runtime-bound versions could not
    // unroll, so every V load waited on the previous FMA; a serial load-use chain that
    // block-level overlap hides at depth and nothing hides shallow. Unrolled, the loads
    // batch.
    float tile_m = neg_inf<float>();
#pragma unroll
    for (int i = 0; i < WarpsPerBlock; ++i) {
      if (i < tile_tokens) {
        tile_m = fmaxf(tile_m, score_shared[tile_base - chunk_start + i]);
      }
    }
    float beta[WarpsPerBlock];
    float tile_l = 0.0f;
#pragma unroll
    for (int i = 0; i < WarpsPerBlock; ++i) {
      if (i < tile_tokens) {
        beta[i] = expf(score_shared[tile_base - chunk_start + i] - tile_m);
        tile_l += beta[i];
      }
    }
    const float new_m = fmaxf(running_m, tile_m);
    const float c_prev = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
    const float c_tile = expf(tile_m - new_m);

    int j = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
      float vbuf[WarpsPerBlock];
#pragma unroll
      for (int i = 0; i < WarpsPerBlock; ++i) {
        if (i < tile_tokens) {
          const int base = cache_index(tile_base + i, kv_head, 0, num_kv_heads, head_dim);
          vbuf[i] = __half2float(v_cache[base + d]);
        }
      }
      float tile_o = 0.0f;
#pragma unroll
      for (int i = 0; i < WarpsPerBlock; ++i) {
        if (i < tile_tokens) {
          tile_o += beta[i] * vbuf[i];
        }
      }
      acc[j] = acc[j] * c_prev + tile_o * c_tile;
    }
    running_m = new_m;
    running_l = running_l * c_prev + tile_l * c_tile;
  }

  if (tid == 0) {
    chunk_m[chunk_index] = running_m;
    chunk_l[chunk_index] = running_l;
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

// any-head_dim split-K wrappers (Gemma 4's 256/512, Qwen3.5's 256), window-aware. The
// stats core already handles arbitrary head_dim (kAccPerThread accumulators); these add the
// window (as a k_start clip) and a reduce whose accumulator survives head_dim > blockDim
// the original reduce's scalar acc is correct only when every thread owns one output dim,
// which is exactly why the split path was gated to head_dim <= 128.
template <int WarpsPerBlock>
__global__ void attention_step_chunk_stats_any_kernel(const half* q, const half* k_cache,
                                                      const half* v_cache, float* chunk_m,
                                                      float* chunk_l, float* chunk_o, int seq_len,
                                                      int num_heads, int num_kv_heads, int head_dim,
                                                      int chunk_size, int scratch_chunks,
                                                      int k_start) {
  attention_step_chunk_stats_core<WarpsPerBlock>(q, k_cache, v_cache, chunk_m, chunk_l, chunk_o,
                                                 seq_len, num_heads, num_kv_heads, head_dim,
                                                 chunk_size, scratch_chunks, k_start);
}

template <int WarpsPerBlock>
__global__ void attention_step_chunk_stats_any_device_pos_kernel(
    const half* q, const half* k_cache, const half* v_cache, float* chunk_m, float* chunk_l,
    float* chunk_o, const int* position_ptr, int num_heads, int num_kv_heads, int head_dim,
    int chunk_size, int scratch_chunks, int window) {
  const int seq_len = position_ptr[0] + 1;
  const int k_start = (window > 0 && seq_len > window) ? seq_len - window : 0;
  attention_step_chunk_stats_core<WarpsPerBlock>(q, k_cache, v_cache, chunk_m, chunk_l, chunk_o,
                                                 seq_len, num_heads, num_kv_heads, head_dim,
                                                 chunk_size, scratch_chunks, k_start);
}

__device__ __forceinline__ void attention_step_chunk_reduce_any_core(
    const float* chunk_m, const float* chunk_l, const float* chunk_o, half* out, int seq_len,
    int num_heads, int head_dim, int chunk_size, int scratch_chunks, int k_start) {
  // Every thread runs the m/l scan redundantly in registers; identical inputs in
  // identical order give identical values, so the old per-chunk single-thread section and
  // its two barriers (x26 chunks) vanish without moving a rounding. The m/l loads are
  // same-address broadcasts.
  // two-pass combine: the running-rescale recurrence (acc = acc*alpha + o*beta) is a
  // serial dependency across chunks, which dominates deep. Pass 1 finds the global max
  // (scalar, broadcast loads); pass 2 accumulates independent o_c * exp(m_c - M)
  // products, which pipeline freely. One exp scale per chunk instead of a running
  // rescale is also numerically tighter.
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int first_chunk = k_start / chunk_size;
  const int chunk_count = (seq_len + chunk_size - 1) / chunk_size;
  float acc[kAccPerThread];
#pragma unroll
  for (int j = 0; j < kAccPerThread; ++j) acc[j] = 0.0f;

  float global_m = neg_inf<float>();
  for (int chunk = first_chunk; chunk < chunk_count; ++chunk) {
    global_m = fmaxf(global_m, chunk_m[head * scratch_chunks + chunk]);
  }
  float total_l = 0.0f;
  for (int chunk = first_chunk; chunk < chunk_count; ++chunk) {
    const int idx = head * scratch_chunks + chunk;
    const float chunk_l_value = chunk_l[idx];
    const float w = (chunk_l_value == 0.0f) ? 0.0f : expf(chunk_m[idx] - global_m);
    total_l += chunk_l_value * w;
    const std::size_t base =
        (static_cast<std::size_t>(head) * static_cast<std::size_t>(scratch_chunks) +
         static_cast<std::size_t>(chunk)) *
        static_cast<std::size_t>(head_dim);
    int j = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
      acc[j] += chunk_o[base + static_cast<std::size_t>(d)] * w;
    }
  }

  const float inv_l = 1.0f / fmaxf(total_l, 1e-8f);
  int j = 0;
  for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
    out[head * head_dim + d] = __float2half(acc[j] * inv_l);
  }
}

__global__ void attention_step_chunk_reduce_any_kernel(const float* chunk_m, const float* chunk_l,
                                                       const float* chunk_o, half* out, int seq_len,
                                                       int num_heads, int head_dim, int chunk_size,
                                                       int scratch_chunks, int k_start) {
  attention_step_chunk_reduce_any_core(chunk_m, chunk_l, chunk_o, out, seq_len, num_heads, head_dim,
                                       chunk_size, scratch_chunks, k_start);
}

__global__ void attention_step_chunk_reduce_any_device_pos_kernel(
    const float* chunk_m, const float* chunk_l, const float* chunk_o, half* out,
    const int* position_ptr, int num_heads, int head_dim, int chunk_size, int scratch_chunks,
    int window) {
  const int seq_len = position_ptr[0] + 1;
  const int k_start = (window > 0 && seq_len > window) ? seq_len - window : 0;
  attention_step_chunk_reduce_any_core(chunk_m, chunk_l, chunk_o, out, seq_len, num_heads, head_dim,
                                       chunk_size, scratch_chunks, k_start);
}

// Multi-query twins for the graphed speculative verify: blockIdx.z / blockIdx.y select the
// query token t, whose causal KV length is (base position + t + 1). Query/output rows use
// the sequence-slot stride [token][heads*head_dim]; scratch adds a leading token axis. The
// per-chunk seq guard makes the fixed max-context grid safe at any live depth, and per-token
// lengths give exact causality inside the verify batch.
template <int WarpsPerBlock>
__global__ void attention_step_chunk_stats_any_mt_device_pos_kernel(
    const half* q, const half* k_cache, const half* v_cache, float* chunk_m, float* chunk_l,
    float* chunk_o, const int* position_ptr, int num_heads, int num_kv_heads, int head_dim,
    int chunk_size, int scratch_chunks, int window) {
  const int t = blockIdx.z;
  const int seq_len = position_ptr[0] + t + 1;
  const int k_start = (window > 0 && seq_len > window) ? seq_len - window : 0;
  const std::size_t scr = static_cast<std::size_t>(t) * num_heads * scratch_chunks;
  attention_step_chunk_stats_core<WarpsPerBlock>(
      q + static_cast<std::size_t>(t) * num_heads * head_dim, k_cache, v_cache, chunk_m + scr,
      chunk_l + scr, chunk_o + scr * head_dim, seq_len, num_heads, num_kv_heads, head_dim,
      chunk_size, scratch_chunks, k_start);
}

__global__ void attention_step_chunk_reduce_any_mt_device_pos_kernel(
    const float* chunk_m, const float* chunk_l, const float* chunk_o, half* out,
    const int* position_ptr, int num_heads, int head_dim, int chunk_size, int scratch_chunks,
    int window) {
  const int t = blockIdx.y;
  const int seq_len = position_ptr[0] + t + 1;
  const int k_start = (window > 0 && seq_len > window) ? seq_len - window : 0;
  const std::size_t scr = static_cast<std::size_t>(t) * num_heads * scratch_chunks;
  attention_step_chunk_reduce_any_core(chunk_m + scr, chunk_l + scr, chunk_o + scr * head_dim,
                                       out + static_cast<std::size_t>(t) * num_heads * head_dim,
                                       seq_len, num_heads, head_dim, chunk_size, scratch_chunks,
                                       k_start);
}

// Tree-verify split-K twins: like the mt twins above, but every row's CACHE
// length is exactly *position (the committed prefix; in-batch causality is
// an ancestor bitmask, not position order), and the in-batch scratch columns
// form one extra virtual chunk at index ceil(cache_len/chunk_size), merged by
// the tree reduce below. Restores the split-K parallelism the one-block-per-
// (row,head) tree kernel gives up (latency-bound at depth: the decode
// attention coarsening trap, again).
template <int WarpsPerBlock>
__global__ void attention_tree_chunk_stats_kernel(const half* q, const half* k_cache,
                                                  const half* v_cache, float* chunk_m,
                                                  float* chunk_l, float* chunk_o,
                                                  const int* position_ptr, int num_heads,
                                                  int num_kv_heads, int head_dim, int chunk_size,
                                                  int scratch_chunks) {
  const int t = blockIdx.z;
  const int seq_len = position_ptr[0];
  const std::size_t scr = static_cast<std::size_t>(t) * num_heads * scratch_chunks;
  attention_step_chunk_stats_core<WarpsPerBlock>(
      q + static_cast<std::size_t>(t) * num_heads * head_dim, k_cache, v_cache, chunk_m + scr,
      chunk_l + scr, chunk_o + scr * head_dim, seq_len, num_heads, num_kv_heads, head_dim,
      chunk_size, scratch_chunks, /*k_start=*/0);
}

// One block per (head, row): softmax stats over the <= 32 in-batch scratch
// columns the row's ancestor bitmask admits, written to the virtual chunk
// slot after the last cache chunk. scr_cols is the scratch column count; the
// verify passes scr_cols == rows (its scratch is the batch itself) while the
// batched draft levels attend a wider node scratch than their own row count.
__global__ void attention_tree_inbatch_stats_kernel(const half* q, const half* k_scratch,
                                                    const half* v_scratch, float* chunk_m,
                                                    float* chunk_l, float* chunk_o,
                                                    const int* position_ptr,
                                                    const unsigned int* anc_mask, int scr_cols,
                                                    int num_heads, int num_kv_heads, int head_dim,
                                                    int chunk_size, int scratch_chunks) {
  const int head = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int lane = tid % warpSize;
  const int warp_id = tid / warpSize;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head =
      ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const int head_pairs = head_dim / 2;
  const unsigned int mask = anc_mask[row];
  const int cache_len = position_ptr[0];
  const int slot = (cache_len + chunk_size - 1) / chunk_size;

  extern __shared__ float smem[];
  float* q_shared = smem;                    // [head_dim]
  float* score_shared = q_shared + head_dim; // [rows]

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = __half2float(q[(static_cast<std::size_t>(row) * num_heads + head) * head_dim + d]);
  }
  __syncthreads();

  // One warp per admitted column (scr_cols <= 32 so one pass of warps covers
  // all columns for blockDim >= 32 * ceil(scr_cols/warps); loop to be safe).
  for (int j = warp_id; j < scr_cols; j += blockDim.x / warpSize) {
    float score = neg_inf<float>();
    if ((mask >> j) & 1u) {
      const half2* k2 = reinterpret_cast<const half2*>(
          k_scratch + (static_cast<std::size_t>(j) * kv_heads_safe + kv_head) * head_dim);
      float partial = 0.0f;
      for (int pair = lane; pair < head_pairs; pair += warpSize) {
        const float2 qv = make_float2(q_shared[2 * pair], q_shared[2 * pair + 1]);
        const float2 kv = __half22float2(k2[pair]);
        partial += qv.x * kv.x + qv.y * kv.y;
      }
      score = warp_sum(partial) * scale;
    }
    if (lane == 0) score_shared[j] = score;
  }
  __syncthreads();

  // Serial softmax stats over <= scr_cols entries, then a parallel V combine.
  float m = neg_inf<float>();
  for (int j = 0; j < scr_cols; ++j) m = fmaxf(m, score_shared[j]);
  float l = 0.0f;
  for (int j = 0; j < scr_cols; ++j) {
    const float b = ((mask >> j) & 1u) ? expf(score_shared[j] - m) : 0.0f;
    score_shared[j] = ((mask >> j) & 1u) ? b : 0.0f;
    l += b;
  }
  __syncthreads();
  const std::size_t scr =
      (static_cast<std::size_t>(row) * num_heads + head) * static_cast<std::size_t>(scratch_chunks);
  for (int d = tid; d < head_dim; d += blockDim.x) {
    float o = 0.0f;
    for (int j = 0; j < scr_cols; ++j) {
      if ((mask >> j) & 1u) {
        o += score_shared[j] *
             __half2float(v_scratch[(static_cast<std::size_t>(j) * kv_heads_safe + kv_head) *
                                        head_dim +
                                    d]);
      }
    }
    chunk_o[(scr + slot) * head_dim + d] = o;
  }
  if (tid == 0) {
    chunk_m[scr + slot] = m;
    chunk_l[scr + slot] = l;
  }
}

__global__ void attention_tree_chunk_reduce_kernel(const float* chunk_m, const float* chunk_l,
                                                   const float* chunk_o, half* out,
                                                   const int* position_ptr, int num_heads,
                                                   int head_dim, int chunk_size,
                                                   int scratch_chunks) {
  const int t = blockIdx.y;
  const int cache_len = position_ptr[0];
  // Cache chunks plus the in-batch virtual chunk: reduce_any_core iterates
  // chunk_count = ceil(seq_len/chunk_size), so a synthetic seq_len one chunk
  // past the cache covers slot ceil(cache_len/chunk_size) exactly.
  const int seq_len = ((cache_len + chunk_size - 1) / chunk_size + 1) * chunk_size;
  const std::size_t scr = static_cast<std::size_t>(t) * num_heads * scratch_chunks;
  attention_step_chunk_reduce_any_core(chunk_m + scr, chunk_l + scr, chunk_o + scr * head_dim,
                                       out + static_cast<std::size_t>(t) * num_heads * head_dim,
                                       seq_len, num_heads, head_dim, chunk_size, scratch_chunks,
                                       /*k_start=*/0);
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

// Block reduction into a shared scalar. SUM picks addition, otherwise maximum.
template <bool SUM>
__device__ __forceinline__ void block_reduce_into(float v, float* red, int tid, float* out) {
  const int lane = tid & 31;
  const int warp = tid >> 5;
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float o = __shfl_down_sync(0xFFFFFFFFu, v, off);
    v = SUM ? v + o : fmaxf(v, o);
  }
  if (lane == 0) red[warp] = v;
  __syncthreads();
  if (tid == 0) {
    const int warps = (blockDim.x + 31) >> 5;
    float t = red[0];
    for (int i = 1; i < warps; ++i) t = SUM ? t + red[i] : fmaxf(t, red[i]);
    *out = t;
  }
}

__global__ void attention_step_chunk_reduce_device_pos_kernel(
    const float* chunk_m, const float* chunk_l, const float* chunk_o, half* out,
    const int* position_ptr, int num_heads, int head_dim, int chunk_size, int scratch_chunks,
    std::int8_t* q8, float* q8_scale, float* q8_sum) {
  const int seq_len = position_ptr[0] + 1;
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int chunk_count = (seq_len + chunk_size - 1) / chunk_size;
  const std::size_t head_base = static_cast<std::size_t>(head) *
                                static_cast<std::size_t>(scratch_chunks);

  // Merge the per-chunk softmax stats in two parallel reductions rather than by
  // walking chunks in an online loop. The online form made thread 0 do a
  // dependent global load per chunk with two __syncthreads around it, so the
  // kernel cost a latency chain as deep as the chunk count: 2643 ns for seven
  // chunks. Taking the global max first turns the rest into independent work:
  // every weight is exp(m_c - m) <= 1, which is the same stability the online
  // rescaling was buying.
  __shared__ float red[32];
  __shared__ float m_shared;
  __shared__ float l_shared;

  float local_m = neg_inf<float>();
  for (int c = tid; c < chunk_count; c += blockDim.x) {
    local_m = fmaxf(local_m, chunk_m[head_base + c]);
  }
  block_reduce_into<false>(local_m, red, tid, &m_shared);
  __syncthreads();
  const float m = m_shared;

  float local_l = 0.0f;
  if (m > neg_inf<float>()) {
    for (int c = tid; c < chunk_count; c += blockDim.x) {
      local_l += chunk_l[head_base + c] * __expf(chunk_m[head_base + c] - m);
    }
  }
  block_reduce_into<true>(local_l, red, tid, &l_shared);
  __syncthreads();

  if (tid >= head_dim) return;

  float acc = 0.0f;
  if (m > neg_inf<float>()) {
    // No barrier in here: the weight is a broadcast read of one float per chunk.
    for (int c = 0; c < chunk_count; ++c) {
      const float w = __expf(chunk_m[head_base + c] - m);
      acc += chunk_o[(head_base + c) * static_cast<std::size_t>(head_dim) + tid] * w;
    }
    acc /= fmaxf(l_shared, 1e-8f);
  }
  const half r = __float2half(acc);
  out[head * head_dim + tid] = r;

  // Also emit the q8_1 form the output projection wants, so the layer does not
  // launch a separate quantize over what was just written. This is the last of
  // the four per-layer quantize calls. head_dim is a multiple of 32 and tid maps
  // one-to-one onto the head's elements, so a 32-value group is exactly one warp
  // (the cleanest of the three producers). Threads that dropped out above did so
  // in whole warps, for the same reason, so the shuffles below are safe.
  if (q8 == nullptr) return;
  const float v = __half2float(r);
  float amax = fabsf(v);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, off));
  }
  const float scale = amax > 0.0f ? amax / 127.0f : 1.0f;
  const float qscale = amax > 0.0f ? 127.0f / amax : 1.0f;
  const int qi = max(-127, min(127, __float2int_rn(v * qscale)));
  const int idx = head * head_dim + tid;
  q8[idx] = static_cast<std::int8_t>(qi);
  float sum = static_cast<float>(qi);
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, off);
  if ((tid & 31) == 0) {
    q8_scale[idx >> 5] = scale;
    q8_sum[idx >> 5] = sum * scale;
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

// Sequence form: appends `rows` K/V rows starting at the device-side base position
// (blockIdx.y = row). Rows past max_context are dropped, matching the scalar kernel.
__global__ void store_kv_seq_device_pos_vec8_kernel(const int4* k, const int4* v, int4* k_cache,
                                                    int4* v_cache, const int* position_ptr,
                                                    int kv_hidden_vec8, int max_context) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= kv_hidden_vec8) {
    return;
  }
  const int row = blockIdx.y;
  const int position = position_ptr[0] + row;
  if (position < 0 || position >= max_context) {
    return;
  }
  const std::size_t src = static_cast<std::size_t>(row) * kv_hidden_vec8 + idx;
  const std::size_t dst = static_cast<std::size_t>(position) * kv_hidden_vec8 + idx;
  k_cache[dst] = k[src];
  v_cache[dst] = v[src];
}

__global__ void store_kv_seq_device_pos_kernel(const half* k, const half* v, half* k_cache,
                                               half* v_cache, const int* position_ptr,
                                               int kv_hidden, int max_context) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= kv_hidden) {
    return;
  }
  const int row = blockIdx.y;
  const int position = position_ptr[0] + row;
  if (position < 0 || position >= max_context) {
    return;
  }
  const std::size_t src = static_cast<std::size_t>(row) * kv_hidden + idx;
  const std::size_t dst = static_cast<std::size_t>(position) * kv_hidden + idx;
  k_cache[dst] = k[src];
  v_cache[dst] = v[src];
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
    __syncthreads();  // scores written; K reads done, safe to overwrite kv_sh

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
// to the split-K path (parallel over KV chunks); see launch_attention_step.
constexpr int kGqaFusedMaxSeq = 256;

// ── GQA + split-K decode attention (best of both) ────────────────────────────
// The per-Q-head split-K kernel reads each K/V token once per query head, so a
// GQA group of `group_size` query heads re-reads the same KV `group_size` times
// from HBM. The GQA-fused kernel avoids that but launches only num_kv_heads
// blocks (poor occupancy at long context). This kernel combines both: grid.x =
// num_kv_heads (one block per KV head, `group_size` warps = one per query head),
// grid.y = KV chunk. Each K/V token is loaded to shared once per group per chunk
// and reused across the group's warps, while num_kv_heads×chunks blocks keep the
// GPU busy. It writes per-Q-head chunk partials in the exact layout the existing
// attention_step_chunk_reduce* kernels consume, so pass 2 is unchanged.
template <int HeadDim>
__device__ __forceinline__ void gqa_split_chunk_stats_core(
    const half* q, const half* k_cache, const half* v_cache, float* chunk_m, float* chunk_l,
    float* chunk_o, int seq_len, int num_kv_heads, int group_size, int chunk_size,
    int blocks_per_chunk, int scratch_chunks) {
  // chunk_size is the split step (<= warpSize=32), so lane `t` owns token `t`. A
  // grid block sweeps blocks_per_chunk such sub-chunks with a running online
  // softmax, so long context runs on fewer, larger, latency-hiding blocks instead
  // of thousands of tiny 32-token ones (which left it at ~14% of peak at batch 1).
  const int kv_head = blockIdx.x;
  const int coarse = blockIdx.y;
  const int first_sub = coarse * blocks_per_chunk;  // first sub-chunk in this coarse chunk
  if (first_sub * chunk_size >= seq_len) return;

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
  constexpr int kOutPerLane = HeadDim / 32;

  // Load this group's query heads once (reused across all sub-chunks/warps).
  for (int d = lane; d < HeadDim; d += 32) {
    q_sh[warp_id * HeadDim + d] = q[q_head * HeadDim + d];
  }

  float running_m = neg_inf<float>();
  float running_l = 0.0f;
  float acc[kOutPerLane];
#pragma unroll
  for (int i = 0; i < kOutPerLane; ++i) acc[i] = 0.0f;

  for (int jb = 0; jb < blocks_per_chunk; ++jb) {
    const int sub_start = (first_sub + jb) * chunk_size;
    if (sub_start >= seq_len) break;
    const int tile_tokens = min(chunk_size, seq_len - sub_start);

    __syncthreads();  // protect the K/V tiles from the prior sub-chunk's V sum
    // Stage K/V into shared with 128-bit loads.
    //
    // Copying one half at a time leaves the warps stalled on memory latency with too few
    // requests in flight; reading int4 (8 halves) cuts the request count 8x.
    //
    // Only the staging width changes, so this is bit-identical: the bytes landing in
    // k_tile/v_tile are the same and every arithmetic op below reads from shared. Alignment
    // holds because each (token, kv_head) row is HeadDim halves and HeadDim is 64 or 128.
    constexpr int kHalvesPerVec = 8;  // int4 = 128 bits
    constexpr int kVecPerRow = HeadDim / kHalvesPerVec;
    static_assert(HeadDim % kHalvesPerVec == 0, "HeadDim must be a multiple of 8 for int4 staging");
    int4* k_tile4 = reinterpret_cast<int4*>(k_tile);
    int4* v_tile4 = reinterpret_cast<int4*>(v_tile);
    const int tile_vecs = tile_tokens * kVecPerRow;
    for (int idx = tid; idx < tile_vecs; idx += blockDim.x) {
      const int t = idx / kVecPerRow;
      const int dv = idx - t * kVecPerRow;
      const int base = cache_index(sub_start + t, kv_head, 0, num_kv_heads, HeadDim);
      k_tile4[idx] = reinterpret_cast<const int4*>(k_cache + base)[dv];
      v_tile4[idx] = reinterpret_cast<const int4*>(v_cache + base)[dv];
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
    // Sub-chunk max, then fold into the running softmax (rescale prior partial).
    float sub_m = score;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sub_m = fmaxf(sub_m, __shfl_xor_sync(0xffffffffu, sub_m, o));
    const float new_m = fmaxf(running_m, sub_m);
    const float corr = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
    running_l *= corr;
#pragma unroll
    for (int i = 0; i < kOutPerLane; ++i) acc[i] *= corr;
    const float weight = (lane < tile_tokens) ? expf(score - new_m) : 0.0f;
    float sub_l = weight;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sub_l += __shfl_xor_sync(0xffffffffu, sub_l, o);
    running_l += sub_l;
    running_m = new_m;
    w_sh[warp_id * chunk_size + lane] = weight;
    __syncwarp();

    // Weighted V sum into the running output (each lane owns HeadDim/32 channels).
#pragma unroll
    for (int i = 0; i < kOutPerLane; ++i) {
      const int d = lane + i * 32;
      float o = 0.0f;
      for (int t = 0; t < tile_tokens; ++t) {
        o += w_sh[warp_id * chunk_size + t] * __half2float(v_tile[t * HeadDim + d]);
      }
      acc[i] += o;
    }
  }

  const int chunk_index = q_head * scratch_chunks + coarse;
#pragma unroll
  for (int i = 0; i < kOutPerLane; ++i) {
    const int d = lane + i * 32;
    chunk_o[static_cast<std::size_t>(chunk_index) * HeadDim + d] = acc[i];
  }
  if (lane == 0) {
    chunk_m[chunk_index] = running_m;
    chunk_l[chunk_index] = running_l;
  }
}

template <int HeadDim>
__global__ void attention_step_chunk_stats_gqa_split_kernel(
    const half* q, const half* k_cache, const half* v_cache, float* chunk_m, float* chunk_l,
    float* chunk_o, int seq_len, int num_kv_heads, int group_size, int chunk_size,
    int blocks_per_chunk, int scratch_chunks) {
  gqa_split_chunk_stats_core<HeadDim>(q, k_cache, v_cache, chunk_m, chunk_l, chunk_o, seq_len,
                                      num_kv_heads, group_size, chunk_size, blocks_per_chunk,
                                      scratch_chunks);
}

template <int HeadDim>
__global__ void attention_step_chunk_stats_gqa_split_device_pos_kernel(
    const half* q, const half* k_cache, const half* v_cache, float* chunk_m, float* chunk_l,
    float* chunk_o, const int* position_ptr, int num_kv_heads, int group_size, int chunk_size,
    int blocks_per_chunk, int scratch_chunks) {
  gqa_split_chunk_stats_core<HeadDim>(q, k_cache, v_cache, chunk_m, chunk_l, chunk_o,
                                      position_ptr[0] + 1, num_kv_heads, group_size, chunk_size,
                                      blocks_per_chunk, scratch_chunks);
}

// Coarsening factor for the split-K decode attention: each grid block sweeps this
// many 32-token sub-chunks with a running online softmax. Pick the largest G (<=
// kMaxBpc) that keeps the grid (lanes*coarse_chunks, lanes = num_kv_heads*batch)
// above a floor for occupancy. CPI_ATTN_BPC overrides.
//
// kMaxBpc is 1: measurement says coarsening costs 2% and never pays back.
//
// The intent was that short context "stays at G=1", but it cannot. G is fixed
// when the decode graph is captured, and capture only knows max_context, so
// scratch_chunks is always large and G always lands at the cap. At a 215-token
// context that leaves one coarse chunk per KV head holding data: eight blocks
// on a 170-SM part, each walking seven sub-chunks serially, every one a
// dependent global-to-shared load behind a __syncthreads. nsys shows the cost is
// almost entirely that chain rather than the KV traffic: 8525 ns at a mean
// context of ~75 against 8838 ns at ~600, so eight times the context costs 3.7%
// more time.
//
// Against llama.cpp on an 8B Q4_K_M, G=1 versus the previous G=8:
//   max_new  400 -> +2.0%   (241.8 -> 246.8 tok/s)
//   max_new 1500 -> +0.2%
//   max_new 3000 -> +0.1%
//   max_new 8000 -> +0.0%   (238.6 -> 238.7)
// so the coarsening never paid for itself at any depth reachable here, and the
// batched paged path is unchanged at B=1..64. The comment it replaces claimed
// tiny 32-token blocks left decode attention at ~14% of peak; whatever that
// measured, it is not reproducible on this GPU for this model, so if you
// reinstate coarsening, do it with numbers at the depth you care about.
static inline int pick_blocks_per_chunk(int total_blocks, long long lanes) {
  static const int env_bpc = [] {
    const char* s = std::getenv("CPI_ATTN_BPC");
    return s ? atoi(s) : 0;
  }();
  const int cap = max(1, total_blocks);
  if (env_bpc > 0) return min(env_bpc, cap);
  constexpr int kMaxBpc = 1;     // see above: coarsening measured as a 2% loss
  constexpr int kMinGrid = 256;  // floor on total grid blocks for occupancy
  int bpc = min(kMaxBpc, cap);
  while (bpc > 1) {
    const int cc = (total_blocks + bpc - 1) / bpc;
    if (lanes * static_cast<long long>(cc) >= kMinGrid) break;
    --bpc;
  }
  return bpc;
}

void launch_attention_step(const half* q, const half* k_cache, const half* v_cache, half* out,
                           int seq_len, int num_heads, int num_kv_heads, int head_dim,
                           cudaStream_t stream, float* scratch_m, float* scratch_l,
                           float* scratch_o, int scratch_chunks, bool allow_split) {
  // Cached once: getenv on every launch (per layer, per token) is needless work and, more to the
  // point, is not required to be thread-safe against a concurrent setenv in a threaded server.
  static const bool force_fallback = [] {
    return std::getenv("CPI_FORCE_FALLBACK_DECODE_ATTENTION") != nullptr;
  }();
  if (force_fallback) {
    constexpr int threads = 128;
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                             static_cast<std::size_t>(threads + 3) * sizeof(float);
    attention_step_kernel_fallback<<<num_heads, threads, smem, stream>>>(
        q, k_cache, v_cache, out, seq_len, num_heads, num_kv_heads, head_dim);
    return;
  }
  // GQA fused: one block per KV head, group_size warps share K/V loads.
  // Gives group_size× KV-bandwidth reduction vs. per-Q-head kernels, but it
  // launches only num_kv_heads blocks and scans the sequence serially per block,
  // so it is bandwidth-optimal at short context yet badly under-parallelized at
  // long context (only num_kv_heads blocks, which starves a large GPU at 2k+ tokens).
  // Above this length, fall through to the split-K path (parallel over KV
  // chunks, ~num_heads×ceil(seq/32) blocks).
  // head_dim 64 and 128 both matter: 128 is Llama's, 64 is Qwen2.5's (and plenty of
  // small models'). Gating these fast paths on 128 alone dropped every head_dim-64 model
  // onto the fallback kernel, which runs one block per head and walks the whole KV inside
  // it; so decode got slower the longer the context grew.
  if (num_kv_heads > 0 && num_heads > num_kv_heads && (num_heads % num_kv_heads) == 0 &&
      (head_dim == 64 || head_dim == 128) && seq_len <= kGqaFusedMaxSeq) {
    const int group_size_val = num_heads / num_kv_heads;
    if (group_size_val <= 32) {
      const int threads_gqa = group_size_val * 32;
      const std::size_t smem_gqa = static_cast<std::size_t>(group_size_val + 1) *
                                       static_cast<std::size_t>(head_dim) * sizeof(half) +
                                   static_cast<std::size_t>(group_size_val) * sizeof(float);
      if (head_dim == 128) {
        gqa_decode_kernel<128><<<num_kv_heads, threads_gqa, smem_gqa, stream>>>(
            q, k_cache, v_cache, out, seq_len, num_heads, num_kv_heads, group_size_val);
      } else {
        gqa_decode_kernel<64><<<num_kv_heads, threads_gqa, smem_gqa, stream>>>(
            q, k_cache, v_cache, out, seq_len, num_heads, num_kv_heads, group_size_val);
      }
      return;
    }
  }

  constexpr int split_chunk_size = 32;
  // Split-K over KV chunks: grid is num_heads x ceil(seq/32) blocks instead of num_heads,
  // which is what stops per-token cost from growing with context. The generic chunk-stats
  // kernel takes head_dim at RUNTIME; only this gate was pinning it to 128.
  if (allow_split && scratch_m && scratch_l && scratch_o && scratch_chunks > 0 &&
      (head_dim == 64 || head_dim == 128) && seq_len >= 64) {
    constexpr int warps = 4;
    constexpr int threads = warps * 32;
    const int sub_chunks = min(scratch_chunks, (seq_len + split_chunk_size - 1) / split_chunk_size);
    // GQA models: share each K/V read across the query-head group (no per-head
    // redundancy) while keeping num_kv_heads×chunks blocks for occupancy.
    const bool gqa = num_kv_heads > 0 && num_heads > num_kv_heads &&
                     (num_heads % num_kv_heads) == 0 && (num_heads / num_kv_heads) * 32 <= 1024;
    int reduce_chunk_size = split_chunk_size;
    if (gqa) {
      const int group_size_val = num_heads / num_kv_heads;
      // Coarsen the split at long context: one grid block sweeps G sub-chunks.
      const int G = pick_blocks_per_chunk(sub_chunks, num_kv_heads);
      const int coarse_chunks = (sub_chunks + G - 1) / G;
      reduce_chunk_size = split_chunk_size * G;
      const std::size_t smem_gqa =
          static_cast<std::size_t>(group_size_val + 2 * split_chunk_size) * head_dim *
              sizeof(half) +
          static_cast<std::size_t>(group_size_val) * split_chunk_size * sizeof(float);
      const dim3 grid(num_kv_heads, coarse_chunks);
      if (head_dim == 128) {
        attention_step_chunk_stats_gqa_split_kernel<128>
            <<<grid, group_size_val * 32, smem_gqa, stream>>>(
                q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, seq_len, num_kv_heads,
                group_size_val, split_chunk_size, G, scratch_chunks);
      } else {
        attention_step_chunk_stats_gqa_split_kernel<64>
            <<<grid, group_size_val * 32, smem_gqa, stream>>>(
                q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, seq_len, num_kv_heads,
                group_size_val, split_chunk_size, G, scratch_chunks);
      }
    } else {
      const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                               static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                               static_cast<std::size_t>(warps * head_dim) * sizeof(half);
      const dim3 grid(num_heads, sub_chunks);
      attention_step_chunk_stats_kernel<warps><<<grid, threads, smem, stream>>>(
          q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, seq_len, num_heads, num_kv_heads,
          head_dim, split_chunk_size, scratch_chunks);
    }
    attention_step_chunk_reduce_kernel<<<num_heads, threads, 0, stream>>>(
        scratch_m, scratch_l, scratch_o, out, seq_len, num_heads, head_dim, reduce_chunk_size,
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
// group). Each block loads one KV chunk once into shared and reuses it across the
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
    int blocks_per_chunk, int scratch_chunks) {
  const int b = blockIdx.z;
  const int seq_len = seq_lens[b];
  const int kv_head = blockIdx.x;
  const int coarse = blockIdx.y;  // coarse chunk = blocks_per_chunk paged blocks
  const int first_block = coarse * blocks_per_chunk;
  if (first_block * block_size >= seq_len) return;

  // K and V are used in disjoint phases (K for scores, V for the output sum), so
  // they share one tile buffer: stage K, score, then overwrite with V. Peak smem
  // is a single block_size*HeadDim tile (unchanged by coarsening; the tile is
  // reused across the coarse chunk's sub-blocks), keeping occupancy high while a
  // single grid block now streams blocks_per_chunk*block_size tokens. Fewer,
  // larger blocks hide memory latency far better at long context (batch 1), where
  // one-paged-block-per-grid-block left the kernel latency-bound at ~14% of peak.
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
  const half* q_seq = q + (static_cast<std::size_t>(b) * num_heads + q_head) * HeadDim;
  constexpr int kHd8 = HeadDim / 8;  // int4 chunks per token
  constexpr int kOutPerLane = HeadDim / 32;
  int4* kv4 = reinterpret_cast<int4*>(kv_tile);

  // Stage this group's query heads once (reused across all sub-blocks/warps).
  for (int d = lane; d < HeadDim; d += 32) q_sh[warp_id * HeadDim + d] = q_seq[d];

  // Running online-softmax state, carried across the coarse chunk's sub-blocks:
  // per-warp max/denominator + each lane's owned output channels.
  float running_m = neg_inf<float>();
  float running_l = 0.0f;
  float acc[kOutPerLane];
#pragma unroll
  for (int i = 0; i < kOutPerLane; ++i) acc[i] = 0.0f;

  for (int jb = 0; jb < blocks_per_chunk; ++jb) {
    const int pb = first_block + jb;  // paged block index
    const int tok0 = pb * block_size;
    if (tok0 >= seq_len) break;
    const int tile_tokens = min(block_size, seq_len - tok0);
    const int phys_row0 = block_table[pb] * block_size;

    // Phase 1: stage the paged K tile (vectorized int4, each token's HeadDim
    // slice is contiguous + 16-byte aligned, so 8 halves move per instruction).
    __syncthreads();  // protect the shared tile from the previous sub-block's V sum
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
    // Sub-block max, then fold into the running softmax (rescale prior partial).
    float sub_m = score;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sub_m = fmaxf(sub_m, __shfl_xor_sync(0xffffffffu, sub_m, o));
    const float new_m = fmaxf(running_m, sub_m);
    const float corr = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
    running_l *= corr;
#pragma unroll
    for (int i = 0; i < kOutPerLane; ++i) acc[i] *= corr;
    const float weight = (lane < tile_tokens) ? expf(score - new_m) : 0.0f;
    float sub_l = weight;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sub_l += __shfl_xor_sync(0xffffffffu, sub_l, o);
    running_l += sub_l;
    running_m = new_m;
    w_sh[warp_id * block_size + lane] = weight;
    __syncthreads();  // all warps done reading K + weights ready; safe to overwrite with V

    // Phase 2: stage the paged V tile into the same buffer, then weighted sum into
    // the running output (each lane owns HeadDim/32 channels).
    for (int i8 = tid; i8 < tile_tokens * kHd8; i8 += blockDim.x) {
      const int t = i8 / kHd8;
      const int c = i8 - t * kHd8;
      const int base = cache_index(phys_row0 + t, kv_head, 0, num_kv_heads, HeadDim);
      kv4[i8] = reinterpret_cast<const int4*>(v_pool + base)[c];
    }
    __syncthreads();
#pragma unroll
    for (int i = 0; i < kOutPerLane; ++i) {
      const int d = lane + i * 32;
      float o = 0.0f;
      for (int t = 0; t < tile_tokens; ++t)
        o += w_sh[warp_id * block_size + t] * __half2float(kv_tile[t * HeadDim + d]);
      acc[i] += o;
    }
  }

  // One partial per (query head, coarse chunk), in the layout the reduce consumes.
  const int chunk_index = (b * num_heads + q_head) * scratch_chunks + coarse;
#pragma unroll
  for (int i = 0; i < kOutPerLane; ++i) {
    const int d = lane + i * 32;
    chunk_o[static_cast<std::size_t>(chunk_index) * HeadDim + d] = acc[i];
  }
  if (lane == 0) {
    chunk_m[chunk_index] = running_m;
    chunk_l[chunk_index] = running_l;
  }
}

template <int HeadDim>
__global__ void attention_step_chunk_stats_gqa_split_batched_kernel(
    const half* q, const half* k_pool, const half* v_pool, const int* __restrict__ block_tables,
    const int* __restrict__ seq_lens, int max_blocks, float* chunk_m, float* chunk_l,
    float* chunk_o, int num_heads, int num_kv_heads, int group_size, int block_size,
    int blocks_per_chunk, int scratch_chunks) {
  gqa_split_chunk_stats_batched_core<HeadDim>(
      q, k_pool, v_pool, block_tables, seq_lens, max_blocks, chunk_m, chunk_l, chunk_o, num_heads,
      num_kv_heads, group_size, block_size, blocks_per_chunk, scratch_chunks);
}

void launch_attention_step_batched_paged(const half* q, const half* k_pool, const half* v_pool,
                                         const int* block_tables, const int* seq_lens,
                                         int max_blocks, int max_seq_len, half* out, int batch,
                                         int num_heads, int num_kv_heads, int head_dim,
                                         int block_size, cudaStream_t stream, float* scratch_m,
                                         float* scratch_l, float* scratch_o, int scratch_chunks) {
  constexpr int warps = 4;
  constexpr int threads = warps * 32;
  // Sizing the grid from a host-side max_seq_len is the one thing keeping this
  // path out of a CUDA graph: capture freezes grid dimensions, so a grid that
  // depends on runtime length cannot be captured. Everything else here is
  // already device-side: seq_lens and block_tables are device pointers, the
  // kernel reads seq_lens[b] itself and guards `chunk_start >= seq_len`, and the
  // reduce derives its per-sequence chunk count from seq_lens too.
  //
  // The fixed grid is available but OFF by default, because unlike the
  // single-stream case it is NOT free here. This grid is heads x chunks x batch,
  // so max-sizing it multiplies by the batch as well: at B=32 with a short
  // context that is ~65k blocks, nearly all of which exist only to hit the guard
  // and return. Measured against the length-dependent grid at B=1/8/32/64:
  // +7.6 / -0.4 / -9.3 / +3.2%, medians of three; noisy, but the loss at 32 has
  // a mechanism behind it and the wins do not.
  //
  // So one always-max graph is the wrong shape for this path. What fits is
  // bucketed capture: a small set of graphs for length ranges, each with a grid
  // fixed to its bucket's ceiling, re-captured when a sequence outgrows it. That
  // keeps the grid honest AND capturable. CPI_ATTN_FIXED_GRID=1 turns on the
  // always-max form for anyone building that.
  static const bool fixed_grid = [] {
    const char* e = std::getenv("CPI_ATTN_FIXED_GRID");
    return e && *e == '1';
  }();
  const int total_blocks =
      fixed_grid ? scratch_chunks
                 : min(scratch_chunks, (max_seq_len + block_size - 1) / block_size);

  // GQA-shared path: one block per (KV head, chunk, sequence) with group_size
  // warps sharing each KV tile, cutting KV traffic by group_size. Requires a real
  // GQA group, block_size <= 32 (lane owns one token), and group_size <= 32 (warps
  // per block). K and V share one shared-memory tile (staged in disjoint phases),
  // so peak smem is a single block_size*HeadDim tile, small enough that both
  // head_dim 128 and 256 keep good occupancy.
  const int kv_hs = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = num_heads / kv_hs;
  const bool gqa_shared = num_kv_heads > 0 && group_size > 1 && (num_heads % kv_hs) == 0 &&
                          block_size <= 32 && group_size <= 32 &&
                          (head_dim == 128 || head_dim == 256);

  // Coarsen the GQA split at long context: each grid block sweeps blocks_per_chunk
  // paged blocks with a running online softmax, so decode runs on fewer, larger,
  // latency-hiding blocks instead of thousands of tiny 32-token ones (which left
  // batch-1 long-context attention latency-bound at ~14% of peak). Keep the grid
  // (num_kv_heads*chunks*batch) above a floor so SMs stay filled, and cap the
  // coarsening so no single block streams too serially.
  // Long context favours fewer, larger blocks (better latency hiding); short context keeps
  // roughly one block per chunk. Tuned empirically; see attention_decode_bench.
  const int blocks_per_chunk =
      gqa_shared ? pick_blocks_per_chunk(total_blocks, static_cast<long long>(num_kv_heads) * batch)
                 : 1;
  const int coarse_chunks = (total_blocks + blocks_per_chunk - 1) / blocks_per_chunk;
  // The reduce derives its per-sequence chunk count from this stride, so it must
  // equal the coarse chunk's token span (ceil(ceil(s/b)/G) == ceil(s/(b*G))).
  const int reduce_chunk_size = block_size * blocks_per_chunk;

  if (gqa_shared) {
    const int threads_g = group_size * 32;
    const std::size_t smem_g = (static_cast<std::size_t>(group_size) * head_dim +
                                static_cast<std::size_t>(block_size) * head_dim) *
                                   sizeof(half) +
                               static_cast<std::size_t>(group_size) * block_size * sizeof(float);
    const dim3 grid_g(num_kv_heads, coarse_chunks, batch);
    if (head_dim == 128) {
      attention_step_chunk_stats_gqa_split_batched_kernel<128>
          <<<grid_g, threads_g, smem_g, stream>>>(q, k_pool, v_pool, block_tables, seq_lens,
                                                  max_blocks, scratch_m, scratch_l, scratch_o,
                                                  num_heads, num_kv_heads, group_size, block_size,
                                                  blocks_per_chunk, scratch_chunks);
    } else {
      attention_step_chunk_stats_gqa_split_batched_kernel<256>
          <<<grid_g, threads_g, smem_g, stream>>>(q, k_pool, v_pool, block_tables, seq_lens,
                                                  max_blocks, scratch_m, scratch_l, scratch_o,
                                                  num_heads, num_kv_heads, group_size, block_size,
                                                  blocks_per_chunk, scratch_chunks);
    }
  } else {
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                             static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                             static_cast<std::size_t>(warps * head_dim) * sizeof(half);
    const dim3 grid(num_heads, total_blocks, batch);
    attention_step_chunk_stats_batched_kernel<warps><<<grid, threads, smem, stream>>>(
        q, k_pool, v_pool, block_tables, seq_lens, max_blocks, scratch_m, scratch_l, scratch_o,
        num_heads, num_kv_heads, head_dim, block_size, scratch_chunks);
  }

  const dim3 rgrid(num_heads, batch);
  attention_step_chunk_reduce_batched_kernel<<<rgrid, threads, 0, stream>>>(
      scratch_m, scratch_l, scratch_o, out, seq_lens, num_heads, head_dim, reduce_chunk_size,
      scratch_chunks);
}

void launch_attention_step_device_pos(const half* q, const half* k_cache, const half* v_cache,
                                      half* out, const int* position, int num_heads,
                                      int num_kv_heads, int head_dim, cudaStream_t stream,
                                      float* scratch_m, float* scratch_l, float* scratch_o,
                                      int scratch_chunks, bool allow_split, int window, bool* emitted_q8) {
  // Cached once: getenv on every launch (per layer, per token) is needless work and, more to the
  // point, is not required to be thread-safe against a concurrent setenv in a threaded server.
  static const bool force_fallback = [] {
    return std::getenv("CPI_FORCE_FALLBACK_DECODE_ATTENTION") != nullptr;
  }();
  if (force_fallback) {
    constexpr int threads = 128;
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                             static_cast<std::size_t>(threads + 3) * sizeof(float);
    attention_step_kernel_fallback_device_pos<<<num_heads, threads, smem, stream>>>(
        q, k_cache, v_cache, out, position, num_heads, num_kv_heads, head_dim, window);
    return;
  }
  // The split-K and GQA-fused device-pos kernels don't support a sliding window
  // (they're head_dim==128 paths; Gemma's windowed layers are head_dim 256/512 and
  // pass no scratch, so they never reach here). Force the tiled/fallback path,
  // which honours the window, whenever one is requested.
  if (window > 0) {
    allow_split = false;
    scratch_m = scratch_l = scratch_o = nullptr;
  }
  // GQA fused dispatch (device-position variant for CUDA graph capture). The
  // graphed decode path can't gate on runtime seq_len (position is device-side),
  // so when the split-K path is available it takes priority for GQA models:
  // split-K parallelizes over KV chunks (fixed num_heads×scratch_chunks grid
  // with a per-chunk seq guard, so it stays graph-safe as the sequence grows)
  // and avoids the GQA-fused kernel's num_kv_heads-block serial scan that
  // dominated long-context decode. GQA-fused stays as the fallback when split is
  // disabled (--no-split-attention).
  // Same head_dim widening as the host-position launcher. This is the launcher the CUDA
  // decode GRAPH captures, so it is the one that actually runs during generation
  // widening only the host-position one changed nothing measurable.
  const bool head_dim_fast = (head_dim == 64 || head_dim == 128);
  const bool split_available =
      allow_split && scratch_m && scratch_l && scratch_o && scratch_chunks > 0 && head_dim_fast;
  if (window <= 0 && !split_available && num_kv_heads > 0 && num_heads > num_kv_heads &&
      (num_heads % num_kv_heads) == 0 && head_dim_fast) {
    const int group_size_val = num_heads / num_kv_heads;
    if (group_size_val <= 32) {
      const int threads_gqa = group_size_val * 32;
      const std::size_t smem_gqa = static_cast<std::size_t>(group_size_val + 1) *
                                       static_cast<std::size_t>(head_dim) * sizeof(half) +
                                   static_cast<std::size_t>(group_size_val) * sizeof(float);
      if (head_dim == 64) {
        gqa_decode_kernel_device_pos<64><<<num_kv_heads, threads_gqa, smem_gqa, stream>>>(
            q, k_cache, v_cache, out, position, num_heads, num_kv_heads, group_size_val);
        return;
      }
      gqa_decode_kernel_device_pos<128><<<num_kv_heads, threads_gqa, smem_gqa, stream>>>(
          q, k_cache, v_cache, out, position, num_heads, num_kv_heads, group_size_val);
      return;
    }
  }

  constexpr int split_chunk_size = 32;
  if (allow_split && scratch_m && scratch_l && scratch_o && scratch_chunks > 0 && head_dim_fast) {
    constexpr int warps = 4;
    constexpr int threads = warps * 32;
    // GQA models: share each K/V read across the query-head group (grid.x =
    // num_kv_heads, group_size warps) while split-K over KV chunks (grid.y)
    // keeps occupancy. Graph-safe: fixed grid, per-chunk seq guard.
    const bool gqa = num_kv_heads > 0 && num_heads > num_kv_heads &&
                     (num_heads % num_kv_heads) == 0 && (num_heads / num_kv_heads) * 32 <= 1024;
    int reduce_chunk_size = split_chunk_size;
    if (gqa) {
      const int group_size_val = num_heads / num_kv_heads;
      // Graph capture can't see runtime seq_len (device-side position), so the
      // coarsening factor is fixed from scratch_chunks (the max). The per-sub-chunk
      // seq guard keeps it correct as the sequence grows; the reduce derives its
      // count from the coarse chunk_size and the device position.
      const int G = pick_blocks_per_chunk(scratch_chunks, num_kv_heads);
      const int coarse_chunks = (scratch_chunks + G - 1) / G;
      reduce_chunk_size = split_chunk_size * G;
      const std::size_t smem_gqa =
          static_cast<std::size_t>(group_size_val + 2 * split_chunk_size) * head_dim *
              sizeof(half) +
          static_cast<std::size_t>(group_size_val) * split_chunk_size * sizeof(float);
      const dim3 grid(num_kv_heads, coarse_chunks);
      if (head_dim == 64) {
        attention_step_chunk_stats_gqa_split_device_pos_kernel<64>
            <<<grid, group_size_val * 32, smem_gqa, stream>>>(
                q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, position, num_kv_heads,
                group_size_val, split_chunk_size, G, scratch_chunks);
      } else {
        attention_step_chunk_stats_gqa_split_device_pos_kernel<128>
            <<<grid, group_size_val * 32, smem_gqa, stream>>>(
                q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, position, num_kv_heads,
                group_size_val, split_chunk_size, G, scratch_chunks);
      }
    } else {
      const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                               static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                               static_cast<std::size_t>(warps * head_dim) * sizeof(half);
      const dim3 grid(num_heads, scratch_chunks);
      attention_step_chunk_stats_device_pos_kernel<warps><<<grid, threads, smem, stream>>>(
          q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, position, num_heads, num_kv_heads,
          head_dim, split_chunk_size, scratch_chunks);
    }
    // The reduce produces the output projection's activation, so it can leave the
    // q8_1 form behind and save the layer a quantize launch. Only when the caller
    // asked, since it claims the shared activation scratch.
    std::int8_t* q8 = nullptr;
    float* q8_scale = nullptr;
    float* q8_sum = nullptr;
    if (emitted_q8 != nullptr && (head_dim % 32) == 0 &&
        acquire_kquant_q8_1_scratch(num_heads * head_dim, &q8, &q8_scale, &q8_sum)) {
      *emitted_q8 = true;
    }
    attention_step_chunk_reduce_device_pos_kernel<<<num_heads, threads, 0, stream>>>(
        scratch_m, scratch_l, scratch_o, out, position, num_heads, head_dim, reduce_chunk_size,
        scratch_chunks, q8, q8_scale, q8_sum);
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
          q, k_cache, v_cache, out, position, num_heads, num_kv_heads, head_dim, window);
      return;
    }

    constexpr int warps = 4;
    constexpr int threads = warps * 32;
    const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                             static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                             static_cast<std::size_t>(warps * head_dim) * sizeof(half);
    attention_step_kernel_tiled_device_pos<warps><<<num_heads, threads, smem, stream>>>(
        q, k_cache, v_cache, out, position, num_heads, num_kv_heads, head_dim, window);
    return;
  }

  constexpr int threads = 128;
  const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                           static_cast<std::size_t>(threads + 3) * sizeof(float);
  attention_step_kernel_fallback_device_pos<<<num_heads, threads, smem, stream>>>(
      q, k_cache, v_cache, out, position, num_heads, num_kv_heads, head_dim, window);
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

void launch_attention_split_any(const half* q, const half* k_cache, const half* v_cache, half* out,
                                int seq_len, int window, int num_heads, int num_kv_heads,
                                int head_dim, float* scratch_m, float* scratch_l, float* scratch_o,
                                int chunk_size, int scratch_chunks, cudaStream_t stream) {
  constexpr int warps = 4;      // merge-tile template arg (arithmetic order pinned)
  constexpr int threads = 256;  // score/merge thread count; runtime-derived in-kernel
  const int k_start = (window > 0 && seq_len > window) ? seq_len - window : 0;
  const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                           static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                           static_cast<std::size_t>(warps * head_dim) * sizeof(half);
  const dim3 grid(num_heads, scratch_chunks);
  attention_step_chunk_stats_any_kernel<warps><<<grid, threads, smem, stream>>>(
      q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, seq_len, num_heads, num_kv_heads,
      head_dim, chunk_size, scratch_chunks, k_start);
  attention_step_chunk_reduce_any_kernel<<<num_heads, threads, 0, stream>>>(
      scratch_m, scratch_l, scratch_o, out, seq_len, num_heads, head_dim, chunk_size,
      scratch_chunks, k_start);
}

void launch_attention_split_any_device_pos(const half* q, const half* k_cache, const half* v_cache,
                                           half* out, const int* position, int window,
                                           int num_heads, int num_kv_heads, int head_dim,
                                           float* scratch_m, float* scratch_l, float* scratch_o,
                                           int chunk_size, int scratch_chunks,
                                           cudaStream_t stream) {
  constexpr int warps = 4;      // merge-tile template arg (arithmetic order pinned)
  constexpr int threads = 256;  // score/merge thread count; runtime-derived in-kernel
  const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                           static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                           static_cast<std::size_t>(warps * head_dim) * sizeof(half);
  const dim3 grid(num_heads, scratch_chunks);
  attention_step_chunk_stats_any_device_pos_kernel<warps><<<grid, threads, smem, stream>>>(
      q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, position, num_heads, num_kv_heads,
      head_dim, chunk_size, scratch_chunks, window);
  attention_step_chunk_reduce_any_device_pos_kernel<<<num_heads, threads, 0, stream>>>(
      scratch_m, scratch_l, scratch_o, out, position, num_heads, head_dim, chunk_size,
      scratch_chunks, window);
}

void launch_attention_split_any_mt_device_pos(const half* q, const half* k_cache,
                                              const half* v_cache, half* out, const int* position,
                                              int tokens, int window, int num_heads,
                                              int num_kv_heads, int head_dim, float* scratch_m,
                                              float* scratch_l, float* scratch_o, int chunk_size,
                                              int scratch_chunks, cudaStream_t stream) {
  constexpr int warps = 4;      // merge-tile template arg (arithmetic order pinned)
  constexpr int threads = 256;  // score/merge thread count; runtime-derived in-kernel
  const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                           static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                           static_cast<std::size_t>(warps * head_dim) * sizeof(half);
  const dim3 grid(num_heads, scratch_chunks, tokens);
  attention_step_chunk_stats_any_mt_device_pos_kernel<warps><<<grid, threads, smem, stream>>>(
      q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, position, num_heads, num_kv_heads,
      head_dim, chunk_size, scratch_chunks, window);
  const dim3 rgrid(num_heads, tokens);
  attention_step_chunk_reduce_any_mt_device_pos_kernel<<<rgrid, threads, 0, stream>>>(
      scratch_m, scratch_l, scratch_o, out, position, num_heads, head_dim, chunk_size,
      scratch_chunks, window);
}

void launch_store_kv_seq_device_pos(const half* k, const half* v, half* k_cache, half* v_cache,
                                    const int* position, int kv_hidden, int rows, int max_context,
                                    cudaStream_t stream) {
  const bool aligned =
      ((reinterpret_cast<std::uintptr_t>(k) | reinterpret_cast<std::uintptr_t>(v) |
        reinterpret_cast<std::uintptr_t>(k_cache) | reinterpret_cast<std::uintptr_t>(v_cache)) &
       (alignof(int4) - 1)) == 0;
  if ((kv_hidden & 7) == 0 && aligned) {
    const int kv_hidden_vec8 = kv_hidden / 8;
    const int threads = choose_copy_threads(kv_hidden);
    const dim3 blocks((kv_hidden_vec8 + threads - 1) / threads, rows);
    store_kv_seq_device_pos_vec8_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const int4*>(k), reinterpret_cast<const int4*>(v),
        reinterpret_cast<int4*>(k_cache), reinterpret_cast<int4*>(v_cache), position,
        kv_hidden_vec8, max_context);
    return;
  }
  constexpr int threads = 256;
  const dim3 blocks((kv_hidden + threads - 1) / threads, rows);
  store_kv_seq_device_pos_kernel<<<blocks, threads, 0, stream>>>(k, v, k_cache, v_cache, position,
                                                                 kv_hidden, max_context);
}

// Tree-verify attention: K draft-tree rows attend the committed cache
// [0, *cache_len) plus the in-batch scratch rows their ancestor bitmask admits
// (bit j of anc_mask[row] = scratch row j visible; a row's own bit is set).
// Draft siblings share a position, so in-batch K/V cannot go through the
// sequential cache-store path; they live in a [rows, kv_hidden] scratch and
// causality is the mask, not position order. One block per (head, row): with
// rows x heads blocks (e.g. 11 x 32) the grid fills the GPU without the
// split-K machinery, and the tiled main loop matches attention_step_kernel's.
// cache_len is read from the device so the fixed-shape launch is
// graph-capturable.
template <int WarpsPerBlock>
__global__ void attention_tree_masked_kernel(
    const half* __restrict__ q, const half* __restrict__ k_cache, const half* __restrict__ v_cache,
    const half* __restrict__ k_scratch, const half* __restrict__ v_scratch, half* __restrict__ out,
    const int* __restrict__ cache_len_ptr, const unsigned int* __restrict__ anc_mask, int rows,
    int num_heads, int num_kv_heads, int head_dim, int q_row_stride) {
  extern __shared__ float smem[];
  float* q_shared = smem;                              // [head_dim]
  float* score_shared = q_shared + head_dim;           // [WarpsPerBlock]
  float* beta_shared = score_shared + WarpsPerBlock;   // [WarpsPerBlock]
  float* stats_shared = beta_shared + WarpsPerBlock;   // [running_m, running_l, tile_m, tile_l]
  half* v_tile = reinterpret_cast<half*>(stats_shared + 4);  // [WarpsPerBlock, head_dim]

  const int head = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane = tid % warpSize;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head =
      ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const int head_pairs = head_dim / 2;
  const int cache_len = cache_len_ptr[0];
  const unsigned int mask = anc_mask[row];

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = __half2float(q[row * q_row_stride + head * head_dim + d]);
  }
  if (tid == 0) {
    stats_shared[0] = -1.0e30f;  // running_m
    stats_shared[1] = 0.0f;      // running_l
  }
  __syncthreads();

  float acc[2];  // head_dim <= 2 * blockDim elements per thread
  acc[0] = 0.0f;
  acc[1] = 0.0f;
  // total_cols: cache columns then the rows in-batch scratch columns; scratch
  // column j is dead unless the mask admits it.
  const int total_cols = cache_len + rows;
  for (int tile_base = 0; tile_base < total_cols; tile_base += WarpsPerBlock) {
    const int tile_tokens = min(WarpsPerBlock, total_cols - tile_base);

    // Phase 1a: one warp scores one column of the tile.
    {
      const int t = tile_base + warp_id;
      float score = -1.0e30f;
      if (warp_id < tile_tokens) {
        const half* k_src = nullptr;
        if (t < cache_len) {
          k_src = k_cache + cache_index(t, kv_head, 0, num_kv_heads, head_dim);
        } else if ((mask >> (t - cache_len)) & 1u) {
          k_src = k_scratch + cache_index(t - cache_len, kv_head, 0, num_kv_heads, head_dim);
        }
        if (k_src != nullptr) {
          const half2* k2 = reinterpret_cast<const half2*>(k_src);
          float partial = 0.0f;
          for (int pair = lane; pair < head_pairs; pair += warpSize) {
            const float2 qv = make_float2(q_shared[2 * pair], q_shared[2 * pair + 1]);
            const float2 kv = __half22float2(k2[pair]);
            partial += qv.x * kv.x + qv.y * kv.y;
          }
          score = warp_sum(partial) * scale;
        }
      }
      if (lane == 0 && warp_id < tile_tokens) {
        score_shared[warp_id] = score;
      }
    }

    // Phase 1b: stage the tile's V rows (masked-out columns stage zeros; their
    // softmax weight underflows to 0 through the -1e30 score anyway).
    for (int i = 0; i < tile_tokens; ++i) {
      const int t = tile_base + i;
      const half* v_src = nullptr;
      if (t < cache_len) {
        v_src = v_cache + cache_index(t, kv_head, 0, num_kv_heads, head_dim);
      } else if ((mask >> (t - cache_len)) & 1u) {
        v_src = v_scratch + cache_index(t - cache_len, kv_head, 0, num_kv_heads, head_dim);
      }
      for (int d = tid; d < head_dim; d += blockDim.x) {
        v_tile[i * head_dim + d] = (v_src != nullptr) ? v_src[d] : __float2half(0.0f);
      }
    }
    __syncthreads();

    // Phase 2: tile softmax stats (serial over <= WarpsPerBlock entries).
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

    // Phase 3: merge the tile into the running accumulator.
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
    out[row * q_row_stride + head * head_dim + d] = __float2half(acc[j] * inv_l);
  }
}

void launch_attention_tree_split(const half* q, const half* k_cache, const half* v_cache,
                                 const half* k_scratch, const half* v_scratch, half* out,
                                 const int* cache_len, const unsigned int* anc_mask, int rows,
                                 int scr_cols, int num_heads, int num_kv_heads, int head_dim,
                                 float* scratch_m, float* scratch_l, float* scratch_o,
                                 int chunk_size, int scratch_chunks, cudaStream_t stream) {
  constexpr int warps = 4;
  constexpr int threads = 256;
  const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                           static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                           static_cast<std::size_t>(warps * head_dim) * sizeof(half);
  // Cache chunks (fixed grid, per-chunk guards; the virtual in-batch chunk
  // needs one spare slot, so pass scratch_chunks - 1 as the cache grid).
  const dim3 grid(num_heads, scratch_chunks - 1, rows);
  attention_tree_chunk_stats_kernel<warps><<<grid, threads, smem, stream>>>(
      q, k_cache, v_cache, scratch_m, scratch_l, scratch_o, cache_len, num_heads, num_kv_heads,
      head_dim, chunk_size, scratch_chunks);
  const dim3 bgrid(num_heads, rows);
  const std::size_t bsmem =
      static_cast<std::size_t>(head_dim) * sizeof(float) + scr_cols * sizeof(float);
  attention_tree_inbatch_stats_kernel<<<bgrid, 128, bsmem, stream>>>(
      q, k_scratch, v_scratch, scratch_m, scratch_l, scratch_o, cache_len, anc_mask, scr_cols,
      num_heads, num_kv_heads, head_dim, chunk_size, scratch_chunks);
  const dim3 rgrid(num_heads, rows);
  attention_tree_chunk_reduce_kernel<<<rgrid, threads, 0, stream>>>(
      scratch_m, scratch_l, scratch_o, out, cache_len, num_heads, head_dim, chunk_size,
      scratch_chunks);
}

void launch_attention_tree_masked(const half* q, const half* k_cache, const half* v_cache,
                                  const half* k_scratch, const half* v_scratch, half* out,
                                  const int* cache_len, const unsigned int* anc_mask, int rows,
                                  int num_heads, int num_kv_heads, int head_dim, int q_row_stride,
                                  cudaStream_t stream) {
  constexpr int kWarps = 4;
  const dim3 grid(num_heads, rows);
  const int threads = kWarps * 32;
  const std::size_t smem_bytes = static_cast<std::size_t>(head_dim) * sizeof(float) +
                                 (2 * kWarps + 4) * sizeof(float) +
                                 static_cast<std::size_t>(kWarps) * head_dim * sizeof(half);
  attention_tree_masked_kernel<kWarps><<<grid, threads, smem_bytes, stream>>>(
      q, k_cache, v_cache, k_scratch, v_scratch, out, cache_len, anc_mask, rows, num_heads,
      num_kv_heads, head_dim, q_row_stride);
}

}  // namespace kernels
