// kernels_attention_decode_int4.cu
//
// CUDA kernels and host launch wrappers for INT4 decode-time attention paths.

#include "runtime/kernels.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <sm_61_intrinsics.h>

namespace kernels {
namespace {

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


template <int WarpsPerBlock>
__global__ void attention_step_chunk_stats_int4_kernel(const half*   q,
                                                        const int8_t* k_cache_i4,
                                                        const int8_t* v_cache_i4,
                                                        const half*   k_scales,
                                                        const half*   v_scales,
                                                        float*        chunk_m,
                                                        float*        chunk_l,
                                                        float*        chunk_o,
                                                        int           seq_len,
                                                        int           num_heads,
                                                        int           num_kv_heads,
                                                        int           head_dim,
                                                        int           chunk_size,
                                                        int           scratch_chunks) {
  extern __shared__ unsigned char smem_bytes[];
  half*  q_shared     = reinterpret_cast<half*>(smem_bytes);
  float* score_shared = reinterpret_cast<float*>(q_shared + head_dim);
  float* beta_shared  = score_shared + WarpsPerBlock;
  float* stats_shared = beta_shared + WarpsPerBlock;  // [running_m, running_l, tile_m, tile_l]
  half*  v_tile       = reinterpret_cast<half*>(stats_shared + 4);

  const int head        = blockIdx.x;
  const int chunk       = blockIdx.y;
  const int chunk_start = chunk * chunk_size;
  if (chunk_start >= seq_len) return;
  const int chunk_end = min(chunk_start + chunk_size, seq_len);

  const int tid      = threadIdx.x;
  const int warp_id  = tid / warpSize;
  const int lane     = tid % warpSize;
  const float scale  = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size    = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head = ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const int packed_per_head = head_dim / 2;
  const int chunk_index     = head * scratch_chunks + chunk;

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

    // Phase 1a: INT4 K dot-product — each warp handles one tile token.
    {
      const int t = tile_base + warp_id;
      float score = neg_inf<float>();
      if (warp_id < tile_tokens) {
        const float kscale  = __half2float(k_scales[t * num_kv_heads + kv_head]);
        const int8_t* k_i4 = k_cache_i4 + (t * num_kv_heads + kv_head) * packed_per_head;
        float partial = 0.0f;
        for (int i = lane; i < packed_per_head; i += warpSize) {
          const int8_t b = k_i4[i];
          const float k0 = static_cast<float>(((int)b << 28) >> 28) * kscale;
          const float k1 = static_cast<float>((int)b >> 4)          * kscale;
          partial += __half2float(q_shared[2 * i])     * k0;
          partial += __half2float(q_shared[2 * i + 1]) * k1;
        }
        score = warp_sum(partial) * scale;
      }
      if (lane == 0 && warp_id < tile_tokens) {
        score_shared[warp_id] = score;
      }
    }

    // Phase 1b: decode INT4 V into shared v_tile.
    for (int i = 0; i < tile_tokens; ++i) {
      const int t         = tile_base + i;
      const float vscale  = __half2float(v_scales[t * num_kv_heads + kv_head]);
      const int8_t* v_i4  = v_cache_i4 + (t * num_kv_heads + kv_head) * packed_per_head;
      half* vt = v_tile + i * head_dim;
      for (int d = tid; d < head_dim; d += blockDim.x) {
        const int8_t b = v_i4[d >> 1];
        const float vval = static_cast<float>((d & 1) ? ((int)b >> 4)
                                                       : (((int)b << 28) >> 28)) * vscale;
        vt[d] = __float2half(vval);
      }
    }
    __syncthreads();

    // Phase 2: tile softmax (thread 0 only).
    if (tid == 0) {
      float tile_m = neg_inf<float>();
      for (int i = 0; i < tile_tokens; ++i) tile_m = fmaxf(tile_m, score_shared[i]);
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

    // Phase 3: accumulate V and merge online softmax stats.
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

  // Write partial stats and output to scratch buffers.
  if (tid == 0) {
    chunk_m[chunk_index] = stats_shared[0];
    chunk_l[chunk_index] = stats_shared[1];
  }
  for (int d = tid; d < head_dim; d += blockDim.x) {
    chunk_o[static_cast<std::size_t>(chunk_index) * static_cast<std::size_t>(head_dim) +
            static_cast<std::size_t>(d)] = acc;
  }
}

// Quantize and store one K/V pair as packed INT4 with per-head scales.
// Grid: dim3(num_kv_heads).  Block: dim3(head_dim).
// Shared memory layout: [k_warp_max, v_warp_max, k_scale_bcast, v_scale_bcast]
//   = (2*num_warps + 2) floats.
__global__ void store_kv_int4_kernel(const half* k,
                                     const half* v,
                                     int8_t*     k_cache_i4,
                                     int8_t*     v_cache_i4,
                                     half*       k_scales,
                                     half*       v_scales,
                                     int         position,
                                     int         num_kv_heads,
                                     int         head_dim,
                                     int         max_context) {
  extern __shared__ float smem[];
  const int kv_head  = blockIdx.x;
  const int tid      = threadIdx.x;
  const int warp_id  = tid / 32;
  const int lane     = tid % 32;
  const int num_warps = blockDim.x / 32;

  if (position < 0 || position >= max_context) {
    return;
  }

  const int head_base = kv_head * head_dim;

  // Load one element per thread; compute absolute value for absmax.
  const float kval = __half2float(k[head_base + tid]);
  const float vval = __half2float(v[head_base + tid]);
  float kabs = fabsf(kval);
  float vabs = fabsf(vval);

  // Warp-level max via shuffle.
  for (int off = 16; off > 0; off >>= 1) {
    kabs = fmaxf(kabs, __shfl_down_sync(0xffffffffu, kabs, off));
    vabs = fmaxf(vabs, __shfl_down_sync(0xffffffffu, vabs, off));
  }

  float* k_warp = smem;                  // [num_warps]
  float* v_warp = smem + num_warps;      // [num_warps]
  float* scales = smem + 2 * num_warps;  // [2]: broadcast k_scale, v_scale

  if (lane == 0) {
    k_warp[warp_id] = kabs;
    v_warp[warp_id] = vabs;
  }
  __syncthreads();

  // Thread 0 reduces across warps, computes and broadcasts scales.
  if (tid == 0) {
    float km = 0.0f, vm = 0.0f;
    for (int i = 0; i < num_warps; ++i) {
      km = fmaxf(km, k_warp[i]);
      vm = fmaxf(vm, v_warp[i]);
    }
    const float ks = (km > 0.0f) ? (km / 7.0f) : 1.0f;
    const float vs = (vm > 0.0f) ? (vm / 7.0f) : 1.0f;
    scales[0] = ks;
    scales[1] = vs;
    const int si = position * num_kv_heads + kv_head;
    k_scales[si] = __float2half(ks);
    v_scales[si] = __float2half(vs);
  }
  __syncthreads();

  // Each of the first head_dim/2 threads packs two adjacent fp16 values into
  // one signed-nibble byte and writes it to the INT4 cache.
  const int packed_per_head = head_dim / 2;
  if (tid < packed_per_head) {
    const float ks = scales[0];
    const float vs = scales[1];
    const float k0 = __half2float(k[head_base + 2 * tid]);
    const float k1 = __half2float(k[head_base + 2 * tid + 1]);
    const float v0 = __half2float(v[head_base + 2 * tid]);
    const float v1 = __half2float(v[head_base + 2 * tid + 1]);
    const int ki0 = max(-8, min(7, __float2int_rn(k0 / ks)));
    const int ki1 = max(-8, min(7, __float2int_rn(k1 / ks)));
    const int vi0 = max(-8, min(7, __float2int_rn(v0 / vs)));
    const int vi1 = max(-8, min(7, __float2int_rn(v1 / vs)));
    const int out = (position * num_kv_heads + kv_head) * packed_per_head + tid;
    k_cache_i4[out] = static_cast<int8_t>((ki0 & 0xF) | ((ki1 & 0xF) << 4));
    v_cache_i4[out] = static_cast<int8_t>((vi0 & 0xF) | ((vi1 & 0xF) << 4));
  }
}

// Tiled decode attention over INT4 KV cache.
// Structurally identical to attention_step_kernel_tiled_device_pos except:
//   - Phase 1a reads INT4 packed K bytes + per-head scale instead of fp16.
//   - Phase 1b decodes packed INT4 V bytes to fp16 into shared memory.
//   - All subsequent phases (softmax, accumulate, normalise) are unchanged.
template <int WarpsPerBlock>
__global__ void attention_step_kernel_int4(const half*   q,
                                           const int8_t* k_cache_i4,
                                           const int8_t* v_cache_i4,
                                           const half*   k_scales,
                                           const half*   v_scales,
                                           half*         out,
                                           int           seq_len,
                                           int           num_heads,
                                           int           num_kv_heads,
                                           int           head_dim) {
  const int packed_per_head = head_dim / 2;
  extern __shared__ unsigned char smem_bytes[];
  half*  q_shared     = reinterpret_cast<half*>(smem_bytes);
  float* score_shared = reinterpret_cast<float*>(q_shared + head_dim);
  float* beta_shared  = score_shared + WarpsPerBlock;
  float* stats_shared = beta_shared + WarpsPerBlock;  // [running_m, running_l, tile_m, tile_l]
  half*  v_tile       = reinterpret_cast<half*>(stats_shared + 4);

  const int head     = blockIdx.x;
  const int tid      = threadIdx.x;
  const int warp_id  = tid / warpSize;
  const int lane     = tid % warpSize;
  const float scale  = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head = ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);

  // Load Q into shared memory.
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

    // Phase 1a: each warp computes K·Q score for its tile token via INT4 K.
    {
      const int t = tile_base + warp_id;
      float score = -1.0e30f;
      if (warp_id < tile_tokens) {
        const float kscale = __half2float(k_scales[t * num_kv_heads + kv_head]);
        const int8_t* k_i4 = k_cache_i4 + (t * num_kv_heads + kv_head) * packed_per_head;
        float partial = 0.0f;
        for (int i = lane; i < packed_per_head; i += warpSize) {
          const int8_t b = k_i4[i];
          // Sign-extend low nibble (bits 3:0) and high nibble (bits 7:4).
          const float k0 = static_cast<float>(((int)b << 28) >> 28) * kscale;
          const float k1 = static_cast<float>((int)b >> 4)          * kscale;
          partial += __half2float(q_shared[2 * i])     * k0;
          partial += __half2float(q_shared[2 * i + 1]) * k1;
        }
        score = warp_sum(partial) * scale;
      }
      if (lane == 0 && warp_id < tile_tokens) {
        score_shared[warp_id] = score;
      }
    }

    // Phase 1b: stage V tile — decode INT4 packed bytes to fp16 in shared mem.
    {
      for (int i = 0; i < tile_tokens; ++i) {
        const int t = tile_base + i;
        const float vscale = __half2float(v_scales[t * num_kv_heads + kv_head]);
        const int8_t* v_i4 = v_cache_i4 + (t * num_kv_heads + kv_head) * packed_per_head;
        half* vt = v_tile + i * head_dim;
        for (int d = tid; d < head_dim; d += blockDim.x) {
          const int8_t b = v_i4[d >> 1];
          // Select low or high nibble depending on element parity.
          const float vval = static_cast<float>((d & 1) ? ((int)b >> 4)
                                                        : (((int)b << 28) >> 28)) * vscale;
          vt[d] = __float2half(vval);
        }
      }
    }
    __syncthreads();

    // Phase 2: tile-local softmax weights (identical to fp16 kernel).
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

    // Phase 3: accumulate V weighted by betas, merge tile stats (identical).
    {
      const float tile_m    = stats_shared[2];
      const float tile_l    = stats_shared[3];
      const float running_m = stats_shared[0];
      const float running_l = stats_shared[1];
      const float new_m     = fmaxf(running_m, tile_m);
      const float c_prev    = (running_l == 0.0f) ? 0.0f : expf(running_m - new_m);
      const float c_tile    = expf(tile_m - new_m);
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


}  // namespace

// Host launch wrappers for INT4 decode attention paths.
void launch_store_kv_int4(const half*  k,
                           const half*  v,
                           int8_t*      k_cache_i4,
                           int8_t*      v_cache_i4,
                           half*        k_scales,
                           half*        v_scales,
                           int          position,
                           int          num_kv_heads,
                           int          head_dim,
                           int          max_context,
                           cudaStream_t stream) {
  const int num_warps = head_dim / 32;
  const std::size_t smem = static_cast<std::size_t>(2 * num_warps + 2) * sizeof(float);
  store_kv_int4_kernel<<<num_kv_heads, head_dim, smem, stream>>>(
      k, v, k_cache_i4, v_cache_i4, k_scales, v_scales,
      position, num_kv_heads, head_dim, max_context);
}

void launch_attention_step_int4(const half*   q,
                                 const int8_t* k_cache_i4,
                                 const int8_t* v_cache_i4,
                                 const half*   k_scales,
                                 const half*   v_scales,
                                 half*         out,
                                 int           seq_len,
                                 int           num_heads,
                                 int           num_kv_heads,
                                 int           head_dim,
                                 cudaStream_t  stream,
                                 float*        scratch_m,
                                 float*        scratch_l,
                                 float*        scratch_o,
                                 int           scratch_chunks,
                                 bool          allow_split) {
  constexpr int warps   = 4;
  constexpr int threads = warps * 32;
  const std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                            static_cast<std::size_t>(2 * warps + 4) * sizeof(float) +
                            static_cast<std::size_t>(warps * head_dim) * sizeof(half);

  // Split-K path: same chunk decomposition as the FP16 attention kernel.
  // Requires head_dim==128 (matches scratch_o element stride), seq_len>=64, and
  // allocated scratch buffers.
  constexpr int split_chunk_size = 32;
  if (allow_split && scratch_m && scratch_l && scratch_o && scratch_chunks > 0
      && head_dim == 128 && seq_len >= 64) {
    const int chunk_count = min(scratch_chunks, (seq_len + split_chunk_size - 1) / split_chunk_size);
    const dim3 grid(num_heads, chunk_count);
    attention_step_chunk_stats_int4_kernel<warps><<<grid, threads, smem, stream>>>(
        q, k_cache_i4, v_cache_i4, k_scales, v_scales,
        scratch_m, scratch_l, scratch_o,
        seq_len, num_heads, num_kv_heads, head_dim,
        split_chunk_size, scratch_chunks);
    // Reuse the FP16 reduce kernel — it only operates on float scratch, not the cache format.
    attention_step_chunk_reduce_kernel<<<num_heads, threads, 0, stream>>>(
        scratch_m, scratch_l, scratch_o, out,
        seq_len, num_heads, head_dim, split_chunk_size, scratch_chunks);
    return;
  }

  // Fallback: single-block serial path (correct for any head_dim, slow at long context).
  attention_step_kernel_int4<warps><<<num_heads, threads, smem, stream>>>(
      q, k_cache_i4, v_cache_i4, k_scales, v_scales, out,
      seq_len, num_heads, num_kv_heads, head_dim);
}


}  // namespace kernels
