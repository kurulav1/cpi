// kernels_attention_decode_int4.cu
//
// CUDA kernels and host launch wrappers for quantized-KV decode attention.
// K and V are stored per (token, kv_head) with an fp16 absmax scale each, at
// 4 or 8 bits per element (template parameters). K can optionally be rotated
// by a head_dim Walsh-Hadamard transform before quantization (QuaRot R3);
// the attention kernels then apply the same transform to Q at read time, so
// scores are computed in the rotated basis and remain invariant.
// The legacy int4 entry points (launch_store_kv_int4 / launch_attention_step_int4)
// are kept as wrappers over the 4-bit unrotated instantiation.

#include <cuda_fp16.h>
#include <sm_61_intrinsics.h>

#include <cstddef>
#include <cstdint>

#include "runtime/kernels.cuh"

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

// In-place Walsh-Hadamard butterfly over buf[0..n). Requires blockDim.x >= n,
// n a power of two, and buf fully populated and synced before the call.
// Unnormalized; the caller applies the 1/sqrt(n) factor.
__device__ __forceinline__ void fwht_shared(float* buf, int tid, int n) {
  for (int len = 1; len < n; len <<= 1) {
    float a = 0.0f, b = 0.0f;
    if (tid < n) {
      a = buf[tid];
      b = buf[tid ^ len];
    }
    __syncthreads();
    if (tid < n) {
      buf[tid] = (tid & len) ? (b - a) : (a + b);
    }
    __syncthreads();
  }
}

// Dequantize element d of a K/V row stored at `row` (int4 packed nibbles or int8).
template <int Bits>
__device__ __forceinline__ float kv_load(const int8_t* row, int d, float scale);

template <>
__device__ __forceinline__ float kv_load<4>(const int8_t* row, int d, float scale) {
  const int8_t b = row[d >> 1];
  return static_cast<float>((d & 1) ? ((int)b >> 4) : (((int)b << 28) >> 28)) * scale;
}

template <>
__device__ __forceinline__ float kv_load<8>(const int8_t* row, int d, float scale) {
  return static_cast<float>(row[d]) * scale;
}

__global__ void attention_step_chunk_reduce_kernel(const float* chunk_m, const float* chunk_l,
                                                   const float* chunk_o, half* out, int seq_len,
                                                   int num_heads, int head_dim, int chunk_size,
                                                   int scratch_chunks) {
  __shared__ float scale_shared[3];
  const int head = blockIdx.x;
  const int tid = threadIdx.x;
  const int chunk_count = (seq_len + chunk_size - 1) / chunk_size;
  // Per-thread accumulator array: this reduce runs at any head_dim, and head_dim
  // can exceed blockDim (256 with 128 threads), so a scalar acc would conflate
  // the strided output dims.
  float acc[kAccPerThread];
#pragma unroll
  for (int i = 0; i < kAccPerThread; ++i) acc[i] = 0.0f;
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
    int j = 0;
    for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
      acc[j] = acc[j] * alpha + chunk_o[base + static_cast<std::size_t>(d)] * beta;
    }
    __syncthreads();
  }

  const float inv_l = 1.0f / fmaxf(scale_shared[2], 1e-8f);
  int j = 0;
  for (int d = tid; d < head_dim; d += blockDim.x, ++j) {
    out[head * head_dim + d] = __float2half(acc[j] * inv_l);
  }
}

// Quantize and store one K/V pair with per-head scales.
// Grid: dim3(num_kv_heads).  Block: dim3(head_dim).
// Shared memory: [head_dim k_buf][head_dim v_buf][num_warps k_warp][num_warps v_warp][2 scales].
// When sink/ring buffers are provided, the (rotated) fp16 values are also
// written there: sinks at absolute positions < sink_n (permanent), and every
// position into ring slot position % win_n. Attention reads these fp16 copies
// for sink and recent-window tokens instead of the quantized cache.
template <int KBits, int VBits, bool RotK>
__global__ void store_kv_quant_kernel(const half* k, const half* v, int8_t* k_cache,
                                      int8_t* v_cache, half* k_scales, half* v_scales, half* k_sink,
                                      half* v_sink, half* k_ring, half* v_ring, int sink_n,
                                      int win_n, int position, int num_kv_heads, int head_dim,
                                      int max_context) {
  extern __shared__ float smem[];
  const int kv_head = blockIdx.x;
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid % 32;
  const int num_warps = blockDim.x / 32;

  if (position < 0 || position >= max_context) {
    return;
  }

  float* k_buf = smem;
  float* v_buf = k_buf + head_dim;
  float* k_warp = v_buf + head_dim;
  float* v_warp = k_warp + num_warps;
  float* scales = v_warp + num_warps;

  const int head_base = kv_head * head_dim;
  k_buf[tid] = __half2float(k[head_base + tid]);
  v_buf[tid] = __half2float(v[head_base + tid]);
  __syncthreads();

  if (RotK) {
    fwht_shared(k_buf, tid, head_dim);
    k_buf[tid] *= rsqrtf(static_cast<float>(head_dim));
    __syncthreads();
  }

  if (win_n > 0 && k_ring != nullptr) {
    const int slot = position % win_n;
    const int ri = (slot * num_kv_heads + kv_head) * head_dim + tid;
    k_ring[ri] = __float2half(k_buf[tid]);
    v_ring[ri] = __float2half(v_buf[tid]);
  }
  if (position < sink_n && k_sink != nullptr) {
    const int si = (position * num_kv_heads + kv_head) * head_dim + tid;
    k_sink[si] = __float2half(k_buf[tid]);
    v_sink[si] = __float2half(v_buf[tid]);
  }

  float kabs = fabsf(k_buf[tid]);
  float vabs = fabsf(v_buf[tid]);
  for (int off = 16; off > 0; off >>= 1) {
    kabs = fmaxf(kabs, __shfl_down_sync(0xffffffffu, kabs, off));
    vabs = fmaxf(vabs, __shfl_down_sync(0xffffffffu, vabs, off));
  }
  if (lane == 0) {
    k_warp[warp_id] = kabs;
    v_warp[warp_id] = vabs;
  }
  __syncthreads();

  constexpr float kMaxQ = (KBits == 4) ? 7.0f : 127.0f;
  constexpr float vMaxQ = (VBits == 4) ? 7.0f : 127.0f;
  if (tid == 0) {
    float km = 0.0f, vm = 0.0f;
    for (int i = 0; i < num_warps; ++i) {
      km = fmaxf(km, k_warp[i]);
      vm = fmaxf(vm, v_warp[i]);
    }
    const float ks = (km > 0.0f) ? (km / kMaxQ) : 1.0f;
    const float vs = (vm > 0.0f) ? (vm / vMaxQ) : 1.0f;
    scales[0] = ks;
    scales[1] = vs;
    const int si = position * num_kv_heads + kv_head;
    k_scales[si] = __float2half(ks);
    v_scales[si] = __float2half(vs);
  }
  __syncthreads();

  const float ks = scales[0];
  const float vs = scales[1];
  if (KBits == 4) {
    const int packed = head_dim / 2;
    if (tid < packed) {
      const int q0 = max(-8, min(7, __float2int_rn(k_buf[2 * tid] / ks)));
      const int q1 = max(-8, min(7, __float2int_rn(k_buf[2 * tid + 1] / ks)));
      const int out = (position * num_kv_heads + kv_head) * packed + tid;
      k_cache[out] = static_cast<int8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
  } else {
    const int qi = max(-127, min(127, __float2int_rn(k_buf[tid] / ks)));
    k_cache[(position * num_kv_heads + kv_head) * head_dim + tid] = static_cast<int8_t>(qi);
  }
  if (VBits == 4) {
    const int packed = head_dim / 2;
    if (tid < packed) {
      const int q0 = max(-8, min(7, __float2int_rn(v_buf[2 * tid] / vs)));
      const int q1 = max(-8, min(7, __float2int_rn(v_buf[2 * tid + 1] / vs)));
      const int out = (position * num_kv_heads + kv_head) * packed + tid;
      v_cache[out] = static_cast<int8_t>((q0 & 0xF) | ((q1 & 0xF) << 4));
    }
  } else {
    const int qi = max(-127, min(127, __float2int_rn(v_buf[tid] / vs)));
    v_cache[(position * num_kv_heads + kv_head) * head_dim + tid] = static_cast<int8_t>(qi);
  }
}

// Rotate Q in shared memory to match a Hadamard-rotated K cache. Uses a float
// scratch region placed by the caller; blockDim.x must be >= head_dim.
__device__ __forceinline__ void rotate_q_shared(half* q_shared, float* qbuf, int tid,
                                                int head_dim) {
  for (int d = tid; d < head_dim; d += blockDim.x) qbuf[d] = __half2float(q_shared[d]);
  __syncthreads();
  fwht_shared(qbuf, tid, head_dim);
  const float rn = rsqrtf(static_cast<float>(head_dim));
  for (int d = tid; d < head_dim; d += blockDim.x) q_shared[d] = __float2half(qbuf[d] * rn);
  __syncthreads();
}

// Sink/window fp16 override: token index t is relative to the (possibly
// offset) cache base; attn_start restores the absolute position. Sink tokens
// (absolute < sink_n) and recent tokens (absolute >= seq_end - win_n) read the
// fp16 side buffers written by the store kernel instead of the quantized cache.
__device__ __forceinline__ const half* kv_fp16_source(const half* sink, const half* ring, int t_abs,
                                                      int seq_end_abs, int sink_n, int win_n,
                                                      int num_kv_heads, int kv_head, int head_dim) {
  if (t_abs < sink_n && sink != nullptr) {
    return sink + (t_abs * num_kv_heads + kv_head) * head_dim;
  }
  if (win_n > 0 && ring != nullptr && t_abs >= seq_end_abs - win_n) {
    return ring + ((t_abs % win_n) * num_kv_heads + kv_head) * head_dim;
  }
  return nullptr;
}

template <int WarpsPerBlock, int KBits, int VBits, bool RotK>
__global__ void attention_step_chunk_stats_quant_kernel(
    const half* q, const int8_t* k_cache, const int8_t* v_cache, const half* k_scales,
    const half* v_scales, const half* k_sink, const half* v_sink, const half* k_ring,
    const half* v_ring, int sink_n, int win_n, int attn_start, float* chunk_m, float* chunk_l,
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
  if (chunk_start >= seq_len) return;
  const int chunk_end = min(chunk_start + chunk_size, seq_len);

  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane = tid % warpSize;
  const float scale = rsqrtf(static_cast<float>(head_dim));
  const int kv_heads_safe = (num_kv_heads > 0) ? num_kv_heads : 1;
  const int group_size = ((num_heads / kv_heads_safe) > 0) ? (num_heads / kv_heads_safe) : 1;
  const int kv_head =
      ((head / group_size) < kv_heads_safe) ? (head / group_size) : (kv_heads_safe - 1);
  const int k_row = head_dim * KBits / 8;
  const int v_row = head_dim * VBits / 8;
  const int chunk_index = head * scratch_chunks + chunk;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[head * head_dim + d];
  }
  if (tid == 0) {
    stats_shared[0] = neg_inf<float>();
    stats_shared[1] = 0.0f;
  }
  __syncthreads();
  if (RotK) {
    float* qbuf = reinterpret_cast<float*>(v_tile + WarpsPerBlock * head_dim);
    rotate_q_shared(q_shared, qbuf, tid, head_dim);
  }

  static_assert(WarpsPerBlock * 32 * kAccPerThread >= kTiledMaxHeadDim,
                "quant decode accumulator too small for kTiledMaxHeadDim");
  float acc[kAccPerThread];
#pragma unroll
  for (int i = 0; i < kAccPerThread; ++i) acc[i] = 0.0f;
  for (int tile_base = chunk_start; tile_base < chunk_end; tile_base += WarpsPerBlock) {
    const int tile_tokens = min(WarpsPerBlock, chunk_end - tile_base);

    // Phase 1a: K dot-product; each warp handles one tile token, reading fp16
    // sink/window buffers where applicable and the quantized cache otherwise.
    {
      const int t = tile_base + warp_id;
      float score = neg_inf<float>();
      if (warp_id < tile_tokens) {
        const int t_abs = attn_start + t;
        const half* kf = kv_fp16_source(k_sink, k_ring, t_abs, attn_start + seq_len, sink_n, win_n,
                                        num_kv_heads, kv_head, head_dim);
        float partial = 0.0f;
        if (kf != nullptr) {
          for (int i = lane; i < head_dim; i += warpSize) {
            partial += __half2float(q_shared[i]) * __half2float(kf[i]);
          }
        } else if (KBits == 4) {
          const float kscale = __half2float(k_scales[t * num_kv_heads + kv_head]);
          const int8_t* k_q = k_cache + (t * num_kv_heads + kv_head) * k_row;
          for (int i = lane; i < head_dim / 2; i += warpSize) {
            const int8_t b = k_q[i];
            const float k0 = static_cast<float>(((int)b << 28) >> 28) * kscale;
            const float k1 = static_cast<float>((int)b >> 4) * kscale;
            partial += __half2float(q_shared[2 * i]) * k0;
            partial += __half2float(q_shared[2 * i + 1]) * k1;
          }
        } else {
          const float kscale = __half2float(k_scales[t * num_kv_heads + kv_head]);
          const int8_t* k_q = k_cache + (t * num_kv_heads + kv_head) * k_row;
          for (int i = lane; i < head_dim; i += warpSize) {
            partial += __half2float(q_shared[i]) * static_cast<float>(k_q[i]) * kscale;
          }
        }
        score = warp_sum(partial) * scale;
      }
      if (lane == 0 && warp_id < tile_tokens) {
        score_shared[warp_id] = score;
      }
    }

    // Phase 1b: stage V into shared v_tile (fp16 copy or dequant).
    for (int i = 0; i < tile_tokens; ++i) {
      const int t = tile_base + i;
      const int t_abs = attn_start + t;
      const half* vf = kv_fp16_source(v_sink, v_ring, t_abs, attn_start + seq_len, sink_n, win_n,
                                      num_kv_heads, kv_head, head_dim);
      half* vt = v_tile + i * head_dim;
      if (vf != nullptr) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
          vt[d] = vf[d];
        }
      } else {
        const float vscale = __half2float(v_scales[t * num_kv_heads + kv_head]);
        const int8_t* v_q = v_cache + (t * num_kv_heads + kv_head) * v_row;
        for (int d = tid; d < head_dim; d += blockDim.x) {
          vt[d] = __float2half(kv_load<VBits>(v_q, d, vscale));
        }
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

  // Write partial stats and output to scratch buffers.
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

// Serial fallback decode attention over the quantized KV cache.
// Structurally identical to attention_step_kernel_tiled_device_pos except for
// the dequantizing K/V reads (and the optional Q rotation).
template <int WarpsPerBlock, int KBits, int VBits, bool RotK>
__global__ void attention_step_kernel_quant(const half* q, const int8_t* k_cache,
                                            const int8_t* v_cache, const half* k_scales,
                                            const half* v_scales, const half* k_sink,
                                            const half* v_sink, const half* k_ring,
                                            const half* v_ring, int sink_n, int win_n,
                                            int attn_start, half* out, int seq_len, int num_heads,
                                            int num_kv_heads, int head_dim) {
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
  const int k_row = head_dim * KBits / 8;
  const int v_row = head_dim * VBits / 8;

  for (int d = tid; d < head_dim; d += blockDim.x) {
    q_shared[d] = q[head * head_dim + d];
  }
  if (tid == 0) {
    stats_shared[0] = -1.0e30f;
    stats_shared[1] = 0.0f;
  }
  __syncthreads();
  if (RotK) {
    float* qbuf = reinterpret_cast<float*>(v_tile + WarpsPerBlock * head_dim);
    rotate_q_shared(q_shared, qbuf, tid, head_dim);
  }

  static_assert(WarpsPerBlock * 32 * kAccPerThread >= kTiledMaxHeadDim,
                "quant decode accumulator too small for kTiledMaxHeadDim");
  float acc[kAccPerThread];
#pragma unroll
  for (int i = 0; i < kAccPerThread; ++i) acc[i] = 0.0f;
  for (int tile_base = 0; tile_base < seq_len; tile_base += WarpsPerBlock) {
    const int tile_tokens = min(WarpsPerBlock, seq_len - tile_base);

    // Phase 1a: each warp computes the K.Q score for its tile token, reading
    // fp16 sink/window buffers where applicable.
    {
      const int t = tile_base + warp_id;
      float score = -1.0e30f;
      if (warp_id < tile_tokens) {
        const int t_abs = attn_start + t;
        const half* kf = kv_fp16_source(k_sink, k_ring, t_abs, attn_start + seq_len, sink_n, win_n,
                                        num_kv_heads, kv_head, head_dim);
        float partial = 0.0f;
        if (kf != nullptr) {
          for (int i = lane; i < head_dim; i += warpSize) {
            partial += __half2float(q_shared[i]) * __half2float(kf[i]);
          }
        } else if (KBits == 4) {
          const float kscale = __half2float(k_scales[t * num_kv_heads + kv_head]);
          const int8_t* k_q = k_cache + (t * num_kv_heads + kv_head) * k_row;
          for (int i = lane; i < head_dim / 2; i += warpSize) {
            const int8_t b = k_q[i];
            const float k0 = static_cast<float>(((int)b << 28) >> 28) * kscale;
            const float k1 = static_cast<float>((int)b >> 4) * kscale;
            partial += __half2float(q_shared[2 * i]) * k0;
            partial += __half2float(q_shared[2 * i + 1]) * k1;
          }
        } else {
          const float kscale = __half2float(k_scales[t * num_kv_heads + kv_head]);
          const int8_t* k_q = k_cache + (t * num_kv_heads + kv_head) * k_row;
          for (int i = lane; i < head_dim; i += warpSize) {
            partial += __half2float(q_shared[i]) * static_cast<float>(k_q[i]) * kscale;
          }
        }
        score = warp_sum(partial) * scale;
      }
      if (lane == 0 && warp_id < tile_tokens) {
        score_shared[warp_id] = score;
      }
    }

    // Phase 1b: stage V tile in shared mem (fp16 copy or dequant).
    {
      for (int i = 0; i < tile_tokens; ++i) {
        const int t = tile_base + i;
        const int t_abs = attn_start + t;
        const half* vf = kv_fp16_source(v_sink, v_ring, t_abs, attn_start + seq_len, sink_n, win_n,
                                        num_kv_heads, kv_head, head_dim);
        half* vt = v_tile + i * head_dim;
        if (vf != nullptr) {
          for (int d = tid; d < head_dim; d += blockDim.x) {
            vt[d] = vf[d];
          }
        } else {
          const float vscale = __half2float(v_scales[t * num_kv_heads + kv_head]);
          const int8_t* v_q = v_cache + (t * num_kv_heads + kv_head) * v_row;
          for (int d = tid; d < head_dim; d += blockDim.x) {
            vt[d] = __float2half(kv_load<VBits>(v_q, d, vscale));
          }
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

constexpr int kQuantAttnWarps = 4;

std::size_t quant_attn_smem(int head_dim, bool rot_k) {
  std::size_t smem = static_cast<std::size_t>(head_dim) * sizeof(half) +
                     static_cast<std::size_t>(2 * kQuantAttnWarps + 4) * sizeof(float) +
                     static_cast<std::size_t>(kQuantAttnWarps * head_dim) * sizeof(half);
  if (rot_k) smem += static_cast<std::size_t>(head_dim) * sizeof(float);
  return smem;
}

}  // namespace

// Host launch wrappers.

void launch_store_kv_quant(const half* k, const half* v, int8_t* k_cache, int8_t* v_cache,
                           half* k_scales, half* v_scales, half* k_sink, half* v_sink, half* k_ring,
                           half* v_ring, int sink_n, int win_n, int position, int num_kv_heads,
                           int head_dim, int max_context, int k_bits, int v_bits, bool rotate_k,
                           cudaStream_t stream) {
  const int num_warps = head_dim / 32;
  const std::size_t smem =
      static_cast<std::size_t>(2 * head_dim + 2 * num_warps + 2) * sizeof(float);
  const dim3 grid(num_kv_heads);
  const dim3 block(head_dim);
  // The attention side can only rotate Q when head_dim matches its block size,
  // so the store side must apply the identical gate or the bases diverge.
  rotate_k = rotate_k && head_dim == 128;
#define CPI_STORE_CASE(KB, VB, RK)                                                               \
  store_kv_quant_kernel<KB, VB, RK><<<grid, block, smem, stream>>>(                              \
      k, v, k_cache, v_cache, k_scales, v_scales, k_sink, v_sink, k_ring, v_ring, sink_n, win_n, \
      position, num_kv_heads, head_dim, max_context)
  if (k_bits == 4 && v_bits == 4 && rotate_k) {
    CPI_STORE_CASE(4, 4, true);
  } else if (k_bits == 4 && v_bits == 4) {
    CPI_STORE_CASE(4, 4, false);
  } else if (k_bits == 8 && v_bits == 4) {
    CPI_STORE_CASE(8, 4, false);
  } else {
    CPI_STORE_CASE(8, 8, false);
  }
#undef CPI_STORE_CASE
}

void launch_attention_step_quant(const half* q, const int8_t* k_cache, const int8_t* v_cache,
                                 const half* k_scales, const half* v_scales, const half* k_sink,
                                 const half* v_sink, const half* k_ring, const half* v_ring,
                                 int sink_n, int win_n, int attn_start, half* out, int seq_len,
                                 int num_heads, int num_kv_heads, int head_dim, int k_bits,
                                 int v_bits, bool rotate_k, cudaStream_t stream, float* scratch_m,
                                 float* scratch_l, float* scratch_o, int scratch_chunks,
                                 bool allow_split) {
  constexpr int warps = kQuantAttnWarps;
  constexpr int threads = warps * 32;
  // Q rotation needs one thread per element of the head.
  const bool rot = rotate_k && head_dim == threads;
  const std::size_t smem = quant_attn_smem(head_dim, rot);

  // Split-K path: same chunk decomposition as the fp16 attention kernel.
  // Requires head_dim==128 (matches scratch_o element stride), seq_len>=64, and
  // allocated scratch buffers.
  constexpr int split_chunk_size = 32;
  const bool split = allow_split && scratch_m && scratch_l && scratch_o && scratch_chunks > 0 &&
                     head_dim == 128 && seq_len >= 64;
  const int chunk_count =
      split ? min(scratch_chunks, (seq_len + split_chunk_size - 1) / split_chunk_size) : 0;
  const dim3 split_grid(num_heads, chunk_count > 0 ? chunk_count : 1);

#define CPI_ATTN_CASE(KB, VB, RK)                                                                 \
  do {                                                                                            \
    if (split) {                                                                                  \
      attention_step_chunk_stats_quant_kernel<warps, KB, VB, RK>                                  \
          <<<split_grid, threads, smem, stream>>>(                                                \
              q, k_cache, v_cache, k_scales, v_scales, k_sink, v_sink, k_ring, v_ring, sink_n,    \
              win_n, attn_start, scratch_m, scratch_l, scratch_o, seq_len, num_heads,             \
              num_kv_heads, head_dim, split_chunk_size, scratch_chunks);                          \
      attention_step_chunk_reduce_kernel<<<num_heads, threads, 0, stream>>>(                      \
          scratch_m, scratch_l, scratch_o, out, seq_len, num_heads, head_dim, split_chunk_size,   \
          scratch_chunks);                                                                        \
    } else {                                                                                      \
      attention_step_kernel_quant<warps, KB, VB, RK><<<num_heads, threads, smem, stream>>>(       \
          q, k_cache, v_cache, k_scales, v_scales, k_sink, v_sink, k_ring, v_ring, sink_n, win_n, \
          attn_start, out, seq_len, num_heads, num_kv_heads, head_dim);                           \
    }                                                                                             \
  } while (0)
  if (k_bits == 4 && v_bits == 4 && rot) {
    CPI_ATTN_CASE(4, 4, true);
  } else if (k_bits == 4 && v_bits == 4) {
    CPI_ATTN_CASE(4, 4, false);
  } else if (k_bits == 8 && v_bits == 4) {
    CPI_ATTN_CASE(8, 4, false);
  } else {
    CPI_ATTN_CASE(8, 8, false);
  }
#undef CPI_ATTN_CASE
}

// Legacy int4 entry points, preserved as wrappers over the unrotated 4-bit path.
void launch_store_kv_int4(const half* k, const half* v, int8_t* k_cache_i4, int8_t* v_cache_i4,
                          half* k_scales, half* v_scales, int position, int num_kv_heads,
                          int head_dim, int max_context, cudaStream_t stream) {
  launch_store_kv_quant(k, v, k_cache_i4, v_cache_i4, k_scales, v_scales, nullptr, nullptr, nullptr,
                        nullptr, 0, 0, position, num_kv_heads, head_dim, max_context, 4, 4, false,
                        stream);
}

void launch_attention_step_int4(const half* q, const int8_t* k_cache_i4, const int8_t* v_cache_i4,
                                const half* k_scales, const half* v_scales, half* out, int seq_len,
                                int num_heads, int num_kv_heads, int head_dim, cudaStream_t stream,
                                float* scratch_m, float* scratch_l, float* scratch_o,
                                int scratch_chunks, bool allow_split) {
  launch_attention_step_quant(q, k_cache_i4, v_cache_i4, k_scales, v_scales, nullptr, nullptr,
                              nullptr, nullptr, 0, 0, 0, out, seq_len, num_heads, num_kv_heads,
                              head_dim, 4, 4, false, stream, scratch_m, scratch_l, scratch_o,
                              scratch_chunks, allow_split);
}

}  // namespace kernels
