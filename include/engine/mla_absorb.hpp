#pragma once

// Absorbed-MLA decode kernels for DeepSeek-V2 on the op-plan engine.
//
// Instead of attending over the materialized per-head K/V (nh*(qkhd + qkhd)
// halves per token), decode scores each head's absorbed query
// [W_UK^T q_nope | q_pe] against the cached 576-dim latent rows
// [kv_a_layernorm latent (kv_lora) | roped+scaled k_pe (qk_rope)] and takes
// V as the first kv_lora dims of the same rows, decompressing once through
// W_UV afterwards. Identical math to the materialized path (q_nope . W_UK c
// == (W_UK^T q_nope) . c; sum_t p_t W_UV c_t == W_UV sum_t p_t c_t), ~9x less
// attention bandwidth at depth. Prefill keeps the materialized path.
//
// The softmax scale must be the materialized head's rsqrt(qkhd) (the yarn
// mscale is already baked into the roped q_pe/k_pe by the assemble kernel);
// rsqrt of the 576-dim absorbed width would be wrong.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace engine {

// Append T rows of [latent | roped k_pe] to the latent cache. The roped
// (and mscale-scaled) k_pe is read from Slot::K's head-0 row at offset
// qk_nope (it is identical in every head's row). lat rows are [kv_lora],
// K rows are [nh*qkhd].
__global__ inline void mla_lat_store_kernel(const __half* lat, const __half* K, __half* cache,
                                            int kv_lora, int qk_nope, int qk_rope, int qkhd,
                                            int nh, int T, int position, const int* d_position,
                                            int max_ctx) {
  const int r = blockIdx.x;
  if (r >= T) return;
  if (d_position) position = *d_position;
  const int pos = position + r;
  if (pos < 0 || pos >= max_ctx) return;
  const int width = kv_lora + qk_rope;
  __half* row = cache + static_cast<size_t>(pos) * width;
  const __half* lrow = lat + static_cast<size_t>(r) * kv_lora;
  const __half* krow = K + static_cast<size_t>(r) * nh * qkhd + qk_nope;
  for (int j = threadIdx.x; j < width; j += blockDim.x) {
    row[j] = (j < kv_lora) ? lrow[j] : krow[j - kv_lora];
  }
}

inline void launch_mla_lat_store(const __half* lat, const __half* K, __half* cache, int kv_lora,
                                 int qk_nope, int qk_rope, int qkhd, int nh, int T, int position,
                                 const int* d_position, int max_ctx, cudaStream_t stream) {
  mla_lat_store_kernel<<<T, 192, 0, stream>>>(lat, K, cache, kv_lora, qk_nope, qk_rope, qkhd, nh,
                                              T, position, d_position, max_ctx);
}

// Absorbed query: MlaQAbs[i] = [W_UK[i]^T q_nope[i] (kv_lora) | q_pe[i] (qk_rope)].
// Q rows are [nh][qkhd] with q_pe already roped+scaled in place; w_uk_t is
// [nh][kv_lora][qk_nope] (output-major so each warp's dot reads a contiguous row).
// Block: 4 warps, one output element per warp; grid (nh, ceil(width/4)).
__global__ inline void mla_q_absorb_kernel(const __half* Q, const __half* w_uk_t, __half* q_abs,
                                           int nh, int qk_nope, int qk_rope, int qkhd,
                                           int kv_lora) {
  const int h = blockIdx.x;
  const int j = blockIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  const int width = kv_lora + qk_rope;
  if (h >= nh || j >= width) return;
  __half* out = q_abs + static_cast<size_t>(h) * width;
  if (j >= kv_lora) {
    if (lane == 0) out[j] = Q[static_cast<size_t>(h) * qkhd + qk_nope + (j - kv_lora)];
    return;
  }
  const __half* w = w_uk_t + (static_cast<size_t>(h) * kv_lora + j) * qk_nope;
  const __half* q = Q + static_cast<size_t>(h) * qkhd;
  float acc = 0.0f;
  for (int d = lane; d < qk_nope; d += 32) {
    acc += __half2float(w[d]) * __half2float(q[d]);
  }
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffffu, acc, off);
  if (lane == 0) out[j] = __float2half(acc);
}

inline void launch_mla_q_absorb(const __half* Q, const __half* w_uk_t, __half* q_abs, int nh,
                                int qk_nope, int qk_rope, int qkhd, int kv_lora,
                                cudaStream_t stream) {
  const int width = kv_lora + qk_rope;
  const dim3 grid(nh, (width + 3) / 4);
  mla_q_absorb_kernel<<<grid, 128, 0, stream>>>(Q, w_uk_t, q_abs, nh, qk_nope, qk_rope, qkhd,
                                                kv_lora);
}

// Split-K pass 1: one block per (head, chunk of 64 tokens), 4 warps. Each
// warp owns every 4th token of the chunk with its OWN running online softmax
// (no per-tile block synchronization): score = half2 dot over kdim, then the
// warp accumulates the token's first vdim dims weighted into 16 per-lane
// register accumulators. The 4 warp partials merge once at the end. Writes
// chunk stats + partial outputs; a fixed max-chunk grid with per-chunk guards
// keeps the kernel graph-capturable at any depth.
__global__ inline void mla_absorbed_attn_stats_kernel(const __half* q_abs, const __half* cache,
                                                      float* chunk_m, float* chunk_l,
                                                      float* chunk_o, int seq_len,
                                                      const int* d_position, int kdim, int vdim,
                                                      float scale, int chunk_size,
                                                      int scratch_chunks) {
  if (d_position) seq_len = *d_position + 1;
  const int h = blockIdx.x;
  const int chunk = blockIdx.y;
  const int chunk_start = chunk * chunk_size;
  if (chunk_start >= seq_len || chunk >= scratch_chunks) return;
  const int chunk_end = min(chunk_start + chunk_size, seq_len);

  extern __shared__ float smem[];
  float* q_sh = smem;                // [kdim] (read as float2 pairs when scoring)
  float* score_sh = q_sh + kdim;     // [chunk_size]

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid & 31;
  const int warps = blockDim.x / 32;

  for (int d = tid; d < kdim; d += blockDim.x) {
    q_sh[d] = __half2float(q_abs[static_cast<size_t>(h) * kdim + d]);
  }
  __syncthreads();

  // Phase 1: score every token in the chunk into shared (one warp per token,
  // striding). float2 q reads keep shared accesses conflict-free.
  const float2* q2 = reinterpret_cast<const float2*>(q_sh);
  for (int t = chunk_start + warp_id; t < chunk_end; t += warps) {
    const __half2* row2 = reinterpret_cast<const __half2*>(cache + static_cast<size_t>(t) * kdim);
    float partial = 0.0f;
    for (int p = lane; p < kdim / 2; p += 32) {
      const float2 qv = q2[p];
      const float2 v = __half22float2(row2[p]);
      partial += qv.x * v.x + qv.y * v.y;
    }
    for (int off = 16; off > 0; off >>= 1)
      partial += __shfl_down_sync(0xffffffffu, partial, off);
    if (lane == 0) score_sh[t - chunk_start] = partial * scale;
  }
  __syncthreads();

  // Phase 2: chunk max scanned redundantly per thread (n <= 32 shared reads),
  // then the softmax weights are materialized ONCE into shared and the V pass
  // reads them as broadcasts (no per-thread expf).
  const int n = chunk_end - chunk_start;
  float cm = -3.4e38f;
  for (int i = 0; i < n; ++i) cm = fmaxf(cm, score_sh[i]);
  __syncthreads();
  if (tid < n) score_sh[tid] = expf(score_sh[tid] - cm);
  __syncthreads();
  float cl = 0.0f;
  for (int i = 0; i < n; ++i) cl += score_sh[i];

  const int ci = h * scratch_chunks + chunk;
  if (tid == 0) {
    chunk_m[ci] = cm;
    chunk_l[ci] = cl;
  }
  float acc0 = 0.0f, acc1 = 0.0f;
  const int d0 = tid;
  const int d1 = tid + blockDim.x;
  for (int i = 0; i < n; ++i) {
    const float w = score_sh[i];
    const __half* row = cache + static_cast<size_t>(chunk_start + i) * kdim;
    if (d0 < vdim) acc0 += w * __half2float(row[d0]);
    if (d1 < vdim) acc1 += w * __half2float(row[d1]);
  }
  if (d0 < vdim) chunk_o[static_cast<size_t>(ci) * vdim + d0] = acc0;
  if (d1 < vdim) chunk_o[static_cast<size_t>(ci) * vdim + d1] = acc1;
}

// Split-K pass 2: barrier-free merge of a head's chunk partials into the
// normalized attention-weighted latent (out_lat). Tiny (grid nh); the
// bandwidth-heavy decompress runs as its own well-parallel kernel after.
__global__ inline void mla_absorbed_reduce_kernel(const float* chunk_m, const float* chunk_l,
                                                  const float* chunk_o, __half* out_lat,
                                                  int seq_len, const int* d_position, int vdim,
                                                  int chunk_size, int scratch_chunks) {
  if (d_position) seq_len = *d_position + 1;
  // Grid (nh, vdim/blockDim): each block owns one dim slice, so the scratch
  // read spreads across nh*slices blocks. The m/l scans are cheap broadcasts.
  const int h = blockIdx.x;
  const int d = blockIdx.y * blockDim.x + threadIdx.x;
  const int chunk_count = min(scratch_chunks, (seq_len + chunk_size - 1) / chunk_size);
  float gm = -3.4e38f;
  for (int c = 0; c < chunk_count; ++c) {
    gm = fmaxf(gm, chunk_m[h * scratch_chunks + c]);
  }
  float gl = 0.0f;
  float acc = 0.0f;
  for (int c = 0; c < chunk_count; ++c) {
    const int idx = h * scratch_chunks + c;
    const float cl = chunk_l[idx];
    const float w = (cl == 0.0f) ? 0.0f : expf(chunk_m[idx] - gm);
    gl += cl * w;
    if (d < vdim) {
      acc += chunk_o[(static_cast<size_t>(h) * scratch_chunks + c) * vdim + d] * w;
    }
  }
  if (d < vdim) {
    out_lat[static_cast<size_t>(h) * vdim + d] = __float2half(acc / fmaxf(gl, 1e-8f));
  }
}

// Decompress: Att[h, d] = W_UV[h][d] . lat[h] for d < v_head, pads zeroed.
// Grid (nh, qkhd/4), one output per warp: the 2MB weight read spreads over
// ~500 blocks instead of nh, which is what makes it bandwidth- rather than
// occupancy-bound. lat rows are L2-hot (1KB per head).
__global__ inline void mla_v_decompress_kernel(const __half* out_lat, const __half* w_uv,
                                               __half* att, int nh, int vdim, int v_head,
                                               int qkhd) {
  const int h = blockIdx.x;
  const int d = blockIdx.y * (blockDim.x / 32) + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  if (h >= nh || d >= qkhd) return;
  __half* out = att + static_cast<size_t>(h) * qkhd;
  if (d >= v_head) {
    if (lane == 0) out[d] = __float2half(0.0f);
    return;
  }
  const __half2* w2 =
      reinterpret_cast<const __half2*>(w_uv + (static_cast<size_t>(h) * v_head + d) * vdim);
  const __half2* l2 =
      reinterpret_cast<const __half2*>(out_lat + static_cast<size_t>(h) * vdim);
  float a = 0.0f;
  for (int p = lane; p < vdim / 2; p += 32) {
    const float2 wv = __half22float2(w2[p]);
    const float2 lv = __half22float2(l2[p]);
    a += wv.x * lv.x + wv.y * lv.y;
  }
  for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
  if (lane == 0) out[d] = __float2half(a);
}

inline void launch_mla_absorbed_attention(const __half* q_abs, const __half* cache,
                                          const __half* w_uv, __half* out_lat, __half* att,
                                          int nh, int kdim, int vdim, int v_head, int qkhd,
                                          float scale, int seq_len, const int* d_position,
                                          float* sm, float* sl, float* so, int scratch_chunks,
                                          bool fixed_grid, cudaStream_t stream) {
  constexpr int kChunk = 32;
  const int chunks =
      fixed_grid ? scratch_chunks
                 : min(scratch_chunks, (seq_len + kChunk - 1) / kChunk);
  const dim3 grid(nh, chunks > 0 ? chunks : 1);
  const size_t smem = (static_cast<size_t>(kdim) + kChunk) * sizeof(float);
  mla_absorbed_attn_stats_kernel<<<grid, 256, smem, stream>>>(
      q_abs, cache, sm, sl, so, seq_len, d_position, kdim, vdim, scale, kChunk, scratch_chunks);
  const dim3 rgrid(nh, (vdim + 127) / 128);
  mla_absorbed_reduce_kernel<<<rgrid, 128, 0, stream>>>(sm, sl, so, out_lat, seq_len, d_position,
                                                        vdim, kChunk, scratch_chunks);
  const dim3 dgrid(nh, (qkhd + 3) / 4);
  mla_v_decompress_kernel<<<dgrid, 128, 0, stream>>>(out_lat, w_uv, att, nh, vdim, v_head, qkhd);
}

}  // namespace engine
