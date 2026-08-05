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
  float* q_sh = smem;              // [kdim]
  float* warp_m = q_sh + kdim;     // [4]
  float* warp_l = warp_m + 4;      // [4]
  float* warp_o = warp_l + 4;      // [4][vdim]

  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid & 31;

  for (int d = tid; d < kdim; d += blockDim.x) {
    q_sh[d] = __half2float(q_abs[static_cast<size_t>(h) * kdim + d]);
  }
  __syncthreads();

  // Per-warp online softmax over this warp's tokens (chunk_start+warp, +4, ...).
  constexpr int kAccPer = 16;  // vdim <= 512 = 32 lanes * 16
  float acc[kAccPer];
#pragma unroll
  for (int i = 0; i < kAccPer; ++i) acc[i] = 0.0f;
  float rm = -3.4e38f, rl = 0.0f;
  for (int t = chunk_start + warp_id; t < chunk_end; t += 4) {
    const __half* row = cache + static_cast<size_t>(t) * kdim;
    const __half2* row2 = reinterpret_cast<const __half2*>(row);
    float partial = 0.0f;
    for (int p = lane; p < kdim / 2; p += 32) {
      const float2 v = __half22float2(row2[p]);
      partial += q_sh[2 * p] * v.x + q_sh[2 * p + 1] * v.y;
    }
    for (int off = 16; off > 0; off >>= 1)
      partial += __shfl_xor_sync(0xffffffffu, partial, off);
    const float s = partial * scale;
    const float nm = fmaxf(rm, s);
    const float corr = (rl == 0.0f) ? 0.0f : expf(rm - nm);
    const float w = expf(s - nm);
    rl = rl * corr + w;
    rm = nm;
#pragma unroll
    for (int i = 0; i < kAccPer; ++i) {
      const int d = lane + 32 * i;
      const float vv = (d < vdim) ? __half2float(row[d]) : 0.0f;
      acc[i] = acc[i] * corr + w * vv;
    }
  }

  // Stage warp partials and merge (contribution of warp w at global max M is
  // acc_w * exp(m_w - M); l merges the same way).
  if (lane == 0) {
    warp_m[warp_id] = rm;
    warp_l[warp_id] = rl;
  }
#pragma unroll
  for (int i = 0; i < kAccPer; ++i) {
    const int d = lane + 32 * i;
    if (d < vdim) warp_o[warp_id * vdim + d] = acc[i];
  }
  __syncthreads();

  const int ci = h * scratch_chunks + chunk;
  float gm = -3.4e38f;
  for (int w = 0; w < 4; ++w) gm = fmaxf(gm, warp_m[w]);
  if (tid == 0) {
    float gl = 0.0f;
    for (int w = 0; w < 4; ++w) {
      gl += (warp_l[w] == 0.0f) ? 0.0f : warp_l[w] * expf(warp_m[w] - gm);
    }
    chunk_m[ci] = gm;
    chunk_l[ci] = gl;
  }
  for (int d = tid; d < vdim; d += blockDim.x) {
    float o = 0.0f;
    for (int w = 0; w < 4; ++w) {
      const float f = (warp_l[w] == 0.0f) ? 0.0f : expf(warp_m[w] - gm);
      o += warp_o[w * vdim + d] * f;
    }
    chunk_o[static_cast<size_t>(ci) * vdim + d] = o;
  }
}

// Split-K pass 2, fused with the V decompression: merge a head's chunk
// partials into the attention-weighted latent (shared memory), then each warp
// computes W_UV rows against it and writes the padded Att row directly. One
// block per head; saves a kernel launch and the MlaAttLat round-trip.
__global__ inline void mla_absorbed_reduce_decompress_kernel(
    const float* chunk_m, const float* chunk_l, const float* chunk_o, const __half* w_uv,
    __half* att, int seq_len, const int* d_position, int vdim, int v_head, int qkhd,
    int chunk_size, int scratch_chunks) {
  if (d_position) seq_len = *d_position + 1;
  extern __shared__ float rsm[];  // [vdim] merged latent + [3] scale broadcast
  float* lat_sh = rsm;
  float* sc = rsm + vdim;
  const int h = blockIdx.x;
  const int tid = threadIdx.x;
  const int chunk_count = min(scratch_chunks, (seq_len + chunk_size - 1) / chunk_size);
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float rm = -3.4e38f, rl = 0.0f;
  for (int c = 0; c < chunk_count; ++c) {
    if (tid == 0) {
      const int idx = h * scratch_chunks + c;
      const float cm = chunk_m[idx], cl = chunk_l[idx];
      const float nm = fmaxf(rm, cm);
      const float alpha = (rl == 0.0f) ? 0.0f : expf(rm - nm);
      const float beta = (cl == 0.0f) ? 0.0f : expf(cm - nm);
      rl = rl * alpha + cl * beta;
      rm = nm;
      sc[0] = alpha;
      sc[1] = beta;
      sc[2] = rl;
    }
    __syncthreads();
    const float alpha = sc[0], beta = sc[1];
    const size_t base = (static_cast<size_t>(h) * scratch_chunks + c) * vdim;
    int j = 0;
    for (int d = tid; d < vdim; d += blockDim.x, ++j) {
      acc[j] = acc[j] * alpha + chunk_o[base + d] * beta;
    }
    __syncthreads();
  }
  const float inv_l = 1.0f / fmaxf(sc[2], 1e-8f);
  int j = 0;
  for (int d = tid; d < vdim; d += blockDim.x, ++j) {
    lat_sh[d] = acc[j] * inv_l;
  }
  __syncthreads();

  // Decompress: each warp walks output dims d = warp, warp+4, ...; pads zeroed.
  const int warp_id = tid / 32;
  const int lane = tid & 31;
  __half* out = att + static_cast<size_t>(h) * qkhd;
  for (int d = warp_id; d < qkhd; d += 4) {
    if (d >= v_head) {
      if (lane == 0) out[d] = __float2half(0.0f);
      continue;
    }
    const __half* w = w_uv + (static_cast<size_t>(h) * v_head + d) * vdim;
    float a = 0.0f;
    for (int k = lane; k < vdim; k += 32) {
      a += __half2float(w[k]) * lat_sh[k];
    }
    for (int off = 16; off > 0; off >>= 1) a += __shfl_down_sync(0xffffffffu, a, off);
    if (lane == 0) out[d] = __float2half(a);
  }
}

inline void launch_mla_absorbed_attention(const __half* q_abs, const __half* cache,
                                          const __half* w_uv, __half* att, int nh, int kdim,
                                          int vdim, int v_head, int qkhd, float scale,
                                          int seq_len, const int* d_position, float* sm,
                                          float* sl, float* so, int scratch_chunks,
                                          bool fixed_grid, cudaStream_t stream) {
  constexpr int kChunk = 64;
  const int chunks =
      fixed_grid ? scratch_chunks
                 : min(scratch_chunks, (seq_len + kChunk - 1) / kChunk);
  const dim3 grid(nh, chunks > 0 ? chunks : 1);
  const size_t smem = (static_cast<size_t>(kdim) + 8 + 4 * static_cast<size_t>(vdim)) * sizeof(float);
  mla_absorbed_attn_stats_kernel<<<grid, 128, smem, stream>>>(
      q_abs, cache, sm, sl, so, seq_len, d_position, kdim, vdim, scale, kChunk, scratch_chunks);
  const size_t rsmem = (static_cast<size_t>(vdim) + 3) * sizeof(float);
  mla_absorbed_reduce_decompress_kernel<<<nh, 128, rsmem, stream>>>(
      sm, sl, so, w_uv, att, seq_len, d_position, vdim, v_head, qkhd, kChunk, scratch_chunks);
}

}  // namespace engine
