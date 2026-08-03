#pragma once

// The one new kernel the in-engine DeepSeek MLA needs: from a single token's projected pieces, assemble
// the per-head K and V the standard Attention op consumes, and apply the interleaved-complex YARN rope
// to q_pe and the shared k_pe. Everything else in MLA (the projections, the latent RMSNorm, the
// attention, o_proj) reuses existing ops. Verified fp16-vs-oracle in deepseek_mla_assemble_test.
//
// One token at absolute `position`. Layouts (fp16 slots, per head h):
//   Q   [nh*qkhd]  in/out : [q_nope(qk_nope) | q_pe(qk_rope)]; q_pe is roped IN PLACE.
//   kvb [nh*kvh]   in     : [k_nope(qk_nope) | v(v_head)]   (kv_b up-projection output)
//   ckv [kv_lora+qk_rope] in : the k_pe (shared) lives at [kv_lora, kv_lora+qk_rope)
//   K   [nh*qkhd]  out    : [k_nope | k_pe_roped]   (k_pe shared across heads, roped)
//   V   [nh*qkhd]  out    : [v | zeros]             (V padded to qkhd so one head_dim serves Q/K/V)
// qkhd = qk_nope+qk_rope, kvh = qk_nope+v_head. Rope is INTERLEAVED: pair (2p,2p+1) rotates by
// position*inv_freq[p], then scales by attn_scaling (the yarn mscale).

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace engine {

__global__ inline void mla_assemble_rope_kernel(__half* Q, const __half* kvb, const __half* ckv,
                                                __half* K, __half* V, const float* inv_freq, int nh,
                                                int qk_nope, int qk_rope, int v_head, int kv_lora,
                                                int position, float attn_scaling) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= nh) return;
  const int qkhd = qk_nope + qk_rope;
  const int kvh = qk_nope + v_head;
  const int half = qk_rope / 2;

  // q_pe rope, in place (interleaved).
  __half* q = Q + (size_t)h * qkhd + qk_nope;
  for (int p = 0; p < half; ++p) {
    const float ang = position * inv_freq[p], c = cosf(ang), s = sinf(ang);
    const float x0 = __half2float(q[2 * p]), x1 = __half2float(q[2 * p + 1]);
    q[2 * p] = __float2half(attn_scaling * (x0 * c - x1 * s));
    q[2 * p + 1] = __float2half(attn_scaling * (x0 * s + x1 * c));
  }

  // K = [k_nope | k_pe_roped].
  __half* k = K + (size_t)h * qkhd;
  const __half* kn = kvb + (size_t)h * kvh;
  for (int a = 0; a < qk_nope; ++a) k[a] = kn[a];
  const __half* kpe = ckv + kv_lora;  // shared rope key
  for (int p = 0; p < half; ++p) {
    const float ang = position * inv_freq[p], c = cosf(ang), s = sinf(ang);
    const float x0 = __half2float(kpe[2 * p]), x1 = __half2float(kpe[2 * p + 1]);
    k[qk_nope + 2 * p] = __float2half(attn_scaling * (x0 * c - x1 * s));
    k[qk_nope + 2 * p + 1] = __float2half(attn_scaling * (x0 * s + x1 * c));
  }

  // V = [v | zeros] (padded to qkhd).
  __half* v = V + (size_t)h * qkhd;
  const __half* vv = kvb + (size_t)h * kvh + qk_nope;
  for (int b = 0; b < v_head; ++b) v[b] = vv[b];
  for (int b = v_head; b < qkhd; ++b) v[b] = __float2half(0.0f);
}

inline void launch_mla_assemble_rope(__half* Q, const __half* kvb, const __half* ckv, __half* K,
                                     __half* V, const float* inv_freq, int nh, int qk_nope,
                                     int qk_rope, int v_head, int kv_lora, int position,
                                     float attn_scaling, cudaStream_t stream) {
  mla_assemble_rope_kernel<<<(nh + 63) / 64, 64, 0, stream>>>(Q, kvb, ckv, K, V, inv_freq, nh,
                                                              qk_nope, qk_rope, v_head, kv_lora,
                                                              position, attn_scaling);
}

}  // namespace engine
