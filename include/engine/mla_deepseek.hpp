#pragma once

// DeepSeek-V2/V3 Multi-head Latent Attention, matching transformers' DeepseekV2Attention exactly (the
// real V2-Lite structure, verified in deepseek_mla_real_test against an HF dump). Differs from the
// generic mla_forward.hpp brick: (1) NO q-lora -- q is a direct projection; (2) a SINGLE
// kv_a_proj_with_mqa produces [latent(kv_lora) | k_pe(qk_rope)] and the k_pe head is shared across all
// heads; (3) kv_a_layernorm (RMSNorm) on the latent before kv_b up-projection; (4) INTERLEAVED-complex
// RoPE (pairs (2p,2p+1)) using the model's inv_freq, with the rotated vectors scaled by
// attention_scaling (the YARN mscale, baked into the rope, not the softmax scale which stays 1/sqrt(hd)).
//
// fp32, naive-reconstruction prefill. inv_freq + attention_scaling are supplied by the caller (dumped
// from the model), so this op doesn't re-derive YARN; computing inv_freq from config is a separate step.

#include <cuda_runtime.h>

#include <cmath>

#include "engine/mla_forward.hpp"  // engine::mla_detail::mm_wt

namespace engine {

struct DSMLADims {
  int H = 0, nh = 0, qk_nope = 0, qk_rope = 0, v_head = 0, kv_lora = 0;
  float softmax_scale = 0.0f, attn_scaling = 1.0f, rms_eps = 1e-6f;
  int qkhd() const { return qk_nope + qk_rope; }         // per-head query/key dim
  int kvh() const { return qk_nope + v_head; }           // per-head [k_nope | v] from kv_b
  int kva() const { return kv_lora + qk_rope; }          // kv_a_proj output width
};

struct DSMLAWeights {
  const float* q_proj = nullptr;   // [nh*qkhd, H]
  const float* kv_a = nullptr;     // [kv_lora+qk_rope, H]  (kv_a_proj_with_mqa)
  const float* kv_a_ln = nullptr;  // [kv_lora]             (kv_a_layernorm weight)
  const float* kv_b = nullptr;     // [nh*(qk_nope+v_head), kv_lora]
  const float* o_proj = nullptr;   // [H, nh*v_head]
  const float* inv_freq = nullptr; // [qk_rope/2]
};

namespace dsmla_detail {

// RMSNorm y[T,D] = (x_row / sqrt(mean(x^2)+eps)) * w, reading x from a strided source (row stride
// `sstride`, starting at `soff`) so the latent slice of kv_a's output can be normed without a copy.
__global__ inline void rmsnorm_strided(const float* src, int sstride, int soff, const float* w,
                                       float* y, int T, int D, float eps) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* xr = src + (size_t)t * sstride + soff;
  float ss = 0.0f;
  for (int d = 0; d < D; ++d) ss += xr[d] * xr[d];
  const float inv = rsqrtf(ss / D + eps);
  float* yr = y + (size_t)t * D;
  for (int d = 0; d < D; ++d) yr[d] = xr[d] * inv * w[d];
}

// Copy the k_pe slice [soff, soff+D) of each row out to a contiguous [T,D] buffer.
__global__ inline void copy_strided(const float* src, int sstride, int soff, float* dst, int T,
                                    int D) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= T * D) return;
  const int t = i / D, d = i % D;
  dst[(size_t)t * D + d] = src[(size_t)t * sstride + soff + d];
}

// Interleaved-complex RoPE on the rope segment [seg, seg+rope_dim) of a [rows, stride] buffer: pair
// (2p, 2p+1) rotates by (row_position * inv_freq[p]) then scales by `scaling`. row_position = row /
// tokens_per_row (q has nh rows per token; the shared k_pe has 1).
__global__ inline void rope_interleaved(float* buf, int rows, int stride, int seg, int rope_dim,
                                        int tokens_per_row, const float* inv_freq, float scaling) {
  const int half = rope_dim / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * half) return;
  const int p = idx % half, row = idx / half;
  const int t = row / tokens_per_row;
  float* b = buf + (size_t)row * stride + seg;
  const float ang = t * inv_freq[p], c = cosf(ang), s = sinf(ang);
  const float x0 = b[2 * p], x1 = b[2 * p + 1];
  b[2 * p] = scaling * (x0 * c - x1 * s);
  b[2 * p + 1] = scaling * (x0 * s + x1 * c);
}

// One thread per (query i, head h). q is [T, nh*qkhd] (nope|rope inline, rope already rotated); kv is
// [T, nh*(qk_nope+v_head)] (per head [k_nope | v]); kpe is [T, qk_rope] (rotated, shared).
__global__ inline void ds_attn(const float* q, const float* kv, const float* kpe, float* out, int T,
                               int nh, int qk_nope, int qk_rope, int v_head, float scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * nh) return;
  const int h = idx % nh, i = idx / nh;
  const int qkhd = qk_nope + qk_rope, kvh = qk_nope + v_head;
  float sc[256];
  float mx = -1e30f;
  const float* qi = q + (size_t)(i * nh + h) * qkhd;
  for (int j = 0; j <= i; ++j) {
    const float* kn = kv + (size_t)(j * nh + h) * kvh;   // k_nope = first qk_nope
    const float* kr = kpe + (size_t)j * qk_rope;
    float d = 0.0f;
    for (int a = 0; a < qk_nope; ++a) d += qi[a] * kn[a];
    for (int r = 0; r < qk_rope; ++r) d += qi[qk_nope + r] * kr[r];
    d *= scale;
    sc[j] = d;
    if (d > mx) mx = d;
  }
  float sum = 0.0f;
  for (int j = 0; j <= i; ++j) {
    sc[j] = expf(sc[j] - mx);
    sum += sc[j];
  }
  float* o = out + (size_t)(i * nh + h) * v_head;
  for (int b = 0; b < v_head; ++b) o[b] = 0.0f;
  for (int j = 0; j <= i; ++j) {
    const float a = sc[j] / sum;
    const float* vj = kv + (size_t)(j * nh + h) * kvh + qk_nope;  // v = after k_nope
    for (int b = 0; b < v_head; ++b) o[b] += a * vj[b];
  }
}

inline float* alloc(size_t n) {
  float* p = nullptr;
  cudaMalloc(&p, n * sizeof(float));
  return p;
}

}  // namespace dsmla_detail

// Device DeepSeek MLA prefill: d_hidden [T,H] -> d_out [T,H]. All device pointers.
inline void mla_prefill_deepseek(const DSMLADims& m, const DSMLAWeights& w, const float* d_hidden,
                                 int T, float* d_out) {
  using namespace dsmla_detail;
  using engine::mla_detail::mm_wt;
  const int QKHD = m.qkhd(), KVH = m.kvh(), KVA = m.kva();

  float* q = alloc((size_t)T * m.nh * QKHD);
  float* ckv = alloc((size_t)T * KVA);
  float* latent = alloc((size_t)T * m.kv_lora);
  float* kpe = alloc((size_t)T * m.qk_rope);
  float* kv = alloc((size_t)T * m.nh * KVH);
  float* outh = alloc((size_t)T * m.nh * m.v_head);

  mm_wt(d_hidden, w.q_proj, q, T, m.nh * QKHD, m.H);
  mm_wt(d_hidden, w.kv_a, ckv, T, KVA, m.H);
  // latent = RMSNorm(ckv[:, :kv_lora]); kpe = ckv[:, kv_lora:]
  rmsnorm_strided<<<(T + 63) / 64, 64>>>(ckv, KVA, 0, w.kv_a_ln, latent, T, m.kv_lora, m.rms_eps);
  copy_strided<<<(T * m.qk_rope + 127) / 128, 128>>>(ckv, KVA, m.kv_lora, kpe, T, m.qk_rope);
  mm_wt(latent, w.kv_b, kv, T, m.nh * KVH, m.kv_lora);
  // RoPE (interleaved, mscale-scaled): q's rope segment per head, and the shared k_pe.
  const int rq = T * m.nh * (m.qk_rope / 2);
  rope_interleaved<<<(rq + 127) / 128, 128>>>(q, T * m.nh, QKHD, m.qk_nope, m.qk_rope, m.nh,
                                              w.inv_freq, m.attn_scaling);
  const int rk = T * (m.qk_rope / 2);
  rope_interleaved<<<(rk + 127) / 128, 128>>>(kpe, T, m.qk_rope, 0, m.qk_rope, 1, w.inv_freq,
                                              m.attn_scaling);
  ds_attn<<<(T * m.nh + 63) / 64, 64>>>(q, kv, kpe, outh, T, m.nh, m.qk_nope, m.qk_rope, m.v_head,
                                        m.softmax_scale);
  mm_wt(outh, w.o_proj, d_out, T, m.H, m.nh * m.v_head);

  cudaFree(q);
  cudaFree(ckv);
  cudaFree(latent);
  cudaFree(kpe);
  cudaFree(kv);
  cudaFree(outh);
}

}  // namespace engine
