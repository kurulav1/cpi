#pragma once

// Reusable Multi-head Latent Attention (MLA) forward, the DeepSeek-V2/V3/R1 attention, promoted out
// of mla_attention_test so it can be composed into a decoder stack (deepseek_mla_stack_test) and,
// once a native-MLA checkpoint is in hand, wired into the engine. fp32, "naive reconstruction"
// form: cache the low-rank latent c_KV + the shared decoupled RoPE key k_R, up-project per-head K/V
// each step. See mla_attention_test for the standalone device-vs-oracle correctness proof (max rel
// ~1.9e-7).
//
// Provides both a device path (small hand-rolled kernels) and an independent host path, so a caller
// can oracle one against the other. Scratch is allocated per call (verification/prototype code, not
// the hot path). Layouts are row-major; weights are [out, in] (y_n = dot(row, weight_n)).

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <vector>

namespace engine {

struct MLADims {
  int H = 0;        // hidden size
  int nh = 0;       // attention heads
  int q_lora = 0;   // query down-projection rank
  int kv_lora = 0;  // KV down-projection rank (the cached latent width)
  int qk_nope = 0;  // per-head non-rope query/key dim
  int qk_rope = 0;  // per-head decoupled-rope query/key dim (k side shared across heads)
  int v_head = 0;   // per-head value dim
  float rope_theta = 10000.0f;
  int hd() const {
    return qk_nope + qk_rope;
  }
};

// Device pointers for the device path, host pointers for the host path; same struct, caller
// supplies the right memory space.
struct MLAWeights {
  const float* WDQ = nullptr;   // [q_lora, H]
  const float* WUQ = nullptr;   // [nh*hd, q_lora]
  const float* WDKV = nullptr;  // [kv_lora, H]
  const float* WKR = nullptr;   // [qk_rope, H]
  const float* WUK = nullptr;   // [nh*qk_nope, kv_lora]
  const float* WUV = nullptr;   // [nh*v_head, kv_lora]
  const float* WO = nullptr;    // [H, nh*v_head]
};

namespace mla_detail {

__global__ inline void mm_wt_kernel(const float* A, const float* W, float* C, int M, int N, int K) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;
  float s = 0.0f;
  for (int k = 0; k < K; ++k) s += A[(size_t)m * K + k] * W[(size_t)n * K + k];
  C[(size_t)m * N + n] = s;
}

__global__ inline void rope_kernel(float* buf, int rows, int stride, int seg, int rope_dim,
                                   int tokens_per_row, float theta) {
  const int half = rope_dim / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * half) return;
  const int p = idx % half;
  const int row = idx / half;
  const int t = row / tokens_per_row;
  float* base = buf + (size_t)row * stride + seg;
  const float freq = powf(theta, -2.0f * p / rope_dim);
  const float ang = t * freq, c = cosf(ang), s = sinf(ang);
  const float x0 = base[p], x1 = base[p + half];
  base[p] = x0 * c - x1 * s;
  base[p + half] = x1 * c + x0 * s;
}

__global__ inline void attn_kernel(const float* q, const float* knope, const float* krope,
                                   const float* v, float* out, int T, int nh, int hd, int nope,
                                   int rope, int vhd, float scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * nh) return;
  const int h = idx % nh, i = idx / nh;
  float sc[256];
  float mx = -1e30f;
  for (int j = 0; j <= i; ++j) {
    const float* qi = q + (size_t)(i * nh + h) * hd;
    const float* kn = knope + (size_t)(j * nh + h) * nope;
    const float* kr = krope + (size_t)j * rope;
    float d = 0.0f;
    for (int a = 0; a < nope; ++a) d += qi[a] * kn[a];
    for (int r = 0; r < rope; ++r) d += qi[nope + r] * kr[r];
    d *= scale;
    sc[j] = d;
    if (d > mx) mx = d;
  }
  float sum = 0.0f;
  for (int j = 0; j <= i; ++j) {
    sc[j] = expf(sc[j] - mx);
    sum += sc[j];
  }
  float* o = out + (size_t)(i * nh + h) * vhd;
  for (int b = 0; b < vhd; ++b) o[b] = 0.0f;
  for (int j = 0; j <= i; ++j) {
    const float a = sc[j] / sum;
    const float* vj = v + (size_t)(j * nh + h) * vhd;
    for (int b = 0; b < vhd; ++b) o[b] += a * vj[b];
  }
}

inline void mm_wt(const float* A, const float* W, float* C, int M, int N, int K) {
  dim3 blk(16, 16), grd((N + 15) / 16, (M + 15) / 16);
  mm_wt_kernel<<<grd, blk>>>(A, W, C, M, N, K);
}
inline float* alloc(size_t n) {
  float* p = nullptr;
  cudaMalloc(&p, n * sizeof(float));
  return p;
}

// Host mirrors (independent implementation; valid as an oracle for the device path).
inline void mm_host(const float* A, const float* W, float* C, int M, int N, int K) {
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float s = 0;
      for (int k = 0; k < K; ++k) s += A[(size_t)m * K + k] * W[(size_t)n * K + k];
      C[(size_t)m * N + n] = s;
    }
}
inline void rope_host(std::vector<float>& buf, int rows, int stride, int seg, int rope_dim,
                      int tokens_per_row, float theta) {
  const int half = rope_dim / 2;
  for (int row = 0; row < rows; ++row) {
    const int t = row / tokens_per_row;
    float* base = buf.data() + (size_t)row * stride + seg;
    for (int p = 0; p < half; ++p) {
      const float freq = std::pow(theta, -2.0f * p / rope_dim);
      const float ang = t * freq, c = std::cos(ang), s = std::sin(ang);
      const float x0 = base[p], x1 = base[p + half];
      base[p] = x0 * c - x1 * s;
      base[p + half] = x1 * c + x0 * s;
    }
  }
}

}  // namespace mla_detail

// Device MLA prefill: d_hidden [T,H] -> d_out [T,H]. All pointers on device 0.
inline void mla_prefill(const MLADims& m, const MLAWeights& w, const float* d_hidden, int T,
                        float* d_out) {
  using namespace mla_detail;
  const int hd = m.hd();
  float* cQ = alloc((size_t)T * m.q_lora);
  float* q = alloc((size_t)T * m.nh * hd);
  float* cKV = alloc((size_t)T * m.kv_lora);
  float* kR = alloc((size_t)T * m.qk_rope);
  float* knope = alloc((size_t)T * m.nh * m.qk_nope);
  float* v = alloc((size_t)T * m.nh * m.v_head);
  float* outh = alloc((size_t)T * m.nh * m.v_head);

  mm_wt(d_hidden, w.WDQ, cQ, T, m.q_lora, m.H);
  mm_wt(cQ, w.WUQ, q, T, m.nh * hd, m.q_lora);
  mm_wt(d_hidden, w.WDKV, cKV, T, m.kv_lora, m.H);
  mm_wt(d_hidden, w.WKR, kR, T, m.qk_rope, m.H);
  mm_wt(cKV, w.WUK, knope, T, m.nh * m.qk_nope, m.kv_lora);
  mm_wt(cKV, w.WUV, v, T, m.nh * m.v_head, m.kv_lora);
  const int rq = T * m.nh * (m.qk_rope / 2);
  rope_kernel<<<(rq + 127) / 128, 128>>>(q, T * m.nh, hd, m.qk_nope, m.qk_rope, m.nh, m.rope_theta);
  const int rk = T * (m.qk_rope / 2);
  rope_kernel<<<(rk + 127) / 128, 128>>>(kR, T, m.qk_rope, 0, m.qk_rope, 1, m.rope_theta);
  const float scale = 1.0f / std::sqrt((float)hd);
  attn_kernel<<<(T * m.nh + 63) / 64, 64>>>(q, knope, kR, v, outh, T, m.nh, hd, m.qk_nope,
                                            m.qk_rope, m.v_head, scale);
  mm_wt(outh, w.WO, d_out, T, m.H, m.nh * m.v_head);

  cudaFree(cQ);
  cudaFree(q);
  cudaFree(cKV);
  cudaFree(kR);
  cudaFree(knope);
  cudaFree(v);
  cudaFree(outh);
}

// Host MLA prefill (independent oracle): h_hidden [T,H] -> h_out [T,H], all host memory.
inline void mla_prefill_host(const MLADims& m, const MLAWeights& w, const float* h_hidden, int T,
                             float* h_out) {
  using namespace mla_detail;
  const int hd = m.hd();
  std::vector<float> cQ((size_t)T * m.q_lora), q((size_t)T * m.nh * hd), cKV((size_t)T * m.kv_lora),
      kR((size_t)T * m.qk_rope), knope((size_t)T * m.nh * m.qk_nope),
      v((size_t)T * m.nh * m.v_head), outh((size_t)T * m.nh * m.v_head, 0.0f);
  mm_host(h_hidden, w.WDQ, cQ.data(), T, m.q_lora, m.H);
  mm_host(cQ.data(), w.WUQ, q.data(), T, m.nh * hd, m.q_lora);
  mm_host(h_hidden, w.WDKV, cKV.data(), T, m.kv_lora, m.H);
  mm_host(h_hidden, w.WKR, kR.data(), T, m.qk_rope, m.H);
  mm_host(cKV.data(), w.WUK, knope.data(), T, m.nh * m.qk_nope, m.kv_lora);
  mm_host(cKV.data(), w.WUV, v.data(), T, m.nh * m.v_head, m.kv_lora);
  rope_host(q, T * m.nh, hd, m.qk_nope, m.qk_rope, m.nh, m.rope_theta);
  rope_host(kR, T, m.qk_rope, 0, m.qk_rope, 1, m.rope_theta);
  const float scale = 1.0f / std::sqrt((float)hd);
  for (int i = 0; i < T; ++i)
    for (int h = 0; h < m.nh; ++h) {
      std::vector<float> sc(i + 1);
      float mx = -1e30f;
      for (int j = 0; j <= i; ++j) {
        const float* qi = q.data() + (size_t)(i * m.nh + h) * hd;
        const float* kn = knope.data() + (size_t)(j * m.nh + h) * m.qk_nope;
        const float* kr = kR.data() + (size_t)j * m.qk_rope;
        float d = 0;
        for (int a = 0; a < m.qk_nope; ++a) d += qi[a] * kn[a];
        for (int r = 0; r < m.qk_rope; ++r) d += qi[m.qk_nope + r] * kr[r];
        d *= scale;
        sc[j] = d;
        if (d > mx) mx = d;
      }
      float sum = 0;
      for (float& x : sc) {
        x = std::exp(x - mx);
        sum += x;
      }
      float* o = outh.data() + (size_t)(i * m.nh + h) * m.v_head;
      for (int j = 0; j <= i; ++j) {
        const float a = sc[j] / sum;
        const float* vj = v.data() + (size_t)(j * m.nh + h) * m.v_head;
        for (int b = 0; b < m.v_head; ++b) o[b] += a * vj[b];
      }
    }
  mm_host(outh.data(), w.WO, h_out, T, m.H, m.nh * m.v_head);
}

}  // namespace engine
