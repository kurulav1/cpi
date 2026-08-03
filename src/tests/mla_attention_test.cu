// Verifies Multi-head Latent Attention (MLA, the DeepSeek-V2/V3/R1 attention) on-device against a host
// fp32 oracle -- the isolated, model-blocked groundwork for supporting native DeepSeek-V3/R1 (whose
// full checkpoints don't fit this box), the same way the MXFP4 and DeepSeek-router bricks were landed.
//
// MLA compresses K and V into a shared low-rank latent c_KV (dim kv_lora_rank) plus a small decoupled
// RoPE key k_R shared across heads; the per-head keys/values are RECONSTRUCTED from that latent, so the
// KV cache stores only (kv_lora_rank + qk_rope) numbers per token instead of the full per-head K,V.
// This brick implements the "naive reconstruction" form (cache the latent, up-project each step) in
// fp32 with small hand-rolled kernels, and checks the full prefill forward equals the oracle. It also
// asserts the cache-compression ratio -- the whole point of MLA.
//
// Per head h, query position i, key position j (causal j<=i):
//   c_Q  = W_DQ  h                       (query down-projection, dim q_lora_rank)
//   q    = W_UQ  c_Q                      (up -> per head [q_nope (qk_nope) | q_rope (qk_rope)])
//   c_KV = W_DKV h                        (KV down-projection, dim kv_lora_rank)  << cached
//   k_R  = RoPE(W_KR h)                   (decoupled rope key, dim qk_rope, shared) << cached
//   k_nope = W_UK c_KV                    (per head, dim qk_nope)   -- reconstructed
//   v      = W_UV c_KV                    (per head, dim v_head_dim) -- reconstructed
//   score  = (q_nope . k_nope + RoPE(q_rope) . k_R) / sqrt(qk_nope+qk_rope)
//   out    = softmax_j(score) . v ; then  y = W_O concat_h(out)
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int MAXT = 64;

// C[M,N] = A[M,K] . W[N,K]^T  (row-major; W stored as [N,K], i.e. y_n = dot(row_m, weight_n)).
__global__ void k_mm_wt(const float* A, const float* W, float* C, int M, int N, int K) {
  const int n = blockIdx.x * blockDim.x + threadIdx.x;
  const int m = blockIdx.y * blockDim.y + threadIdx.y;
  if (m >= M || n >= N) return;
  float s = 0.0f;
  for (int k = 0; k < K; ++k) s += A[(size_t)m * K + k] * W[(size_t)n * K + k];
  C[(size_t)m * N + n] = s;
}

// Half-split RoPE (position = token index) applied in place to the rope segment of a [rows, stride]
// buffer. q: one segment per (t,h) at offset `seg` within each head's head_dim row. k: the whole row.
__global__ void k_rope(float* buf, int rows, int stride, int seg, int rope_dim, int tokens_per_row,
                       float theta) {
  const int half = rope_dim / 2;
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * half) return;
  const int p = idx % half;
  const int row = idx / half;
  const int t = row / tokens_per_row;  // token position for this row
  float* base = buf + (size_t)row * stride + seg;
  const float freq = powf(theta, -2.0f * p / rope_dim);
  const float ang = t * freq;
  const float c = cosf(ang), s = sinf(ang);
  const float x0 = base[p], x1 = base[p + half];
  base[p] = x0 * c - x1 * s;
  base[p + half] = x1 * c + x0 * s;
}

// One thread per (query i, head h): causal softmax over keys and the value-weighted sum.
__global__ void k_mla_attn(const float* q, const float* knope, const float* krope, const float* v,
                           float* out, int T, int nh, int hd, int nope, int rope, int vhd,
                           float scale) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= T * nh) return;
  const int h = idx % nh;
  const int i = idx / nh;
  float sc[MAXT];
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

struct MLA {
  int H, nh, dq, dc, qk_nope, qk_rope, v_hd;
  int hd() const { return qk_nope + qk_rope; }
  float theta = 10000.0f;
};

void mm_wt(const float* A, const float* W, float* C, int M, int N, int K) {
  dim3 blk(16, 16), grd((N + 15) / 16, (M + 15) / 16);
  k_mm_wt<<<grd, blk>>>(A, W, C, M, N, K);
}

// ---- host fp32 oracle: the same forward, in plain C++ ----
void rope_host(std::vector<float>& buf, int rows, int stride, int seg, int rope_dim,
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
std::vector<float> mm_host(const std::vector<float>& A, const std::vector<float>& W, int M, int N,
                           int K) {
  std::vector<float> C((size_t)M * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      float s = 0;
      for (int k = 0; k < K; ++k) s += A[(size_t)m * K + k] * W[(size_t)n * K + k];
      C[(size_t)m * N + n] = s;
    }
  return C;
}

std::vector<float> oracle(const MLA& m, int T, const std::vector<float>& hid,
                          const std::vector<float>& WDQ, const std::vector<float>& WUQ,
                          const std::vector<float>& WDKV, const std::vector<float>& WKR,
                          const std::vector<float>& WUK, const std::vector<float>& WUV,
                          const std::vector<float>& WO) {
  const int hd = m.hd();
  auto cQ = mm_host(hid, WDQ, T, m.dq, m.H);
  auto q = mm_host(cQ, WUQ, T, m.nh * hd, m.dq);
  auto cKV = mm_host(hid, WDKV, T, m.dc, m.H);
  auto kR = mm_host(hid, WKR, T, m.qk_rope, m.H);
  auto knope = mm_host(cKV, WUK, T, m.nh * m.qk_nope, m.dc);
  auto v = mm_host(cKV, WUV, T, m.nh * m.v_hd, m.dc);
  // RoPE q_rope segment (per head) and the shared k_R.
  rope_host(q, T * m.nh, hd, m.qk_nope, m.qk_rope, m.nh, m.theta);
  rope_host(kR, T, m.qk_rope, 0, m.qk_rope, 1, m.theta);
  // Attention.
  const float scale = 1.0f / std::sqrt((float)hd);
  std::vector<float> outh((size_t)T * m.nh * m.v_hd, 0.0f);
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
        mx = std::max(mx, d);
      }
      float sum = 0;
      for (float& x : sc) {
        x = std::exp(x - mx);
        sum += x;
      }
      float* o = outh.data() + (size_t)(i * m.nh + h) * m.v_hd;
      for (int j = 0; j <= i; ++j) {
        const float a = sc[j] / sum;
        const float* vj = v.data() + (size_t)(j * m.nh + h) * m.v_hd;
        for (int b = 0; b < m.v_hd; ++b) o[b] += a * vj[b];
      }
    }
  return mm_host(outh, WO, T, m.H, m.nh * m.v_hd);
}

float* dev(const std::vector<float>& h) {
  float* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(float));
  cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
  return d;
}

}  // namespace

int main() {
  MLA m{/*H*/ 128, /*nh*/ 4, /*dq*/ 24, /*dc*/ 32, /*qk_nope*/ 16, /*qk_rope*/ 8, /*v_hd*/ 16};
  const int T = 5, hd = m.hd();
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 0.2f);
  auto rnd = [&](int n) {
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
  };
  auto hid = rnd(T * m.H);
  auto WDQ = rnd(m.dq * m.H), WUQ = rnd(m.nh * hd * m.dq), WDKV = rnd(m.dc * m.H),
       WKR = rnd(m.qk_rope * m.H), WUK = rnd(m.nh * m.qk_nope * m.dc),
       WUV = rnd(m.nh * m.v_hd * m.dc), WO = rnd(m.H * m.nh * m.v_hd);

  const std::vector<float> ref =
      oracle(m, T, hid, WDQ, WUQ, WDKV, WKR, WUK, WUV, WO);

  // ---- device forward ----
  float *dHid = dev(hid), *dWDQ = dev(WDQ), *dWUQ = dev(WUQ), *dWDKV = dev(WDKV), *dWKR = dev(WKR),
        *dWUK = dev(WUK), *dWUV = dev(WUV), *dWO = dev(WO);
  float *dCQ, *dQ, *dCKV, *dKR, *dKnope, *dV, *dOutH, *dY;
  cudaMalloc(&dCQ, (size_t)T * m.dq * sizeof(float));
  cudaMalloc(&dQ, (size_t)T * m.nh * hd * sizeof(float));
  cudaMalloc(&dCKV, (size_t)T * m.dc * sizeof(float));
  cudaMalloc(&dKR, (size_t)T * m.qk_rope * sizeof(float));
  cudaMalloc(&dKnope, (size_t)T * m.nh * m.qk_nope * sizeof(float));
  cudaMalloc(&dV, (size_t)T * m.nh * m.v_hd * sizeof(float));
  cudaMalloc(&dOutH, (size_t)T * m.nh * m.v_hd * sizeof(float));
  cudaMalloc(&dY, (size_t)T * m.H * sizeof(float));

  mm_wt(dHid, dWDQ, dCQ, T, m.dq, m.H);
  mm_wt(dCQ, dWUQ, dQ, T, m.nh * hd, m.dq);
  mm_wt(dHid, dWDKV, dCKV, T, m.dc, m.H);
  mm_wt(dHid, dWKR, dKR, T, m.qk_rope, m.H);
  mm_wt(dCKV, dWUK, dKnope, T, m.nh * m.qk_nope, m.dc);
  mm_wt(dCKV, dWUV, dV, T, m.nh * m.v_hd, m.dc);
  // RoPE.
  {
    const int rq = T * m.nh * (m.qk_rope / 2);
    k_rope<<<(rq + 127) / 128, 128>>>(dQ, T * m.nh, hd, m.qk_nope, m.qk_rope, m.nh, m.theta);
    const int rk = T * (m.qk_rope / 2);
    k_rope<<<(rk + 127) / 128, 128>>>(dKR, T, m.qk_rope, 0, m.qk_rope, 1, m.theta);
  }
  const float scale = 1.0f / std::sqrt((float)hd);
  k_mla_attn<<<(T * m.nh + 63) / 64, 64>>>(dQ, dKnope, dKR, dV, dOutH, T, m.nh, hd, m.qk_nope,
                                           m.qk_rope, m.v_hd, scale);
  mm_wt(dOutH, dWO, dY, T, m.H, m.nh * m.v_hd);
  cudaDeviceSynchronize();

  std::vector<float> y((size_t)T * m.H);
  cudaMemcpy(y.data(), dY, y.size() * sizeof(float), cudaMemcpyDeviceToHost);

  float maxabs = 0, denom = 1e-6f;
  for (size_t i = 0; i < ref.size(); ++i) {
    maxabs = std::max(maxabs, std::fabs(ref[i] - y[i]));
    denom = std::max(denom, std::fabs(ref[i]));
  }
  const float rel = maxabs / denom;
  const bool pass = rel < 1e-4f;
  std::printf("%s[MLA prefill]: device vs fp32 oracle, T=%d heads=%d qk=%d(+%d rope) v=%d, max rel %.2e\n",
              pass ? "PASS" : "FAIL", T, m.nh, m.qk_nope, m.qk_rope, m.v_hd, rel);

  // The point of MLA: the KV cache stores the latent (dc + qk_rope) per token, not the full per-head
  // K and V (nh * (hd + v_hd)). Assert the compression.
  const int mla_cache = m.dc + m.qk_rope;
  const int full_cache = m.nh * (hd + m.v_hd);
  std::printf("      KV cache/token: MLA=%d floats vs full MHA=%d floats -> %.2fx smaller\n",
              mla_cache, full_cache, (double)full_cache / mla_cache);
  const bool win = mla_cache < full_cache;
  if (!win) std::printf("FAIL: MLA cache is not smaller\n");

  return (pass && win) ? 0 : 1;
}
