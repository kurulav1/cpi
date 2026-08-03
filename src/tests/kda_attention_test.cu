// Verifies Kimi Delta Attention (KDA) -- the gated delta-rule linear attention -- on-device against a
// host oracle. Isolated, model-blocked groundwork for Kimi K3 (whose checkpoint doesn't fit this box).
// CPI already runs the Qwen3.5 gated delta-net with a PER-HEAD SCALAR decay (verified). KDA's
// distinction is FINE-GRAINED gating: a decay vector over the key channels, so each key dimension of
// the recurrent state forgets at its own rate. This brick implements that per-channel gated delta rule
// as a standalone recurrence and checks it against a plain-C++ oracle.
//
// Per head, running state S[key_dim, value_dim] (no KV cache -- a recurrent state), for token t:
//   S <- diag(g_t) S                       (per-key-channel decay; g_t in (0,1)^key_dim)   << KDA
//   u  = (v_t - Sᵀ k_t) * beta_t           (delta correction, beta_t in (0,1))
//   S <- S + k_t ⊗ u                       (rank-1 update)
//   o_t = Sᵀ q_t                           (readout, value_dim)
// (The QK-norm and the RMS/SiLU output gate CPI applies around this are orthogonal, separately-tested
// stages; this brick isolates the recurrence, which is the KDA-specific part.)
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int KMAX = 16, VMAX = 16;  // state kept in registers/local; tiny by design

// One thread per head: sequential gated delta-rule recurrence over the whole sequence.
__global__ void k_kda(const float* q, const float* k, const float* v, const float* g,
                      const float* beta, float* out, int T, int nh, int kd, int vd) {
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= nh) return;
  float S[KMAX * VMAX];
  for (int i = 0; i < kd * vd; ++i) S[i] = 0.0f;
  for (int t = 0; t < T; ++t) {
    const float* qt = q + (size_t)(t * nh + h) * kd;
    const float* kt = k + (size_t)(t * nh + h) * kd;
    const float* vt = v + (size_t)(t * nh + h) * vd;
    const float* gt = g + (size_t)(t * nh + h) * kd;
    const float bt = beta[t * nh + h];
    // decay (per key channel)
    for (int a = 0; a < kd; ++a)
      for (int d = 0; d < vd; ++d) S[a * vd + d] *= gt[a];
    // u = (v - Sᵀk) * beta
    float u[VMAX];
    for (int d = 0; d < vd; ++d) {
      float m = 0.0f;
      for (int a = 0; a < kd; ++a) m += S[a * vd + d] * kt[a];
      u[d] = (vt[d] - m) * bt;
    }
    // rank-1 update + readout
    float* o = out + (size_t)(t * nh + h) * vd;
    for (int d = 0; d < vd; ++d) o[d] = 0.0f;
    for (int a = 0; a < kd; ++a) {
      const float ka = kt[a], qa = qt[a];
      for (int d = 0; d < vd; ++d) {
        S[a * vd + d] += ka * u[d];
        o[d] += S[a * vd + d] * qa;
      }
    }
  }
}

std::vector<float> oracle(const std::vector<float>& q, const std::vector<float>& k,
                          const std::vector<float>& v, const std::vector<float>& g,
                          const std::vector<float>& beta, int T, int nh, int kd, int vd) {
  std::vector<float> out((size_t)T * nh * vd, 0.0f);
  std::vector<float> S((size_t)kd * vd);
  for (int h = 0; h < nh; ++h) {
    std::fill(S.begin(), S.end(), 0.0f);
    for (int t = 0; t < T; ++t) {
      const float* qt = q.data() + (size_t)(t * nh + h) * kd;
      const float* kt = k.data() + (size_t)(t * nh + h) * kd;
      const float* vt = v.data() + (size_t)(t * nh + h) * vd;
      const float* gt = g.data() + (size_t)(t * nh + h) * kd;
      const float bt = beta[t * nh + h];
      for (int a = 0; a < kd; ++a)
        for (int d = 0; d < vd; ++d) S[a * vd + d] *= gt[a];
      std::vector<float> u(vd);
      for (int d = 0; d < vd; ++d) {
        float m = 0.0f;
        for (int a = 0; a < kd; ++a) m += S[a * vd + d] * kt[a];
        u[d] = (vt[d] - m) * bt;
      }
      float* o = out.data() + (size_t)(t * nh + h) * vd;
      for (int a = 0; a < kd; ++a)
        for (int d = 0; d < vd; ++d) {
          S[a * vd + d] += kt[a] * u[d];
          o[d] += S[a * vd + d] * qt[a];
        }
    }
  }
  return out;
}

float* dev(const std::vector<float>& h) {
  float* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(float));
  cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
  return d;
}

}  // namespace

int main() {
  const int T = 6, nh = 3, kd = 8, vd = 8;
  std::mt19937 rng(13);
  std::normal_distribution<float> nd(0.0f, 0.3f);
  std::uniform_real_distribution<float> ud(0.0f, 1.0f);
  auto rnd = [&](int n) {
    std::vector<float> x(n);
    for (auto& e : x) e = nd(rng);
    return x;
  };
  const auto q = rnd(T * nh * kd), k = rnd(T * nh * kd), v = rnd(T * nh * vd);
  std::vector<float> g(T * nh * kd), beta(T * nh);
  for (auto& e : g) e = 0.85f + 0.149f * ud(rng);  // per-channel decay in (0.85, ~1)
  for (auto& e : beta) e = ud(rng);                // delta strength in (0,1)

  const auto ref = oracle(q, k, v, g, beta, T, nh, kd, vd);

  float *dq = dev(q), *dk = dev(k), *dv = dev(v), *dg = dev(g), *db = dev(beta), *dout = nullptr;
  cudaMalloc(&dout, (size_t)T * nh * vd * sizeof(float));
  k_kda<<<(nh + 63) / 64, 64>>>(dq, dk, dv, dg, db, dout, T, nh, kd, vd);
  cudaDeviceSynchronize();
  std::vector<float> out((size_t)T * nh * vd);
  cudaMemcpy(out.data(), dout, out.size() * sizeof(float), cudaMemcpyDeviceToHost);

  float maxabs = 0, denom = 1e-6f;
  for (size_t i = 0; i < ref.size(); ++i) {
    maxabs = std::max(maxabs, std::fabs(ref[i] - out[i]));
    denom = std::max(denom, std::fabs(ref[i]));
  }
  const float rel = maxabs / denom;
  const bool pass = rel < 1e-4f;
  std::printf(
      "%s[KDA]: per-channel gated delta-rule, device vs oracle, T=%d heads=%d kdim=%d vdim=%d, "
      "max rel %.2e\n",
      pass ? "PASS" : "FAIL", T, nh, kd, vd, rel);
  std::printf("      recurrent state = %d floats/head, NO KV cache (state size independent of T)\n",
              kd * vd);
  return pass ? 0 : 1;
}
