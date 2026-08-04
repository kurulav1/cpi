// Composes the reusable MLA op (engine/mla_forward.hpp) into a small multi-layer DeepSeek-style decoder
// stack and verifies the whole forward on-device against a host oracle. This is the "wire MLA toward a
// DeepSeek-arch config" step: structurally a real DeepSeek decoder; pre-norm layers of
//   h += MLA(RMSNorm(h));  h += SwiGLU-MLP(RMSNorm(h))
//; on a small SYNTHETIC config with random weights (a native-MLA checkpoint like DeepSeek-V2-Lite
// doesn't reside on this box). It proves the MLA op composes correctly across residual layers with
// norms and MLP; everything except the real weights, which is the only remaining end-to-end gate.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "engine/mla_forward.hpp"

namespace {

using engine::mla_detail::mm_host;
using engine::mla_detail::mm_wt;

__global__ void k_rmsnorm(const float* x, const float* w, float* y, int T, int H, float eps) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  const float* xr = x + (size_t)t * H;
  float* yr = y + (size_t)t * H;
  float ss = 0.0f;
  for (int d = 0; d < H; ++d) ss += xr[d] * xr[d];
  const float inv = rsqrtf(ss / H + eps);
  for (int d = 0; d < H; ++d) yr[d] = xr[d] * inv * w[d];
}
__global__ void k_silu_mul(const float* g, const float* u, float* y, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float gv = g[i];
  y[i] = (gv / (1.0f + expf(-gv))) * u[i];
}
__global__ void k_add(float* a, const float* b, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) a[i] += b[i];
}

void rmsnorm_host(const std::vector<float>& x, const std::vector<float>& w, std::vector<float>& y,
                  int T, int H, float eps) {
  for (int t = 0; t < T; ++t) {
    float ss = 0;
    for (int d = 0; d < H; ++d) ss += x[t * H + d] * x[t * H + d];
    const float inv = 1.0f / std::sqrt(ss / H + eps);
    for (int d = 0; d < H; ++d) y[t * H + d] = x[t * H + d] * inv * w[d];
  }
}
float silu(float v) { return v / (1.0f + std::exp(-v)); }

float* dev(const std::vector<float>& h) {
  float* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(float));
  cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
  return d;
}

struct LayerH {  // host weights for one layer
  std::vector<float> ln1, ln2, WDQ, WUQ, WDKV, WKR, WUK, WUV, WO, Wg, Wu, Wd;
};

}  // namespace

int main() {
  engine::MLADims m{/*H*/ 96, /*nh*/ 4, /*q_lora*/ 24, /*kv_lora*/ 32,
                    /*qk_nope*/ 16, /*qk_rope*/ 8, /*v_head*/ 16};
  const int T = 5, L = 3, ffn = 192, hd = m.hd();
  const float eps = 1e-6f;
  std::mt19937 rng(21);
  std::normal_distribution<float> nd(0.0f, 0.2f);
  auto rnd = [&](int n) {
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
  };
  auto ones_ish = [&](int n) {  // norm weights near 1
    std::vector<float> v(n);
    for (auto& x : v) x = 1.0f + 0.05f * nd(rng);
    return v;
  };

  std::vector<LayerH> layers(L);
  for (auto& ly : layers) {
    ly.ln1 = ones_ish(m.H);
    ly.ln2 = ones_ish(m.H);
    ly.WDQ = rnd(m.q_lora * m.H);
    ly.WUQ = rnd(m.nh * hd * m.q_lora);
    ly.WDKV = rnd(m.kv_lora * m.H);
    ly.WKR = rnd(m.qk_rope * m.H);
    ly.WUK = rnd(m.nh * m.qk_nope * m.kv_lora);
    ly.WUV = rnd(m.nh * m.v_head * m.kv_lora);
    ly.WO = rnd(m.H * m.nh * m.v_head);
    ly.Wg = rnd(ffn * m.H);
    ly.Wu = rnd(ffn * m.H);
    ly.Wd = rnd(m.H * ffn);
  }
  const auto lnf = ones_ish(m.H);
  const auto x0 = rnd(T * m.H);

  // ---------- host oracle ----------
  std::vector<float> h = x0;
  for (const auto& ly : layers) {
    std::vector<float> xn(T * m.H), attn(T * m.H);
    rmsnorm_host(h, ly.ln1, xn, T, m.H, eps);
    engine::MLAWeights hw{ly.WDQ.data(), ly.WUQ.data(), ly.WDKV.data(), ly.WKR.data(),
                          ly.WUK.data(), ly.WUV.data(), ly.WO.data()};
    engine::mla_prefill_host(m, hw, xn.data(), T, attn.data());
    for (int i = 0; i < T * m.H; ++i) h[i] += attn[i];
    // SwiGLU MLP
    rmsnorm_host(h, ly.ln2, xn, T, m.H, eps);
    std::vector<float> g(T * ffn), u(T * ffn), act(T * ffn), mlp(T * m.H);
    mm_host(xn.data(), ly.Wg.data(), g.data(), T, ffn, m.H);
    mm_host(xn.data(), ly.Wu.data(), u.data(), T, ffn, m.H);
    for (int i = 0; i < T * ffn; ++i) act[i] = silu(g[i]) * u[i];
    mm_host(act.data(), ly.Wd.data(), mlp.data(), T, m.H, ffn);
    for (int i = 0; i < T * m.H; ++i) h[i] += mlp[i];
  }
  std::vector<float> ref(T * m.H);
  rmsnorm_host(h, lnf, ref, T, m.H, eps);

  // ---------- device forward ----------
  float* dh = dev(x0);
  float* dxn = nullptr;
  float *dattn, *dg, *du, *dact, *dmlp;
  cudaMalloc(&dxn, (size_t)T * m.H * sizeof(float));
  cudaMalloc(&dattn, (size_t)T * m.H * sizeof(float));
  cudaMalloc(&dg, (size_t)T * ffn * sizeof(float));
  cudaMalloc(&du, (size_t)T * ffn * sizeof(float));
  cudaMalloc(&dact, (size_t)T * ffn * sizeof(float));
  cudaMalloc(&dmlp, (size_t)T * m.H * sizeof(float));
  const int NH = T * m.H, NF = T * ffn;
  for (const auto& ly : layers) {
    float *dln1 = dev(ly.ln1), *dln2 = dev(ly.ln2), *dWDQ = dev(ly.WDQ), *dWUQ = dev(ly.WUQ),
          *dWDKV = dev(ly.WDKV), *dWKR = dev(ly.WKR), *dWUK = dev(ly.WUK), *dWUV = dev(ly.WUV),
          *dWO = dev(ly.WO), *dWg = dev(ly.Wg), *dWu = dev(ly.Wu), *dWd = dev(ly.Wd);
    k_rmsnorm<<<(T + 63) / 64, 64>>>(dh, dln1, dxn, T, m.H, eps);
    engine::MLAWeights dw{dWDQ, dWUQ, dWDKV, dWKR, dWUK, dWUV, dWO};
    engine::mla_prefill(m, dw, dxn, T, dattn);
    k_add<<<(NH + 127) / 128, 128>>>(dh, dattn, NH);
    k_rmsnorm<<<(T + 63) / 64, 64>>>(dh, dln2, dxn, T, m.H, eps);
    mm_wt(dxn, dWg, dg, T, ffn, m.H);
    mm_wt(dxn, dWu, du, T, ffn, m.H);
    k_silu_mul<<<(NF + 127) / 128, 128>>>(dg, du, dact, NF);
    mm_wt(dact, dWd, dmlp, T, m.H, ffn);
    k_add<<<(NH + 127) / 128, 128>>>(dh, dmlp, NH);
    cudaDeviceSynchronize();
    for (float* p : {dln1, dln2, dWDQ, dWUQ, dWDKV, dWKR, dWUK, dWUV, dWO, dWg, dWu, dWd})
      cudaFree(p);
  }
  float* dlnf = dev(lnf);
  float* dref = nullptr;
  cudaMalloc(&dref, (size_t)T * m.H * sizeof(float));
  k_rmsnorm<<<(T + 63) / 64, 64>>>(dh, dlnf, dref, T, m.H, eps);
  cudaDeviceSynchronize();
  std::vector<float> y(T * m.H);
  cudaMemcpy(y.data(), dref, y.size() * sizeof(float), cudaMemcpyDeviceToHost);

  float maxabs = 0, denom = 1e-6f;
  for (int i = 0; i < T * m.H; ++i) {
    maxabs = std::max(maxabs, std::fabs(ref[i] - y[i]));
    denom = std::max(denom, std::fabs(ref[i]));
  }
  const float rel = maxabs / denom;
  const bool pass = rel < 1e-4f;
  std::printf("%s[DeepSeek MLA stack]: %d layers (MLA+SwiGLU, pre-norm) device vs oracle, max rel %.2e\n",
              pass ? "PASS" : "FAIL", L, rel);
  return pass ? 0 : 1;
}
