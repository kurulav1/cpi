// Verifies the MlaAssembleRope kernel (engine/mla_assemble.hpp) fp16-on-device vs an fp32 host oracle:
// assemble per-head K = [k_nope | roped shared k_pe] and V = [v | zeros], and rope q_pe in place. This
// is the one new kernel the in-engine DeepSeek MLA adds; everything else reuses existing ops.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "engine/mla_assemble.hpp"

namespace {
__half* devh(const std::vector<__half>& h) {
  __half* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(__half));
  cudaMemcpy(d, h.data(), h.size() * sizeof(__half), cudaMemcpyHostToDevice);
  return d;
}
}  // namespace

int main() {
  const int nh = 4, qk_nope = 16, qk_rope = 8, v_head = 16, kv_lora = 32, position = 3;
  const int qkhd = qk_nope + qk_rope, kvh = qk_nope + v_head, half = qk_rope / 2;
  const float scaling = 1.0f;
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0.0f, 0.4f);
  auto rndh = [&](int n) {
    std::vector<__half> v(n);
    for (auto& x : v) x = __float2half(nd(rng));
    return v;
  };
  auto Q = rndh(nh * qkhd), kvb = rndh(nh * kvh), ckv = rndh(kv_lora + qk_rope);
  std::vector<float> inv_freq(half);
  for (int p = 0; p < half; ++p) inv_freq[p] = std::pow(10000.0f, -2.0f * p / qk_rope);

  // ---- fp32 oracle ----
  auto Qo = Q;  // copy (roped in place)
  std::vector<float> Ko(nh * qkhd, 0), Vo(nh * qkhd, 0);
  for (int h = 0; h < nh; ++h) {
    float* q = reinterpret_cast<float*>(nullptr);  // placeholder to keep structure clear
    (void)q;
    // q_pe rope
    for (int p = 0; p < half; ++p) {
      const float ang = position * inv_freq[p], c = std::cos(ang), s = std::sin(ang);
      const int base = h * qkhd + qk_nope;
      const float x0 = __half2float(Q[base + 2 * p]), x1 = __half2float(Q[base + 2 * p + 1]);
      Qo[base + 2 * p] = __float2half(scaling * (x0 * c - x1 * s));
      Qo[base + 2 * p + 1] = __float2half(scaling * (x0 * s + x1 * c));
    }
    for (int a = 0; a < qk_nope; ++a) Ko[h * qkhd + a] = __half2float(kvb[h * kvh + a]);
    for (int p = 0; p < half; ++p) {
      const float ang = position * inv_freq[p], c = std::cos(ang), s = std::sin(ang);
      const float x0 = __half2float(ckv[kv_lora + 2 * p]), x1 = __half2float(ckv[kv_lora + 2 * p + 1]);
      Ko[h * qkhd + qk_nope + 2 * p] = scaling * (x0 * c - x1 * s);
      Ko[h * qkhd + qk_nope + 2 * p + 1] = scaling * (x0 * s + x1 * c);
    }
    for (int b = 0; b < v_head; ++b) Vo[h * qkhd + b] = __half2float(kvb[h * kvh + qk_nope + b]);
  }

  // ---- device ----
  __half* dQ = devh(Q);
  __half* dkvb = devh(kvb);
  __half* dckv = devh(ckv);
  __half *dK, *dV;
  float* dinv;
  cudaMalloc(&dK, (size_t)nh * qkhd * sizeof(__half));
  cudaMalloc(&dV, (size_t)nh * qkhd * sizeof(__half));
  cudaMalloc(&dinv, half * sizeof(float));
  cudaMemcpy(dinv, inv_freq.data(), half * sizeof(float), cudaMemcpyHostToDevice);
  engine::launch_mla_assemble_rope(dQ, dkvb, dckv, dK, dV, dinv, nh, qk_nope, qk_rope, v_head,
                                   kv_lora, position, scaling, 0);
  cudaDeviceSynchronize();
  std::vector<__half> Kd(nh * qkhd), Vd(nh * qkhd), Qd(nh * qkhd);
  cudaMemcpy(Kd.data(), dK, Kd.size() * sizeof(__half), cudaMemcpyDeviceToHost);
  cudaMemcpy(Vd.data(), dV, Vd.size() * sizeof(__half), cudaMemcpyDeviceToHost);
  cudaMemcpy(Qd.data(), dQ, Qd.size() * sizeof(__half), cudaMemcpyDeviceToHost);

  float maxrel = 0, denom = 1e-4f;
  auto acc = [&](float ref, float got) {
    maxrel = std::max(maxrel, std::fabs(ref - got));
    denom = std::max(denom, std::fabs(ref));
  };
  for (int i = 0; i < nh * qkhd; ++i) {
    acc(Ko[i], __half2float(Kd[i]));
    acc(Vo[i], __half2float(Vd[i]));
    acc(__half2float(Qo[i]), __half2float(Qd[i]));
  }
  const float rel = maxrel / denom;
  const bool pass = rel < 5e-3f;  // fp16 rounding
  std::printf("%s[MLA assemble+rope]: K/V/Q device vs fp32 oracle, max rel %.2e\n",
              pass ? "PASS" : "FAIL", rel);
  return pass ? 0 : 1;
}
