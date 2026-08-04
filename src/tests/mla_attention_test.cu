// Verifies Multi-head Latent Attention (MLA, the DeepSeek-V2/V3/R1 attention) on-device against a host
// fp32 oracle; isolated, model-blocked groundwork for native DeepSeek-V3/R1 (full checkpoints don't
// fit this box), the same way the MXFP4 and DeepSeek-router bricks were landed. The MLA forward itself
// lives in engine/mla_forward.hpp (reused by deepseek_mla_stack_test); this test exercises a single
// layer and cross-checks the header's device path against its independent host path.
//
// MLA compresses K and V into a shared low-rank latent c_KV (dim kv_lora_rank) plus a small decoupled
// RoPE key k_R shared across heads; per-head keys/values are RECONSTRUCTED from that latent, so the KV
// cache stores only (kv_lora_rank + qk_rope) numbers per token instead of the full per-head K,V.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "engine/mla_forward.hpp"

namespace {
float* dev(const std::vector<float>& h) {
  float* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(float));
  cudaMemcpy(d, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice);
  return d;
}
}  // namespace

int main() {
  engine::MLADims m{/*H*/ 128, /*nh*/ 4, /*q_lora*/ 24, /*kv_lora*/ 32,
                    /*qk_nope*/ 16, /*qk_rope*/ 8, /*v_head*/ 16};
  const int T = 5, hd = m.hd();
  std::mt19937 rng(7);
  std::normal_distribution<float> nd(0.0f, 0.2f);
  auto rnd = [&](int n) {
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng);
    return v;
  };
  const auto hid = rnd(T * m.H);
  const auto WDQ = rnd(m.q_lora * m.H), WUQ = rnd(m.nh * hd * m.q_lora), WDKV = rnd(m.kv_lora * m.H),
             WKR = rnd(m.qk_rope * m.H), WUK = rnd(m.nh * m.qk_nope * m.kv_lora),
             WUV = rnd(m.nh * m.v_head * m.kv_lora), WO = rnd(m.H * m.nh * m.v_head);

  // Host oracle via the header's independent host path.
  engine::MLAWeights hw{WDQ.data(), WUQ.data(), WDKV.data(), WKR.data(),
                        WUK.data(), WUV.data(), WO.data()};
  std::vector<float> ref(T * m.H);
  engine::mla_prefill_host(m, hw, hid.data(), T, ref.data());

  // Device path via the header's CUDA kernels.
  float* dHid = dev(hid);
  float *dWDQ = dev(WDQ), *dWUQ = dev(WUQ), *dWDKV = dev(WDKV), *dWKR = dev(WKR), *dWUK = dev(WUK),
        *dWUV = dev(WUV), *dWO = dev(WO), *dY = nullptr;
  cudaMalloc(&dY, (size_t)T * m.H * sizeof(float));
  engine::MLAWeights dw{dWDQ, dWUQ, dWDKV, dWKR, dWUK, dWUV, dWO};
  engine::mla_prefill(m, dw, dHid, T, dY);
  cudaDeviceSynchronize();
  std::vector<float> y(T * m.H);
  cudaMemcpy(y.data(), dY, y.size() * sizeof(float), cudaMemcpyDeviceToHost);

  float maxabs = 0, denom = 1e-6f;
  for (size_t i = 0; i < ref.size(); ++i) {
    maxabs = std::max(maxabs, std::fabs(ref[i] - y[i]));
    denom = std::max(denom, std::fabs(ref[i]));
  }
  const float rel = maxabs / denom;
  const bool pass = rel < 1e-4f;
  std::printf("%s[MLA prefill]: device vs fp32 oracle, T=%d heads=%d qk=%d(+%d rope) v=%d, max rel %.2e\n",
              pass ? "PASS" : "FAIL", T, m.nh, m.qk_nope, m.qk_rope, m.v_head, rel);

  const int mla_cache = m.kv_lora + m.qk_rope;
  const int full_cache = m.nh * (hd + m.v_head);
  std::printf("      KV cache/token: MLA=%d floats vs full MHA=%d floats -> %.2fx smaller\n",
              mla_cache, full_cache, (double)full_cache / mla_cache);
  const bool win = mla_cache < full_cache;
  return (pass && win) ? 0 : 1;
}
