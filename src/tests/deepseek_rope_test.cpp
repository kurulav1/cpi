// Verifies the C++ YARN rope frequencies (engine/deepseek_rope.hpp) reproduce transformers' inv_freq
// for DeepSeek-V2-Lite. The reference (artifacts/hf_inv_freq.f32, 32 float32 dumped from the model's
// rotary) is generated out-of-band; the test skips cleanly if absent. This is the rope prerequisite
// for the in-engine MLA op; getting inv_freq wrong is a silent, fluent-garbage failure, so it is
// pinned to the real values here.
#include <cmath>
#include <cstdio>
#include <fstream>
#include <vector>

#include "engine/deepseek_rope.hpp"

int main() {
  // V2-Lite rope config.
  const int rope_dim = 64;
  const auto y = engine::deepseek_yarn_rope(rope_dim, /*base*/ 10000.0f, /*factor*/ 40.0f,
                                            /*beta_fast*/ 32.0f, /*beta_slow*/ 1.0f,
                                            /*orig_max_pos*/ 4096, /*mscale*/ 0.707f,
                                            /*mscale_all_dim*/ 0.707f);

  std::ifstream f("artifacts/hf_inv_freq.f32", std::ios::binary);
  if (!f) {
    std::printf("SKIP: artifacts/hf_inv_freq.f32 not found (dump it from the model first)\n");
    return 0;
  }
  std::vector<float> hf(rope_dim / 2);
  f.read(reinterpret_cast<char*>(hf.data()), hf.size() * sizeof(float));
  if (!f) {
    std::printf("FAIL: could not read %zu floats from hf_inv_freq.f32\n", hf.size());
    return 1;
  }

  float maxrel = 0.0f, denom = 1e-12f;
  for (size_t i = 0; i < hf.size(); ++i) {
    maxrel = std::max(maxrel, std::fabs(y.inv_freq[i] - hf[i]));
    denom = std::max(denom, std::fabs(hf[i]));
  }
  const float rel = maxrel / denom;
  const bool freq_ok = rel < 1e-5f;
  // attention_scaling for V2-Lite: mscale == mscale_all_dim -> ratio is exactly 1.0.
  const bool scale_ok = std::fabs(y.attention_scaling - 1.0f) < 1e-6f;
  const bool pass = freq_ok && scale_ok;
  std::printf("%s[DeepSeek YARN rope]: inv_freq vs HF max rel %.2e, attention_scaling %.4f\n",
              pass ? "PASS" : "FAIL", rel, y.attention_scaling);
  return pass ? 0 : 1;
}
