#pragma once

// DeepSeek-V2 YARN RoPE frequencies, matching transformers' yarn rope init exactly (verified in
// deepseek_rope_test against the model's dumped inv_freq to ~1e-8). MLA applies INTERLEAVED-complex
// rope on the qk_rope_head_dim slice using these inv_freq, scaling the rotated vectors by
// attention_scaling (the yarn mscale). For V2-Lite attention_scaling == 1.0 (the mscale ratio
// cancels because mscale == mscale_all_dim); this helper returns it generally.
//
// Header-only host math so the loader (and a standalone test) can build the rope table without
// CUDA.

#include <cmath>
#include <vector>

namespace engine {

struct YarnRope {
  std::vector<float> inv_freq;     // [qk_rope_head_dim/2]
  float attention_scaling = 1.0f;  // multiplies the rotated q_pe/k_pe (yarn mscale)
};

// Compute the yarn-interpolated inverse frequencies for `rope_dim` (= qk_rope_head_dim) under a
// scale `factor`, ramping between beta_fast/beta_slow rotations against `orig_max_pos`. Mirrors
// transformers _compute_yarn_parameters + _yarn_get_mscale.
inline YarnRope deepseek_yarn_rope(int rope_dim, float base, float factor, float beta_fast,
                                   float beta_slow, int orig_max_pos, float mscale,
                                   float mscale_all_dim) {
  const int half = rope_dim / 2;
  YarnRope out;
  out.inv_freq.resize(half);

  constexpr float kPi = 3.14159265358979323846f;
  auto correction_dim = [&](float num_rot) {
    return rope_dim * std::log(orig_max_pos / (num_rot * 2.0f * kPi)) / (2.0f * std::log(base));
  };
  float low = std::floor(correction_dim(beta_fast));
  float high = std::ceil(correction_dim(beta_slow));
  low = std::max(low, 0.0f);
  high = std::min(high, static_cast<float>(rope_dim - 1));
  if (low == high) high += 0.001f;

  for (int i = 0; i < half; ++i) {
    const float freq = std::pow(base, static_cast<float>(2 * i) / rope_dim);  // plain pos_freq
    const float inv_extrap = 1.0f / freq;
    const float inv_interp = 1.0f / (factor * freq);
    float ramp = (static_cast<float>(i) - low) / (high - low);  // linear ramp
    ramp = std::min(std::max(ramp, 0.0f), 1.0f);
    const float extrap_factor = 1.0f - ramp;  // 1 near low-freq dims, 0 near high-freq
    out.inv_freq[i] = inv_interp * (1.0f - extrap_factor) + inv_extrap * extrap_factor;
  }

  // yarn mscale: DeepSeek uses the RATIO
  // get_mscale(factor,mscale)/get_mscale(factor,mscale_all_dim).
  auto get_mscale = [&](float m) {
    return factor <= 1.0f ? 1.0f : (0.1f * m * std::log(factor) + 1.0f);
  };
  out.attention_scaling =
      mscale_all_dim != 0.0f ? get_mscale(mscale) / get_mscale(mscale_all_dim) : get_mscale(mscale);
  return out;
}

}  // namespace engine
