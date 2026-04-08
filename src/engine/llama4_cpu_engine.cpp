#include "engine/llama4_cpu_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace engine {
namespace {

constexpr float kPi = 3.14159265358979323846f;

static inline float bf16_to_f32(std::uint16_t h) {
  std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

static inline __m256 load_bf16_to_fp32(const std::uint16_t* ptr) {
  const __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(ptr));
  const __m256i e = _mm256_cvtepu16_epi32(h);
  const __m256i s = _mm256_slli_epi32(e, 16);
  return _mm256_castsi256_ps(s);
}

static inline float hsum256(__m256 v) {
  const __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 sum = _mm_add_ps(lo, hi);
  sum = _mm_hadd_ps(sum, sum);
  sum = _mm_hadd_ps(sum, sum);
  return _mm_cvtss_f32(sum);
}

void rmsnorm_optional_weight(const float* x, const std::uint16_t* w, float* out, int n) {
  float ss = 0.0f;
  for (int i = 0; i < n; ++i) {
    ss += x[i] * x[i];
  }
  const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n) + 1e-5f);
  if (w) {
    for (int i = 0; i < n; ++i) {
      out[i] = x[i] * inv * bf16_to_f32(w[i]);
    }
    return;
  }
  for (int i = 0; i < n; ++i) {
    out[i] = x[i] * inv;
  }
}

float silu(float x) {
  return x / (1.0f + std::exp(-x));
}

}  // namespace

void Llama4CpuEngine::gemv_bf16(const std::uint16_t* W, const float* x, float* y, int M, int N) {
  const int M4 = (M / 4) * 4;

#pragma omp parallel for schedule(static)
  for (int i = 0; i < M4; i += 4) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m256 acc2 = _mm256_setzero_ps();
    __m256 acc3 = _mm256_setzero_ps();

    const std::uint16_t* r0 = W + static_cast<std::size_t>(i) * static_cast<std::size_t>(N);
    const std::uint16_t* r1 = W + static_cast<std::size_t>(i + 1) * static_cast<std::size_t>(N);
    const std::uint16_t* r2 = W + static_cast<std::size_t>(i + 2) * static_cast<std::size_t>(N);
    const std::uint16_t* r3 = W + static_cast<std::size_t>(i + 3) * static_cast<std::size_t>(N);

    int j = 0;
    for (; j + 8 <= N; j += 8) {
      _mm_prefetch(reinterpret_cast<const char*>(r0 + j + 160), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(r1 + j + 160), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(r2 + j + 160), _MM_HINT_T0);
      _mm_prefetch(reinterpret_cast<const char*>(r3 + j + 160), _MM_HINT_T0);

      const __m256 xv = _mm256_loadu_ps(x + j);
      acc0 = _mm256_fmadd_ps(load_bf16_to_fp32(r0 + j), xv, acc0);
      acc1 = _mm256_fmadd_ps(load_bf16_to_fp32(r1 + j), xv, acc1);
      acc2 = _mm256_fmadd_ps(load_bf16_to_fp32(r2 + j), xv, acc2);
      acc3 = _mm256_fmadd_ps(load_bf16_to_fp32(r3 + j), xv, acc3);
    }

    float s0 = hsum256(acc0);
    float s1 = hsum256(acc1);
    float s2 = hsum256(acc2);
    float s3 = hsum256(acc3);
    for (; j < N; ++j) {
      s0 += bf16_to_f32(r0[j]) * x[j];
      s1 += bf16_to_f32(r1[j]) * x[j];
      s2 += bf16_to_f32(r2[j]) * x[j];
      s3 += bf16_to_f32(r3[j]) * x[j];
    }

    y[i] = s0;
    y[i + 1] = s1;
    y[i + 2] = s2;
    y[i + 3] = s3;
  }

#pragma omp parallel for schedule(static)
  for (int i = M4; i < M; ++i) {
    const std::uint16_t* row = W + static_cast<std::size_t>(i) * static_cast<std::size_t>(N);
    __m256 acc = _mm256_setzero_ps();
    int j = 0;
    for (; j + 8 <= N; j += 8) {
      acc = _mm256_fmadd_ps(load_bf16_to_fp32(row + j), _mm256_loadu_ps(x + j), acc);
    }
    float sum = hsum256(acc);
    for (; j < N; ++j) {
      sum += bf16_to_f32(row[j]) * x[j];
    }
    y[i] = sum;
  }
}

void Llama4CpuEngine::gemv_bf16_T(const std::uint16_t* W, const float* x, float* y, int in_dim, int out_dim) {
  std::fill(y, y + out_dim, 0.0f);
  constexpr int tile = 256;

#pragma omp parallel for schedule(dynamic, 1)
  for (int j0 = 0; j0 < out_dim; j0 += tile) {
    const int jend = std::min(j0 + tile, out_dim);
    const int jlen = jend - j0;
    for (int i = 0; i < in_dim; ++i) {
      const float xi = x[i];
      if (xi == 0.0f) {
        continue;
      }
      const __m256 xv = _mm256_set1_ps(xi);
      const std::uint16_t* row = W + static_cast<std::size_t>(i) * static_cast<std::size_t>(out_dim) +
                                 static_cast<std::size_t>(j0);
      float* yp = y + j0;
      int j = 0;
      for (; j + 8 <= jlen; j += 8) {
        const __m256 wv = load_bf16_to_fp32(row + j);
        __m256 yv = _mm256_loadu_ps(yp + j);
        yv = _mm256_fmadd_ps(xv, wv, yv);
        _mm256_storeu_ps(yp + j, yv);
      }
      for (; j < jlen; ++j) {
        yp[j] += xi * bf16_to_f32(row[j]);
      }
    }
  }
}

void Llama4CpuEngine::rmsnorm(const float* x, const std::uint16_t* w, float* out, int n) {
  rmsnorm_optional_weight(x, w, out, n);
}

void Llama4CpuEngine::rope(float* q, float* k, int pos) {
  const int half = head_dim_ / 2;

  for (int h = 0; h < n_heads_; ++h) {
    float* qh = q + static_cast<std::size_t>(h) * static_cast<std::size_t>(head_dim_);
    for (int d = 0; d < half; ++d) {
      const float q0 = qh[d];
      const float q1 = qh[d + half];
      const float c = rope_cos_[static_cast<std::size_t>(pos) * static_cast<std::size_t>(half) + d];
      const float s = rope_sin_[static_cast<std::size_t>(pos) * static_cast<std::size_t>(half) + d];
      qh[d] = q0 * c - q1 * s;
      qh[d + half] = q0 * s + q1 * c;
    }
  }

  for (int h = 0; h < n_kv_heads_; ++h) {
    float* kh = k + static_cast<std::size_t>(h) * static_cast<std::size_t>(head_dim_);
    for (int d = 0; d < half; ++d) {
      const float k0 = kh[d];
      const float k1 = kh[d + half];
      const float c = rope_cos_[static_cast<std::size_t>(pos) * static_cast<std::size_t>(half) + d];
      const float s = rope_sin_[static_cast<std::size_t>(pos) * static_cast<std::size_t>(half) + d];
      kh[d] = k0 * c - k1 * s;
      kh[d + half] = k0 * s + k1 * c;
    }
  }
}

void Llama4CpuEngine::attention(int pos, int layer) {
  const int kv_mul = n_heads_ / n_kv_heads_;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim_));
  float* k_layer = k_cache_.data() + static_cast<std::size_t>(layer) * static_cast<std::size_t>(max_ctx_) *
                                       static_cast<std::size_t>(kv_dim_);
  float* v_layer = v_cache_.data() + static_cast<std::size_t>(layer) * static_cast<std::size_t>(max_ctx_) *
                                       static_cast<std::size_t>(kv_dim_);

  std::memcpy(k_layer + static_cast<std::size_t>(pos) * static_cast<std::size_t>(kv_dim_), k_.data(),
              static_cast<std::size_t>(kv_dim_) * sizeof(float));
  std::memcpy(v_layer + static_cast<std::size_t>(pos) * static_cast<std::size_t>(kv_dim_), v_.data(),
              static_cast<std::size_t>(kv_dim_) * sizeof(float));

  const int seq_len = pos + 1;

#pragma omp parallel for schedule(static)
  for (int h = 0; h < n_heads_; ++h) {
    const int kv_head = h / kv_mul;
    const float* qh = q_.data() + static_cast<std::size_t>(h) * static_cast<std::size_t>(head_dim_);
    float* head_scores =
        scores_.data() + static_cast<std::size_t>(h) * static_cast<std::size_t>(max_ctx_);

    float max_score = -1.0e30f;
    for (int t = 0; t < seq_len; ++t) {
      const float* kt = k_layer + static_cast<std::size_t>(t) * static_cast<std::size_t>(kv_dim_) +
                        static_cast<std::size_t>(kv_head) * static_cast<std::size_t>(head_dim_);
      float dot = 0.0f;
      for (int d = 0; d < head_dim_; ++d) {
        dot += qh[d] * kt[d];
      }
      head_scores[t] = dot * scale;
      max_score = std::max(max_score, head_scores[t]);
    }

    float sum = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
      head_scores[t] = std::exp(head_scores[t] - max_score);
      sum += head_scores[t];
    }
    const float inv_sum = 1.0f / std::max(sum, 1e-20f);
    for (int t = 0; t < seq_len; ++t) {
      head_scores[t] *= inv_sum;
    }

    float* out_head = att_.data() + static_cast<std::size_t>(h) * static_cast<std::size_t>(head_dim_);
    std::fill(out_head, out_head + head_dim_, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
      const float* vt = v_layer + static_cast<std::size_t>(t) * static_cast<std::size_t>(kv_dim_) +
                        static_cast<std::size_t>(kv_head) * static_cast<std::size_t>(head_dim_);
      const float w = head_scores[t];
      for (int d = 0; d < head_dim_; ++d) {
        out_head[d] += w * vt[d];
      }
    }
  }
}

void Llama4CpuEngine::moe_ffn(int layer) {
  const LayerWeights& lw = layers_[static_cast<std::size_t>(layer)];
  const int half_int = inter_mlp_ / 2;

  gemv_bf16(lw.router, x_norm_.data(), router_logits_.data(), n_experts_, hidden_);

  int expert_idx = 0;
  for (int e = 1; e < n_experts_; ++e) {
    if (router_logits_[static_cast<std::size_t>(e)] > router_logits_[static_cast<std::size_t>(expert_idx)]) {
      expert_idx = e;
    }
  }

  const std::uint16_t* gate_up_ptr =
      lw.gate_up + static_cast<std::size_t>(expert_idx) * static_cast<std::size_t>(hidden_) *
                       static_cast<std::size_t>(inter_mlp_);
  gemv_bf16_T(gate_up_ptr, x_norm_.data(), gate_buf_.data(), hidden_, inter_mlp_);
  for (int i = 0; i < half_int; ++i) {
    down_buf_[static_cast<std::size_t>(i)] =
        silu(gate_buf_[static_cast<std::size_t>(i)]) * gate_buf_[static_cast<std::size_t>(i + half_int)];
  }

  const std::uint16_t* down_ptr =
      lw.down_exp + static_cast<std::size_t>(expert_idx) * static_cast<std::size_t>(half_int) *
                        static_cast<std::size_t>(hidden_);
  gemv_bf16_T(down_ptr, down_buf_.data(), shared_buf_.data(), half_int, hidden_);

  gemv_bf16(lw.sh_gate, x_norm_.data(), shared_gate_.data(), inter_shared_, hidden_);
  gemv_bf16(lw.sh_up, x_norm_.data(), shared_up_.data(), inter_shared_, hidden_);
  for (int i = 0; i < inter_shared_; ++i) {
    up_buf_[static_cast<std::size_t>(i)] =
        silu(shared_gate_[static_cast<std::size_t>(i)]) * shared_up_[static_cast<std::size_t>(i)];
  }

  float* shared_out = att_.data();
  gemv_bf16(lw.sh_down, up_buf_.data(), shared_out, hidden_, inter_shared_);
  for (int i = 0; i < hidden_; ++i) {
    x_[static_cast<std::size_t>(i)] += shared_buf_[static_cast<std::size_t>(i)] + shared_out[i];
  }
}

void Llama4CpuEngine::forward_token(int token, int pos) {
  const std::uint16_t* emb_row =
      tok_embeddings_ + static_cast<std::size_t>(token) * static_cast<std::size_t>(hidden_);
  for (int i = 0; i < hidden_; ++i) {
    x_[static_cast<std::size_t>(i)] = bf16_to_f32(emb_row[i]);
  }

  for (int l = 0; l < n_layers_; ++l) {
    const LayerWeights& lw = layers_[static_cast<std::size_t>(l)];
    rmsnorm(x_.data(), lw.norm_att, x_norm_.data(), hidden_);

    gemv_bf16(lw.wq, x_norm_.data(), q_.data(), n_heads_ * head_dim_, hidden_);
    gemv_bf16(lw.wk, x_norm_.data(), k_.data(), kv_dim_, hidden_);
    gemv_bf16(lw.wv, x_norm_.data(), v_.data(), kv_dim_, hidden_);

    if (use_qk_norm_) {
      for (int h = 0; h < n_heads_; ++h) {
        float* q_head = q_.data() + static_cast<std::size_t>(h) * static_cast<std::size_t>(head_dim_);
        rmsnorm_optional_weight(q_head, lw.q_norm, q_head, head_dim_);
      }
      for (int h = 0; h < n_kv_heads_; ++h) {
        float* k_head = k_.data() + static_cast<std::size_t>(h) * static_cast<std::size_t>(head_dim_);
        rmsnorm_optional_weight(k_head, lw.k_norm, k_head, head_dim_);
      }
    }

    rope(q_.data(), k_.data(), pos);
    attention(pos, l);

    gemv_bf16(lw.wo, att_.data(), x_norm_.data(), hidden_, hidden_);
    for (int i = 0; i < hidden_; ++i) {
      x_[static_cast<std::size_t>(i)] += x_norm_[static_cast<std::size_t>(i)];
    }

    rmsnorm(x_.data(), lw.norm_ffn, x_norm_.data(), hidden_);
    moe_ffn(l);
  }

  rmsnorm(x_.data(), norm_out_, x_norm_.data(), hidden_);
  gemv_bf16(lm_head_, x_norm_.data(), logits_.data(), vocab_size_, hidden_);
}

int Llama4CpuEngine::sample_token(float temperature, int top_k, const std::vector<int>& history, float rep_penalty) {
  if (rep_penalty != 1.0f) {
    for (int tok : history) {
      if (tok < 0 || tok >= vocab_size_) {
        continue;
      }
      float& logit = logits_[static_cast<std::size_t>(tok)];
      logit = (logit > 0.0f) ? (logit / rep_penalty) : (logit * rep_penalty);
    }
  }

  if (temperature <= 0.0f || top_k == 1) {
    return static_cast<int>(std::max_element(logits_.begin(), logits_.end()) - logits_.begin());
  }

  const int k = std::clamp(top_k, 1, vocab_size_);
  std::vector<int> ids(static_cast<std::size_t>(vocab_size_));
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                    [&](int left, int right) { return logits_[static_cast<std::size_t>(left)] >
                                                      logits_[static_cast<std::size_t>(right)]; });

  const float max_logit = logits_[static_cast<std::size_t>(ids[0])];
  std::vector<float> probs(static_cast<std::size_t>(k));
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    probs[static_cast<std::size_t>(i)] =
        std::exp((logits_[static_cast<std::size_t>(ids[static_cast<std::size_t>(i)])] - max_logit) / temperature);
    sum += probs[static_cast<std::size_t>(i)];
  }
  if (sum <= 0.0f) {
    return ids[0];
  }
  for (float& p : probs) {
    p /= sum;
  }

  const float top_p = std::clamp(options_.top_p, 0.0f, 1.0f);
  int limit = k;
  if (top_p > 0.0f && top_p < 1.0f) {
    float cumulative = 0.0f;
    limit = 0;
    for (; limit < k; ++limit) {
      cumulative += probs[static_cast<std::size_t>(limit)];
      if (cumulative >= top_p) {
        ++limit;
        break;
      }
    }
    limit = std::clamp(limit, 1, k);
    float renorm = 0.0f;
    for (int i = 0; i < limit; ++i) {
      renorm += probs[static_cast<std::size_t>(i)];
    }
    for (int i = 0; i < limit; ++i) {
      probs[static_cast<std::size_t>(i)] /= renorm;
    }
  }

  static thread_local std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float draw = dist(rng);
  float cumulative = 0.0f;
  for (int i = 0; i < limit; ++i) {
    cumulative += probs[static_cast<std::size_t>(i)];
    if (draw <= cumulative) {
      return ids[static_cast<std::size_t>(i)];
    }
  }
  return ids[static_cast<std::size_t>(limit - 1)];
}

void Llama4CpuEngine::initialize(const EngineOptions& options) {
  options_ = options;
  if (options.max_context > 0) {
    max_ctx_ = std::min(options.max_context, rope_orig_max_pos_ * static_cast<int>(rope_scale_));
  }
  kv_dim_ = n_kv_heads_ * head_dim_;

  weights_.open(options.model_path);
  auto load_tensor = [&](const std::string& name) -> const std::uint16_t* {
    return reinterpret_cast<const std::uint16_t*>(weights_.tensor_ptr(name));
  };

  tok_embeddings_ = load_tensor("language_model.model.embed_tokens.weight");
  norm_out_ = load_tensor("language_model.model.norm.weight");
  lm_head_ = load_tensor("language_model.lm_head.weight");

  layers_.resize(static_cast<std::size_t>(n_layers_));
  for (int l = 0; l < n_layers_; ++l) {
    const std::string prefix = "language_model.model.layers." + std::to_string(l);
    LayerWeights& lw = layers_[static_cast<std::size_t>(l)];
    lw.norm_att = load_tensor(prefix + ".input_layernorm.weight");
    lw.norm_ffn = load_tensor(prefix + ".post_attention_layernorm.weight");
    lw.wq = load_tensor(prefix + ".self_attn.q_proj.weight");
    lw.wk = load_tensor(prefix + ".self_attn.k_proj.weight");
    lw.wv = load_tensor(prefix + ".self_attn.v_proj.weight");
    lw.wo = load_tensor(prefix + ".self_attn.o_proj.weight");
    lw.router = load_tensor(prefix + ".feed_forward.router.weight");
    lw.gate_up = load_tensor(prefix + ".feed_forward.experts.gate_up_proj");
    lw.down_exp = load_tensor(prefix + ".feed_forward.experts.down_proj");
    lw.sh_gate = load_tensor(prefix + ".feed_forward.shared_expert.gate_proj.weight");
    lw.sh_up = load_tensor(prefix + ".feed_forward.shared_expert.up_proj.weight");
    lw.sh_down = load_tensor(prefix + ".feed_forward.shared_expert.down_proj.weight");

    const std::string q_norm_name = prefix + ".self_attn.q_norm.weight";
    const std::string k_norm_name = prefix + ".self_attn.k_norm.weight";
    lw.q_norm = weights_.has_tensor(q_norm_name) ? load_tensor(q_norm_name) : nullptr;
    lw.k_norm = weights_.has_tensor(k_norm_name) ? load_tensor(k_norm_name) : nullptr;
  }

  const int half = head_dim_ / 2;
  rope_cos_.resize(static_cast<std::size_t>(max_ctx_) * static_cast<std::size_t>(half));
  rope_sin_.resize(static_cast<std::size_t>(max_ctx_) * static_cast<std::size_t>(half));
  const float low_wavelen = static_cast<float>(rope_orig_max_pos_) / rope_low_freq_factor_;
  const float high_wavelen = static_cast<float>(rope_orig_max_pos_) / rope_high_freq_factor_;

  for (int d = 0; d < half; ++d) {
    float theta_d =
        std::pow(rope_theta_, -2.0f * static_cast<float>(d) / static_cast<float>(head_dim_));
    const float wavelen = 2.0f * kPi / theta_d;
    if (wavelen > low_wavelen) {
      theta_d /= rope_scale_;
    } else if (wavelen >= high_wavelen) {
      const float smooth =
          (static_cast<float>(rope_orig_max_pos_) / wavelen - rope_low_freq_factor_) /
          (rope_high_freq_factor_ - rope_low_freq_factor_);
      const float scaled_theta = theta_d / rope_scale_;
      theta_d = (1.0f - smooth) * scaled_theta + smooth * theta_d;
    }

    for (int p = 0; p < max_ctx_; ++p) {
      const float angle = static_cast<float>(p) * theta_d;
      rope_cos_[static_cast<std::size_t>(p) * static_cast<std::size_t>(half) + d] = std::cos(angle);
      rope_sin_[static_cast<std::size_t>(p) * static_cast<std::size_t>(half) + d] = std::sin(angle);
    }
  }

  k_cache_.assign(static_cast<std::size_t>(n_layers_) * static_cast<std::size_t>(max_ctx_) *
                      static_cast<std::size_t>(kv_dim_),
                  0.0f);
  v_cache_.assign(static_cast<std::size_t>(n_layers_) * static_cast<std::size_t>(max_ctx_) *
                      static_cast<std::size_t>(kv_dim_),
                  0.0f);

  x_.resize(static_cast<std::size_t>(hidden_));
  x_norm_.resize(static_cast<std::size_t>(hidden_));
  q_.resize(static_cast<std::size_t>(n_heads_ * head_dim_));
  k_.resize(static_cast<std::size_t>(kv_dim_));
  v_.resize(static_cast<std::size_t>(kv_dim_));
  att_.resize(static_cast<std::size_t>(hidden_));
  router_logits_.resize(static_cast<std::size_t>(n_experts_));
  gate_buf_.resize(static_cast<std::size_t>(inter_mlp_));
  up_buf_.resize(static_cast<std::size_t>(inter_shared_));
  down_buf_.resize(static_cast<std::size_t>(inter_mlp_ / 2));
  shared_buf_.resize(static_cast<std::size_t>(hidden_));
  shared_gate_.resize(static_cast<std::size_t>(inter_shared_));
  shared_up_.resize(static_cast<std::size_t>(inter_shared_));
  logits_.resize(static_cast<std::size_t>(vocab_size_));
  scores_.resize(static_cast<std::size_t>(n_heads_) * static_cast<std::size_t>(max_ctx_));

  if (options_.verbose) {
    std::fprintf(stderr,
                 "[llama4_cpu] layers=%d hidden=%d heads=%d/%d experts=%d vocab=%d max_ctx=%d kv_dim=%d\n",
                 n_layers_,
                 hidden_,
                 n_heads_,
                 n_kv_heads_,
                 n_experts_,
                 vocab_size_,
                 max_ctx_,
                 kv_dim_);
  }
}

std::vector<int> Llama4CpuEngine::generate(const std::vector<int>& prompt_tokens,
                                           int max_new_tokens,
                                           float temperature) {
  return generate_stream(prompt_tokens, max_new_tokens, temperature, [](int) { return true; });
}

std::vector<int> Llama4CpuEngine::generate_stream(const std::vector<int>& prompt_tokens,
                                                  int max_new_tokens,
                                                  float temperature,
                                                  const std::function<bool(int)>& on_token) {
  stats_ = BenchmarkStats{};
  stats_.prompt_tokens = static_cast<int>(prompt_tokens.size());
  std::fill(k_cache_.begin(), k_cache_.end(), 0.0f);
  std::fill(v_cache_.begin(), v_cache_.end(), 0.0f);

  std::vector<int> output = prompt_tokens;
  std::vector<int> history;

  const auto prefill_start = std::chrono::steady_clock::now();
  int pos = 0;
  if (prompt_tokens.empty()) {
    forward_token(bos_id_, 0);
    pos = 1;
  } else {
    for (int i = 0; i < static_cast<int>(prompt_tokens.size()) && i < max_ctx_; ++i) {
      forward_token(prompt_tokens[static_cast<std::size_t>(i)], i);
      pos = i + 1;
    }
  }
  const auto prefill_end = std::chrono::steady_clock::now();
  stats_.prefill_ms =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

  const auto decode_start = std::chrono::steady_clock::now();
  for (int step = 0; step < max_new_tokens; ++step) {
    const int next = sample_token(temperature, options_.top_k, history, options_.repetition_penalty);
    history.push_back(next);
    output.push_back(next);
    ++stats_.generated_tokens;

    if (next == eos_id_) {
      break;
    }
    if (!on_token(next)) {
      break;
    }
    if (pos >= max_ctx_) {
      break;
    }

    forward_token(next, pos);
    ++pos;
  }
  const auto decode_end = std::chrono::steady_clock::now();
  stats_.decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
  return output;
}

std::vector<std::pair<int, float>> Llama4CpuEngine::inspect_next_logits(const std::vector<int>& prompt_tokens,
                                                                        int top_k) {
  std::fill(k_cache_.begin(), k_cache_.end(), 0.0f);
  std::fill(v_cache_.begin(), v_cache_.end(), 0.0f);

  if (prompt_tokens.empty()) {
    forward_token(bos_id_, 0);
  } else {
    for (int i = 0; i < static_cast<int>(prompt_tokens.size()) && i < max_ctx_; ++i) {
      forward_token(prompt_tokens[static_cast<std::size_t>(i)], i);
    }
  }

  const int k = std::clamp(top_k, 0, vocab_size_);
  std::vector<std::pair<int, float>> pairs;
  pairs.reserve(static_cast<std::size_t>(vocab_size_));
  for (int i = 0; i < vocab_size_; ++i) {
    pairs.emplace_back(i, logits_[static_cast<std::size_t>(i)]);
  }
  std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                    [](const auto& left, const auto& right) { return left.second > right.second; });
  pairs.resize(static_cast<std::size_t>(k));
  return pairs;
}

}  // namespace engine
