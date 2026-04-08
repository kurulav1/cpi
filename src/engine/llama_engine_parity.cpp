#include "engine/llama_engine.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "common.hpp"

namespace engine {
namespace {

const __half* tensor_half(const model::WeightLoader& weights,
                          const std::string& name) {
  return reinterpret_cast<const __half*>(weights.tensor_data(name));
}

void matvec_rowmajor(const __half* w,
                     const std::vector<float>& x,
                     int out_features,
                     int in_features,
                     std::vector<float>* y) {
  y->assign(static_cast<std::size_t>(out_features), 0.0f);
  for (int o = 0; o < out_features; ++o) {
    float acc = 0.0f;
    for (int i = 0; i < in_features; ++i) {
      acc += __half2float(
                 w[static_cast<std::size_t>(o) *
                       static_cast<std::size_t>(in_features) +
                   i]) *
             x[i];
    }
    (*y)[o] = acc;
  }
}

void normalize_cpu(const std::vector<float>& x,
                   const __half* w,
                   const __half* b,
                   bool use_layernorm,
                   float eps,
                   std::vector<float>* y) {
  y->resize(x.size());
  if (use_layernorm) {
    float sum = 0.0f;
    float sq = 0.0f;
    for (float v : x) {
      sum += v;
      sq += v * v;
    }
    const float mean = sum / static_cast<float>(x.size());
    const float var = std::max(0.0f, sq / static_cast<float>(x.size()) - mean * mean);
    const float inv = 1.0f / std::sqrt(var + eps);
    for (std::size_t i = 0; i < x.size(); ++i) {
      const float bb = b ? __half2float(b[i]) : 0.0f;
      (*y)[i] = (x[i] - mean) * inv * __half2float(w[i]) + bb;
    }
    return;
  }

  float sq = 0.0f;
  for (float v : x) {
    sq += v * v;
  }
  const float inv = 1.0f / std::sqrt(sq / static_cast<float>(x.size()) + eps);
  for (std::size_t i = 0; i < x.size(); ++i) {
    const float bb = b ? __half2float(b[i]) : 0.0f;
    (*y)[i] = x[i] * inv * __half2float(w[i]) + bb;
  }
}

void rope_cpu(std::vector<float>* q,
              std::vector<float>* k,
              int num_heads_q,
              int num_heads_k,
              int head_dim,
              int position,
              float rope_theta = 10000.0f) {
  const int half_dim = head_dim / 2;
  for (int h = 0; h < num_heads_q; ++h) {
    for (int i = 0; i < half_dim; ++i) {
      const float theta = std::pow(
          rope_theta, -2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
      const float angle = static_cast<float>(position) * theta;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      const int i0 = h * head_dim + i;
      const int i1 = h * head_dim + i + half_dim;
      const float q0 = (*q)[i0];
      const float q1 = (*q)[i1];
      (*q)[i0] = q0 * c - q1 * s;
      (*q)[i1] = q1 * c + q0 * s;
    }
  }

  for (int h = 0; h < num_heads_k; ++h) {
    for (int i = 0; i < half_dim; ++i) {
      const float theta = std::pow(
          rope_theta, -2.0f * static_cast<float>(i) / static_cast<float>(head_dim));
      const float angle = static_cast<float>(position) * theta;
      const float c = std::cos(angle);
      const float s = std::sin(angle);
      const int i0 = h * head_dim + i;
      const int i1 = h * head_dim + i + half_dim;
      const float k0 = (*k)[i0];
      const float k1 = (*k)[i1];
      (*k)[i0] = k0 * c - k1 * s;
      (*k)[i1] = k1 * c + k0 * s;
    }
  }
}

}  // namespace

void LlamaEngine::run_parity_check(const std::vector<int>& prompt_tokens) {
  if (prompt_tokens.empty()) {
    LLAMA_ENGINE_THROW("parity check requires non-empty prompt");
  }

  const auto& cfg = weights_.config();
  const int hidden = cfg.hidden_size;
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : hidden;
  const int inter = cfg.intermediate_size;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (hidden / cfg.num_heads);
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const int seq_len = static_cast<int>(prompt_tokens.size());
  if (seq_len > options_.max_context) {
    LLAMA_ENGINE_THROW("parity prompt exceeds max_context");
  }

  reset_kv_cache();
  enforce_host_resource_limits("parity.prefill");

  std::vector<float> gpu_logits;
  prefill_prompt(prompt_tokens);
  forward_token_logits(prompt_tokens.back(), seq_len - 1, &gpu_logits, nullptr);

  std::vector<float> x(static_cast<std::size_t>(hidden), 0.0f);
  std::vector<float> x_norm;
  std::vector<float> q;
  std::vector<float> k;
  std::vector<float> v;
  std::vector<float> att(static_cast<std::size_t>(q_hidden), 0.0f);
  std::vector<float> tmp;
  std::vector<float> ff1;
  std::vector<float> ff2;
  std::vector<float> ff3;

  const std::size_t kv_elems = static_cast<std::size_t>(cfg.num_layers) *
                               static_cast<std::size_t>(seq_len) *
                               static_cast<std::size_t>(kv_hidden);
  std::vector<float> k_cache(kv_elems, 0.0f);
  std::vector<float> v_cache(kv_elems, 0.0f);

  const auto* emb = tensor_half(weights_, "tok_embeddings.weight");
  const auto* norm_out = tensor_half(weights_, "norm.weight");
  const auto* norm_out_bias = weights_.has_tensor("norm.bias")
                                  ? tensor_half(weights_, "norm.bias")
                                  : nullptr;
  const auto* lm_head = tensor_half(
      weights_, weights_.has_tensor("output.weight") ? "output.weight"
                                                      : "tok_embeddings.weight");
  const auto* lm_head_bias = weights_.has_tensor("output.bias")
                                 ? tensor_half(weights_, "output.bias")
                                 : nullptr;

  std::vector<float> cpu_logits(static_cast<std::size_t>(cfg.vocab_size), 0.0f);

  for (int pos = 0; pos < seq_len; ++pos) {
    const int tok = prompt_tokens[pos];
    for (int i = 0; i < hidden; ++i) {
      x[i] = __half2float(
          emb[static_cast<std::size_t>(tok) * static_cast<std::size_t>(hidden) +
              i]);
    }

    for (int layer = 0; layer < cfg.num_layers; ++layer) {
      const std::string p = "layers." + std::to_string(layer);
      const auto* norm_att = tensor_half(weights_, p + ".attention_norm.weight");
      const auto* norm_att_bias = weights_.has_tensor(p + ".attention_norm.bias")
          ? tensor_half(weights_, p + ".attention_norm.bias")
          : nullptr;
      const auto* wq = tensor_half(weights_, p + ".attention.wq");
      const auto* wk = tensor_half(weights_, p + ".attention.wk");
      const auto* wv = tensor_half(weights_, p + ".attention.wv");
      const auto* wo = tensor_half(weights_, p + ".attention.wo");
      const auto* bo = weights_.has_tensor(p + ".attention.bo")
          ? tensor_half(weights_, p + ".attention.bo")
          : nullptr;
      const auto* norm_ffn = tensor_half(weights_, p + ".ffn_norm.weight");
      const auto* norm_ffn_bias = weights_.has_tensor(p + ".ffn_norm.bias")
          ? tensor_half(weights_, p + ".ffn_norm.bias")
          : nullptr;
      const auto* w1 = tensor_half(weights_, p + ".feed_forward.w1");
      const auto* w2 = tensor_half(weights_, p + ".feed_forward.w2");
      const auto* w3 = tensor_half(weights_, p + ".feed_forward.w3");

      normalize_cpu(x, norm_att, norm_att_bias, cfg.use_layernorm, cfg.norm_eps, &x_norm);
      matvec_rowmajor(wq, x_norm, q_hidden, hidden, &q);
      matvec_rowmajor(wk, x_norm, kv_hidden, hidden, &k);
      matvec_rowmajor(wv, x_norm, kv_hidden, hidden, &v);
      const float parity_rope_theta =
          (options_.rope_theta > 0.0f) ? options_.rope_theta
                                       : cfg.effective_rope_theta();
      rope_cpu(&q, &k, cfg.num_heads, cfg.num_kv_heads, head_dim, pos,
               parity_rope_theta);

      const std::size_t layer_off =
          static_cast<std::size_t>(layer) * static_cast<std::size_t>(seq_len) *
          static_cast<std::size_t>(kv_hidden);
      for (int i = 0; i < kv_hidden; ++i) {
        k_cache[layer_off +
                static_cast<std::size_t>(pos) *
                    static_cast<std::size_t>(kv_hidden) +
                i] = k[i];
        v_cache[layer_off +
                static_cast<std::size_t>(pos) *
                    static_cast<std::size_t>(kv_hidden) +
                i] = v[i];
      }

      std::fill(att.begin(), att.end(), 0.0f);
      const int group_size = cfg.num_heads / cfg.num_kv_heads;
      for (int h = 0; h < cfg.num_heads; ++h) {
        const int kv_head = h / group_size;
        std::vector<float> scores(static_cast<std::size_t>(pos + 1), 0.0f);
        float max_s = -1e30f;
        for (int t = 0; t <= pos; ++t) {
          float dot = 0.0f;
          for (int d = 0; d < head_dim; ++d) {
            const int q_idx = h * head_dim + d;
            const int kv_idx = kv_head * head_dim + d;
            const float kv =
                k_cache[layer_off +
                        static_cast<std::size_t>(t) *
                            static_cast<std::size_t>(kv_hidden) +
                        kv_idx];
            dot += q[q_idx] * kv;
          }
          dot /= std::sqrt(static_cast<float>(head_dim));
          scores[static_cast<std::size_t>(t)] = dot;
          max_s = std::max(max_s, dot);
        }
        float denom = 0.0f;
        for (float& s : scores) {
          s = std::exp(s - max_s);
          denom += s;
        }
        denom = std::max(denom, 1e-8f);
        for (int d = 0; d < head_dim; ++d) {
          const int q_idx = h * head_dim + d;
          const int kv_idx = kv_head * head_dim + d;
          float acc = 0.0f;
          for (int t = 0; t <= pos; ++t) {
            const float p_att = scores[static_cast<std::size_t>(t)] / denom;
            const float vv =
                v_cache[layer_off +
                        static_cast<std::size_t>(t) *
                            static_cast<std::size_t>(kv_hidden) +
                        kv_idx];
            acc += p_att * vv;
          }
          att[static_cast<std::size_t>(q_idx)] = acc;
        }
      }

      matvec_rowmajor(wo, att, hidden, q_hidden, &tmp);
      for (int i = 0; i < hidden; ++i) {
        x[i] += tmp[i] + (bo ? __half2float(bo[i]) : 0.0f);
      }

      normalize_cpu(x, norm_ffn, norm_ffn_bias, cfg.use_layernorm, cfg.norm_eps, &x_norm);
      matvec_rowmajor(w1, x_norm, inter, hidden, &ff1);
      matvec_rowmajor(w3, x_norm, inter, hidden, &ff2);
      for (int i = 0; i < inter; ++i) {
        const float g = ff1[static_cast<std::size_t>(i)];
        ff2[static_cast<std::size_t>(i)] *= g / (1.0f + std::exp(-g));
      }
      matvec_rowmajor(w2, ff2, hidden, inter, &ff3);
      for (int i = 0; i < hidden; ++i) {
        x[i] += ff3[static_cast<std::size_t>(i)];
      }
    }

    normalize_cpu(x, norm_out, norm_out_bias, cfg.use_layernorm, cfg.norm_eps, &x_norm);
    matvec_rowmajor(lm_head, x_norm, cfg.vocab_size, hidden, &cpu_logits);
    if (lm_head_bias) {
      for (int i = 0; i < cfg.vocab_size; ++i) {
        cpu_logits[static_cast<std::size_t>(i)] += __half2float(lm_head_bias[static_cast<std::size_t>(i)]);
      }
    }
  }

  double max_abs = 0.0;
  double mean_abs = 0.0;
  for (std::size_t i = 0; i < cpu_logits.size(); ++i) {
    const double d = std::abs(static_cast<double>(cpu_logits[i]) -
                              static_cast<double>(gpu_logits[i]));
    max_abs = std::max(max_abs, d);
    mean_abs += d;
  }
  mean_abs /= static_cast<double>(cpu_logits.size());

  const int cpu_top = static_cast<int>(
      std::max_element(cpu_logits.begin(), cpu_logits.end()) - cpu_logits.begin());
  const int gpu_top = static_cast<int>(
      std::max_element(gpu_logits.begin(), gpu_logits.end()) - gpu_logits.begin());
  std::cout << "[parity] top_token_cpu=" << cpu_top
            << " top_token_gpu=" << gpu_top << "\n";
  std::cout << "[parity] max_abs_diff=" << max_abs
            << " mean_abs_diff=" << mean_abs << "\n";
}

}  // namespace engine
