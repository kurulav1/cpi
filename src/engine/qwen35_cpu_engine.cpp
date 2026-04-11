#include "engine/qwen35_cpu_engine.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iterator>
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

float silu(float x) {
  return x / (1.0f + std::exp(-x));
}

float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

float softplus(float x) {
  if (x > 20.0f) {
    return x;
  }
  if (x < -20.0f) {
    return std::exp(x);
  }
  return std::log1p(std::exp(x));
}

std::string read_text_file(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) {
    throw std::runtime_error("failed to open config file: " + path.string());
  }
  return std::string((std::istreambuf_iterator<char>(in)),
                     std::istreambuf_iterator<char>());
}

void skip_ws(const std::string& json, std::size_t& pos) {
  while (pos < json.size() &&
         std::isspace(static_cast<unsigned char>(json[pos]))) {
    ++pos;
  }
}

std::size_t json_find_key(const std::string& json, const std::string& key) {
  const std::string needle = "\"" + key + "\"";
  std::size_t pos = 0;
  while ((pos = json.find(needle, pos)) != std::string::npos) {
    std::size_t value_pos = pos + needle.size();
    skip_ws(json, value_pos);
    if (value_pos < json.size() && json[value_pos] == ':') {
      return value_pos + 1;
    }
    pos += needle.size();
  }
  return std::string::npos;
}

std::string json_read_string(const std::string& json, std::size_t& pos) {
  if (pos >= json.size() || json[pos] != '"') {
    return "";
  }
  ++pos;
  std::string result;
  while (pos < json.size()) {
    const char c = json[pos++];
    if (c == '"') {
      break;
    }
    if (c == '\\' && pos < json.size()) {
      const char escaped = json[pos++];
      switch (escaped) {
        case '"': result.push_back('"'); break;
        case '\\': result.push_back('\\'); break;
        case '/': result.push_back('/'); break;
        case 'b': result.push_back('\b'); break;
        case 'f': result.push_back('\f'); break;
        case 'n': result.push_back('\n'); break;
        case 'r': result.push_back('\r'); break;
        case 't': result.push_back('\t'); break;
        default: result.push_back(escaped); break;
      }
      continue;
    }
    result.push_back(c);
  }
  return result;
}

std::string json_extract_object(const std::string& json, const std::string& key) {
  std::size_t pos = json_find_key(json, key);
  if (pos == std::string::npos) {
    return "";
  }
  skip_ws(json, pos);
  if (pos >= json.size() || json[pos] != '{') {
    return "";
  }
  const std::size_t start = pos;
  int depth = 0;
  bool in_string = false;
  for (; pos < json.size(); ++pos) {
    const char c = json[pos];
    if (in_string) {
      if (c == '\\') {
        ++pos;
      } else if (c == '"') {
        in_string = false;
      }
      continue;
    }
    if (c == '"') {
      in_string = true;
      continue;
    }
    if (c == '{') {
      ++depth;
      continue;
    }
    if (c == '}') {
      --depth;
      if (depth == 0) {
        return json.substr(start, pos - start + 1);
      }
    }
  }
  return "";
}

std::string json_get_string(const std::string& json,
                            const std::string& key,
                            const std::string& def = "") {
  std::size_t pos = json_find_key(json, key);
  if (pos == std::string::npos) {
    return def;
  }
  skip_ws(json, pos);
  if (pos >= json.size() || json[pos] != '"') {
    return def;
  }
  return json_read_string(json, pos);
}

int json_get_int(const std::string& json, const std::string& key, int def = 0) {
  std::size_t pos = json_find_key(json, key);
  if (pos == std::string::npos) {
    return def;
  }
  skip_ws(json, pos);
  std::size_t end = pos;
  while (end < json.size() &&
         (std::isdigit(static_cast<unsigned char>(json[end])) ||
          json[end] == '-' || json[end] == '+')) {
    ++end;
  }
  if (end == pos) {
    return def;
  }
  try {
    return std::stoi(json.substr(pos, end - pos));
  } catch (...) {
    return def;
  }
}

float json_get_float(const std::string& json,
                     const std::string& key,
                     float def = 0.0f) {
  std::size_t pos = json_find_key(json, key);
  if (pos == std::string::npos) {
    return def;
  }
  skip_ws(json, pos);
  std::size_t end = pos;
  while (end < json.size() &&
         (std::isdigit(static_cast<unsigned char>(json[end])) ||
          json[end] == '-' || json[end] == '+' || json[end] == '.' ||
          json[end] == 'e' || json[end] == 'E')) {
    ++end;
  }
  if (end == pos) {
    return def;
  }
  try {
    return std::stof(json.substr(pos, end - pos));
  } catch (...) {
    return def;
  }
}

std::vector<std::string> json_get_string_array(const std::string& json,
                                               const std::string& key) {
  std::size_t pos = json_find_key(json, key);
  if (pos == std::string::npos) {
    return {};
  }
  skip_ws(json, pos);
  if (pos >= json.size() || json[pos] != '[') {
    return {};
  }
  ++pos;
  std::vector<std::string> out;
  while (pos < json.size()) {
    skip_ws(json, pos);
    if (pos >= json.size() || json[pos] == ']') {
      break;
    }
    if (json[pos] == '"') {
      out.push_back(json_read_string(json, pos));
    } else {
      ++pos;
    }
    while (pos < json.size() && json[pos] != '"' && json[pos] != ']') {
      ++pos;
    }
  }
  return out;
}

const std::uint16_t* require_bf16_tensor(const model::SafetensorsLoader& loader,
                                         const std::string& name) {
  if (!loader.has_tensor(name)) {
    throw std::runtime_error("missing tensor: " + name);
  }
  return reinterpret_cast<const std::uint16_t*>(loader.tensor_ptr(name));
}

const float* require_f32_tensor(const model::SafetensorsLoader& loader,
                                const std::string& name) {
  if (!loader.has_tensor(name)) {
    throw std::runtime_error("missing tensor: " + name);
  }
  return reinterpret_cast<const float*>(loader.tensor_ptr(name));
}

void l2norm_inplace(float* x, int n, float eps = 1e-6f) {
  float ss = 0.0f;
  for (int i = 0; i < n; ++i) {
    ss += x[i] * x[i];
  }
  const float inv = 1.0f / std::sqrt(ss + eps);
  for (int i = 0; i < n; ++i) {
    x[i] *= inv;
  }
}

}  // namespace

void Qwen35CpuEngine::gemv_bf16(const std::uint16_t* W,
                                const float* x,
                                float* y,
                                int M,
                                int N) {
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

void Qwen35CpuEngine::rmsnorm_offset(const float* x,
                                     const std::uint16_t* weight,
                                     float* out,
                                     int n,
                                     float eps) {
  float ss = 0.0f;
  for (int i = 0; i < n; ++i) {
    ss += x[i] * x[i];
  }
  const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
  for (int i = 0; i < n; ++i) {
    out[i] = x[i] * inv * (1.0f + bf16_to_f32(weight[i]));
  }
}

void Qwen35CpuEngine::rmsnorm_offset_inplace(float* x,
                                             const std::uint16_t* weight,
                                             int n,
                                             float eps) {
  float ss = 0.0f;
  for (int i = 0; i < n; ++i) {
    ss += x[i] * x[i];
  }
  const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
  for (int i = 0; i < n; ++i) {
    x[i] = x[i] * inv * (1.0f + bf16_to_f32(weight[i]));
  }
}

void Qwen35CpuEngine::rmsnorm_direct_gated_inplace(float* x,
                                                   const float* weight,
                                                   const float* gate,
                                                   int n,
                                                   float eps) {
  float ss = 0.0f;
  for (int i = 0; i < n; ++i) {
    ss += x[i] * x[i];
  }
  const float inv = 1.0f / std::sqrt(ss / static_cast<float>(n) + eps);
  for (int i = 0; i < n; ++i) {
    x[i] = x[i] * inv * weight[i] * silu(gate[i]);
  }
}

void Qwen35CpuEngine::load_config(const std::string& model_dir) {
  const auto config_path = std::filesystem::path(model_dir) / "config.json";
  const std::string raw = read_text_file(config_path);
  const std::string text_config = json_extract_object(raw, "text_config");
  if (text_config.empty()) {
    throw std::runtime_error("Qwen3.5 config is missing text_config");
  }

  const std::string root_model_type = json_get_string(raw, "model_type", "");
  const std::string model_type = json_get_string(text_config, "model_type", root_model_type);
  if (root_model_type.find("qwen3_5") == std::string::npos &&
      model_type.find("qwen3_5") == std::string::npos) {
    throw std::runtime_error("config.json does not describe a Qwen3.5 text model");
  }

  cfg_.vocab_size = json_get_int(text_config, "vocab_size", 0);
  cfg_.hidden_size = json_get_int(text_config, "hidden_size", 0);
  cfg_.intermediate_size = json_get_int(text_config, "intermediate_size", 0);
  cfg_.num_layers = json_get_int(text_config, "num_hidden_layers", 0);
  cfg_.num_attention_heads = json_get_int(text_config, "num_attention_heads", 0);
  cfg_.num_key_value_heads = json_get_int(text_config, "num_key_value_heads", 0);
  cfg_.head_dim = json_get_int(text_config, "head_dim", 0);
  cfg_.linear_num_key_heads = json_get_int(text_config, "linear_num_key_heads", 0);
  cfg_.linear_num_value_heads = json_get_int(text_config, "linear_num_value_heads", 0);
  cfg_.linear_key_head_dim = json_get_int(text_config, "linear_key_head_dim", 0);
  cfg_.linear_value_head_dim = json_get_int(text_config, "linear_value_head_dim", 0);
  cfg_.linear_conv_kernel_dim = json_get_int(text_config, "linear_conv_kernel_dim", 0);
  cfg_.max_position_embeddings = json_get_int(text_config, "max_position_embeddings", 0);
  cfg_.eos_token_id = json_get_int(text_config, "eos_token_id",
                                   json_get_int(raw, "eos_token_id", -1));
  cfg_.rms_norm_eps = json_get_float(text_config, "rms_norm_eps", 1e-6f);
  const std::string rope_parameters = json_extract_object(text_config, "rope_parameters");
  cfg_.rope_theta = json_get_float(rope_parameters, "rope_theta", 10000000.0f);
  cfg_.partial_rotary_factor = json_get_float(rope_parameters, "partial_rotary_factor", 1.0f);

  const auto layer_types = json_get_string_array(text_config, "layer_types");
  cfg_.layer_kinds.reserve(layer_types.size());
  for (const std::string& value : layer_types) {
    cfg_.layer_kinds.push_back(
        value == "full_attention" ? LayerKind::FullAttention
                                   : LayerKind::LinearAttention);
  }

  if (cfg_.vocab_size <= 0 || cfg_.hidden_size <= 0 || cfg_.intermediate_size <= 0 ||
      cfg_.num_layers <= 0 || cfg_.num_attention_heads <= 0 || cfg_.num_key_value_heads <= 0 ||
      cfg_.head_dim <= 0 || cfg_.linear_num_key_heads <= 0 ||
      cfg_.linear_num_value_heads <= 0 || cfg_.linear_key_head_dim <= 0 ||
      cfg_.linear_value_head_dim <= 0 || cfg_.linear_conv_kernel_dim <= 0) {
    throw std::runtime_error("Qwen3.5 config.json is missing required text_config fields");
  }
  if (cfg_.layer_kinds.size() != static_cast<std::size_t>(cfg_.num_layers)) {
    throw std::runtime_error("Qwen3.5 config.json layer_types does not match num_hidden_layers");
  }
}

void Qwen35CpuEngine::allocate_runtime_buffers() {
  max_ctx_ = options_.max_context > 0
      ? std::min(options_.max_context, cfg_.max_position_embeddings)
      : std::min(2048, cfg_.max_position_embeddings);
  bos_id_ = cfg_.eos_token_id >= 0 ? cfg_.eos_token_id : 0;
  rotary_dim_ = static_cast<int>(std::round(static_cast<float>(cfg_.head_dim) * cfg_.partial_rotary_factor));
  if (rotary_dim_ <= 0 || (rotary_dim_ % 2) != 0) {
    rotary_dim_ = std::max(2, cfg_.head_dim);
    if ((rotary_dim_ % 2) != 0) {
      --rotary_dim_;
    }
  }
  full_q_dim_ = cfg_.num_attention_heads * cfg_.head_dim;
  full_kv_dim_ = cfg_.num_key_value_heads * cfg_.head_dim;
  linear_k_dim_ = cfg_.linear_num_key_heads * cfg_.linear_key_head_dim;
  linear_v_dim_ = cfg_.linear_num_value_heads * cfg_.linear_value_head_dim;
  linear_conv_dim_ = linear_k_dim_ * 2 + linear_v_dim_;
  linear_head_repeat_ = cfg_.linear_num_value_heads / cfg_.linear_num_key_heads;

  if (linear_head_repeat_ <= 0 ||
      cfg_.linear_num_value_heads % cfg_.linear_num_key_heads != 0) {
    throw std::runtime_error("Qwen3.5 linear head counts are not divisible");
  }

  x_.resize(static_cast<std::size_t>(cfg_.hidden_size));
  x_norm_.resize(static_cast<std::size_t>(cfg_.hidden_size));
  logits_.resize(static_cast<std::size_t>(cfg_.vocab_size));

  full_qkv_.resize(static_cast<std::size_t>(full_q_dim_) * 2);
  full_q_.resize(static_cast<std::size_t>(full_q_dim_));
  full_q_gate_.resize(static_cast<std::size_t>(full_q_dim_));
  full_k_.resize(static_cast<std::size_t>(full_kv_dim_));
  full_v_.resize(static_cast<std::size_t>(full_kv_dim_));
  full_att_.resize(static_cast<std::size_t>(full_q_dim_));
  full_scores_.resize(static_cast<std::size_t>(cfg_.num_attention_heads) *
                      static_cast<std::size_t>(max_ctx_));

  linear_qkv_mix_.resize(static_cast<std::size_t>(linear_conv_dim_));
  linear_q_.resize(static_cast<std::size_t>(cfg_.linear_num_value_heads) *
                   static_cast<std::size_t>(cfg_.linear_key_head_dim));
  linear_k_.resize(static_cast<std::size_t>(cfg_.linear_num_value_heads) *
                   static_cast<std::size_t>(cfg_.linear_key_head_dim));
  linear_v_.resize(static_cast<std::size_t>(linear_v_dim_));
  linear_z_.resize(static_cast<std::size_t>(linear_v_dim_));
  linear_a_.resize(static_cast<std::size_t>(cfg_.linear_num_value_heads));
  linear_b_.resize(static_cast<std::size_t>(cfg_.linear_num_value_heads));
  linear_att_.resize(static_cast<std::size_t>(linear_v_dim_));

  mlp_gate_buf_.resize(static_cast<std::size_t>(cfg_.intermediate_size));
  mlp_up_buf_.resize(static_cast<std::size_t>(cfg_.intermediate_size));
  mlp_down_buf_.resize(static_cast<std::size_t>(
      std::max(cfg_.intermediate_size, cfg_.linear_value_head_dim)));

  const int rotary_half = rotary_dim_ / 2;
  rope_cos_.resize(static_cast<std::size_t>(max_ctx_) *
                   static_cast<std::size_t>(rotary_half));
  rope_sin_.resize(static_cast<std::size_t>(max_ctx_) *
                   static_cast<std::size_t>(rotary_half));
  for (int i = 0; i < rotary_half; ++i) {
    const float inv_freq = std::pow(
        cfg_.rope_theta,
        -2.0f * static_cast<float>(i) / static_cast<float>(rotary_dim_));
    for (int pos = 0; pos < max_ctx_; ++pos) {
      const float angle = static_cast<float>(pos) * inv_freq;
      rope_cos_[static_cast<std::size_t>(pos) * static_cast<std::size_t>(rotary_half) +
                static_cast<std::size_t>(i)] = std::cos(angle);
      rope_sin_[static_cast<std::size_t>(pos) * static_cast<std::size_t>(rotary_half) +
                static_cast<std::size_t>(i)] = std::sin(angle);
    }
  }

  full_k_cache_.assign(static_cast<std::size_t>(cfg_.num_layers), {});
  full_v_cache_.assign(static_cast<std::size_t>(cfg_.num_layers), {});
  linear_conv_state_.assign(static_cast<std::size_t>(cfg_.num_layers), {});
  linear_recurrent_state_.assign(static_cast<std::size_t>(cfg_.num_layers), {});

  for (int layer = 0; layer < cfg_.num_layers; ++layer) {
    if (cfg_.layer_kinds[static_cast<std::size_t>(layer)] ==
        LayerKind::FullAttention) {
      full_k_cache_[static_cast<std::size_t>(layer)].assign(
          static_cast<std::size_t>(max_ctx_) *
              static_cast<std::size_t>(full_kv_dim_),
          0.0f);
      full_v_cache_[static_cast<std::size_t>(layer)].assign(
          static_cast<std::size_t>(max_ctx_) *
              static_cast<std::size_t>(full_kv_dim_),
          0.0f);
    } else {
      linear_conv_state_[static_cast<std::size_t>(layer)].assign(
          static_cast<std::size_t>(linear_conv_dim_) *
              static_cast<std::size_t>(cfg_.linear_conv_kernel_dim - 1),
          0.0f);
      linear_recurrent_state_[static_cast<std::size_t>(layer)].assign(
          static_cast<std::size_t>(cfg_.linear_num_value_heads) *
              static_cast<std::size_t>(cfg_.linear_key_head_dim) *
              static_cast<std::size_t>(cfg_.linear_value_head_dim),
          0.0f);
    }
  }
}

void Qwen35CpuEngine::load_weight_pointers() {
  tok_embeddings_ =
      require_bf16_tensor(weights_, "model.language_model.embed_tokens.weight");
  norm_out_ = require_bf16_tensor(weights_, "model.language_model.norm.weight");
  lm_head_ = require_bf16_tensor(weights_, "lm_head.weight");

  layers_.resize(static_cast<std::size_t>(cfg_.num_layers));
  for (int layer = 0; layer < cfg_.num_layers; ++layer) {
    const std::string prefix = "model.language_model.layers." +
                               std::to_string(layer);
    LayerWeights& lw = layers_[static_cast<std::size_t>(layer)];
    lw.kind = cfg_.layer_kinds[static_cast<std::size_t>(layer)];
    lw.norm_att =
        require_bf16_tensor(weights_, prefix + ".input_layernorm.weight");
    lw.norm_ffn =
        require_bf16_tensor(weights_, prefix + ".post_attention_layernorm.weight");
    lw.mlp_gate = require_bf16_tensor(weights_, prefix + ".mlp.gate_proj.weight");
    lw.mlp_up = require_bf16_tensor(weights_, prefix + ".mlp.up_proj.weight");
    lw.mlp_down = require_bf16_tensor(weights_, prefix + ".mlp.down_proj.weight");

    if (lw.kind == LayerKind::FullAttention) {
      lw.full_q = require_bf16_tensor(weights_, prefix + ".self_attn.q_proj.weight");
      lw.full_k = require_bf16_tensor(weights_, prefix + ".self_attn.k_proj.weight");
      lw.full_v = require_bf16_tensor(weights_, prefix + ".self_attn.v_proj.weight");
      lw.full_o = require_bf16_tensor(weights_, prefix + ".self_attn.o_proj.weight");
      lw.full_q_norm =
          require_bf16_tensor(weights_, prefix + ".self_attn.q_norm.weight");
      lw.full_k_norm =
          require_bf16_tensor(weights_, prefix + ".self_attn.k_norm.weight");
    } else {
      lw.linear_qkv =
          require_bf16_tensor(weights_, prefix + ".linear_attn.in_proj_qkv.weight");
      lw.linear_z =
          require_bf16_tensor(weights_, prefix + ".linear_attn.in_proj_z.weight");
      lw.linear_a =
          require_bf16_tensor(weights_, prefix + ".linear_attn.in_proj_a.weight");
      lw.linear_b =
          require_bf16_tensor(weights_, prefix + ".linear_attn.in_proj_b.weight");
      lw.linear_out =
          require_bf16_tensor(weights_, prefix + ".linear_attn.out_proj.weight");
      lw.linear_conv =
          require_bf16_tensor(weights_, prefix + ".linear_attn.conv1d.weight");
      lw.linear_norm =
          require_f32_tensor(weights_, prefix + ".linear_attn.norm.weight");
      lw.linear_A_log =
          require_f32_tensor(weights_, prefix + ".linear_attn.A_log");
      lw.linear_dt_bias =
          require_bf16_tensor(weights_, prefix + ".linear_attn.dt_bias");
    }
  }
}

void Qwen35CpuEngine::apply_rope_partial(float* q, float* k, int position) {
  const int rotary_half = rotary_dim_ / 2;
  const float* cos_row = rope_cos_.data() +
      static_cast<std::size_t>(position) * static_cast<std::size_t>(rotary_half);
  const float* sin_row = rope_sin_.data() +
      static_cast<std::size_t>(position) * static_cast<std::size_t>(rotary_half);

  for (int head = 0; head < cfg_.num_attention_heads; ++head) {
    float* q_head = q + static_cast<std::size_t>(head) *
                          static_cast<std::size_t>(cfg_.head_dim);
    for (int d = 0; d < rotary_half; ++d) {
      const float q0 = q_head[d];
      const float q1 = q_head[d + rotary_half];
      const float c = cos_row[d];
      const float s = sin_row[d];
      q_head[d] = q0 * c - q1 * s;
      q_head[d + rotary_half] = q1 * c + q0 * s;
    }
  }

  for (int head = 0; head < cfg_.num_key_value_heads; ++head) {
    float* k_head = k + static_cast<std::size_t>(head) *
                          static_cast<std::size_t>(cfg_.head_dim);
    for (int d = 0; d < rotary_half; ++d) {
      const float k0 = k_head[d];
      const float k1 = k_head[d + rotary_half];
      const float c = cos_row[d];
      const float s = sin_row[d];
      k_head[d] = k0 * c - k1 * s;
      k_head[d + rotary_half] = k1 * c + k0 * s;
    }
  }
}

void Qwen35CpuEngine::run_full_attention_layer(int layer, int position) {
  const LayerWeights& lw = layers_[static_cast<std::size_t>(layer)];
  rmsnorm_offset(x_.data(), lw.norm_att, x_norm_.data(), cfg_.hidden_size,
                 cfg_.rms_norm_eps);

  gemv_bf16(lw.full_q, x_norm_.data(), full_qkv_.data(), full_q_dim_ * 2,
            cfg_.hidden_size);
  for (int head = 0; head < cfg_.num_attention_heads; ++head) {
    const float* src = full_qkv_.data() +
        static_cast<std::size_t>(head) * static_cast<std::size_t>(cfg_.head_dim) * 2;
    float* dst_q = full_q_.data() +
        static_cast<std::size_t>(head) * static_cast<std::size_t>(cfg_.head_dim);
    float* dst_gate = full_q_gate_.data() +
        static_cast<std::size_t>(head) * static_cast<std::size_t>(cfg_.head_dim);
    std::memcpy(dst_q, src, static_cast<std::size_t>(cfg_.head_dim) * sizeof(float));
    std::memcpy(dst_gate, src + static_cast<std::size_t>(cfg_.head_dim),
                static_cast<std::size_t>(cfg_.head_dim) * sizeof(float));
  }
  gemv_bf16(lw.full_k, x_norm_.data(), full_k_.data(), full_kv_dim_,
            cfg_.hidden_size);
  gemv_bf16(lw.full_v, x_norm_.data(), full_v_.data(), full_kv_dim_,
            cfg_.hidden_size);

  for (int head = 0; head < cfg_.num_attention_heads; ++head) {
    rmsnorm_offset_inplace(
        full_q_.data() +
            static_cast<std::size_t>(head) * static_cast<std::size_t>(cfg_.head_dim),
        lw.full_q_norm, cfg_.head_dim, cfg_.rms_norm_eps);
  }
  for (int head = 0; head < cfg_.num_key_value_heads; ++head) {
    rmsnorm_offset_inplace(
        full_k_.data() +
            static_cast<std::size_t>(head) * static_cast<std::size_t>(cfg_.head_dim),
        lw.full_k_norm, cfg_.head_dim, cfg_.rms_norm_eps);
  }

  apply_rope_partial(full_q_.data(), full_k_.data(), position);

  float* k_cache = full_k_cache_[static_cast<std::size_t>(layer)].data();
  float* v_cache = full_v_cache_[static_cast<std::size_t>(layer)].data();
  std::memcpy(k_cache + static_cast<std::size_t>(position) *
                           static_cast<std::size_t>(full_kv_dim_),
              full_k_.data(),
              static_cast<std::size_t>(full_kv_dim_) * sizeof(float));
  std::memcpy(v_cache + static_cast<std::size_t>(position) *
                           static_cast<std::size_t>(full_kv_dim_),
              full_v_.data(),
              static_cast<std::size_t>(full_kv_dim_) * sizeof(float));

  const int seq_len = position + 1;
  const int kv_mul = cfg_.num_attention_heads / cfg_.num_key_value_heads;
  const float scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));

#pragma omp parallel for schedule(static)
  for (int head = 0; head < cfg_.num_attention_heads; ++head) {
    const int kv_head = head / kv_mul;
    const float* q_head = full_q_.data() +
        static_cast<std::size_t>(head) * static_cast<std::size_t>(cfg_.head_dim);
    float* scores = full_scores_.data() +
        static_cast<std::size_t>(head) * static_cast<std::size_t>(max_ctx_);

    float max_score = -1.0e30f;
    for (int t = 0; t < seq_len; ++t) {
      const float* k_head =
          k_cache + static_cast<std::size_t>(t) * static_cast<std::size_t>(full_kv_dim_) +
          static_cast<std::size_t>(kv_head) * static_cast<std::size_t>(cfg_.head_dim);
      float dot = 0.0f;
      for (int d = 0; d < cfg_.head_dim; ++d) {
        dot += q_head[d] * k_head[d];
      }
      scores[t] = dot * scale;
      max_score = std::max(max_score, scores[t]);
    }

    float sum = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
      scores[t] = std::exp(scores[t] - max_score);
      sum += scores[t];
    }
    const float inv_sum = 1.0f / std::max(sum, 1e-20f);
    for (int t = 0; t < seq_len; ++t) {
      scores[t] *= inv_sum;
    }

    float* out_head = full_att_.data() +
        static_cast<std::size_t>(head) * static_cast<std::size_t>(cfg_.head_dim);
    std::fill(out_head, out_head + cfg_.head_dim, 0.0f);
    for (int t = 0; t < seq_len; ++t) {
      const float* v_head =
          v_cache + static_cast<std::size_t>(t) * static_cast<std::size_t>(full_kv_dim_) +
          static_cast<std::size_t>(kv_head) * static_cast<std::size_t>(cfg_.head_dim);
      const float weight = scores[t];
      for (int d = 0; d < cfg_.head_dim; ++d) {
        out_head[d] += weight * v_head[d];
      }
    }
  }

  for (int i = 0; i < full_q_dim_; ++i) {
    full_att_[static_cast<std::size_t>(i)] *=
        sigmoid(full_q_gate_[static_cast<std::size_t>(i)]);
  }

  gemv_bf16(lw.full_o, full_att_.data(), x_norm_.data(), cfg_.hidden_size,
            full_q_dim_);
  for (int i = 0; i < cfg_.hidden_size; ++i) {
    x_[static_cast<std::size_t>(i)] += x_norm_[static_cast<std::size_t>(i)];
  }
}

void Qwen35CpuEngine::run_linear_attention_layer(int layer) {
  const LayerWeights& lw = layers_[static_cast<std::size_t>(layer)];
  rmsnorm_offset(x_.data(), lw.norm_att, x_norm_.data(), cfg_.hidden_size,
                 cfg_.rms_norm_eps);

  gemv_bf16(lw.linear_qkv, x_norm_.data(), linear_qkv_mix_.data(),
            linear_conv_dim_, cfg_.hidden_size);
  gemv_bf16(lw.linear_z, x_norm_.data(), linear_z_.data(), linear_v_dim_,
            cfg_.hidden_size);
  gemv_bf16(lw.linear_b, x_norm_.data(), linear_b_.data(),
            cfg_.linear_num_value_heads, cfg_.hidden_size);
  gemv_bf16(lw.linear_a, x_norm_.data(), linear_a_.data(),
            cfg_.linear_num_value_heads, cfg_.hidden_size);

  const int kernel = cfg_.linear_conv_kernel_dim;
  std::vector<float>& conv_state = linear_conv_state_[static_cast<std::size_t>(layer)];
  if (kernel > 1) {
    const int state_len = kernel - 1;
    for (int channel = 0; channel < linear_conv_dim_; ++channel) {
      const std::uint16_t* w = lw.linear_conv +
          static_cast<std::size_t>(channel) * static_cast<std::size_t>(kernel);
      float out = 0.0f;
      float* state_row = conv_state.data() +
          static_cast<std::size_t>(channel) * static_cast<std::size_t>(state_len);
      for (int j = 0; j < state_len; ++j) {
        out += bf16_to_f32(w[j]) * state_row[j];
      }
      out += bf16_to_f32(w[kernel - 1]) * linear_qkv_mix_[static_cast<std::size_t>(channel)];
      for (int j = 0; j + 1 < state_len; ++j) {
        state_row[j] = state_row[j + 1];
      }
      state_row[state_len - 1] = linear_qkv_mix_[static_cast<std::size_t>(channel)];
      linear_qkv_mix_[static_cast<std::size_t>(channel)] = silu(out);
    }
  } else {
    for (float& value : linear_qkv_mix_) {
      value = silu(value);
    }
  }

  const float* q_raw = linear_qkv_mix_.data();
  const float* k_raw = q_raw + linear_k_dim_;
  const float* v_raw = k_raw + linear_k_dim_;
  std::memcpy(linear_v_.data(), v_raw,
              static_cast<std::size_t>(linear_v_dim_) * sizeof(float));

  for (int k_head = 0; k_head < cfg_.linear_num_key_heads; ++k_head) {
    const float* src_q = q_raw + static_cast<std::size_t>(k_head) *
                                     static_cast<std::size_t>(cfg_.linear_key_head_dim);
    const float* src_k = k_raw + static_cast<std::size_t>(k_head) *
                                     static_cast<std::size_t>(cfg_.linear_key_head_dim);
    for (int rep = 0; rep < linear_head_repeat_; ++rep) {
      const int v_head = k_head * linear_head_repeat_ + rep;
      float* dst_q = linear_q_.data() + static_cast<std::size_t>(v_head) *
                                         static_cast<std::size_t>(cfg_.linear_key_head_dim);
      float* dst_k = linear_k_.data() + static_cast<std::size_t>(v_head) *
                                         static_cast<std::size_t>(cfg_.linear_key_head_dim);
      std::memcpy(dst_q, src_q,
                  static_cast<std::size_t>(cfg_.linear_key_head_dim) * sizeof(float));
      std::memcpy(dst_k, src_k,
                  static_cast<std::size_t>(cfg_.linear_key_head_dim) * sizeof(float));
    }
  }

  std::vector<float>& recurrent =
      linear_recurrent_state_[static_cast<std::size_t>(layer)];
  const float scale = 1.0f /
      std::sqrt(static_cast<float>(cfg_.linear_key_head_dim));
  for (int head = 0; head < cfg_.linear_num_value_heads; ++head) {
    float* q_head = linear_q_.data() + static_cast<std::size_t>(head) *
                                       static_cast<std::size_t>(cfg_.linear_key_head_dim);
    float* k_head = linear_k_.data() + static_cast<std::size_t>(head) *
                                       static_cast<std::size_t>(cfg_.linear_key_head_dim);
    float* v_head = linear_v_.data() + static_cast<std::size_t>(head) *
                                       static_cast<std::size_t>(cfg_.linear_value_head_dim);
    float* z_head = linear_z_.data() + static_cast<std::size_t>(head) *
                                       static_cast<std::size_t>(cfg_.linear_value_head_dim);
    float* out_head = linear_att_.data() + static_cast<std::size_t>(head) *
                                         static_cast<std::size_t>(cfg_.linear_value_head_dim);
    float* state = recurrent.data() + static_cast<std::size_t>(head) *
                       static_cast<std::size_t>(cfg_.linear_key_head_dim) *
                       static_cast<std::size_t>(cfg_.linear_value_head_dim);

    l2norm_inplace(q_head, cfg_.linear_key_head_dim);
    l2norm_inplace(k_head, cfg_.linear_key_head_dim);
    for (int d = 0; d < cfg_.linear_key_head_dim; ++d) {
      q_head[d] *= scale;
    }

    const float beta = sigmoid(linear_b_[static_cast<std::size_t>(head)]);
    const float a = linear_a_[static_cast<std::size_t>(head)];
    const float dt_bias = bf16_to_f32(lw.linear_dt_bias[head]);
    const float decay =
        std::exp(-std::exp(lw.linear_A_log[head]) * softplus(a + dt_bias));

    for (int k = 0; k < cfg_.linear_key_head_dim; ++k) {
      float* state_row = state + static_cast<std::size_t>(k) *
                                     static_cast<std::size_t>(cfg_.linear_value_head_dim);
      for (int dv = 0; dv < cfg_.linear_value_head_dim; ++dv) {
        state_row[dv] *= decay;
      }
    }

    float* scratch = mlp_down_buf_.data();
    for (int dv = 0; dv < cfg_.linear_value_head_dim; ++dv) {
      float kv_mem = 0.0f;
      for (int k = 0; k < cfg_.linear_key_head_dim; ++k) {
        kv_mem += state[static_cast<std::size_t>(k) *
                            static_cast<std::size_t>(cfg_.linear_value_head_dim) +
                        static_cast<std::size_t>(dv)] *
                  k_head[k];
      }
      scratch[dv] = (v_head[dv] - kv_mem) * beta;
    }

    for (int k = 0; k < cfg_.linear_key_head_dim; ++k) {
      float* state_row = state + static_cast<std::size_t>(k) *
                                     static_cast<std::size_t>(cfg_.linear_value_head_dim);
      const float kk = k_head[k];
      for (int dv = 0; dv < cfg_.linear_value_head_dim; ++dv) {
        state_row[dv] += kk * scratch[dv];
      }
    }

    for (int dv = 0; dv < cfg_.linear_value_head_dim; ++dv) {
      float sum = 0.0f;
      for (int k = 0; k < cfg_.linear_key_head_dim; ++k) {
        sum += state[static_cast<std::size_t>(k) *
                         static_cast<std::size_t>(cfg_.linear_value_head_dim) +
                     static_cast<std::size_t>(dv)] *
               q_head[k];
      }
      out_head[dv] = sum;
    }

    rmsnorm_direct_gated_inplace(out_head, lw.linear_norm, z_head,
                                 cfg_.linear_value_head_dim, cfg_.rms_norm_eps);
  }

  gemv_bf16(lw.linear_out, linear_att_.data(), x_norm_.data(),
            cfg_.hidden_size, linear_v_dim_);
  for (int i = 0; i < cfg_.hidden_size; ++i) {
    x_[static_cast<std::size_t>(i)] += x_norm_[static_cast<std::size_t>(i)];
  }
}

void Qwen35CpuEngine::run_mlp_layer(int layer) {
  const LayerWeights& lw = layers_[static_cast<std::size_t>(layer)];
  rmsnorm_offset(x_.data(), lw.norm_ffn, x_norm_.data(), cfg_.hidden_size,
                 cfg_.rms_norm_eps);
  gemv_bf16(lw.mlp_gate, x_norm_.data(), mlp_gate_buf_.data(),
            cfg_.intermediate_size, cfg_.hidden_size);
  gemv_bf16(lw.mlp_up, x_norm_.data(), mlp_up_buf_.data(),
            cfg_.intermediate_size, cfg_.hidden_size);
  for (int i = 0; i < cfg_.intermediate_size; ++i) {
    mlp_down_buf_[static_cast<std::size_t>(i)] =
        silu(mlp_gate_buf_[static_cast<std::size_t>(i)]) *
        mlp_up_buf_[static_cast<std::size_t>(i)];
  }
  gemv_bf16(lw.mlp_down, mlp_down_buf_.data(), x_norm_.data(),
            cfg_.hidden_size, cfg_.intermediate_size);
  for (int i = 0; i < cfg_.hidden_size; ++i) {
    x_[static_cast<std::size_t>(i)] += x_norm_[static_cast<std::size_t>(i)];
  }
}

void Qwen35CpuEngine::forward_token(int token, int position) {
  const std::uint16_t* emb_row =
      tok_embeddings_ + static_cast<std::size_t>(token) *
                            static_cast<std::size_t>(cfg_.hidden_size);
  for (int i = 0; i < cfg_.hidden_size; ++i) {
    x_[static_cast<std::size_t>(i)] = bf16_to_f32(emb_row[i]);
  }

  for (int layer = 0; layer < cfg_.num_layers; ++layer) {
    if (layers_[static_cast<std::size_t>(layer)].kind == LayerKind::FullAttention) {
      run_full_attention_layer(layer, position);
    } else {
      run_linear_attention_layer(layer);
    }
    run_mlp_layer(layer);
  }

  rmsnorm_offset(x_.data(), norm_out_, x_norm_.data(), cfg_.hidden_size,
                 cfg_.rms_norm_eps);
  gemv_bf16(lm_head_, x_norm_.data(), logits_.data(), cfg_.vocab_size,
            cfg_.hidden_size);
}

int Qwen35CpuEngine::sample_token(float temperature,
                                  int top_k,
                                  const std::vector<int>& history,
                                  float repetition_penalty) {
  if (repetition_penalty != 1.0f) {
    for (int token : history) {
      if (token < 0 || token >= cfg_.vocab_size) {
        continue;
      }
      float& logit = logits_[static_cast<std::size_t>(token)];
      logit = (logit > 0.0f) ? (logit / repetition_penalty)
                             : (logit * repetition_penalty);
    }
  }

  if (temperature <= 0.0f || top_k == 1) {
    return static_cast<int>(
        std::max_element(logits_.begin(), logits_.end()) - logits_.begin());
  }

  const int k = std::clamp(top_k, 1, cfg_.vocab_size);
  std::vector<int> ids(static_cast<std::size_t>(cfg_.vocab_size));
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                    [&](int left, int right) {
                      return logits_[static_cast<std::size_t>(left)] >
                             logits_[static_cast<std::size_t>(right)];
                    });

  const float max_logit = logits_[static_cast<std::size_t>(ids[0])];
  std::vector<float> probs(static_cast<std::size_t>(k));
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    probs[static_cast<std::size_t>(i)] =
        std::exp((logits_[static_cast<std::size_t>(
                      ids[static_cast<std::size_t>(i)])] -
                  max_logit) /
                 temperature);
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

void Qwen35CpuEngine::initialize(const EngineOptions& options) {
  options_ = options;
  load_config(options.model_path);
  weights_.open(options.model_path);
  allocate_runtime_buffers();
  load_weight_pointers();

  if (options_.verbose) {
    std::fprintf(stderr,
                 "[qwen3_5_cpu] layers=%d hidden=%d heads=%d/%d linear_heads=%d/%d vocab=%d max_ctx=%d\n",
                 cfg_.num_layers, cfg_.hidden_size, cfg_.num_attention_heads,
                 cfg_.num_key_value_heads, cfg_.linear_num_key_heads,
                 cfg_.linear_num_value_heads, cfg_.vocab_size, max_ctx_);
  }
}

std::vector<int> Qwen35CpuEngine::generate(const std::vector<int>& prompt_tokens,
                                           int max_new_tokens,
                                           float temperature) {
  return generate_stream(prompt_tokens, max_new_tokens, temperature,
                         [](int) { return true; });
}

std::vector<int> Qwen35CpuEngine::generate_stream(
    const std::vector<int>& prompt_tokens,
    int max_new_tokens,
    float temperature,
    const std::function<bool(int)>& on_token) {
  stats_ = BenchmarkStats{};
  stats_.prompt_tokens = static_cast<int>(prompt_tokens.size());
  for (auto& cache : full_k_cache_) {
    std::fill(cache.begin(), cache.end(), 0.0f);
  }
  for (auto& cache : full_v_cache_) {
    std::fill(cache.begin(), cache.end(), 0.0f);
  }
  for (auto& state : linear_conv_state_) {
    std::fill(state.begin(), state.end(), 0.0f);
  }
  for (auto& state : linear_recurrent_state_) {
    std::fill(state.begin(), state.end(), 0.0f);
  }

  std::vector<int> output = prompt_tokens;
  std::vector<int> history;

  const auto prefill_start = std::chrono::steady_clock::now();
  int pos = 0;
  if (prompt_tokens.empty()) {
    forward_token(bos_id_, 0);
    pos = 1;
  } else {
    for (int i = 0; i < static_cast<int>(prompt_tokens.size()) && i < max_ctx_;
         ++i) {
      forward_token(prompt_tokens[static_cast<std::size_t>(i)], i);
      pos = i + 1;
    }
  }
  const auto prefill_end = std::chrono::steady_clock::now();
  stats_.prefill_ms = std::chrono::duration<double, std::milli>(
                          prefill_end - prefill_start)
                          .count();

  const auto decode_start = std::chrono::steady_clock::now();
  for (int step = 0; step < max_new_tokens; ++step) {
    const int next =
        sample_token(temperature, options_.top_k, history, options_.repetition_penalty);
    history.push_back(next);
    output.push_back(next);
    ++stats_.generated_tokens;

    if (next == options_.eos_token_id || next == cfg_.eos_token_id) {
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
  stats_.decode_ms = std::chrono::duration<double, std::milli>(
                         decode_end - decode_start)
                         .count();
  return output;
}

std::vector<std::pair<int, float>> Qwen35CpuEngine::inspect_next_logits(
    const std::vector<int>& prompt_tokens,
    int top_k) {
  for (auto& cache : full_k_cache_) {
    std::fill(cache.begin(), cache.end(), 0.0f);
  }
  for (auto& cache : full_v_cache_) {
    std::fill(cache.begin(), cache.end(), 0.0f);
  }
  for (auto& state : linear_conv_state_) {
    std::fill(state.begin(), state.end(), 0.0f);
  }
  for (auto& state : linear_recurrent_state_) {
    std::fill(state.begin(), state.end(), 0.0f);
  }

  if (prompt_tokens.empty()) {
    forward_token(bos_id_, 0);
  } else {
    for (int i = 0; i < static_cast<int>(prompt_tokens.size()) && i < max_ctx_;
         ++i) {
      forward_token(prompt_tokens[static_cast<std::size_t>(i)], i);
    }
  }

  const int k = std::clamp(top_k, 0, cfg_.vocab_size);
  std::vector<std::pair<int, float>> pairs;
  pairs.reserve(static_cast<std::size_t>(cfg_.vocab_size));
  for (int i = 0; i < cfg_.vocab_size; ++i) {
    pairs.emplace_back(i, logits_[static_cast<std::size_t>(i)]);
  }
  std::partial_sort(pairs.begin(), pairs.begin() + k, pairs.end(),
                    [](const auto& left, const auto& right) {
                      return left.second > right.second;
                    });
  pairs.resize(static_cast<std::size_t>(k));
  return pairs;
}

}  // namespace engine
