#include "engine/qwen35_cuda_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_fp16.h>

#include "common.hpp"
#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"

namespace engine {
namespace {

float bf16_to_float(std::uint16_t bits) {
  const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16;
  float out = 0.0f;
  std::memcpy(&out, &word, sizeof(out));
  return out;
}

std::uint16_t float_to_half_bits(float value) {
  const __half h = __float2half(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

float half_bits_to_float(std::uint16_t bits) {
  __half h{};
  std::memcpy(&h, &bits, sizeof(bits));
  return __half2float(h);
}

float silu(float x) {
  return x / (1.0f + std::exp(-x));
}

float sigmoid(float x) {
  return 1.0f / (1.0f + std::exp(-x));
}

float softplus(float x) {
  if (x > 20.0f) return x;
  if (x < -20.0f) return std::exp(x);
  return std::log1p(std::exp(x));
}

void l2norm_inplace(float* x, int n, float eps = 1.0e-6f) {
  float ss = 0.0f;
  for (int i = 0; i < n; ++i) ss += x[i] * x[i];
  const float inv = 1.0f / std::sqrt(ss + eps);
  for (int i = 0; i < n; ++i) x[i] *= inv;
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
    if (c == '"') break;
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
  if (pos == std::string::npos) return "";
  skip_ws(json, pos);
  if (pos >= json.size() || json[pos] != '{') return "";
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
  if (pos == std::string::npos) return def;
  skip_ws(json, pos);
  if (pos >= json.size() || json[pos] != '"') return def;
  return json_read_string(json, pos);
}

int json_get_int(const std::string& json, const std::string& key, int def = 0) {
  std::size_t pos = json_find_key(json, key);
  if (pos == std::string::npos) return def;
  skip_ws(json, pos);
  std::size_t end = pos;
  while (end < json.size() &&
         (std::isdigit(static_cast<unsigned char>(json[end])) ||
          json[end] == '-' || json[end] == '+')) {
    ++end;
  }
  if (end == pos) return def;
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
  if (pos == std::string::npos) return def;
  skip_ws(json, pos);
  std::size_t end = pos;
  while (end < json.size() &&
         (std::isdigit(static_cast<unsigned char>(json[end])) ||
          json[end] == '-' || json[end] == '+' || json[end] == '.' ||
          json[end] == 'e' || json[end] == 'E')) {
    ++end;
  }
  if (end == pos) return def;
  try {
    return std::stof(json.substr(pos, end - pos));
  } catch (...) {
    return def;
  }
}

std::vector<std::string> json_get_string_array(const std::string& json,
                                               const std::string& key) {
  std::size_t pos = json_find_key(json, key);
  if (pos == std::string::npos) return {};
  skip_ws(json, pos);
  if (pos >= json.size() || json[pos] != '[') return {};
  ++pos;
  std::vector<std::string> out;
  while (pos < json.size()) {
    skip_ws(json, pos);
    if (pos >= json.size() || json[pos] == ']') break;
    if (json[pos] == '"') {
      out.push_back(json_read_string(json, pos));
    } else {
      ++pos;
    }
    while (pos < json.size() && json[pos] != '"' && json[pos] != ']') ++pos;
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

void free_device_void(void*& ptr) {
  if (ptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

template <typename T>
void free_device_ptr(T*& ptr) {
  if (ptr) {
    cudaFree(ptr);
    ptr = nullptr;
  }
}

std::vector<std::uint16_t> convert_bf16_to_fp16_host(const std::uint16_t* src,
                                                     std::size_t elems) {
  std::vector<std::uint16_t> out(elems);
  for (std::size_t i = 0; i < elems; ++i) {
    out[i] = float_to_half_bits(bf16_to_float(src[i]));
  }
  return out;
}

}  // namespace

Qwen35CudaEngine::~Qwen35CudaEngine() {
  destroy();
}

void Qwen35CudaEngine::destroy() {
  auto free_matrix = [](DeviceMatrix& m) {
    free_device_void(m.data);
    free_device_ptr(m.scales);
    m = DeviceMatrix{};
  };
  for (auto& layer : layers_) {
    free_matrix(layer.norm_att);
    free_matrix(layer.norm_ffn);
    free_matrix(layer.mlp_gate);
    free_matrix(layer.mlp_up);
    free_matrix(layer.mlp_down);
    free_matrix(layer.full_q);
    free_matrix(layer.full_k);
    free_matrix(layer.full_v);
    free_matrix(layer.full_o);
    free_matrix(layer.linear_qkv);
    free_matrix(layer.linear_z);
    free_matrix(layer.linear_a);
    free_matrix(layer.linear_b);
    free_matrix(layer.linear_out);
    free_device_void(layer.full_q_norm);
    free_device_void(layer.full_k_norm);
    layer.linear_conv = nullptr;
    layer.linear_norm = nullptr;
    layer.linear_A_log = nullptr;
    layer.linear_dt_bias = nullptr;
  }
  layers_.clear();
  free_device_void(d_norm_out_);
  free_matrix(lm_head_);
  free_device_void(d_rope_cos_);
  free_device_void(d_rope_sin_);
  free_device_void(d_x_);
  free_device_void(d_x_norm_);
  free_device_void(d_q_pair_);
  free_device_void(d_q_);
  free_device_void(d_q_gate_);
  free_device_void(d_k_);
  free_device_void(d_v_);
  free_device_void(d_att_);
  free_device_void(d_tmp_hidden_);
  free_device_void(d_logits_);
  free_device_ptr(d_argmax_);
  free_device_void(d_mlp_gate_);
  free_device_void(d_mlp_up_);
  free_device_void(d_mlp_inter_);
  free_device_void(d_linear_qkv_mix_);
  free_device_void(d_linear_z_);
  free_device_void(d_linear_a_);
  free_device_void(d_linear_b_);
  free_device_void(d_linear_att_);
  free_device_void(d_k_cache_);
  free_device_void(d_v_cache_);
  if (cublas_) {
    cublasDestroy(cublas_);
    cublas_ = nullptr;
  }
  if (compute_stream_) {
    cudaStreamDestroy(compute_stream_);
    compute_stream_ = nullptr;
  }
  tok_embeddings_ = nullptr;
  h_token_embedding_fp16_.clear();
  h_logits_.clear();
  h_linear_qkv_mix_bits_.clear();
  h_linear_z_bits_.clear();
  h_linear_a_bits_.clear();
  h_linear_b_bits_.clear();
  h_linear_att_bits_.clear();
  h_linear_qkv_mix_.clear();
  h_linear_z_.clear();
  h_linear_a_.clear();
  h_linear_b_.clear();
  h_linear_att_.clear();
  h_linear_q_.clear();
  h_linear_k_.clear();
  h_linear_v_.clear();
  linear_conv_state_.clear();
  linear_recurrent_state_.clear();
}

void Qwen35CudaEngine::load_config(const std::string& model_dir) {
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

void Qwen35CudaEngine::allocate_runtime_buffers() {
  max_ctx_ = options_.max_context > 0
      ? std::min(options_.max_context, cfg_.max_position_embeddings)
      : std::min(2048, cfg_.max_position_embeddings);
  bos_id_ = cfg_.eos_token_id >= 0 ? cfg_.eos_token_id : 0;
  rotary_dim_ = static_cast<int>(std::round(static_cast<float>(cfg_.head_dim) * cfg_.partial_rotary_factor));
  if (rotary_dim_ <= 0 || (rotary_dim_ % 2) != 0) {
    rotary_dim_ = cfg_.head_dim - (cfg_.head_dim % 2);
  }
  full_q_dim_ = cfg_.num_attention_heads * cfg_.head_dim;
  full_kv_dim_ = cfg_.num_key_value_heads * cfg_.head_dim;
  linear_k_dim_ = cfg_.linear_num_key_heads * cfg_.linear_key_head_dim;
  linear_v_dim_ = cfg_.linear_num_value_heads * cfg_.linear_value_head_dim;
  linear_conv_dim_ = linear_k_dim_ * 2 + linear_v_dim_;
  linear_head_repeat_ = cfg_.linear_num_value_heads / cfg_.linear_num_key_heads;

  auto malloc_device = [](void** ptr, std::size_t bytes) {
    CUDA_CHECK(cudaMalloc(ptr, bytes));
  };
  malloc_device(&d_rope_cos_, static_cast<std::size_t>(max_ctx_) * static_cast<std::size_t>(rotary_dim_ / 2) * sizeof(float));
  malloc_device(&d_rope_sin_, static_cast<std::size_t>(max_ctx_) * static_cast<std::size_t>(rotary_dim_ / 2) * sizeof(float));
  malloc_device(&d_x_, static_cast<std::size_t>(cfg_.hidden_size) * sizeof(__half));
  malloc_device(&d_x_norm_, static_cast<std::size_t>(cfg_.hidden_size) * sizeof(__half));
  malloc_device(&d_q_pair_, static_cast<std::size_t>(full_q_dim_ * 2) * sizeof(__half));
  malloc_device(&d_q_, static_cast<std::size_t>(full_q_dim_) * sizeof(__half));
  malloc_device(&d_q_gate_, static_cast<std::size_t>(full_q_dim_) * sizeof(__half));
  malloc_device(&d_k_, static_cast<std::size_t>(full_kv_dim_) * sizeof(__half));
  malloc_device(&d_v_, static_cast<std::size_t>(full_kv_dim_) * sizeof(__half));
  malloc_device(&d_att_, static_cast<std::size_t>(cfg_.hidden_size) * sizeof(__half));
  malloc_device(&d_tmp_hidden_, static_cast<std::size_t>(cfg_.hidden_size) * sizeof(__half));
  malloc_device(&d_logits_, static_cast<std::size_t>(cfg_.vocab_size) * sizeof(float));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_argmax_), sizeof(int)));
  malloc_device(&d_mlp_gate_, static_cast<std::size_t>(cfg_.intermediate_size) * sizeof(__half));
  malloc_device(&d_mlp_up_, static_cast<std::size_t>(cfg_.intermediate_size) * sizeof(__half));
  malloc_device(&d_mlp_inter_, static_cast<std::size_t>(cfg_.intermediate_size) * sizeof(__half));
  malloc_device(&d_linear_qkv_mix_, static_cast<std::size_t>(linear_conv_dim_) * sizeof(__half));
  malloc_device(&d_linear_z_, static_cast<std::size_t>(linear_v_dim_) * sizeof(__half));
  malloc_device(&d_linear_a_, static_cast<std::size_t>(cfg_.linear_num_value_heads) * sizeof(__half));
  malloc_device(&d_linear_b_, static_cast<std::size_t>(cfg_.linear_num_value_heads) * sizeof(__half));
  malloc_device(&d_linear_att_, static_cast<std::size_t>(linear_v_dim_) * sizeof(__half));
  malloc_device(&d_k_cache_, static_cast<std::size_t>(cfg_.num_layers) * static_cast<std::size_t>(max_ctx_) * static_cast<std::size_t>(full_kv_dim_) * sizeof(__half));
  malloc_device(&d_v_cache_, static_cast<std::size_t>(cfg_.num_layers) * static_cast<std::size_t>(max_ctx_) * static_cast<std::size_t>(full_kv_dim_) * sizeof(__half));

  h_token_embedding_fp16_.resize(static_cast<std::size_t>(cfg_.hidden_size));
  h_logits_.resize(static_cast<std::size_t>(cfg_.vocab_size));
  h_linear_qkv_mix_bits_.resize(static_cast<std::size_t>(linear_conv_dim_));
  h_linear_z_bits_.resize(static_cast<std::size_t>(linear_v_dim_));
  h_linear_a_bits_.resize(static_cast<std::size_t>(cfg_.linear_num_value_heads));
  h_linear_b_bits_.resize(static_cast<std::size_t>(cfg_.linear_num_value_heads));
  h_linear_att_bits_.resize(static_cast<std::size_t>(linear_v_dim_));
  h_linear_qkv_mix_.resize(static_cast<std::size_t>(linear_conv_dim_));
  h_linear_z_.resize(static_cast<std::size_t>(linear_v_dim_));
  h_linear_a_.resize(static_cast<std::size_t>(cfg_.linear_num_value_heads));
  h_linear_b_.resize(static_cast<std::size_t>(cfg_.linear_num_value_heads));
  h_linear_att_.resize(static_cast<std::size_t>(linear_v_dim_));
  h_linear_q_.resize(static_cast<std::size_t>(cfg_.linear_num_value_heads) * static_cast<std::size_t>(cfg_.linear_key_head_dim));
  h_linear_k_.resize(static_cast<std::size_t>(cfg_.linear_num_value_heads) * static_cast<std::size_t>(cfg_.linear_key_head_dim));
  h_linear_v_.resize(static_cast<std::size_t>(linear_v_dim_));

  linear_conv_state_.assign(static_cast<std::size_t>(cfg_.num_layers), {});
  linear_recurrent_state_.assign(static_cast<std::size_t>(cfg_.num_layers), {});
  for (int layer = 0; layer < cfg_.num_layers; ++layer) {
    if (cfg_.layer_kinds[static_cast<std::size_t>(layer)] == LayerKind::LinearAttention) {
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

void Qwen35CudaEngine::build_rope_tables() {
  const int half_dim = rotary_dim_ / 2;
  std::vector<float> rope_cos(
      static_cast<std::size_t>(max_ctx_) * static_cast<std::size_t>(half_dim));
  std::vector<float> rope_sin(
      static_cast<std::size_t>(max_ctx_) * static_cast<std::size_t>(half_dim));

  for (int i = 0; i < half_dim; ++i) {
    const float inv_freq = std::pow(
        cfg_.rope_theta,
        -2.0f * static_cast<float>(i) / static_cast<float>(rotary_dim_));
    for (int pos = 0; pos < max_ctx_; ++pos) {
      const float angle = static_cast<float>(pos) * inv_freq;
      rope_cos[static_cast<std::size_t>(pos) * static_cast<std::size_t>(half_dim) +
               static_cast<std::size_t>(i)] = std::cos(angle);
      rope_sin[static_cast<std::size_t>(pos) * static_cast<std::size_t>(half_dim) +
               static_cast<std::size_t>(i)] = std::sin(angle);
    }
  }

  CUDA_CHECK(cudaMemcpyAsync(d_rope_cos_,
                             rope_cos.data(),
                             rope_cos.size() * sizeof(float),
                             cudaMemcpyHostToDevice,
                             compute_stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_rope_sin_,
                             rope_sin.data(),
                             rope_sin.size() * sizeof(float),
                             cudaMemcpyHostToDevice,
                             compute_stream_));
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
}

void Qwen35CudaEngine::reset_state() {
  const std::size_t cache_bytes =
      static_cast<std::size_t>(cfg_.num_layers) *
      static_cast<std::size_t>(max_ctx_) *
      static_cast<std::size_t>(full_kv_dim_) *
      sizeof(__half);
  CUDA_CHECK(cudaMemsetAsync(d_k_cache_, 0, cache_bytes, compute_stream_));
  CUDA_CHECK(cudaMemsetAsync(d_v_cache_, 0, cache_bytes, compute_stream_));
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  for (auto& state : linear_conv_state_) {
    std::fill(state.begin(), state.end(), 0.0f);
  }
  for (auto& state : linear_recurrent_state_) {
    std::fill(state.begin(), state.end(), 0.0f);
  }
}

void Qwen35CudaEngine::load_token_embedding_to_device(int token) {
  token = std::clamp(token, 0, cfg_.vocab_size - 1);
  const std::uint16_t* row =
      tok_embeddings_ +
      static_cast<std::size_t>(token) * static_cast<std::size_t>(cfg_.hidden_size);
  for (int i = 0; i < cfg_.hidden_size; ++i) {
    h_token_embedding_fp16_[static_cast<std::size_t>(i)] =
        float_to_half_bits(bf16_to_float(row[i]));
  }
  CUDA_CHECK(cudaMemcpyAsync(d_x_,
                             h_token_embedding_fp16_.data(),
                             static_cast<std::size_t>(cfg_.hidden_size) *
                                 sizeof(std::uint16_t),
                             cudaMemcpyHostToDevice,
                             compute_stream_));
}

void Qwen35CudaEngine::project(const DeviceMatrix& matrix, const void* x, void* y) {
  switch (matrix.kind) {
    case MatrixKind::Fp16:
      kernels::launch_rowmajor_half_gemv_f16(static_cast<const __half*>(matrix.data),
                                             static_cast<const __half*>(x),
                                             static_cast<__half*>(y),
                                             matrix.rows,
                                             matrix.cols,
                                             compute_stream_);
      return;
    case MatrixKind::Int8:
      kernels::launch_weight_only_int8_matvec(static_cast<const std::int8_t*>(matrix.data),
                                              matrix.scales,
                                              static_cast<const __half*>(x),
                                              static_cast<__half*>(y),
                                              matrix.rows,
                                              matrix.cols,
                                              compute_stream_);
      return;
    case MatrixKind::Int4:
      kernels::launch_weight_only_int4_matvec(static_cast<const std::int8_t*>(matrix.data),
                                              matrix.scales,
                                              static_cast<const __half*>(x),
                                              static_cast<__half*>(y),
                                              matrix.rows,
                                              matrix.cols,
                                              compute_stream_);
      return;
  }
}

void Qwen35CudaEngine::rowmajor_projection_float(const DeviceMatrix& matrix,
                                                 const void* x,
                                                 void* y) {
  if (matrix.kind != MatrixKind::Fp16) {
    LLAMA_ENGINE_THROW("float projection requires fp16 weights");
  }
  kernels::launch_rowmajor_half_gemv_f32(static_cast<const __half*>(matrix.data),
                                         static_cast<const __half*>(x),
                                         static_cast<float*>(y),
                                         matrix.rows,
                                         matrix.cols,
                                         compute_stream_);
}

void Qwen35CudaEngine::load_weights() {
  auto load_vector_fp16 = [&](const std::string& name, int elems) -> void* {
    const auto* src = require_bf16_tensor(weights_, name);
    const auto host = convert_bf16_to_fp16_host(src, static_cast<std::size_t>(elems));
    void* dst = nullptr;
    CUDA_CHECK(cudaMalloc(&dst, static_cast<std::size_t>(elems) * sizeof(__half)));
    CUDA_CHECK(cudaMemcpyAsync(dst,
                               host.data(),
                               host.size() * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice,
                               compute_stream_));
    return dst;
  };

  auto quantize_rowwise = [&](const std::uint16_t* src,
                              int rows,
                              int cols,
                              DeviceMatrix* out) {
    const int max_q = options_.streaming_quant_bits == 4 ? 7 : 127;
    const bool int4 = options_.streaming_quant_bits == 4;
    std::vector<float> scales(static_cast<std::size_t>(rows), 1.0f);
    std::vector<std::int8_t> q;
    if (int4) {
      q.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>((cols + 1) / 2), 0);
    } else {
      q.assign(static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols), 0);
    }
    for (int row = 0; row < rows; ++row) {
      const std::size_t row_off = static_cast<std::size_t>(row) * static_cast<std::size_t>(cols);
      float max_abs = 0.0f;
      for (int col = 0; col < cols; ++col) {
        max_abs = std::max(max_abs, std::fabs(bf16_to_float(src[row_off + static_cast<std::size_t>(col)])));
      }
      const float scale = max_abs > 0.0f ? (max_abs / static_cast<float>(max_q)) : 1.0f;
      scales[static_cast<std::size_t>(row)] = scale;
      if (int4) {
        const std::size_t dst_off =
            static_cast<std::size_t>(row) * static_cast<std::size_t>((cols + 1) / 2);
        for (int col = 0; col < cols; col += 2) {
          const auto pack_val = [&](int c) -> std::uint8_t {
            if (c >= cols) return 0;
            int v = static_cast<int>(std::lrint(
                bf16_to_float(src[row_off + static_cast<std::size_t>(c)]) / scale));
            v = std::clamp(v, -8, 7);
            return static_cast<std::uint8_t>(v < 0 ? v + 16 : v);
          };
          const std::uint8_t lo = pack_val(col);
          const std::uint8_t hi = pack_val(col + 1);
          q[dst_off + static_cast<std::size_t>(col / 2)] =
              static_cast<std::int8_t>(lo | (hi << 4));
        }
      } else {
        for (int col = 0; col < cols; ++col) {
          int v = static_cast<int>(std::lrint(
              bf16_to_float(src[row_off + static_cast<std::size_t>(col)]) / scale));
          v = std::clamp(v, -127, 127);
          q[row_off + static_cast<std::size_t>(col)] = static_cast<std::int8_t>(v);
        }
      }
    }

    out->kind = int4 ? MatrixKind::Int4 : MatrixKind::Int8;
    out->rows = rows;
    out->cols = cols;
    CUDA_CHECK(cudaMalloc(&out->data, q.size() * sizeof(std::int8_t)));
    CUDA_CHECK(cudaMemcpyAsync(out->data,
                               q.data(),
                               q.size() * sizeof(std::int8_t),
                               cudaMemcpyHostToDevice,
                               compute_stream_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&out->scales),
                          scales.size() * sizeof(float)));
    CUDA_CHECK(cudaMemcpyAsync(out->scales,
                               scales.data(),
                               scales.size() * sizeof(float),
                               cudaMemcpyHostToDevice,
                               compute_stream_));
  };

  auto load_matrix = [&](const std::string& name, int rows, int cols) -> DeviceMatrix {
    const auto* src = require_bf16_tensor(weights_, name);
    DeviceMatrix out;
    out.rows = rows;
    out.cols = cols;
    if (!options_.int8_streaming) {
      out.kind = MatrixKind::Fp16;
      const auto host = convert_bf16_to_fp16_host(
          src, static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
      CUDA_CHECK(cudaMalloc(&out.data, host.size() * sizeof(std::uint16_t)));
      CUDA_CHECK(cudaMemcpyAsync(out.data,
                                 host.data(),
                                 host.size() * sizeof(std::uint16_t),
                                 cudaMemcpyHostToDevice,
                                 compute_stream_));
      return out;
    }
    quantize_rowwise(src, rows, cols, &out);
    return out;
  };

  auto load_matrix_fp16_only = [&](const std::string& name, int rows, int cols) -> DeviceMatrix {
    const auto* src = require_bf16_tensor(weights_, name);
    DeviceMatrix out;
    out.kind = MatrixKind::Fp16;
    out.rows = rows;
    out.cols = cols;
    const auto host = convert_bf16_to_fp16_host(
        src, static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    CUDA_CHECK(cudaMalloc(&out.data, host.size() * sizeof(std::uint16_t)));
    CUDA_CHECK(cudaMemcpyAsync(out.data,
                               host.data(),
                               host.size() * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice,
                               compute_stream_));
    return out;
  };

  tok_embeddings_ =
      require_bf16_tensor(weights_, "model.language_model.embed_tokens.weight");
  d_norm_out_ = load_vector_fp16("model.language_model.norm.weight", cfg_.hidden_size);
  lm_head_.kind = MatrixKind::Fp16;
  lm_head_.rows = cfg_.vocab_size;
  lm_head_.cols = cfg_.hidden_size;
  lm_head_.data = load_vector_fp16("lm_head.weight", cfg_.vocab_size * cfg_.hidden_size);

  layers_.resize(static_cast<std::size_t>(cfg_.num_layers));
  for (int layer = 0; layer < cfg_.num_layers; ++layer) {
    const std::string prefix = "model.language_model.layers." + std::to_string(layer);
    LayerWeights& lw = layers_[static_cast<std::size_t>(layer)];
    lw.kind = cfg_.layer_kinds[static_cast<std::size_t>(layer)];
    lw.norm_att = load_matrix_fp16_only(prefix + ".input_layernorm.weight", 1, cfg_.hidden_size);
    lw.norm_ffn = load_matrix_fp16_only(prefix + ".post_attention_layernorm.weight", 1, cfg_.hidden_size);
    lw.mlp_gate = load_matrix(prefix + ".mlp.gate_proj.weight", cfg_.intermediate_size, cfg_.hidden_size);
    lw.mlp_up = load_matrix(prefix + ".mlp.up_proj.weight", cfg_.intermediate_size, cfg_.hidden_size);
    lw.mlp_down = load_matrix(prefix + ".mlp.down_proj.weight", cfg_.hidden_size, cfg_.intermediate_size);
    if (lw.kind == LayerKind::FullAttention) {
      lw.full_q = load_matrix(prefix + ".self_attn.q_proj.weight", full_q_dim_ * 2, cfg_.hidden_size);
      lw.full_k = load_matrix(prefix + ".self_attn.k_proj.weight", full_kv_dim_, cfg_.hidden_size);
      lw.full_v = load_matrix(prefix + ".self_attn.v_proj.weight", full_kv_dim_, cfg_.hidden_size);
      lw.full_o = load_matrix(prefix + ".self_attn.o_proj.weight", cfg_.hidden_size, full_q_dim_);
      lw.full_q_norm = load_vector_fp16(prefix + ".self_attn.q_norm.weight", cfg_.head_dim);
      lw.full_k_norm = load_vector_fp16(prefix + ".self_attn.k_norm.weight", cfg_.head_dim);
    } else {
      lw.linear_qkv = load_matrix(prefix + ".linear_attn.in_proj_qkv.weight", linear_conv_dim_, cfg_.hidden_size);
      lw.linear_z = load_matrix(prefix + ".linear_attn.in_proj_z.weight", linear_v_dim_, cfg_.hidden_size);
      lw.linear_a = load_matrix(prefix + ".linear_attn.in_proj_a.weight", cfg_.linear_num_value_heads, cfg_.hidden_size);
      lw.linear_b = load_matrix(prefix + ".linear_attn.in_proj_b.weight", cfg_.linear_num_value_heads, cfg_.hidden_size);
      lw.linear_out = load_matrix(prefix + ".linear_attn.out_proj.weight", cfg_.hidden_size, linear_v_dim_);
      lw.linear_conv = require_bf16_tensor(weights_, prefix + ".linear_attn.conv1d.weight");
      lw.linear_norm = require_f32_tensor(weights_, prefix + ".linear_attn.norm.weight");
      lw.linear_A_log = require_f32_tensor(weights_, prefix + ".linear_attn.A_log");
      lw.linear_dt_bias = require_bf16_tensor(weights_, prefix + ".linear_attn.dt_bias");
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
}

void Qwen35CudaEngine::forward_token(int token,
                                     int position,
                                     bool compute_logits,
                                     std::vector<float>* out_logits,
                                     int* out_argmax) {
  if (position < 0 || position >= max_ctx_) {
    LLAMA_ENGINE_THROW("decode position exceeds max context");
  }

  load_token_embedding_to_device(token);

  for (int layer = 0; layer < cfg_.num_layers; ++layer) {
    LayerWeights& lw = layers_[static_cast<std::size_t>(layer)];

    if (lw.kind == LayerKind::FullAttention) {
      kernels::launch_rmsnorm_offset(static_cast<const __half*>(d_x_),
                                     static_cast<const __half*>(lw.norm_att.data),
                                     static_cast<__half*>(d_x_norm_),
                                     1,
                                     cfg_.hidden_size,
                                     cfg_.rms_norm_eps,
                                     compute_stream_);
      project(lw.full_q, d_x_norm_, d_q_pair_);
      project(lw.full_k, d_x_norm_, d_k_);
      project(lw.full_v, d_x_norm_, d_v_);
      kernels::launch_split_interleaved_head_halves(
          static_cast<const __half*>(d_q_pair_),
          static_cast<__half*>(d_q_),
          static_cast<__half*>(d_q_gate_),
          cfg_.num_attention_heads,
          cfg_.head_dim,
          compute_stream_);

      kernels::launch_rmsnorm_offset(static_cast<const __half*>(d_q_),
                                     static_cast<const __half*>(lw.full_q_norm),
                                     static_cast<__half*>(d_q_),
                                     cfg_.num_attention_heads,
                                     cfg_.head_dim,
                                     cfg_.rms_norm_eps,
                                     compute_stream_);
      kernels::launch_rmsnorm_offset(static_cast<const __half*>(d_k_),
                                     static_cast<const __half*>(lw.full_k_norm),
                                     static_cast<__half*>(d_k_),
                                     cfg_.num_key_value_heads,
                                     cfg_.head_dim,
                                     cfg_.rms_norm_eps,
                                     compute_stream_);

      kernels::launch_rope_inplace_partial_table(static_cast<__half*>(d_q_),
                                                 static_cast<__half*>(d_k_),
                                                 cfg_.num_attention_heads,
                                                 cfg_.num_key_value_heads,
                                                 cfg_.head_dim,
                                                 rotary_dim_,
                                                 position,
                                                 static_cast<const float*>(d_rope_cos_),
                                                 static_cast<const float*>(d_rope_sin_),
                                                 compute_stream_);

      auto* k_cache_layer =
          static_cast<__half*>(d_k_cache_) +
          static_cast<std::size_t>(layer) * static_cast<std::size_t>(max_ctx_) *
              static_cast<std::size_t>(full_kv_dim_);
      auto* v_cache_layer =
          static_cast<__half*>(d_v_cache_) +
          static_cast<std::size_t>(layer) * static_cast<std::size_t>(max_ctx_) *
              static_cast<std::size_t>(full_kv_dim_);
      CUDA_CHECK(cudaMemcpyAsync(
          k_cache_layer + static_cast<std::size_t>(position) * static_cast<std::size_t>(full_kv_dim_),
          d_k_,
          static_cast<std::size_t>(full_kv_dim_) * sizeof(__half),
          cudaMemcpyDeviceToDevice,
          compute_stream_));
      CUDA_CHECK(cudaMemcpyAsync(
          v_cache_layer + static_cast<std::size_t>(position) * static_cast<std::size_t>(full_kv_dim_),
          d_v_,
          static_cast<std::size_t>(full_kv_dim_) * sizeof(__half),
          cudaMemcpyDeviceToDevice,
          compute_stream_));

      kernels::launch_attention_step(static_cast<const __half*>(d_q_),
                                     k_cache_layer,
                                     v_cache_layer,
                                     static_cast<__half*>(d_att_),
                                     position + 1,
                                     cfg_.num_attention_heads,
                                     cfg_.num_key_value_heads,
                                     cfg_.head_dim,
                                     compute_stream_);
      kernels::launch_apply_sigmoid_gate_inplace(static_cast<__half*>(d_att_),
                                                 static_cast<const __half*>(d_q_gate_),
                                                 full_q_dim_,
                                                 compute_stream_);
      project(lw.full_o, d_att_, d_tmp_hidden_);
      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  static_cast<const __half*>(d_tmp_hidden_),
                                  cfg_.hidden_size,
                                  compute_stream_);
    } else {
      kernels::launch_rmsnorm_offset(static_cast<const __half*>(d_x_),
                                     static_cast<const __half*>(lw.norm_att.data),
                                     static_cast<__half*>(d_x_norm_),
                                     1,
                                     cfg_.hidden_size,
                                     cfg_.rms_norm_eps,
                                     compute_stream_);
      project(lw.linear_qkv, d_x_norm_, d_linear_qkv_mix_);
      project(lw.linear_z, d_x_norm_, d_linear_z_);
      project(lw.linear_a, d_x_norm_, d_linear_a_);
      project(lw.linear_b, d_x_norm_, d_linear_b_);

      CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
      CUDA_CHECK(cudaMemcpy(h_linear_qkv_mix_bits_.data(),
                            d_linear_qkv_mix_,
                            h_linear_qkv_mix_bits_.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_linear_z_bits_.data(),
                            d_linear_z_,
                            h_linear_z_bits_.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_linear_a_bits_.data(),
                            d_linear_a_,
                            h_linear_a_bits_.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_linear_b_bits_.data(),
                            d_linear_b_,
                            h_linear_b_bits_.size() * sizeof(std::uint16_t),
                            cudaMemcpyDeviceToHost));

      for (std::size_t i = 0; i < h_linear_qkv_mix_bits_.size(); ++i) {
        h_linear_qkv_mix_[i] = half_bits_to_float(h_linear_qkv_mix_bits_[i]);
      }
      for (std::size_t i = 0; i < h_linear_z_bits_.size(); ++i) {
        h_linear_z_[i] = half_bits_to_float(h_linear_z_bits_[i]);
      }
      for (std::size_t i = 0; i < h_linear_a_bits_.size(); ++i) {
        h_linear_a_[i] = half_bits_to_float(h_linear_a_bits_[i]);
        h_linear_b_[i] = half_bits_to_float(h_linear_b_bits_[i]);
      }

      const int kernel = cfg_.linear_conv_kernel_dim;
      auto& conv_state = linear_conv_state_[static_cast<std::size_t>(layer)];
      if (kernel > 1) {
        const int state_len = kernel - 1;
        for (int channel = 0; channel < linear_conv_dim_; ++channel) {
          const std::uint16_t* w =
              lw.linear_conv + static_cast<std::size_t>(channel) * static_cast<std::size_t>(kernel);
          float out = 0.0f;
          float* state_row = conv_state.data() +
              static_cast<std::size_t>(channel) * static_cast<std::size_t>(state_len);
          for (int j = 0; j < state_len; ++j) out += bf16_to_float(w[j]) * state_row[j];
          out += bf16_to_float(w[kernel - 1]) * h_linear_qkv_mix_[static_cast<std::size_t>(channel)];
          for (int j = 0; j + 1 < state_len; ++j) state_row[j] = state_row[j + 1];
          state_row[state_len - 1] = h_linear_qkv_mix_[static_cast<std::size_t>(channel)];
          h_linear_qkv_mix_[static_cast<std::size_t>(channel)] = silu(out);
        }
      } else {
        for (float& value : h_linear_qkv_mix_) value = silu(value);
      }

      const float* q_raw = h_linear_qkv_mix_.data();
      const float* k_raw = q_raw + linear_k_dim_;
      const float* v_raw = k_raw + linear_k_dim_;
      std::memcpy(h_linear_v_.data(), v_raw, static_cast<std::size_t>(linear_v_dim_) * sizeof(float));

      for (int k_head = 0; k_head < cfg_.linear_num_key_heads; ++k_head) {
        const float* src_q = q_raw + static_cast<std::size_t>(k_head) * static_cast<std::size_t>(cfg_.linear_key_head_dim);
        const float* src_k = k_raw + static_cast<std::size_t>(k_head) * static_cast<std::size_t>(cfg_.linear_key_head_dim);
        for (int rep = 0; rep < linear_head_repeat_; ++rep) {
          const int v_head = k_head * linear_head_repeat_ + rep;
          float* dst_q = h_linear_q_.data() + static_cast<std::size_t>(v_head) * static_cast<std::size_t>(cfg_.linear_key_head_dim);
          float* dst_k = h_linear_k_.data() + static_cast<std::size_t>(v_head) * static_cast<std::size_t>(cfg_.linear_key_head_dim);
          std::memcpy(dst_q, src_q, static_cast<std::size_t>(cfg_.linear_key_head_dim) * sizeof(float));
          std::memcpy(dst_k, src_k, static_cast<std::size_t>(cfg_.linear_key_head_dim) * sizeof(float));
        }
      }

      auto& recurrent = linear_recurrent_state_[static_cast<std::size_t>(layer)];
      const float q_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.linear_key_head_dim));
      std::vector<float> scratch(static_cast<std::size_t>(cfg_.linear_value_head_dim), 0.0f);
      for (int head = 0; head < cfg_.linear_num_value_heads; ++head) {
        float* q_head = h_linear_q_.data() + static_cast<std::size_t>(head) * static_cast<std::size_t>(cfg_.linear_key_head_dim);
        float* k_head = h_linear_k_.data() + static_cast<std::size_t>(head) * static_cast<std::size_t>(cfg_.linear_key_head_dim);
        float* v_head = h_linear_v_.data() + static_cast<std::size_t>(head) * static_cast<std::size_t>(cfg_.linear_value_head_dim);
        float* z_head = h_linear_z_.data() + static_cast<std::size_t>(head) * static_cast<std::size_t>(cfg_.linear_value_head_dim);
        float* out_head = h_linear_att_.data() + static_cast<std::size_t>(head) * static_cast<std::size_t>(cfg_.linear_value_head_dim);
        float* state = recurrent.data() + static_cast<std::size_t>(head) *
                           static_cast<std::size_t>(cfg_.linear_key_head_dim) *
                           static_cast<std::size_t>(cfg_.linear_value_head_dim);

        l2norm_inplace(q_head, cfg_.linear_key_head_dim);
        l2norm_inplace(k_head, cfg_.linear_key_head_dim);
        for (int d = 0; d < cfg_.linear_key_head_dim; ++d) q_head[d] *= q_scale;

        const float beta = sigmoid(h_linear_b_[static_cast<std::size_t>(head)]);
        const float a = h_linear_a_[static_cast<std::size_t>(head)];
        const float dt_bias = bf16_to_float(lw.linear_dt_bias[head]);
        const float decay = std::exp(-std::exp(lw.linear_A_log[head]) * softplus(a + dt_bias));

        for (int k = 0; k < cfg_.linear_key_head_dim; ++k) {
          float* row = state + static_cast<std::size_t>(k) * static_cast<std::size_t>(cfg_.linear_value_head_dim);
          for (int dv = 0; dv < cfg_.linear_value_head_dim; ++dv) row[dv] *= decay;
        }

        for (int dv = 0; dv < cfg_.linear_value_head_dim; ++dv) {
          float kv_mem = 0.0f;
          for (int k = 0; k < cfg_.linear_key_head_dim; ++k) {
            kv_mem += state[static_cast<std::size_t>(k) * static_cast<std::size_t>(cfg_.linear_value_head_dim) +
                            static_cast<std::size_t>(dv)] * k_head[k];
          }
          scratch[static_cast<std::size_t>(dv)] = (v_head[dv] - kv_mem) * beta;
        }
        for (int k = 0; k < cfg_.linear_key_head_dim; ++k) {
          float* row = state + static_cast<std::size_t>(k) * static_cast<std::size_t>(cfg_.linear_value_head_dim);
          const float kk = k_head[k];
          for (int dv = 0; dv < cfg_.linear_value_head_dim; ++dv) row[dv] += kk * scratch[static_cast<std::size_t>(dv)];
        }
        for (int dv = 0; dv < cfg_.linear_value_head_dim; ++dv) {
          float sum = 0.0f;
          for (int k = 0; k < cfg_.linear_key_head_dim; ++k) {
            sum += state[static_cast<std::size_t>(k) * static_cast<std::size_t>(cfg_.linear_value_head_dim) +
                         static_cast<std::size_t>(dv)] * q_head[k];
          }
          out_head[dv] = sum;
        }

        float ss = 0.0f;
        for (int dv = 0; dv < cfg_.linear_value_head_dim; ++dv) ss += out_head[dv] * out_head[dv];
        const float inv = 1.0f / std::sqrt(ss / static_cast<float>(cfg_.linear_value_head_dim) + cfg_.rms_norm_eps);
        for (int dv = 0; dv < cfg_.linear_value_head_dim; ++dv) {
          out_head[dv] = out_head[dv] * inv * lw.linear_norm[dv] * silu(z_head[dv]);
        }
      }

      for (std::size_t i = 0; i < h_linear_att_.size(); ++i) {
        h_linear_att_bits_[i] = float_to_half_bits(h_linear_att_[i]);
      }
      CUDA_CHECK(cudaMemcpyAsync(d_linear_att_,
                                 h_linear_att_bits_.data(),
                                 h_linear_att_bits_.size() * sizeof(std::uint16_t),
                                 cudaMemcpyHostToDevice,
                                 compute_stream_));
      project(lw.linear_out, d_linear_att_, d_tmp_hidden_);
      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  static_cast<const __half*>(d_tmp_hidden_),
                                  cfg_.hidden_size,
                                  compute_stream_);
    }

    kernels::launch_rmsnorm_offset(static_cast<const __half*>(d_x_),
                                   static_cast<const __half*>(lw.norm_ffn.data),
                                   static_cast<__half*>(d_x_norm_),
                                   1,
                                   cfg_.hidden_size,
                                   cfg_.rms_norm_eps,
                                   compute_stream_);
    project(lw.mlp_gate, d_x_norm_, d_mlp_gate_);
    project(lw.mlp_up, d_x_norm_, d_mlp_up_);
    kernels::launch_silu_mul(static_cast<const __half*>(d_mlp_gate_),
                             static_cast<const __half*>(d_mlp_up_),
                             static_cast<__half*>(d_mlp_inter_),
                             cfg_.intermediate_size,
                             compute_stream_);
    project(lw.mlp_down, d_mlp_inter_, d_tmp_hidden_);
    kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                static_cast<const __half*>(d_tmp_hidden_),
                                cfg_.hidden_size,
                                compute_stream_);
  }

  if (!compute_logits && out_logits == nullptr && out_argmax == nullptr) {
    return;
  }
  kernels::launch_rmsnorm_offset(static_cast<const __half*>(d_x_),
                                 static_cast<const __half*>(d_norm_out_),
                                 static_cast<__half*>(d_x_norm_),
                                 1,
                                 cfg_.hidden_size,
                                 cfg_.rms_norm_eps,
                                 compute_stream_);
  rowmajor_projection_float(lm_head_, d_x_norm_, d_logits_);
  if (out_argmax) {
    kernels::launch_argmax_float(static_cast<const float*>(d_logits_),
                                 cfg_.vocab_size,
                                 d_argmax_,
                                 compute_stream_);
  }
  if (out_argmax) {
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    CUDA_CHECK(cudaMemcpy(out_argmax, d_argmax_, sizeof(int), cudaMemcpyDeviceToHost));
  }
  if (out_logits) {
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    CUDA_CHECK(cudaMemcpy(out_logits->data(),
                          d_logits_,
                          static_cast<std::size_t>(cfg_.vocab_size) * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
}

int Qwen35CudaEngine::sample_next_token(float temperature,
                                        const std::vector<int>& history) {
  const bool greedy_fast_path =
      temperature <= 0.0f &&
      options_.repetition_penalty <= 1.0f &&
      options_.no_repeat_ngram_size <= 1;
  if (greedy_fast_path) {
    kernels::launch_argmax_float(static_cast<const float*>(d_logits_),
                                 cfg_.vocab_size,
                                 d_argmax_,
                                 compute_stream_);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    int next = 0;
    CUDA_CHECK(cudaMemcpy(&next, d_argmax_, sizeof(int), cudaMemcpyDeviceToHost));
    return next;
  }

  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  CUDA_CHECK(cudaMemcpy(h_logits_.data(),
                        d_logits_,
                        static_cast<std::size_t>(cfg_.vocab_size) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  std::vector<float> logits = h_logits_;
  for (int token_id : history) {
    if (token_id >= 0 && token_id < cfg_.vocab_size && options_.repetition_penalty > 1.0f) {
      float& logit = logits[static_cast<std::size_t>(token_id)];
      logit = (logit > 0.0f) ? (logit / options_.repetition_penalty)
                             : (logit * options_.repetition_penalty);
    }
  }
  if (temperature <= 0.0f) {
    return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
  }
  const int k = std::clamp(options_.top_k, 1, cfg_.vocab_size);
  std::vector<int> ids(static_cast<std::size_t>(cfg_.vocab_size));
  std::iota(ids.begin(), ids.end(), 0);
  std::partial_sort(ids.begin(), ids.begin() + k, ids.end(),
                    [&](int left, int right) {
                      return logits[static_cast<std::size_t>(left)] >
                             logits[static_cast<std::size_t>(right)];
                    });
  const float max_logit = logits[static_cast<std::size_t>(ids[0])];
  std::vector<float> probs(static_cast<std::size_t>(k));
  float sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    probs[static_cast<std::size_t>(i)] =
        std::exp((logits[static_cast<std::size_t>(ids[static_cast<std::size_t>(i)])] - max_logit) /
                 temperature);
    sum += probs[static_cast<std::size_t>(i)];
  }
  if (sum <= 0.0f) return ids[0];
  for (float& p : probs) p /= sum;
  static thread_local std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float draw = dist(rng);
  float cumulative = 0.0f;
  for (int i = 0; i < k; ++i) {
    cumulative += probs[static_cast<std::size_t>(i)];
    if (draw <= cumulative) return ids[static_cast<std::size_t>(i)];
  }
  return ids[static_cast<std::size_t>(k - 1)];
}

void Qwen35CudaEngine::initialize(const EngineOptions& options) {
  destroy();
  options_ = options;
  load_config(options.model_path);

  try {
    weights_.open(options.model_path);
    CUDA_CHECK(cudaStreamCreate(&compute_stream_));
    CUBLAS_CHECK(cublasCreate(&cublas_));
    CUBLAS_CHECK(cublasSetStream(cublas_, compute_stream_));

    allocate_runtime_buffers();
    load_weights();
    build_rope_tables();
    reset_state();

    if (options_.verbose) {
      std::cout << "[qwen3_5_cuda] layers=" << cfg_.num_layers
                << " hidden=" << cfg_.hidden_size
                << " heads=" << cfg_.num_attention_heads << "/" << cfg_.num_key_value_heads
                << " linear_heads=" << cfg_.linear_num_key_heads << "/" << cfg_.linear_num_value_heads
                << " quant=" << (options_.int8_streaming
                                      ? (options_.streaming_quant_bits == 4 ? "int4" : "int8")
                                      : "fp16")
                << " max_ctx=" << max_ctx_
                << "\n";
    }
  } catch (...) {
    destroy();
    throw;
  }
}

std::vector<int> Qwen35CudaEngine::generate(const std::vector<int>& prompt_tokens,
                                            int max_new_tokens,
                                            float temperature) {
  return generate_stream(prompt_tokens, max_new_tokens, temperature, [](int) { return true; });
}

std::vector<int> Qwen35CudaEngine::generate_stream(
    const std::vector<int>& prompt_tokens,
    int max_new_tokens,
    float temperature,
    const std::function<bool(int)>& on_token) {
  if (max_new_tokens < 0) {
    LLAMA_ENGINE_THROW("max_new_tokens must be >= 0");
  }
  if (static_cast<int>(prompt_tokens.size()) > max_ctx_) {
    LLAMA_ENGINE_THROW("prompt length exceeds max context");
  }

  reset_state();
  stats_ = {};
  stats_.prompt_tokens = static_cast<int>(prompt_tokens.size());

  std::vector<int> out = prompt_tokens;
  out.reserve(prompt_tokens.size() + static_cast<std::size_t>(max_new_tokens));
  std::vector<int> history = prompt_tokens;
  int current = bos_id_;
  int pos = 0;

  const auto prefill_start = std::chrono::steady_clock::now();
  if (prompt_tokens.empty()) {
    current = bos_id_;
    pos = 0;
    history.push_back(bos_id_);
    forward_token(current, pos, true, nullptr, nullptr);
  } else {
    for (int i = 0; i + 1 < static_cast<int>(prompt_tokens.size()); ++i) {
      forward_token(prompt_tokens[static_cast<std::size_t>(i)], i, false, nullptr, nullptr);
    }
    current = prompt_tokens.back();
    pos = static_cast<int>(prompt_tokens.size()) - 1;
    forward_token(current, pos, true, nullptr, nullptr);
  }
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  const auto prefill_end = std::chrono::steady_clock::now();
  stats_.prefill_ms =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();

  const auto decode_start = std::chrono::steady_clock::now();
  for (int step = 0; step < max_new_tokens; ++step) {
    const int next = sample_next_token(temperature, history);
    history.push_back(next);
    out.push_back(next);
    if (on_token && !on_token(next)) {
      break;
    }
    if (next == cfg_.eos_token_id || next == options_.eos_token_id) {
      break;
    }
    if (step + 1 >= max_new_tokens || pos + 1 >= max_ctx_) {
      break;
    }
    ++pos;
    current = next;
    forward_token(current, pos, true, nullptr, nullptr);
  }
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  const auto decode_end = std::chrono::steady_clock::now();
  stats_.decode_ms =
      std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
  stats_.generated_tokens =
      static_cast<int>((out.size() > prompt_tokens.size())
                           ? (out.size() - prompt_tokens.size())
                           : 0);
  return out;
}

std::vector<std::pair<int, float>> Qwen35CudaEngine::inspect_next_logits(
    const std::vector<int>& prompt_tokens,
    int top_k) {
  if (top_k <= 0) {
    return {};
  }
  if (static_cast<int>(prompt_tokens.size()) > max_ctx_) {
    LLAMA_ENGINE_THROW("prompt length exceeds max context");
  }

  reset_state();
  if (prompt_tokens.empty()) {
    forward_token(bos_id_, 0, true, nullptr, nullptr);
  } else {
    for (int i = 0; i + 1 < static_cast<int>(prompt_tokens.size()); ++i) {
      forward_token(prompt_tokens[static_cast<std::size_t>(i)], i, false, nullptr, nullptr);
    }
    forward_token(prompt_tokens.back(),
                  static_cast<int>(prompt_tokens.size()) - 1,
                  true,
                  nullptr,
                  nullptr);
  }

  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  CUDA_CHECK(cudaMemcpy(h_logits_.data(),
                        d_logits_,
                        static_cast<std::size_t>(cfg_.vocab_size) * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::vector<int> order(static_cast<std::size_t>(cfg_.vocab_size));
  std::iota(order.begin(), order.end(), 0);
  const int k = std::min(top_k, cfg_.vocab_size);
  std::partial_sort(order.begin(), order.begin() + k, order.end(),
                    [&](int a, int b) {
                      return h_logits_[static_cast<std::size_t>(a)] >
                             h_logits_[static_cast<std::size_t>(b)];
                    });

  std::vector<std::pair<int, float>> out;
  out.reserve(static_cast<std::size_t>(k));
  for (int i = 0; i < k; ++i) {
    const int id = order[static_cast<std::size_t>(i)];
    out.emplace_back(id, h_logits_[static_cast<std::size_t>(id)]);
  }
  return out;
}

}  // namespace engine
