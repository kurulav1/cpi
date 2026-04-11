#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "engine/engine_types.hpp"
#include "model/safetensors_loader.hpp"

namespace engine {

class Qwen35CpuEngine {
 public:
  ~Qwen35CpuEngine() = default;

  void initialize(const EngineOptions& options);

  std::vector<int> generate(const std::vector<int>& prompt_tokens,
                            int max_new_tokens,
                            float temperature);

  std::vector<int> generate_stream(const std::vector<int>& prompt_tokens,
                                   int max_new_tokens,
                                   float temperature,
                                   const std::function<bool(int)>& on_token);

  std::vector<std::pair<int, float>> inspect_next_logits(
      const std::vector<int>& prompt_tokens, int top_k);

  const BenchmarkStats& last_benchmark_stats() const { return stats_; }

 private:
  enum class LayerKind {
    LinearAttention,
    FullAttention,
  };

  struct LayerWeights {
    LayerKind kind = LayerKind::LinearAttention;
    const std::uint16_t* norm_att = nullptr;
    const std::uint16_t* norm_ffn = nullptr;
    const std::uint16_t* mlp_gate = nullptr;
    const std::uint16_t* mlp_up = nullptr;
    const std::uint16_t* mlp_down = nullptr;

    const std::uint16_t* full_q = nullptr;
    const std::uint16_t* full_k = nullptr;
    const std::uint16_t* full_v = nullptr;
    const std::uint16_t* full_o = nullptr;
    const std::uint16_t* full_q_norm = nullptr;
    const std::uint16_t* full_k_norm = nullptr;

    const std::uint16_t* linear_qkv = nullptr;
    const std::uint16_t* linear_z = nullptr;
    const std::uint16_t* linear_a = nullptr;
    const std::uint16_t* linear_b = nullptr;
    const std::uint16_t* linear_out = nullptr;
    const std::uint16_t* linear_conv = nullptr;
    const float* linear_norm = nullptr;
    const float* linear_A_log = nullptr;
    const std::uint16_t* linear_dt_bias = nullptr;
  };

  struct ModelConfig {
    int vocab_size = 0;
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int linear_num_key_heads = 0;
    int linear_num_value_heads = 0;
    int linear_key_head_dim = 0;
    int linear_value_head_dim = 0;
    int linear_conv_kernel_dim = 0;
    int max_position_embeddings = 0;
    int eos_token_id = -1;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 10000000.0f;
    float partial_rotary_factor = 1.0f;
    std::vector<LayerKind> layer_kinds;
  };

  void load_config(const std::string& model_dir);
  void allocate_runtime_buffers();
  void load_weight_pointers();

  void gemv_bf16(const std::uint16_t* weights,
                 const float* x,
                 float* y,
                 int out_dim,
                 int in_dim);

  void rmsnorm_offset(const float* x,
                      const std::uint16_t* weight,
                      float* out,
                      int n,
                      float eps);

  void rmsnorm_offset_inplace(float* x,
                              const std::uint16_t* weight,
                              int n,
                              float eps);

  void rmsnorm_direct_gated_inplace(float* x,
                                    const float* weight,
                                    const float* gate,
                                    int n,
                                    float eps);

  void apply_rope_partial(float* q, float* k, int position);
  void run_full_attention_layer(int layer, int position);
  void run_linear_attention_layer(int layer);
  void run_mlp_layer(int layer);
  void forward_token(int token, int position);

  int sample_token(float temperature,
                   int top_k,
                   const std::vector<int>& history,
                   float repetition_penalty);

  model::SafetensorsLoader weights_;
  EngineOptions options_{};
  BenchmarkStats stats_{};
  ModelConfig cfg_{};

  std::vector<LayerWeights> layers_;

  const std::uint16_t* tok_embeddings_ = nullptr;
  const std::uint16_t* norm_out_ = nullptr;
  const std::uint16_t* lm_head_ = nullptr;

  std::vector<float> x_;
  std::vector<float> x_norm_;
  std::vector<float> logits_;

  std::vector<float> full_qkv_;
  std::vector<float> full_q_;
  std::vector<float> full_q_gate_;
  std::vector<float> full_k_;
  std::vector<float> full_v_;
  std::vector<float> full_att_;
  std::vector<float> full_scores_;

  std::vector<float> linear_qkv_mix_;
  std::vector<float> linear_q_;
  std::vector<float> linear_k_;
  std::vector<float> linear_v_;
  std::vector<float> linear_z_;
  std::vector<float> linear_a_;
  std::vector<float> linear_b_;
  std::vector<float> linear_att_;

  std::vector<float> mlp_gate_buf_;
  std::vector<float> mlp_up_buf_;
  std::vector<float> mlp_down_buf_;

  std::vector<float> rope_cos_;
  std::vector<float> rope_sin_;

  std::vector<std::vector<float>> full_k_cache_;
  std::vector<std::vector<float>> full_v_cache_;
  std::vector<std::vector<float>> linear_conv_state_;
  std::vector<std::vector<float>> linear_recurrent_state_;

  int max_ctx_ = 2048;
  int bos_id_ = 0;
  int rotary_dim_ = 0;
  int full_q_dim_ = 0;
  int full_kv_dim_ = 0;
  int linear_k_dim_ = 0;
  int linear_v_dim_ = 0;
  int linear_conv_dim_ = 0;
  int linear_head_repeat_ = 1;
};

}  // namespace engine
