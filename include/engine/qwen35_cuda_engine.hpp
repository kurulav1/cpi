#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "engine/engine_types.hpp"
#include "model/safetensors_loader.hpp"

namespace engine {

class Qwen35CudaEngine {
 public:
  ~Qwen35CudaEngine();

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

  const BenchmarkStats& last_benchmark_stats() const {
    return stats_;
  }

 private:
  enum class LayerKind {
    LinearAttention,
    FullAttention,
  };

  enum class MatrixKind {
    Fp16,
    Int8,
    Int4,
  };

  struct DeviceMatrix {
    MatrixKind kind = MatrixKind::Fp16;
    int rows = 0;
    int cols = 0;
    void* data = nullptr;
    float* scales = nullptr;
  };

  struct LayerWeights {
    LayerKind kind = LayerKind::LinearAttention;
    DeviceMatrix norm_att{};
    DeviceMatrix norm_ffn{};
    DeviceMatrix mlp_gate{};
    DeviceMatrix mlp_up{};
    DeviceMatrix mlp_down{};

    DeviceMatrix full_q{};
    DeviceMatrix full_k{};
    DeviceMatrix full_v{};
    DeviceMatrix full_o{};
    void* full_q_norm = nullptr;
    void* full_k_norm = nullptr;

    DeviceMatrix linear_qkv{};
    DeviceMatrix linear_z{};
    DeviceMatrix linear_a{};
    DeviceMatrix linear_b{};
    DeviceMatrix linear_out{};
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

  void destroy();
  void load_config(const std::string& model_dir);
  void allocate_runtime_buffers();
  void build_rope_tables();
  void load_weights();
  void reset_state();
  void load_token_embedding_to_device(int token);
  void project(const DeviceMatrix& matrix, const void* x, void* y);
  void rowmajor_projection_float(const DeviceMatrix& matrix, const void* x, void* y);
  void forward_token(int token,
                     int position,
                     bool compute_logits,
                     std::vector<float>* out_logits,
                     int* out_argmax);
  int sample_next_token(float temperature, const std::vector<int>& history);

  model::SafetensorsLoader weights_;
  EngineOptions options_{};
  BenchmarkStats stats_{};
  ModelConfig cfg_{};

  cublasHandle_t cublas_ = nullptr;
  cudaStream_t compute_stream_ = nullptr;

  std::vector<LayerWeights> layers_;

  const std::uint16_t* tok_embeddings_ = nullptr;
  void* d_norm_out_ = nullptr;
  DeviceMatrix lm_head_{};

  void* d_rope_cos_ = nullptr;
  void* d_rope_sin_ = nullptr;

  void* d_x_ = nullptr;
  void* d_x_norm_ = nullptr;
  void* d_q_pair_ = nullptr;
  void* d_q_ = nullptr;
  void* d_q_gate_ = nullptr;
  void* d_k_ = nullptr;
  void* d_v_ = nullptr;
  void* d_att_ = nullptr;
  void* d_tmp_hidden_ = nullptr;
  void* d_logits_ = nullptr;
  int* d_argmax_ = nullptr;

  void* d_mlp_gate_ = nullptr;
  void* d_mlp_up_ = nullptr;
  void* d_mlp_inter_ = nullptr;

  void* d_linear_qkv_mix_ = nullptr;
  void* d_linear_z_ = nullptr;
  void* d_linear_a_ = nullptr;
  void* d_linear_b_ = nullptr;
  void* d_linear_att_ = nullptr;

  void* d_k_cache_ = nullptr;
  void* d_v_cache_ = nullptr;

  std::vector<std::uint16_t> h_token_embedding_fp16_;
  std::vector<float> h_logits_;
  std::vector<std::uint16_t> h_linear_qkv_mix_bits_;
  std::vector<std::uint16_t> h_linear_z_bits_;
  std::vector<std::uint16_t> h_linear_a_bits_;
  std::vector<std::uint16_t> h_linear_b_bits_;
  std::vector<std::uint16_t> h_linear_att_bits_;

  std::vector<float> h_linear_qkv_mix_;
  std::vector<float> h_linear_z_;
  std::vector<float> h_linear_a_;
  std::vector<float> h_linear_b_;
  std::vector<float> h_linear_att_;
  std::vector<float> h_linear_q_;
  std::vector<float> h_linear_k_;
  std::vector<float> h_linear_v_;

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
