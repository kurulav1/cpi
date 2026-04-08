#pragma once

// Llama4CudaEngine: CUDA inference for Llama4 Scout (MoE) from safetensors.
//
// Design summary:
// - Attention and norm weights are converted to fp16 and kept resident on GPU.
// - The LM head is converted to fp16 and kept resident on GPU.
// - Token embeddings stay on host and only the active row is staged each step.
// - Expert routing runs on CPU from a copied RMSNorm activation.
// - The selected expert plus shared expert weights are streamed to GPU per layer.

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "engine/llama_engine.hpp"
#include "model/safetensors_loader.hpp"

namespace engine {

class Llama4CudaEngine {
 public:
  ~Llama4CudaEngine();

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
    return last_benchmark_stats_;
  }

 private:
  struct LayerDeviceWeights {
    void* norm_att = nullptr;
    void* norm_ffn = nullptr;
    void* wq = nullptr;
    void* wk = nullptr;
    void* wv = nullptr;
    void* wo = nullptr;
    void* q_norm = nullptr;
    void* k_norm = nullptr;
  };

  struct LayerHostMoEWeights {
    const std::uint16_t* router = nullptr;   // [n_experts, hidden] BF16
    const std::uint16_t* gate_up = nullptr;  // [hidden, 2*inter] BF16 row-major
    const std::uint16_t* down_exp = nullptr; // [inter, hidden] BF16 row-major
    const std::uint16_t* sh_gate = nullptr;  // [inter_shared, hidden] BF16 row-major
    const std::uint16_t* sh_up = nullptr;    // [inter_shared, hidden] BF16 row-major
    const std::uint16_t* sh_down = nullptr;  // [hidden, inter_shared] BF16 row-major
  };

  // Shared expert FP16 weights resident on GPU.  The shared expert is active
  // every layer every step, so keeping it GPU-resident eliminates ~12 GB of
  // H2D transfers per decode token (half of the total MoE transfer budget).
  struct LayerDeviceSharedWeights {
    void* sh_gate = nullptr;  // [inter_shared, hidden] FP16
    void* sh_up = nullptr;    // [inter_shared, hidden] FP16
    void* sh_down = nullptr;  // [hidden, inter_shared] FP16
  };

  void destroy();
  void load_resident_weights();
  void allocate_runtime_buffers();
  void build_rope_tables();
  void reset_kv_cache();

  void copy_bf16_tensor_to_fp16_device(const std::uint16_t* src,
                                       void* dst,
                                       std::size_t elems);
  void load_token_embedding_to_device(int token);
  int select_expert_cpu(int layer);
  void load_layer_moe_weights_to_device(int layer, int expert_idx);

  void rowmajor_projection_half(const void* w,
                                const void* x,
                                void* y,
                                int out_features,
                                int in_features);
  void rowmajor_projection_float(const void* w,
                                 const void* x,
                                 void* y,
                                 int out_features,
                                 int in_features);
  void transposed_projection_half(const void* w_in_out,
                                  const void* x,
                                  void* y,
                                  int in_features,
                                  int out_features);

  void forward_token(int token,
                     int position,
                     bool compute_logits,
                     std::vector<float>* out_logits,
                     int* out_argmax);
  void forward_token_logits(int token,
                            int position,
                            std::vector<float>* out_logits,
                            int* out_argmax);
  int sample_next_token(float temperature,
                        const std::vector<int>& history);

  // Llama4 Scout architecture.
  int vocab_size_ = 202048;
  int hidden_ = 5120;
  int n_heads_ = 40;
  int n_kv_heads_ = 8;
  int head_dim_ = 128;
  int n_layers_ = 48;
  int n_experts_ = 16;
  int inter_full_ = 16384;  // gate+up concatenated
  int inter_expert_ = 8192;
  int inter_shared_ = 8192;
  float rope_theta_ = 500000.0f;
  float rope_scale_ = 16.0f;
  float rope_low_freq_factor_ = 1.0f;
  float rope_high_freq_factor_ = 4.0f;
  int rope_orig_max_pos_ = 8192;
  int max_ctx_ = 2048;
  int kv_dim_ = 0;
  int bos_id_ = 128000;
  int eos_id_ = 128009;
  bool use_qk_norm_ = true;

  EngineOptions options_{};
  model::SafetensorsLoader weights_;
  BenchmarkStats last_benchmark_stats_{};

  cublasHandle_t cublas_ = nullptr;
  cudaStream_t compute_stream_ = nullptr;

  const std::uint16_t* h_tok_embeddings_bf16_ = nullptr;
  void* d_norm_out_ = nullptr;
  void* d_lm_head_ = nullptr;

  std::vector<LayerDeviceWeights> layer_device_;
  std::vector<LayerHostMoEWeights> layer_host_moe_;
  std::vector<LayerDeviceSharedWeights> layer_device_shared_;  // Shared expert FP16, GPU-resident.
  int n_shared_gpu_ = 0;  // How many layers have shared expert weights cached on GPU.

  void* d_q_norm_unit_ = nullptr;
  void* d_k_norm_unit_ = nullptr;
  void* d_rope_cos_ = nullptr;
  void* d_rope_sin_ = nullptr;

  void* d_x_ = nullptr;
  void* d_x_norm_ = nullptr;
  void* d_q_ = nullptr;
  void* d_k_ = nullptr;
  void* d_v_ = nullptr;
  void* d_att_ = nullptr;
  void* d_tmp_hidden_ = nullptr;
  void* d_ff13_ = nullptr;
  void* d_ff_inter_ = nullptr;
  void* d_shared_gate_out_ = nullptr;
  void* d_shared_up_out_ = nullptr;
  void* d_logits_ = nullptr;
  int* d_argmax_ = nullptr;

  void* d_k_cache_ = nullptr;
  void* d_v_cache_ = nullptr;

  // Streamed MoE weight buffers.
  void* d_expert_gate_up_w_ = nullptr;  // [hidden, 2*inter] converted to fp16
  void* d_expert_down_w_ = nullptr;     // [inter, hidden] converted to fp16
  void* d_streamed_rowmajor_w_ = nullptr; // reused for shared gate/up/down
  std::uint16_t* d_bf16_stage_ = nullptr; // conversion staging buffer
  std::size_t bf16_stage_elems_ = 0;

  // Host-side scratch/state.
  std::vector<std::uint16_t> h_token_embedding_fp16_;
  std::vector<std::uint16_t> h_x_norm_fp16_;
  std::vector<float> h_x_norm_f32_;
  std::vector<float> h_logits_;
};

}  // namespace engine
