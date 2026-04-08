#pragma once

// Llama4CpuEngine: CPU inference for Llama4 Scout (MoE) from safetensors.
// Weights are disk-streamed via OS mmap page eviction. Fits a 240GB model
// in 32GB RAM by only keeping recently-accessed pages resident.
//
// Architecture: 48 layers, hidden=5120, 40/8 GQA heads, head_dim=128,
//   16 MoE experts per layer (top-1 routing), shared expert per layer,
//   BF16 weights, vocab=202048, Llama3 RoPE scaling (theta=500000, scale=16).

#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "engine/llama_engine.hpp"   // EngineOptions, BenchmarkStats
#include "model/safetensors_loader.hpp"

namespace engine {

class Llama4CpuEngine {
 public:
  ~Llama4CpuEngine() = default;

  // options.model_path = path to the directory with .safetensors files
  void initialize(const EngineOptions& options);

  std::vector<int> generate(const std::vector<int>& prompt_tokens,
                            int max_new_tokens, float temperature);

  std::vector<int> generate_stream(const std::vector<int>& prompt_tokens,
                                   int max_new_tokens, float temperature,
                                   const std::function<bool(int)>& on_token);

  std::vector<std::pair<int, float>> inspect_next_logits(
      const std::vector<int>& prompt_tokens, int top_k);

  const BenchmarkStats& last_benchmark_stats() const { return stats_; }

 private:
  // ---- Architecture config (fixed for Llama4 Scout 17B-16E) ----
  int vocab_size_     = 202048;
  int hidden_        = 5120;
  int n_heads_       = 40;
  int n_kv_heads_    = 8;
  int head_dim_      = 128;
  int n_layers_      = 48;
  int n_experts_     = 16;
  int inter_mlp_     = 16384;   // per-expert gate/up output dim (2 * 8192)
  int inter_shared_  = 8192;    // shared expert intermediate dim
  float rope_theta_  = 500000.0f;
  float rope_scale_  = 16.0f;   // Llama3 RoPE scale factor
  float rope_low_freq_factor_   = 1.0f;
  float rope_high_freq_factor_  = 4.0f;
  int   rope_orig_max_pos_      = 8192;
  int max_ctx_       = 2048;
  int kv_dim_        = 0;       // n_kv_heads * head_dim
  int bos_id_        = 128000;  // <|begin_of_text|>
  int eos_id_        = 128009;  // <|eot_id|>
  bool use_qk_norm_  = true;    // Llama4 applies per-head Q/K RMSNorm.

  // ---- Weight loader ----
  model::SafetensorsLoader weights_;

  // ---- Per-layer weight pointers (BF16 into mmap) ----
  struct LayerWeights {
    const uint16_t* norm_att  = nullptr;  // [hidden]
    const uint16_t* norm_ffn  = nullptr;  // [hidden]
    const uint16_t* wq        = nullptr;  // [n_heads*head_dim, hidden]
    const uint16_t* wk        = nullptr;  // [n_kv_heads*head_dim, hidden]
    const uint16_t* wv        = nullptr;  // [n_kv_heads*head_dim, hidden]
    const uint16_t* wo        = nullptr;  // [hidden, hidden]
    // Router: [n_experts, hidden]
    const uint16_t* router    = nullptr;
    // MoE experts (transposed layout [in, out]):
    //   gate_up: [n_experts, hidden, 2*inter_mlp_]
    //   down:    [n_experts, inter_mlp_, hidden]
    const uint16_t* gate_up   = nullptr;
    const uint16_t* down_exp  = nullptr;
    // Shared expert (standard [out, in]):
    const uint16_t* sh_gate   = nullptr;  // [inter_shared_, hidden]
    const uint16_t* sh_up     = nullptr;  // [inter_shared_, hidden]
    const uint16_t* sh_down   = nullptr;  // [hidden, inter_shared_]
    // QK norm (optional)
    const uint16_t* q_norm    = nullptr;  // [head_dim] or nullptr
    const uint16_t* k_norm    = nullptr;
  };
  std::vector<LayerWeights> layers_;

  // ---- Static weights ----
  const uint16_t* tok_embeddings_ = nullptr;  // [vocab, hidden]
  const uint16_t* norm_out_       = nullptr;  // [hidden]
  const uint16_t* lm_head_        = nullptr;  // [vocab, hidden]

  // ---- Scratch buffers ----
  std::vector<float> x_, x_norm_;
  std::vector<float> q_, k_, v_, att_;
  std::vector<float> router_logits_;   // [n_experts]
  std::vector<float> gate_buf_, up_buf_, down_buf_;  // expert FFN temps
  std::vector<float> shared_buf_, shared_gate_, shared_up_;
  std::vector<float> logits_, scores_;

  // ---- KV cache [n_layers, max_ctx, kv_dim] ----
  std::vector<float> k_cache_, v_cache_;

  // ---- RoPE tables [max_ctx, head_dim/2] ----
  std::vector<float> rope_cos_, rope_sin_;

  BenchmarkStats stats_{};
  EngineOptions options_;

  // ---- Core operations ----
  void rmsnorm(const float* x, const uint16_t* w, float* out, int n);
  // y = W * x  where W is [M, N] BF16 standard row-major [out, in]
  void gemv_bf16(const uint16_t* W, const float* x, float* y, int M, int N);
  // y = x @ W  where W is [in_dim, out_dim] BF16 transposed layout
  void gemv_bf16_T(const uint16_t* W, const float* x, float* y, int in_dim, int out_dim);
  void rope(float* q, float* k, int pos);
  void attention(int pos, int layer);
  void moe_ffn(int layer);
  void forward_token(int token, int pos);
  int sample_token(float temperature, int top_k,
                   const std::vector<int>& history, float rep_penalty);
};

}  // namespace engine
