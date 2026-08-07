#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "engine/engine_types.hpp"
#include "engine/generation_constraints.hpp"
#include "model/weight_loader.hpp"

namespace engine {

class CpuLlamaEngine {
public:
  ~CpuLlamaEngine() = default;

  void initialize(const EngineOptions& options);

  std::vector<int> generate(const std::vector<int>& prompt_tokens, int max_new_tokens,
                            float temperature);

  std::vector<int> generate_stream(const std::vector<int>& prompt_tokens, int max_new_tokens,
                                   float temperature, const std::function<bool(int)>& on_token,
                                   const GenerationConstraints* constraints = nullptr);

  std::vector<std::pair<int, float>> inspect_next_logits(const std::vector<int>& prompt_tokens,
                                                         int top_k);

  const BenchmarkStats& last_benchmark_stats() const {
    return last_benchmark_stats_;
  }

private:
  struct LayerWeights {
    const uint16_t* norm_att = nullptr;
    const uint16_t* norm_ffn = nullptr;
    const uint16_t* norm_att_bias = nullptr;
    const uint16_t* norm_ffn_bias = nullptr;
    const uint16_t* wq = nullptr;
    const uint16_t* wk = nullptr;
    const uint16_t* wv = nullptr;
    const uint16_t* wo = nullptr;
    const uint16_t* bo = nullptr;
    const uint16_t* bqkv = nullptr;  // fused q||k||v projection bias (Qwen2 et al.)
    // Per-head RMSNorm on Q and K, applied after projection and before RoPE (Qwen3).
    // One [head_dim] weight shared across heads. Without these the CPU engine
    // silently produces a wrong model for every Qwen3 checkpoint; it does not
    // crash, it just computes something else.
    const uint16_t* q_norm = nullptr;
    const uint16_t* k_norm = nullptr;

    const uint16_t* w1_fp16 = nullptr;
    const uint16_t* w2_fp16 = nullptr;
    const uint16_t* w3_fp16 = nullptr;
    const float* w1_fp32 = nullptr;
    const float* w2_fp32 = nullptr;
    const float* w3_fp32 = nullptr;

    const uint16_t* router_fp16 = nullptr;
    const float* router_fp32 = nullptr;
    std::vector<const uint16_t*> expert_w1_fp16;
    std::vector<const uint16_t*> expert_w2_fp16;
    std::vector<const uint16_t*> expert_w3_fp16;
    std::vector<const float*> expert_w1_fp32;
    std::vector<const float*> expert_w2_fp32;
    std::vector<const float*> expert_w3_fp32;
  };

  void normalize(const float* x, const uint16_t* w, const uint16_t* b, float* out, int n);
  void qk_norm_heads(float* x, const uint16_t* w, int heads, int head_dim);

  void gemv_fp16(const uint16_t* W, const float* x, float* y, int M, int N);
  void gemv_fp32(const float* W, const float* x, float* y, int M, int N);
  // Batched projection for prefill: Y[B][M] = X[B][N] * W^T, W row-major fp16.
  // Cache-blocked so each weight row block is read from DRAM once per call
  // (not once per token), which is the entire difference between prefill at
  // decode speed and prefill at compute speed on CPU.
  void gemm_fp16(const uint16_t* W, const float* X, float* Y, int M, int N, int B);

  void rope(float* q, float* k, int pos, int n_heads, int n_kv_heads, int head_dim);
  void attention(int pos, int layer);

  void mlp(int layer);
  void mlp_moe(int layer);
  void forward_token(int token, int pos);
  // Final RMSNorm + LM head from the current residual x_ into logits_
  // (the tail of forward_token, shared with the chunked prefill).
  void compute_logits_from_x();
  // Batched prefill: run `count` prompt tokens starting at absolute position
  // base_pos through all layers (no LM head), leaving the last token's
  // residual in x_. Full-attention / sliding-window dense fp16 models only;
  // callers gate on batched_prefill_supported().
  void forward_prompt_chunk(const int* tokens, int base_pos, int count);
  bool batched_prefill_supported() const;
  // Prefill dispatch shared by generate_stream and inspect_next_logits:
  // chunked when supported (CPI_CPU_PREFILL_TOKENWISE=1 forces the legacy
  // token-by-token path), token-wise otherwise. Returns the next position and
  // leaves logits_ primed from the last prompt token.
  int prefill_prompt(const std::vector<int>& prompt_tokens);

  int sample_token(float temperature, int top_k, const std::vector<int>& history,
                   float rep_penalty);

  model::WeightLoader weights_;
  model::LlamaConfig cfg_;
  EngineOptions options_;
  BenchmarkStats last_benchmark_stats_{};
  // Active grammar for the in-flight generate_stream call (see generation_constraints.hpp).
  grammar::GrammarSampler* active_grammar_ = nullptr;

  std::vector<LayerWeights> layers_;

  const uint16_t* tok_embeddings_ = nullptr;
  const uint16_t* norm_out_ = nullptr;
  const uint16_t* norm_out_bias_ = nullptr;
  const uint16_t* lm_head_ = nullptr;
  const uint16_t* lm_head_bias_ = nullptr;

  std::vector<float> x_;
  std::vector<float> x_norm_;
  std::vector<float> q_;
  std::vector<float> k_;
  std::vector<float> v_;
  std::vector<float> att_;
  std::vector<float> ff1_;
  std::vector<float> ff3_;
  std::vector<float> ff2_;
  std::vector<float> logits_;
  std::vector<float> scores_;

  std::vector<std::vector<float>> dequant_mlp_;
  std::vector<std::vector<float>> dequant_moe_;

  std::vector<float> k_cache_;
  std::vector<float> v_cache_;

  // Batched-prefill scratch, [chunk][dim] row-major (sized on first use).
  std::vector<float> xb_;
  std::vector<float> xnormb_;
  std::vector<float> qb_;
  std::vector<float> kb_;
  std::vector<float> vb_;
  std::vector<float> attb_;
  std::vector<float> ff1b_;
  std::vector<float> ff3b_;
  std::vector<float> ff2b_;

  std::vector<float> rope_cos_;
  std::vector<float> rope_sin_;
  std::vector<float> moe_router_logits_;
  std::vector<float> moe_accum_;

  int q_dim_ = 0;
  int head_dim_ = 0;
  int kv_dim_ = 0;
};

}  // namespace engine
