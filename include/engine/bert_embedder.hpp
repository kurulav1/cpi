#pragma once

// CUDA BERT-family sentence-embedding encoder (the embedding "core"). Loads a
// HuggingFace BERT encoder (bge / gte / e5 / ...) from safetensors, runs a
// bidirectional forward pass, pools (CLS or mean), and L2-normalizes, producing
// a retrieval embedding. Architecture and behaviour are driven entirely by
// EmbeddingConfig, so new models are served by config, not code.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

#include "engine/embedding_config.hpp"
#include "model/safetensors_loader.hpp"

namespace engine {

class BertEmbedder {
public:
  ~BertEmbedder();

  // Loads config + weights from `model_dir` onto the GPU. Throws on error.
  void initialize(const std::string& model_dir);

  // Encodes one tokenized sequence ([CLS] ... [SEP] ids) into a pooled,
  // (optionally) L2-normalized embedding of size dim().
  std::vector<float> embed(const std::vector<int>& token_ids);

  int dim() const {
    return cfg_.dimension;
  }
  int max_tokens() const {
    return cfg_.max_tokens;
  }
  const EmbeddingConfig& config() const {
    return cfg_;
  }

private:
  struct LayerWeights {
    void* q_w = nullptr;
    void* q_b = nullptr;  // [H,H], [H]
    void* k_w = nullptr;
    void* k_b = nullptr;
    void* v_w = nullptr;
    void* v_b = nullptr;
    void* o_w = nullptr;
    void* o_b = nullptr;  // attention output dense
    void* attn_ln_w = nullptr;
    void* attn_ln_b = nullptr;
    void* inter_w = nullptr;
    void* inter_b = nullptr;  // [I,H], [I]
    void* out_w = nullptr;
    void* out_b = nullptr;  // [H,I], [H]
    void* out_ln_w = nullptr;
    void* out_ln_b = nullptr;
  };

  void* upload_fp16(const std::string& name, std::size_t expected_elems);
  // y[rows,out] = x[rows,in] @ W[out,in]^T  (+ bias if not null), all fp16.
  void linear(const void* w, const void* bias, const void* x, void* y, int out, int in, int rows);
  void destroy();

  EmbeddingConfig cfg_{};
  model::SafetensorsLoader weights_;
  cublasHandle_t cublas_ = nullptr;
  cudaStream_t stream_ = nullptr;

  // Embedding tables (fp16, device).
  void* word_emb_ = nullptr;  // [vocab, H]
  void* pos_emb_ = nullptr;   // [max_pos, H]
  void* type_emb_ = nullptr;  // [type_vocab, H]
  void* emb_ln_w_ = nullptr;
  void* emb_ln_b_ = nullptr;
  std::vector<LayerWeights> layers_;

  // Per-call scratch (sized for max_tokens).
  int* d_tokens_ = nullptr;
  void* d_x_ = nullptr;    // [L,H] hidden state
  void* d_tmp_ = nullptr;  // [L,H] sublayer output
  void* d_q_ = nullptr;
  void* d_k_ = nullptr;
  void* d_v_ = nullptr;        // [L,H]
  void* d_att_ = nullptr;      // [L,H] attention output
  void* d_inter_ = nullptr;    // [L,I]
  float* d_pooled_ = nullptr;  // [H] pooled embedding (fp32)
  std::vector<float> h_pooled_;
};

}  // namespace engine
