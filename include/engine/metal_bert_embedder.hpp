#pragma once

// Metal BERT-family sentence-embedding encoder; the Apple Silicon counterpart of
// engine::BertEmbedder, which is CUDA (cublas + six __global__ kernels) and therefore left
// embeddings, RAG and folder-search dead on a Mac.
//
// The interface deliberately mirrors BertEmbedder rather than generalising it: same
// initialize/embed/dim/max_tokens/config surface, so cpi_embed picks one at compile time and the
// serving protocol above it does not change. A shared abstract base would buy nothing here
// there are exactly two implementations and they share no state.
//
// The port is mostly assembly. Of the six CUDA kernels only two had no Metal equivalent
// (see src/kernels/metal/85_bert.metal); the rest are kernels the Qwen3.5 vision tower already
// needed; cpi_layernorm, cpi_gemm_f16 (which applies its own bias), cpi_gelu_erf and
// cpi_attention_bidirectional.

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/embedding_config.hpp"
#include "model/safetensors_loader.hpp"
#include "runtime/metal_context.hpp"

namespace engine {

class MetalBertEmbedder {
public:
  // Loads config + weights from `model_dir` onto the GPU. Throws on error.
  void initialize(const std::string& model_dir);

  // Encodes one tokenized sequence ([CLS] ... [SEP] ids) into a pooled, (optionally)
  // L2-normalized embedding of size dim().
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
  bool available() const {
    return ctx_.available();
  }
  const std::string& last_error() const {
    return ctx_.last_error();
  }

private:
  struct LayerWeights {
    runtime::MetalBuffer q_w, q_b, k_w, k_b, v_w, v_b;
    runtime::MetalBuffer o_w, o_b;  // attention output dense
    runtime::MetalBuffer attn_ln_w, attn_ln_b;
    runtime::MetalBuffer inter_w, inter_b;  // [I,H], [I]
    runtime::MetalBuffer out_w, out_b;      // [H,I], [H]
    runtime::MetalBuffer out_ln_w, out_ln_b;
  };

  // Uploads a safetensors tensor as fp16. Requires an F32 source, exactly as the CUDA path does:
  // matching its accepted dtypes matters more than accepting more, because a checkpoint that
  // loads on one backend and not the other is a worse failure than one that loads on neither.
  runtime::MetalBuffer upload_fp16(const std::string& name, std::size_t expected_elems);
  // y[rows,out] = x[rows,in] @ W[out,in]^T + bias, all fp16, via cpi_gemm_f16.
  void linear(const runtime::MetalBuffer& w, const runtime::MetalBuffer& bias,
              const runtime::MetalBuffer& x, runtime::MetalBuffer& y, int out, int in, int rows);
  // out = LayerNorm(src + residual), the shape every BERT sublayer ends in. cpi_layernorm has no
  // residual input, so the add happens first with cpi_add_inplace.
  void layernorm_residual(runtime::MetalBuffer& src, runtime::MetalBuffer& residual,
                          const runtime::MetalBuffer& w, const runtime::MetalBuffer& b,
                          runtime::MetalBuffer& out, int rows, int cols);

  EmbeddingConfig cfg_{};
  model::SafetensorsLoader weights_;
  runtime::MetalContext ctx_;

  runtime::MetalBuffer word_emb_, pos_emb_, type_emb_, emb_ln_w_, emb_ln_b_;
  std::vector<LayerWeights> layers_;

  // Per-call scratch, sized once for max_tokens.
  runtime::MetalBuffer tokens_, x_, tmp_, q_, k_, v_, att_, inter_, pooled_;
  bool ready_ = false;
};

}  // namespace engine
