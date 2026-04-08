#pragma once

// Model configuration and manifest types for LLaMA-family models.
// LlamaConfig holds the architectural hyper-parameters that describe a model
// variant (e.g., 7B vs 13B).  ModelManifest pairs a config with the path to
// the serialised weight file so that a single object carries everything needed
// to open a model.

#include <cstddef>
#include <cstdint>
#include <string>

namespace model {

// Identifies the model family, used to select architecture-specific behaviour
// such as default RoPE frequency, activation function, and special token IDs.
enum class ModelFamily : std::int32_t {
  Unknown  = 0,
  LLaMA2   = 1,  // Meta LLaMA 2 (rope_theta=10000, vocab=32000)
  LLaMA3   = 2,  // Meta LLaMA 3/3.1/3.2 (rope_theta=500000, vocab=128256)
  Mistral  = 3,  // Mistral 7B / Mixtral (rope_theta=10000, sliding window)
  Phi3     = 4,  // Microsoft Phi-3 (partial rotary, rope_theta=10000)
  Qwen2    = 5,  // Alibaba Qwen2 (rope_theta=1000000, QKV biases)
  Mixtral  = 6,  // Mixtral sparse-MoE (top-k routed experts, rope_theta=10000).
};

// Returns the default RoPE base frequency for a given model family.
inline float default_rope_theta(ModelFamily family) {
  switch (family) {
    case ModelFamily::LLaMA3: return 500000.0f;
    case ModelFamily::Qwen2:  return 1000000.0f;
    default:                  return 10000.0f;
  }
}

// Architectural hyper-parameters for a LLaMA-family transformer model.
// All fields correspond directly to the matching entries in the model's
// config.json.  Default values match the LLaMA-7B configuration.
struct LlamaConfig {
  std::int32_t vocab_size = 32000;        // Number of token embeddings.
  std::int32_t hidden_size = 4096;        // Dimension of the residual stream (d_model).
  std::int32_t intermediate_size = 11008; // Inner dimension of the SwiGLU MLP block.
  std::int32_t num_layers = 32;           // Total number of transformer layers.
  std::int32_t num_heads = 32;            // Number of query attention heads.
  std::int32_t num_kv_heads = 32;         // Number of key/value heads (< num_heads for GQA).
  std::int32_t max_seq_len = 4096;        // Maximum supported sequence length.
  std::int32_t tensor_parallel = 1;       // Number of tensor-parallel shards (currently unused).
  std::string dtype = "fp16";             // Storage data type of the weights ("fp16" or "fp32").

  // Extended fields stored in HeaderV3 (.ll2c format version 3+).
  // Files with an older header get conservative defaults.
  ModelFamily  model_family = ModelFamily::Unknown; // Architecture variant identifier.
  float        rope_theta = 0.0f;         // RoPE base frequency (0 = use family default or CLI override).
  float        norm_eps = 1e-5f;          // Epsilon added to RMSNorm denominator.
  std::int32_t sliding_window = 0;        // Attention window size (0 = full context, Mistral uses 4096).
  bool         has_qkv_bias = false;      // QKV projections have additive bias vectors (Qwen2).
  bool         use_layernorm = false;     // Use true LayerNorm (mean+variance) instead of RMSNorm.
  bool         tie_word_embeddings = false; // lm_head shares weights with tok_embeddings (some Phi variants).
  std::int32_t num_local_experts = 0;     // Number of MoE experts per layer (0 = dense FFN).
  std::int32_t num_experts_per_tok = 0;   // Router top-k experts selected per token (Mixtral uses 2).
  std::int32_t expert_intermediate_size = 0; // Expert FFN hidden size (0 = use intermediate_size).

  // Returns the effective RoPE theta: uses the stored value if set, otherwise
  // falls back to the family default.
  float effective_rope_theta() const {
    return rope_theta > 0.0f ? rope_theta : default_rope_theta(model_family);
  }

  [[nodiscard]] bool is_moe() const {
    return num_local_experts > 0;
  }

  [[nodiscard]] std::int32_t effective_expert_intermediate_size() const {
    return expert_intermediate_size > 0 ? expert_intermediate_size : intermediate_size;
  }
};

// Combines a model's architectural config with the filesystem path to its
// compiled weight file, forming a self-contained description of a model
// artefact that can be passed to WeightLoader::open().
struct ModelManifest {
  LlamaConfig config;       // Architectural hyper-parameters for this model.
  std::string weights_path; // Absolute or relative path to the binary weight file.
};

}  // namespace model
