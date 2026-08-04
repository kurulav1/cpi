#pragma once

// Backend-neutral model config for the op-plan engines.
//
// This is the config that PlanCudaEngine and PlanMetalEngine both need in order to BUILD a plan:
// geometry, per-layer attention kinds, RoPE parameters, KV sharing. None of it is CUDA-specific
// it is the answer to "what shape is this model", which is the same question on every backend.
//
// It used to be a struct nested inside PlanCudaEngine, in a header that includes <cuda_runtime.h>.
// That is the only reason Gemma 4 was CUDA-only: not a missing kernel (every OpKind it needs is
// implemented on Metal too), but a config parser that could not be #included without the CUDA
// toolkit. Lifting it here is what lets the plan builder move to op_plan_builder.cpp beside
// build_llama_plan and build_qwen35_plan.
//
// PlanCudaEngine aliases its private Family/Config to these, so the field names below must stay
// exactly as they were; ~1800 lines of `cfg_.<field>` depend on them. Renaming a field here is
// not a cleanup, it is a silent behaviour change in whichever engine you forget to update.
//
// This file must compile with no CUDA and no Metal toolkit present.

#include <string>
#include <vector>

#include "engine/op_plan_builder.hpp"
#include "model/llama_config.hpp"

namespace engine {

// Which model recipe builds the plan. The EXECUTOR, sampler, quantizer, decode graph and KV/state
// machinery are shared; only the config parse, weight load and plan recipe differ; so a "new
// model" is a recipe, not an engine.
enum class PlanFamily { Gemma4, Qwen35, DeepSeekV2 };

struct PlanModelConfig {
  PlanFamily family = PlanFamily::Gemma4;
  int num_layers = 0, hidden = 0, num_heads = 0, num_kv_heads = 1;
  int head_dim = 256, global_head_dim = 512, intermediate = 0, vocab = 0;
  // Actual per-layer-type head_dim from the weights (E2B: 256/512; 12B: 256/512).
  int head_dim_sliding = 256, head_dim_full = 512;
  // num_kv_heads can differ per layer type (12B: sliding GQA-8, full MQA-1).
  int num_kv_heads_sliding = 1, num_kv_heads_full = 1;
  bool attention_k_eq_v = false;  // 12B full layers: V shares k_proj (no v_proj)
  int hidden_size_per_layer_input = 256, num_kv_shared_layers = 0, first_shared_layer = 0;
  int sliding_window = 0, bos_token_id = 2, eos_token_id = 1;
  float rms_eps = 1e-6f, final_logit_softcapping = 0.0f;
  float rope_theta_full = 1e6f, rope_theta_sliding = 1e4f, partial_rotary_full = 0.25f;
  bool use_double_wide_mlp = false, tie_word_embeddings = true;
  bool enable_moe_block = false;  // 26B-A4B: dense-MLP + top-k experts (not yet run)
  int num_experts = 0, top_k_experts = 0, moe_intermediate_size = 0;
  std::vector<int> layer_full;  // 1 if full_attention, 0 if sliding/linear
  std::vector<int> kv_source;   // which layer's K/V each layer uses

  // Per-Layer Embeddings vocabulary (E2B). Was read as a local in the CUDA parser and used only
  // for the shape registry; it is config, so it lives with the rest of it. 0 => fall back to vocab.
  int vocab_size_per_layer_input = 0;

  // ── Qwen3.5 (hybrid full-attention + gated delta-net) ──
  // layer_full doubles as the layer-kind mask here: 1 = full_attention,
  // 0 = linear_attention (delta-net), exactly as for Gemma's sliding layers.
  int linear_num_key_heads = 0, linear_num_value_heads = 0;
  int linear_key_head_dim = 0, linear_value_head_dim = 0;
  int linear_conv_kernel_dim = 0;
  int max_position_embeddings = 0;
  float rope_theta = 1e7f;
  float partial_rotary_factor = 1.0f;

  // ── DeepSeek-V2 (MLA latent attention + fine-grained MoE) ──
  // MLA geometry: query is a direct proj when q_lora_rank == 0 (V2-Lite) else down/up through it.
  int q_lora_rank = 0;             // query LoRA rank (0 = direct q_proj)
  int kv_lora_rank = 0;            // cached KV latent width
  int qk_nope_head_dim = 0;        // per-head non-rope q/k dim
  int qk_rope_head_dim = 0;        // per-head decoupled-rope q/k dim (k side shared across heads)
  int v_head_dim = 0;              // per-head value dim (may differ from qk head dim)
  int n_shared_experts = 0;        // always-on shared experts (added to the routed output)
  int first_k_dense_replace = 0;   // first N layers are a dense MLP; layers >= N are MoE
  float routed_scaling_factor = 1.0f;
  // YARN rope parameters, used to build inv_freq + the attention (mscale) scaling at load.
  float yarn_factor = 1.0f, yarn_mscale = 1.0f, yarn_beta_fast = 32.0f, yarn_beta_slow = 1.0f;
  int yarn_original_max_pos = 0;

  // True when this layer reuses another layer's K/V rather than projecting its own.
  [[nodiscard]] bool layer_is_kv_shared(int layer) const {
    return first_shared_layer > 0 && layer >= first_shared_layer;
  }
};

// Parses Gemma 4's geometry out of a HuggingFace `config.json`.
//
// Takes the whole file text and reaches into `text_config` itself, because that nesting is part of
// Gemma 4's format rather than the caller's business. Throws if `text_config` is absent or if the
// required fields (vocab / hidden / layers) are missing; a config that parses to zeros would
// otherwise produce a model that allocates nothing and generates garbage.
//
// Also derives what the checkpoint states only implicitly: `layer_full` from `layer_types`, and
// `kv_source` from `num_kv_shared_layers` (the last N layers reuse the cache of the last
// non-shared layer OF the same TYPE; sliding and full are tracked separately, and getting it
// wrong still runs and still produces fluent-looking garbage).
PlanModelConfig parse_gemma4_text_config(const std::string& config_json);

// Same geometry, expressed as a model::LlamaConfig.
//
// The Metal engine builds its slots, geometry and plan from LlamaConfig, so this is how Gemma 4
// reaches it. It is a pure FIELD MAPPING and deliberately does no parsing: every JSON key name
// lives in parse_gemma4_text_config and nowhere else, so the two cannot drift. Reading the config
// on Metal is therefore parse -> map, not a second parser.
model::LlamaConfig gemma4_to_llama_config(const PlanModelConfig& g);

// True when a HuggingFace config.json describes a Gemma 4 model. Checks `model_type` in
// text_config, falling back to the root; the same place PlanCudaEngine::open looks, and the same
// place probe_model's directory sniff looks.
bool config_json_is_gemma4(const std::string& config_json);

// Gemma 4's `vision_config`, parsed into the shared vision-plan geometry. Returns
// present == false when the checkpoint has no vision_config block (text-only export).
opplan::Gemma4VisionGeometry parse_gemma4_vision_config(const std::string& config_json);

}  // namespace engine
