#pragma once

// LlamaConfig <-> JSON, for the `.cpi` (safetensors) container's `__metadata__` block.
//
// why this file HAS AN X-MACRO INSTEAD OF two HAND-WRITTEN FUNCTIONS
//
// A config serializer is exactly the shape of the bug that shipped container v7 with all-zero
// vision geometry: `pack_ll2c.py::load_config` is a WHITELIST, so a field present upstream and
// absent from the list was dropped silently, and the container reported success. Two hand-written
// functions here would reintroduce that; add a field to the writer, forget the reader, and the
// model loads with a zeroed field and generates fluent nonsense.
//
// So both directions are generated from one list. A field added to CPI_CONFIG_FIELDS appears in
// the writer and the reader and the round-trip test simultaneously; it is not possible to add it
// to one and not the others. A field left OUT of the list is missing from all three, which is
// what the round-trip test (config_json_test) exists to catch: it fills a config with distinct
// values, round-trips it, and compares field by field using this same list.
//
// The three non-scalar members are handled explicitly below the macro, and the round-trip test
// covers them too.

#include <cstdint>
#include <string>
#include <vector>

#include "model/json_mini.hpp"
#include "model/llama_config.hpp"

namespace model {

// X(member, json_key, kind); kind is one of I (int32), F (float), B (bool).
#define CPI_CONFIG_FIELDS(X)                                                  \
  X(vocab_size, "vocab_size", I)                                              \
  X(hidden_size, "hidden_size", I)                                            \
  X(intermediate_size, "intermediate_size", I)                                \
  X(num_layers, "num_layers", I)                                              \
  X(num_heads, "num_heads", I)                                                \
  X(num_kv_heads, "num_kv_heads", I)                                          \
  X(max_seq_len, "max_seq_len", I)                                            \
  X(tensor_parallel, "tensor_parallel", I)                                    \
  X(rope_theta, "rope_theta", F)                                              \
  X(norm_eps, "norm_eps", F)                                                  \
  X(sliding_window, "sliding_window", I)                                      \
  X(has_qkv_bias, "has_qkv_bias", B)                                          \
  X(has_qk_norm, "has_qk_norm", B)                                            \
  X(mlp_gelu, "mlp_gelu", B)                                                  \
  X(scale_embeddings, "scale_embeddings", B)                                  \
  X(attn_output_gate, "attn_output_gate", B)                                  \
  X(use_layernorm, "use_layernorm", B)                                        \
  X(tie_word_embeddings, "tie_word_embeddings", B)                            \
  X(num_local_experts, "num_local_experts", I)                                \
  X(num_experts_per_tok, "num_experts_per_tok", I)                            \
  X(expert_intermediate_size, "expert_intermediate_size", I)                  \
  X(partial_rotary_factor, "partial_rotary_factor", F)                        \
  X(linear_num_key_heads, "linear_num_key_heads", I)                          \
  X(linear_num_value_heads, "linear_num_value_heads", I)                      \
  X(linear_key_head_dim, "linear_key_head_dim", I)                            \
  X(linear_value_head_dim, "linear_value_head_dim", I)                        \
  X(linear_conv_kernel_dim, "linear_conv_kernel_dim", I)                      \
  X(vision_depth, "vision_depth", I)                                          \
  X(vision_hidden_size, "vision_hidden_size", I)                              \
  X(vision_num_heads, "vision_num_heads", I)                                  \
  X(vision_intermediate_size, "vision_intermediate_size", I)                  \
  X(vision_patch_size, "vision_patch_size", I)                                \
  X(vision_temporal_patch_size, "vision_temporal_patch_size", I)              \
  X(vision_in_channels, "vision_in_channels", I)                              \
  X(vision_spatial_merge_size, "vision_spatial_merge_size", I)                \
  X(vision_num_position_embeddings, "vision_num_position_embeddings", I)      \
  X(vision_out_hidden_size, "vision_out_hidden_size", I)                       \
  X(hidden_size_per_layer_input, "hidden_size_per_layer_input", I)             \
  X(vocab_size_per_layer_input, "vocab_size_per_layer_input", I)               \
  X(num_kv_shared_layers, "num_kv_shared_layers", I)                           \
  X(first_shared_layer, "first_shared_layer", I)                               \
  X(head_dim_sliding, "head_dim_sliding", I)                                   \
  X(head_dim_full, "head_dim_full", I)                                         \
  X(num_kv_heads_sliding, "num_kv_heads_sliding", I)                           \
  X(num_kv_heads_full, "num_kv_heads_full", I)                                 \
  X(rope_theta_sliding, "rope_theta_sliding", F)                               \
  X(rope_theta_full, "rope_theta_full", F)                                     \
  X(partial_rotary_full, "partial_rotary_full", F)                             \
  X(use_double_wide_mlp, "use_double_wide_mlp", B)                             \
  X(attention_k_eq_v, "attention_k_eq_v", B)                                   \
  X(final_logit_softcapping, "final_logit_softcapping", F)

// Serializes to a flat JSON object. Values are strings, because safetensors' `__metadata__` is
// specified as a string->string map and readers in the wild (including HF's) reject anything else
//; so the container stays valid safetensors and `safetensors.safe_open` can read it.
inline std::string config_to_json(const LlamaConfig& c) {
  std::string s = "{";
  bool first = true;
  auto comma = [&]() {
    if (!first) s += ",";
    first = false;
  };
  auto put = [&](const char* k, const std::string& v) {
    comma();
    s += "\"";
    s += k;
    s += "\":\"";
    s += v;
    s += "\"";
  };

#define CPI_EMIT_I(member, key, kind) put(key, std::to_string(c.member));
#define CPI_EMIT_F(member, key, kind) put(key, std::to_string(c.member));
#define CPI_EMIT_B(member, key, kind) put(key, c.member ? "1" : "0");
#define CPI_EMIT(member, key, kind) CPI_EMIT_##kind(member, key, kind)
  CPI_CONFIG_FIELDS(CPI_EMIT)
#undef CPI_EMIT
#undef CPI_EMIT_I
#undef CPI_EMIT_F
#undef CPI_EMIT_B

  put("dtype", c.dtype);
  put("model_family", std::to_string(static_cast<int>(c.model_family)));
  // Per-layer attention kinds as one digit each ('0' full, '1' sliding, '2' linear). A compact
  // string rather than a JSON array because __metadata__ values must be strings.
  std::string kinds;
  kinds.reserve(c.layer_attention_kinds.size());
  for (AttentionKind k : c.layer_attention_kinds) {
    kinds += static_cast<char>('0' + static_cast<int>(k));
  }
  put("layer_attention_kinds", kinds);
  // Gemma 4's KV sharing: which layer's K/V each layer reads. COMMA-SEPARATED, not one digit each
  // like the kinds above; these are layer INDICES and a 35-layer model already exceeds 9, so the
  // compact form would silently truncate layer 10 to '1'. Empty for every other family.
  std::string kvsrc;
  for (std::size_t i = 0; i < c.kv_source.size(); ++i) {
    if (i) kvsrc += ',';
    kvsrc += std::to_string(c.kv_source[i]);
  }
  put("kv_source", kvsrc);
  s += "}";
  return s;
}

inline LlamaConfig config_from_json(const std::string& j) {
  LlamaConfig c;

  // json_get_int/float read a NUMBER; these are quoted strings, so pull the string and convert.
  auto get_str = [&](const char* k) { return engine::mini::json_get_string(j, k); };

#define CPI_READ_I(member, key, kind)                    \
  {                                                      \
    const std::string v = get_str(key);                  \
    if (!v.empty()) c.member = std::stoi(v);             \
  }
#define CPI_READ_F(member, key, kind)                    \
  {                                                      \
    const std::string v = get_str(key);                  \
    if (!v.empty()) c.member = std::stof(v);             \
  }
#define CPI_READ_B(member, key, kind)                    \
  {                                                      \
    const std::string v = get_str(key);                  \
    if (!v.empty()) c.member = (v != "0");               \
  }
#define CPI_READ(member, key, kind) CPI_READ_##kind(member, key, kind)
  CPI_CONFIG_FIELDS(CPI_READ)
#undef CPI_READ
#undef CPI_READ_I
#undef CPI_READ_F
#undef CPI_READ_B

  const std::string dt = get_str("dtype");
  if (!dt.empty()) c.dtype = dt;
  const std::string fam = get_str("model_family");
  if (!fam.empty()) c.model_family = static_cast<ModelFamily>(std::stoi(fam));
  const std::string kinds = get_str("layer_attention_kinds");
  c.layer_attention_kinds.clear();
  c.layer_attention_kinds.reserve(kinds.size());
  for (char ch : kinds) {
    c.layer_attention_kinds.push_back(static_cast<AttentionKind>(ch - '0'));
  }
  // Gemma 4 KV sharing, comma-separated layer indices (see the writer).
  const std::string kvsrc = get_str("kv_source");
  c.kv_source.clear();
  for (std::size_t i = 0; i < kvsrc.size();) {
    std::size_t j = kvsrc.find(',', i);
    if (j == std::string::npos) j = kvsrc.size();
    if (j > i) c.kv_source.push_back(std::stoi(kvsrc.substr(i, j - i)));
    i = j + 1;
  }
  return c;
}

}  // namespace model
