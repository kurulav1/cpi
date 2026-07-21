#include "engine/plan_model_config.hpp"

#include <stdexcept>

#include "model/json_mini.hpp"

namespace engine {

// NOTE: no `namespace mini = engine::mini;` alias here -- we are already inside `engine`, so the
// alias collides with the namespace it names (error C2386) and `mini::` resolves on its own.

PlanModelConfig parse_gemma4_text_config(const std::string& config_json) {
  const std::string tc = mini::json_extract_object(config_json, "text_config");
  if (tc.empty()) throw std::runtime_error("Gemma 4 config is missing text_config");

  PlanModelConfig c;
  c.family = PlanFamily::Gemma4;
  c.vocab = mini::json_get_int(tc, "vocab_size", 0);
  c.hidden = mini::json_get_int(tc, "hidden_size", 0);
  c.intermediate = mini::json_get_int(tc, "intermediate_size", 0);
  c.num_layers = mini::json_get_int(tc, "num_hidden_layers", 0);
  c.num_heads = mini::json_get_int(tc, "num_attention_heads", 0);
  c.head_dim = mini::json_get_int(tc, "head_dim", 256);
  c.head_dim_sliding = c.head_dim;
  c.head_dim_full = mini::json_get_int(tc, "global_head_dim", c.head_dim);
  c.num_kv_heads_sliding = mini::json_get_int(tc, "num_key_value_heads", 1);
  c.num_kv_heads_full = mini::json_get_int(tc, "num_global_key_value_heads", c.num_kv_heads_sliding);
  c.num_kv_heads = c.num_kv_heads_sliding;
  c.hidden_size_per_layer_input = mini::json_get_int(tc, "hidden_size_per_layer_input", 0);
  c.num_kv_shared_layers = mini::json_get_int(tc, "num_kv_shared_layers", 0);
  c.sliding_window = mini::json_get_int(tc, "sliding_window", 0);
  c.rms_eps = mini::json_get_float(tc, "rms_norm_eps", 1e-6f);
  c.final_logit_softcapping = mini::json_get_float(tc, "final_logit_softcapping", 0.0f);
  c.attention_k_eq_v = mini::json_get_bool(tc, "attention_k_eq_v", false);
  c.use_double_wide_mlp = mini::json_get_bool(tc, "use_double_wide_mlp", false);
  c.tie_word_embeddings = mini::json_get_bool(tc, "tie_word_embeddings", true);
  c.eos_token_id = mini::json_get_int(tc, "eos_token_id", 1);
  c.bos_token_id = mini::json_get_int(tc, "bos_token_id", 2);
  // Only meaningful when there are per-layer embeddings; the shape registry falls back to `vocab`.
  c.vocab_size_per_layer_input = mini::json_get_int(tc, "vocab_size_per_layer_input", 0);

  c.enable_moe_block = mini::json_get_bool(tc, "enable_moe_block", false);
  c.num_experts = mini::json_get_int(tc, "num_experts", 0);
  c.top_k_experts = mini::json_get_int(tc, "top_k_experts", 0);
  c.moe_intermediate_size = mini::json_get_int(tc, "moe_intermediate_size", 0);
  if (c.enable_moe_block &&
      (c.num_experts <= 0 || c.top_k_experts <= 0 || c.moe_intermediate_size <= 0))
    throw std::runtime_error("Gemma 4 MoE config is missing num_experts/top_k/moe_intermediate");

  // layer_types: "full_attention" vs "sliding_attention", per layer.
  const auto types = mini::json_get_string_array(tc, "layer_types");
  c.layer_full.assign(c.num_layers, 0);
  for (int L = 0; L < c.num_layers && L < static_cast<int>(types.size()); ++L)
    c.layer_full[L] = (types[L] == "full_attention") ? 1 : 0;
  // KV sharing: the last num_kv_shared_layers layers do NOT project their own K/V --
  // they reuse the cache of the LAST NON-SHARED LAYER OF THE SAME TYPE (sliding vs
  // full). E2B shares 20 of its 35 layers; the MoE and 12B share none. Getting this
  // wrong still runs, and still produces fluent-looking garbage.
  c.first_shared_layer = c.num_kv_shared_layers > 0 ? c.num_layers - c.num_kv_shared_layers : 0;
  c.kv_source.assign(c.num_layers, 0);
  for (int L = 0; L < c.num_layers; ++L) {
    if (c.first_shared_layer > 0 && L >= c.first_shared_layer) {
      int src = L;
      for (int i = 0; i < c.first_shared_layer; ++i)
        if (c.layer_full[i] == c.layer_full[L]) src = i;  // last match wins
      c.kv_source[L] = src;
    } else {
      c.kv_source[L] = L;
    }
  }

  if (c.vocab <= 0 || c.hidden <= 0 || c.num_layers <= 0)
    throw std::runtime_error("Gemma 4 config.json is missing required text_config fields");

  return c;
}

}  // namespace engine
