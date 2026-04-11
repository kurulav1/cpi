#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace engine {

struct EngineOptions {
  std::string model_path;
  int max_batch = 1;
  int max_context = 2048;
  int tensor_parallel = 1;
  std::size_t vram_safety_margin_mb = 1024;
  int gpu_cache_layers = -1;
  std::size_t gpu_cache_limit_mb = 0;
  bool gpu_cache_all = false;
  int top_k = 40;
  float top_p = 0.9f;
  float repetition_penalty = 1.0f;
  int no_repeat_ngram_size = 0;
  bool int8_streaming = false;
  int streaming_quant_bits = 8;
  bool prefer_lowbit_cache = false;
  bool profile_decode_phases = false;
  bool disable_split_attention = false;
  bool loop_guard = true;
  bool paged_kv_cache = false;
  bool kv_cache_int4 = false;
  bool enable_tq_cached = false;
  std::string tq_mode = "auto";
  bool verbose = true;
  float rope_theta = 0.0f;
  bool enable_host_resource_limits = true;
  double max_cpu_percent = 85.0;
  double max_memory_percent = 85.0;
  int resource_sample_interval_ms = 250;
  int resource_sustain_ms = 5000;
  int resource_throttle_sleep_ms = 50;
  int tq_cached_init_timeout_ms = 180000;
  int tq_first_token_timeout_ms = 120000;
  int eos_token_id = 2;
};

struct BenchmarkStats {
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  double transfer_ms = 0.0;
  double decode_rmsnorm_ms = 0.0;
  double decode_qkv_ms = 0.0;
  double decode_kv_store_ms = 0.0;
  double decode_attention_ms = 0.0;
  double decode_wo_ms = 0.0;
  double decode_mlp_ms = 0.0;
  double decode_moe_router_ms = 0.0;
  double decode_moe_expert_ms = 0.0;
  double decode_moe_merge_ms = 0.0;
  double decode_lm_head_ms = 0.0;
  int prompt_tokens = 0;
  int generated_tokens = 0;
  int streamed_layer_copies = 0;
  int tq3_cached_active = 0;
  int moe_topk_layers = 0;
  int moe_topk_k = 0;
  std::vector<int> moe_topk_indices{};
  std::vector<float> moe_topk_probs{};
  std::string moe_quant_mode = "none";
};

}  // namespace engine
