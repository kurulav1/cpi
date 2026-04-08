#include "llama_engine_internal.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>
#include <cuda_fp16.h>
#include "common.hpp"
#include "runtime/cuda_utils.cuh"
#include "runtime/system_info.hpp"
namespace engine {
namespace {
std::size_t bytes_for_matrix(int rows, int cols) {
  return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(__half);
}
}  // namespace
void LlamaEngine::enforce_host_resource_limits(const char* stage) {
  if (!options_.enable_host_resource_limits) {
    return;
  }
  const bool check_cpu = options_.max_cpu_percent > 0.0;
  const bool check_mem = options_.max_memory_percent > 0.0;
  if (!check_cpu && !check_mem) {
    return;
  }

  const auto now = std::chrono::steady_clock::now();
  const int sample_interval_ms = std::max(0, options_.resource_sample_interval_ms);
  const bool should_sample = !resource_sample_ready_ || sample_interval_ms == 0 ||
                             std::chrono::duration_cast<std::chrono::milliseconds>(
                                 now - last_resource_sample_time_).count() >= sample_interval_ms;

  if (should_sample) {
    const runtime::HostResourceUsage usage = runtime::query_host_resource_usage();
    sampled_cpu_percent_ = usage.cpu_percent;
    sampled_memory_percent_ = usage.memory_percent;
    last_resource_sample_time_ = now;
    resource_sample_ready_ = true;
  }

  const bool cpu_over = check_cpu && sampled_cpu_percent_ >= 0.0 &&
                        sampled_cpu_percent_ > options_.max_cpu_percent;
  const bool mem_over = check_mem && sampled_memory_percent_ >= 0.0 &&
                        sampled_memory_percent_ > options_.max_memory_percent;
  const bool over = cpu_over || mem_over;

  if (!over) {
    over_limit_active_ = false;
    return;
  }

  if (!over_limit_active_) {
    over_limit_active_ = true;
    over_limit_since_ = now;
    last_over_limit_log_time_ = std::chrono::steady_clock::time_point{};
  }

  const auto over_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - over_limit_since_).count();
  const int sustain_ms = std::max(1, options_.resource_sustain_ms);
  if (options_.verbose) {
    const bool can_log = last_over_limit_log_time_.time_since_epoch().count() == 0 ||
                         std::chrono::duration_cast<std::chrono::milliseconds>(
                             now - last_over_limit_log_time_).count() >= 1000;
    if (can_log) {
      std::cout << "[limits] over-threshold stage=" << (stage ? stage : "unknown")
                << " cpu=" << std::fixed << std::setprecision(1) << sampled_cpu_percent_
                << "% mem=" << sampled_memory_percent_
                << "% sustained_ms=" << over_ms << "/" << sustain_ms << "\n";
      last_over_limit_log_time_ = now;
    }
  }

  if (over_ms >= sustain_ms) {
    std::ostringstream oss;
    oss << "host resource limit exceeded at stage=" << (stage ? stage : "unknown")
        << " cpu=" << std::fixed << std::setprecision(1) << sampled_cpu_percent_
        << "% (limit=" << options_.max_cpu_percent << "%)"
        << " mem=" << sampled_memory_percent_
        << "% (limit=" << options_.max_memory_percent << "%)"
        << " sustained_ms=" << over_ms;
    LLAMA_ENGINE_THROW(oss.str());
  }

  const int throttle_ms = std::max(0, options_.resource_throttle_sleep_ms);
  if (throttle_ms > 0) {
    std::this_thread::sleep_for(std::chrono::milliseconds(throttle_ms));
  }
}

void LlamaEngine::check_tq_cached_init_timeout(const std::chrono::steady_clock::time_point& start,
                                               int layer_index) {
  if (!tq3_enabled_ || !options_.enable_tq_cached || options_.tq_cached_init_timeout_ms <= 0) {
    return;
  }
  const auto now = std::chrono::steady_clock::now();
  const auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
  if (elapsed_ms <= options_.tq_cached_init_timeout_ms) {
    return;
  }
  std::ostringstream oss;
  oss << "TurboQuant cached init timeout at layer=" << layer_index
      << " elapsed_ms=" << elapsed_ms
      << " limit_ms=" << options_.tq_cached_init_timeout_ms;
  LLAMA_ENGINE_THROW(oss.str());
}

void LlamaEngine::load_static_weights() {
  const auto& cfg = weights_.config();

  const std::string emb_name = "tok_embeddings.weight";
  if (!weights_.has_tensor(emb_name)) {
    LLAMA_ENGINE_THROW("missing tensor: " + emb_name);
  }
  const std::size_t emb_bytes = weights_.tensor_bytes(emb_name);
  CUDA_CHECK(cudaMalloc(&d_tok_embeddings_, emb_bytes));
  CUDA_CHECK(cudaMemcpy(d_tok_embeddings_, weights_.tensor_data(emb_name), emb_bytes, cudaMemcpyHostToDevice));

  const std::string norm_name = "norm.weight";
  if (!weights_.has_tensor(norm_name)) {
    LLAMA_ENGINE_THROW("missing tensor: " + norm_name);
  }
  const std::size_t norm_bytes = weights_.tensor_bytes(norm_name);
  CUDA_CHECK(cudaMalloc(&d_norm_out_, norm_bytes));
  CUDA_CHECK(cudaMemcpy(d_norm_out_, weights_.tensor_data(norm_name), norm_bytes, cudaMemcpyHostToDevice));
  if (weights_.has_tensor("norm.bias")) {
    CUDA_CHECK(cudaMalloc(&d_norm_out_bias_, norm_bytes));
    CUDA_CHECK(cudaMemcpy(d_norm_out_bias_, weights_.tensor_data("norm.bias"), norm_bytes, cudaMemcpyHostToDevice));
  }

  const std::string out_name = weights_.has_tensor("output.weight") ? "output.weight" : emb_name;
  const std::size_t out_bytes = weights_.tensor_bytes(out_name);
  CUDA_CHECK(cudaMalloc(&d_lm_head_, out_bytes));
  CUDA_CHECK(cudaMemcpy(d_lm_head_, weights_.tensor_data(out_name), out_bytes, cudaMemcpyHostToDevice));
  if (weights_.has_tensor("output.bias")) {
    const std::size_t out_bias_bytes = static_cast<std::size_t>(cfg.vocab_size) * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_lm_head_bias_, out_bias_bytes));
    CUDA_CHECK(cudaMemcpy(d_lm_head_bias_, weights_.tensor_data("output.bias"), out_bias_bytes, cudaMemcpyHostToDevice));
  }

    // Detect TurboQuant metadata and load shared parameters.
  if (weights_.has_tensor("tq3_codebook") && weights_.has_tensor("tq3_signs_hidden")) {
    tq3_enabled_ = true;
    CUDA_CHECK(cudaMalloc(&d_tq3_codebook_, 8 * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_tq3_codebook_, weights_.tensor_data("tq3_codebook"),
                          8 * sizeof(__half), cudaMemcpyHostToDevice));
    const std::size_t signs_bytes = static_cast<std::size_t>(cfg.hidden_size) * sizeof(int8_t);
    CUDA_CHECK(cudaMalloc(&d_tq3_signs_, signs_bytes));
    CUDA_CHECK(cudaMemcpy(d_tq3_signs_, weights_.tensor_data("tq3_signs_hidden"),
                          signs_bytes, cudaMemcpyHostToDevice));

    // Block-diagonal WHT block size (new files store tq3_block_size; old files
    // used full-width WHT so block_size defaults to hidden_size for compat).
    tq3_block_size_ = cfg.hidden_size;
    if (weights_.has_tensor("tq3_block_size")) {
      std::int32_t bs = 0;
      std::memcpy(&bs, weights_.tensor_data("tq3_block_size"), sizeof(std::int32_t));
      if (bs > 0) tq3_block_size_ = static_cast<int>(bs);
    }
    std::cout << "[engine] tq3_block_size=" << tq3_block_size_ << "\n";

    tq_objective_file_ = 0;
    if (weights_.has_tensor("tq_objective")) {
      std::int32_t objective = 0;
      std::memcpy(&objective, weights_.tensor_data("tq_objective"), sizeof(std::int32_t));
      tq_objective_file_ = static_cast<int>(objective);
    }

    const bool has_qjl_meta =
        weights_.has_tensor("tq_qjl_dim") &&
        weights_.has_tensor("tq_qjl_seed") &&
        weights_.has_tensor("tq_qjl_indices_hidden") &&
        weights_.has_tensor("tq_qjl_signs_hidden");

    if (has_qjl_meta) {
      std::int32_t qjl_dim_i32 = 0;
      std::memcpy(&qjl_dim_i32, weights_.tensor_data("tq_qjl_dim"), sizeof(std::int32_t));
      tq_qjl_dim_ = static_cast<int>(qjl_dim_i32);
      if (tq_qjl_dim_ <= 0 || tq_qjl_dim_ > cfg.hidden_size) {
        LLAMA_ENGINE_THROW("invalid tq_qjl_dim in model metadata");
      }
      tq_qjl_words_ = (tq_qjl_dim_ + 31) / 32;
      const std::size_t qjl_idx_bytes = static_cast<std::size_t>(tq_qjl_dim_) * sizeof(std::int32_t);
      const std::size_t qjl_sign_bytes = static_cast<std::size_t>(tq_qjl_dim_) * sizeof(std::int8_t);
      CUDA_CHECK(cudaMalloc(&d_tq_qjl_indices_, qjl_idx_bytes));
      CUDA_CHECK(cudaMemcpy(d_tq_qjl_indices_, weights_.tensor_data("tq_qjl_indices_hidden"),
                            qjl_idx_bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMalloc(&d_tq_qjl_signs_, qjl_sign_bytes));
      CUDA_CHECK(cudaMemcpy(d_tq_qjl_signs_, weights_.tensor_data("tq_qjl_signs_hidden"),
                            qjl_sign_bytes, cudaMemcpyHostToDevice));
    }

    const std::string requested_mode = options_.tq_mode.empty() ? "auto" : options_.tq_mode;
    if (requested_mode != "auto" && requested_mode != "mse" && requested_mode != "prod") {
      LLAMA_ENGINE_THROW("invalid tq_mode option: expected auto|mse|prod");
    }
    if (requested_mode == "prod") {
      if (!has_qjl_meta) {
        LLAMA_ENGINE_THROW("--tq-mode=prod requested but Qprod metadata is missing from model");
      }
      tq_prod_enabled_ = true;
    } else if (requested_mode == "mse") {
      tq_prod_enabled_ = false;
    } else {
      tq_prod_enabled_ = (tq_objective_file_ == 1) && has_qjl_meta;
    }

    if (options_.verbose) {
      std::cout << "[engine] TurboQuant detected: objective_file="
                << (tq_objective_file_ == 1 ? "prod" : "mse")
                << " runtime_mode=" << (tq_prod_enabled_ ? "prod" : "mse")
                << " qjl_dim=" << tq_qjl_dim_ << "\n";
    }
  }

  (void)cfg;
}

void LlamaEngine::allocate_runtime_buffers() {
  const auto& cfg = weights_.config();
  const int hidden = cfg.hidden_size;
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : hidden;
  const int inter = cfg.intermediate_size;
  const int expert_inter = cfg.effective_expert_intermediate_size() > 0
      ? cfg.effective_expert_intermediate_size()
      : inter;
  const int ffn_inter = (expert_inter > inter) ? expert_inter : inter;
  kv_int4_enabled_ = options_.kv_cache_int4;
  if (kv_int4_enabled_ && options_.paged_kv_cache) {
    LLAMA_ENGINE_THROW("kv_cache_int4 and paged_kv_cache are mutually exclusive");
  }
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (cfg.hidden_size / cfg.num_heads);
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const int rows = prefill_chunk_size_;
  int max_lowbit_cols = (hidden > inter) ? hidden : inter;
  if (expert_inter > max_lowbit_cols) {
    max_lowbit_cols = expert_inter;
  }

  CUDA_CHECK(cudaMalloc(&d_token_id_, static_cast<std::size_t>(rows) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_x_, bytes_for_matrix(rows, hidden)));
  CUDA_CHECK(cudaMalloc(&d_x_norm_, bytes_for_matrix(rows, hidden)));
  if (tq3_enabled_) {
    CUDA_CHECK(cudaMalloc(&d_x_tq3_, bytes_for_matrix(1, hidden)));
    if (tq_prod_enabled_ && tq_qjl_words_ > 0) {
      CUDA_CHECK(cudaMalloc(&d_tq_qjl_x_bits_, static_cast<std::size_t>(tq_qjl_words_) * sizeof(std::uint32_t)));
    }
  }
  CUDA_CHECK(cudaMalloc(&d_qkv_, bytes_for_matrix(rows, q_hidden + 2 * kv_hidden)));
  d_q_ = d_qkv_;
  d_k_ = static_cast<void*>(static_cast<__half*>(d_qkv_) + q_hidden);
  d_v_ = static_cast<void*>(static_cast<__half*>(d_qkv_) + q_hidden + kv_hidden);
  CUDA_CHECK(cudaMalloc(&d_prefill_q_, bytes_for_matrix(rows, q_hidden)));
  CUDA_CHECK(cudaMalloc(&d_prefill_k_, bytes_for_matrix(rows, kv_hidden)));
  CUDA_CHECK(cudaMalloc(&d_prefill_v_, bytes_for_matrix(rows, kv_hidden)));
  CUDA_CHECK(cudaMalloc(&d_att_, bytes_for_matrix(rows, std::max(hidden, q_hidden))));
  CUDA_CHECK(cudaMalloc(&d_ff13_, bytes_for_matrix(rows, 2 * inter)));
  d_ff1_ = d_ff13_;
  d_ff2_ = static_cast<void*>(static_cast<__half*>(d_ff13_) + inter);
  CUDA_CHECK(cudaMalloc(&d_prefill_ff1_, bytes_for_matrix(rows, ffn_inter)));
  CUDA_CHECK(cudaMalloc(&d_prefill_ff2_, bytes_for_matrix(rows, ffn_inter)));
  CUDA_CHECK(cudaMalloc(&d_prefill_i8_, static_cast<std::size_t>(rows) * static_cast<std::size_t>(max_lowbit_cols)));
  CUDA_CHECK(cudaMalloc(&d_prefill_i8_scales_, static_cast<std::size_t>(rows) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ff3_, bytes_for_matrix(rows, hidden)));
  CUDA_CHECK(cudaMalloc(&d_logits_, static_cast<std::size_t>(cfg.vocab_size) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_argmax_, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_decode_position_, sizeof(int)));
  const int half_dim = head_dim / 2;
  const std::size_t rope_table_elems =
      static_cast<std::size_t>(options_.max_context) * static_cast<std::size_t>(half_dim);
  std::vector<float> rope_cos(rope_table_elems);
  std::vector<float> rope_sin(rope_table_elems);
  // Resolve effective RoPE theta: CLI override wins, then model file, then family default.
  const float eff_rope_theta = (options_.rope_theta > 0.0f) ? options_.rope_theta : cfg.effective_rope_theta();
  if (options_.verbose) {
    std::cout << "[engine] rope_theta=" << eff_rope_theta
              << " (source=" << (options_.rope_theta > 0.0f ? "cli" :
                                 (cfg.rope_theta > 0.0f ? "model_file" : "family_default")) << ")\n";
  }

  for (int pos = 0; pos < options_.max_context; ++pos) {
    for (int pair = 0; pair < half_dim; ++pair) {
      const float theta = std::pow(eff_rope_theta, -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
      const float angle = static_cast<float>(pos) * theta;
      rope_cos[static_cast<std::size_t>(pos) * static_cast<std::size_t>(half_dim) + static_cast<std::size_t>(pair)] = std::cos(angle);
      rope_sin[static_cast<std::size_t>(pos) * static_cast<std::size_t>(half_dim) + static_cast<std::size_t>(pair)] = std::sin(angle);
    }
  }
  CUDA_CHECK(cudaMalloc(&d_rope_cos_, rope_table_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_rope_sin_, rope_table_elems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_rope_cos_, rope_cos.data(), rope_table_elems * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rope_sin_, rope_sin.data(), rope_table_elems * sizeof(float), cudaMemcpyHostToDevice));

  attn_chunk_capacity_ = std::max(1, (options_.max_context + 31) / 32);
  CUDA_CHECK(cudaMalloc(&d_attn_chunk_m_,
                        static_cast<std::size_t>(cfg.num_heads) * static_cast<std::size_t>(attn_chunk_capacity_) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_attn_chunk_l_,
                        static_cast<std::size_t>(cfg.num_heads) * static_cast<std::size_t>(attn_chunk_capacity_) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_attn_chunk_o_,
                        static_cast<std::size_t>(cfg.num_heads) * static_cast<std::size_t>(attn_chunk_capacity_) *
                            static_cast<std::size_t>(head_dim) * sizeof(float)));

  if (cfg.is_moe()) {
    const int experts = std::max(1, cfg.num_local_experts);
    const int top_k = std::max(1, std::min(cfg.num_experts_per_tok > 0 ? cfg.num_experts_per_tok : 2, experts));
    const std::size_t expert_w13_bytes =
        static_cast<std::size_t>(expert_inter) * static_cast<std::size_t>(hidden) * sizeof(__half);
    const std::size_t expert_w2_bytes =
        static_cast<std::size_t>(hidden) * static_cast<std::size_t>(expert_inter) * sizeof(__half);
    const std::size_t expert_w13_q8_bytes =
        static_cast<std::size_t>(expert_inter) * static_cast<std::size_t>(hidden);
    const std::size_t expert_w13_q4_bytes =
        static_cast<std::size_t>(expert_inter) * static_cast<std::size_t>((hidden + 1) / 2);
    const std::size_t expert_w2_q8_bytes =
        static_cast<std::size_t>(hidden) * static_cast<std::size_t>(expert_inter);
    const std::size_t expert_w2_q4_bytes =
        static_cast<std::size_t>(hidden) * static_cast<std::size_t>((expert_inter + 1) / 2);
    const std::size_t router_fp16_bytes =
        static_cast<std::size_t>(experts) * static_cast<std::size_t>(hidden) * sizeof(__half);
    const std::size_t router_logits_bytes =
        static_cast<std::size_t>(experts) * sizeof(__half);
    const std::size_t router_q8_bytes =
        static_cast<std::size_t>(experts) * static_cast<std::size_t>(hidden);
    const std::size_t router_q4_bytes =
        static_cast<std::size_t>(experts) * static_cast<std::size_t>((hidden + 1) / 2);

    CUDA_CHECK(cudaMalloc(&d_moe_router_w_, router_fp16_bytes));
    CUDA_CHECK(cudaMalloc(&d_moe_router_logits_, router_logits_bytes));
    CUDA_CHECK(cudaMalloc(&d_moe_router_w_q_, std::max(router_q8_bytes, router_q4_bytes)));
    CUDA_CHECK(cudaMalloc(&d_moe_router_scales_, static_cast<std::size_t>(experts) * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_moe_w1_, expert_w13_bytes));
    CUDA_CHECK(cudaMalloc(&d_moe_w2_, expert_w2_bytes));
    CUDA_CHECK(cudaMalloc(&d_moe_w3_, expert_w13_bytes));

    CUDA_CHECK(cudaMalloc(&d_moe_w1_q_, std::max(expert_w13_q8_bytes, expert_w13_q4_bytes)));
    CUDA_CHECK(cudaMalloc(&d_moe_w2_q_, std::max(expert_w2_q8_bytes, expert_w2_q4_bytes)));
    CUDA_CHECK(cudaMalloc(&d_moe_w3_q_, std::max(expert_w13_q8_bytes, expert_w13_q4_bytes)));

    CUDA_CHECK(cudaMalloc(&d_moe_s_w1_, static_cast<std::size_t>(expert_inter) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_moe_s_w2_, static_cast<std::size_t>(hidden) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_moe_s_w3_, static_cast<std::size_t>(expert_inter) * sizeof(float)));

    CUDA_CHECK(cudaMalloc(&d_moe_topk_idx_, static_cast<std::size_t>(top_k) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_moe_topk_prob_, static_cast<std::size_t>(top_k) * sizeof(float)));
  }

  const std::size_t kv_bytes = static_cast<std::size_t>(cfg.num_layers) * static_cast<std::size_t>(options_.max_context) *
                               static_cast<std::size_t>(kv_hidden) * sizeof(__half);
  if (kv_int4_enabled_) {
    // INT4 KV: packed nibbles + per-head fp16 scales; FP16 KV buffers not allocated.
    const int packed_per_head = head_dim / 2;
    const std::size_t i4_bytes = static_cast<std::size_t>(cfg.num_layers) *
                                  static_cast<std::size_t>(options_.max_context) *
                                  static_cast<std::size_t>(cfg.num_kv_heads) *
                                  static_cast<std::size_t>(packed_per_head);
    const std::size_t sc_bytes = static_cast<std::size_t>(cfg.num_layers) *
                                  static_cast<std::size_t>(options_.max_context) *
                                  static_cast<std::size_t>(cfg.num_kv_heads) * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_k_cache_i4_, i4_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_cache_i4_, i4_bytes));
    CUDA_CHECK(cudaMalloc(&d_k_scales_, sc_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_scales_, sc_bytes));
    CUDA_CHECK(cudaMemset(d_k_cache_i4_, 0, i4_bytes));
    CUDA_CHECK(cudaMemset(d_v_cache_i4_, 0, i4_bytes));
    CUDA_CHECK(cudaMemset(d_k_scales_, 0, sc_bytes));
    CUDA_CHECK(cudaMemset(d_v_scales_, 0, sc_bytes));
    if (options_.verbose) {
      std::cout << "[engine] kv_cache_int4=on  KV VRAM: "
                << (i4_bytes * 2 + sc_bytes * 2) / (1024 * 1024) << " MiB"
                << " (vs " << (kv_bytes * 2) / (1024 * 1024) << " MiB fp16)\n";
    }
  } else if (options_.paged_kv_cache) {
    const std::size_t stage_bytes =
        static_cast<std::size_t>(options_.max_context) * static_cast<std::size_t>(kv_hidden) * sizeof(__half);
    CUDA_CHECK(cudaHostAlloc(&h_k_cache_, kv_bytes, cudaHostAllocPortable));
    CUDA_CHECK(cudaHostAlloc(&h_v_cache_, kv_bytes, cudaHostAllocPortable));
    std::memset(h_k_cache_, 0, kv_bytes);
    std::memset(h_v_cache_, 0, kv_bytes);
    CUDA_CHECK(cudaMalloc(&d_k_cache_, stage_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_cache_, stage_bytes));
    CUDA_CHECK(cudaMemset(d_k_cache_, 0, stage_bytes));
    CUDA_CHECK(cudaMemset(d_v_cache_, 0, stage_bytes));
  } else {
    CUDA_CHECK(cudaMalloc(&d_k_cache_, kv_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_cache_, kv_bytes));
    CUDA_CHECK(cudaMemset(d_k_cache_, 0, kv_bytes));
    CUDA_CHECK(cudaMemset(d_v_cache_, 0, kv_bytes));
  }

  for (auto& lw : streaming_layer_weights_) {
    CUDA_CHECK(cudaMalloc(&lw.wqkv, bytes_for_matrix(q_hidden + 2 * kv_hidden, hidden)));
    CUDA_CHECK(cudaMalloc(&lw.wo, bytes_for_matrix(hidden, q_hidden)));
    CUDA_CHECK(cudaMalloc(&lw.bo, bytes_for_matrix(1, hidden)));
    CUDA_CHECK(cudaMalloc(&lw.w13, bytes_for_matrix(2 * inter, hidden)));
    CUDA_CHECK(cudaMalloc(&lw.w2, bytes_for_matrix(hidden, inter)));
    CUDA_CHECK(cudaMalloc(&lw.norm_att, bytes_for_matrix(1, hidden)));
    CUDA_CHECK(cudaMalloc(&lw.norm_ffn, bytes_for_matrix(1, hidden)));
    CUDA_CHECK(cudaMalloc(&lw.norm_att_bias, bytes_for_matrix(1, hidden)));
    CUDA_CHECK(cudaMalloc(&lw.norm_ffn_bias, bytes_for_matrix(1, hidden)));
    CUDA_CHECK(cudaMemsetAsync(lw.bo, 0, bytes_for_matrix(1, hidden), compute_stream_));
    CUDA_CHECK(cudaMemsetAsync(lw.norm_att_bias, 0, bytes_for_matrix(1, hidden), compute_stream_));
    CUDA_CHECK(cudaMemsetAsync(lw.norm_ffn_bias, 0, bytes_for_matrix(1, hidden), compute_stream_));
    if (cfg.has_qkv_bias) {
      CUDA_CHECK(cudaMalloc(&lw.bqkv, bytes_for_matrix(1, q_hidden + 2 * kv_hidden)));
    }
  }

  if (lowbit_streaming_enabled(options_)) {
    for (auto& iw : streaming_layer_weights_i8_) {
      CUDA_CHECK(cudaMalloc(&iw.w1, static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden)));
      CUDA_CHECK(cudaMalloc(&iw.w2, static_cast<std::size_t>(hidden) * static_cast<std::size_t>(inter)));
      CUDA_CHECK(cudaMalloc(&iw.w3, static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden)));
      CUDA_CHECK(cudaMalloc(&iw.s_w1, static_cast<std::size_t>(inter) * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&iw.s_w2, static_cast<std::size_t>(hidden) * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&iw.s_w3, static_cast<std::size_t>(inter) * sizeof(float)));
    }
  }
}

void LlamaEngine::reset_kv_cache() {
  const auto& cfg = weights_.config();
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (cfg.hidden_size / cfg.num_heads);
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const std::size_t kv_bytes = static_cast<std::size_t>(cfg.num_layers) * static_cast<std::size_t>(options_.max_context) *
                               static_cast<std::size_t>(kv_hidden) * sizeof(__half);
  if (kv_int4_enabled_) {
    const int packed_per_head = head_dim / 2;
    const std::size_t i4_bytes = static_cast<std::size_t>(cfg.num_layers) *
                                  static_cast<std::size_t>(options_.max_context) *
                                  static_cast<std::size_t>(cfg.num_kv_heads) *
                                  static_cast<std::size_t>(packed_per_head);
    const std::size_t sc_bytes = static_cast<std::size_t>(cfg.num_layers) *
                                  static_cast<std::size_t>(options_.max_context) *
                                  static_cast<std::size_t>(cfg.num_kv_heads) * sizeof(__half);
    CUDA_CHECK(cudaMemset(d_k_cache_i4_, 0, i4_bytes));
    CUDA_CHECK(cudaMemset(d_v_cache_i4_, 0, i4_bytes));
    CUDA_CHECK(cudaMemset(d_k_scales_, 0, sc_bytes));
    CUDA_CHECK(cudaMemset(d_v_scales_, 0, sc_bytes));
    return;
  }
  if (options_.paged_kv_cache) {
    const std::size_t stage_bytes =
        static_cast<std::size_t>(options_.max_context) * static_cast<std::size_t>(kv_hidden) * sizeof(__half);
    if (h_k_cache_) {
      std::memset(h_k_cache_, 0, kv_bytes);
    }
    if (h_v_cache_) {
      std::memset(h_v_cache_, 0, kv_bytes);
    }
    CUDA_CHECK(cudaMemset(d_k_cache_, 0, stage_bytes));
    CUDA_CHECK(cudaMemset(d_v_cache_, 0, stage_bytes));
  } else {
    CUDA_CHECK(cudaMemset(d_k_cache_, 0, kv_bytes));
    CUDA_CHECK(cudaMemset(d_v_cache_, 0, kv_bytes));
  }
}
}  // namespace engine
