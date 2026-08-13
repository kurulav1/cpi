#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <thread>
#include <vector>

#include "common.hpp"
#include "llama_engine_internal.hpp"
#include "model/gguf_kquant.hpp"
#include "runtime/kernels.cuh"
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
  const bool should_sample =
      !resource_sample_ready_ || sample_interval_ms == 0 ||
      std::chrono::duration_cast<std::chrono::milliseconds>(now - last_resource_sample_time_)
              .count() >= sample_interval_ms;

  if (should_sample) {
    const runtime::HostResourceUsage usage = runtime::query_host_resource_usage();
    sampled_cpu_percent_ = usage.cpu_percent;
    sampled_memory_percent_ = usage.memory_percent;
    last_resource_sample_time_ = now;
    resource_sample_ready_ = true;
  }

  const bool cpu_over =
      check_cpu && sampled_cpu_percent_ >= 0.0 && sampled_cpu_percent_ > options_.max_cpu_percent;
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

  const auto over_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - over_limit_since_).count();
  const int sustain_ms = std::max(1, options_.resource_sustain_ms);
  if (options_.verbose) {
    const bool can_log =
        last_over_limit_log_time_.time_since_epoch().count() == 0 ||
        std::chrono::duration_cast<std::chrono::milliseconds>(now - last_over_limit_log_time_)
                .count() >= 1000;
    if (can_log) {
      std::cout << "[limits] over-threshold stage=" << (stage ? stage : "unknown")
                << " cpu=" << std::fixed << std::setprecision(1) << sampled_cpu_percent_
                << "% mem=" << sampled_memory_percent_ << "% sustained_ms=" << over_ms << "/"
                << sustain_ms << "\n";
      last_over_limit_log_time_ = now;
    }
  }

  if (over_ms >= sustain_ms) {
    std::ostringstream oss;
    oss << "host resource limit exceeded at stage=" << (stage ? stage : "unknown")
        << " cpu=" << std::fixed << std::setprecision(1) << sampled_cpu_percent_
        << "% (limit=" << options_.max_cpu_percent << "%)"
        << " mem=" << sampled_memory_percent_ << "% (limit=" << options_.max_memory_percent << "%)"
        << " sustained_ms=" << over_ms;
    CPI_THROW(oss.str());
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
  const auto elapsed_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count();
  if (elapsed_ms <= options_.tq_cached_init_timeout_ms) {
    return;
  }
  std::ostringstream oss;
  oss << "TurboQuant cached init timeout at layer=" << layer_index << " elapsed_ms=" << elapsed_ms
      << " limit_ms=" << options_.tq_cached_init_timeout_ms;
  CPI_THROW(oss.str());
}

void LlamaEngine::load_static_weights() {
  const auto& cfg = weights_.config();

  const std::string emb_name = "tok_embeddings.weight";
  if (!weights_.has_tensor(emb_name)) {
    CPI_THROW("missing tensor: " + emb_name);
  }
  const std::size_t emb_bytes = weights_.tensor_bytes(emb_name);
  CUDA_CHECK(cudaMalloc(&d_tok_embeddings_, emb_bytes));
  // A quantized GGUF embedding table used to take tensor_data(), which
  // dequantizes on the host into a table-sized vector and copies that, pageable
  // (~630 ms of an 8B warm load, most of static-load). Ship the packed blocks
  // through the staging ring and expand them on the device instead: a quarter
  // of the bytes across the bus and no host-side expansion. The embedding table
  // carries no RoPE permute, so the blocks are usable as stored.
  bool emb_loaded = false;
  if (weights_.is_gguf() && weights_.gguf() != nullptr) {
    const auto pk = weights_.gguf()->packed_kquant(emb_name);
    const std::size_t emb_elems = emb_bytes / sizeof(__half);
    if (pk.valid() && pk.permute_heads <= 0 &&
        static_cast<std::size_t>(pk.rows) * static_cast<std::size_t>(pk.cols) == emb_elems &&
        emb_elems % model::kquant::kSuperBlock == 0) {
      void* d_packed = nullptr;
      if (cudaMalloc(&d_packed, pk.bytes) == cudaSuccess) {
        if (staged_h2d(d_packed, pk.data, pk.bytes)) {
          kernels::launch_dequant_kquant(static_cast<const std::uint8_t*>(d_packed),
                                         static_cast<kernels::KQuantType>(pk.kind),
                                         emb_elems / model::kquant::kSuperBlock,
                                         static_cast<__half*>(d_tok_embeddings_),
                                         transfer_stream_);
          CUDA_CHECK(cudaStreamSynchronize(transfer_stream_));
          emb_loaded = true;
        }
        cudaFree(d_packed);
      }
    }
  }
  if (!emb_loaded) {
    CUDA_CHECK(cudaMemcpy(d_tok_embeddings_, weights_.tensor_data(emb_name), emb_bytes,
                          cudaMemcpyHostToDevice));
  }

  const std::string norm_name = "norm.weight";
  if (!weights_.has_tensor(norm_name)) {
    CPI_THROW("missing tensor: " + norm_name);
  }
  const std::size_t norm_bytes = weights_.tensor_bytes(norm_name);
  CUDA_CHECK(cudaMalloc(&d_norm_out_, norm_bytes));
  CUDA_CHECK(
      cudaMemcpy(d_norm_out_, weights_.tensor_data(norm_name), norm_bytes, cudaMemcpyHostToDevice));
  if (weights_.has_tensor("norm.bias")) {
    CUDA_CHECK(cudaMalloc(&d_norm_out_bias_, norm_bytes));
    CUDA_CHECK(cudaMemcpy(d_norm_out_bias_, weights_.tensor_data("norm.bias"), norm_bytes,
                          cudaMemcpyHostToDevice));
  }

  const std::string out_name = weights_.has_tensor("output.weight") ? "output.weight" : emb_name;
  // The head is read in full for every token, so leaving it fp16 costs more
  // bandwidth than any single layer matrix. It carries no RoPE permutation, so
  // it can be served packed as soon as the container offers it.
  stage_packed_weight(out_name, &lm_head_packed_, 8);
  if (!lm_head_packed_.active()) {
    const std::size_t out_bytes = weights_.tensor_bytes(out_name);
    CUDA_CHECK(cudaMalloc(&d_lm_head_, out_bytes));
    CUDA_CHECK(
        cudaMemcpy(d_lm_head_, weights_.tensor_data(out_name), out_bytes, cudaMemcpyHostToDevice));
  }

  // Weight-only int8 LM head, when the user asked for a quantized model.
  //
  // Only the weights are quantized; the activation stays fp16 and the dot accumulates in
  // fp32 (see launch_weight_only_int8_gemv_f32). This is the one layer that decides the output
  // token, so it does not go through the dp4a path that would also quantize the activation.
  //
  // Row-wise scales: one per vocabulary row, so a few loud rows cannot squash the rest.
  // CPI_FP16_LM_HEAD=1 keeps the LM head in fp16 even under --weight-quant. This is
  // the one change in the engine that alters model output (quantization error lands directly in
  // the logits), so it needs an A/B switch for the perplexity gate.
  const bool force_fp16_lm_head = [] {
    const char* e = std::getenv("CPI_FP16_LM_HEAD");
    return e && *e == '1';
  }();
  if (lowbit_streaming_enabled(options_) && !force_fp16_lm_head) {
    const auto& lm_cfg = weights_.config();
    const int vocab = lm_cfg.vocab_size;
    const int lm_in = lm_cfg.hidden_size;
    CUDA_CHECK(cudaMalloc(&d_lm_head_i8_,
                          static_cast<std::size_t>(vocab) * static_cast<std::size_t>(lm_in)));
    CUDA_CHECK(cudaMalloc(&d_lm_head_i8_scales_, static_cast<std::size_t>(vocab) * sizeof(float)));
    kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_lm_head_),
                                                  d_lm_head_i8_, d_lm_head_i8_scales_, vocab, lm_in,
                                                  compute_stream_);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    lm_head_int8_ = true;
    if (options_.verbose) {
      std::cout << "[engine] lm_head=int8 (" << vocab << "x" << lm_in << ")\n";
    }
  }
  if (weights_.has_tensor("output.bias")) {
    const std::size_t out_bias_bytes = static_cast<std::size_t>(cfg.vocab_size) * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_lm_head_bias_, out_bias_bytes));
    CUDA_CHECK(cudaMemcpy(d_lm_head_bias_, weights_.tensor_data("output.bias"), out_bias_bytes,
                          cudaMemcpyHostToDevice));
  }

  // Detect TurboQuant metadata and load shared parameters.
  if (weights_.has_tensor("tq3_codebook") && weights_.has_tensor("tq3_signs_hidden")) {
    tq3_enabled_ = true;
    CUDA_CHECK(cudaMalloc(&d_tq3_codebook_, 8 * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(d_tq3_codebook_, weights_.tensor_data("tq3_codebook"), 8 * sizeof(__half),
                          cudaMemcpyHostToDevice));
    const std::size_t signs_bytes = static_cast<std::size_t>(cfg.hidden_size) * sizeof(int8_t);
    CUDA_CHECK(cudaMalloc(&d_tq3_signs_, signs_bytes));
    CUDA_CHECK(cudaMemcpy(d_tq3_signs_, weights_.tensor_data("tq3_signs_hidden"), signs_bytes,
                          cudaMemcpyHostToDevice));

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
        weights_.has_tensor("tq_qjl_dim") && weights_.has_tensor("tq_qjl_seed") &&
        weights_.has_tensor("tq_qjl_indices_hidden") && weights_.has_tensor("tq_qjl_signs_hidden");

    if (has_qjl_meta) {
      std::int32_t qjl_dim_i32 = 0;
      std::memcpy(&qjl_dim_i32, weights_.tensor_data("tq_qjl_dim"), sizeof(std::int32_t));
      tq_qjl_dim_ = static_cast<int>(qjl_dim_i32);
      if (tq_qjl_dim_ <= 0 || tq_qjl_dim_ > cfg.hidden_size) {
        CPI_THROW("invalid tq_qjl_dim in model metadata");
      }
      tq_qjl_words_ = (tq_qjl_dim_ + 31) / 32;
      const std::size_t qjl_idx_bytes =
          static_cast<std::size_t>(tq_qjl_dim_) * sizeof(std::int32_t);
      const std::size_t qjl_sign_bytes =
          static_cast<std::size_t>(tq_qjl_dim_) * sizeof(std::int8_t);
      CUDA_CHECK(cudaMalloc(&d_tq_qjl_indices_, qjl_idx_bytes));
      CUDA_CHECK(cudaMemcpy(d_tq_qjl_indices_, weights_.tensor_data("tq_qjl_indices_hidden"),
                            qjl_idx_bytes, cudaMemcpyHostToDevice));
      CUDA_CHECK(cudaMalloc(&d_tq_qjl_signs_, qjl_sign_bytes));
      CUDA_CHECK(cudaMemcpy(d_tq_qjl_signs_, weights_.tensor_data("tq_qjl_signs_hidden"),
                            qjl_sign_bytes, cudaMemcpyHostToDevice));
    }

    const std::string requested_mode = options_.tq_mode.empty() ? "auto" : options_.tq_mode;
    if (requested_mode != "auto" && requested_mode != "mse" && requested_mode != "prod") {
      CPI_THROW("invalid tq_mode option: expected auto|mse|prod");
    }
    if (requested_mode == "prod") {
      if (!has_qjl_meta) {
        CPI_THROW("--tq-mode=prod requested but Qprod metadata is missing from model");
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
  kv_quant_kbits_ = 4;
  kv_quant_vbits_ = 4;
  kv_quant_rot_ = false;
  if (const char* env = std::getenv("CPI_KV_QUANT")) {
    const std::string mode(env);
    if (mode == "4" || mode == "44") {
      kv_quant_kbits_ = 4; kv_quant_vbits_ = 4; kv_quant_rot_ = true; kv_int4_enabled_ = true;
    } else if (mode == "4nr") {
      kv_quant_kbits_ = 4; kv_quant_vbits_ = 4; kv_quant_rot_ = false; kv_int4_enabled_ = true;
    } else if (mode == "84") {
      kv_quant_kbits_ = 8; kv_quant_vbits_ = 4; kv_int4_enabled_ = true;
    } else if (mode == "8" || mode == "88") {
      kv_quant_kbits_ = 8; kv_quant_vbits_ = 8; kv_int4_enabled_ = true;
    } else if (mode != "0" && mode != "off") {
      CPI_THROW("CPI_KV_QUANT must be one of 4, 4nr, 84, 8, off");
    }
  }
  if (kv_int4_enabled_ && options_.paged_kv_cache) {
    CPI_THROW("quantized KV cache and paged_kv_cache are mutually exclusive");
  }
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (cfg.hidden_size / cfg.num_heads);
  if (kv_quant_rot_ && head_dim != 128) {
    // The R3 rotation kernels assume head_dim 128; fall back to unrotated K.
    kv_quant_rot_ = false;
  }
  if (const char* env = std::getenv("CPI_KV_SINK")) {
    kv_quant_sink_ = std::max(0, std::atoi(env));
  }
  if (const char* env = std::getenv("CPI_KV_WIN")) {
    kv_quant_win_ = std::max(0, std::atoi(env));
  }
  if (kv_int4_enabled_ && options_.paged_blocks) {
    const int bs = options_.paged_block_size > 0 ? options_.paged_block_size : 32;
    if (!kernels::paged_quant_attention_supported(cfg.num_heads, cfg.num_kv_heads, head_dim, bs)) {
      CPI_THROW(
          "quantized KV with --paged-blocks requires head_dim 128, a GQA group of 4..32 heads, "
          "and block size <= 32");
    }
  }
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
      CUDA_CHECK(cudaMalloc(&d_tq_qjl_x_bits_,
                            static_cast<std::size_t>(tq_qjl_words_) * sizeof(std::uint32_t)));
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
  CUDA_CHECK(cudaMalloc(
      &d_prefill_i8_, static_cast<std::size_t>(rows) * static_cast<std::size_t>(max_lowbit_cols)));
  CUDA_CHECK(cudaMalloc(&d_prefill_i8_scales_, static_cast<std::size_t>(rows) * sizeof(float)));
  // Per-group(32) activation scales for the perm8-g32 quant feeding the grouped int4 dp4a MLP.
  CUDA_CHECK(cudaMalloc(
      &d_prefill_perm8_scales_, static_cast<std::size_t>(rows) *
                                    ((static_cast<std::size_t>(max_lowbit_cols) + 31) / 32) *
                                    sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_ff3_, bytes_for_matrix(rows, hidden)));
  CUDA_CHECK(cudaMalloc(&d_logits_, static_cast<std::size_t>(cfg.vocab_size) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_argmax_, sizeof(int)));
  // Scratch for the two-phase argmax. Without it the greedy argmax runs as one BLOCK over the
  // whole vocab (one SM out of 170) and cost ~150 us/token on a 151936-wide head; 12% of all
  // GPU time, to read 608 KB that peak bandwidth delivers in 0.34 us.
  argmax_parts_ = kernels::argmax_partition_count(cfg.vocab_size);
  CUDA_CHECK(
      cudaMalloc(&d_argmax_part_val_, static_cast<std::size_t>(argmax_parts_) * sizeof(float)));
  CUDA_CHECK(
      cudaMalloc(&d_argmax_part_idx_, static_cast<std::size_t>(argmax_parts_) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_decode_position_, sizeof(int)));
  const int half_dim = head_dim / 2;
  const std::size_t rope_table_elems =
      static_cast<std::size_t>(options_.max_context) * static_cast<std::size_t>(half_dim);
  std::vector<float> rope_cos(rope_table_elems);
  std::vector<float> rope_sin(rope_table_elems);
  // Resolve effective RoPE theta: CLI override wins, then model file, then family default.
  const float eff_rope_theta =
      (options_.rope_theta > 0.0f) ? options_.rope_theta : cfg.effective_rope_theta();
  if (options_.verbose) {
    std::cout << "[engine] rope_theta=" << eff_rope_theta << " (source="
              << (options_.rope_theta > 0.0f
                      ? "cli"
                      : (cfg.rope_theta > 0.0f ? "model_file" : "family_default"))
              << ")\n";
  }

  // Physical per-layer KV capacity. Default = max_context (holds one sequence).
  // With --paged-blocks it can be enlarged (explicit override here; VRAM-derived
  // default is a follow-up) so the shared block pool holds many concurrent
  // sequences. Every stride into d_k_cache_/d_v_cache_ uses kv_capacity_tokens_,
  // so read/write layout stays consistent regardless of the value.
  kv_capacity_tokens_ = options_.max_context;
  if (options_.paged_blocks) {
    const int bs = options_.paged_block_size > 0 ? options_.paged_block_size : 32;
    if (const char* ov = std::getenv("CPI_KV_POOL_TOKENS")) {
      const int req = std::atoi(ov);
      if (req > kv_capacity_tokens_) kv_capacity_tokens_ = req;
    } else {
      // Auto-size the pool to free VRAM so continuous batching can hold many
      // concurrent sequences. This runs before the resident weight cache is
      // built (cache-copy phase), so cudaMemGetInfo still counts that VRAM as
      // free; subtract a conservative full-fp16 layer-cache reserve plus fixed
      // headroom (streaming staging, activations, cuBLAS, fragmentation). The
      // reserve is an upper bound (quantized caches are smaller), so we only
      // ever under-size the pool, never OOM the later weight load.
      std::size_t free_b = 0, total_b = 0;
      cudaMemGetInfo(&free_b, &total_b);
      const std::size_t per_layer_weight_b =
          (static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * hidden +  // wqkv
           static_cast<std::size_t>(hidden) * q_hidden +                  // wo
           static_cast<std::size_t>(2) * inter * hidden +                 // w1|w3
           static_cast<std::size_t>(hidden) * inter) *                    // w2
          sizeof(__half);
      const std::size_t weight_reserve_b =
          static_cast<std::size_t>(cfg.num_layers) * per_layer_weight_b;
      const std::size_t headroom_b = static_cast<std::size_t>(3) << 30;  // 3 GiB
      // Quantized KV stores head_dim*bits/8 bytes per element plus one fp16
      // scale per (token, head) for K and V each, so the same VRAM budget holds
      // proportionally more tokens (the capacity multiplier).
      const std::size_t per_token_kv_b =
          kv_int4_enabled_
              ? static_cast<std::size_t>(cfg.num_layers) *
                    static_cast<std::size_t>(cfg.num_kv_heads) *
                    (static_cast<std::size_t>(head_dim * (kv_quant_kbits_ + kv_quant_vbits_) / 8) +
                     2 * sizeof(__half))
              : static_cast<std::size_t>(cfg.num_layers) * static_cast<std::size_t>(kv_hidden) * 2 *
                    sizeof(__half);
      if (free_b > weight_reserve_b + headroom_b && per_token_kv_b > 0) {
        const std::size_t budget_b = free_b - weight_reserve_b - headroom_b;
        const long long auto_tokens = static_cast<long long>(budget_b / per_token_kv_b);
        const long long capped = std::min<long long>(auto_tokens, 1 << 20);  // cap 1M tok
        if (capped > kv_capacity_tokens_) kv_capacity_tokens_ = static_cast<int>(capped);
      }
    }
    kv_capacity_tokens_ = ((kv_capacity_tokens_ + bs - 1) / bs) * bs;  // whole blocks
    const int num_blocks = kv_capacity_tokens_ / bs;
    if (options_.verbose) {
      std::cout << "[engine] kv_pool_tokens=" << kv_capacity_tokens_ << " (max_context="
                << options_.max_context << ", " << num_blocks << " blocks)\n";
    }
    block_alloc_ = std::make_unique<BlockAllocator>(num_blocks);
    seq_blocks_ = std::make_unique<SequenceBlockTable>(block_alloc_.get(), bs);
    // Device block table for the paged decode-attention kernel. Phase 2c uses a
    // single contiguous sequence => identity mapping (logical chunk c -> block c),
    // so KV writes stay in place and paged reads are byte-identical to contiguous.
    // (Non-contiguous/multi-sequence tables come with continuous batching.)
    CUDA_CHECK(cudaMalloc(&d_block_table_, static_cast<std::size_t>(num_blocks) * sizeof(int)));
    std::vector<int> table(static_cast<std::size_t>(num_blocks));
    // Identity mapping (byte-identical). CPI_PAGED_SCRAMBLE reverses it: a
    // diagnostic that must corrupt output iff the paged gather is truly live
    // (writes are still contiguous), proving the paged read path is engaged.
    const bool scramble = std::getenv("CPI_PAGED_SCRAMBLE") != nullptr;
    for (int i = 0; i < num_blocks; ++i) {
      table[static_cast<std::size_t>(i)] = scramble ? (num_blocks - 1 - i) : i;
    }
    CUDA_CHECK(cudaMemcpy(d_block_table_, table.data(),
                          static_cast<std::size_t>(num_blocks) * sizeof(int),
                          cudaMemcpyHostToDevice));
    if (options_.verbose) {
      std::cout << "[engine] paged_blocks=on block_size=" << bs << " num_blocks=" << num_blocks
                << " (P3 phase 2c: paged decode attention, identity table, byte-identical)\n";
    }
  }

  // llama3 rope scaling (CPI_ROPE_SCALING=llama3, params overridable via
  // CPI_ROPE_SCALING_PARAMS="factor,low,high,orig_max"): Llama-3.1 trains at
  // 8192 and reaches its long context through frequency-banded scaling of the
  // inverse frequencies. Without it the model degrades beyond ~16k (needle
  // retrieval: 16k OK, 30k fails). Opt-in because the model file does not say
  // whether a checkpoint uses it (Llama-3.0 shares theta 500000 and does not).
  const char* rs_env = std::getenv("CPI_ROPE_SCALING");
  const bool llama3_scaling = rs_env && std::string(rs_env) == "llama3";
  float rs_factor = 8.0f, rs_low = 1.0f, rs_high = 4.0f, rs_orig = 8192.0f;
  if (const char* p = std::getenv("CPI_ROPE_SCALING_PARAMS")) {
    std::sscanf(p, "%f,%f,%f,%f", &rs_factor, &rs_low, &rs_high, &rs_orig);
  }
  if (options_.verbose && llama3_scaling) {
    std::cout << "[engine] rope_scaling=llama3 factor=" << rs_factor << " low=" << rs_low
              << " high=" << rs_high << " orig_max=" << rs_orig << "\n";
  }
  for (int pos = 0; pos < options_.max_context; ++pos) {
    for (int pair = 0; pair < half_dim; ++pair) {
      float theta =
          std::pow(eff_rope_theta, -2.0f * static_cast<float>(pair) / static_cast<float>(head_dim));
      if (llama3_scaling) {
        const float wavelen = 2.0f * 3.14159265358979323846f / theta;
        if (wavelen > rs_orig / rs_low) {
          theta /= rs_factor;
        } else if (wavelen >= rs_orig / rs_high) {
          const float smooth = (rs_orig / wavelen - rs_low) / (rs_high - rs_low);
          theta = (1.0f - smooth) * theta / rs_factor + smooth * theta;
        }
      }
      const float angle = static_cast<float>(pos) * theta;
      rope_cos[static_cast<std::size_t>(pos) * static_cast<std::size_t>(half_dim) +
               static_cast<std::size_t>(pair)] = std::cos(angle);
      rope_sin[static_cast<std::size_t>(pos) * static_cast<std::size_t>(half_dim) +
               static_cast<std::size_t>(pair)] = std::sin(angle);
    }
  }
  CUDA_CHECK(cudaMalloc(&d_rope_cos_, rope_table_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_rope_sin_, rope_table_elems * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_rope_cos_, rope_cos.data(), rope_table_elems * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_rope_sin_, rope_sin.data(), rope_table_elems * sizeof(float),
                        cudaMemcpyHostToDevice));

  attn_chunk_capacity_ = std::max(1, (options_.max_context + 31) / 32);
  CUDA_CHECK(cudaMalloc(&d_attn_chunk_m_, static_cast<std::size_t>(cfg.num_heads) *
                                              static_cast<std::size_t>(attn_chunk_capacity_) *
                                              sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_attn_chunk_l_, static_cast<std::size_t>(cfg.num_heads) *
                                              static_cast<std::size_t>(attn_chunk_capacity_) *
                                              sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_attn_chunk_o_, static_cast<std::size_t>(cfg.num_heads) *
                                              static_cast<std::size_t>(attn_chunk_capacity_) *
                                              static_cast<std::size_t>(head_dim) * sizeof(float)));

  if (cfg.is_moe()) {
    const int experts = std::max(1, cfg.num_local_experts);
    const int top_k =
        std::max(1, std::min(cfg.effective_experts_per_tok(), experts));
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
    const std::size_t router_logits_bytes = static_cast<std::size_t>(experts) * sizeof(__half);
    const std::size_t router_q8_bytes =
        static_cast<std::size_t>(experts) * static_cast<std::size_t>(hidden);
    const std::size_t router_q4_bytes =
        static_cast<std::size_t>(experts) * static_cast<std::size_t>((hidden + 1) / 2);

    CUDA_CHECK(cudaMalloc(&d_moe_router_w_, router_fp16_bytes));
    CUDA_CHECK(cudaMalloc(&d_moe_router_logits_, router_logits_bytes));
    CUDA_CHECK(cudaMalloc(&d_moe_router_w_q_, std::max(router_q8_bytes, router_q4_bytes)));
    CUDA_CHECK(
        cudaMalloc(&d_moe_router_scales_, static_cast<std::size_t>(experts) * sizeof(float)));

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

  const std::size_t kv_bytes = static_cast<std::size_t>(cfg.num_layers) *
                               static_cast<std::size_t>(kv_capacity_tokens_) *
                               static_cast<std::size_t>(kv_hidden) * sizeof(__half);
  if (kv_int4_enabled_) {
    // Quantized KV: 4- or 8-bit rows + per-head fp16 scales; fp16 KV buffers not
    // allocated. Strided by kv_capacity_tokens_ so the enlarged paged pool works.
    const std::size_t rows = static_cast<std::size_t>(cfg.num_layers) *
                             static_cast<std::size_t>(kv_capacity_tokens_) *
                             static_cast<std::size_t>(cfg.num_kv_heads);
    const std::size_t k_bytes = rows * static_cast<std::size_t>(head_dim * kv_quant_kbits_ / 8);
    const std::size_t v_bytes = rows * static_cast<std::size_t>(head_dim * kv_quant_vbits_ / 8);
    const std::size_t sc_bytes = rows * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_k_cache_i4_, k_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_cache_i4_, v_bytes));
    CUDA_CHECK(cudaMalloc(&d_k_scales_, sc_bytes));
    CUDA_CHECK(cudaMalloc(&d_v_scales_, sc_bytes));
    CUDA_CHECK(cudaMemset(d_k_cache_i4_, 0, k_bytes));
    CUDA_CHECK(cudaMemset(d_v_cache_i4_, 0, v_bytes));
    CUDA_CHECK(cudaMemset(d_k_scales_, 0, sc_bytes));
    CUDA_CHECK(cudaMemset(d_v_scales_, 0, sc_bytes));
    // fp16 sink and recent-window side buffers (small: (sink+win) tokens/layer).
    const std::size_t sink_bytes = static_cast<std::size_t>(cfg.num_layers) *
                                   static_cast<std::size_t>(kv_quant_sink_) *
                                   static_cast<std::size_t>(kv_hidden) * sizeof(__half);
    const std::size_t ring_bytes = static_cast<std::size_t>(cfg.num_layers) *
                                   static_cast<std::size_t>(kv_quant_win_) *
                                   static_cast<std::size_t>(kv_hidden) * sizeof(__half);
    if (sink_bytes > 0) {
      CUDA_CHECK(cudaMalloc(&d_kv_sink_k_, sink_bytes));
      CUDA_CHECK(cudaMalloc(&d_kv_sink_v_, sink_bytes));
      CUDA_CHECK(cudaMemset(d_kv_sink_k_, 0, sink_bytes));
      CUDA_CHECK(cudaMemset(d_kv_sink_v_, 0, sink_bytes));
    }
    if (ring_bytes > 0) {
      CUDA_CHECK(cudaMalloc(&d_kv_ring_k_, ring_bytes));
      CUDA_CHECK(cudaMalloc(&d_kv_ring_v_, ring_bytes));
      CUDA_CHECK(cudaMemset(d_kv_ring_k_, 0, ring_bytes));
      CUDA_CHECK(cudaMemset(d_kv_ring_v_, 0, ring_bytes));
    }
    if (options_.verbose) {
      std::cout << "[engine] kv_quant=K" << kv_quant_kbits_ << "V" << kv_quant_vbits_
                << (kv_quant_rot_ ? "+rot" : "") << " sink=" << kv_quant_sink_
                << " win=" << kv_quant_win_ << "  KV VRAM: "
                << (k_bytes + v_bytes + sc_bytes * 2 + (sink_bytes + ring_bytes) * 2) /
                       (1024 * 1024)
                << " MiB (vs " << (kv_bytes * 2) / (1024 * 1024) << " MiB fp16)\n";
    }
  } else if (options_.paged_kv_cache) {
    const std::size_t stage_bytes = static_cast<std::size_t>(options_.max_context) *
                                    static_cast<std::size_t>(kv_hidden) * sizeof(__half);
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
      CUDA_CHECK(
          cudaMalloc(&iw.w1, static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden)));
      CUDA_CHECK(
          cudaMalloc(&iw.w2, static_cast<std::size_t>(hidden) * static_cast<std::size_t>(inter)));
      CUDA_CHECK(
          cudaMalloc(&iw.w3, static_cast<std::size_t>(inter) * static_cast<std::size_t>(hidden)));
      CUDA_CHECK(cudaMalloc(&iw.s_w1, static_cast<std::size_t>(inter) * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&iw.s_w2, static_cast<std::size_t>(hidden) * sizeof(float)));
      CUDA_CHECK(cudaMalloc(&iw.s_w3, static_cast<std::size_t>(inter) * sizeof(float)));
    }
  }
}

void LlamaEngine::reset_kv_cache() {
  // The KV cache is being wiped, so any resident prompt prefix is no longer
  // valid for reuse (callers like inspect_next_logits / verify also reset here).
  resident_prefix_.clear();
  if (seq_blocks_) seq_blocks_->clear();  // release paged blocks alongside the KV wipe
  // The paged shared-prefix cache references KV blocks that this wipe invalidates;
  // drop it so no later request adopts stale blocks. It lives in the shared scheduler now,
  // which may not exist yet (it is built on first admit).
  if (scheduler_) scheduler_->clear_prefix_cache();
  const auto& cfg = weights_.config();
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (cfg.hidden_size / cfg.num_heads);
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const std::size_t kv_bytes = static_cast<std::size_t>(cfg.num_layers) *
                               static_cast<std::size_t>(kv_capacity_tokens_) *
                               static_cast<std::size_t>(kv_hidden) * sizeof(__half);
  if (kv_int4_enabled_) {
    const std::size_t rows = static_cast<std::size_t>(cfg.num_layers) *
                             static_cast<std::size_t>(kv_capacity_tokens_) *
                             static_cast<std::size_t>(cfg.num_kv_heads);
    const std::size_t k_bytes = rows * static_cast<std::size_t>(head_dim * kv_quant_kbits_ / 8);
    const std::size_t v_bytes = rows * static_cast<std::size_t>(head_dim * kv_quant_vbits_ / 8);
    const std::size_t sc_bytes = rows * sizeof(__half);
    CUDA_CHECK(cudaMemset(d_k_cache_i4_, 0, k_bytes));
    CUDA_CHECK(cudaMemset(d_v_cache_i4_, 0, v_bytes));
    CUDA_CHECK(cudaMemset(d_k_scales_, 0, sc_bytes));
    CUDA_CHECK(cudaMemset(d_v_scales_, 0, sc_bytes));
    // The fp16 sink/ring quality buffers are deliberately NOT cleared: the attention
    // only ever reads sink positions t < min(sink_n, seq_len) and ring positions
    // >= seq_len - win_n of the owning sequence's slot, all of which that sequence's
    // own prefill + decode freshly wrote (prefix adoption is disabled under this
    // tier). At batched slot counts these buffers are ~GB-scale, and this reset runs
    // per generation on the single-sequence path -- clearing them was measured at
    // ~180ms/reset (the B=48/64 batch-bench collapse).
    return;
  }
  if (options_.paged_kv_cache) {
    const std::size_t stage_bytes = static_cast<std::size_t>(options_.max_context) *
                                    static_cast<std::size_t>(kv_hidden) * sizeof(__half);
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
