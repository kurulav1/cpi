#include "engine/llama_engine.hpp"
#include "llama_engine_internal.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cuda_fp16.h>
#include "common.hpp"
#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"
namespace engine {
namespace {
std::size_t bytes_for_matrix(int rows, int cols) {
  return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(__half);
}
}  // namespace
void LlamaEngine::forward_decode_layers(int token, int position) {
  const auto& cfg = weights_.config();
  const int hidden = cfg.hidden_size;
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : hidden;
  const int inter = cfg.intermediate_size;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (cfg.hidden_size / cfg.num_heads);
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const auto attention_bounds = [&](int layer, int pos) {
    const int full = pos + 1;
    switch (cfg.attention_kind_for_layer(layer)) {
      case model::AttentionKind::Full:
        return std::pair<int, int>{0, full};
      case model::AttentionKind::SlidingWindow: {
        const int window = cfg.attention_window_for_layer(layer);
        const int seq_len = (window > 0) ? std::min(window, full) : full;
        const int start = full - seq_len;
        return std::pair<int, int>{start, seq_len};
      }
      case model::AttentionKind::Linear:
        LLAMA_ENGINE_THROW("native CUDA runtime does not support linear-attention layers yet");
      default:
        LLAMA_ENGINE_THROW("unknown attention kind in model metadata");
    }
  };
  const bool can_use_dp4a_decode = ((hidden & 3) == 0) && ((inter & 3) == 0);
  const bool resident_fast_path = cached_layer_count_ == cfg.num_layers && !options_.paged_kv_cache;
  // Use the int8 path whenever int8 projections are active (QKV/wo freed as fp16)
  // or when the full MLP is also stored as int8.
  const bool resident_all_packed_mlp =
      resident_fast_path && (cached_int8_proj_enabled_ || (cached_int8_mlp_enabled_ && static_cast<int>(layer_cache_i8_.size()) == cfg.num_layers));
  const bool phase_profile = options_.profile_decode_phases;
  cublasLtHandle_t matmul_lt = cublas_lt_;
  const auto run_profiled = [&](double& acc, const auto& fn) {
    if (!phase_profile) {
      fn();
      return;
    }
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start, compute_stream_));
    fn();
    CUDA_CHECK(cudaEventRecord(stop, compute_stream_));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaEventDestroy(start));
    acc += static_cast<double>(ms);
  };
  if (position >= options_.max_context) {
    LLAMA_ENGINE_THROW("context length exceeded max_context");
  }
  // Helper: compute INT4 layer-local pointers and dispatch KV store + attention.
  // Captures layer, position, kv_hidden, head_dim, cfg from the surrounding scope.
  const auto do_kv_int4 = [&](int layer) {
    const auto [attn_start, attn_seq_len] = attention_bounds(layer, position);
    const int packed_per_head = head_dim / 2;
    const std::size_t i4_stride = static_cast<std::size_t>(options_.max_context) *
                                   static_cast<std::size_t>(cfg.num_kv_heads) *
                                   static_cast<std::size_t>(packed_per_head);
    const std::size_t sc_stride = static_cast<std::size_t>(options_.max_context) *
                                   static_cast<std::size_t>(cfg.num_kv_heads);
    auto* ki4 = d_k_cache_i4_ + static_cast<std::size_t>(layer) * i4_stride;
    auto* vi4 = d_v_cache_i4_ + static_cast<std::size_t>(layer) * i4_stride;
    auto* ks  = d_k_scales_   + static_cast<std::size_t>(layer) * sc_stride;
    auto* vs  = d_v_scales_   + static_cast<std::size_t>(layer) * sc_stride;
    kernels::launch_store_kv_int4(static_cast<const __half*>(d_k_),
                                   static_cast<const __half*>(d_v_),
                                   ki4, vi4, ks, vs,
                                   position, cfg.num_kv_heads, head_dim,
                                   options_.max_context, compute_stream_);
    kernels::launch_attention_step_int4(static_cast<const __half*>(d_q_),
                                        ki4 + static_cast<std::size_t>(attn_start) *
                                                  static_cast<std::size_t>(cfg.num_kv_heads) *
                                                  static_cast<std::size_t>(packed_per_head),
                                        vi4 + static_cast<std::size_t>(attn_start) *
                                                  static_cast<std::size_t>(cfg.num_kv_heads) *
                                                  static_cast<std::size_t>(packed_per_head),
                                        ks + static_cast<std::size_t>(attn_start) * static_cast<std::size_t>(cfg.num_kv_heads),
                                        vs + static_cast<std::size_t>(attn_start) * static_cast<std::size_t>(cfg.num_kv_heads),
                                        static_cast<__half*>(d_att_),
                                        attn_seq_len,
                                        cfg.num_heads, cfg.num_kv_heads, head_dim,
                                        compute_stream_,
                                        d_attn_chunk_m_, d_attn_chunk_l_, d_attn_chunk_o_,
                                        attn_chunk_capacity_, !options_.disable_split_attention);
  };
  if (cfg.is_moe()) {
    const int experts = std::max(1, cfg.num_local_experts);
    const int top_k = std::max(1, std::min(cfg.num_experts_per_tok > 0 ? cfg.num_experts_per_tok : 2, experts));
    const int expert_inter = cfg.effective_expert_intermediate_size() > 0
        ? cfg.effective_expert_intermediate_size()
        : inter;
    LayerDeviceWeights* lw = &streaming_layer_weights_[0];
    auto* moe_router_logits = static_cast<__half*>(d_moe_router_logits_);
    auto* moe_ff_gate = static_cast<__half*>(d_prefill_ff1_);
    auto* moe_ff_up = static_cast<__half*>(d_prefill_ff2_);
    auto* moe_accum = static_cast<__half*>(d_att_);
    std::vector<float> h_scale_staging;
    std::vector<int> h_topk_idx(static_cast<std::size_t>(top_k), 0);
    std::vector<float> h_topk_prob(static_cast<std::size_t>(top_k), 0.0f);
    last_benchmark_stats_.moe_topk_layers = cfg.num_layers;
    last_benchmark_stats_.moe_topk_k = top_k;
    last_benchmark_stats_.moe_topk_indices.assign(
        static_cast<std::size_t>(cfg.num_layers) * static_cast<std::size_t>(top_k),
        -1);
    last_benchmark_stats_.moe_topk_probs.assign(
        static_cast<std::size_t>(cfg.num_layers) * static_cast<std::size_t>(top_k),
        0.0f);
    last_benchmark_stats_.moe_quant_mode = "fp16";

    const auto promote_moe_quant_mode = [&](const char* mode) {
      if (std::strcmp(mode, "int4") == 0) {
        last_benchmark_stats_.moe_quant_mode = "int4";
        return;
      }
      if (std::strcmp(mode, "int8") == 0 && last_benchmark_stats_.moe_quant_mode != "int4") {
        last_benchmark_stats_.moe_quant_mode = "int8";
      }
    };

    CUDA_CHECK(cudaMemcpyAsync(d_token_id_, &token, sizeof(int), cudaMemcpyHostToDevice, compute_stream_));
    kernels::launch_embedding_lookup(static_cast<const __half*>(d_tok_embeddings_),
                                     d_token_id_,
                                     static_cast<__half*>(d_x_),
                                     1,
                                     hidden,
                                     compute_stream_);

    const auto copy_fp16 = [&](const std::string& name, void* dst, std::size_t bytes) {
      if (!weights_.has_tensor(name)) {
        LLAMA_ENGINE_THROW("missing tensor: " + name);
      }
      CUDA_CHECK(cudaMemcpyAsync(dst, weights_.tensor_data(name), bytes, cudaMemcpyHostToDevice, compute_stream_));
    };
    const auto copy_optional_fp16 = [&](const std::string& name, void* dst, std::size_t bytes) {
      if (weights_.has_tensor(name)) {
        CUDA_CHECK(cudaMemcpyAsync(dst, weights_.tensor_data(name), bytes, cudaMemcpyHostToDevice, compute_stream_));
      } else {
        CUDA_CHECK(cudaMemsetAsync(dst, 0, bytes, compute_stream_));
      }
    };

    const auto copy_row_scales = [&](const std::string& scale_name, int rows, float* dst) {
      if (!weights_.has_tensor(scale_name)) {
        LLAMA_ENGINE_THROW("missing quant scales: " + scale_name);
      }
      const auto* src = reinterpret_cast<const float*>(weights_.tensor_data(scale_name));
      const std::size_t scale_count = weights_.tensor_bytes(scale_name) / sizeof(float);
      if (scale_count == static_cast<std::size_t>(rows)) {
        CUDA_CHECK(cudaMemcpyAsync(dst, src, static_cast<std::size_t>(rows) * sizeof(float), cudaMemcpyHostToDevice, compute_stream_));
        return;
      }
      if (scale_count == 1) {
        h_scale_staging.assign(static_cast<std::size_t>(rows), src[0]);
        CUDA_CHECK(cudaMemcpyAsync(dst,
                                   h_scale_staging.data(),
                                   static_cast<std::size_t>(rows) * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   compute_stream_));
        return;
      }
      LLAMA_ENGINE_THROW("invalid quant scale size for " + scale_name);
    };

    const auto matvec_device_weight = [&](const std::string& base,
                                          int rows,
                                          int cols,
                                          const __half* x,
                                          __half* y,
                                          void* fp16_weight_dst,
                                          std::int8_t* packed_weight_dst,
                                          float* scale_dst) {
      const std::string i8name = int8_tensor_name(base);
      const std::string i4name = int4_tensor_name(base);
      const std::string sname = quant_scale_name(base);
      const bool has_fp16 = weights_.has_tensor(base);
      const bool has_scales = weights_.has_tensor(sname);
      const bool has_i8 = has_scales && weights_.has_tensor(i8name);
      const bool has_i4 = has_scales && weights_.has_tensor(i4name);
      const bool prefer_lowbit = lowbit_streaming_enabled(options_);
      const int quant_bits = clamp_streaming_quant_bits(options_.streaming_quant_bits);

      bool use_i4 = false;
      bool use_i8 = false;
      if (prefer_lowbit) {
        if (quant_bits == 4) {
          use_i4 = has_i4;
          use_i8 = !use_i4 && has_i8;
        } else {
          use_i8 = has_i8;
          use_i4 = !use_i8 && has_i4;
        }
      } else if (!has_fp16) {
        use_i8 = has_i8;
        use_i4 = !use_i8 && has_i4;
      }

      if (!use_i8 && !use_i4) {
        if (!has_fp16) {
          LLAMA_ENGINE_THROW("missing MoE tensor: " + base);
        }
        copy_fp16(base, fp16_weight_dst, bytes_for_matrix(rows, cols));
        kernels::launch_rowmajor_half_gemv_f16(static_cast<const __half*>(fp16_weight_dst),
                                               x,
                                               y,
                                               rows,
                                               cols,
                                               compute_stream_);
        return;
      }

      if (!has_scales) {
        LLAMA_ENGINE_THROW("missing MoE tensor: " + base);
      }
      copy_row_scales(sname, rows, scale_dst);
      if (use_i8) {
        promote_moe_quant_mode("int8");
        CUDA_CHECK(cudaMemcpyAsync(packed_weight_dst,
                                   weights_.tensor_data(i8name),
                                   static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols),
                                   cudaMemcpyHostToDevice,
                                   compute_stream_));
        kernels::launch_weight_only_int8_matvec(packed_weight_dst,
                                                scale_dst,
                                                x,
                                                y,
                                                rows,
                                                cols,
                                                compute_stream_);
        return;
      }
      const int packed_cols = (cols + 1) / 2;
      promote_moe_quant_mode("int4");
      CUDA_CHECK(cudaMemcpyAsync(packed_weight_dst,
                                 weights_.tensor_data(i4name),
                                 static_cast<std::size_t>(rows) * static_cast<std::size_t>(packed_cols),
                                 cudaMemcpyHostToDevice,
                                 compute_stream_));
      kernels::launch_weight_only_int4_matvec(packed_weight_dst,
                                              scale_dst,
                                              x,
                                              y,
                                              rows,
                                              cols,
                                              compute_stream_);
    };

    for (int layer = 0; layer < cfg.num_layers; ++layer) {
      const std::string p = "layers." + std::to_string(layer);

      copy_fp16(p + ".attention_norm.weight", lw->norm_att, bytes_for_matrix(1, hidden));
      copy_fp16(p + ".ffn_norm.weight", lw->norm_ffn, bytes_for_matrix(1, hidden));
      copy_optional_fp16(p + ".attention_norm.bias", lw->norm_att_bias, bytes_for_matrix(1, hidden));
      copy_optional_fp16(p + ".ffn_norm.bias", lw->norm_ffn_bias, bytes_for_matrix(1, hidden));
      copy_fp16(p + ".attention.wo", lw->wo, bytes_for_matrix(hidden, q_hidden));
      copy_optional_fp16(p + ".attention.bo", lw->bo, bytes_for_matrix(1, hidden));
      auto* wqkv_base = static_cast<__half*>(lw->wqkv);
      copy_fp16(p + ".attention.wq", wqkv_base, bytes_for_matrix(q_hidden, hidden));
      copy_fp16(p + ".attention.wk",
                wqkv_base + static_cast<std::size_t>(q_hidden) * static_cast<std::size_t>(hidden),
                bytes_for_matrix(kv_hidden, hidden));
      copy_fp16(p + ".attention.wv",
                wqkv_base + static_cast<std::size_t>(q_hidden + kv_hidden) * static_cast<std::size_t>(hidden),
                bytes_for_matrix(kv_hidden, hidden));
      if (cfg.has_qkv_bias && lw->bqkv && weights_.has_tensor(p + ".attention.bqkv")) {
        copy_fp16(p + ".attention.bqkv", lw->bqkv, bytes_for_matrix(1, q_hidden + 2 * kv_hidden));
      }

      launch_norm(d_x_, lw->norm_att, lw->norm_att_bias, d_x_norm_, 1, hidden);

      resident_projection_half(lw->wqkv,
                               d_x_norm_,
                               d_q_,
                               q_hidden + 2 * kv_hidden,
                               hidden,
                               resident_qkv_warps_,
                               resident_qkv_tile_pairs_,
                               resident_qkv_rows_per_warp_);
      if (lw->bqkv && cfg.has_qkv_bias && weights_.has_tensor(p + ".attention.bqkv")) {
        kernels::launch_add_inplace(static_cast<__half*>(d_q_),
                                    static_cast<const __half*>(lw->bqkv),
                                    q_hidden + 2 * kv_hidden,
                                    compute_stream_);
      }
      kernels::launch_rope_inplace_table(static_cast<__half*>(d_q_),
                                         static_cast<__half*>(d_k_),
                                         cfg.num_heads,
                                         cfg.num_kv_heads,
                                         head_dim,
                                         position,
                                         d_rope_cos_,
                                         d_rope_sin_,
                                         compute_stream_);

      if (kv_int4_enabled_) {
        do_kv_int4(layer);
      } else {
        const auto [attn_start, attn_seq_len] = attention_bounds(layer, position);
        const std::size_t kv_bytes = bytes_for_matrix(1, kv_hidden);
        const std::size_t layer_stride = static_cast<std::size_t>(options_.max_context) * static_cast<std::size_t>(kv_hidden);
        __half* k_layer = nullptr;
        __half* v_layer = nullptr;
        if (options_.paged_kv_cache) {
          auto* h_k_layer = static_cast<__half*>(h_k_cache_) + static_cast<std::size_t>(layer) * layer_stride;
          auto* h_v_layer = static_cast<__half*>(h_v_cache_) + static_cast<std::size_t>(layer) * layer_stride;
          CUDA_CHECK(cudaMemcpyAsync(h_k_layer + static_cast<std::size_t>(position) * kv_hidden,
                                     d_k_, kv_bytes, cudaMemcpyDeviceToHost, compute_stream_));
          CUDA_CHECK(cudaMemcpyAsync(h_v_layer + static_cast<std::size_t>(position) * kv_hidden,
                                     d_v_, kv_bytes, cudaMemcpyDeviceToHost, compute_stream_));
          const std::size_t window_bytes =
              static_cast<std::size_t>(attn_seq_len) * static_cast<std::size_t>(kv_hidden) * sizeof(__half);
          CUDA_CHECK(cudaMemcpyAsync(d_k_cache_,
                                     h_k_layer + static_cast<std::size_t>(attn_start) * kv_hidden,
                                     window_bytes,
                                     cudaMemcpyHostToDevice,
                                     compute_stream_));
          CUDA_CHECK(cudaMemcpyAsync(d_v_cache_,
                                     h_v_layer + static_cast<std::size_t>(attn_start) * kv_hidden,
                                     window_bytes,
                                     cudaMemcpyHostToDevice,
                                     compute_stream_));
          k_layer = static_cast<__half*>(d_k_cache_);
          v_layer = static_cast<__half*>(d_v_cache_);
          kernels::launch_attention_step(static_cast<const __half*>(d_q_),
                                         k_layer,
                                         v_layer,
                                         static_cast<__half*>(d_att_),
                                         attn_seq_len,
                                         cfg.num_heads,
                                         cfg.num_kv_heads,
                                         head_dim,
                                         compute_stream_,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         0,
                                         !options_.disable_split_attention);
        } else {
          auto* k_store = static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride;
          auto* v_store = static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride;
          CUDA_CHECK(cudaMemcpyAsync(k_store + static_cast<std::size_t>(position) * kv_hidden,
                                     d_k_, kv_bytes, cudaMemcpyDeviceToDevice, compute_stream_));
          CUDA_CHECK(cudaMemcpyAsync(v_store + static_cast<std::size_t>(position) * kv_hidden,
                                     d_v_, kv_bytes, cudaMemcpyDeviceToDevice, compute_stream_));
          k_layer = k_store + static_cast<std::size_t>(attn_start) * static_cast<std::size_t>(kv_hidden);
          v_layer = v_store + static_cast<std::size_t>(attn_start) * static_cast<std::size_t>(kv_hidden);
          kernels::launch_attention_step(static_cast<const __half*>(d_q_),
                                         k_layer,
                                         v_layer,
                                         static_cast<__half*>(d_att_),
                                         attn_seq_len,
                                         cfg.num_heads,
                                         cfg.num_kv_heads,
                                         head_dim,
                                         compute_stream_,
                                         d_attn_chunk_m_,
                                         d_attn_chunk_l_,
                                         d_attn_chunk_o_,
                                         attn_chunk_capacity_,
                                         !options_.disable_split_attention);
        }
      }

      resident_projection_half(lw->wo,
                               d_att_,
                               d_ff3_,
                               hidden,
                               q_hidden,
                               resident_wo_warps_,
                               resident_wo_tile_pairs_,
                               resident_wo_rows_per_warp_);
      maybe_add_half_bias(d_ff3_, lw->bo, 1, hidden);
      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  static_cast<const __half*>(d_ff3_),
                                  hidden,
                                  compute_stream_);

      launch_norm(d_x_, lw->norm_ffn, lw->norm_ffn_bias, d_x_norm_, 1, hidden);

      run_profiled(last_benchmark_stats_.decode_moe_router_ms, [&] {
        matvec_device_weight(p + ".feed_forward.router",
                             experts,
                             hidden,
                             static_cast<const __half*>(d_x_norm_),
                             moe_router_logits,
                             d_moe_router_w_,
                             d_moe_router_w_q_,
                             d_moe_router_scales_);
        kernels::launch_moe_router_topk_softmax(moe_router_logits,
                                                experts,
                                                top_k,
                                                d_moe_topk_idx_,
                                                d_moe_topk_prob_,
                                                compute_stream_);
      });
      CUDA_CHECK(cudaMemcpyAsync(h_topk_idx.data(),
                                 d_moe_topk_idx_,
                                 static_cast<std::size_t>(top_k) * sizeof(int),
                                 cudaMemcpyDeviceToHost,
                                 compute_stream_));
      CUDA_CHECK(cudaMemcpyAsync(h_topk_prob.data(),
                                 d_moe_topk_prob_,
                                 static_cast<std::size_t>(top_k) * sizeof(float),
                                 cudaMemcpyDeviceToHost,
                                 compute_stream_));
      CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
      for (int k = 0; k < top_k; ++k) {
        const std::size_t flat =
            static_cast<std::size_t>(layer) * static_cast<std::size_t>(top_k) +
            static_cast<std::size_t>(k);
        last_benchmark_stats_.moe_topk_indices[flat] =
            h_topk_idx[static_cast<std::size_t>(k)];
        last_benchmark_stats_.moe_topk_probs[flat] =
            h_topk_prob[static_cast<std::size_t>(k)];
      }

      CUDA_CHECK(cudaMemsetAsync(moe_accum,
                                 0,
                                 static_cast<std::size_t>(hidden) * sizeof(__half),
                                 compute_stream_));
      for (int k = 0; k < top_k; ++k) {
        int expert_idx = h_topk_idx[static_cast<std::size_t>(k)];
        if (expert_idx < 0 || expert_idx >= experts) {
          continue;
        }
        const float gate = h_topk_prob[static_cast<std::size_t>(k)];
        if (gate <= 0.0f) {
          continue;
        }
        const std::string ebase = p + ".feed_forward.experts." + std::to_string(expert_idx);
        run_profiled(last_benchmark_stats_.decode_moe_expert_ms, [&] {
          matvec_device_weight(ebase + ".w1",
                               expert_inter,
                               hidden,
                               static_cast<const __half*>(d_x_norm_),
                               moe_ff_gate,
                               d_moe_w1_,
                               d_moe_w1_q_,
                               d_moe_s_w1_);
          matvec_device_weight(ebase + ".w3",
                               expert_inter,
                               hidden,
                               static_cast<const __half*>(d_x_norm_),
                               moe_ff_up,
                               d_moe_w3_,
                               d_moe_w3_q_,
                               d_moe_s_w3_);
          kernels::launch_silu_mul(moe_ff_gate,
                                   moe_ff_up,
                                   moe_ff_up,
                                   expert_inter,
                                   compute_stream_);
          matvec_device_weight(ebase + ".w2",
                               hidden,
                               expert_inter,
                               moe_ff_up,
                               static_cast<__half*>(d_ff3_),
                               d_moe_w2_,
                               d_moe_w2_q_,
                               d_moe_s_w2_);
        });
        run_profiled(last_benchmark_stats_.decode_moe_merge_ms, [&] {
          kernels::launch_scale_add_inplace(moe_accum,
                                            static_cast<const __half*>(d_ff3_),
                                            hidden,
                                            gate,
                                            compute_stream_);
        });
      }

      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  moe_accum,
                                  hidden,
                                  compute_stream_);
      enforce_host_resource_limits("decode.layer_end");
    }
    return;
  }
  CUDA_CHECK(cudaMemcpyAsync(d_token_id_, &token, sizeof(int), cudaMemcpyHostToDevice, compute_stream_));
  kernels::launch_embedding_lookup(static_cast<const __half*>(d_tok_embeddings_),
                                   d_token_id_,
                                   static_cast<__half*>(d_x_),
                                   1,
                                   hidden,
                                   compute_stream_);
  const auto run_layer = [&](int layer, const LayerDeviceWeights* lw, const LayerDeviceInt8Weights* lw_i8) {
    launch_norm(d_x_, lw->norm_att, lw->norm_att_bias, d_x_norm_, 1, hidden);
    if (lw_i8 && lw_i8->wqkv) {
      kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_x_norm_),
                                                    static_cast<std::int8_t*>(d_prefill_i8_),
                                                    static_cast<float*>(d_prefill_i8_scales_),
                                                    1,
                                                    hidden,
                                                    compute_stream_);
      if (lw_i8->proj_int4) {
        kernels::launch_weight_only_int4_matvec_dp4a(lw_i8->wqkv,
                                                     lw_i8->s_wqkv,
                                                     static_cast<const std::int8_t*>(d_prefill_i8_),
                                                     static_cast<const float*>(d_prefill_i8_scales_),
                                                     static_cast<__half*>(d_q_),
                                                     q_hidden + 2 * kv_hidden,
                                                     hidden,
                                                     compute_stream_,
                                                     resident_int8_qkv_warps_,
                                                     resident_int8_qkv_tile_packed4_,
                                                     resident_int8_qkv_warps_per_row_);
      } else {
        kernels::launch_weight_only_int8_matvec_dp4a(lw_i8->wqkv,
                                                     lw_i8->s_wqkv,
                                                     static_cast<const std::int8_t*>(d_prefill_i8_),
                                                     static_cast<const float*>(d_prefill_i8_scales_),
                                                     static_cast<__half*>(d_q_),
                                                     q_hidden + 2 * kv_hidden,
                                                     hidden,
                                                     compute_stream_,
                                                     resident_int8_qkv_warps_,
                                                     resident_int8_qkv_tile_packed4_,
                                                     resident_int8_qkv_warps_per_row_);
      }
    } else {
      resident_projection_half(lw->wqkv,
                               d_x_norm_,
                               d_q_,
                               q_hidden + 2 * kv_hidden,
                               hidden,
                               resident_qkv_warps_,
                               resident_qkv_tile_pairs_,
                               resident_qkv_rows_per_warp_);
    }
    if (lw->bqkv) {
      kernels::launch_add_inplace(static_cast<__half*>(d_q_),
                                  static_cast<const __half*>(lw->bqkv),
                                  q_hidden + 2 * kv_hidden,
                                  compute_stream_);
    }
    kernels::launch_rope_inplace_table(static_cast<__half*>(d_q_),
                                       static_cast<__half*>(d_k_),
                                       cfg.num_heads,
                                       cfg.num_kv_heads,
                                       head_dim,
                                       position,
                                       d_rope_cos_,
                                       d_rope_sin_,
                                       compute_stream_);
    if (kv_int4_enabled_) {
      do_kv_int4(layer);
    } else {
      const auto [attn_start, attn_seq_len] = attention_bounds(layer, position);
      const std::size_t kv_bytes = bytes_for_matrix(1, kv_hidden);
      const std::size_t layer_stride = static_cast<std::size_t>(options_.max_context) * static_cast<std::size_t>(kv_hidden);
      __half* k_layer = nullptr;
      __half* v_layer = nullptr;
      if (options_.paged_kv_cache) {
        auto* h_k_layer = static_cast<__half*>(h_k_cache_) + static_cast<std::size_t>(layer) * layer_stride;
        auto* h_v_layer = static_cast<__half*>(h_v_cache_) + static_cast<std::size_t>(layer) * layer_stride;
        CUDA_CHECK(cudaMemcpyAsync(h_k_layer + static_cast<std::size_t>(position) * kv_hidden,
                                   d_k_, kv_bytes, cudaMemcpyDeviceToHost, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(h_v_layer + static_cast<std::size_t>(position) * kv_hidden,
                                   d_v_, kv_bytes, cudaMemcpyDeviceToHost, compute_stream_));
        const std::size_t window_bytes =
            static_cast<std::size_t>(attn_seq_len) * static_cast<std::size_t>(kv_hidden) * sizeof(__half);
        CUDA_CHECK(cudaMemcpyAsync(d_k_cache_,
                                   h_k_layer + static_cast<std::size_t>(attn_start) * kv_hidden,
                                   window_bytes,
                                   cudaMemcpyHostToDevice,
                                   compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(d_v_cache_,
                                   h_v_layer + static_cast<std::size_t>(attn_start) * kv_hidden,
                                   window_bytes,
                                   cudaMemcpyHostToDevice,
                                   compute_stream_));
        k_layer = static_cast<__half*>(d_k_cache_);
        v_layer = static_cast<__half*>(d_v_cache_);
      } else {
        k_layer = static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride +
                  static_cast<std::size_t>(attn_start) * static_cast<std::size_t>(kv_hidden);
        v_layer = static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride +
                  static_cast<std::size_t>(attn_start) * static_cast<std::size_t>(kv_hidden);
        auto* k_store = static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride;
        auto* v_store = static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride;
        CUDA_CHECK(cudaMemcpyAsync(k_store + static_cast<std::size_t>(position) * kv_hidden,
                                   d_k_, kv_bytes, cudaMemcpyDeviceToDevice, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(v_store + static_cast<std::size_t>(position) * kv_hidden,
                                   d_v_, kv_bytes, cudaMemcpyDeviceToDevice, compute_stream_));
      }
      kernels::launch_attention_step(static_cast<const __half*>(d_q_),
                                     k_layer, v_layer, static_cast<__half*>(d_att_),
                                     attn_seq_len, cfg.num_heads, cfg.num_kv_heads, head_dim,
                                     compute_stream_, nullptr, nullptr, nullptr, 0,
                                     !options_.disable_split_attention);
    }
    if (lw_i8 && lw_i8->wo) {
      kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_att_),
                                                    static_cast<std::int8_t*>(d_prefill_i8_),
                                                    static_cast<float*>(d_prefill_i8_scales_),
                                                    1,
                                                    q_hidden,
                                                    compute_stream_);
      if (lw_i8->proj_int4) {
        kernels::launch_weight_only_int4_matvec_dp4a(lw_i8->wo,
                                                     lw_i8->s_wo,
                                                     static_cast<const std::int8_t*>(d_prefill_i8_),
                                                     static_cast<const float*>(d_prefill_i8_scales_),
                                                     static_cast<__half*>(d_ff3_),
                                                     hidden,
                                                     q_hidden,
                                                     compute_stream_,
                                                     resident_int8_wo_warps_,
                                                     resident_int8_wo_tile_packed4_,
                                                     resident_int8_wo_warps_per_row_);
      } else {
        kernels::launch_weight_only_int8_matvec_dp4a(lw_i8->wo,
                                                     lw_i8->s_wo,
                                                     static_cast<const std::int8_t*>(d_prefill_i8_),
                                                     static_cast<const float*>(d_prefill_i8_scales_),
                                                     static_cast<__half*>(d_ff3_),
                                                     hidden,
                                                     q_hidden,
                                                     compute_stream_,
                                                     resident_int8_wo_warps_,
                                                     resident_int8_wo_tile_packed4_,
                                                     resident_int8_wo_warps_per_row_);
      }
    } else {
      resident_projection_half(lw->wo,
                               d_att_,
                               d_ff3_,
                               hidden,
                               q_hidden,
                               resident_wo_warps_,
                               resident_wo_tile_pairs_,
                               resident_wo_rows_per_warp_);
    }
    maybe_add_half_bias(d_ff3_, lw->bo, 1, hidden);

    kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                static_cast<const __half*>(d_ff3_),
                                hidden,
                                compute_stream_);

    launch_norm(d_x_, lw->norm_ffn, lw->norm_ffn_bias, d_x_norm_, 1, hidden);

    if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3) {
      if (lw_i8->mlp_int4) {
        kernels::launch_weight_only_int4_matvec(
            lw_i8->w1, lw_i8->s_w1, static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff1_), inter, hidden, compute_stream_);
        kernels::launch_weight_only_int4_matvec(
            lw_i8->w3, lw_i8->s_w3, static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff2_), inter, hidden, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec(
            lw_i8->w1, lw_i8->s_w1, static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff1_), inter, hidden, compute_stream_);
        kernels::launch_weight_only_int8_matvec(
            lw_i8->w3, lw_i8->s_w3, static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff2_), inter, hidden, compute_stream_);
      }
    } else {
      resident_projection_half(lw->w13,
                               d_x_norm_,
                               d_ff1_,
                               2 * inter,
                               hidden,
                               resident_qkv_warps_,
                               resident_qkv_tile_pairs_,
                               resident_qkv_rows_per_warp_);
    }

    kernels::launch_silu_mul(static_cast<const __half*>(d_ff1_),
                             static_cast<const __half*>(d_ff2_),
                             static_cast<__half*>(d_ff2_),
                             inter,
                             compute_stream_);

    if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3) {
      if (lw_i8->mlp_int4) {
        kernels::launch_weight_only_int4_matvec(
            lw_i8->w2, lw_i8->s_w2, static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_), hidden, inter, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec(
            lw_i8->w2, lw_i8->s_w2, static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_), hidden, inter, compute_stream_);
      }
    } else {
      resident_projection_half(lw->w2,
                               d_ff2_,
                               d_ff3_,
                               hidden,
                               inter,
                               resident_wo_warps_,
                               resident_wo_tile_pairs_,
                               resident_wo_rows_per_warp_);
    }

    kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                static_cast<const __half*>(d_ff3_),
                                hidden,
                                compute_stream_);
  };

  const auto run_layer_resident_fp16 = [&](int layer, const LayerDeviceWeights* lw) {
    run_profiled(last_benchmark_stats_.decode_rmsnorm_ms, [&] {
      launch_norm(d_x_, lw->norm_att, lw->norm_att_bias, d_x_norm_, 1, hidden);
    });

    run_profiled(last_benchmark_stats_.decode_qkv_ms, [&] {
      if (resident_custom_qkv_) {
        resident_projection_half(
            lw->wqkv,
            d_x_norm_,
            d_q_,
            q_hidden + 2 * kv_hidden,
            hidden,
            resident_qkv_warps_,
            resident_qkv_tile_pairs_,
            resident_qkv_rows_per_warp_);
      } else {
        resident_projection_half(lw->wqkv,
                                 d_x_norm_,
                                 d_q_,
                                 q_hidden + 2 * kv_hidden,
                                 hidden,
                                 resident_qkv_warps_,
                                 resident_qkv_tile_pairs_,
                                 resident_qkv_rows_per_warp_);
      }
    });

    if (lw->bqkv) {
      kernels::launch_add_inplace(static_cast<__half*>(d_q_),
                                  static_cast<const __half*>(lw->bqkv),
                                  q_hidden + 2 * kv_hidden,
                                  compute_stream_);
    }

    kernels::launch_rope_inplace_table(static_cast<__half*>(d_q_),
                                       static_cast<__half*>(d_k_),
                                       cfg.num_heads,
                                       cfg.num_kv_heads,
                                       head_dim,
                                       position,
                                       d_rope_cos_,
                                       d_rope_sin_,
                                       compute_stream_);

    run_profiled(last_benchmark_stats_.decode_kv_store_ms, [&] {
      if (kv_int4_enabled_) {
        do_kv_int4(layer);
      } else {
        const std::size_t kv_bytes = bytes_for_matrix(1, kv_hidden);
        const std::size_t layer_stride = static_cast<std::size_t>(options_.max_context) * static_cast<std::size_t>(kv_hidden);
        auto* k_layer = static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride;
        auto* v_layer = static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride;
        CUDA_CHECK(cudaMemcpyAsync(k_layer + static_cast<std::size_t>(position) * kv_hidden,
                                   d_k_, kv_bytes, cudaMemcpyDeviceToDevice, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(v_layer + static_cast<std::size_t>(position) * kv_hidden,
                                   d_v_, kv_bytes, cudaMemcpyDeviceToDevice, compute_stream_));
      }
    });

    if (!kv_int4_enabled_) {
      run_profiled(last_benchmark_stats_.decode_attention_ms, [&] {
        const auto [attn_start, attn_seq_len] = attention_bounds(layer, position);
        const std::size_t layer_stride = static_cast<std::size_t>(options_.max_context) * static_cast<std::size_t>(kv_hidden);
        auto* k_layer = static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride +
                        static_cast<std::size_t>(attn_start) * static_cast<std::size_t>(kv_hidden);
        auto* v_layer = static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride +
                        static_cast<std::size_t>(attn_start) * static_cast<std::size_t>(kv_hidden);
        kernels::launch_attention_step(static_cast<const __half*>(d_q_),
                                       k_layer, v_layer, static_cast<__half*>(d_att_),
                                       attn_seq_len, cfg.num_heads, cfg.num_kv_heads, head_dim,
                                       compute_stream_, d_attn_chunk_m_, d_attn_chunk_l_, d_attn_chunk_o_,
                                       attn_chunk_capacity_, !options_.disable_split_attention);
      });
    }

    run_profiled(last_benchmark_stats_.decode_wo_ms, [&] {
      if (resident_custom_wo_) {
        resident_projection_half(
            lw->wo, d_att_, d_ff3_, hidden, q_hidden, resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      } else {
        resident_projection_half(lw->wo,
                                 d_att_,
                                 d_ff3_,
                                 hidden,
                                 q_hidden,
                                 resident_wo_warps_,
                                 resident_wo_tile_pairs_,
                                 resident_wo_rows_per_warp_);
      }
      maybe_add_half_bias(d_ff3_, lw->bo, 1, hidden);
      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  static_cast<const __half*>(d_ff3_),
                                  hidden,
                                  compute_stream_);
    });

    run_profiled(last_benchmark_stats_.decode_rmsnorm_ms, [&] {
      launch_norm(d_x_, lw->norm_ffn, lw->norm_ffn_bias, d_x_norm_, 1, hidden);
    });

    run_profiled(last_benchmark_stats_.decode_mlp_ms, [&] {
      resident_projection_half(lw->w13,
                               d_x_norm_,
                               d_ff1_,
                               2 * inter,
                               hidden,
                               resident_qkv_warps_,
                               resident_qkv_tile_pairs_,
                               resident_qkv_rows_per_warp_);

      kernels::launch_silu_mul(static_cast<const __half*>(d_ff1_),
                               static_cast<const __half*>(d_ff2_),
                               static_cast<__half*>(d_ff2_),
                               inter,
                               compute_stream_);

      resident_projection_half(lw->w2,
                               d_ff2_,
                               d_ff3_,
                               hidden,
                               inter,
                               resident_wo_warps_,
                               resident_wo_tile_pairs_,
                               resident_wo_rows_per_warp_);

      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  static_cast<const __half*>(d_ff3_),
                                  hidden,
                                  compute_stream_);
    });
  };

  const auto run_layer_resident_int8 = [&](int layer,
                                           const LayerDeviceWeights* lw,
                                           const LayerDeviceInt8Weights* lw_i8) {
    run_profiled(last_benchmark_stats_.decode_rmsnorm_ms, [&] {
      launch_norm(d_x_, lw->norm_att, lw->norm_att_bias, d_x_norm_, 1, hidden);
    });

    run_profiled(last_benchmark_stats_.decode_qkv_ms, [&] {
      if (cached_int8_proj_enabled_) {
        kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_x_norm_),
                                                      static_cast<std::int8_t*>(d_prefill_i8_),
                                                      static_cast<float*>(d_prefill_i8_scales_),
                                                      1,
                                                      hidden,
                                                      compute_stream_);
        if (lw_i8->proj_int4) {
          kernels::launch_weight_only_int4_matvec_dp4a(lw_i8->wqkv,
                                                       lw_i8->s_wqkv,
                                                       static_cast<const std::int8_t*>(d_prefill_i8_),
                                                       static_cast<const float*>(d_prefill_i8_scales_),
                                                       static_cast<__half*>(d_q_),
                                                       q_hidden + 2 * kv_hidden,
                                                       hidden,
                                                       compute_stream_,
                                                       resident_int8_qkv_warps_,
                                                       resident_int8_qkv_tile_packed4_,
                                                       resident_int8_qkv_warps_per_row_);
        } else {
          kernels::launch_weight_only_int8_matvec_dp4a(lw_i8->wqkv,
                                                       lw_i8->s_wqkv,
                                                       static_cast<const std::int8_t*>(d_prefill_i8_),
                                                       static_cast<const float*>(d_prefill_i8_scales_),
                                                       static_cast<__half*>(d_q_),
                                                       q_hidden + 2 * kv_hidden,
                                                       hidden,
                                                       compute_stream_,
                                                       resident_int8_qkv_warps_,
                                                       resident_int8_qkv_tile_packed4_,
                                                       resident_int8_qkv_warps_per_row_);
        }
      } else if (resident_custom_qkv_) {
        resident_projection_half(
            lw->wqkv,
            d_x_norm_,
            d_q_,
            q_hidden + 2 * kv_hidden,
            hidden,
            resident_qkv_warps_,
            resident_qkv_tile_pairs_,
            resident_qkv_rows_per_warp_);
      } else {
        detail::dispatch_linear_rowmajor_weight(cublas_,
                               matmul_lt,
                               &lt_plan_cache_,
                               lt_workspace_,
                               lt_workspace_bytes_,
                               compute_stream_,
                               lw->wqkv,
                               d_x_norm_,
                               d_q_,
                               q_hidden + 2 * kv_hidden,
                               hidden,
                               1,
                               CUDA_R_16F);
      }
    });

    if (lw->bqkv) {
      kernels::launch_add_inplace(static_cast<__half*>(d_q_),
                                  static_cast<const __half*>(lw->bqkv),
                                  q_hidden + 2 * kv_hidden,
                                  compute_stream_);
    }

    kernels::launch_rope_inplace_table(static_cast<__half*>(d_q_),
                                       static_cast<__half*>(d_k_),
                                       cfg.num_heads,
                                       cfg.num_kv_heads,
                                       head_dim,
                                       position,
                                       d_rope_cos_,
                                       d_rope_sin_,
                                       compute_stream_);

    run_profiled(last_benchmark_stats_.decode_kv_store_ms, [&] {
      if (kv_int4_enabled_) {
        do_kv_int4(layer);
      } else {
        const std::size_t kv_bytes = bytes_for_matrix(1, kv_hidden);
        const std::size_t layer_stride = static_cast<std::size_t>(options_.max_context) * static_cast<std::size_t>(kv_hidden);
        auto* k_layer = static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride;
        auto* v_layer = static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride;
        CUDA_CHECK(cudaMemcpyAsync(k_layer + static_cast<std::size_t>(position) * kv_hidden,
                                   d_k_, kv_bytes, cudaMemcpyDeviceToDevice, compute_stream_));
        CUDA_CHECK(cudaMemcpyAsync(v_layer + static_cast<std::size_t>(position) * kv_hidden,
                                   d_v_, kv_bytes, cudaMemcpyDeviceToDevice, compute_stream_));
      }
    });

    if (!kv_int4_enabled_) {
      run_profiled(last_benchmark_stats_.decode_attention_ms, [&] {
        const auto [attn_start, attn_seq_len] = attention_bounds(layer, position);
        const std::size_t layer_stride = static_cast<std::size_t>(options_.max_context) * static_cast<std::size_t>(kv_hidden);
        auto* k_layer = static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride +
                        static_cast<std::size_t>(attn_start) * static_cast<std::size_t>(kv_hidden);
        auto* v_layer = static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride +
                        static_cast<std::size_t>(attn_start) * static_cast<std::size_t>(kv_hidden);
        kernels::launch_attention_step(static_cast<const __half*>(d_q_),
                                       k_layer, v_layer, static_cast<__half*>(d_att_),
                                       attn_seq_len, cfg.num_heads, cfg.num_kv_heads, head_dim,
                                       compute_stream_, d_attn_chunk_m_, d_attn_chunk_l_, d_attn_chunk_o_,
                                       attn_chunk_capacity_, !options_.disable_split_attention);
      });
    }

    run_profiled(last_benchmark_stats_.decode_wo_ms, [&] {
      if (cached_int8_proj_enabled_) {
        kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_att_),
                                                      static_cast<std::int8_t*>(d_prefill_i8_),
                                                      static_cast<float*>(d_prefill_i8_scales_),
                                                      1,
                                                      q_hidden,
                                                      compute_stream_);
        if (lw_i8->proj_int4) {
          kernels::launch_weight_only_int4_matvec_dp4a(lw_i8->wo,
                                                       lw_i8->s_wo,
                                                       static_cast<const std::int8_t*>(d_prefill_i8_),
                                                       static_cast<const float*>(d_prefill_i8_scales_),
                                                       static_cast<__half*>(d_ff3_),
                                                       hidden,
                                                       q_hidden,
                                                       compute_stream_,
                                                       resident_int8_wo_warps_,
                                                       resident_int8_wo_tile_packed4_,
                                                       resident_int8_wo_warps_per_row_);
        } else {
          kernels::launch_weight_only_int8_matvec_dp4a(lw_i8->wo,
                                                       lw_i8->s_wo,
                                                       static_cast<const std::int8_t*>(d_prefill_i8_),
                                                       static_cast<const float*>(d_prefill_i8_scales_),
                                                       static_cast<__half*>(d_ff3_),
                                                       hidden,
                                                       q_hidden,
                                                       compute_stream_,
                                                       resident_int8_wo_warps_,
                                                       resident_int8_wo_tile_packed4_,
                                                       resident_int8_wo_warps_per_row_);
        }
      } else if (resident_custom_wo_) {
        resident_projection_half(
            lw->wo, d_att_, d_ff3_, hidden, q_hidden, resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      } else {
        detail::dispatch_linear_rowmajor_weight(cublas_,
                               matmul_lt,
                               &lt_plan_cache_,
                               lt_workspace_,
                               lt_workspace_bytes_,
                               compute_stream_,
                               lw->wo,
                               d_att_,
                               d_ff3_,
                               hidden,
                               q_hidden,
                               1,
                               CUDA_R_16F);
      }
      maybe_add_half_bias(d_ff3_, lw->bo, 1, hidden);
      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  static_cast<const __half*>(d_ff3_),
                                  hidden,
                                  compute_stream_);
    });

    run_profiled(last_benchmark_stats_.decode_rmsnorm_ms, [&] {
      launch_norm(d_x_, lw->norm_ffn, lw->norm_ffn_bias, d_x_norm_, 1, hidden);
    });

    run_profiled(last_benchmark_stats_.decode_mlp_ms, [&] {
      if (lw_i8->w1) {
        // INT8 MLP path (pre-quantised weights available).
        if (can_use_dp4a_decode) {
          kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_x_norm_),
                                                        d_prefill_i8_,
                                                        d_prefill_i8_scales_,
                                                        1,
                                                        hidden,
                                                        compute_stream_);
          resident_int8_mlp_w13(*lw_i8, inter, hidden);
        } else {
          if (lw_i8->mlp_int4) {
            kernels::launch_weight_only_int4_matvec(lw_i8->w1,
                                                    lw_i8->s_w1,
                                                    static_cast<const __half*>(d_x_norm_),
                                                    static_cast<__half*>(d_ff1_),
                                                    inter,
                                                    hidden,
                                                    compute_stream_);
            kernels::launch_weight_only_int4_matvec(lw_i8->w3,
                                                    lw_i8->s_w3,
                                                    static_cast<const __half*>(d_x_norm_),
                                                    static_cast<__half*>(d_ff2_),
                                                    inter,
                                                    hidden,
                                                    compute_stream_);
          } else {
            kernels::launch_weight_only_int8_matvec(lw_i8->w1,
                                                    lw_i8->s_w1,
                                                    static_cast<const __half*>(d_x_norm_),
                                                    static_cast<__half*>(d_ff1_),
                                                    inter,
                                                    hidden,
                                                    compute_stream_);
            kernels::launch_weight_only_int8_matvec(lw_i8->w3,
                                                    lw_i8->s_w3,
                                                    static_cast<const __half*>(d_x_norm_),
                                                    static_cast<__half*>(d_ff2_),
                                                    inter,
                                                    hidden,
                                                    compute_stream_);
          }
        }

        kernels::launch_silu_mul(static_cast<const __half*>(d_ff1_),
                                 static_cast<const __half*>(d_ff2_),
                                 static_cast<__half*>(d_ff2_),
                                 inter,
                                 compute_stream_);

        if (can_use_dp4a_decode) {
          kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_ff2_),
                                                        d_prefill_i8_,
                                                        d_prefill_i8_scales_,
                                                        1,
                                                        inter,
                                                        compute_stream_);
          resident_int8_mlp_w2(*lw_i8, hidden, inter);
        } else {
          if (lw_i8->mlp_int4) {
            kernels::launch_weight_only_int4_matvec(lw_i8->w2,
                                                    lw_i8->s_w2,
                                                    static_cast<const __half*>(d_ff2_),
                                                    static_cast<__half*>(d_ff3_),
                                                    hidden,
                                                    inter,
                                                    compute_stream_);
          } else {
            kernels::launch_weight_only_int8_matvec(lw_i8->w2,
                                                    lw_i8->s_w2,
                                                    static_cast<const __half*>(d_ff2_),
                                                    static_cast<__half*>(d_ff3_),
                                                    hidden,
                                                    inter,
                                                    compute_stream_);
          }
        }
      } else {
        // FP16 MLP fallback (no pre-quantised MLP weights; int8 proj only).
        // Use resident_projection_half: always graph-capturable, unlike cuBLASLt/cublasGemmEx.
        resident_projection_half(lw->w13, d_x_norm_, d_ff1_, 2 * inter, hidden,
                                 resident_qkv_warps_, resident_qkv_tile_pairs_, resident_qkv_rows_per_warp_);

        kernels::launch_silu_mul(static_cast<const __half*>(d_ff1_),
                                 static_cast<const __half*>(d_ff2_),
                                 static_cast<__half*>(d_ff2_),
                                 inter,
                                 compute_stream_);

        resident_projection_half(lw->w2, d_ff2_, d_ff3_, hidden, inter,
                                 resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      }

      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  static_cast<const __half*>(d_ff3_),
                                  hidden,
                                  compute_stream_);
    });
  };

  // TQ3 resident fast path: wqkv/wo/w13 use 3-bit TurboQuant GEMV; w2 uses fp16.
  // The input activation is rotated with the Hadamard transform before each
  // projection that has TQ3 weights.  The rotation is applied to a scratch
  // buffer (d_x_tq3_) so d_x_norm_ stays intact for w2.
  const auto apply_qprod_residual = [&](const uint32_t* row_bits,
                                        const half* residual_scales,
                                        const half* rotated_x,
                                        half* y,
                                        int out_features) {
    if (!tq_prod_enabled_ || tq_qjl_dim_ <= 0 || !row_bits || !residual_scales || !rotated_x || !y ||
        !d_tq_qjl_indices_ || !d_tq_qjl_signs_ || !d_tq_qjl_x_bits_) {
      return;
    }
    kernels::launch_tq_qjl_pack_sign_bits(rotated_x,
                                          d_tq_qjl_indices_,
                                          d_tq_qjl_signs_,
                                          d_tq_qjl_x_bits_,
                                          tq_qjl_dim_,
                                          compute_stream_);
    kernels::launch_tq_qjl_residual_add_f16(row_bits,
                                            residual_scales,
                                            d_tq_qjl_x_bits_,
                                            y,
                                            out_features,
                                            tq_qjl_dim_,
                                            compute_stream_);
  };

  const bool resident_tq3_path = tq3_enabled_ && resident_fast_path &&
                                  !layer_cache_tq3_.empty() &&
                                  static_cast<int>(layer_cache_tq3_.size()) == cfg.num_layers;

  const auto run_layer_resident_tq3 = [&](int layer,
                                          const LayerDeviceWeights* lw,
                                          const LayerDeviceTq3Weights* tq) {
    // --- Attention block ---
    run_profiled(last_benchmark_stats_.decode_rmsnorm_ms, [&] {
      launch_norm(d_x_, lw->norm_att, lw->norm_att_bias, d_x_norm_, 1, hidden);
    });

    run_profiled(last_benchmark_stats_.decode_qkv_ms, [&] {
      // Rotate x_norm Ã¢â€ â€™ d_x_tq3_ then run TQ3 GEMV for wqkv.
      CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_x_norm_, static_cast<std::size_t>(hidden) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, compute_stream_));
      kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_),
                                           d_tq3_signs_, hidden, tq3_block_size_, compute_stream_);
      if (tq->wqkv) {
        const int out_qkv = q_hidden + 2 * kv_hidden;
        const int words   = (hidden + 9) / 10;
        kernels::launch_tq3_gemv_f16(tq->wqkv, d_tq3_codebook_, tq->s_wqkv,
                                     static_cast<const __half*>(d_x_tq3_),
                                     static_cast<__half*>(d_q_),
                                     out_qkv, hidden, compute_stream_);
        apply_qprod_residual(tq->r_wqkv, tq->rs_wqkv,
                             static_cast<const __half*>(d_x_tq3_),
                             static_cast<__half*>(d_q_),
                             out_qkv);
        (void)words;
      } else {
        // Fallback: fp16 (wq/wk/wv not quantised in this file).
        resident_projection_half(lw->wqkv, d_x_norm_, d_q_,
                                 q_hidden + 2 * kv_hidden, hidden,
                                 resident_qkv_warps_, resident_qkv_tile_pairs_, resident_qkv_rows_per_warp_);
      }
    });

    if (lw->bqkv) {
      kernels::launch_add_inplace(static_cast<__half*>(d_q_),
                                  static_cast<const __half*>(lw->bqkv),
                                  q_hidden + 2 * kv_hidden,
                                  compute_stream_);
    }

    kernels::launch_rope_inplace_table(static_cast<__half*>(d_q_),
                                       static_cast<__half*>(d_k_),
                                       cfg.num_heads,
                                       cfg.num_kv_heads,
                                       head_dim,
                                       position,
                                       d_rope_cos_,
                                       d_rope_sin_,
                                       compute_stream_);

    run_profiled(last_benchmark_stats_.decode_kv_store_ms, [&] {
      const std::size_t kv_bytes    = bytes_for_matrix(1, kv_hidden);
      const std::size_t layer_stride = static_cast<std::size_t>(options_.max_context) * kv_hidden;
      auto* k_layer = static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride;
      auto* v_layer = static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride;
      CUDA_CHECK(cudaMemcpyAsync(k_layer + static_cast<std::size_t>(position) * kv_hidden,
                                 d_k_, kv_bytes, cudaMemcpyDeviceToDevice, compute_stream_));
      CUDA_CHECK(cudaMemcpyAsync(v_layer + static_cast<std::size_t>(position) * kv_hidden,
                                 d_v_, kv_bytes, cudaMemcpyDeviceToDevice, compute_stream_));
    });

    run_profiled(last_benchmark_stats_.decode_attention_ms, [&] {
      const auto [attn_start, attn_seq_len] = attention_bounds(layer, position);
      const std::size_t layer_stride = static_cast<std::size_t>(options_.max_context) * kv_hidden;
      auto* k_layer = static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride +
                      static_cast<std::size_t>(attn_start) * static_cast<std::size_t>(kv_hidden);
      auto* v_layer = static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride +
                      static_cast<std::size_t>(attn_start) * static_cast<std::size_t>(kv_hidden);
      kernels::launch_attention_step(static_cast<const __half*>(d_q_),
                                     k_layer, v_layer, static_cast<__half*>(d_att_),
                                     attn_seq_len, cfg.num_heads, cfg.num_kv_heads, head_dim,
                                     compute_stream_, d_attn_chunk_m_, d_attn_chunk_l_, d_attn_chunk_o_,
                                     attn_chunk_capacity_, !options_.disable_split_attention);
    });

    run_profiled(last_benchmark_stats_.decode_wo_ms, [&] {
      // Rotate att Ã¢â€ â€™ d_x_tq3_ for TQ3 wo GEMV.
      CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_att_, static_cast<std::size_t>(hidden) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, compute_stream_));
      kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_),
                                           d_tq3_signs_, hidden, tq3_block_size_, compute_stream_);
      if (tq->wo) {
        kernels::launch_tq3_gemv_f16(tq->wo, d_tq3_codebook_, tq->s_wo,
                                     static_cast<const __half*>(d_x_tq3_),
                                     static_cast<__half*>(d_ff3_),
                                     hidden, hidden, compute_stream_);
        apply_qprod_residual(tq->r_wo, tq->rs_wo,
                             static_cast<const __half*>(d_x_tq3_),
                             static_cast<__half*>(d_ff3_),
                             hidden);
      } else {
        resident_projection_half(lw->wo, d_att_, d_ff3_, hidden, q_hidden,
                                 resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      }
      maybe_add_half_bias(d_ff3_, lw->bo, 1, hidden);
      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  static_cast<const __half*>(d_ff3_),
                                  hidden, compute_stream_);
    });

    // --- FFN block ---
    run_profiled(last_benchmark_stats_.decode_rmsnorm_ms, [&] {
      launch_norm(d_x_, lw->norm_ffn, lw->norm_ffn_bias, d_x_norm_, 1, hidden);
    });

    run_profiled(last_benchmark_stats_.decode_mlp_ms, [&] {
      if (tq->w13) {
        // Rotate x_norm Ã¢â€ â€™ d_x_tq3_ for TQ3 w13 GEMV.
        CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_x_norm_, static_cast<std::size_t>(hidden) * sizeof(__half),
                                   cudaMemcpyDeviceToDevice, compute_stream_));
        kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_),
                                             d_tq3_signs_, hidden, tq3_block_size_, compute_stream_);
        kernels::launch_tq3_gemv_f16(tq->w13, d_tq3_codebook_, tq->s_w13,
                                     static_cast<const __half*>(d_x_tq3_),
                                     static_cast<__half*>(d_ff1_),
                                     2 * inter, hidden, compute_stream_);
        apply_qprod_residual(tq->r_w13, tq->rs_w13,
                             static_cast<const __half*>(d_x_tq3_),
                             static_cast<__half*>(d_ff1_),
                             2 * inter);
      } else {
        resident_projection_half(lw->w13, d_x_norm_, d_ff1_, 2 * inter, hidden,
                                 resident_qkv_warps_, resident_qkv_tile_pairs_, resident_qkv_rows_per_warp_);
      }

      kernels::launch_silu_mul(static_cast<const __half*>(d_ff1_),
                               static_cast<const __half*>(d_ff2_),
                               static_cast<__half*>(d_ff2_),
                               inter, compute_stream_);

      // w2 is always fp16. Use resident_projection_half (always graph-capturable
      // and avoids cuBLASLt autotuning overhead on first call).
      resident_projection_half(lw->w2, d_ff2_, d_ff3_, hidden, inter,
                               resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);

      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  static_cast<const __half*>(d_ff3_),
                                  hidden, compute_stream_);
    });
  };

  if (resident_tq3_path) {
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
      run_layer_resident_tq3(layer,
                             &layer_cache_[static_cast<std::size_t>(layer)],
                             &layer_cache_tq3_[static_cast<std::size_t>(layer)]);
    }
  } else if (resident_all_packed_mlp) {
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
      run_layer_resident_int8(layer,
                              &layer_cache_[static_cast<std::size_t>(layer)],
                              &layer_cache_i8_[static_cast<std::size_t>(layer)]);
    }
  } else if (resident_fast_path) {
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
      run_layer_resident_fp16(layer, &layer_cache_[static_cast<std::size_t>(layer)]);
    }
  } else {
    int uncached_index = 0;
    copy_layer_weights_to_device(cached_layer_count_,
                                 &streaming_layer_weights_[0],
                                 options_.int8_streaming ? &streaming_layer_weights_i8_[0] : nullptr,
                                 transfer_stream_);
    CUDA_CHECK(cudaEventRecord(streaming_ready_[0], transfer_stream_));

    for (int layer = 0; layer < cfg.num_layers; ++layer) {
      const LayerDeviceWeights* lw = nullptr;
      const LayerDeviceInt8Weights* lw_i8 = nullptr;
      if (layer < cached_layer_count_) {
        lw = &layer_cache_[static_cast<std::size_t>(layer)];
        if (layer < static_cast<int>(layer_cache_i8_.size())) {
          lw_i8 = &layer_cache_i8_[static_cast<std::size_t>(layer)];
        }
      } else {
        const int slot = uncached_index % 2;
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream_, streaming_ready_[slot], 0));
        lw = &streaming_layer_weights_[slot];
        if (options_.int8_streaming) {
          lw_i8 = &streaming_layer_weights_i8_[slot];
        }

        const int next_layer = layer + 1;
        if (next_layer < cfg.num_layers) {
          const int next_slot = (uncached_index + 1) % 2;
          CUDA_CHECK(cudaStreamWaitEvent(transfer_stream_, streaming_consumed_[next_slot], 0));
          copy_layer_weights_to_device(next_layer,
                                       &streaming_layer_weights_[next_slot],
                                       options_.int8_streaming ? &streaming_layer_weights_i8_[next_slot] : nullptr,
                                       transfer_stream_);
          CUDA_CHECK(cudaEventRecord(streaming_ready_[next_slot], transfer_stream_));
        }
        ++uncached_index;
      }

      run_layer(layer, lw, lw_i8);

      if (layer >= cached_layer_count_) {
        const int consumed_slot = (uncached_index - 1) % 2;
        CUDA_CHECK(cudaEventRecord(streaming_consumed_[consumed_slot], compute_stream_));
      }
      enforce_host_resource_limits("decode.layer_end");
    }
  }

}
}  // namespace engine
