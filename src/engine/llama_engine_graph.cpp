#include "engine/llama_engine.hpp"
#include "llama_engine_internal.hpp"

#include <iostream>

#include <cuda_fp16.h>

#include "common.hpp"
#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"

namespace engine {

bool LlamaEngine::can_use_greedy_decode_graph() const {
  return cached_layer_count_ == weights_.config().num_layers && !options_.paged_kv_cache &&
         !options_.profile_decode_phases && !kv_int4_enabled_ &&
         !weights_.config().is_moe() && !weights_.config().use_layernorm &&
         (attn_q_hidden_ <= 0 || attn_q_hidden_ == weights_.config().hidden_size) &&
         !has_any_layer_norm_bias_ && !has_any_layer_output_bias_ &&
         !weights_.has_tensor("norm.bias") && !weights_.has_tensor("output.bias") &&
         weights_.config().sliding_window <= 0;
}

void LlamaEngine::destroy_greedy_decode_graph() {
  if (greedy_decode_graph_exec_) {
    cudaGraphExecDestroy(greedy_decode_graph_exec_);
    greedy_decode_graph_exec_ = nullptr;
  }
  if (greedy_decode_graph_) {
    cudaGraphDestroy(greedy_decode_graph_);
    greedy_decode_graph_ = nullptr;
  }
  greedy_decode_graph_ready_ = false;
  greedy_decode_graph_state_valid_ = false;
}

void LlamaEngine::destroy_logits_decode_graph() {
  if (logits_decode_graph_exec_) {
    cudaGraphExecDestroy(logits_decode_graph_exec_);
    logits_decode_graph_exec_ = nullptr;
  }
  if (logits_decode_graph_) {
    cudaGraphDestroy(logits_decode_graph_);
    logits_decode_graph_ = nullptr;
  }
  logits_decode_graph_ready_ = false;
}

void LlamaEngine::init_greedy_decode_graph() {
  if (greedy_decode_graph_ready_) {
    return;
  }
  if (!can_use_greedy_decode_graph()) {
    return;
  }
  if (options_.verbose) {
    std::cout << "[engine] init_greedy_decode_graph: starting warmup\n";
  }

  const auto& cfg = weights_.config();
  const int hidden = cfg.hidden_size;
  const int inter = cfg.intermediate_size;
  const int head_dim = cfg.hidden_size / cfg.num_heads;
  const int kv_hidden = cfg.num_kv_heads * head_dim;
  const bool can_use_dp4a_decode = ((hidden & 3) == 0) && ((inter & 3) == 0);
  const auto apply_qprod_residual = [&](const uint32_t* row_bits,
                                        const half* residual_scales,
                                        const half* rotated_x,
                                        half* y,
                                        int out_features) {
    if (!tq_prod_enabled_ || tq_qjl_dim_ <= 0 || !row_bits || !residual_scales || !rotated_x || !y ||
        !d_tq_qjl_indices_ || !d_tq_qjl_signs_ || !d_tq_qjl_x_bits_) {
      return;
    }
    kernels::launch_tq_qjl_pack_sign_bits(rotated_x, d_tq_qjl_indices_, d_tq_qjl_signs_,
                                          d_tq_qjl_x_bits_, tq_qjl_dim_, compute_stream_);
    kernels::launch_tq_qjl_residual_add_f16(row_bits, residual_scales, d_tq_qjl_x_bits_,
                                            y, out_features, tq_qjl_dim_, compute_stream_);
  };

  // Warm cuBLASLt plans before capture so the graph records steady-state launches.
  // Skip fp16 QKV/WO/MLP warmup when INT8 projections are active: those fp16 weight
  // pointers are freed after INT8 packing and must not be passed to cuBLAS.
  if (!layer_cache_.empty()) {
    const auto& lw = layer_cache_.front();
    if (tq3_enabled_ && !layer_cache_tq3_.empty()) {
      // TQ3: warm packed GEMV kernels; skip any weight that wasn't loaded (null guard).
      const auto& tq = layer_cache_tq3_.front();
      if (tq.wqkv) {
        kernels::launch_tq3_gemv_f16(tq.wqkv, d_tq3_codebook_, tq.s_wqkv,
                                     static_cast<const __half*>(d_x_tq3_), static_cast<__half*>(d_q_),
                                     hidden + 2 * kv_hidden, hidden, compute_stream_);
      }
      if (tq.wo) {
        kernels::launch_tq3_gemv_f16(tq.wo, d_tq3_codebook_, tq.s_wo,
                                     static_cast<const __half*>(d_x_tq3_), static_cast<__half*>(d_ff3_),
                                     hidden, hidden, compute_stream_);
      }
      if (tq.w13) {
        kernels::launch_tq3_gemv_f16(tq.w13, d_tq3_codebook_, tq.s_w13,
                                     static_cast<const __half*>(d_x_tq3_), static_cast<__half*>(d_ff1_),
                                     2 * inter, hidden, compute_stream_);
      }
      // Warm w2 (fp16, always) using resident_projection_half — always graph-capturable.
      resident_projection_half(lw.w2, d_ff2_, d_ff3_, hidden, inter,
                               resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
    } else if (!cached_int8_proj_enabled_) {
      if (resident_custom_qkv_) {
        resident_projection_half(lw.wqkv, d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden,
                                 resident_qkv_warps_, resident_qkv_tile_pairs_, resident_qkv_rows_per_warp_);
      } else {
        detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_,
                               compute_stream_, lw.wqkv, d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden, 1, CUDA_R_16F);
      }
      if (resident_custom_wo_) {
        resident_projection_half(lw.wo, d_att_, d_ff3_, hidden, hidden,
                                 resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      } else {
        detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_,
                               compute_stream_, lw.wo, d_att_, d_ff3_, hidden, hidden, 1, CUDA_R_16F);
      }
    }
    if (!tq3_enabled_) {
      if (cached_int8_mlp_enabled_ && !layer_cache_i8_.empty() && layer_cache_i8_.front().w1) {
        if (can_use_dp4a_decode) {
          resident_int8_mlp_w13(layer_cache_i8_.front(), inter, hidden);
          resident_int8_mlp_w2(layer_cache_i8_.front(), hidden, inter);
        } else {
          if (layer_cache_i8_.front().mlp_int4) {
            kernels::launch_weight_only_int4_matvec(layer_cache_i8_.front().w1, layer_cache_i8_.front().s_w1,
                                                    static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff1_),
                                                    inter, hidden, compute_stream_);
            kernels::launch_weight_only_int4_matvec(layer_cache_i8_.front().w2, layer_cache_i8_.front().s_w2,
                                                    static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_),
                                                    hidden, inter, compute_stream_);
          } else {
            kernels::launch_weight_only_int8_matvec(layer_cache_i8_.front().w1, layer_cache_i8_.front().s_w1,
                                                    static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff1_),
                                                    inter, hidden, compute_stream_);
            kernels::launch_weight_only_int8_matvec(layer_cache_i8_.front().w2, layer_cache_i8_.front().s_w2,
                                                    static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_),
                                                    hidden, inter, compute_stream_);
          }
        }
      } else if (!cached_int8_proj_enabled_) {
        detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_,
                               compute_stream_, lw.w13, d_x_norm_, d_ff1_, 2 * inter, hidden, 1, CUDA_R_16F);
        detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_,
                               compute_stream_, lw.w2, d_ff2_, d_ff3_, hidden, inter, 1, CUDA_R_16F);
      } else if (lw.w13) {
        // INT8 proj + FP16 MLP (e.g., TinyLlama): warm resident_projection_half since
        // cuBLASLt may not find a graph-capturable plan for all dimension combinations.
        resident_projection_half(lw.w13, d_x_norm_, d_ff1_, 2 * inter, hidden,
                                 resident_qkv_warps_, resident_qkv_tile_pairs_, resident_qkv_rows_per_warp_);
        resident_projection_half(lw.w2, d_ff2_, d_ff3_, hidden, inter,
                                 resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      }
    }
    // Always warm the custom LM-head kernel (used in all graph captures to avoid cuBLAS fallback issues).
    resident_projection_float(d_lm_head_, d_x_norm_, d_logits_, cfg.vocab_size, hidden,
                              resident_lm_head_warps_, resident_lm_head_tile_pairs_, resident_lm_head_rows_per_warp_);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  }

  destroy_greedy_decode_graph();
  CUDA_CHECK(cudaStreamBeginCapture(compute_stream_, cudaStreamCaptureModeGlobal));

  kernels::launch_embedding_lookup(static_cast<const __half*>(d_tok_embeddings_),
                                   d_token_id_,
                                   static_cast<__half*>(d_x_),
                                   1,
                                   hidden,
                                   compute_stream_);

  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    const auto* lw = &layer_cache_[static_cast<std::size_t>(layer)];
    // Include lw_i8 when either int8 MLP weights (w1) OR int8 projection weights
    // (wqkv) are present.  Using only w1 as the gate causes lw_i8=nullptr for
    // models like TinyLlama that have int8 projections but no int8 MLP, which
    // falls through to cublasGemmEx inside graph capture and triggers INVALID_VALUE.
    const LayerDeviceInt8Weights* lw_i8 =
        (layer < static_cast<int>(layer_cache_i8_.size()) &&
         (layer_cache_i8_[static_cast<std::size_t>(layer)].w1 ||
          layer_cache_i8_[static_cast<std::size_t>(layer)].wqkv))
            ? &layer_cache_i8_[static_cast<std::size_t>(layer)]
            : nullptr;
    const LayerDeviceTq3Weights* tq =
        (tq3_enabled_ && layer < static_cast<int>(layer_cache_tq3_.size()))
            ? &layer_cache_tq3_[static_cast<std::size_t>(layer)]
            : nullptr;
    auto* k_layer =
        static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * static_cast<std::size_t>(options_.max_context) *
                                               static_cast<std::size_t>(kv_hidden);
    auto* v_layer =
        static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * static_cast<std::size_t>(options_.max_context) *
                                               static_cast<std::size_t>(kv_hidden);

    kernels::launch_rmsnorm(static_cast<const __half*>(d_x_),
                            static_cast<const __half*>(lw->norm_att),
                            static_cast<__half*>(d_x_norm_),
                            1,
                            hidden,
                            1e-5f,
                            compute_stream_);

    if (tq && tq->wqkv) {
      CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_x_norm_, static_cast<std::size_t>(hidden) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, compute_stream_));
      kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_), d_tq3_signs_, hidden, tq3_block_size_, compute_stream_);
      kernels::launch_tq3_gemv_f16(tq->wqkv, d_tq3_codebook_, tq->s_wqkv,
                                   static_cast<const __half*>(d_x_tq3_), static_cast<__half*>(d_q_),
                                   hidden + 2 * kv_hidden, hidden, compute_stream_);
      apply_qprod_residual(tq->r_wqkv, tq->rs_wqkv,
                           static_cast<const __half*>(d_x_tq3_),
                           static_cast<__half*>(d_q_),
                           hidden + 2 * kv_hidden);
    } else if (cached_int8_proj_enabled_ && lw_i8 && lw_i8->wqkv) {
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
                                                     hidden + 2 * kv_hidden,
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
                                                     hidden + 2 * kv_hidden,
                                                     hidden,
                                                     compute_stream_,
                                                     resident_int8_qkv_warps_,
                                                     resident_int8_qkv_tile_packed4_,
                                                     resident_int8_qkv_warps_per_row_);
      }
    } else if (resident_custom_qkv_) {
      resident_projection_half(
          lw->wqkv, d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden, resident_qkv_warps_, resident_qkv_tile_pairs_);
    } else {
      detail::dispatch_linear_rowmajor_weight(cublas_,
                             cublas_lt_,
                             &lt_plan_cache_,
                             lt_workspace_,
                             lt_workspace_bytes_,
                             compute_stream_,
                             lw->wqkv,
                             d_x_norm_,
                             d_q_,
                             hidden + 2 * kv_hidden,
                             hidden,
                             1,
                             CUDA_R_16F);
    }

    if (lw->bqkv) {
      kernels::launch_add_inplace(static_cast<__half*>(d_q_),
                                  static_cast<const __half*>(lw->bqkv),
                                  hidden + 2 * kv_hidden,
                                  compute_stream_);
    }

    kernels::launch_rope_inplace_device_pos(static_cast<__half*>(d_q_),
                                            static_cast<__half*>(d_k_),
                                            cfg.num_heads,
                                            cfg.num_kv_heads,
                                            head_dim,
                                            d_decode_position_,
                                            d_rope_cos_,
                                            d_rope_sin_,
                                            compute_stream_);
    kernels::launch_store_kv_device_pos(static_cast<const __half*>(d_k_),
                                        static_cast<const __half*>(d_v_),
                                        k_layer,
                                        v_layer,
                                        d_decode_position_,
                                        kv_hidden,
                                        options_.max_context,
                                        compute_stream_);
    kernels::launch_attention_step_device_pos(static_cast<const __half*>(d_q_),
                                              k_layer,
                                              v_layer,
                                              static_cast<__half*>(d_att_),
                                              d_decode_position_,
                                              cfg.num_heads,
                                              cfg.num_kv_heads,
                                              head_dim,
                                              compute_stream_,
                                              d_attn_chunk_m_,
                                              d_attn_chunk_l_,
                                              d_attn_chunk_o_,
                                              attn_chunk_capacity_,
                                              !options_.disable_split_attention);

    if (tq && tq->wo) {
      CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_att_, static_cast<std::size_t>(hidden) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, compute_stream_));
      kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_), d_tq3_signs_, hidden, tq3_block_size_, compute_stream_);
      kernels::launch_tq3_gemv_f16(tq->wo, d_tq3_codebook_, tq->s_wo,
                                   static_cast<const __half*>(d_x_tq3_), static_cast<__half*>(d_ff3_),
                                   hidden, hidden, compute_stream_);
      apply_qprod_residual(tq->r_wo, tq->rs_wo,
                           static_cast<const __half*>(d_x_tq3_),
                           static_cast<__half*>(d_ff3_),
                           hidden);
    } else if (cached_int8_proj_enabled_ && lw_i8 && lw_i8->wo) {
      kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_att_),
                                                    static_cast<std::int8_t*>(d_prefill_i8_),
                                                    static_cast<float*>(d_prefill_i8_scales_),
                                                    1,
                                                    hidden,
                                                    compute_stream_);
      if (lw_i8->proj_int4) {
        kernels::launch_weight_only_int4_matvec_dp4a(lw_i8->wo,
                                                     lw_i8->s_wo,
                                                     static_cast<const std::int8_t*>(d_prefill_i8_),
                                                     static_cast<const float*>(d_prefill_i8_scales_),
                                                     static_cast<__half*>(d_ff3_),
                                                     hidden,
                                                     hidden,
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
                                                     hidden,
                                                     compute_stream_,
                                                     resident_int8_wo_warps_,
                                                     resident_int8_wo_tile_packed4_,
                                                     resident_int8_wo_warps_per_row_);
      }
    } else if (resident_custom_wo_) {
      resident_projection_half(
          lw->wo, d_att_, d_ff3_, hidden, hidden, resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
    } else {
      detail::dispatch_linear_rowmajor_weight(cublas_,
                             cublas_lt_,
                             &lt_plan_cache_,
                             lt_workspace_,
                             lt_workspace_bytes_,
                             compute_stream_,
                             lw->wo,
                             d_att_,
                             d_ff3_,
                             hidden,
                             hidden,
                             1,
                             CUDA_R_16F);
    }

    kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                static_cast<const __half*>(d_ff3_),
                                hidden,
                                compute_stream_);

    kernels::launch_rmsnorm(static_cast<const __half*>(d_x_),
                            static_cast<const __half*>(lw->norm_ffn),
                            static_cast<__half*>(d_x_norm_),
                            1,
                            hidden,
                            1e-5f,
                            compute_stream_);

    if (tq && tq->w13) {
      // TQ3 w13: hadamard-rotate x_norm, then packed GEMV.
      CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_x_norm_, static_cast<std::size_t>(hidden) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, compute_stream_));
      kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_), d_tq3_signs_, hidden, tq3_block_size_, compute_stream_);
      kernels::launch_tq3_gemv_f16(tq->w13, d_tq3_codebook_, tq->s_w13,
                                   static_cast<const __half*>(d_x_tq3_), static_cast<__half*>(d_ff1_),
                                   2 * inter, hidden, compute_stream_);
      apply_qprod_residual(tq->r_w13, tq->rs_w13,
                           static_cast<const __half*>(d_x_tq3_),
                           static_cast<__half*>(d_ff1_),
                           2 * inter);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && can_use_dp4a_decode) {
      kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_x_norm_),
                                                    static_cast<std::int8_t*>(d_prefill_i8_),
                                                    static_cast<float*>(d_prefill_i8_scales_),
                                                    1,
                                                    hidden,
                                                    compute_stream_);
      resident_int8_mlp_w13(*lw_i8, inter, hidden);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w3) {
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
      // FP16 MLP fallback in graph capture: use resident_projection_half (always
      // graph-capturable) instead of linear_rowmajor_weight / cuBLASLt which may
      // fall through to cublasGemmEx and cause CUDA_STATUS_INVALID_VALUE crashes.
      resident_projection_half(lw->w13, d_x_norm_, d_ff1_, 2 * inter, hidden,
                               resident_qkv_warps_, resident_qkv_tile_pairs_, resident_qkv_rows_per_warp_);
    }

    kernels::launch_silu_mul(static_cast<const __half*>(d_ff1_),
                             static_cast<const __half*>(d_ff2_),
                             static_cast<__half*>(d_ff2_),
                             inter,
                             compute_stream_);

    if (tq) {
      // w2 stays fp16 for TQ3 (intermediate_size is not power-of-2).
      // Use resident_projection_half — always graph-capturable, consistent with logits graph.
      resident_projection_half(lw->w2, d_ff2_, d_ff3_, hidden, inter,
                               resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && can_use_dp4a_decode) {
      kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_ff2_),
                                                    static_cast<std::int8_t*>(d_prefill_i8_),
                                                    static_cast<float*>(d_prefill_i8_scales_),
                                                    1,
                                                    inter,
                                                    compute_stream_);
      resident_int8_mlp_w2(*lw_i8, hidden, inter);
    } else if (lw_i8 && lw_i8->w2) {
      if (lw_i8->mlp_int4) {
        kernels::launch_weight_only_int4_matvec(
            lw_i8->w2, lw_i8->s_w2, static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_), hidden, inter, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec(
            lw_i8->w2, lw_i8->s_w2, static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_), hidden, inter, compute_stream_);
      }
    } else {
      resident_projection_half(lw->w2, d_ff2_, d_ff3_, hidden, inter,
                               resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
    }

    kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                static_cast<const __half*>(d_ff3_),
                                hidden,
                                compute_stream_);
  }

  kernels::launch_rmsnorm(static_cast<const __half*>(d_x_),
                          static_cast<const __half*>(d_norm_out_),
                          static_cast<__half*>(d_x_norm_),
                          1,
                          hidden,
                          1e-5f,
                          compute_stream_);
  // Always use custom kernel in graph capture: cuBLASLt may fall through to
  // cublasGemmEx which is not graph-capturable, causing INVALID_VALUE errors.
  resident_projection_float(d_lm_head_, d_x_norm_, d_logits_, cfg.vocab_size, hidden,
                             resident_lm_head_warps_, resident_lm_head_tile_pairs_,
                             resident_lm_head_rows_per_warp_);
  kernels::launch_argmax_float(static_cast<const float*>(d_logits_), cfg.vocab_size, d_argmax_, compute_stream_);
  kernels::launch_copy_int(d_argmax_, d_token_id_, compute_stream_);
  kernels::launch_increment_int(d_decode_position_, compute_stream_);

  CUDA_CHECK(cudaStreamEndCapture(compute_stream_, &greedy_decode_graph_));
  if (options_.verbose) {
    std::cout << "[engine] init_greedy_decode_graph: captured, instantiating\n";
  }
  CUDA_CHECK(cudaGraphInstantiate(&greedy_decode_graph_exec_, greedy_decode_graph_, nullptr, nullptr, 0));
  greedy_decode_graph_ready_ = true;
  greedy_decode_graph_state_valid_ = false;
  if (options_.verbose) {
    std::cout << "[engine] greedy_decode_graph: enabled\n";
  }
}

// Same transformer body as init_greedy_decode_graph but without argmax/copy/increment
// at the end — outputs to d_logits_ so the sampling path can read them back to CPU.
void LlamaEngine::init_logits_decode_graph() {
  if (logits_decode_graph_ready_) {
    return;
  }
  if (!can_use_greedy_decode_graph()) {
    return;
  }

  const auto& cfg = weights_.config();
  const int hidden = cfg.hidden_size;
  const int inter = cfg.intermediate_size;
  const int head_dim = cfg.hidden_size / cfg.num_heads;
  const int kv_hidden = cfg.num_kv_heads * head_dim;
  const bool can_use_dp4a_decode = ((hidden & 3) == 0) && ((inter & 3) == 0);
  const auto apply_qprod_residual = [&](const uint32_t* row_bits,
                                        const half* residual_scales,
                                        const half* rotated_x,
                                        half* y,
                                        int out_features) {
    if (!tq_prod_enabled_ || tq_qjl_dim_ <= 0 || !row_bits || !residual_scales || !rotated_x || !y ||
        !d_tq_qjl_indices_ || !d_tq_qjl_signs_ || !d_tq_qjl_x_bits_) {
      return;
    }
    kernels::launch_tq_qjl_pack_sign_bits(rotated_x, d_tq_qjl_indices_, d_tq_qjl_signs_,
                                          d_tq_qjl_x_bits_, tq_qjl_dim_, compute_stream_);
    kernels::launch_tq_qjl_residual_add_f16(row_bits, residual_scales, d_tq_qjl_x_bits_,
                                            y, out_features, tq_qjl_dim_, compute_stream_);
  };

  // Warm cuBLASLt plans before capture. Skip fp16 QKV/WO/MLP warmup when INT8
  // projections are active: those fp16 weight pointers are freed after packing.
  if (!layer_cache_.empty()) {
    const auto& lw = layer_cache_.front();
    if (tq3_enabled_ && !layer_cache_tq3_.empty()) {
      // TQ3: warm packed GEMV kernels; skip any weight that wasn't loaded (null guard).
      const auto& tq = layer_cache_tq3_.front();
      if (tq.wqkv) {
        kernels::launch_tq3_gemv_f16(tq.wqkv, d_tq3_codebook_, tq.s_wqkv,
                                     static_cast<const __half*>(d_x_tq3_), static_cast<__half*>(d_q_),
                                     hidden + 2 * kv_hidden, hidden, compute_stream_);
      }
      if (tq.wo) {
        kernels::launch_tq3_gemv_f16(tq.wo, d_tq3_codebook_, tq.s_wo,
                                     static_cast<const __half*>(d_x_tq3_), static_cast<__half*>(d_ff3_),
                                     hidden, hidden, compute_stream_);
      }
      if (tq.w13) {
        kernels::launch_tq3_gemv_f16(tq.w13, d_tq3_codebook_, tq.s_w13,
                                     static_cast<const __half*>(d_x_tq3_), static_cast<__half*>(d_ff1_),
                                     2 * inter, hidden, compute_stream_);
      }
      // Warm w2 (fp16, always) using resident_projection_half — always graph-capturable.
      resident_projection_half(lw.w2, d_ff2_, d_ff3_, hidden, inter,
                               resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
    } else if (!cached_int8_proj_enabled_) {
      if (resident_custom_qkv_) {
        resident_projection_half(lw.wqkv, d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden,
                                 resident_qkv_warps_, resident_qkv_tile_pairs_, resident_qkv_rows_per_warp_);
      } else {
        detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_,
                               compute_stream_, lw.wqkv, d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden, 1, CUDA_R_16F);
      }
      if (resident_custom_wo_) {
        resident_projection_half(lw.wo, d_att_, d_ff3_, hidden, hidden,
                                 resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      } else {
        detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_,
                               compute_stream_, lw.wo, d_att_, d_ff3_, hidden, hidden, 1, CUDA_R_16F);
      }
    }
    if (!tq3_enabled_) {
      if (cached_int8_mlp_enabled_ && !layer_cache_i8_.empty() && layer_cache_i8_.front().w1) {
        if (can_use_dp4a_decode) {
          resident_int8_mlp_w13(layer_cache_i8_.front(), inter, hidden);
          resident_int8_mlp_w2(layer_cache_i8_.front(), hidden, inter);
        } else {
          if (layer_cache_i8_.front().mlp_int4) {
            kernels::launch_weight_only_int4_matvec(layer_cache_i8_.front().w1, layer_cache_i8_.front().s_w1,
                                                    static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff1_),
                                                    inter, hidden, compute_stream_);
            kernels::launch_weight_only_int4_matvec(layer_cache_i8_.front().w2, layer_cache_i8_.front().s_w2,
                                                    static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_),
                                                    hidden, inter, compute_stream_);
          } else {
            kernels::launch_weight_only_int8_matvec(layer_cache_i8_.front().w1, layer_cache_i8_.front().s_w1,
                                                    static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff1_),
                                                    inter, hidden, compute_stream_);
            kernels::launch_weight_only_int8_matvec(layer_cache_i8_.front().w2, layer_cache_i8_.front().s_w2,
                                                    static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_),
                                                    hidden, inter, compute_stream_);
          }
        }
      } else if (!cached_int8_proj_enabled_) {
        detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_,
                               compute_stream_, lw.w13, d_x_norm_, d_ff1_, 2 * inter, hidden, 1, CUDA_R_16F);
        detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_,
                               compute_stream_, lw.w2, d_ff2_, d_ff3_, hidden, inter, 1, CUDA_R_16F);
      } else if (lw.w13) {
        // INT8 proj + FP16 MLP (e.g., TinyLlama): warm resident_projection_half since
        // cuBLASLt may not find a graph-capturable plan for all dimension combinations.
        resident_projection_half(lw.w13, d_x_norm_, d_ff1_, 2 * inter, hidden,
                                 resident_qkv_warps_, resident_qkv_tile_pairs_, resident_qkv_rows_per_warp_);
        resident_projection_half(lw.w2, d_ff2_, d_ff3_, hidden, inter,
                                 resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      }
    }
    resident_projection_float(d_lm_head_, d_x_norm_, d_logits_, cfg.vocab_size, hidden,
                              resident_lm_head_warps_, resident_lm_head_tile_pairs_, resident_lm_head_rows_per_warp_);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  }

  destroy_logits_decode_graph();
  CUDA_CHECK(cudaStreamBeginCapture(compute_stream_, cudaStreamCaptureModeGlobal));

  kernels::launch_embedding_lookup(static_cast<const __half*>(d_tok_embeddings_),
                                   d_token_id_,
                                   static_cast<__half*>(d_x_),
                                   1,
                                   hidden,
                                   compute_stream_);

  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    const auto* lw = &layer_cache_[static_cast<std::size_t>(layer)];
    const LayerDeviceInt8Weights* lw_i8 =
        (layer < static_cast<int>(layer_cache_i8_.size()) &&
         (layer_cache_i8_[static_cast<std::size_t>(layer)].w1 ||
          layer_cache_i8_[static_cast<std::size_t>(layer)].wqkv))
            ? &layer_cache_i8_[static_cast<std::size_t>(layer)]
            : nullptr;
    const LayerDeviceTq3Weights* tq =
        (tq3_enabled_ && layer < static_cast<int>(layer_cache_tq3_.size()))
            ? &layer_cache_tq3_[static_cast<std::size_t>(layer)]
            : nullptr;
    auto* k_layer = static_cast<__half*>(d_k_cache_) +
                    static_cast<std::size_t>(layer) * static_cast<std::size_t>(options_.max_context) *
                        static_cast<std::size_t>(kv_hidden);
    auto* v_layer = static_cast<__half*>(d_v_cache_) +
                    static_cast<std::size_t>(layer) * static_cast<std::size_t>(options_.max_context) *
                        static_cast<std::size_t>(kv_hidden);

    kernels::launch_rmsnorm(static_cast<const __half*>(d_x_), static_cast<const __half*>(lw->norm_att),
                            static_cast<__half*>(d_x_norm_), 1, hidden, 1e-5f, compute_stream_);

    if (tq && tq->wqkv) {
      CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_x_norm_, static_cast<std::size_t>(hidden) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, compute_stream_));
      kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_), d_tq3_signs_, hidden, tq3_block_size_, compute_stream_);
      kernels::launch_tq3_gemv_f16(tq->wqkv, d_tq3_codebook_, tq->s_wqkv,
                                   static_cast<const __half*>(d_x_tq3_), static_cast<__half*>(d_q_),
                                   hidden + 2 * kv_hidden, hidden, compute_stream_);
      apply_qprod_residual(tq->r_wqkv, tq->rs_wqkv,
                           static_cast<const __half*>(d_x_tq3_),
                           static_cast<__half*>(d_q_),
                           hidden + 2 * kv_hidden);
    } else if (cached_int8_proj_enabled_ && lw_i8 && lw_i8->wqkv) {
      kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_x_norm_),
                                                    static_cast<std::int8_t*>(d_prefill_i8_),
                                                    static_cast<float*>(d_prefill_i8_scales_),
                                                    1, hidden, compute_stream_);
      if (lw_i8->proj_int4) {
        kernels::launch_weight_only_int4_matvec_dp4a(lw_i8->wqkv, lw_i8->s_wqkv,
                                                     static_cast<const std::int8_t*>(d_prefill_i8_),
                                                     static_cast<const float*>(d_prefill_i8_scales_),
                                                     static_cast<__half*>(d_q_), hidden + 2 * kv_hidden, hidden,
                                                     compute_stream_, resident_int8_qkv_warps_,
                                                     resident_int8_qkv_tile_packed4_, resident_int8_qkv_warps_per_row_);
      } else {
        kernels::launch_weight_only_int8_matvec_dp4a(lw_i8->wqkv, lw_i8->s_wqkv,
                                                     static_cast<const std::int8_t*>(d_prefill_i8_),
                                                     static_cast<const float*>(d_prefill_i8_scales_),
                                                     static_cast<__half*>(d_q_), hidden + 2 * kv_hidden, hidden,
                                                     compute_stream_, resident_int8_qkv_warps_,
                                                     resident_int8_qkv_tile_packed4_, resident_int8_qkv_warps_per_row_);
      }
    } else if (resident_custom_qkv_) {
      resident_projection_half(lw->wqkv, d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden,
                               resident_qkv_warps_, resident_qkv_tile_pairs_);
    } else {
      detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_,
                             compute_stream_, lw->wqkv, d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden, 1, CUDA_R_16F);
    }

    if (lw->bqkv) {
      kernels::launch_add_inplace(static_cast<__half*>(d_q_), static_cast<const __half*>(lw->bqkv),
                                  hidden + 2 * kv_hidden, compute_stream_);
    }

    kernels::launch_rope_inplace_device_pos(static_cast<__half*>(d_q_), static_cast<__half*>(d_k_),
                                            cfg.num_heads, cfg.num_kv_heads, head_dim,
                                            d_decode_position_, d_rope_cos_, d_rope_sin_, compute_stream_);
    kernels::launch_store_kv_device_pos(static_cast<const __half*>(d_k_), static_cast<const __half*>(d_v_),
                                        k_layer, v_layer, d_decode_position_, kv_hidden,
                                        options_.max_context, compute_stream_);
    kernels::launch_attention_step_device_pos(static_cast<const __half*>(d_q_), k_layer, v_layer,
                                              static_cast<__half*>(d_att_), d_decode_position_,
                                              cfg.num_heads, cfg.num_kv_heads, head_dim, compute_stream_,
                                              d_attn_chunk_m_, d_attn_chunk_l_, d_attn_chunk_o_,
                                              attn_chunk_capacity_, !options_.disable_split_attention);

    if (tq && tq->wo) {
      CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_att_, static_cast<std::size_t>(hidden) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, compute_stream_));
      kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_), d_tq3_signs_, hidden, tq3_block_size_, compute_stream_);
      kernels::launch_tq3_gemv_f16(tq->wo, d_tq3_codebook_, tq->s_wo,
                                   static_cast<const __half*>(d_x_tq3_), static_cast<__half*>(d_ff3_),
                                   hidden, hidden, compute_stream_);
      apply_qprod_residual(tq->r_wo, tq->rs_wo,
                           static_cast<const __half*>(d_x_tq3_),
                           static_cast<__half*>(d_ff3_),
                           hidden);
    } else if (cached_int8_proj_enabled_ && lw_i8 && lw_i8->wo) {
      kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_att_),
                                                    static_cast<std::int8_t*>(d_prefill_i8_),
                                                    static_cast<float*>(d_prefill_i8_scales_),
                                                    1, hidden, compute_stream_);
      if (lw_i8->proj_int4) {
        kernels::launch_weight_only_int4_matvec_dp4a(lw_i8->wo, lw_i8->s_wo,
                                                     static_cast<const std::int8_t*>(d_prefill_i8_),
                                                     static_cast<const float*>(d_prefill_i8_scales_),
                                                     static_cast<__half*>(d_ff3_), hidden, hidden,
                                                     compute_stream_, resident_int8_wo_warps_,
                                                     resident_int8_wo_tile_packed4_, resident_int8_wo_warps_per_row_);
      } else {
        kernels::launch_weight_only_int8_matvec_dp4a(lw_i8->wo, lw_i8->s_wo,
                                                     static_cast<const std::int8_t*>(d_prefill_i8_),
                                                     static_cast<const float*>(d_prefill_i8_scales_),
                                                     static_cast<__half*>(d_ff3_), hidden, hidden,
                                                     compute_stream_, resident_int8_wo_warps_,
                                                     resident_int8_wo_tile_packed4_, resident_int8_wo_warps_per_row_);
      }
    } else if (resident_custom_wo_) {
      resident_projection_half(lw->wo, d_att_, d_ff3_, hidden, hidden,
                               resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
    } else {
      detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_,
                             compute_stream_, lw->wo, d_att_, d_ff3_, hidden, hidden, 1, CUDA_R_16F);
    }

    kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                hidden, compute_stream_);

    kernels::launch_rmsnorm(static_cast<const __half*>(d_x_), static_cast<const __half*>(lw->norm_ffn),
                            static_cast<__half*>(d_x_norm_), 1, hidden, 1e-5f, compute_stream_);

    if (tq && tq->w13) {
      CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_x_norm_, static_cast<std::size_t>(hidden) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, compute_stream_));
      kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_), d_tq3_signs_, hidden, tq3_block_size_, compute_stream_);
      kernels::launch_tq3_gemv_f16(tq->w13, d_tq3_codebook_, tq->s_w13,
                                   static_cast<const __half*>(d_x_tq3_), static_cast<__half*>(d_ff1_),
                                   2 * inter, hidden, compute_stream_);
      apply_qprod_residual(tq->r_w13, tq->rs_w13,
                           static_cast<const __half*>(d_x_tq3_),
                           static_cast<__half*>(d_ff1_),
                           2 * inter);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && can_use_dp4a_decode) {
      kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_x_norm_),
                                                    static_cast<std::int8_t*>(d_prefill_i8_),
                                                    static_cast<float*>(d_prefill_i8_scales_),
                                                    1, hidden, compute_stream_);
      resident_int8_mlp_w13(*lw_i8, inter, hidden);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w3) {
      if (lw_i8->mlp_int4) {
        kernels::launch_weight_only_int4_matvec(lw_i8->w1, lw_i8->s_w1,
                                                static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff1_),
                                                inter, hidden, compute_stream_);
        kernels::launch_weight_only_int4_matvec(lw_i8->w3, lw_i8->s_w3,
                                                static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff2_),
                                                inter, hidden, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec(lw_i8->w1, lw_i8->s_w1,
                                                static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff1_),
                                                inter, hidden, compute_stream_);
        kernels::launch_weight_only_int8_matvec(lw_i8->w3, lw_i8->s_w3,
                                                static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff2_),
                                                inter, hidden, compute_stream_);
      }
    } else {
      // FP16 MLP fallback in graph capture: use resident_projection_half (always graph-capturable).
      resident_projection_half(lw->w13, d_x_norm_, d_ff1_, 2 * inter, hidden,
                               resident_qkv_warps_, resident_qkv_tile_pairs_, resident_qkv_rows_per_warp_);
    }

    kernels::launch_silu_mul(static_cast<const __half*>(d_ff1_), static_cast<const __half*>(d_ff2_),
                             static_cast<__half*>(d_ff2_), inter, compute_stream_);

    if (tq) {
      resident_projection_half(lw->w2, d_ff2_, d_ff3_, hidden, inter,
                               resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && can_use_dp4a_decode) {
      kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_ff2_),
                                                    static_cast<std::int8_t*>(d_prefill_i8_),
                                                    static_cast<float*>(d_prefill_i8_scales_),
                                                    1, inter, compute_stream_);
      resident_int8_mlp_w2(*lw_i8, hidden, inter);
    } else if (lw_i8 && lw_i8->w2) {
      if (lw_i8->mlp_int4) {
        kernels::launch_weight_only_int4_matvec(lw_i8->w2, lw_i8->s_w2,
                                                static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_),
                                                hidden, inter, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec(lw_i8->w2, lw_i8->s_w2,
                                                static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_),
                                                hidden, inter, compute_stream_);
      }
    } else {
      resident_projection_half(lw->w2, d_ff2_, d_ff3_, hidden, inter,
                               resident_wo_warps_, resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
    }

    kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                hidden, compute_stream_);
  }

  kernels::launch_rmsnorm(static_cast<const __half*>(d_x_), static_cast<const __half*>(d_norm_out_),
                          static_cast<__half*>(d_x_norm_), 1, hidden, 1e-5f, compute_stream_);
  // Always use custom kernel in graph capture — same reason as greedy graph.
  resident_projection_float(d_lm_head_, d_x_norm_, d_logits_, cfg.vocab_size, hidden,
                             resident_lm_head_warps_, resident_lm_head_tile_pairs_,
                             resident_lm_head_rows_per_warp_);
  // Note: no argmax/copy_int/increment_int — caller reads d_logits_ for sampling.

  CUDA_CHECK(cudaStreamEndCapture(compute_stream_, &logits_decode_graph_));
  CUDA_CHECK(cudaGraphInstantiate(&logits_decode_graph_exec_, logits_decode_graph_, nullptr, nullptr, 0));
  logits_decode_graph_ready_ = true;
  if (options_.verbose) {
    std::cout << "[engine] logits_decode_graph: enabled\n";
  }
}

void LlamaEngine::decode_next_token_logits_graph(int token, int position, std::vector<float>& h_logits) {
  if (!logits_decode_graph_ready_) {
    init_logits_decode_graph();
  }
  if (!logits_decode_graph_ready_) {
    forward_token_logits(token, position, &h_logits, nullptr);
    return;
  }
  // Always upload token+position since the graph doesn't auto-increment position.
  CUDA_CHECK(cudaMemcpyAsync(d_token_id_, &token, sizeof(int), cudaMemcpyHostToDevice, compute_stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_decode_position_, &position, sizeof(int), cudaMemcpyHostToDevice, compute_stream_));
  CUDA_CHECK(cudaGraphLaunch(logits_decode_graph_exec_, compute_stream_));
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  const int vocab = weights_.config().vocab_size;
  h_logits.resize(static_cast<std::size_t>(vocab));
  CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits_,
                        static_cast<std::size_t>(vocab) * sizeof(float), cudaMemcpyDeviceToHost));
}

int LlamaEngine::decode_next_token_graph(int token, int position) {
  if (!greedy_decode_graph_ready_) {
    init_greedy_decode_graph();
  }
  if (!greedy_decode_graph_ready_) {
    int next = 0;
    forward_token_logits(token, position, nullptr, &next);
    return next;
  }

  if (!greedy_decode_graph_state_valid_ || token != greedy_decode_graph_expected_token_ ||
      position != greedy_decode_graph_expected_position_) {
    CUDA_CHECK(cudaMemcpyAsync(d_token_id_, &token, sizeof(int), cudaMemcpyHostToDevice, compute_stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_decode_position_, &position, sizeof(int), cudaMemcpyHostToDevice, compute_stream_));
    greedy_decode_graph_state_valid_ = true;
    greedy_decode_graph_expected_token_ = token;
    greedy_decode_graph_expected_position_ = position;
  }
  CUDA_CHECK(cudaGraphLaunch(greedy_decode_graph_exec_, compute_stream_));
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));

  int next = 0;
  CUDA_CHECK(cudaMemcpy(&next, d_argmax_, sizeof(int), cudaMemcpyDeviceToHost));
  greedy_decode_graph_expected_token_ = next;
  greedy_decode_graph_expected_position_ = position + 1;
  return next;
}



}  // namespace engine
