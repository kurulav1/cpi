#include <cuda_fp16.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>

#include "engine/llama_engine.hpp"
#include "engine/sampling.hpp"
#include "llama_engine_internal.hpp"
#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"

namespace engine {

bool LlamaEngine::can_use_greedy_decode_graph() const {
  // Greedy decode CUDA graph. Verified byte-identical to the non-graph path for
  // fp16, int8 and int4 once the RMSNorm eps was sourced from the model config
  // (it had been hardcoded to 1e-5, which corrupted models like Qwen2.5 that use
  // 1e-6; the corruption compounded badly through int8/int4 quantization).
  // Requires: all layers resident, full attention, standard attention dims, and
  // no biases / LayerNorm / paged-KV / int4-KV / MoE / TQ3 (TQ3 untested in the
  // graph and unused here).
  const auto& cfg = weights_.config();
  // The graph's device-position split-K attention launches a fixed grid sized to
  // the full max_context (scratch_chunks = ceil(max_context/32)) every step,
  // since the grid can't depend on the runtime seq_len once captured. For large
  // max_context that fixed grid dwarfs the per-step launch savings and makes
  // decode scale with the *allocated* context rather than the *filled* one (e.g.
  // Qwen2.5-7B: ~82 tok/s at 128K but ~26 at 256K on an 8-token prompt). Above
  // this threshold, skip the graph so decode uses the host-launched path whose
  // grid is min(scratch_chunks, ceil(seq_len/32)), i.e. scales with filled tokens.
  constexpr int kGreedyGraphMaxContext = 32768;
  return cached_layer_count_ == cfg.num_layers && !options_.paged_kv_cache &&
         options_.max_context <= kGreedyGraphMaxContext &&
         !options_.paged_blocks &&  // paged decode uses the non-graph split-K block-gather path
         !options_.profile_decode_phases && !kv_int4_enabled_ && !tq3_enabled_ && !cfg.is_moe() &&
         !cfg.use_layernorm && !cfg.has_qk_norm && !cfg.scale_embeddings &&
         (attn_q_hidden_ <= 0 || attn_q_hidden_ == cfg.hidden_size) && !has_any_layer_norm_bias_ &&
         !has_any_layer_output_bias_ && !weights_.has_tensor("norm.bias") &&
         !weights_.has_tensor("output.bias") && !cfg.uses_non_full_attention();
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
  const float norm_eps = cfg.norm_eps > 0.0f ? cfg.norm_eps : 1e-5f;
  const int hidden = cfg.hidden_size;
  const int inter = cfg.intermediate_size;
  const int head_dim = cfg.hidden_size / cfg.num_heads;
  const int kv_hidden = cfg.num_kv_heads * head_dim;
  const bool can_use_dp4a_decode = ((hidden & 3) == 0) && ((inter & 3) == 0);
  const auto apply_qprod_residual = [&](const uint32_t* row_bits, const half* residual_scales,
                                        const half* rotated_x, half* y, int out_features) {
    if (!tq_prod_enabled_ || tq_qjl_dim_ <= 0 || !row_bits || !residual_scales || !rotated_x ||
        !y || !d_tq_qjl_indices_ || !d_tq_qjl_signs_ || !d_tq_qjl_x_bits_) {
      return;
    }
    kernels::launch_tq_qjl_pack_sign_bits(rotated_x, d_tq_qjl_indices_, d_tq_qjl_signs_,
                                          d_tq_qjl_x_bits_, tq_qjl_dim_, compute_stream_);
    kernels::launch_tq_qjl_residual_add_f16(row_bits, residual_scales, d_tq_qjl_x_bits_, y,
                                            out_features, tq_qjl_dim_, compute_stream_);
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
        kernels::launch_tq3_gemv_f16(
            tq.wqkv, d_tq3_codebook_, tq.s_wqkv, static_cast<const __half*>(d_x_tq3_),
            static_cast<__half*>(d_q_), hidden + 2 * kv_hidden, hidden, compute_stream_);
      }
      if (tq.wo) {
        kernels::launch_tq3_gemv_f16(tq.wo, d_tq3_codebook_, tq.s_wo,
                                     static_cast<const __half*>(d_x_tq3_),
                                     static_cast<__half*>(d_ff3_), hidden, hidden, compute_stream_);
      }
      if (tq.w13) {
        kernels::launch_tq3_gemv_f16(
            tq.w13, d_tq3_codebook_, tq.s_w13, static_cast<const __half*>(d_x_tq3_),
            static_cast<__half*>(d_ff1_), 2 * inter, hidden, compute_stream_);
      }
      // Warm w2 (fp16 under TQ3) using resident_projection_half; always graph-capturable.
      if (lw.w2 != nullptr) {
        resident_projection_half(lw.w2, d_ff2_, d_ff3_, hidden, inter, resident_wo_warps_,
                                 resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      }
    } else if (!cached_int8_proj_enabled_) {
      if (resident_custom_qkv_) {
        if (!packed_qkv_matvec(lw, d_x_norm_, d_q_, compute_stream_)) {
          resident_projection_half(lw.wqkv, d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden,
                                   resident_qkv_warps_, resident_qkv_tile_pairs_,
                                   resident_qkv_rows_per_warp_);
        }
      } else if (!packed_qkv_matvec(lw, d_x_norm_, d_q_, compute_stream_)) {
        detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                                lt_workspace_bytes_, compute_stream_, lw.wqkv,
                                                d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden, 1,
                                                CUDA_R_16F);
      }
      if (lw.wo_packed.active()) {
        // Warm what capture will actually record. A packed matrix has no fp16
        // buffer, so warming the fp16 kernel would hand cuBLAS a null pointer.
        kernels::launch_kquant_matvec(static_cast<const std::uint8_t*>(lw.wo_packed.data),
                                      static_cast<kernels::KQuantType>(lw.wo_packed.kind),
                                      static_cast<const __half*>(d_att_),
                                      static_cast<__half*>(d_ff3_), lw.wo_packed.rows,
                                      lw.wo_packed.cols, compute_stream_);
      } else if (resident_custom_wo_) {
        resident_projection_half(lw.wo, d_att_, d_ff3_, hidden, hidden, resident_wo_warps_,
                                 resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      } else {
        detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                                lt_workspace_bytes_, compute_stream_, lw.wo, d_att_,
                                                d_ff3_, hidden, hidden, 1, CUDA_R_16F);
      }
    }
    if (!tq3_enabled_) {
      if (cached_int8_mlp_enabled_ && !layer_cache_i8_.empty() && layer_cache_i8_.front().w1) {
        if (can_use_dp4a_decode) {
          resident_int8_mlp_w13(layer_cache_i8_.front(), inter, hidden);
          resident_int8_mlp_w2(layer_cache_i8_.front(), hidden, inter);
        } else {
          if (layer_cache_i8_.front().mlp_int4) {
            kernels::launch_weight_only_int4_matvec(
                layer_cache_i8_.front().w1, layer_cache_i8_.front().s_w1,
                static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff1_), inter, hidden,
                compute_stream_);
            kernels::launch_weight_only_int4_matvec(
                layer_cache_i8_.front().w2, layer_cache_i8_.front().s_w2,
                static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_), hidden, inter,
                compute_stream_);
          } else {
            kernels::launch_weight_only_int8_matvec(
                layer_cache_i8_.front().w1, layer_cache_i8_.front().s_w1,
                static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff1_), inter, hidden,
                compute_stream_);
            kernels::launch_weight_only_int8_matvec(
                layer_cache_i8_.front().w2, layer_cache_i8_.front().s_w2,
                static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_), hidden, inter,
                compute_stream_);
          }
        }
      } else if (!cached_int8_proj_enabled_) {
        if (lw.w13_packed.active()) {
          kernels::launch_kquant_matvec(static_cast<const std::uint8_t*>(lw.w13_packed.data),
                                        static_cast<kernels::KQuantType>(lw.w13_packed.kind),
                                        static_cast<const __half*>(d_x_norm_),
                                        static_cast<__half*>(d_ff1_), lw.w13_packed.rows,
                                        lw.w13_packed.cols, compute_stream_);
        } else {
          detail::dispatch_linear_rowmajor_weight(
              cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_,
              compute_stream_, lw.w13, d_x_norm_, d_ff1_, 2 * inter, hidden, 1, CUDA_R_16F);
        }
        if (lw.w2_packed.active()) {
          kernels::launch_kquant_matvec(static_cast<const std::uint8_t*>(lw.w2_packed.data),
                                        static_cast<kernels::KQuantType>(lw.w2_packed.kind),
                                        static_cast<const __half*>(d_ff2_),
                                        static_cast<__half*>(d_ff3_), lw.w2_packed.rows,
                                        lw.w2_packed.cols, compute_stream_);
        } else {
          detail::dispatch_linear_rowmajor_weight(
              cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_,
              compute_stream_, lw.w2, d_ff2_, d_ff3_, hidden, inter, 1, CUDA_R_16F);
        }
      } else if (lw.w13) {
        // INT8 proj + FP16 MLP (e.g., TinyLlama): warm resident_projection_half since
        // cuBLASLt may not find a graph-capturable plan for all dimension combinations.
        resident_projection_half(lw.w13, d_x_norm_, d_ff1_, 2 * inter, hidden, resident_qkv_warps_,
                                 resident_qkv_tile_pairs_, resident_qkv_rows_per_warp_);
        resident_projection_half(lw.w2, d_ff2_, d_ff3_, hidden, inter, resident_wo_warps_,
                                 resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      }
    }
    // Always warm the custom LM-head kernel (used in all graph captures to avoid cuBLAS fallback
    // issues).
    project_lm_head_logits(static_cast<const __half*>(d_x_norm_), static_cast<float*>(d_logits_));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  }

  destroy_greedy_decode_graph();
  CUDA_CHECK(cudaStreamBeginCapture(compute_stream_, cudaStreamCaptureModeGlobal));

  kernels::launch_embedding_lookup(static_cast<const __half*>(d_tok_embeddings_), d_token_id_,
                                   static_cast<__half*>(d_x_), 1, hidden, compute_stream_);

  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    // Set when a projection folded the residual add into its own epilogue, so the shared
    // add_inplace below is skipped. The int8 / tq3 branches do not fold it and still need
    // the add; moving the add into one branch would silently drop the residual for them.
    bool fused_residual = false;
    const auto* lw = &layer_cache_[static_cast<std::size_t>(layer)];
    // Include lw_i8 when either int8 MLP weights (w1) or int8 projection weights
    // (wqkv) are present.  Using only w1 as the gate causes lw_i8=nullptr for
    // models like TinyLlama that have int8 projections but no int8 MLP, which
    // falls through to cublasGemmEx inside graph capture and triggers INVALID_VALUE.
    const LayerDeviceInt8Weights* lw_i8 = (layer < static_cast<int>(layer_cache_i8_.size()) &&
                                           (layer_cache_i8_[static_cast<std::size_t>(layer)].w1 ||
                                            layer_cache_i8_[static_cast<std::size_t>(layer)].wqkv))
                                              ? &layer_cache_i8_[static_cast<std::size_t>(layer)]
                                              : nullptr;
    const LayerDeviceTq3Weights* tq =
        (tq3_enabled_ && layer < static_cast<int>(layer_cache_tq3_.size()))
            ? &layer_cache_tq3_[static_cast<std::size_t>(layer)]
            : nullptr;
    auto* k_layer =
        static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) *
                                               static_cast<std::size_t>(options_.max_context) *
                                               static_cast<std::size_t>(kv_hidden);
    auto* v_layer =
        static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) *
                                               static_cast<std::size_t>(options_.max_context) *
                                               static_cast<std::size_t>(kv_hidden);

    kernels::launch_rmsnorm(static_cast<const __half*>(d_x_),
                            static_cast<const __half*>(lw->norm_att),
                            static_cast<__half*>(d_x_norm_), 1, hidden, norm_eps, compute_stream_);

    if (tq && tq->wqkv) {
      CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_x_norm_,
                                 static_cast<std::size_t>(hidden) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, compute_stream_));
      kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_), d_tq3_signs_, hidden,
                                           tq3_block_size_, compute_stream_);
      kernels::launch_tq3_gemv_f16(tq->wqkv, d_tq3_codebook_, tq->s_wqkv,
                                   static_cast<const __half*>(d_x_tq3_), static_cast<__half*>(d_q_),
                                   hidden + 2 * kv_hidden, hidden, compute_stream_);
      apply_qprod_residual(tq->r_wqkv, tq->rs_wqkv, static_cast<const __half*>(d_x_tq3_),
                           static_cast<__half*>(d_q_), hidden + 2 * kv_hidden);
    } else if (cached_int8_proj_enabled_ && lw_i8 && lw_i8->wqkv) {
      kernels::launch_quantize_rowwise_fp16_to_int8(
          static_cast<const __half*>(d_x_norm_), static_cast<std::int8_t*>(d_prefill_i8_),
          static_cast<float*>(d_prefill_i8_scales_), 1, hidden, compute_stream_);
      if (lw_i8->proj_int4) {
        kernels::launch_weight_only_int4_matvec_dp4a(
            lw_i8->wqkv, lw_i8->s_wqkv, static_cast<const std::int8_t*>(d_prefill_i8_),
            static_cast<const float*>(d_prefill_i8_scales_), static_cast<__half*>(d_q_),
            hidden + 2 * kv_hidden, hidden, compute_stream_, resident_int8_qkv_warps_,
            resident_int8_qkv_tile_packed4_, resident_int8_qkv_warps_per_row_);
      } else {
        kernels::launch_weight_only_int8_matvec_dp4a(
            lw_i8->wqkv, lw_i8->s_wqkv, static_cast<const std::int8_t*>(d_prefill_i8_),
            static_cast<const float*>(d_prefill_i8_scales_), static_cast<__half*>(d_q_),
            hidden + 2 * kv_hidden, hidden, compute_stream_, resident_int8_qkv_warps_,
            resident_int8_qkv_tile_packed4_, resident_int8_qkv_warps_per_row_);
      }
    } else if (resident_custom_qkv_) {
      if (!packed_qkv_matvec(*lw, d_x_norm_, d_q_, compute_stream_)) {
        resident_projection_half(lw->wqkv, d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden,
                                 resident_qkv_warps_, resident_qkv_tile_pairs_);
      }
    } else if (!packed_qkv_matvec(*lw, d_x_norm_, d_q_, compute_stream_)) {
      detail::dispatch_linear_rowmajor_weight(
          cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
          lw->wqkv, d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden, 1, CUDA_R_16F);
    }

    if (lw->bqkv) {
      kernels::launch_add_inplace(static_cast<__half*>(d_q_), static_cast<const __half*>(lw->bqkv),
                                  hidden + 2 * kv_hidden, compute_stream_);
    }

    kernels::launch_rope_inplace_device_pos(
        static_cast<__half*>(d_q_), static_cast<__half*>(d_k_), cfg.num_heads, cfg.num_kv_heads,
        head_dim, d_decode_position_, d_rope_cos_, d_rope_sin_, compute_stream_);
    kernels::launch_store_kv_device_pos(
        static_cast<const __half*>(d_k_), static_cast<const __half*>(d_v_), k_layer, v_layer,
        d_decode_position_, kv_hidden, options_.max_context, compute_stream_);
    kernels::launch_attention_step_device_pos(
        static_cast<const __half*>(d_q_), k_layer, v_layer, static_cast<__half*>(d_att_),
        d_decode_position_, cfg.num_heads, cfg.num_kv_heads, head_dim, compute_stream_,
        d_attn_chunk_m_, d_attn_chunk_l_, d_attn_chunk_o_, attn_chunk_capacity_,
        !options_.disable_split_attention);

    if (tq && tq->wo) {
      CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_att_,
                                 static_cast<std::size_t>(hidden) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, compute_stream_));
      kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_), d_tq3_signs_, hidden,
                                           tq3_block_size_, compute_stream_);
      kernels::launch_tq3_gemv_f16(tq->wo, d_tq3_codebook_, tq->s_wo,
                                   static_cast<const __half*>(d_x_tq3_),
                                   static_cast<__half*>(d_ff3_), hidden, hidden, compute_stream_);
      apply_qprod_residual(tq->r_wo, tq->rs_wo, static_cast<const __half*>(d_x_tq3_),
                           static_cast<__half*>(d_ff3_), hidden);
    } else if (cached_int8_proj_enabled_ && lw_i8 && lw_i8->wo) {
      kernels::launch_quantize_rowwise_fp16_to_int8(
          static_cast<const __half*>(d_att_), static_cast<std::int8_t*>(d_prefill_i8_),
          static_cast<float*>(d_prefill_i8_scales_), 1, hidden, compute_stream_);
      if (lw_i8->proj_int4) {
        kernels::launch_weight_only_int4_matvec_dp4a(
            lw_i8->wo, lw_i8->s_wo, static_cast<const std::int8_t*>(d_prefill_i8_),
            static_cast<const float*>(d_prefill_i8_scales_), static_cast<__half*>(d_ff3_), hidden,
            hidden, compute_stream_, resident_int8_wo_warps_, resident_int8_wo_tile_packed4_,
            resident_int8_wo_warps_per_row_);
      } else {
        kernels::launch_weight_only_int8_matvec_dp4a(
            lw_i8->wo, lw_i8->s_wo, static_cast<const std::int8_t*>(d_prefill_i8_),
            static_cast<const float*>(d_prefill_i8_scales_), static_cast<__half*>(d_ff3_), hidden,
            hidden, compute_stream_, resident_int8_wo_warps_, resident_int8_wo_tile_packed4_,
            resident_int8_wo_warps_per_row_);
      }
    } else if (lw->wo_packed.active()) {
      // Packed k-quant: multiply the container's own blocks. Nothing is fused
      // into this one, so the shared add_inplace below still runs.
      ++packed_matvec_calls_;
      kernels::launch_kquant_matvec(static_cast<const std::uint8_t*>(lw->wo_packed.data),
                                    static_cast<kernels::KQuantType>(lw->wo_packed.kind),
                                    static_cast<const __half*>(d_att_), static_cast<__half*>(d_ff3_),
                                    lw->wo_packed.rows, lw->wo_packed.cols, compute_stream_);
    } else if (resident_custom_wo_) {
      // Residual add folded into this projection's epilogue, so the shared add_inplace
      // below is skipped. At batch 1 a kernel costs a fixed ~1.7 us whatever it does.
      resident_projection_half_residual(lw->wo, d_att_, d_x_, hidden, hidden, resident_wo_warps_,
                                        resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      fused_residual = true;
    } else {
      detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                              lt_workspace_bytes_, compute_stream_, lw->wo, d_att_,
                                              d_ff3_, hidden, hidden, 1, CUDA_R_16F);
    }

    if (!fused_residual) {
      kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                  hidden, compute_stream_);
    }
    fused_residual = false;

    kernels::launch_rmsnorm(static_cast<const __half*>(d_x_),
                            static_cast<const __half*>(lw->norm_ffn),
                            static_cast<__half*>(d_x_norm_), 1, hidden, norm_eps, compute_stream_);

    // True when the gate+up GEMV and silu_mul collapsed into one kernel.
    bool fused_glu = false;
    if (tq && tq->w13) {
      // TQ3 w13: hadamard-rotate x_norm, then packed GEMV.
      CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_x_norm_,
                                 static_cast<std::size_t>(hidden) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, compute_stream_));
      kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_), d_tq3_signs_, hidden,
                                           tq3_block_size_, compute_stream_);
      kernels::launch_tq3_gemv_f16(
          tq->w13, d_tq3_codebook_, tq->s_w13, static_cast<const __half*>(d_x_tq3_),
          static_cast<__half*>(d_ff1_), 2 * inter, hidden, compute_stream_);
      apply_qprod_residual(tq->r_w13, tq->rs_w13, static_cast<const __half*>(d_x_tq3_),
                           static_cast<__half*>(d_ff1_), 2 * inter);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && can_use_dp4a_decode) {
      kernels::launch_quantize_rowwise_fp16_to_int8(
          static_cast<const __half*>(d_x_norm_), static_cast<std::int8_t*>(d_prefill_i8_),
          static_cast<float*>(d_prefill_i8_scales_), 1, hidden, compute_stream_);
      resident_int8_mlp_w13(*lw_i8, inter, hidden);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w3) {
      if (lw_i8->mlp_int4) {
        kernels::launch_weight_only_int4_matvec(
            lw_i8->w1, lw_i8->s_w1, static_cast<const __half*>(d_x_norm_),
            static_cast<__half*>(d_ff1_), inter, hidden, compute_stream_);
        kernels::launch_weight_only_int4_matvec(
            lw_i8->w3, lw_i8->s_w3, static_cast<const __half*>(d_x_norm_),
            static_cast<__half*>(d_ff2_), inter, hidden, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec(
            lw_i8->w1, lw_i8->s_w1, static_cast<const __half*>(d_x_norm_),
            static_cast<__half*>(d_ff1_), inter, hidden, compute_stream_);
        kernels::launch_weight_only_int8_matvec(
            lw_i8->w3, lw_i8->s_w3, static_cast<const __half*>(d_x_norm_),
            static_cast<__half*>(d_ff2_), inter, hidden, compute_stream_);
      }
      fused_glu = false;
    } else if (lw->w13_packed.active()) {
      // Gate and up share one packed buffer, so a single matvec covers both row
      // blocks and the gated activation below finishes the job.
      ++packed_matvec_calls_;
      kernels::launch_kquant_matvec(static_cast<const std::uint8_t*>(lw->w13_packed.data),
                                    static_cast<kernels::KQuantType>(lw->w13_packed.kind),
                                    static_cast<const __half*>(d_x_norm_),
                                    static_cast<__half*>(d_ff1_), lw->w13_packed.rows,
                                    lw->w13_packed.cols, compute_stream_);
      fused_glu = false;
    } else if (!weights_.config().mlp_gelu) {
      // fused: gate+up GEMV and silu_mul in one kernel. At batch 1 every kernel costs a
      // fixed ~2.7 us of scheduling regardless of how little it does, and that tax; not
      // bandwidth; is what holds a small model off the roofline. The fused kernel rounds
      // g and u to fp16 before silu(g)*u, exactly as the unfused path does when it stores
      // them between kernels, so the output is byte-identical.
      kernels::launch_swiglu_gemv_f16(
          static_cast<const __half*>(lw->w13),
          static_cast<const __half*>(lw->w13) + static_cast<std::size_t>(inter) * hidden,
          static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff2_), inter, hidden,
          compute_stream_);
      fused_glu = true;
    } else {
      // FP16 MLP fallback in graph capture: use resident_projection_half (always
      // graph-capturable) instead of linear_rowmajor_weight / cuBLASLt which may
      // fall through to cublasGemmEx and cause CUDA_STATUS_INVALID_VALUE crashes.
      resident_projection_half(lw->w13, d_x_norm_, d_ff1_, 2 * inter, hidden, resident_qkv_warps_,
                               resident_qkv_tile_pairs_, resident_qkv_rows_per_warp_);
      fused_glu = false;
    }

    if (!fused_glu) {
      detail::launch_gated_glu(weights_.config().mlp_gelu, static_cast<const __half*>(d_ff1_),
                               static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff2_),
                               inter, compute_stream_);
    }

    if (tq) {
      // w2 stays fp16 for TQ3 (intermediate_size is not power-of-2).
      // Use resident_projection_half; always graph-capturable, consistent with logits graph.
      resident_projection_half(lw->w2, d_ff2_, d_ff3_, hidden, inter, resident_wo_warps_,
                               resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && can_use_dp4a_decode) {
      kernels::launch_quantize_rowwise_fp16_to_int8(
          static_cast<const __half*>(d_ff2_), static_cast<std::int8_t*>(d_prefill_i8_),
          static_cast<float*>(d_prefill_i8_scales_), 1, inter, compute_stream_);
      resident_int8_mlp_w2(*lw_i8, hidden, inter);
    } else if (lw_i8 && lw_i8->w2) {
      if (lw_i8->mlp_int4) {
        kernels::launch_weight_only_int4_matvec(
            lw_i8->w2, lw_i8->s_w2, static_cast<const __half*>(d_ff2_),
            static_cast<__half*>(d_ff3_), hidden, inter, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec(
            lw_i8->w2, lw_i8->s_w2, static_cast<const __half*>(d_ff2_),
            static_cast<__half*>(d_ff3_), hidden, inter, compute_stream_);
      }
    } else if (lw->w2_packed.active()) {
      ++packed_matvec_calls_;
      kernels::launch_kquant_matvec(static_cast<const std::uint8_t*>(lw->w2_packed.data),
                                    static_cast<kernels::KQuantType>(lw->w2_packed.kind),
                                    static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_),
                                    lw->w2_packed.rows, lw->w2_packed.cols, compute_stream_);
    } else {
      // Residual add folded into the down-projection's epilogue.
      resident_projection_half_residual(lw->w2, d_ff2_, d_x_, hidden, inter, resident_wo_warps_,
                                        resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      fused_residual = true;
    }

    if (!fused_residual) {
      kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                  hidden, compute_stream_);
    }
  }

  kernels::launch_rmsnorm(static_cast<const __half*>(d_x_), static_cast<const __half*>(d_norm_out_),
                          static_cast<__half*>(d_x_norm_), 1, hidden, norm_eps, compute_stream_);
  // Always use custom kernel in graph capture: cuBLASLt may fall through to
  // cublasGemmEx which is not graph-capturable, causing INVALID_VALUE errors.
  project_lm_head_logits(static_cast<const __half*>(d_x_norm_), static_cast<float*>(d_logits_));
  kernels::launch_argmax_float(static_cast<const float*>(d_logits_), cfg.vocab_size, d_argmax_,
                               compute_stream_, d_argmax_part_val_, d_argmax_part_idx_,
                               argmax_parts_);
  kernels::launch_copy_int(d_argmax_, d_token_id_, compute_stream_);
  kernels::launch_increment_int(d_decode_position_, compute_stream_);

  CUDA_CHECK(cudaStreamEndCapture(compute_stream_, &greedy_decode_graph_));
  if (options_.verbose) {
    std::cout << "[engine] init_greedy_decode_graph: captured, instantiating\n";
  }
  CUDA_CHECK(
      cudaGraphInstantiate(&greedy_decode_graph_exec_, greedy_decode_graph_, nullptr, nullptr, 0));

  // How many kernels does one decoded token actually cost? At batch 1 on a small model,
  // per-kernel scheduling overhead, not bandwidth, is what separates us from the
  // roofline, so the node count is the number to optimise against.
  if (std::getenv("CPI_GRAPH_NODES")) {
    std::size_t n = 0;
    cudaGraphGetNodes(greedy_decode_graph_, nullptr, &n);
    std::cerr << "[graph] greedy decode graph: " << n << " nodes ("
              << (cfg.num_layers > 0 ? static_cast<double>(n) / cfg.num_layers : 0.0)
              << " per layer)\n";
  }
  greedy_decode_graph_ready_ = true;
  greedy_decode_graph_state_valid_ = false;
  if (options_.verbose) {
    std::cout << "[engine] greedy_decode_graph: enabled\n";
  }
}

// Same transformer body as init_greedy_decode_graph but without argmax/copy/increment
// at the end; outputs to d_logits_ so the sampling path can read them back to CPU.
void LlamaEngine::init_logits_decode_graph() {
  if (logits_decode_graph_ready_) {
    return;
  }
  if (!can_use_greedy_decode_graph()) {
    return;
  }

  const auto& cfg = weights_.config();
  const float norm_eps = cfg.norm_eps > 0.0f ? cfg.norm_eps : 1e-5f;
  const int hidden = cfg.hidden_size;
  const int inter = cfg.intermediate_size;
  const int head_dim = cfg.hidden_size / cfg.num_heads;
  const int kv_hidden = cfg.num_kv_heads * head_dim;
  const bool can_use_dp4a_decode = ((hidden & 3) == 0) && ((inter & 3) == 0);
  const auto apply_qprod_residual = [&](const uint32_t* row_bits, const half* residual_scales,
                                        const half* rotated_x, half* y, int out_features) {
    if (!tq_prod_enabled_ || tq_qjl_dim_ <= 0 || !row_bits || !residual_scales || !rotated_x ||
        !y || !d_tq_qjl_indices_ || !d_tq_qjl_signs_ || !d_tq_qjl_x_bits_) {
      return;
    }
    kernels::launch_tq_qjl_pack_sign_bits(rotated_x, d_tq_qjl_indices_, d_tq_qjl_signs_,
                                          d_tq_qjl_x_bits_, tq_qjl_dim_, compute_stream_);
    kernels::launch_tq_qjl_residual_add_f16(row_bits, residual_scales, d_tq_qjl_x_bits_, y,
                                            out_features, tq_qjl_dim_, compute_stream_);
  };

  // Warm cuBLASLt plans before capture. Skip fp16 QKV/WO/MLP warmup when INT8
  // projections are active: those fp16 weight pointers are freed after packing.
  if (!layer_cache_.empty()) {
    const auto& lw = layer_cache_.front();
    if (tq3_enabled_ && !layer_cache_tq3_.empty()) {
      // TQ3: warm packed GEMV kernels; skip any weight that wasn't loaded (null guard).
      const auto& tq = layer_cache_tq3_.front();
      if (tq.wqkv) {
        kernels::launch_tq3_gemv_f16(
            tq.wqkv, d_tq3_codebook_, tq.s_wqkv, static_cast<const __half*>(d_x_tq3_),
            static_cast<__half*>(d_q_), hidden + 2 * kv_hidden, hidden, compute_stream_);
      }
      if (tq.wo) {
        kernels::launch_tq3_gemv_f16(tq.wo, d_tq3_codebook_, tq.s_wo,
                                     static_cast<const __half*>(d_x_tq3_),
                                     static_cast<__half*>(d_ff3_), hidden, hidden, compute_stream_);
      }
      if (tq.w13) {
        kernels::launch_tq3_gemv_f16(
            tq.w13, d_tq3_codebook_, tq.s_w13, static_cast<const __half*>(d_x_tq3_),
            static_cast<__half*>(d_ff1_), 2 * inter, hidden, compute_stream_);
      }
      // Warm w2 (fp16 under TQ3) using resident_projection_half; always graph-capturable.
      if (lw.w2 != nullptr) {
        resident_projection_half(lw.w2, d_ff2_, d_ff3_, hidden, inter, resident_wo_warps_,
                                 resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      }
    } else if (!cached_int8_proj_enabled_) {
      if (resident_custom_qkv_) {
        if (!packed_qkv_matvec(lw, d_x_norm_, d_q_, compute_stream_)) {
          resident_projection_half(lw.wqkv, d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden,
                                   resident_qkv_warps_, resident_qkv_tile_pairs_,
                                   resident_qkv_rows_per_warp_);
        }
      } else if (!packed_qkv_matvec(lw, d_x_norm_, d_q_, compute_stream_)) {
        detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                                lt_workspace_bytes_, compute_stream_, lw.wqkv,
                                                d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden, 1,
                                                CUDA_R_16F);
      }
      if (lw.wo_packed.active()) {
        // Warm what capture will actually record. A packed matrix has no fp16
        // buffer, so warming the fp16 kernel would hand cuBLAS a null pointer.
        kernels::launch_kquant_matvec(static_cast<const std::uint8_t*>(lw.wo_packed.data),
                                      static_cast<kernels::KQuantType>(lw.wo_packed.kind),
                                      static_cast<const __half*>(d_att_),
                                      static_cast<__half*>(d_ff3_), lw.wo_packed.rows,
                                      lw.wo_packed.cols, compute_stream_);
      } else if (resident_custom_wo_) {
        resident_projection_half(lw.wo, d_att_, d_ff3_, hidden, hidden, resident_wo_warps_,
                                 resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      } else {
        detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                                lt_workspace_bytes_, compute_stream_, lw.wo, d_att_,
                                                d_ff3_, hidden, hidden, 1, CUDA_R_16F);
      }
    }
    if (!tq3_enabled_) {
      if (cached_int8_mlp_enabled_ && !layer_cache_i8_.empty() && layer_cache_i8_.front().w1) {
        if (can_use_dp4a_decode) {
          resident_int8_mlp_w13(layer_cache_i8_.front(), inter, hidden);
          resident_int8_mlp_w2(layer_cache_i8_.front(), hidden, inter);
        } else {
          if (layer_cache_i8_.front().mlp_int4) {
            kernels::launch_weight_only_int4_matvec(
                layer_cache_i8_.front().w1, layer_cache_i8_.front().s_w1,
                static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff1_), inter, hidden,
                compute_stream_);
            kernels::launch_weight_only_int4_matvec(
                layer_cache_i8_.front().w2, layer_cache_i8_.front().s_w2,
                static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_), hidden, inter,
                compute_stream_);
          } else {
            kernels::launch_weight_only_int8_matvec(
                layer_cache_i8_.front().w1, layer_cache_i8_.front().s_w1,
                static_cast<const __half*>(d_x_norm_), static_cast<__half*>(d_ff1_), inter, hidden,
                compute_stream_);
            kernels::launch_weight_only_int8_matvec(
                layer_cache_i8_.front().w2, layer_cache_i8_.front().s_w2,
                static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_), hidden, inter,
                compute_stream_);
          }
        }
      } else if (!cached_int8_proj_enabled_) {
        if (lw.w13_packed.active()) {
          kernels::launch_kquant_matvec(static_cast<const std::uint8_t*>(lw.w13_packed.data),
                                        static_cast<kernels::KQuantType>(lw.w13_packed.kind),
                                        static_cast<const __half*>(d_x_norm_),
                                        static_cast<__half*>(d_ff1_), lw.w13_packed.rows,
                                        lw.w13_packed.cols, compute_stream_);
        } else {
          detail::dispatch_linear_rowmajor_weight(
              cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_,
              compute_stream_, lw.w13, d_x_norm_, d_ff1_, 2 * inter, hidden, 1, CUDA_R_16F);
        }
        if (lw.w2_packed.active()) {
          kernels::launch_kquant_matvec(static_cast<const std::uint8_t*>(lw.w2_packed.data),
                                        static_cast<kernels::KQuantType>(lw.w2_packed.kind),
                                        static_cast<const __half*>(d_ff2_),
                                        static_cast<__half*>(d_ff3_), lw.w2_packed.rows,
                                        lw.w2_packed.cols, compute_stream_);
        } else {
          detail::dispatch_linear_rowmajor_weight(
              cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_,
              compute_stream_, lw.w2, d_ff2_, d_ff3_, hidden, inter, 1, CUDA_R_16F);
        }
      } else if (lw.w13) {
        // INT8 proj + FP16 MLP (e.g., TinyLlama): warm resident_projection_half since
        // cuBLASLt may not find a graph-capturable plan for all dimension combinations.
        resident_projection_half(lw.w13, d_x_norm_, d_ff1_, 2 * inter, hidden, resident_qkv_warps_,
                                 resident_qkv_tile_pairs_, resident_qkv_rows_per_warp_);
        resident_projection_half(lw.w2, d_ff2_, d_ff3_, hidden, inter, resident_wo_warps_,
                                 resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      }
    }
    project_lm_head_logits(static_cast<const __half*>(d_x_norm_), static_cast<float*>(d_logits_));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  }

  destroy_logits_decode_graph();
  CUDA_CHECK(cudaStreamBeginCapture(compute_stream_, cudaStreamCaptureModeGlobal));

  kernels::launch_embedding_lookup(static_cast<const __half*>(d_tok_embeddings_), d_token_id_,
                                   static_cast<__half*>(d_x_), 1, hidden, compute_stream_);

  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    // Set when a projection folded the residual add into its own epilogue, so the shared
    // add_inplace below is skipped. The int8 / tq3 branches do not fold it and still need
    // the add; moving the add into one branch would silently drop the residual for them.
    bool fused_residual = false;
    const auto* lw = &layer_cache_[static_cast<std::size_t>(layer)];
    const LayerDeviceInt8Weights* lw_i8 = (layer < static_cast<int>(layer_cache_i8_.size()) &&
                                           (layer_cache_i8_[static_cast<std::size_t>(layer)].w1 ||
                                            layer_cache_i8_[static_cast<std::size_t>(layer)].wqkv))
                                              ? &layer_cache_i8_[static_cast<std::size_t>(layer)]
                                              : nullptr;
    const LayerDeviceTq3Weights* tq =
        (tq3_enabled_ && layer < static_cast<int>(layer_cache_tq3_.size()))
            ? &layer_cache_tq3_[static_cast<std::size_t>(layer)]
            : nullptr;
    auto* k_layer =
        static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) *
                                               static_cast<std::size_t>(options_.max_context) *
                                               static_cast<std::size_t>(kv_hidden);
    auto* v_layer =
        static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) *
                                               static_cast<std::size_t>(options_.max_context) *
                                               static_cast<std::size_t>(kv_hidden);

    kernels::launch_rmsnorm(static_cast<const __half*>(d_x_),
                            static_cast<const __half*>(lw->norm_att),
                            static_cast<__half*>(d_x_norm_), 1, hidden, norm_eps, compute_stream_);

    if (tq && tq->wqkv) {
      CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_x_norm_,
                                 static_cast<std::size_t>(hidden) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, compute_stream_));
      kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_), d_tq3_signs_, hidden,
                                           tq3_block_size_, compute_stream_);
      kernels::launch_tq3_gemv_f16(tq->wqkv, d_tq3_codebook_, tq->s_wqkv,
                                   static_cast<const __half*>(d_x_tq3_), static_cast<__half*>(d_q_),
                                   hidden + 2 * kv_hidden, hidden, compute_stream_);
      apply_qprod_residual(tq->r_wqkv, tq->rs_wqkv, static_cast<const __half*>(d_x_tq3_),
                           static_cast<__half*>(d_q_), hidden + 2 * kv_hidden);
    } else if (cached_int8_proj_enabled_ && lw_i8 && lw_i8->wqkv) {
      kernels::launch_quantize_rowwise_fp16_to_int8(
          static_cast<const __half*>(d_x_norm_), static_cast<std::int8_t*>(d_prefill_i8_),
          static_cast<float*>(d_prefill_i8_scales_), 1, hidden, compute_stream_);
      if (lw_i8->proj_int4) {
        kernels::launch_weight_only_int4_matvec_dp4a(
            lw_i8->wqkv, lw_i8->s_wqkv, static_cast<const std::int8_t*>(d_prefill_i8_),
            static_cast<const float*>(d_prefill_i8_scales_), static_cast<__half*>(d_q_),
            hidden + 2 * kv_hidden, hidden, compute_stream_, resident_int8_qkv_warps_,
            resident_int8_qkv_tile_packed4_, resident_int8_qkv_warps_per_row_);
      } else {
        kernels::launch_weight_only_int8_matvec_dp4a(
            lw_i8->wqkv, lw_i8->s_wqkv, static_cast<const std::int8_t*>(d_prefill_i8_),
            static_cast<const float*>(d_prefill_i8_scales_), static_cast<__half*>(d_q_),
            hidden + 2 * kv_hidden, hidden, compute_stream_, resident_int8_qkv_warps_,
            resident_int8_qkv_tile_packed4_, resident_int8_qkv_warps_per_row_);
      }
    } else if (resident_custom_qkv_) {
      if (!packed_qkv_matvec(*lw, d_x_norm_, d_q_, compute_stream_)) {
        resident_projection_half(lw->wqkv, d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden,
                                 resident_qkv_warps_, resident_qkv_tile_pairs_);
      }
    } else if (!packed_qkv_matvec(*lw, d_x_norm_, d_q_, compute_stream_)) {
      detail::dispatch_linear_rowmajor_weight(
          cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
          lw->wqkv, d_x_norm_, d_q_, hidden + 2 * kv_hidden, hidden, 1, CUDA_R_16F);
    }

    if (lw->bqkv) {
      kernels::launch_add_inplace(static_cast<__half*>(d_q_), static_cast<const __half*>(lw->bqkv),
                                  hidden + 2 * kv_hidden, compute_stream_);
    }

    kernels::launch_rope_inplace_device_pos(
        static_cast<__half*>(d_q_), static_cast<__half*>(d_k_), cfg.num_heads, cfg.num_kv_heads,
        head_dim, d_decode_position_, d_rope_cos_, d_rope_sin_, compute_stream_);
    kernels::launch_store_kv_device_pos(
        static_cast<const __half*>(d_k_), static_cast<const __half*>(d_v_), k_layer, v_layer,
        d_decode_position_, kv_hidden, options_.max_context, compute_stream_);
    kernels::launch_attention_step_device_pos(
        static_cast<const __half*>(d_q_), k_layer, v_layer, static_cast<__half*>(d_att_),
        d_decode_position_, cfg.num_heads, cfg.num_kv_heads, head_dim, compute_stream_,
        d_attn_chunk_m_, d_attn_chunk_l_, d_attn_chunk_o_, attn_chunk_capacity_,
        !options_.disable_split_attention);

    if (tq && tq->wo) {
      CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_att_,
                                 static_cast<std::size_t>(hidden) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, compute_stream_));
      kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_), d_tq3_signs_, hidden,
                                           tq3_block_size_, compute_stream_);
      kernels::launch_tq3_gemv_f16(tq->wo, d_tq3_codebook_, tq->s_wo,
                                   static_cast<const __half*>(d_x_tq3_),
                                   static_cast<__half*>(d_ff3_), hidden, hidden, compute_stream_);
      apply_qprod_residual(tq->r_wo, tq->rs_wo, static_cast<const __half*>(d_x_tq3_),
                           static_cast<__half*>(d_ff3_), hidden);
    } else if (cached_int8_proj_enabled_ && lw_i8 && lw_i8->wo) {
      kernels::launch_quantize_rowwise_fp16_to_int8(
          static_cast<const __half*>(d_att_), static_cast<std::int8_t*>(d_prefill_i8_),
          static_cast<float*>(d_prefill_i8_scales_), 1, hidden, compute_stream_);
      if (lw_i8->proj_int4) {
        kernels::launch_weight_only_int4_matvec_dp4a(
            lw_i8->wo, lw_i8->s_wo, static_cast<const std::int8_t*>(d_prefill_i8_),
            static_cast<const float*>(d_prefill_i8_scales_), static_cast<__half*>(d_ff3_), hidden,
            hidden, compute_stream_, resident_int8_wo_warps_, resident_int8_wo_tile_packed4_,
            resident_int8_wo_warps_per_row_);
      } else {
        kernels::launch_weight_only_int8_matvec_dp4a(
            lw_i8->wo, lw_i8->s_wo, static_cast<const std::int8_t*>(d_prefill_i8_),
            static_cast<const float*>(d_prefill_i8_scales_), static_cast<__half*>(d_ff3_), hidden,
            hidden, compute_stream_, resident_int8_wo_warps_, resident_int8_wo_tile_packed4_,
            resident_int8_wo_warps_per_row_);
      }
    } else if (lw->wo_packed.active()) {
      // Packed k-quant: multiply the container's own blocks. Nothing is fused
      // into this one, so the shared add_inplace below still runs.
      ++packed_matvec_calls_;
      kernels::launch_kquant_matvec(static_cast<const std::uint8_t*>(lw->wo_packed.data),
                                    static_cast<kernels::KQuantType>(lw->wo_packed.kind),
                                    static_cast<const __half*>(d_att_), static_cast<__half*>(d_ff3_),
                                    lw->wo_packed.rows, lw->wo_packed.cols, compute_stream_);
    } else if (resident_custom_wo_) {
      // Residual add folded into this projection's epilogue, so the shared add_inplace
      // below is skipped. At batch 1 a kernel costs a fixed ~1.7 us whatever it does.
      resident_projection_half_residual(lw->wo, d_att_, d_x_, hidden, hidden, resident_wo_warps_,
                                        resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      fused_residual = true;
    } else {
      detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                              lt_workspace_bytes_, compute_stream_, lw->wo, d_att_,
                                              d_ff3_, hidden, hidden, 1, CUDA_R_16F);
    }

    if (!fused_residual) {
      kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                  hidden, compute_stream_);
    }
    fused_residual = false;

    kernels::launch_rmsnorm(static_cast<const __half*>(d_x_),
                            static_cast<const __half*>(lw->norm_ffn),
                            static_cast<__half*>(d_x_norm_), 1, hidden, norm_eps, compute_stream_);

    if (tq && tq->w13) {
      CUDA_CHECK(cudaMemcpyAsync(d_x_tq3_, d_x_norm_,
                                 static_cast<std::size_t>(hidden) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, compute_stream_));
      kernels::launch_hadamard_rotate_fp16(static_cast<__half*>(d_x_tq3_), d_tq3_signs_, hidden,
                                           tq3_block_size_, compute_stream_);
      kernels::launch_tq3_gemv_f16(
          tq->w13, d_tq3_codebook_, tq->s_w13, static_cast<const __half*>(d_x_tq3_),
          static_cast<__half*>(d_ff1_), 2 * inter, hidden, compute_stream_);
      apply_qprod_residual(tq->r_w13, tq->rs_w13, static_cast<const __half*>(d_x_tq3_),
                           static_cast<__half*>(d_ff1_), 2 * inter);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && can_use_dp4a_decode) {
      kernels::launch_quantize_rowwise_fp16_to_int8(
          static_cast<const __half*>(d_x_norm_), static_cast<std::int8_t*>(d_prefill_i8_),
          static_cast<float*>(d_prefill_i8_scales_), 1, hidden, compute_stream_);
      resident_int8_mlp_w13(*lw_i8, inter, hidden);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w3) {
      if (lw_i8->mlp_int4) {
        kernels::launch_weight_only_int4_matvec(
            lw_i8->w1, lw_i8->s_w1, static_cast<const __half*>(d_x_norm_),
            static_cast<__half*>(d_ff1_), inter, hidden, compute_stream_);
        kernels::launch_weight_only_int4_matvec(
            lw_i8->w3, lw_i8->s_w3, static_cast<const __half*>(d_x_norm_),
            static_cast<__half*>(d_ff2_), inter, hidden, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec(
            lw_i8->w1, lw_i8->s_w1, static_cast<const __half*>(d_x_norm_),
            static_cast<__half*>(d_ff1_), inter, hidden, compute_stream_);
        kernels::launch_weight_only_int8_matvec(
            lw_i8->w3, lw_i8->s_w3, static_cast<const __half*>(d_x_norm_),
            static_cast<__half*>(d_ff2_), inter, hidden, compute_stream_);
      }
    } else if (lw->w13_packed.active()) {
      ++packed_matvec_calls_;
      kernels::launch_kquant_matvec(static_cast<const std::uint8_t*>(lw->w13_packed.data),
                                    static_cast<kernels::KQuantType>(lw->w13_packed.kind),
                                    static_cast<const __half*>(d_x_norm_),
                                    static_cast<__half*>(d_ff1_), lw->w13_packed.rows,
                                    lw->w13_packed.cols, compute_stream_);
    } else {
      // FP16 MLP fallback in graph capture: use resident_projection_half (always graph-capturable).
      resident_projection_half(lw->w13, d_x_norm_, d_ff1_, 2 * inter, hidden, resident_qkv_warps_,
                               resident_qkv_tile_pairs_, resident_qkv_rows_per_warp_);
    }

    detail::launch_gated_glu(weights_.config().mlp_gelu, static_cast<const __half*>(d_ff1_),
                             static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff2_),
                             inter, compute_stream_);

    if (tq) {
      resident_projection_half(lw->w2, d_ff2_, d_ff3_, hidden, inter, resident_wo_warps_,
                               resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && can_use_dp4a_decode) {
      kernels::launch_quantize_rowwise_fp16_to_int8(
          static_cast<const __half*>(d_ff2_), static_cast<std::int8_t*>(d_prefill_i8_),
          static_cast<float*>(d_prefill_i8_scales_), 1, inter, compute_stream_);
      resident_int8_mlp_w2(*lw_i8, hidden, inter);
    } else if (lw_i8 && lw_i8->w2) {
      if (lw_i8->mlp_int4) {
        kernels::launch_weight_only_int4_matvec(
            lw_i8->w2, lw_i8->s_w2, static_cast<const __half*>(d_ff2_),
            static_cast<__half*>(d_ff3_), hidden, inter, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec(
            lw_i8->w2, lw_i8->s_w2, static_cast<const __half*>(d_ff2_),
            static_cast<__half*>(d_ff3_), hidden, inter, compute_stream_);
      }
    } else if (lw->w2_packed.active()) {
      ++packed_matvec_calls_;
      kernels::launch_kquant_matvec(static_cast<const std::uint8_t*>(lw->w2_packed.data),
                                    static_cast<kernels::KQuantType>(lw->w2_packed.kind),
                                    static_cast<const __half*>(d_ff2_), static_cast<__half*>(d_ff3_),
                                    lw->w2_packed.rows, lw->w2_packed.cols, compute_stream_);
    } else {
      // Residual add folded into the down-projection's epilogue.
      resident_projection_half_residual(lw->w2, d_ff2_, d_x_, hidden, inter, resident_wo_warps_,
                                        resident_wo_tile_pairs_, resident_wo_rows_per_warp_);
      fused_residual = true;
    }

    if (!fused_residual) {
      kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                  hidden, compute_stream_);
    }
  }

  kernels::launch_rmsnorm(static_cast<const __half*>(d_x_), static_cast<const __half*>(d_norm_out_),
                          static_cast<__half*>(d_x_norm_), 1, hidden, norm_eps, compute_stream_);
  // Always use custom kernel in graph capture; same reason as greedy graph.
  project_lm_head_logits(static_cast<const __half*>(d_x_norm_), static_cast<float*>(d_logits_));
  // Note: no argmax/copy_int/increment_int; caller reads d_logits_ for sampling.

  CUDA_CHECK(cudaStreamEndCapture(compute_stream_, &logits_decode_graph_));
  CUDA_CHECK(
      cudaGraphInstantiate(&logits_decode_graph_exec_, logits_decode_graph_, nullptr, nullptr, 0));
  logits_decode_graph_ready_ = true;
  if (options_.verbose) {
    std::cout << "[engine] logits_decode_graph: enabled\n";
  }
}

bool LlamaEngine::run_logits_decode_graph(int token, int position) {
  if (!logits_decode_graph_ready_) {
    init_logits_decode_graph();
  }
  if (!logits_decode_graph_ready_) {
    return false;
  }
  // Always upload token+position since the graph doesn't auto-increment position.
  CUDA_CHECK(
      cudaMemcpyAsync(d_token_id_, &token, sizeof(int), cudaMemcpyHostToDevice, compute_stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_decode_position_, &position, sizeof(int), cudaMemcpyHostToDevice,
                             compute_stream_));
  CUDA_CHECK(cudaGraphLaunch(logits_decode_graph_exec_, compute_stream_));
  return true;
}

void LlamaEngine::decode_next_token_logits_graph(int token, int position,
                                                 std::vector<float>& h_logits) {
  if (!run_logits_decode_graph(token, position)) {
    forward_token_logits(token, position, &h_logits, nullptr);
    return;
  }
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  const int vocab = weights_.config().vocab_size;
  h_logits.resize(static_cast<std::size_t>(vocab));
  CUDA_CHECK(cudaMemcpy(h_logits.data(), d_logits_, static_cast<std::size_t>(vocab) * sizeof(float),
                        cudaMemcpyDeviceToHost));
}

namespace {
// Device top-k limits. kCandCapacity leaves room for ties at the k-th logit, which can push
// the candidate set past k; overflowing it falls back to the host path rather than silently
// sampling from a truncated set.
constexpr int kMaxDeviceTopK = 1024;
constexpr int kCandCapacity = 4096;
}  // namespace

void LlamaEngine::ensure_device_topk_buffers() {
  if (device_topk_ready_) {
    return;
  }
  const int vocab = weights_.config().vocab_size;
  const int parts = kernels::topk_partition_count(vocab);
  const std::size_t part_n = static_cast<std::size_t>(parts) * kMaxDeviceTopK;
  CUDA_CHECK(cudaMalloc(&d_topk_part_val_, part_n * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_topk_part_idx_, part_n * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_topk_val_, static_cast<std::size_t>(kMaxDeviceTopK) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_topk_idx_, static_cast<std::size_t>(kMaxDeviceTopK) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cand_idx_, static_cast<std::size_t>(kCandCapacity) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_cand_val_, static_cast<std::size_t>(kCandCapacity) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_cand_count_, sizeof(int)));
  device_topk_ready_ = true;
}

// Opt-out for A/B and bisection: CPI_HOST_SAMPLING=1 forces the old full-vocab
// host path. Used by tools/gemv_output_gate.py to prove the two agree token-for-token.
static bool host_sampling_forced() {
  static const bool v = [] {
    const char* e = std::getenv("CPI_HOST_SAMPLING");
    return e && *e == '1';
  }();
  return v;
}

bool LlamaEngine::decode_next_token_device_topk(int token, int position, float temperature,
                                                const std::vector<int>& history, int* out_token) {
  (void)history;  // eligibility already excludes the history-dependent penalties
  const int vocab = weights_.config().vocab_size;
  const int k = options_.top_k;
  if (host_sampling_forced() || !can_use_greedy_decode_graph() || temperature <= 0.0f || k <= 0 ||
      k > kMaxDeviceTopK || k >= vocab) {
    return false;
  }
  ensure_device_topk_buffers();

  if (!run_logits_decode_graph(token, position)) {
    return false;
  }
  CUDA_CHECK(cudaMemsetAsync(d_cand_count_, 0, sizeof(int), compute_stream_));
  kernels::launch_topk_float(static_cast<const float*>(d_logits_), vocab, k, d_topk_part_val_,
                             d_topk_part_idx_, d_topk_val_, d_topk_idx_, compute_stream_);
  // Threshold = the k-th largest logit, so {i : logit_i >= kth} is exactly the candidate set
  // the host sampler would have built; ties at the k-th value included.
  kernels::launch_gather_ge_threshold(static_cast<const float*>(d_logits_), vocab,
                                      d_topk_val_ + (k - 1), d_cand_idx_, d_cand_val_,
                                      d_cand_count_, kCandCapacity, compute_stream_);
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));

  int count = 0;
  CUDA_CHECK(cudaMemcpy(&count, d_cand_count_, sizeof(int), cudaMemcpyDeviceToHost));
  if (count <= 0 || count > kCandCapacity) {
    return false;  // pathological tie count; let the host path handle it
  }
  std::vector<int> idx(static_cast<std::size_t>(count));
  std::vector<float> val(static_cast<std::size_t>(count));
  CUDA_CHECK(cudaMemcpy(idx.data(), d_cand_idx_, static_cast<std::size_t>(count) * sizeof(int),
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(val.data(), d_cand_val_, static_cast<std::size_t>(count) * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::vector<detail::SampleCandidate> cand(static_cast<std::size_t>(count));
  for (int i = 0; i < count; ++i) {
    cand[static_cast<std::size_t>(i)] = {idx[static_cast<std::size_t>(i)],
                                         val[static_cast<std::size_t>(i)]};
  }
  // The gather appends atomically, so its order is arbitrary; the host builds candidates in
  // index order. Sort to match, so the shared sampler sees an identical vector and a seeded
  // run produces the same token on both paths.
  std::sort(cand.begin(), cand.end(),
            [](const detail::SampleCandidate& a, const detail::SampleCandidate& b) {
              return a.id < b.id;
            });
  *out_token = detail::dispatch_sample_from_candidates(cand, temperature, options_.top_p);
  return true;
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
    CUDA_CHECK(
        cudaMemcpyAsync(d_token_id_, &token, sizeof(int), cudaMemcpyHostToDevice, compute_stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_decode_position_, &position, sizeof(int), cudaMemcpyHostToDevice,
                               compute_stream_));
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
