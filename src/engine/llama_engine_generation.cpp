#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include "common.hpp"
#include "engine/llama_engine.hpp"
#include "grammar/grammar_sampler.hpp"
#include "llama_engine_internal.hpp"
#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"

namespace engine {

std::vector<int> LlamaEngine::generate(const std::vector<int>& prompt_tokens, int max_new_tokens,
                                       float temperature) {
  return generate_stream(prompt_tokens, max_new_tokens, temperature, {});
}

void LlamaEngine::prefill_prompt_sequential(const std::vector<int>& prompt_tokens, int start_pos) {
  if (prompt_tokens.size() <= 1) {
    return;
  }
  const bool needs_streaming_barrier = cached_layer_count_ < weights_.config().num_layers;
  for (int i = std::max(0, start_pos); i < static_cast<int>(prompt_tokens.size()) - 1; ++i) {
    enforce_host_resource_limits("prefill.sequential");
    forward_token(prompt_tokens[static_cast<std::size_t>(i)], i, false, nullptr, nullptr);
    if (needs_streaming_barrier) {
      CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    }
  }
}

void LlamaEngine::prefill_prompt(const std::vector<int>& prompt_tokens, int start_pos) {
  if (prompt_tokens.size() <= 1) {
    return;
  }
  const auto& cfg = weights_.config();
  const bool all_layers_cached = cached_layer_count_ == cfg.num_layers;
  // Batched prefill runs the attention projections through the batched low-bit
  // kernels in --weight-quant int8/int4 mode (the resident cache frees the
  // fp16 wqkv/wo copies after quantisation) and through fp16 cuBLAS GEMMs
  // otherwise. Streamed layers may not have their projections resident in
  // either form, so quant + streaming falls back to sequential prefill.
  if (options_.paged_kv_cache || prefill_chunk_size_ <= 1 ||
      (cached_int8_proj_enabled_ && !all_layers_cached) || kv_int4_enabled_ || tq3_enabled_ ||
      cfg.is_moe() || cfg.uses_non_full_attention()) {
    prefill_prompt_sequential(prompt_tokens, start_pos);
    return;
  }
  const int prompt_count = static_cast<int>(prompt_tokens.size()) - 1;

  /*
   * Process prompt tokens in small fp16 chunks so we can reuse larger GEMMs
   * and batched attention without changing the exact decode path.
   */
  for (int chunk_start = std::max(0, start_pos); chunk_start < prompt_count;
       chunk_start += prefill_chunk_size_) {
    enforce_host_resource_limits("prefill.chunk_begin");
    const int rows = std::min(prefill_chunk_size_, prompt_count - chunk_start);
    CUDA_CHECK(cudaMemcpyAsync(d_token_id_, prompt_tokens.data() + chunk_start,
                               static_cast<std::size_t>(rows) * sizeof(int), cudaMemcpyHostToDevice,
                               compute_stream_));
    run_batched_chunk(rows, chunk_start);
  }
  if (!all_layers_cached) {
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  }
  enforce_host_resource_limits("prefill.done");
}

// Runs `rows` tokens (already uploaded to d_token_id_) through the full
// transformer in a single batched pass at absolute positions
// [base_pos, base_pos+rows), writing their K/V into the cache. Shared by
// prefill_prompt (per chunk, base_pos=chunk_start) and verify_tokens
// (speculative decoding, base_pos=accepted frontier).
void LlamaEngine::run_batched_chunk(int rows, int base_pos) {
  const auto& cfg = weights_.config();
  const bool all_layers_cached = cached_layer_count_ == cfg.num_layers;
  const int hidden = cfg.hidden_size;
  const int inter = cfg.intermediate_size;
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : hidden;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (hidden / cfg.num_heads);
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const bool can_use_dp4a_prefill = ((hidden & 3) == 0) && ((inter & 3) == 0);
  // d_prefill_i8_ holds rows x max(hidden, inter) int8 activations; the wo
  // input is rows x q_hidden, so q_hidden must fit within that buffer.
  const bool can_use_dp4a_proj =
      ((hidden & 3) == 0) && ((q_hidden & 3) == 0) && (q_hidden <= std::max(hidden, inter));
  const std::size_t q_row_bytes = static_cast<std::size_t>(q_hidden) * sizeof(__half);
  const std::size_t kv_row_bytes = static_cast<std::size_t>(kv_hidden) * sizeof(__half);
  const std::size_t ff_row_bytes = static_cast<std::size_t>(inter) * sizeof(__half);
  const std::size_t qkv_stride_bytes =
      static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * sizeof(__half);
  const std::size_t ff13_stride_bytes = static_cast<std::size_t>(2 * inter) * sizeof(__half);
  const std::size_t layer_stride =
      static_cast<std::size_t>(kv_capacity_tokens_) * static_cast<std::size_t>(kv_hidden);
  auto* qkv_base = static_cast<const __half*>(d_qkv_);
  auto* ff13_base = static_cast<const __half*>(d_ff13_);

  kernels::launch_embedding_lookup(static_cast<const __half*>(d_tok_embeddings_), d_token_id_,
                                   static_cast<__half*>(d_x_), rows, hidden, compute_stream_);
  if (cfg.scale_embeddings)  // Gemma: scale token embeddings by sqrt(hidden)
    kernels::launch_scale_copy(static_cast<__half*>(d_x_), static_cast<const __half*>(d_x_),
                               rows * hidden, std::sqrt(static_cast<float>(hidden)), compute_stream_);

  const auto run_layer = [&](int layer, const LayerDeviceWeights* lw,
                             const LayerDeviceInt8Weights* lw_i8) {
    // In --weight-quant mode the resident cache only keeps low-bit
    // projection weights (fp16 wqkv/wo are freed after quantisation), so the
    // projections go through the batched low-bit kernels, mirroring the MLP.
    const bool lowbit_proj = cached_int8_proj_enabled_ && lw_i8 && lw_i8->wqkv && lw_i8->wo;

    launch_norm(d_x_, lw->norm_att, lw->norm_att_bias, d_x_norm_, rows, hidden);

    if (lowbit_proj && can_use_dp4a_proj) {
      kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_x_norm_),
                                                    d_prefill_i8_, d_prefill_i8_scales_, rows,
                                                    hidden, compute_stream_);
      if (lw_i8->proj_int4) {
        kernels::launch_weight_only_int4_matvec_batched_dp4a(
            lw_i8->wqkv, lw_i8->s_wqkv, d_prefill_i8_, d_prefill_i8_scales_,
            static_cast<__half*>(d_qkv_), rows, q_hidden + 2 * kv_hidden, hidden, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec_batched_dp4a(
            lw_i8->wqkv, lw_i8->s_wqkv, d_prefill_i8_, d_prefill_i8_scales_,
            static_cast<__half*>(d_qkv_), rows, q_hidden + 2 * kv_hidden, hidden, compute_stream_);
      }
    } else if (lowbit_proj) {
      if (lw_i8->proj_int4) {
        kernels::launch_weight_only_int4_matvec_batched(
            lw_i8->wqkv, lw_i8->s_wqkv, static_cast<const __half*>(d_x_norm_),
            static_cast<__half*>(d_qkv_), rows, q_hidden + 2 * kv_hidden, hidden, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec_batched(
            lw_i8->wqkv, lw_i8->s_wqkv, static_cast<const __half*>(d_x_norm_),
            static_cast<__half*>(d_qkv_), rows, q_hidden + 2 * kv_hidden, hidden, compute_stream_);
      }
    } else {
      detail::dispatch_linear_rowmajor_weight(
          cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
          lw->wqkv, d_x_norm_, d_qkv_, q_hidden + 2 * kv_hidden, hidden, rows, CUDA_R_16F);
    }
    maybe_add_half_bias(d_ff3_, lw->bo, rows, hidden);

    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_q_, q_row_bytes, qkv_base, qkv_stride_bytes, q_row_bytes,
                                 rows, cudaMemcpyDeviceToDevice, compute_stream_));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_k_, kv_row_bytes, qkv_base + q_hidden, qkv_stride_bytes,
                                 kv_row_bytes, rows, cudaMemcpyDeviceToDevice, compute_stream_));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_v_, kv_row_bytes, qkv_base + q_hidden + kv_hidden,
                                 qkv_stride_bytes, kv_row_bytes, rows, cudaMemcpyDeviceToDevice,
                                 compute_stream_));

    if (lw->bqkv) {
      const auto* bqkv_half = static_cast<const __half*>(lw->bqkv);
      kernels::launch_add_bias_broadcast(static_cast<__half*>(d_prefill_q_), bqkv_half, rows,
                                         q_hidden, compute_stream_);
      kernels::launch_add_bias_broadcast(static_cast<__half*>(d_prefill_k_), bqkv_half + q_hidden,
                                         rows, kv_hidden, compute_stream_);
      kernels::launch_add_bias_broadcast(static_cast<__half*>(d_prefill_v_),
                                         bqkv_half + q_hidden + kv_hidden, rows, kv_hidden,
                                         compute_stream_);
    }

    if (lw->q_norm && lw->k_norm) {
      // Qwen3: per-head RMSNorm on Q and K (each head's head_dim slice), pre-RoPE.
      kernels::launch_rmsnorm(static_cast<const __half*>(d_prefill_q_),
                                     static_cast<const __half*>(lw->q_norm),
                                     static_cast<__half*>(d_prefill_q_), rows * cfg.num_heads,
                                     head_dim, cfg.norm_eps, compute_stream_);
      kernels::launch_rmsnorm(static_cast<const __half*>(d_prefill_k_),
                                     static_cast<const __half*>(lw->k_norm),
                                     static_cast<__half*>(d_prefill_k_), rows * cfg.num_kv_heads,
                                     head_dim, cfg.norm_eps, compute_stream_);
    }

    kernels::launch_rope_inplace_batched(
        static_cast<__half*>(d_prefill_q_), static_cast<__half*>(d_prefill_k_), rows, cfg.num_heads,
        cfg.num_kv_heads, head_dim, base_pos, d_rope_cos_, d_rope_sin_, compute_stream_);

    auto* k_layer =
        static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride;
    auto* v_layer =
        static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride;
    if (options_.paged_blocks && d_block_table_ != nullptr) {
      // P3 phase 2d: scatter this chunk's KV into the block pool via the block
      // table, and run paged prefill attention (gather via the same table).
      const int bs = options_.paged_block_size > 0 ? options_.paged_block_size : 32;
      kernels::launch_store_kv_paged(k_layer, v_layer, static_cast<const __half*>(d_prefill_k_),
                                     static_cast<const __half*>(d_prefill_v_), d_block_table_,
                                     base_pos, rows, kv_hidden, bs, compute_stream_);
      kernels::launch_attention_prefill_paged(static_cast<const __half*>(d_prefill_q_), k_layer,
                                              v_layer, d_block_table_, static_cast<__half*>(d_att_),
                                              rows, base_pos, cfg.num_heads, cfg.num_kv_heads,
                                              head_dim, bs, compute_stream_);
    } else {
      CUDA_CHECK(cudaMemcpy2DAsync(k_layer + static_cast<std::size_t>(base_pos) * kv_hidden,
                                   kv_row_bytes, d_prefill_k_, kv_row_bytes, kv_row_bytes, rows,
                                   cudaMemcpyDeviceToDevice, compute_stream_));
      CUDA_CHECK(cudaMemcpy2DAsync(v_layer + static_cast<std::size_t>(base_pos) * kv_hidden,
                                   kv_row_bytes, d_prefill_v_, kv_row_bytes, kv_row_bytes, rows,
                                   cudaMemcpyDeviceToDevice, compute_stream_));

      // Tensor cores first: Q.K^T and P.V as cuBLAS batched GEMMs. Falls back to the kernel if
      // the geometry is unsupported or the score matrix would be too big. This is plain causal
      // text prefill -- no per-token limits, no sliding window -- which is exactly the case the
      // GEMM path covers; the vision/bidirectional paths never reach here.
      if (!prefill_attention_tensorcore(d_prefill_q_, k_layer, v_layer, d_att_, rows, base_pos,
                                        cfg.num_heads, cfg.num_kv_heads, head_dim)) {
        kernels::launch_attention_prefill(static_cast<const __half*>(d_prefill_q_), k_layer,
                                          v_layer, static_cast<__half*>(d_att_), rows, base_pos,
                                          cfg.num_heads, cfg.num_kv_heads, head_dim,
                                          compute_stream_);
      }
    }

    if (lowbit_proj && can_use_dp4a_proj) {
      kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_att_),
                                                    d_prefill_i8_, d_prefill_i8_scales_, rows,
                                                    q_hidden, compute_stream_);
      if (lw_i8->proj_int4) {
        kernels::launch_weight_only_int4_matvec_batched_dp4a(
            lw_i8->wo, lw_i8->s_wo, d_prefill_i8_, d_prefill_i8_scales_,
            static_cast<__half*>(d_ff3_), rows, hidden, q_hidden, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec_batched_dp4a(
            lw_i8->wo, lw_i8->s_wo, d_prefill_i8_, d_prefill_i8_scales_,
            static_cast<__half*>(d_ff3_), rows, hidden, q_hidden, compute_stream_);
      }
    } else if (lowbit_proj) {
      if (lw_i8->proj_int4) {
        kernels::launch_weight_only_int4_matvec_batched(
            lw_i8->wo, lw_i8->s_wo, static_cast<const __half*>(d_att_),
            static_cast<__half*>(d_ff3_), rows, hidden, q_hidden, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec_batched(
            lw_i8->wo, lw_i8->s_wo, static_cast<const __half*>(d_att_),
            static_cast<__half*>(d_ff3_), rows, hidden, q_hidden, compute_stream_);
      }
    } else {
      detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                              lt_workspace_bytes_, compute_stream_, lw->wo, d_att_,
                                              d_ff3_, hidden, q_hidden, rows, CUDA_R_16F);
    }

    kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                rows * hidden, compute_stream_);

    launch_norm(d_x_, lw->norm_ffn, lw->norm_ffn_bias, d_x_norm_, rows, hidden);

    if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && can_use_dp4a_prefill) {
      kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_x_norm_),
                                                    d_prefill_i8_, d_prefill_i8_scales_, rows,
                                                    hidden, compute_stream_);
      if (lw_i8->mlp_int4) {
        kernels::launch_weight_only_int4_matvec_batched_dp4a(
            lw_i8->w1, lw_i8->s_w1, d_prefill_i8_, d_prefill_i8_scales_,
            static_cast<__half*>(d_prefill_ff1_), rows, inter, hidden, compute_stream_);
        kernels::launch_weight_only_int4_matvec_batched_dp4a(
            lw_i8->w3, lw_i8->s_w3, d_prefill_i8_, d_prefill_i8_scales_,
            static_cast<__half*>(d_prefill_ff2_), rows, inter, hidden, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec_batched_dp4a(
            lw_i8->w1, lw_i8->s_w1, d_prefill_i8_, d_prefill_i8_scales_,
            static_cast<__half*>(d_prefill_ff1_), rows, inter, hidden, compute_stream_);
        kernels::launch_weight_only_int8_matvec_batched_dp4a(
            lw_i8->w3, lw_i8->s_w3, d_prefill_i8_, d_prefill_i8_scales_,
            static_cast<__half*>(d_prefill_ff2_), rows, inter, hidden, compute_stream_);
      }
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3) {
      if (lw_i8->mlp_int4) {
        kernels::launch_weight_only_int4_matvec_batched(
            lw_i8->w1, lw_i8->s_w1, static_cast<const __half*>(d_x_norm_),
            static_cast<__half*>(d_prefill_ff1_), rows, inter, hidden, compute_stream_);
        kernels::launch_weight_only_int4_matvec_batched(
            lw_i8->w3, lw_i8->s_w3, static_cast<const __half*>(d_x_norm_),
            static_cast<__half*>(d_prefill_ff2_), rows, inter, hidden, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec_batched(
            lw_i8->w1, lw_i8->s_w1, static_cast<const __half*>(d_x_norm_),
            static_cast<__half*>(d_prefill_ff1_), rows, inter, hidden, compute_stream_);
        kernels::launch_weight_only_int8_matvec_batched(
            lw_i8->w3, lw_i8->s_w3, static_cast<const __half*>(d_x_norm_),
            static_cast<__half*>(d_prefill_ff2_), rows, inter, hidden, compute_stream_);
      }
    } else {
      detail::dispatch_linear_rowmajor_weight(
          cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
          lw->w13, d_x_norm_, d_ff13_, 2 * inter, hidden, rows, CUDA_R_16F);

      CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_ff1_, ff_row_bytes, ff13_base, ff13_stride_bytes,
                                   ff_row_bytes, rows, cudaMemcpyDeviceToDevice, compute_stream_));
      CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_ff2_, ff_row_bytes, ff13_base + inter,
                                   ff13_stride_bytes, ff_row_bytes, rows, cudaMemcpyDeviceToDevice,
                                   compute_stream_));
    }

    detail::launch_gated_glu(weights_.config().mlp_gelu, static_cast<const __half*>(d_prefill_ff1_),
                             static_cast<const __half*>(d_prefill_ff2_),
                             static_cast<__half*>(d_prefill_ff2_), rows * inter, compute_stream_);

    if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && can_use_dp4a_prefill) {
      kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_prefill_ff2_),
                                                    d_prefill_i8_, d_prefill_i8_scales_, rows,
                                                    inter, compute_stream_);
      if (lw_i8->mlp_int4) {
        kernels::launch_weight_only_int4_matvec_batched_dp4a(
            lw_i8->w2, lw_i8->s_w2, d_prefill_i8_, d_prefill_i8_scales_,
            static_cast<__half*>(d_ff3_), rows, hidden, inter, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec_batched_dp4a(
            lw_i8->w2, lw_i8->s_w2, d_prefill_i8_, d_prefill_i8_scales_,
            static_cast<__half*>(d_ff3_), rows, hidden, inter, compute_stream_);
      }
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3) {
      if (lw_i8->mlp_int4) {
        kernels::launch_weight_only_int4_matvec_batched(
            lw_i8->w2, lw_i8->s_w2, static_cast<const __half*>(d_prefill_ff2_),
            static_cast<__half*>(d_ff3_), rows, hidden, inter, compute_stream_);
      } else {
        kernels::launch_weight_only_int8_matvec_batched(
            lw_i8->w2, lw_i8->s_w2, static_cast<const __half*>(d_prefill_ff2_),
            static_cast<__half*>(d_ff3_), rows, hidden, inter, compute_stream_);
      }
    } else {
      detail::dispatch_linear_rowmajor_weight(
          cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
          lw->w2, d_prefill_ff2_, d_ff3_, hidden, inter, rows, CUDA_R_16F);
    }

    kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                rows * hidden, compute_stream_);
  };

  if (all_layers_cached) {
    for (int layer = 0; layer < cfg.num_layers; ++layer) {
      const auto* lw = &layer_cache_[static_cast<std::size_t>(layer)];
      const LayerDeviceInt8Weights* lw_i8 = nullptr;
      if (options_.int8_streaming && layer < static_cast<int>(layer_cache_i8_.size()) &&
          layer_cache_i8_[static_cast<std::size_t>(layer)].w1) {
        lw_i8 = &layer_cache_i8_[static_cast<std::size_t>(layer)];
      }
      run_layer(layer, lw, lw_i8);
      enforce_host_resource_limits("prefill.chunk_layer");
    }
  } else {
    int uncached_index = 0;
    CUDA_CHECK(cudaStreamWaitEvent(transfer_stream_, streaming_consumed_[0], 0));
    copy_layer_weights_to_device(
        cached_layer_count_, &streaming_layer_weights_[0],
        options_.int8_streaming ? &streaming_layer_weights_i8_[0] : nullptr, transfer_stream_);
    CUDA_CHECK(cudaEventRecord(streaming_ready_[0], transfer_stream_));

    for (int layer = 0; layer < cfg.num_layers; ++layer) {
      const LayerDeviceWeights* lw = nullptr;
      const LayerDeviceInt8Weights* lw_i8 = nullptr;
      if (layer < cached_layer_count_) {
        lw = &layer_cache_[static_cast<std::size_t>(layer)];
        if (options_.int8_streaming && layer < static_cast<int>(layer_cache_i8_.size()) &&
            layer_cache_i8_[static_cast<std::size_t>(layer)].w1) {
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
          copy_layer_weights_to_device(
              next_layer, &streaming_layer_weights_[next_slot],
              options_.int8_streaming ? &streaming_layer_weights_i8_[next_slot] : nullptr,
              transfer_stream_);
          CUDA_CHECK(cudaEventRecord(streaming_ready_[next_slot], transfer_stream_));
        }
        ++uncached_index;
      }

      run_layer(layer, lw, lw_i8);
      enforce_host_resource_limits("prefill.chunk_layer");

      if (layer >= cached_layer_count_) {
        const int consumed_slot = (uncached_index - 1) % 2;
        CUDA_CHECK(cudaEventRecord(streaming_consumed_[consumed_slot], compute_stream_));
      }
    }
  }
}

std::vector<int> LlamaEngine::generate_stream(const std::vector<int>& prompt_tokens,
                                              int max_new_tokens, float temperature,
                                              const std::function<bool(int)>& on_token,
                                              const GenerationConstraints* constraints) {
  if (prompt_tokens.empty()) {
    LLAMA_ENGINE_THROW("prompt token list is empty");
  }
  // Bounds guard: the KV cache and position tables are allocated for exactly
  // options_.max_context positions (see llama_engine_lifecycle.cpp). A prompt
  // that meets or exceeds the window would prefill KV out of bounds, corrupting
  // memory — observed as garbage output on some builds and a 0xC0000409
  // stack-buffer-overrun on others. Reject with a clear error before touching
  // any buffer, leaving at least one slot for a generated token.
  if (static_cast<int>(prompt_tokens.size()) >= options_.max_context) {
    LLAMA_ENGINE_THROW("prompt length " + std::to_string(prompt_tokens.size()) +
                       " exceeds the model context window " + std::to_string(options_.max_context) +
                       " (reduce the prompt or raise --max-context / LLAMA_MAX_CONTEXT)");
  }

  // Bind the per-request grammar for the duration of this call; cleared on exit.
  active_grammar_ = constraints ? constraints->grammar : nullptr;
  const int min_new_tokens = constraints ? std::max(0, constraints->min_new_tokens) : 0;
  if (min_new_tokens > 0) {
    std::cerr << "[engine] min_new_tokens=" << min_new_tokens << " (EOS suppressed until then)\n";
  }
  suppress_eos_ = false;
  struct DecodeScope {
    grammar::GrammarSampler** grammar_slot;
    bool* suppress_slot;
    ~DecodeScope() {
      *grammar_slot = nullptr;
      *suppress_slot = false;
    }
  } decode_scope{&active_grammar_, &suppress_eos_};

  // Reproducible sampling when a seed is supplied (affects temperature>0 only).
  if (constraints != nullptr && constraints->seed >= 0) {
    detail::dispatch_seed_sampler_rng(static_cast<unsigned>(constraints->seed));
  }

  const auto& cfg = weights_.config();

  // Prefix KV reuse: if the head of this prompt matches the tokens whose KV is
  // already resident, skip re-prefilling that shared prefix. This wins big for
  // workloads that resend a large fixed system prompt every request, varying
  // only the trailing user message. KV for an identical prefix at identical positions is bit-exact
  // (causal attention + position-based RoPE), so the output is unchanged. Only
  // the simple contiguous fp16 KV layout is eligible; paged / int4-KV / TQ3 /
  // MoE / sliding-window configs take a full reset + prefill.
  const bool prefix_cacheable =
      !options_.paged_kv_cache && !options_.paged_blocks &&  // paged: each request gets its own
      !kv_int4_enabled_ && !tq3_enabled_ &&             // block table, so prior KV isn't at the
      !cfg.is_moe() && !cfg.uses_non_full_attention();  // same physical blocks (paged shared-prefix
                                                        // is a later phase)
  int reuse = 0;
  if (prefix_cacheable && !resident_prefix_.empty()) {
    // Cap at prompt size - 1 so at least one token remains to seed decode.
    const int cap = std::min(static_cast<int>(resident_prefix_.size()),
                             static_cast<int>(prompt_tokens.size()) - 1);
    while (reuse < cap && prompt_tokens[static_cast<std::size_t>(reuse)] ==
                              resident_prefix_[static_cast<std::size_t>(reuse)]) {
      ++reuse;
    }
  }
  if (reuse == 0) {
    reset_kv_cache();  // wipes KV and clears resident_prefix_
  }
  // Invalidate the tracked prefix until this generation re-establishes it after
  // the decode loop; the reuse count above already captured the prior value.
  resident_prefix_.clear();

  // P3 phase 2a: mirror this sequence's positions in the paged block table so the
  // allocator is exercised against real sequence lengths (allocation + pool
  // sizing + no-exhaustion + release across requests). Bookkeeping only in 2a —
  // the KV hot path still uses the flat cache, so output stays byte-identical.
  // Physical block ids are intentionally NOT contiguous/identity (the free-list
  // hands blocks back LIFO across requests) — that permutation is exactly what
  // paging enables, and is consumed by the phase-2b gather kernel that will read
  // KV through this table instead of a flat offset.
  if (seq_blocks_) {
    // Phase 2d: allocate this sequence's blocks and publish its block table
    // (host + device). The allocator's free-list is LIFO, so requests after the
    // first get NON-CONTIGUOUS physical blocks — exercising real paging. Prefill
    // + decode KV writes and both attention reads all go through this table, so
    // byte-identical output proves non-contiguous paging works.
    seq_blocks_->clear();
    const int total = std::min(
        options_.max_context, static_cast<int>(prompt_tokens.size()) + std::max(0, max_new_tokens));
    if (!seq_blocks_->ensure_position(total - 1)) {
      LLAMA_ENGINE_THROW("paged KV pool exhausted for sequence length " + std::to_string(total));
    }
    const int bs = options_.paged_block_size > 0 ? options_.paged_block_size : 32;
    const int nblk = (total + bs - 1) / bs;
    block_table_host_.assign(static_cast<std::size_t>(nblk), 0);
    for (int c = 0; c < nblk; ++c) {
      const int blk = seq_blocks_->block_for(c * bs);
      if (blk == BlockAllocator::kInvalidBlock) {
        LLAMA_ENGINE_THROW("paged block table has no block for chunk " + std::to_string(c));
      }
      block_table_host_[static_cast<std::size_t>(c)] = blk;
    }
    CUDA_CHECK(cudaMemcpy(d_block_table_, block_table_host_.data(),
                          static_cast<std::size_t>(nblk) * sizeof(int), cudaMemcpyHostToDevice));
  }

  enforce_host_resource_limits("generate.begin");
  std::vector<int> out = prompt_tokens;
  out.reserve(prompt_tokens.size() + max_new_tokens);
  last_benchmark_stats_ = {};
  last_benchmark_stats_.prompt_tokens = static_cast<int>(prompt_tokens.size());
  if (!cfg.is_moe()) {
    last_benchmark_stats_.moe_quant_mode = "none";
  }
  const bool tq3_cached_active = tq3_enabled_ && cached_layer_count_ == cfg.num_layers &&
                                 !layer_cache_tq3_.empty() &&
                                 static_cast<int>(layer_cache_tq3_.size()) == cfg.num_layers;
  last_benchmark_stats_.tq3_cached_active = tq3_cached_active ? 1 : 0;
  if (options_.verbose && tq3_enabled_) {
    std::cout << "[engine] tq3_cached_active=" << (tq3_cached_active ? 1 : 0) << "\n";
  }
  benchmark_transfer_active_ = false;
  greedy_decode_graph_state_valid_ = false;

  if (options_.verbose && reuse > 0) {
    std::cout << "[engine] prefix_reuse tokens=" << reuse << "/" << prompt_tokens.size() << "\n";
  }
  const auto prefill_start = std::chrono::steady_clock::now();
  prefill_prompt(prompt_tokens, reuse);
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  const auto prefill_end = std::chrono::steady_clock::now();
  last_benchmark_stats_.prefill_ms =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start).count();
  if (options_.verbose) {
    std::cout << "[engine] prefill_done ms=" << last_benchmark_stats_.prefill_ms << "\n";
  }

  int current = prompt_tokens.back();
  int pos = static_cast<int>(prompt_tokens.size()) - 1;
  const auto decode_start = std::chrono::steady_clock::now();
  for (int i = 0; i < max_new_tokens; ++i) {
    enforce_host_resource_limits("decode.step");
    // Suppress EOS until at least min_new_tokens have been generated, so greedy
    // decoding can't terminate early on a collapsed/repeated-token distribution.
    suppress_eos_ = (i < min_new_tokens);
    if (options_.verbose && i < 3) {
      std::cout << "[engine] decode_step i=" << i << " pos=" << pos << "\n";
    }
    const auto token_start = std::chrono::steady_clock::now();
    const int next = decode_next_token(current, pos, temperature, out);
    if (i == 0 && tq3_cached_active && options_.tq_first_token_timeout_ms > 0) {
      const auto token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                                std::chrono::steady_clock::now() - token_start)
                                .count();
      if (token_ms > options_.tq_first_token_timeout_ms) {
        std::ostringstream oss;
        oss << "TurboQuant cached first-token timeout: elapsed_ms=" << token_ms
            << " limit_ms=" << options_.tq_first_token_timeout_ms;
        LLAMA_ENGINE_THROW(oss.str());
      }
    }
    if (active_grammar_ != nullptr) {
      active_grammar_->accept(next);  // advance grammar state by the chosen token
    }
    out.push_back(next);
    if (on_token && !on_token(next)) {
      break;
    }
    if (options_.loop_guard && detail::dispatch_has_degenerate_tail(out, prompt_tokens.size())) {
      if (options_.verbose) {
        std::cout << "[engine] stopping early due to repetitive decode loop\n";
      }
      break;
    }
    if (options_.eos_token_id >= 0 && next == options_.eos_token_id) {
      break;
    }
    current = next;
    ++pos;
  }
  enforce_host_resource_limits("decode.done");
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  if (benchmark_transfer_active_) {
    CUDA_CHECK(cudaStreamSynchronize(transfer_stream_));
    float transfer_ms = 0.0f;
    CUDA_CHECK(
        cudaEventElapsedTime(&transfer_ms, benchmark_transfer_start_, benchmark_transfer_end_));
    last_benchmark_stats_.transfer_ms = static_cast<double>(transfer_ms);
  }
  const auto decode_end = std::chrono::steady_clock::now();
  last_benchmark_stats_.decode_ms =
      std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
  last_benchmark_stats_.generated_tokens = static_cast<int>(
      (out.size() > prompt_tokens.size()) ? (out.size() - prompt_tokens.size()) : 0);

  // Record the resident prefix for the next request's reuse. KV[0, prompt_len)
  // is valid only once at least one decode step has run (it writes the last
  // prompt position's KV); if none ran, only prefill's [0, prompt_len-1) is valid.
  if (prefix_cacheable) {
    const std::size_t valid = (out.size() > prompt_tokens.size())
                                  ? prompt_tokens.size()
                                  : (prompt_tokens.empty() ? 0 : prompt_tokens.size() - 1);
    resident_prefix_.assign(prompt_tokens.begin(),
                            prompt_tokens.begin() + static_cast<std::ptrdiff_t>(valid));
  }

  return out;
}

std::vector<std::pair<int, float>> LlamaEngine::inspect_next_logits(
    const std::vector<int>& prompt_tokens, int top_k) {
  if (prompt_tokens.empty()) {
    LLAMA_ENGINE_THROW("inspect_next_logits requires non-empty prompt");
  }
  if (top_k <= 0) {
    return {};
  }

  reset_kv_cache();
  enforce_host_resource_limits("inspect.prefill");
  prefill_prompt(prompt_tokens);
  std::vector<float> logits;
  forward_token_logits(prompt_tokens.back(), static_cast<int>(prompt_tokens.size()) - 1, &logits,
                       nullptr);

  std::vector<int> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  const int k = std::min<int>(top_k, static_cast<int>(order.size()));
  std::partial_sort(order.begin(), order.begin() + k, order.end(), [&](int a, int b) {
    return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)];
  });

  std::vector<std::pair<int, float>> out;
  out.reserve(static_cast<std::size_t>(k));
  for (int i = 0; i < k; ++i) {
    const int id = order[static_cast<std::size_t>(i)];
    out.push_back({id, logits[static_cast<std::size_t>(id)]});
  }
  return out;
}

void LlamaEngine::verify_tokens(const std::vector<int>& tokens, int start_pos,
                                std::vector<int>& out_argmax) {
  const auto& cfg = weights_.config();
  const int K = static_cast<int>(tokens.size());
  out_argmax.assign(static_cast<std::size_t>(K), 0);
  if (K == 0) {
    return;
  }

  // Verify needs the same batched full-attention path as prefill; the scratch
  // buffers are sized for prefill_chunk_size_ rows.
  const bool all_layers_cached = cached_layer_count_ == cfg.num_layers;
  if (options_.paged_kv_cache || prefill_chunk_size_ <= 1 ||
      (cached_int8_proj_enabled_ && !all_layers_cached) || kv_int4_enabled_ || tq3_enabled_ ||
      cfg.is_moe() || cfg.uses_non_full_attention()) {
    LLAMA_ENGINE_THROW("verify_tokens requires the batched full-attention path");
  }
  if (K > prefill_chunk_size_) {
    LLAMA_ENGINE_THROW("verify_tokens: K exceeds prefill_chunk_size_");
  }

  const int hidden = cfg.hidden_size;

  // Run the K tokens through all layers in one batched pass at absolute
  // positions [start_pos, start_pos+K); this writes their K/V into the cache.
  CUDA_CHECK(cudaMemcpyAsync(d_token_id_, tokens.data(), static_cast<std::size_t>(K) * sizeof(int),
                             cudaMemcpyHostToDevice, compute_stream_));
  run_batched_chunk(K, start_pos);
  if (!all_layers_cached) {
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  }

  // Final RMSNorm over all K rows, then per-position LM head + argmax. d_logits_
  // and d_argmax_ are reused each iteration; stream ordering guarantees the
  // i-th argmax is copied to the host before the (i+1)-th overwrites it.
  launch_norm(d_x_, d_norm_out_, d_norm_out_bias_, d_x_norm_, K, hidden);

  // Batched LM head: one GEMM projects all K normed rows in d_x_norm_ through the
  // lm_head, reading its ~1 GB weight ONCE — the per-position loop re-read it K
  // times (K x the lm_head bandwidth per verify). Consistent with the verify's
  // layer projections, which already run through the batched cuBLAS path. Then a
  // per-row device argmax (only K ints cross the bus).
  batched_lm_head(K, hidden, cfg.vocab_size);
  for (int i = 0; i < K; ++i) {
    kernels::launch_argmax_float(
        d_batch_logits_ + static_cast<std::size_t>(i) * static_cast<std::size_t>(cfg.vocab_size),
        cfg.vocab_size, d_argmax_, compute_stream_);
    CUDA_CHECK(cudaMemcpyAsync(out_argmax.data() + i, d_argmax_, sizeof(int),
                               cudaMemcpyDeviceToHost, compute_stream_));
  }
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
}

}  // namespace engine
