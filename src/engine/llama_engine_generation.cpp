#include "engine/llama_engine.hpp"
#include "llama_engine_internal.hpp"

#include <algorithm>
#include <chrono>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <cuda_fp16.h>

#include "common.hpp"
#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"

namespace engine {

std::vector<int> LlamaEngine::generate(const std::vector<int>& prompt_tokens,
                                       int max_new_tokens,
                                       float temperature) {
  return generate_stream(prompt_tokens, max_new_tokens, temperature, {});
}

void LlamaEngine::prefill_prompt_sequential(const std::vector<int>& prompt_tokens) {
  if (prompt_tokens.size() <= 1) {
    return;
  }
  const bool needs_streaming_barrier = cached_layer_count_ < weights_.config().num_layers;
  for (int i = 0; i < static_cast<int>(prompt_tokens.size()) - 1; ++i) {
    enforce_host_resource_limits("prefill.sequential");
    forward_token(prompt_tokens[static_cast<std::size_t>(i)], i, false, nullptr, nullptr);
    if (needs_streaming_barrier) {
      CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    }
  }
}

void LlamaEngine::prefill_prompt(const std::vector<int>& prompt_tokens) {
  if (prompt_tokens.size() <= 1) {
    return;
  }
  const auto& cfg = weights_.config();
  // Batched prefill requires fp16 projection weights; fall back to sequential when int8 proj or TQ3 is active.
  if (options_.paged_kv_cache || prefill_chunk_size_ <= 1 || cached_int8_proj_enabled_ ||
      kv_int4_enabled_ || tq3_enabled_ || cfg.is_moe() || cfg.sliding_window > 0) {
    prefill_prompt_sequential(prompt_tokens);
    return;
  }
  const int hidden = cfg.hidden_size;
  const int inter = cfg.intermediate_size;
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : hidden;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (hidden / cfg.num_heads);
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const int prompt_count = static_cast<int>(prompt_tokens.size()) - 1;
  const bool can_use_dp4a_prefill = ((hidden & 3) == 0) && ((inter & 3) == 0);
  const bool all_layers_cached = cached_layer_count_ == cfg.num_layers;
  const std::size_t q_row_bytes = static_cast<std::size_t>(q_hidden) * sizeof(__half);
  const std::size_t kv_row_bytes = static_cast<std::size_t>(kv_hidden) * sizeof(__half);
  const std::size_t ff_row_bytes = static_cast<std::size_t>(inter) * sizeof(__half);
  const std::size_t qkv_stride_bytes = static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * sizeof(__half);
  const std::size_t ff13_stride_bytes = static_cast<std::size_t>(2 * inter) * sizeof(__half);
  const std::size_t layer_stride = static_cast<std::size_t>(options_.max_context) * static_cast<std::size_t>(kv_hidden);
  auto* qkv_base = static_cast<const __half*>(d_qkv_);
  auto* ff13_base = static_cast<const __half*>(d_ff13_);

  /*
   * Process prompt tokens in small fp16 chunks so we can reuse larger GEMMs
   * and batched attention without changing the exact decode path.
   */
  for (int chunk_start = 0; chunk_start < prompt_count; chunk_start += prefill_chunk_size_) {
    enforce_host_resource_limits("prefill.chunk_begin");
    const int rows = std::min(prefill_chunk_size_, prompt_count - chunk_start);
    CUDA_CHECK(cudaMemcpyAsync(d_token_id_,
                               prompt_tokens.data() + chunk_start,
                               static_cast<std::size_t>(rows) * sizeof(int),
                               cudaMemcpyHostToDevice,
                               compute_stream_));
    kernels::launch_embedding_lookup(static_cast<const __half*>(d_tok_embeddings_),
                                     d_token_id_,
                                     static_cast<__half*>(d_x_),
                                     rows,
                                     hidden,
                                     compute_stream_);

    const auto run_layer = [&](int layer, const LayerDeviceWeights* lw, const LayerDeviceInt8Weights* lw_i8) {
      launch_norm(d_x_, lw->norm_att, lw->norm_att_bias, d_x_norm_, rows, hidden);

      detail::dispatch_linear_rowmajor_weight(cublas_,
                             cublas_lt_,
                             &lt_plan_cache_,
                             lt_workspace_,
                             lt_workspace_bytes_,
                             compute_stream_,
                             lw->wqkv,
                             d_x_norm_,
                             d_qkv_,
                             q_hidden + 2 * kv_hidden,
                             hidden,
                             rows,
                             CUDA_R_16F);
      maybe_add_half_bias(d_ff3_, lw->bo, rows, hidden);

      CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_q_,
                                   q_row_bytes,
                                   qkv_base,
                                   qkv_stride_bytes,
                                   q_row_bytes,
                                   rows,
                                   cudaMemcpyDeviceToDevice,
                                   compute_stream_));
      CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_k_,
                                   kv_row_bytes,
                                   qkv_base + q_hidden,
                                   qkv_stride_bytes,
                                   kv_row_bytes,
                                   rows,
                                   cudaMemcpyDeviceToDevice,
                                   compute_stream_));
      CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_v_,
                                   kv_row_bytes,
                                   qkv_base + q_hidden + kv_hidden,
                                   qkv_stride_bytes,
                                   kv_row_bytes,
                                   rows,
                                   cudaMemcpyDeviceToDevice,
                                   compute_stream_));

      if (lw->bqkv) {
        const auto* bqkv_half = static_cast<const __half*>(lw->bqkv);
        kernels::launch_add_bias_broadcast(static_cast<__half*>(d_prefill_q_), bqkv_half, rows, q_hidden, compute_stream_);
        kernels::launch_add_bias_broadcast(static_cast<__half*>(d_prefill_k_), bqkv_half + q_hidden, rows, kv_hidden, compute_stream_);
        kernels::launch_add_bias_broadcast(static_cast<__half*>(d_prefill_v_), bqkv_half + q_hidden + kv_hidden, rows, kv_hidden, compute_stream_);
      }

      kernels::launch_rope_inplace_batched(static_cast<__half*>(d_prefill_q_),
                                           static_cast<__half*>(d_prefill_k_),
                                           rows,
                                           cfg.num_heads,
                                           cfg.num_kv_heads,
                                           head_dim,
                                           chunk_start,
                                           d_rope_cos_,
                                           d_rope_sin_,
                                           compute_stream_);

      auto* k_layer = static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride;
      auto* v_layer = static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride;
      CUDA_CHECK(cudaMemcpy2DAsync(k_layer + static_cast<std::size_t>(chunk_start) * kv_hidden,
                                   kv_row_bytes,
                                   d_prefill_k_,
                                   kv_row_bytes,
                                   kv_row_bytes,
                                   rows,
                                   cudaMemcpyDeviceToDevice,
                                   compute_stream_));
      CUDA_CHECK(cudaMemcpy2DAsync(v_layer + static_cast<std::size_t>(chunk_start) * kv_hidden,
                                   kv_row_bytes,
                                   d_prefill_v_,
                                   kv_row_bytes,
                                   kv_row_bytes,
                                   rows,
                                   cudaMemcpyDeviceToDevice,
                                   compute_stream_));

      kernels::launch_attention_prefill(static_cast<const __half*>(d_prefill_q_),
                                        k_layer,
                                        v_layer,
                                        static_cast<__half*>(d_att_),
                                        rows,
                                        chunk_start,
                                        cfg.num_heads,
                                        cfg.num_kv_heads,
                                        head_dim,
                                        compute_stream_);

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
                             q_hidden,
                             rows,
                             CUDA_R_16F);

      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  static_cast<const __half*>(d_ff3_),
                                  rows * hidden,
                                  compute_stream_);

      launch_norm(d_x_, lw->norm_ffn, lw->norm_ffn_bias, d_x_norm_, rows, hidden);

      if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && can_use_dp4a_prefill) {
        kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_x_norm_),
                                                      d_prefill_i8_,
                                                      d_prefill_i8_scales_,
                                                      rows,
                                                      hidden,
                                                      compute_stream_);
        if (lw_i8->mlp_int4) {
          kernels::launch_weight_only_int4_matvec_batched_dp4a(lw_i8->w1,
                                                               lw_i8->s_w1,
                                                               d_prefill_i8_,
                                                               d_prefill_i8_scales_,
                                                               static_cast<__half*>(d_prefill_ff1_),
                                                               rows,
                                                               inter,
                                                               hidden,
                                                               compute_stream_);
          kernels::launch_weight_only_int4_matvec_batched_dp4a(lw_i8->w3,
                                                               lw_i8->s_w3,
                                                               d_prefill_i8_,
                                                               d_prefill_i8_scales_,
                                                               static_cast<__half*>(d_prefill_ff2_),
                                                               rows,
                                                               inter,
                                                               hidden,
                                                               compute_stream_);
        } else {
          kernels::launch_weight_only_int8_matvec_batched_dp4a(lw_i8->w1,
                                                               lw_i8->s_w1,
                                                               d_prefill_i8_,
                                                               d_prefill_i8_scales_,
                                                               static_cast<__half*>(d_prefill_ff1_),
                                                               rows,
                                                               inter,
                                                               hidden,
                                                               compute_stream_);
          kernels::launch_weight_only_int8_matvec_batched_dp4a(lw_i8->w3,
                                                               lw_i8->s_w3,
                                                               d_prefill_i8_,
                                                               d_prefill_i8_scales_,
                                                               static_cast<__half*>(d_prefill_ff2_),
                                                               rows,
                                                               inter,
                                                               hidden,
                                                               compute_stream_);
        }
      } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3) {
        if (lw_i8->mlp_int4) {
          kernels::launch_weight_only_int4_matvec_batched(lw_i8->w1,
                                                          lw_i8->s_w1,
                                                          static_cast<const __half*>(d_x_norm_),
                                                          static_cast<__half*>(d_prefill_ff1_),
                                                          rows,
                                                          inter,
                                                          hidden,
                                                          compute_stream_);
          kernels::launch_weight_only_int4_matvec_batched(lw_i8->w3,
                                                          lw_i8->s_w3,
                                                          static_cast<const __half*>(d_x_norm_),
                                                          static_cast<__half*>(d_prefill_ff2_),
                                                          rows,
                                                          inter,
                                                          hidden,
                                                          compute_stream_);
        } else {
          kernels::launch_weight_only_int8_matvec_batched(lw_i8->w1,
                                                          lw_i8->s_w1,
                                                          static_cast<const __half*>(d_x_norm_),
                                                          static_cast<__half*>(d_prefill_ff1_),
                                                          rows,
                                                          inter,
                                                          hidden,
                                                          compute_stream_);
          kernels::launch_weight_only_int8_matvec_batched(lw_i8->w3,
                                                          lw_i8->s_w3,
                                                          static_cast<const __half*>(d_x_norm_),
                                                          static_cast<__half*>(d_prefill_ff2_),
                                                          rows,
                                                          inter,
                                                          hidden,
                                                          compute_stream_);
        }
      } else {
        detail::dispatch_linear_rowmajor_weight(cublas_,
                               cublas_lt_,
                               &lt_plan_cache_,
                               lt_workspace_,
                               lt_workspace_bytes_,
                               compute_stream_,
                               lw->w13,
                               d_x_norm_,
                               d_ff13_,
                               2 * inter,
                               hidden,
                               rows,
                               CUDA_R_16F);

        CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_ff1_,
                                     ff_row_bytes,
                                     ff13_base,
                                     ff13_stride_bytes,
                                     ff_row_bytes,
                                     rows,
                                     cudaMemcpyDeviceToDevice,
                                     compute_stream_));
        CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_ff2_,
                                     ff_row_bytes,
                                     ff13_base + inter,
                                     ff13_stride_bytes,
                                     ff_row_bytes,
                                     rows,
                                     cudaMemcpyDeviceToDevice,
                                     compute_stream_));
      }

      kernels::launch_silu_mul(static_cast<const __half*>(d_prefill_ff1_),
                               static_cast<const __half*>(d_prefill_ff2_),
                               static_cast<__half*>(d_prefill_ff2_),
                               rows * inter,
                               compute_stream_);

      if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && can_use_dp4a_prefill) {
        kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_prefill_ff2_),
                                                      d_prefill_i8_,
                                                      d_prefill_i8_scales_,
                                                      rows,
                                                      inter,
                                                      compute_stream_);
        if (lw_i8->mlp_int4) {
          kernels::launch_weight_only_int4_matvec_batched_dp4a(lw_i8->w2,
                                                               lw_i8->s_w2,
                                                               d_prefill_i8_,
                                                               d_prefill_i8_scales_,
                                                               static_cast<__half*>(d_ff3_),
                                                               rows,
                                                               hidden,
                                                               inter,
                                                               compute_stream_);
        } else {
          kernels::launch_weight_only_int8_matvec_batched_dp4a(lw_i8->w2,
                                                               lw_i8->s_w2,
                                                               d_prefill_i8_,
                                                               d_prefill_i8_scales_,
                                                               static_cast<__half*>(d_ff3_),
                                                               rows,
                                                               hidden,
                                                               inter,
                                                               compute_stream_);
        }
      } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3) {
        if (lw_i8->mlp_int4) {
          kernels::launch_weight_only_int4_matvec_batched(lw_i8->w2,
                                                          lw_i8->s_w2,
                                                          static_cast<const __half*>(d_prefill_ff2_),
                                                          static_cast<__half*>(d_ff3_),
                                                          rows,
                                                          hidden,
                                                          inter,
                                                          compute_stream_);
        } else {
          kernels::launch_weight_only_int8_matvec_batched(lw_i8->w2,
                                                          lw_i8->s_w2,
                                                          static_cast<const __half*>(d_prefill_ff2_),
                                                          static_cast<__half*>(d_ff3_),
                                                          rows,
                                                          hidden,
                                                          inter,
                                                          compute_stream_);
        }
      } else {
        detail::dispatch_linear_rowmajor_weight(cublas_,
                               cublas_lt_,
                               &lt_plan_cache_,
                               lt_workspace_,
                               lt_workspace_bytes_,
                               compute_stream_,
                               lw->w2,
                               d_prefill_ff2_,
                               d_ff3_,
                               hidden,
                               inter,
                               rows,
                               CUDA_R_16F);
      }

      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  static_cast<const __half*>(d_ff3_),
                                  rows * hidden,
                                  compute_stream_);
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
            copy_layer_weights_to_device(next_layer,
                                         &streaming_layer_weights_[next_slot],
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

  if (!all_layers_cached) {
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  }
  enforce_host_resource_limits("prefill.done");
}

std::vector<int> LlamaEngine::generate_stream(const std::vector<int>& prompt_tokens,
                                              int max_new_tokens,
                                              float temperature,
                                              const std::function<bool(int)>& on_token) {
  if (prompt_tokens.empty()) {
    LLAMA_ENGINE_THROW("prompt token list is empty");
  }

  reset_kv_cache();
  enforce_host_resource_limits("generate.begin");
  std::vector<int> out = prompt_tokens;
  out.reserve(prompt_tokens.size() + max_new_tokens);
  last_benchmark_stats_ = {};
  last_benchmark_stats_.prompt_tokens = static_cast<int>(prompt_tokens.size());
  const auto& cfg = weights_.config();
  if (!cfg.is_moe()) {
    last_benchmark_stats_.moe_quant_mode = "none";
  }
  const bool tq3_cached_active =
      tq3_enabled_ && cached_layer_count_ == cfg.num_layers && !layer_cache_tq3_.empty() &&
      static_cast<int>(layer_cache_tq3_.size()) == cfg.num_layers;
  last_benchmark_stats_.tq3_cached_active = tq3_cached_active ? 1 : 0;
  if (options_.verbose && tq3_enabled_) {
    std::cout << "[engine] tq3_cached_active=" << (tq3_cached_active ? 1 : 0) << "\n";
  }
  benchmark_transfer_active_ = false;
  greedy_decode_graph_state_valid_ = false;

  const auto prefill_start = std::chrono::steady_clock::now();
  prefill_prompt(prompt_tokens);
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
    if (options_.verbose && i < 3) {
      std::cout << "[engine] decode_step i=" << i << " pos=" << pos << "\n";
    }
    const auto token_start = std::chrono::steady_clock::now();
    const int next = decode_next_token(current, pos, temperature, out);
    if (i == 0 && tq3_cached_active && options_.tq_first_token_timeout_ms > 0) {
      const auto token_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - token_start).count();
      if (token_ms > options_.tq_first_token_timeout_ms) {
        std::ostringstream oss;
        oss << "TurboQuant cached first-token timeout: elapsed_ms=" << token_ms
            << " limit_ms=" << options_.tq_first_token_timeout_ms;
        LLAMA_ENGINE_THROW(oss.str());
      }
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
    CUDA_CHECK(cudaEventElapsedTime(&transfer_ms, benchmark_transfer_start_, benchmark_transfer_end_));
    last_benchmark_stats_.transfer_ms = static_cast<double>(transfer_ms);
  }
  const auto decode_end = std::chrono::steady_clock::now();
  last_benchmark_stats_.decode_ms =
      std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
  last_benchmark_stats_.generated_tokens =
      static_cast<int>((out.size() > prompt_tokens.size()) ? (out.size() - prompt_tokens.size()) : 0);

  return out;
}

std::vector<std::pair<int, float>> LlamaEngine::inspect_next_logits(const std::vector<int>& prompt_tokens, int top_k) {
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
  forward_token_logits(prompt_tokens.back(),
                       static_cast<int>(prompt_tokens.size()) - 1,
                       &logits,
                       nullptr);

  std::vector<int> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  const int k = std::min<int>(top_k, static_cast<int>(order.size()));
  std::partial_sort(order.begin(),
                    order.begin() + k,
                    order.end(),
                    [&](int a, int b) { return logits[static_cast<std::size_t>(a)] > logits[static_cast<std::size_t>(b)]; });

  std::vector<std::pair<int, float>> out;
  out.reserve(static_cast<std::size_t>(k));
  for (int i = 0; i < k; ++i) {
    const int id = order[static_cast<std::size_t>(i)];
    out.push_back({id, logits[static_cast<std::size_t>(id)]});
  }
  return out;
}


}  // namespace engine
