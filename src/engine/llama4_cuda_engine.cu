#include "engine/llama4_cuda_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

#include <cuda_fp16.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/mman.h>
#endif

#include "common.hpp"
#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"

namespace engine {
namespace {


// Issue a non-blocking OS hint to preload pages from a memory-mapped range
// into the working set.  Returns immediately; the OS satisfies the I/O in
// the background, so subsequent access will not page-fault.
void prefetch_range(const void* ptr, std::size_t bytes) {
  if (!ptr || bytes == 0) return;
#ifdef _WIN32
  WIN32_MEMORY_RANGE_ENTRY entry{const_cast<PVOID>(ptr), bytes};
  PrefetchVirtualMemory(GetCurrentProcess(), 1, &entry, 0);
#else
  madvise(const_cast<void*>(ptr), bytes, MADV_WILLNEED);
#endif
}




template <typename Launch>
double measure_stream_ms(cudaStream_t stream, Launch&& launch) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, stream));
  launch();
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  return static_cast<double>(ms);
}

int sample_from_logits(std::vector<float>& logits,
                       float temperature,
                       int top_k,
                       float top_p,
                       float repetition_penalty,
                       int no_repeat_ngram_size,
                       const std::vector<int>& history) {
  if (logits.empty()) {
    return 0;
  }

  for (float& v : logits) {
    if (!std::isfinite(v)) {
      v = -std::numeric_limits<float>::infinity();
      continue;
    }
    if (v > 80.0f) {
      v = 80.0f;
    } else if (v < -80.0f) {
      v = -80.0f;
    }
  }

  if (repetition_penalty > 1.0f && !history.empty()) {
    std::unordered_set<int> seen(history.begin(), history.end());
    for (int id : seen) {
      if (id < 0 || id >= static_cast<int>(logits.size())) {
        continue;
      }
      if (logits[static_cast<std::size_t>(id)] > 0.0f) {
        logits[static_cast<std::size_t>(id)] /= repetition_penalty;
      } else {
        logits[static_cast<std::size_t>(id)] *= repetition_penalty;
      }
    }
  }

  if (no_repeat_ngram_size > 1 &&
      history.size() + 1 >= static_cast<std::size_t>(no_repeat_ngram_size)) {
    const int n = no_repeat_ngram_size;
    const int prefix_len = n - 1;
    const int hist_size = static_cast<int>(history.size());
    std::vector<int> prefix(static_cast<std::size_t>(prefix_len));
    for (int i = 0; i < prefix_len; ++i) {
      prefix[static_cast<std::size_t>(i)] =
          history[static_cast<std::size_t>(hist_size - prefix_len + i)];
    }

    std::vector<char> banned(logits.size(), 0);
    for (int i = 0; i + n <= hist_size; ++i) {
      bool match = true;
      for (int j = 0; j < prefix_len; ++j) {
        if (history[static_cast<std::size_t>(i + j)] !=
            prefix[static_cast<std::size_t>(j)]) {
          match = false;
          break;
        }
      }
      if (match) {
        const int next_id = history[static_cast<std::size_t>(i + prefix_len)];
        if (next_id >= 0 && next_id < static_cast<int>(banned.size())) {
          banned[static_cast<std::size_t>(next_id)] = 1;
        }
      }
    }

    bool has_candidate = false;
    for (std::size_t i = 0; i < logits.size(); ++i) {
      if (banned[i]) {
        logits[i] = -std::numeric_limits<float>::infinity();
      } else if (std::isfinite(logits[i])) {
        has_candidate = true;
      }
    }
    if (!has_candidate) {
      return 0;
    }
  }

  if (temperature <= 0.0f) {
    return static_cast<int>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
  }

  const float inv_temp = 1.0f / temperature;
  float max_logit = -std::numeric_limits<float>::infinity();
  for (float v : logits) {
    if (std::isfinite(v)) {
      max_logit = std::max(max_logit, v * inv_temp);
    }
  }
  if (!std::isfinite(max_logit)) {
    return 0;
  }

  float sum = 0.0f;
  for (float& v : logits) {
    if (!std::isfinite(v)) {
      v = 0.0f;
      continue;
    }
    v = std::exp(v * inv_temp - max_logit);
    sum += v;
  }
  if (sum <= 0.0f || !std::isfinite(sum)) {
    return 0;
  }
  const float inv_sum = 1.0f / sum;
  for (float& v : logits) {
    v *= inv_sum;
  }

  std::vector<int> order(logits.size());
  std::iota(order.begin(), order.end(), 0);
  const int k = std::clamp(top_k, 1, static_cast<int>(order.size()));
  std::partial_sort(order.begin(),
                    order.begin() + k,
                    order.end(),
                    [&](int a, int b) {
                      return logits[static_cast<std::size_t>(a)] >
                             logits[static_cast<std::size_t>(b)];
                    });

  std::vector<int> kept_ids;
  kept_ids.reserve(static_cast<std::size_t>(k));
  float kept_sum = 0.0f;
  for (int i = 0; i < k; ++i) {
    const int id = order[static_cast<std::size_t>(i)];
    const float p = logits[static_cast<std::size_t>(id)];
    if (p <= 0.0f || !std::isfinite(p)) {
      continue;
    }
    kept_ids.push_back(id);
    kept_sum += p;
    if (top_p > 0.0f && top_p < 1.0f && kept_sum >= top_p) {
      break;
    }
  }
  if (kept_ids.empty()) {
    return order.front();
  }

  float renorm = 0.0f;
  for (int id : kept_ids) {
    renorm += logits[static_cast<std::size_t>(id)];
  }
  if (renorm <= 0.0f || !std::isfinite(renorm)) {
    return kept_ids.front();
  }

  static thread_local std::mt19937 rng(std::random_device{}());
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float draw = dist(rng);
  float running = 0.0f;
  for (int id : kept_ids) {
    running += logits[static_cast<std::size_t>(id)] / renorm;
    if (draw <= running) {
      return id;
    }
  }
  return kept_ids.back();
}

}  // namespace

Llama4CudaEngine::~Llama4CudaEngine() {
  destroy();
}




void Llama4CudaEngine::forward_token(int token,
                                     int position,
                                     bool compute_logits,
                                     std::vector<float>* out_logits,
                                     int* out_argmax) {
  if (position < 0 || position >= max_ctx_) {
    LLAMA_ENGINE_THROW("decode position exceeds max context");
  }

  auto time_or_launch = [&](double* slot, auto&& launch) {
    if (slot && options_.profile_decode_phases) {
      *slot += measure_stream_ms(compute_stream_, launch);
    } else {
      launch();
    }
  };

  load_token_embedding_to_device(token);

  constexpr float kRmsNormEps = 1.0e-5f;
  for (int layer = 0; layer < n_layers_; ++layer) {
    const auto& weights = layer_device_[static_cast<std::size_t>(layer)];
    auto* k_cache_layer =
        static_cast<__half*>(d_k_cache_) +
        static_cast<std::size_t>(layer) * static_cast<std::size_t>(max_ctx_) *
            static_cast<std::size_t>(kv_dim_);
    auto* v_cache_layer =
        static_cast<__half*>(d_v_cache_) +
        static_cast<std::size_t>(layer) * static_cast<std::size_t>(max_ctx_) *
            static_cast<std::size_t>(kv_dim_);

    time_or_launch(&last_benchmark_stats_.decode_rmsnorm_ms, [&] {
      kernels::launch_rmsnorm(static_cast<const __half*>(d_x_),
                              static_cast<const __half*>(weights.norm_att),
                              static_cast<__half*>(d_x_norm_),
                              1,
                              hidden_,
                              kRmsNormEps,
                              compute_stream_);
    });

    time_or_launch(&last_benchmark_stats_.decode_qkv_ms, [&] {
      rowmajor_projection_half(weights.wq, d_x_norm_, d_q_, hidden_, hidden_);
      rowmajor_projection_half(weights.wk, d_x_norm_, d_k_, kv_dim_, hidden_);
      rowmajor_projection_half(weights.wv, d_x_norm_, d_v_, kv_dim_, hidden_);
    });

    if (use_qk_norm_) {
      kernels::launch_rmsnorm(static_cast<const __half*>(d_q_),
                              static_cast<const __half*>(weights.q_norm
                                                             ? weights.q_norm
                                                             : d_q_norm_unit_),
                              static_cast<__half*>(d_q_),
                              n_heads_,
                              head_dim_,
                              kRmsNormEps,
                              compute_stream_);
      kernels::launch_rmsnorm(static_cast<const __half*>(d_k_),
                              static_cast<const __half*>(weights.k_norm
                                                             ? weights.k_norm
                                                             : d_k_norm_unit_),
                              static_cast<__half*>(d_k_),
                              n_kv_heads_,
                              head_dim_,
                              kRmsNormEps,
                              compute_stream_);
    }

    kernels::launch_rope_inplace_table(static_cast<__half*>(d_q_),
                                       static_cast<__half*>(d_k_),
                                       n_heads_,
                                       n_kv_heads_,
                                       head_dim_,
                                       position,
                                       static_cast<const float*>(d_rope_cos_),
                                       static_cast<const float*>(d_rope_sin_),
                                       compute_stream_);

    time_or_launch(&last_benchmark_stats_.decode_kv_store_ms, [&] {
      CUDA_CHECK(cudaMemcpyAsync(
          k_cache_layer +
              static_cast<std::size_t>(position) *
                  static_cast<std::size_t>(kv_dim_),
          d_k_,
          static_cast<std::size_t>(kv_dim_) * sizeof(__half),
          cudaMemcpyDeviceToDevice,
          compute_stream_));
      CUDA_CHECK(cudaMemcpyAsync(
          v_cache_layer +
              static_cast<std::size_t>(position) *
                  static_cast<std::size_t>(kv_dim_),
          d_v_,
          static_cast<std::size_t>(kv_dim_) * sizeof(__half),
          cudaMemcpyDeviceToDevice,
          compute_stream_));
    });

    time_or_launch(&last_benchmark_stats_.decode_attention_ms, [&] {
      kernels::launch_attention_step(static_cast<const __half*>(d_q_),
                                     k_cache_layer,
                                     v_cache_layer,
                                     static_cast<__half*>(d_att_),
                                     position + 1,
                                     n_heads_,
                                     n_kv_heads_,
                                     head_dim_,
                                     compute_stream_);
    });

    time_or_launch(&last_benchmark_stats_.decode_wo_ms, [&] {
      rowmajor_projection_half(weights.wo, d_att_, d_tmp_hidden_, hidden_, hidden_);
      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  static_cast<const __half*>(d_tmp_hidden_),
                                  hidden_,
                                  compute_stream_);
    });

    time_or_launch(&last_benchmark_stats_.decode_rmsnorm_ms, [&] {
      kernels::launch_rmsnorm(static_cast<const __half*>(d_x_),
                              static_cast<const __half*>(weights.norm_ffn),
                              static_cast<__half*>(d_x_norm_),
                              1,
                              hidden_,
                              kRmsNormEps,
                              compute_stream_);
    });

    const int expert_idx = select_expert_cpu(layer);
    const auto& dev_sh = layer_device_shared_[static_cast<std::size_t>(layer)];
    const auto& host   = layer_host_moe_[static_cast<std::size_t>(layer)];

    // Non-blocking prefetch of the NEXT layer's shared expert pages.
    // The shared expert is always active, so we know exactly what to prefetch.
    // This runs concurrently with the current layer's expert H2D transfer (~225 ms),
    // giving the OS time to load the next shared expert into RAM before we need it.
    if (layer + 1 < n_layers_) {
      const int next = layer + 1;
      if (layer_device_shared_[static_cast<std::size_t>(next)].sh_gate == nullptr) {
        const auto& nh = layer_host_moe_[static_cast<std::size_t>(next)];
        const std::size_t sh_elems =
            static_cast<std::size_t>(inter_shared_) *
            static_cast<std::size_t>(hidden_);
        prefetch_range(nh.sh_gate, sh_elems * sizeof(std::uint16_t));
        prefetch_range(nh.sh_up,   sh_elems * sizeof(std::uint16_t));
        prefetch_range(nh.sh_down, sh_elems * sizeof(std::uint16_t));
        prefetch_range(nh.router,
                       static_cast<std::size_t>(n_experts_) *
                           static_cast<std::size_t>(hidden_) *
                           sizeof(std::uint16_t));
      }
    }

    // Load routed expert weights to GPU.
    if (options_.profile_decode_phases) {
      last_benchmark_stats_.transfer_ms +=
          measure_stream_ms(compute_stream_, [&] {
            load_layer_moe_weights_to_device(layer, expert_idx);
          });
    } else {
      load_layer_moe_weights_to_device(layer, expert_idx);
    }

    // MLP: expert FFN then shared expert FFN, results accumulated into d_x_.
    auto run_mlp = [&] {
      // Expert FFN (routed expert, weights already on GPU).
      transposed_projection_half(d_expert_gate_up_w_, d_x_norm_, d_ff13_,
                                 hidden_, inter_full_);
      kernels::launch_silu_mul(static_cast<const __half*>(d_ff13_),
                               static_cast<const __half*>(d_ff13_) + inter_expert_,
                               static_cast<__half*>(d_ff_inter_),
                               inter_expert_, compute_stream_);
      transposed_projection_half(d_expert_down_w_, d_ff_inter_, d_tmp_hidden_,
                                 inter_expert_, hidden_);

      // Shared expert FFN.
      if (dev_sh.sh_gate != nullptr) {
        // GPU-resident path: pure compute, no disk I/O.
        rowmajor_projection_half(dev_sh.sh_gate, d_x_norm_, d_shared_gate_out_,
                                 inter_shared_, hidden_);
        rowmajor_projection_half(dev_sh.sh_up, d_x_norm_, d_shared_up_out_,
                                 inter_shared_, hidden_);
        kernels::launch_silu_mul(static_cast<const __half*>(d_shared_gate_out_),
                                 static_cast<const __half*>(d_shared_up_out_),
                                 static_cast<__half*>(d_ff_inter_),
                                 inter_shared_, compute_stream_);
        rowmajor_projection_half(dev_sh.sh_down, d_ff_inter_, d_att_,
                                 hidden_, inter_shared_);
      } else {
        // Streaming path: pages were prefetched during the expert H2D transfer.
        copy_bf16_tensor_to_fp16_device(
            host.sh_gate, d_streamed_rowmajor_w_,
            static_cast<std::size_t>(inter_shared_) *
                static_cast<std::size_t>(hidden_));
        rowmajor_projection_half(d_streamed_rowmajor_w_, d_x_norm_,
                                 d_shared_gate_out_, inter_shared_, hidden_);
        copy_bf16_tensor_to_fp16_device(
            host.sh_up, d_streamed_rowmajor_w_,
            static_cast<std::size_t>(inter_shared_) *
                static_cast<std::size_t>(hidden_));
        rowmajor_projection_half(d_streamed_rowmajor_w_, d_x_norm_,
                                 d_shared_up_out_, inter_shared_, hidden_);
        kernels::launch_silu_mul(static_cast<const __half*>(d_shared_gate_out_),
                                 static_cast<const __half*>(d_shared_up_out_),
                                 static_cast<__half*>(d_ff_inter_),
                                 inter_shared_, compute_stream_);
        copy_bf16_tensor_to_fp16_device(
            host.sh_down, d_streamed_rowmajor_w_,
            static_cast<std::size_t>(hidden_) *
                static_cast<std::size_t>(inter_shared_));
        rowmajor_projection_half(d_streamed_rowmajor_w_, d_ff_inter_, d_att_,
                                 hidden_, inter_shared_);
      }

      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  static_cast<const __half*>(d_tmp_hidden_),
                                  hidden_, compute_stream_);
      kernels::launch_add_inplace(static_cast<__half*>(d_x_),
                                  static_cast<const __half*>(d_att_),
                                  hidden_, compute_stream_);
    };
    time_or_launch(&last_benchmark_stats_.decode_mlp_ms, run_mlp);
  }

  if (!compute_logits && out_logits == nullptr && out_argmax == nullptr) {
    return;
  }

  time_or_launch(&last_benchmark_stats_.decode_lm_head_ms, [&] {
    kernels::launch_rmsnorm(static_cast<const __half*>(d_x_),
                            static_cast<const __half*>(d_norm_out_),
                            static_cast<__half*>(d_x_norm_),
                            1,
                            hidden_,
                            kRmsNormEps,
                            compute_stream_);
    rowmajor_projection_float(d_lm_head_, d_x_norm_, d_logits_, vocab_size_, hidden_);
    if (out_argmax) {
      kernels::launch_argmax_float(static_cast<const float*>(d_logits_),
                                   vocab_size_,
                                   d_argmax_,
                                   compute_stream_);
    }
  });

  if (out_argmax) {
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    CUDA_CHECK(cudaMemcpy(out_argmax, d_argmax_, sizeof(int), cudaMemcpyDeviceToHost));
  }
  if (out_logits) {
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    CUDA_CHECK(cudaMemcpy(out_logits->data(),
                          d_logits_,
                          static_cast<std::size_t>(vocab_size_) * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
}

void Llama4CudaEngine::forward_token_logits(int token,
                                            int position,
                                            std::vector<float>* out_logits,
                                            int* out_argmax) {
  forward_token(token, position, true, out_logits, out_argmax);
}

int Llama4CudaEngine::sample_next_token(float temperature,
                                        const std::vector<int>& history) {
  const bool greedy_fast_path =
      temperature <= 0.0f &&
      options_.repetition_penalty <= 1.0f &&
      options_.no_repeat_ngram_size <= 1;
  if (greedy_fast_path) {
    kernels::launch_argmax_float(static_cast<const float*>(d_logits_),
                                 vocab_size_,
                                 d_argmax_,
                                 compute_stream_);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    int next = 0;
    CUDA_CHECK(cudaMemcpy(&next, d_argmax_, sizeof(int), cudaMemcpyDeviceToHost));
    return next;
  }

  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  CUDA_CHECK(cudaMemcpy(h_logits_.data(),
                        d_logits_,
                        static_cast<std::size_t>(vocab_size_) * sizeof(float),
                        cudaMemcpyDeviceToHost));
  return sample_from_logits(h_logits_,
                            temperature,
                            options_.top_k,
                            options_.top_p,
                            options_.repetition_penalty,
                            options_.no_repeat_ngram_size,
                            history);
}


std::vector<int> Llama4CudaEngine::generate(const std::vector<int>& prompt_tokens,
                                            int max_new_tokens,
                                            float temperature) {
  return generate_stream(prompt_tokens,
                         max_new_tokens,
                         temperature,
                         [](int) { return true; });
}

std::vector<int> Llama4CudaEngine::generate_stream(
    const std::vector<int>& prompt_tokens,
    int max_new_tokens,
    float temperature,
    const std::function<bool(int)>& on_token) {
  if (max_new_tokens < 0) {
    LLAMA_ENGINE_THROW("max_new_tokens must be >= 0");
  }
  if (static_cast<int>(prompt_tokens.size()) > max_ctx_) {
    LLAMA_ENGINE_THROW("prompt length exceeds max context");
  }

  reset_kv_cache();
  last_benchmark_stats_ = {};
  last_benchmark_stats_.prompt_tokens = static_cast<int>(prompt_tokens.size());

  std::vector<int> out = prompt_tokens;
  out.reserve(prompt_tokens.size() + static_cast<std::size_t>(max_new_tokens));

  std::vector<int> history = prompt_tokens;
  int current = bos_id_;
  int pos = 0;

  const auto prefill_start = std::chrono::steady_clock::now();
  if (prompt_tokens.empty()) {
    current = bos_id_;
    pos = 0;
    history.push_back(bos_id_);
    forward_token(current, pos, true, nullptr, nullptr);
  } else {
    for (int i = 0; i + 1 < static_cast<int>(prompt_tokens.size()); ++i) {
      forward_token(prompt_tokens[static_cast<std::size_t>(i)],
                    i,
                    false,
                    nullptr,
                    nullptr);
    }
    current = prompt_tokens.back();
    pos = static_cast<int>(prompt_tokens.size()) - 1;
    forward_token(current, pos, true, nullptr, nullptr);
  }
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  const auto prefill_end = std::chrono::steady_clock::now();
  last_benchmark_stats_.prefill_ms =
      std::chrono::duration<double, std::milli>(prefill_end - prefill_start)
          .count();

  const auto decode_start = std::chrono::steady_clock::now();
  for (int step = 0; step < max_new_tokens; ++step) {
    const int next = sample_next_token(temperature, history);
    history.push_back(next);
    out.push_back(next);
    if (on_token && !on_token(next)) {
      break;
    }
    if (next == eos_id_) {
      break;
    }
    if (step + 1 >= max_new_tokens) {
      break;
    }
    if (pos + 1 >= max_ctx_) {
      break;
    }

    ++pos;
    current = next;
    forward_token(current, pos, true, nullptr, nullptr);
  }
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  const auto decode_end = std::chrono::steady_clock::now();
  last_benchmark_stats_.decode_ms =
      std::chrono::duration<double, std::milli>(decode_end - decode_start)
          .count();
  last_benchmark_stats_.generated_tokens =
      static_cast<int>((out.size() > prompt_tokens.size())
                           ? (out.size() - prompt_tokens.size())
                           : 0);

  return out;
}

std::vector<std::pair<int, float>> Llama4CudaEngine::inspect_next_logits(
    const std::vector<int>& prompt_tokens,
    int top_k) {
  if (top_k <= 0) {
    return {};
  }
  if (static_cast<int>(prompt_tokens.size()) > max_ctx_) {
    LLAMA_ENGINE_THROW("prompt length exceeds max context");
  }

  reset_kv_cache();
  if (prompt_tokens.empty()) {
    forward_token_logits(bos_id_, 0, nullptr, nullptr);
  } else {
    for (int i = 0; i + 1 < static_cast<int>(prompt_tokens.size()); ++i) {
      forward_token(prompt_tokens[static_cast<std::size_t>(i)],
                    i,
                    false,
                    nullptr,
                    nullptr);
    }
    forward_token_logits(prompt_tokens.back(),
                         static_cast<int>(prompt_tokens.size()) - 1,
                         nullptr,
                         nullptr);
  }

  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  CUDA_CHECK(cudaMemcpy(h_logits_.data(),
                        d_logits_,
                        static_cast<std::size_t>(vocab_size_) * sizeof(float),
                        cudaMemcpyDeviceToHost));

  std::vector<int> order(static_cast<std::size_t>(vocab_size_));
  std::iota(order.begin(), order.end(), 0);
  const int k = std::min(top_k, vocab_size_);
  std::partial_sort(order.begin(),
                    order.begin() + k,
                    order.end(),
                    [&](int a, int b) {
                      return h_logits_[static_cast<std::size_t>(a)] >
                             h_logits_[static_cast<std::size_t>(b)];
                    });

  std::vector<std::pair<int, float>> out;
  out.reserve(static_cast<std::size_t>(k));
  for (int i = 0; i < k; ++i) {
    const int id = order[static_cast<std::size_t>(i)];
    out.emplace_back(id, h_logits_[static_cast<std::size_t>(id)]);
  }
  return out;
}

}  // namespace engine
