#include "engine/llama_engine.hpp"
#include "llama_engine_internal.hpp"

#include <algorithm>
#include <utility>

#include <cuda_fp16.h>

#include "common.hpp"
#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"

namespace engine {
namespace {

template <typename Launch>
double timed_cuda_launch_ms(cudaStream_t stream, int warmup, int iters, Launch&& launch) {
  const int safe_warmup = std::max(0, warmup);
  const int safe_iters = std::max(1, iters);
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  for (int i = 0; i < safe_warmup; ++i) {
    launch();
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaEventRecord(start, stream));
  for (int i = 0; i < safe_iters; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(stop, stream));
  CUDA_CHECK(cudaEventSynchronize(stop));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));
  return static_cast<double>(ms);
}

}  // namespace

void LlamaEngine::forward_token_logits(int token, int position, std::vector<float>* out_logits, int* out_argmax) {
  forward_token(token, position, true, out_logits, out_argmax);
}

void LlamaEngine::forward_token(int token,
                                int position,
                                bool compute_logits,
                                std::vector<float>* out_logits,
                                int* out_argmax) {
  const auto& cfg = weights_.config();
  const int hidden = cfg.hidden_size;
  const bool resident_fast_path = cached_layer_count_ == cfg.num_layers && !options_.paged_kv_cache;
  const bool phase_profile = options_.profile_decode_phases && resident_fast_path;
  cublasLtHandle_t matmul_lt = cublas_lt_;

  forward_decode_layers(token, position);

  const auto run_profiled = [&](double& acc, const auto& fn) {
    if (phase_profile) {
      acc += timed_cuda_launch_ms(compute_stream_, /*warmup=*/0, /*iters=*/1, fn);
    } else {
      fn();
    }
  };

  if (!compute_logits) {
    return;
  }

  run_profiled(last_benchmark_stats_.decode_rmsnorm_ms, [&] {
    launch_norm(d_x_, d_norm_out_, d_norm_out_bias_, d_x_norm_, 1, hidden);
  });

  run_profiled(last_benchmark_stats_.decode_lm_head_ms, [&] {
    resident_projection_float(
        d_lm_head_,
        d_x_norm_,
        d_logits_,
        cfg.vocab_size,
        hidden,
        resident_lm_head_warps_,
        resident_lm_head_tile_pairs_,
        resident_lm_head_rows_per_warp_);
    if (d_lm_head_bias_) {
      kernels::launch_add_bias_inplace_float_from_half(static_cast<float*>(d_logits_),
                                                       static_cast<const __half*>(d_lm_head_bias_),
                                                       cfg.vocab_size,
                                                       compute_stream_);
    }
  });

  if (out_logits) {
    out_logits->resize(static_cast<std::size_t>(cfg.vocab_size));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    CUDA_CHECK(cudaMemcpy(out_logits->data(), d_logits_, out_logits->size() * sizeof(float), cudaMemcpyDeviceToHost));
  } else if (out_argmax) {
    kernels::launch_argmax_float(static_cast<const float*>(d_logits_), cfg.vocab_size, d_argmax_, compute_stream_);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    CUDA_CHECK(cudaMemcpy(out_argmax, d_argmax_, sizeof(int), cudaMemcpyDeviceToHost));
  }
}



}  // namespace engine
