#include "llama_engine_internal.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>
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

template <typename Launch>
double benchmark_cuda_launch(cudaStream_t stream, int warmup, int iters, Launch&& launch) {
  const int safe_iters = std::max(1, iters);
  const double total_ms =
      timed_cuda_launch_ms(stream, warmup, safe_iters, std::forward<Launch>(launch));
  return total_ms / static_cast<double>(safe_iters);
}

}  // namespace
void LlamaEngine::resident_projection_half(const void* w,
                                           const void* x,
                                           void* y,
                                           int out_features,
                                           int in_features,
                                           int warps_per_block,
                                           int tile_pairs,
                                           int rows_per_warp) {
  kernels::launch_rowmajor_half_gemv_f16(static_cast<const __half*>(w),
                                         static_cast<const __half*>(x),
                                         static_cast<__half*>(y),
                                         out_features,
                                         in_features,
                                         compute_stream_,
                                         warps_per_block,
                                         tile_pairs,
                                         rows_per_warp);
}

void LlamaEngine::resident_projection_float(const void* w,
                                            const void* x,
                                            void* y,
                                            int out_features,
                                            int in_features,
                                            int warps_per_block,
                                            int tile_pairs,
                                            int rows_per_warp) {
  kernels::launch_rowmajor_half_gemv_f32(static_cast<const __half*>(w),
                                         static_cast<const __half*>(x),
                                         static_cast<float*>(y),
                                         out_features,
                                         in_features,
                                         compute_stream_,
                                         warps_per_block,
                                         tile_pairs,
                                         rows_per_warp);
}

void LlamaEngine::resident_int8_mlp_w13(const LayerDeviceInt8Weights& lw_i8, int inter, int hidden) {
  if (lw_i8.mlp_int4) {
    kernels::launch_weight_only_int4_matvec_dual_dp4a(lw_i8.w1,
                                                      lw_i8.s_w1,
                                                      lw_i8.w3,
                                                      lw_i8.s_w3,
                                                      d_prefill_i8_,
                                                      d_prefill_i8_scales_,
                                                      static_cast<__half*>(d_ff1_),
                                                      static_cast<__half*>(d_ff2_),
                                                      inter,
                                                      hidden,
                                                      compute_stream_,
                                                      resident_mlp_w13_warps_,
                                                      resident_mlp_w13_tile_packed4_,
                                                      resident_mlp_w13_warps_per_row_);
    return;
  }
  kernels::launch_weight_only_int8_matvec_dual_dp4a(lw_i8.w1,
                                                    lw_i8.s_w1,
                                                    lw_i8.w3,
                                                    lw_i8.s_w3,
                                                    d_prefill_i8_,
                                                    d_prefill_i8_scales_,
                                                    static_cast<__half*>(d_ff1_),
                                                    static_cast<__half*>(d_ff2_),
                                                    inter,
                                                    hidden,
                                                    compute_stream_,
                                                    resident_mlp_w13_warps_,
                                                    resident_mlp_w13_tile_packed4_,
                                                    resident_mlp_w13_warps_per_row_);
}

void LlamaEngine::resident_int8_mlp_w2(const LayerDeviceInt8Weights& lw_i8, int hidden, int inter) {
  if (lw_i8.mlp_int4) {
    kernels::launch_weight_only_int4_matvec_dp4a(lw_i8.w2,
                                                 lw_i8.s_w2,
                                                 d_prefill_i8_,
                                                 d_prefill_i8_scales_,
                                                 static_cast<__half*>(d_ff3_),
                                                 hidden,
                                                 inter,
                                                 compute_stream_,
                                                 resident_mlp_w2_warps_,
                                                 resident_mlp_w2_tile_packed4_,
                                                 resident_mlp_w2_warps_per_row_);
    return;
  }
  kernels::launch_weight_only_int8_matvec_dp4a(lw_i8.w2,
                                               lw_i8.s_w2,
                                               d_prefill_i8_,
                                               d_prefill_i8_scales_,
                                               static_cast<__half*>(d_ff3_),
                                               hidden,
                                               inter,
                                               compute_stream_,
                                               resident_mlp_w2_warps_,
                                               resident_mlp_w2_tile_packed4_,
                                               resident_mlp_w2_warps_per_row_);
}

void LlamaEngine::tune_resident_projection_backends() {
  resident_custom_qkv_ = false;
  resident_custom_wo_ = false;
  resident_custom_lm_head_ = false;
  resident_qkv_warps_ = 8;
  resident_qkv_tile_pairs_ = 128;
  resident_qkv_rows_per_warp_ = 1;
  resident_wo_warps_ = 4;
  resident_wo_tile_pairs_ = 128;
  resident_wo_rows_per_warp_ = 1;
  resident_lm_head_warps_ = 8;
  resident_lm_head_tile_pairs_ = 128;
  resident_lm_head_rows_per_warp_ = 1;
  resident_mlp_w13_warps_ = 8;
  resident_mlp_w13_tile_packed4_ = 256;
  resident_mlp_w13_warps_per_row_ = 2;
  resident_mlp_w2_warps_ = 8;
  resident_mlp_w2_tile_packed4_ = 256;
  resident_mlp_w2_warps_per_row_ = 2;
  return;

  const auto& cfg = weights_.config();
  if (cached_layer_count_ != cfg.num_layers || options_.paged_kv_cache || layer_cache_.empty()) {
    return;
  }

  const int hidden = cfg.hidden_size;
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : hidden;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (cfg.hidden_size / cfg.num_heads);
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const auto& lw = layer_cache_.front();
  const int warmup = std::max(0, env_int_or_default("LLAMA_INFER_TUNE_WARMUP", 3));
  const int iters = std::max(1, env_int_or_default("LLAMA_INFER_TUNE_ITERS", 20));
  struct ProjectionConfig {
    int warps = 4;
    int tile = 128;
    int warps_per_row = 1;
  };
  const std::vector<ProjectionConfig> qkv_configs = {{8, 128, 1}, {8, 128, 2}, {8, 256, 1}, {8, 256, 2}, {16, 128, 1}, {16, 128, 2}, {16, 256, 1}, {16, 256, 2}, {8, 512, 1}, {8, 512, 2}, {16, 512, 1}, {16, 512, 2}};
  const std::vector<ProjectionConfig> wo_configs = {{4, 128, 1}, {4, 128, 2}, {4, 256, 1}, {4, 256, 2}, {8, 128, 1}, {8, 128, 2}, {8, 256, 1}, {8, 256, 2}, {8, 512, 1}, {8, 512, 2}};
  const std::vector<ProjectionConfig> lm_configs = {{8, 128, 1}, {8, 128, 2}, {8, 256, 1}, {16, 128, 1}, {8, 512, 1}, {8, 512, 2}, {16, 512, 1}};
  const std::vector<ProjectionConfig> mlp_w13_configs = {
      {8, 256, 2}, {8, 256, 1}, {16, 256, 2}, {16, 256, 1}, {8, 512, 2}, {8, 512, 1}, {8, 128, 2}, {8, 128, 1}};
  const std::vector<ProjectionConfig> mlp_w2_configs = {
      {8, 256, 2}, {8, 256, 1}, {4, 256, 1}, {8, 128, 2}, {8, 128, 1}, {4, 128, 1}, {16, 128, 2}, {16, 128, 1}};

  // Skip fp16 QKV/wo tuning when they've been freed (int8 proj always active for GPU-cached layers).
  double qkv_cublas_ms = 0.0;
  double qkv_custom_ms = 0.0;
  if (!cached_int8_proj_enabled_ && lw.wqkv) {
    qkv_cublas_ms = benchmark_cuda_launch(compute_stream_, warmup, iters, [&] {
      detail::dispatch_linear_rowmajor_weight(cublas_,
                             cublas_lt_,
                             &lt_plan_cache_,
                             lt_workspace_,
                             lt_workspace_bytes_,
                             compute_stream_,
                             lw.wqkv,
                             d_x_norm_,
                             d_q_,
                             q_hidden + 2 * kv_hidden,
                             hidden,
                             1,
                             CUDA_R_16F);
    });
    qkv_custom_ms = qkv_cublas_ms;
    for (const auto& cfg_candidate : qkv_configs) {
      const double ms = benchmark_cuda_launch(compute_stream_, warmup, iters, [&] {
        resident_projection_half(lw.wqkv,
                                 d_x_norm_,
                                 d_q_,
                                 q_hidden + 2 * kv_hidden,
                                 hidden,
                                 cfg_candidate.warps,
                                 cfg_candidate.tile,
                                 cfg_candidate.warps_per_row);
      });
      if (ms < qkv_custom_ms * 0.99) {
        qkv_custom_ms = ms;
        resident_qkv_warps_ = cfg_candidate.warps;
        resident_qkv_tile_pairs_ = cfg_candidate.tile;
        resident_qkv_rows_per_warp_ = cfg_candidate.warps_per_row;
      } else if (qkv_custom_ms == qkv_cublas_ms) {
        resident_qkv_warps_ = cfg_candidate.warps;
        resident_qkv_tile_pairs_ = cfg_candidate.tile;
        resident_qkv_rows_per_warp_ = cfg_candidate.warps_per_row;
      }
    }
    // Keep cuBLASLt unless the custom QKV path wins by a strong margin.
    resident_custom_qkv_ = qkv_custom_ms < qkv_cublas_ms * 0.95;
  }

  double wo_cublas_ms = 0.0;
  double wo_custom_ms = 0.0;
  if (!cached_int8_proj_enabled_ && lw.wo) {
    wo_cublas_ms = benchmark_cuda_launch(compute_stream_, warmup, iters, [&] {
    detail::dispatch_linear_rowmajor_weight(cublas_,
                           cublas_lt_,
                           &lt_plan_cache_,
                           lt_workspace_,
                           lt_workspace_bytes_,
                           compute_stream_,
                           lw.wo,
                           d_att_,
                           d_ff3_,
                           hidden,
                           hidden,
                           1,
                           CUDA_R_16F);
  });
    wo_custom_ms = wo_cublas_ms;
    for (const auto& cfg_candidate : wo_configs) {
      const double ms = benchmark_cuda_launch(compute_stream_, warmup, iters, [&] {
        resident_projection_half(lw.wo, d_att_, d_ff3_, hidden, q_hidden, cfg_candidate.warps, cfg_candidate.tile, cfg_candidate.warps_per_row);
      });
      if (ms < wo_custom_ms * 0.99) {
        wo_custom_ms = ms;
        resident_wo_warps_ = cfg_candidate.warps;
        resident_wo_tile_pairs_ = cfg_candidate.tile;
        resident_wo_rows_per_warp_ = cfg_candidate.warps_per_row;
      } else if (wo_custom_ms == wo_cublas_ms) {
        resident_wo_warps_ = cfg_candidate.warps;
        resident_wo_tile_pairs_ = cfg_candidate.tile;
        resident_wo_rows_per_warp_ = cfg_candidate.warps_per_row;
      }
    }
    resident_custom_wo_ = wo_custom_ms < wo_cublas_ms * 0.98;
  }

  const double lm_cublas_ms = benchmark_cuda_launch(compute_stream_, warmup, iters, [&] {
    detail::dispatch_linear_rowmajor_weight(cublas_,
                           cublas_lt_,
                           &lt_plan_cache_,
                           lt_workspace_,
                           lt_workspace_bytes_,
                           compute_stream_,
                           d_lm_head_,
                           d_x_norm_,
                           d_logits_,
                           cfg.vocab_size,
                           hidden,
                           1,
                           CUDA_R_32F);
  });
  double lm_custom_ms = lm_cublas_ms;
  for (const auto& cfg_candidate : lm_configs) {
    const double ms = benchmark_cuda_launch(compute_stream_, warmup, iters, [&] {
      resident_projection_float(
          d_lm_head_, d_x_norm_, d_logits_, cfg.vocab_size, hidden, cfg_candidate.warps, cfg_candidate.tile, cfg_candidate.warps_per_row);
    });
    if (ms < lm_custom_ms * 0.99) {
      lm_custom_ms = ms;
      resident_lm_head_warps_ = cfg_candidate.warps;
      resident_lm_head_tile_pairs_ = cfg_candidate.tile;
      resident_lm_head_rows_per_warp_ = cfg_candidate.warps_per_row;
    } else if (lm_custom_ms == lm_cublas_ms) {
      resident_lm_head_warps_ = cfg_candidate.warps;
      resident_lm_head_tile_pairs_ = cfg_candidate.tile;
      resident_lm_head_rows_per_warp_ = cfg_candidate.warps_per_row;
    }
  }
  // The LM head is especially sensitive to measurement noise because the
  // kernel is small and launch overhead matters. Require a larger margin.
  resident_custom_lm_head_ = lm_custom_ms < lm_cublas_ms * 0.96;

  double mlp_w13_best_ms = 0.0;
  double mlp_w2_best_ms = 0.0;
  bool have_mlp_int8_tuning = cached_int8_mlp_enabled_ && !layer_cache_i8_.empty() && layer_cache_i8_.front().w1 &&
                              layer_cache_i8_.front().w2 && layer_cache_i8_.front().w3;
  if (have_mlp_int8_tuning) {
    const auto& lw_i8 = layer_cache_i8_.front();
    kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_x_norm_),
                                                  d_prefill_i8_,
                                                  d_prefill_i8_scales_,
                                                  1,
                                                  hidden,
                                                  compute_stream_);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    mlp_w13_best_ms = std::numeric_limits<double>::max();
    for (const auto& cfg_candidate : mlp_w13_configs) {
      const double ms = benchmark_cuda_launch(compute_stream_, warmup, iters, [&] {
        kernels::launch_weight_only_int8_matvec_dual_dp4a(lw_i8.w1,
                                                          lw_i8.s_w1,
                                                          lw_i8.w3,
                                                          lw_i8.s_w3,
                                                          d_prefill_i8_,
                                                          d_prefill_i8_scales_,
                                                          static_cast<__half*>(d_ff1_),
                                                          static_cast<__half*>(d_ff2_),
                                                          cfg.intermediate_size,
                                                          hidden,
                                                          compute_stream_,
                                                          cfg_candidate.warps,
                                                          cfg_candidate.tile,
                                                          cfg_candidate.warps_per_row);
      });
      if (ms < mlp_w13_best_ms * 0.97) {
        mlp_w13_best_ms = ms;
        resident_mlp_w13_warps_ = cfg_candidate.warps;
        resident_mlp_w13_tile_packed4_ = cfg_candidate.tile;
        resident_mlp_w13_warps_per_row_ = cfg_candidate.warps_per_row;
      } else if (mlp_w13_best_ms == std::numeric_limits<double>::max()) {
        mlp_w13_best_ms = ms;
        resident_mlp_w13_warps_ = cfg_candidate.warps;
        resident_mlp_w13_tile_packed4_ = cfg_candidate.tile;
        resident_mlp_w13_warps_per_row_ = cfg_candidate.warps_per_row;
      }
    }

    kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_ff2_),
                                                  d_prefill_i8_,
                                                  d_prefill_i8_scales_,
                                                  1,
                                                  cfg.intermediate_size,
                                                  compute_stream_);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    mlp_w2_best_ms = std::numeric_limits<double>::max();
    for (const auto& cfg_candidate : mlp_w2_configs) {
      const double ms = benchmark_cuda_launch(compute_stream_, warmup, iters, [&] {
        kernels::launch_weight_only_int8_matvec_dp4a(lw_i8.w2,
                                                     lw_i8.s_w2,
                                                     d_prefill_i8_,
                                                     d_prefill_i8_scales_,
                                                     static_cast<__half*>(d_ff3_),
                                                     hidden,
                                                     cfg.intermediate_size,
                                                     compute_stream_,
                                                     cfg_candidate.warps,
                                                     cfg_candidate.tile,
                                                     cfg_candidate.warps_per_row);
      });
      if (ms < mlp_w2_best_ms * 0.97) {
        mlp_w2_best_ms = ms;
        resident_mlp_w2_warps_ = cfg_candidate.warps;
        resident_mlp_w2_tile_packed4_ = cfg_candidate.tile;
        resident_mlp_w2_warps_per_row_ = cfg_candidate.warps_per_row;
      } else if (mlp_w2_best_ms == std::numeric_limits<double>::max()) {
        mlp_w2_best_ms = ms;
        resident_mlp_w2_warps_ = cfg_candidate.warps;
        resident_mlp_w2_tile_packed4_ = cfg_candidate.tile;
        resident_mlp_w2_warps_per_row_ = cfg_candidate.warps_per_row;
      }
    }
  }

  // Tune int8 QKV and wo kernels when the projection weights are int8-cached.
  double int8_qkv_best_ms = 0.0;
  double int8_wo_best_ms = 0.0;
  bool have_proj_int8_tuning = cached_int8_proj_enabled_ && !layer_cache_i8_.empty() &&
                               layer_cache_i8_.front().wqkv && layer_cache_i8_.front().wo;
  if (have_proj_int8_tuning) {
    const auto& lw_i8 = layer_cache_i8_.front();
    // Quantize d_x_norm_ (activation) once for QKV benchmark.
    kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_x_norm_),
                                                  static_cast<std::int8_t*>(d_prefill_i8_),
                                                  static_cast<float*>(d_prefill_i8_scales_),
                                                  1,
                                                  hidden,
                                                  compute_stream_);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    int8_qkv_best_ms = std::numeric_limits<double>::max();
    for (const auto& cfg_candidate : mlp_w13_configs) {
      const double ms = benchmark_cuda_launch(compute_stream_, warmup, iters, [&] {
        if (lw_i8.proj_int4) {
          kernels::launch_weight_only_int4_matvec_dp4a(lw_i8.wqkv,
                                                       lw_i8.s_wqkv,
                                                       static_cast<const std::int8_t*>(d_prefill_i8_),
                                                       static_cast<const float*>(d_prefill_i8_scales_),
                                                       static_cast<__half*>(d_q_),
                                                       q_hidden + 2 * kv_hidden,
                                                       hidden,
                                                       compute_stream_,
                                                       cfg_candidate.warps,
                                                       cfg_candidate.tile,
                                                       cfg_candidate.warps_per_row);
        } else {
          kernels::launch_weight_only_int8_matvec_dp4a(lw_i8.wqkv,
                                                       lw_i8.s_wqkv,
                                                       static_cast<const std::int8_t*>(d_prefill_i8_),
                                                       static_cast<const float*>(d_prefill_i8_scales_),
                                                       static_cast<__half*>(d_q_),
                                                       q_hidden + 2 * kv_hidden,
                                                       hidden,
                                                       compute_stream_,
                                                       cfg_candidate.warps,
                                                       cfg_candidate.tile,
                                                       cfg_candidate.warps_per_row);
        }
      });
      if (ms < int8_qkv_best_ms * 0.97) {
        int8_qkv_best_ms = ms;
        resident_int8_qkv_warps_ = cfg_candidate.warps;
        resident_int8_qkv_tile_packed4_ = cfg_candidate.tile;
        resident_int8_qkv_warps_per_row_ = cfg_candidate.warps_per_row;
      } else if (int8_qkv_best_ms == std::numeric_limits<double>::max()) {
        int8_qkv_best_ms = ms;
        resident_int8_qkv_warps_ = cfg_candidate.warps;
        resident_int8_qkv_tile_packed4_ = cfg_candidate.tile;
        resident_int8_qkv_warps_per_row_ = cfg_candidate.warps_per_row;
      }
    }

    // Quantize d_att_ (activation) for wo benchmark.
    kernels::launch_quantize_rowwise_fp16_to_int8(static_cast<const __half*>(d_att_),
                                                  static_cast<std::int8_t*>(d_prefill_i8_),
                                                  static_cast<float*>(d_prefill_i8_scales_),
                                                  1,
                                                  q_hidden,
                                                  compute_stream_);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    int8_wo_best_ms = std::numeric_limits<double>::max();
    for (const auto& cfg_candidate : mlp_w2_configs) {
      const double ms = benchmark_cuda_launch(compute_stream_, warmup, iters, [&] {
        if (lw_i8.proj_int4) {
          kernels::launch_weight_only_int4_matvec_dp4a(lw_i8.wo,
                                                       lw_i8.s_wo,
                                                       static_cast<const std::int8_t*>(d_prefill_i8_),
                                                       static_cast<const float*>(d_prefill_i8_scales_),
                                                       static_cast<__half*>(d_ff3_),
                                                       hidden,
                                                       q_hidden,
                                                       compute_stream_,
                                                       cfg_candidate.warps,
                                                       cfg_candidate.tile,
                                                       cfg_candidate.warps_per_row);
        } else {
          kernels::launch_weight_only_int8_matvec_dp4a(lw_i8.wo,
                                                       lw_i8.s_wo,
                                                       static_cast<const std::int8_t*>(d_prefill_i8_),
                                                       static_cast<const float*>(d_prefill_i8_scales_),
                                                       static_cast<__half*>(d_ff3_),
                                                       hidden,
                                                       q_hidden,
                                                       compute_stream_,
                                                       cfg_candidate.warps,
                                                       cfg_candidate.tile,
                                                       cfg_candidate.warps_per_row);
        }
      });
      if (ms < int8_wo_best_ms * 0.97) {
        int8_wo_best_ms = ms;
        resident_int8_wo_warps_ = cfg_candidate.warps;
        resident_int8_wo_tile_packed4_ = cfg_candidate.tile;
        resident_int8_wo_warps_per_row_ = cfg_candidate.warps_per_row;
      } else if (int8_wo_best_ms == std::numeric_limits<double>::max()) {
        int8_wo_best_ms = ms;
        resident_int8_wo_warps_ = cfg_candidate.warps;
        resident_int8_wo_tile_packed4_ = cfg_candidate.tile;
        resident_int8_wo_warps_per_row_ = cfg_candidate.warps_per_row;
      }
    }
  }

  if (options_.verbose) {
    const bool proj_is_int4 = cached_int8_proj_enabled_ && !layer_cache_i8_.empty() && layer_cache_i8_.front().proj_int4;
    const char* proj_mode = proj_is_int4 ? "int4" : "int8";
    std::cout << std::fixed << std::setprecision(3)
              << "[engine] resident_projection_backend"
              << " qkv=" << (cached_int8_proj_enabled_ ? proj_mode : (resident_custom_qkv_ ? "custom" : "cublaslt"))
              << " wo=" << (cached_int8_proj_enabled_ ? proj_mode : (resident_custom_wo_ ? "custom" : "cublaslt"))
              << " lm_head=" << (resident_custom_lm_head_ ? "custom" : "cublaslt");
    if (!cached_int8_proj_enabled_) {
      std::cout << " qkv_ms=" << qkv_cublas_ms << "/" << qkv_custom_ms
                << " qkv_cfg=" << resident_qkv_warps_ << "x" << resident_qkv_tile_pairs_ << "@"
                << resident_qkv_rows_per_warp_
                << " wo_ms=" << wo_cublas_ms << "/" << wo_custom_ms
                << " wo_cfg=" << resident_wo_warps_ << "x" << resident_wo_tile_pairs_ << "@"
                << resident_wo_rows_per_warp_;
    }
    std::cout << " lm_ms=" << lm_cublas_ms << "/" << lm_custom_ms
              << " lm_cfg=" << resident_lm_head_warps_ << "x" << resident_lm_head_tile_pairs_ << "@"
              << resident_lm_head_rows_per_warp_;
    if (have_mlp_int8_tuning) {
      std::cout << " mlp_w13_ms=" << mlp_w13_best_ms
                << " mlp_w13_cfg=" << resident_mlp_w13_warps_ << "x" << resident_mlp_w13_tile_packed4_
                << "@" << resident_mlp_w13_warps_per_row_
                << " mlp_w2_ms=" << mlp_w2_best_ms
                << " mlp_w2_cfg=" << resident_mlp_w2_warps_ << "x" << resident_mlp_w2_tile_packed4_
                << "@" << resident_mlp_w2_warps_per_row_;
    }
    if (have_proj_int8_tuning) {
      std::cout << " " << proj_mode << "_qkv_ms=" << int8_qkv_best_ms
                << " " << proj_mode << "_qkv_cfg=" << resident_int8_qkv_warps_ << "x" << resident_int8_qkv_tile_packed4_
                << "@" << resident_int8_qkv_warps_per_row_
                << " " << proj_mode << "_wo_ms=" << int8_wo_best_ms
                << " " << proj_mode << "_wo_cfg=" << resident_int8_wo_warps_ << "x" << resident_int8_wo_tile_packed4_
                << "@" << resident_int8_wo_warps_per_row_;
    }
    std::cout << "\n";
  }
}
}  // namespace engine
