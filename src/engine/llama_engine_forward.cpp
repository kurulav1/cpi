#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <utility>
#include <vector>

#include "engine/llama_engine.hpp"
#include "llama_engine_internal.hpp"
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

void LlamaEngine::forward_token_logits(int token, int position, std::vector<float>* out_logits,
                                       int* out_argmax) {
  forward_token(token, position, true, out_logits, out_argmax);
}

void LlamaEngine::forward_token(int token, int position, bool compute_logits,
                                std::vector<float>* out_logits, int* out_argmax) {
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

  run_profiled(last_benchmark_stats_.decode_rmsnorm_ms,
               [&] { launch_norm(d_x_, d_norm_out_, d_norm_out_bias_, d_x_norm_, 1, hidden); });

  run_profiled(last_benchmark_stats_.decode_lm_head_ms, [&] {
    project_lm_head_logits(static_cast<const __half*>(d_x_norm_), static_cast<float*>(d_logits_));
    if (d_lm_head_bias_) {
      kernels::launch_add_bias_inplace_float_from_half(static_cast<float*>(d_logits_),
                                                       static_cast<const __half*>(d_lm_head_bias_),
                                                       cfg.vocab_size, compute_stream_);
    }
  });

  if (out_logits) {
    out_logits->resize(static_cast<std::size_t>(cfg.vocab_size));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    CUDA_CHECK(cudaMemcpy(out_logits->data(), d_logits_, out_logits->size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
  } else if (out_argmax) {
    kernels::launch_argmax_float(static_cast<const float*>(d_logits_), cfg.vocab_size, d_argmax_,
                                 compute_stream_, d_argmax_part_val_, d_argmax_part_idx_,
                                 argmax_parts_);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    CUDA_CHECK(cudaMemcpy(out_argmax, d_argmax_, sizeof(int), cudaMemcpyDeviceToHost));
  }
}

// ---------------------------------------------------------------------------------------
// Tensor-core prefill attention.
//
// Q.K^T and P.V are ordinary GEMMs; routing them through cuBLAS runs them on the tensor cores,
// where attention_prefill_kernel_tiled uses only the scalar FMA pipe.
//
// Done in chunks of query rows so the score matrix stays bounded -- a full [heads][seq][seq]
// matrix is tens of GB at long context, whereas chunked it is heads x chunk x keys.
//
// GQA requires cublasGemmBatchedEx (pointer arrays) rather than the strided-batched call: the
// K/V pointer for query head h is (h / group) * head_dim, which is not a constant stride in h,
// so a single strideA cannot express it.
// ---------------------------------------------------------------------------------------

// Eligibility and scratch allocation, decided once per prefill chunk.
//
// Kept separate from the attention call because the caller must know the answer before the
// layer loop: it skips the Q/K/V split copies when the tensor-core path is available, so a
// mid-loop refusal would leave no contiguous Q to fall back to.
bool LlamaEngine::prefill_tc_prepare(int rows, int base_pos, int num_heads, int num_kv_heads,
                                     int head_dim) {
  static const bool legacy = [] {
    const char* e = std::getenv("CPI_LEGACY_PREFILL_ATTN");
    return e && *e == '1';
  }();
  if (legacy || rows <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
      (num_heads % num_kv_heads) != 0) {
    return false;
  }
  constexpr int kPrepChunk = 256;
  const int prep_keys = base_pos + rows;
  const std::size_t need = static_cast<std::size_t>(num_heads) * kPrepChunk *
                           static_cast<std::size_t>(prep_keys) * sizeof(__half);
  if (need > (std::size_t{1} << 30)) {
    return false;
  }
  if (need > attn_scores_bytes_) {
    if (d_attn_scores_) {
      cudaFree(d_attn_scores_);
      d_attn_scores_ = nullptr;
    }
    if (cudaMalloc(&d_attn_scores_, need) != cudaSuccess) {
      d_attn_scores_ = nullptr;
      attn_scores_bytes_ = 0;
      return false;
    }
    attn_scores_bytes_ = need;
  }
  const std::size_t ptrs_needed = 6 * static_cast<std::size_t>(num_heads);
  if (ptrs_needed > gemm_ptrs_capacity_) {
    if (d_gemm_ptrs_) {
      cudaFree(d_gemm_ptrs_);
      d_gemm_ptrs_ = nullptr;
    }
    gemm_ptrs_capacity_ = 0;
    if (cudaMalloc(&d_gemm_ptrs_, sizeof(void*) * ptrs_needed) != cudaSuccess) {
      d_gemm_ptrs_ = nullptr;
      return false;
    }
    gemm_ptrs_capacity_ = ptrs_needed;
  }
  return true;
}

bool LlamaEngine::prefill_attention_tensorcore(const void* q, const void* k_layer,
                                               const void* v_layer, void* out, int rows,
                                               int base_pos, int num_heads, int num_kv_heads,
                                               int head_dim, int q_stride) {
  if (!prefill_tc_prepare(rows, base_pos, num_heads, num_kv_heads, head_dim)) {
    return false;
  }

  // 256, not 128: halves the number of chunk iterations (each carries a pointer-array copy
  // and three launches) and gives the GEMMs twice the n dimension to work with.
  constexpr int kChunk = 256;
  const int keys = base_pos + rows;  // this chunk's KV is already stored; the mask does the rest
  const int group = num_heads / num_kv_heads;
  // q_stride comes from the caller: num_heads*head_dim for a split Q buffer, or the full
  // fused QKV row stride when Q is read in place.
  const int kv_stride = num_kv_heads * head_dim;
  const int out_stride = num_heads * head_dim;

  const __half* qh = static_cast<const __half*>(q);
  const __half* kh = static_cast<const __half*>(k_layer);
  const __half* vh = static_cast<const __half*>(v_layer);
  __half* oh = static_cast<__half*>(out);
  __half* sh = static_cast<__half*>(d_attn_scores_);

  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const float zero = 0.0f;
  const float one = 1.0f;

  const std::size_t ptrs_needed = 6 * static_cast<std::size_t>(num_heads);
  if (ptrs_needed > gemm_ptrs_capacity_) {
    if (d_gemm_ptrs_) {
      cudaFree(d_gemm_ptrs_);
      d_gemm_ptrs_ = nullptr;
    }
    gemm_ptrs_capacity_ = 0;
    if (cudaMalloc(&d_gemm_ptrs_, sizeof(void*) * ptrs_needed) != cudaSuccess) {
      d_gemm_ptrs_ = nullptr;
      return false;
    }
    gemm_ptrs_capacity_ = ptrs_needed;
  }

  for (int c0 = 0; c0 < rows; c0 += kChunk) {
    const int chunk = std::min(kChunk, rows - c0);

    // Pointer arrays are built ON DEVICE, on this stream. Staging them from the host into a
    // reused pinned buffer was a race -- this function runs once per LAYER, so the next layer
    // overwrote the buffer while the previous async copy was still in flight, and the GEMMs
    // could read the wrong layer's K/V. It gave different logits on every run.
    kernels::launch_build_attention_ptrs(kh, vh, qh, sh, oh, d_gemm_ptrs_, num_heads, group,
                                         head_dim, kChunk, keys, q_stride, out_stride, c0,
                                         compute_stream_);

    const void* const* A1 = const_cast<const void* const*>(d_gemm_ptrs_);
    const void* const* B1 = const_cast<const void* const*>(d_gemm_ptrs_ + num_heads);
    void* const* C1 = d_gemm_ptrs_ + 2 * num_heads;
    const void* const* A2 = const_cast<const void* const*>(d_gemm_ptrs_ + 3 * num_heads);
    const void* const* B2 = const_cast<const void* const*>(d_gemm_ptrs_ + 4 * num_heads);
    void* const* C2 = d_gemm_ptrs_ + 5 * num_heads;

    // S(keys x chunk, col-major, ld=keys) = alpha * K^T(keys x D) * Q(D x chunk)
    //
    // K is row-major [keys, D] with row stride kv_stride, which cuBLAS reads column-major as
    // [D, keys] with ld=kv_stride -- so OP_T gives [keys, D]. Q is row-major [chunk, D] with row
    // stride q_stride, read column-major as [D, chunk] -- OP_N. The column-major result
    // [keys, chunk] with ld=keys IS the row-major [chunk, keys] the softmax wants.
    // 1/sqrt(head_dim) rides in as alpha, so no separate scaling pass.
    CUBLAS_CHECK(cublasGemmBatchedEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, keys, chunk, head_dim,
                                     &scale, A1, CUDA_R_16F, kv_stride, B1, CUDA_R_16F, q_stride,
                                     &zero, C1, CUDA_R_16F, keys, num_heads, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    kernels::launch_softmax_causal_rows(sh, num_heads, kChunk, chunk, keys, base_pos + c0,
                                        compute_stream_);

    // O(D x chunk, col-major, ld=out_stride) = V(D x keys) * P(keys x chunk)
    // V is row-major [keys, D] -> column-major [D, keys], OP_N. P is the row-major
    // [chunk, keys] we just wrote -> column-major [keys, chunk] with ld=keys, OP_N.
    CUBLAS_CHECK(cublasGemmBatchedEx(cublas_, CUBLAS_OP_N, CUBLAS_OP_N, head_dim, chunk, keys, &one,
                                     A2, CUDA_R_16F, kv_stride, B2, CUDA_R_16F, keys, &zero, C2,
                                     CUDA_R_16F, out_stride, num_heads, CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT_TENSOR_OP));
  }
  return true;
}

void LlamaEngine::project_lm_head_logits(const __half* x_norm, float* logits) {
  const auto& cfg = weights_.config();
  if (lm_head_int8_ && d_lm_head_i8_ != nullptr) {
    kernels::launch_weight_only_int8_gemv_f32(d_lm_head_i8_, d_lm_head_i8_scales_, x_norm, logits,
                                              cfg.vocab_size, cfg.hidden_size, compute_stream_);
  } else {
    resident_projection_float(d_lm_head_, x_norm, logits, cfg.vocab_size, cfg.hidden_size,
                              resident_lm_head_warps_, resident_lm_head_tile_pairs_,
                              resident_lm_head_rows_per_warp_);
  }
}

}  // namespace engine
