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
// Done in chunks of query rows so the score matrix stays bounded; a full [heads][seq][seq]
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

    // Pointer arrays are built on device, on this stream. Staging them from the host into a
    // reused pinned buffer was a race; this function runs once per layer, so the next layer
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
    // [D, keys] with ld=kv_stride; so OP_T gives [keys, D]. Q is row-major [chunk, D] with row
    // stride q_stride, read column-major as [D, chunk]; OP_N. The column-major result
    // [keys, chunk] with ld=keys is the row-major [chunk, keys] the softmax wants.
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

bool LlamaEngine::ensure_q8_scratch(int batch, int cols) {
  const std::size_t xb = static_cast<std::size_t>(batch) * static_cast<std::size_t>(cols);
  const std::size_t mb =
      static_cast<std::size_t>(batch) * static_cast<std::size_t>(cols / 32) * sizeof(float);
  if (xb > q8_x_bytes_) {
    if (d_q8_x_) cudaFree(d_q8_x_);
    d_q8_x_ = nullptr;
    q8_x_bytes_ = 0;
    if (cudaMalloc(&d_q8_x_, xb) != cudaSuccess) return false;
    q8_x_bytes_ = xb;
  }
  if (mb > q8_meta_bytes_) {
    if (d_q8_scale_) cudaFree(d_q8_scale_);
    if (d_q8_sum_) cudaFree(d_q8_sum_);
    d_q8_scale_ = nullptr;
    d_q8_sum_ = nullptr;
    q8_meta_bytes_ = 0;
    if (cudaMalloc(&d_q8_scale_, mb) != cudaSuccess) return false;
    if (cudaMalloc(&d_q8_sum_, mb) != cudaSuccess) return false;
    q8_meta_bytes_ = mb;
  }
  return true;
}

bool LlamaEngine::packed_matmul(const PackedWeight& w, const void* x, void* y, int batch, int ldy,
                                int row0, cudaStream_t stream, bool reuse_x) {
  if (!w.active()) return false;
  // A single row is a matvec. Without this the batched path falls through to
  // expand-and-cuBLAS, which re-expands the whole weight for one token: at B=1
  // that measured 22 ms per token against 5.3 ms for the same model on the
  // single-stream path, because a decode step was moving ~28 GB to multiply by
  // one vector.
  if (batch == 1) {
    ++packed_matmul_calls_;
    kernels::launch_kquant_matvec(static_cast<const std::uint8_t*>(w.data),
                                  static_cast<kernels::KQuantType>(w.kind),
                                  static_cast<const __half*>(x),
                                  static_cast<__half*>(y) + row0, w.rows, w.cols, stream, reuse_x);
    return true;
  }
  // Int8 tensor cores (CPI_KQUANT_MMQ=1). Correct, gated, and still SLOWER than
  // expand-and-cuBLAS: 535 against 738 tok/s at B=32. It keeps the weight packed
  // and does reach the hardware cuBLAS wins on, so the architecture is right --
  // what is missing is tuning. Two known costs: the M tile is a fixed 64 rows,
  // so a batch of 2 computes 62 rows of padding, and the staging/occupancy work
  // llama.cpp carries per-architecture tile configs for is simply absent here.
  // MMQ is on by default now, but only across the batch range where it actually
  // wins. It keeps the weight packed, so it avoids expanding the whole matrix to
  // fp16 every step -- w13 alone is 235 MB of fp16 per layer per step -- but its
  // fixed 64-row M tile means small batches compute mostly padding, and at large
  // batches cuBLAS's tiling pulls ahead again. Measured against expand-and-cuBLAS
  // on an 8B Q4_K_M, batched decode tok/s:
  //     B:      1     2     4     8    16    32    48    64
  //   cuBLAS: 93.1 165.6 267.2 286.7 387.1 597.4 701.0 765.5
  //   MMQ:    89.7 151.1 262.4 387.6 514.9 680.5 680.3 739.0
  //           -4%   -9%   -2%  +35%  +33%  +14%   -3%   -3%
  // so the band is [8, 40]. CPI_KQUANT_MMQ=1 forces it everywhere and =0 disables
  // it, for re-measuring the edges.
  static const int mmq_force = []() {
    const char* e = std::getenv("CPI_KQUANT_MMQ");
    return e == nullptr ? -1 : (e[0] == '1' ? 1 : 0);
  }();
  static const int mmq_lo = []() {
    const char* e = std::getenv("CPI_KQUANT_MMQ_MIN_BATCH");
    return e ? atoi(e) : 8;
  }();
  static const int mmq_hi = []() {
    const char* e = std::getenv("CPI_KQUANT_MMQ_MAX_BATCH");
    return e ? atoi(e) : 40;
  }();
  const bool mmq_on =
      mmq_force == 1 || (mmq_force != 0 && batch >= mmq_lo && batch <= mmq_hi);
  static const bool mma_ok = []() {
    int dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) != cudaSuccess) return false;
    if (cudaGetDeviceProperties(&prop, dev) != cudaSuccess) return false;
    return prop.major >= 8;  // mma.m16n8k32.s8 is sm_80+
  }();
  // kind 0 is Q4_K, kind 2 is Q6_K; both have an MMQ path now, which is what
  // stops ffn_down and wv falling back to expand-and-cuBLAS.
  if (mmq_on && mma_ok && batch >= 2 && batch <= 64 && (w.kind == 0 || w.kind == 2) &&
      w.cols % 256 == 0 &&
      ensure_q8_scratch(batch, w.cols) &&
      kernels::launch_kquant_mmq(static_cast<const std::uint8_t*>(w.data),
                                 static_cast<kernels::KQuantType>(w.kind),
                                 static_cast<const __half*>(x), static_cast<__half*>(y) + row0,
                                 w.rows, w.cols, batch, ldy, d_q8_x_, d_q8_scale_, d_q8_sum_,
                                 stream)) {
    ++packed_matmul_calls_;
    return true;
  }

  // dp4a form, opt-in (CPI_KQUANT_DP4A=1) and currently NOT profitable.
  //
  // The kernel is correct -- kquant_matvec_test gates it to 0.7% against the
  // fp16 reference -- and the integer dots are genuinely cheaper. What sinks it
  // is the plumbing around it: activations are quantized inside this call, so
  // the same d_x_norm_ is re-quantized for qkv and again for w13, and a batched
  // step pays 32 layers x 4 weights of extra launches and traffic. Measured at
  // B=2..64 it runs 77.9/104.0/93.4/109.4/96.6/99.0 against 85.9/113.3/116.7/
  // 128.1/150.2/160.5 without it.
  //
  // Hoisting the quantization to once per distinct activation is what llama.cpp
  // does (src1 is converted once per matmul op, not per tile), and is the next
  // step if this is picked up.
  static const bool dp4a_on = []() {
    const char* e = std::getenv("CPI_KQUANT_DP4A");
    return e != nullptr && e[0] == '1';
  }();
  if (dp4a_on && batch >= 2 && w.kind != 2 && w.cols % 32 == 0 &&
      ensure_q8_scratch(batch, w.cols) &&
      kernels::launch_kquant_matmul_dp4a(
          static_cast<const std::uint8_t*>(w.data), static_cast<kernels::KQuantType>(w.kind),
          static_cast<const __half*>(x), static_cast<__half*>(y) + row0, w.rows, w.cols, batch, ldy,
          d_q8_x_, d_q8_scale_, d_q8_sum_, stream)) {
    ++packed_matmul_calls_;
    return true;
  }
  const bool ok = kernels::launch_kquant_matmul(
      static_cast<const std::uint8_t*>(w.data), static_cast<kernels::KQuantType>(w.kind),
      static_cast<const __half*>(x), static_cast<__half*>(y) + row0, w.rows, w.cols, batch, ldy,
      stream);
  if (ok) {
    ++packed_matmul_calls_;
  } else {
    ++packed_matmul_declined_;
  }
  return ok;
}

// Batched QKV off the packed blocks, in whichever shape this layer has. All or
// nothing: a partial success would leave some row ranges unwritten.
bool LlamaEngine::packed_qkv_matmul(const LayerDeviceWeights& lw, const void* x, void* y, int batch,
                                    int ldy, cudaStream_t stream) {
  if (lw.wqkv_packed.active()) {
    return packed_matmul(lw.wqkv_packed, x, y, batch, ldy, 0, stream);
  }
  if (!lw.wq_packed.active() || !lw.wv_packed.active()) return false;
  const int rows_q = lw.wq_packed.rows;
  const int rows_k = lw.wk_packed.active() ? lw.wk_packed.rows : 0;
  // Probe the first part before committing: the launcher declines batches it is
  // not worth doing, and it must decline all three or none.
  // All three read the same activation, so only the first pays to quantize it.
  if (!packed_matmul(lw.wq_packed, x, y, batch, ldy, 0, stream)) return false;
  if (rows_k > 0 && !packed_matmul(lw.wk_packed, x, y, batch, ldy, rows_q, stream, true)) {
    return false;
  }
  return packed_matmul(lw.wv_packed, x, y, batch, ldy, rows_q + rows_k, stream, true);
}

bool LlamaEngine::packed_qkv_matvec(const LayerDeviceWeights& lw, const void* x_norm, void* qkv,
                                    cudaStream_t stream, bool x_pre_quantized) {
  // reuse says x_norm was already quantized by the preceding projection: all of
  // q, k and v read the same normed activation, and this model cannot fuse them
  // into one packed matrix because wv is Q6_K where wq and wk are Q4_K.
  const auto run = [&](const PackedWeight& w, int row_offset, bool reuse = false) {
    ++packed_matvec_calls_;
    kernels::launch_kquant_matvec(static_cast<const std::uint8_t*>(w.data),
                                  static_cast<kernels::KQuantType>(w.kind),
                                  static_cast<const __half*>(x_norm),
                                  static_cast<__half*>(qkv) + row_offset, w.rows, w.cols, stream,
                                  reuse);
  };
  if (lw.wqkv_packed.active()) {
    run(lw.wqkv_packed, 0, x_pre_quantized);
    return true;
  }
  // Separate buffers write into the same row ranges the fused matrix would have
  // produced, so everything downstream is unchanged. wq_packed may already hold
  // Q and K together, in which case wk_packed is empty and there are two
  // launches rather than three.
  if (lw.wq_packed.active() && lw.wv_packed.active()) {
    run(lw.wq_packed, 0, x_pre_quantized);
    int written = lw.wq_packed.rows;
    if (lw.wk_packed.active()) {
      run(lw.wk_packed, written, true);
      written += lw.wk_packed.rows;
    }
    run(lw.wv_packed, written, true);
    return true;
  }
  return false;
}

void LlamaEngine::project_lm_head_logits(const __half* x_norm, float* logits) {
  const auto& cfg = weights_.config();
  if (lm_head_packed_.active()) {
    ++packed_matvec_calls_;
    kernels::launch_kquant_matvec_f32(static_cast<const std::uint8_t*>(lm_head_packed_.data),
                                      static_cast<kernels::KQuantType>(lm_head_packed_.kind), x_norm,
                                      logits, lm_head_packed_.rows, lm_head_packed_.cols,
                                      compute_stream_);
  } else if (lm_head_int8_ && d_lm_head_i8_ != nullptr) {
    kernels::launch_weight_only_int8_gemv_f32(d_lm_head_i8_, d_lm_head_i8_scales_, x_norm, logits,
                                              cfg.vocab_size, cfg.hidden_size, compute_stream_);
  } else {
    resident_projection_float(d_lm_head_, x_norm, logits, cfg.vocab_size, cfg.hidden_size,
                              resident_lm_head_warps_, resident_lm_head_tile_pairs_,
                              resident_lm_head_rows_per_warp_);
  }
}

}  // namespace engine
