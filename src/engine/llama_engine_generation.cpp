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

namespace {
// Which quantized prefill matrices go through dequantize + tensor-core GEMM
// instead of the batched matvec kernels. A bitmask rather than one switch so
// each matrix can be enabled and its output read on its own: the two earlier
// attempts at this both measured a convincing speedup while producing nonsense,
// and an all-or-nothing flag cannot tell you which matrix lied.
//   bit 0 (1) wqkv, bit 1 (2) wo, bit 2 (4) w1/w3, bit 3 (8) w2
// All four are on by default; CPI_QUANT_PREFILL_GEMM=<mask> overrides, and 0
// restores the pure matvec path.
int quant_prefill_gemm_mask() {
  static const int value = []() {
    const char* env = std::getenv("CPI_QUANT_PREFILL_GEMM");
    return env ? std::max(0, std::atoi(env)) : 15;
  }();
  return value;
}
// Row count below which the matvec kernels still win (they are built for decode).
constexpr int kQuantPrefillGemmMinRows = 16;
}  // namespace

// Dequantizes one cached low-bit weight into the prefill scratch, in the layout
// llama_engine_lowbit_utils.cpp packs (row-wise scales; int4 as sequential column
// pairs). Returns null when the scratch cannot be sized, so the caller falls back.
const void* LlamaEngine::dequant_weight_for_gemm(const std::int8_t* w, const float* scales,
                                                 bool int4, int rows, int cols) {
  if (w == nullptr || scales == nullptr || rows <= 0 || cols <= 0) return nullptr;
  const std::size_t need =
      static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols) * sizeof(__half);
  if (need > d_prefill_wdq_bytes_) {
    // Queued GEMMs may still be reading the old buffer.
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    if (d_prefill_wdq_ != nullptr) cudaFree(d_prefill_wdq_);
    d_prefill_wdq_ = nullptr;
    d_prefill_wdq_bytes_ = 0;
    if (cudaMalloc(&d_prefill_wdq_, need) != cudaSuccess) {
      d_prefill_wdq_ = nullptr;
      return nullptr;
    }
    d_prefill_wdq_bytes_ = need;
  }
  kernels::launch_dequant_weight_rowwise_to_fp16(w, scales, static_cast<__half*>(d_prefill_wdq_),
                                                 rows, cols, int4, compute_stream_);
  return d_prefill_wdq_;
}

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
  // Quantized KV can batched-prefill through the paged pool (store + attention
  // quant kernels); the contiguous quantized cache still prefills sequentially.
  const bool kv_quant_needs_sequential = kv_int4_enabled_ && !options_.paged_blocks;
  if (options_.paged_kv_cache || prefill_chunk_size_ <= 1 ||
      (cached_int8_proj_enabled_ && !all_layers_cached) || kv_quant_needs_sequential ||
      tq3_enabled_ || cfg.is_moe() || cfg.uses_non_full_attention()) {
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
    if (eagle_enabled_) {
      // Feed this chunk's (post-norm feature, next token) pairs into the draft
      // cache: pair p consumes the token at position p+1, so the last prompt
      // token pairs with the final chunk's last feature row.
      launch_norm(d_x_, d_norm_out_, d_norm_out_bias_, d_x_norm_, rows, cfg.hidden_size);
      eagle_prefill_pairs(prompt_tokens.data() + chunk_start + 1, rows, chunk_start);
    }
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
    // Dequantize-then-GEMM is only worth it at prompt row counts, and only for
    // the row-wise scale layout (grouped int4 scales are per column group).
    const int wgemm_mask = quant_prefill_gemm_mask();
    const bool wgemm_rows_ok = rows >= kQuantPrefillGemmMinRows;
    const bool lowbit_mlp_rowwise =
        lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && lw_i8->mlp_group <= 0;
    bool fused_glu = false;  // fp16 path reads gate/up in place; quant paths still need the split

    launch_norm(d_x_, lw->norm_att, lw->norm_att_bias, d_x_norm_, rows, hidden);

    const void* wqkv_dq =
        (lowbit_proj && wgemm_rows_ok && (wgemm_mask & 1))
            ? dequant_weight_for_gemm(lw_i8->wqkv, lw_i8->s_wqkv, lw_i8->proj_int4,
                                      q_hidden + 2 * kv_hidden, hidden)
            : nullptr;
    if (wqkv_dq != nullptr) {
      detail::dispatch_linear_rowmajor_weight(
          cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
          const_cast<void*>(wqkv_dq), d_x_norm_, d_qkv_, q_hidden + 2 * kv_hidden, hidden, rows,
          CUDA_R_16F);
    } else if (lowbit_proj && can_use_dp4a_proj) {
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

    // Keep Q's split copy; drop K's and V's, and fold the QKV bias into one broadcast.
    //
    // Prefill is bound by host-side CUDA API calls, and three of the seven per-layer 2-D copies
    // exist only to un-interleave Q/K/V out of the fused QKV buffer. RoPE takes independent row
    // strides and the K/V cache stores can read strided, so K and V never need materialising.
    // The bias vector is exactly one fused QKV row wide, so one broadcast replaces three.
    //
    // Q's copy stays deliberately: reading Q in place changes the attention GEMM's leading
    // dimension, which makes cuBLAS select a different algorithm that sums in a different order
    // and perturbs the logits. Keeping the copy pins the stride and the result stays
    // bit-identical; and it also measured faster, the better-aligned GEMM more than paying for
    // the copy.
    //
    // Not eligible when:
    //   - QK-norm (Qwen3): launch_rmsnorm treats its input as contiguous [rows*heads, head_dim],
    //     which K inside the strided QKV buffer is not; it would normalise the wrong slices.
    //   - paged blocks: launch_store_kv_paged requires contiguous K/V.
    const bool fused_kv = !(lw->q_norm && lw->k_norm) &&
                          !(options_.paged_blocks && d_block_table_ != nullptr);
    auto* qkv_rw = static_cast<__half*>(d_qkv_);
    const int qkv_row = q_hidden + 2 * kv_hidden;

    // Bias before the split, so the copy carries the biased values (it used to be applied to the
    // three split buffers afterwards; same arithmetic, one launch instead of three).
    if (lw->bqkv && fused_kv) {
      kernels::launch_add_bias_broadcast(qkv_rw, static_cast<const __half*>(lw->bqkv), rows,
                                         qkv_row, compute_stream_);
    }

    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_q_, q_row_bytes, qkv_base, qkv_stride_bytes, q_row_bytes,
                                 rows, cudaMemcpyDeviceToDevice, compute_stream_));
    if (!fused_kv) {
      CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_k_, kv_row_bytes, qkv_base + q_hidden,
                                   qkv_stride_bytes, kv_row_bytes, rows, cudaMemcpyDeviceToDevice,
                                   compute_stream_));
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
    }

    if (lw->q_norm && lw->k_norm) {
      // Qwen3: per-head RMSNorm on Q and K, pre-RoPE. Only reachable on the split path.
      kernels::launch_rmsnorm(static_cast<const __half*>(d_prefill_q_),
                                     static_cast<const __half*>(lw->q_norm),
                                     static_cast<__half*>(d_prefill_q_), rows * cfg.num_heads,
                                     head_dim, cfg.norm_eps, compute_stream_);
      kernels::launch_rmsnorm(static_cast<const __half*>(d_prefill_k_),
                                     static_cast<const __half*>(lw->k_norm),
                                     static_cast<__half*>(d_prefill_k_), rows * cfg.num_kv_heads,
                                     head_dim, cfg.norm_eps, compute_stream_);
    }

    // Q is contiguous (its copy stayed); K lives in the fused buffer. Independent strides.
    kernels::launch_rope_inplace_batched_strided(
        static_cast<__half*>(d_prefill_q_), fused_kv ? (qkv_rw + q_hidden)
                                                     : static_cast<__half*>(d_prefill_k_),
        rows, cfg.num_heads, cfg.num_kv_heads, head_dim, base_pos, d_rope_cos_, d_rope_sin_,
        q_hidden, fused_kv ? qkv_row : kv_hidden, compute_stream_);

    auto* k_layer =
        static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride;
    auto* v_layer =
        static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride;
    if (options_.paged_blocks && d_block_table_ != nullptr) {
      // P3 phase 2d: scatter this chunk's KV into the block pool via the block
      // table, and run paged prefill attention (gather via the same table).
      const int bs = options_.paged_block_size > 0 ? options_.paged_block_size : 32;
      if (kv_int4_enabled_) {
        const std::size_t k_row = static_cast<std::size_t>(head_dim * kv_quant_kbits_ / 8);
        const std::size_t v_row = static_cast<std::size_t>(head_dim * kv_quant_vbits_ / 8);
        const std::size_t token_heads = static_cast<std::size_t>(kv_capacity_tokens_) *
                                        static_cast<std::size_t>(cfg.num_kv_heads);
        auto* kq = d_k_cache_i4_ + static_cast<std::size_t>(layer) * token_heads * k_row;
        auto* vq = d_v_cache_i4_ + static_cast<std::size_t>(layer) * token_heads * v_row;
        auto* ks = d_k_scales_ + static_cast<std::size_t>(layer) * token_heads;
        auto* vs = d_v_scales_ + static_cast<std::size_t>(layer) * token_heads;
        // Slot-resolved fp16 sink/ring bases for this sequence's quality slot
        // (prefill_kv_slot_; 0 for the single-sequence path). Side-buffer layout
        // is [layers, slots, n, kv_heads, head_dim].
        const std::size_t sink_n_elems = static_cast<std::size_t>(kv_quant_sink_) * kv_hidden;
        const std::size_t ring_n_elems = static_cast<std::size_t>(kv_quant_win_) * kv_hidden;
        const std::size_t sink_off =
            (static_cast<std::size_t>(layer) * kv_quality_slots_ + prefill_kv_slot_) *
            sink_n_elems;
        const std::size_t ring_off =
            (static_cast<std::size_t>(layer) * kv_quality_slots_ + prefill_kv_slot_) *
            ring_n_elems;
        auto* sink_k = d_kv_sink_k_ ? d_kv_sink_k_ + sink_off : nullptr;
        auto* sink_v = d_kv_sink_v_ ? d_kv_sink_v_ + sink_off : nullptr;
        auto* ring_k = d_kv_ring_k_ ? d_kv_ring_k_ + ring_off : nullptr;
        auto* ring_v = d_kv_ring_v_ ? d_kv_ring_v_ + ring_off : nullptr;
        kernels::launch_store_kv_paged_quant(
            static_cast<const __half*>(d_prefill_k_), static_cast<const __half*>(d_prefill_v_),
            kv_hidden, kq, vq, ks, vs, d_block_table_, base_pos, rows, cfg.num_kv_heads, head_dim,
            bs, kv_quant_kbits_, kv_quant_vbits_, kv_quant_rot_, compute_stream_, sink_k, sink_v,
            ring_k, ring_v, kv_quant_sink_, kv_quant_win_);
        kernels::launch_attention_prefill_paged_quant(
            static_cast<const __half*>(d_prefill_q_), kq, vq, ks, vs, d_block_table_,
            static_cast<__half*>(d_att_), rows, base_pos, cfg.num_heads, cfg.num_kv_heads,
            head_dim, bs, kv_quant_kbits_, kv_quant_vbits_, kv_quant_rot_, compute_stream_,
            static_cast<const __half*>(d_prefill_k_), static_cast<const __half*>(d_prefill_v_),
            kv_hidden, sink_k, sink_v, kv_quant_sink_);
      } else {
        kernels::launch_store_kv_paged(k_layer, v_layer, static_cast<const __half*>(d_prefill_k_),
                                       static_cast<const __half*>(d_prefill_v_), d_block_table_,
                                       base_pos, rows, kv_hidden, bs, compute_stream_);
        kernels::launch_attention_prefill_paged(
            static_cast<const __half*>(d_prefill_q_), k_layer, v_layer, d_block_table_,
            static_cast<__half*>(d_att_), rows, base_pos, cfg.num_heads, cfg.num_kv_heads,
            head_dim, bs, compute_stream_);
      }
    } else {
      // Store K/V into the cache. On the in-place path these read straight out of the fused QKV
      // buffer (source pitch = the whole QKV row) instead of from the split copies; same bytes,
      // one fewer copy each to produce them.
      const __half* k_src = fused_kv ? (qkv_rw + q_hidden) : static_cast<__half*>(d_prefill_k_);
      const __half* v_src =
          fused_kv ? (qkv_rw + q_hidden + kv_hidden) : static_cast<__half*>(d_prefill_v_);
      const std::size_t kv_src_pitch = fused_kv ? qkv_stride_bytes : kv_row_bytes;
      CUDA_CHECK(cudaMemcpy2DAsync(k_layer + static_cast<std::size_t>(base_pos) * kv_hidden,
                                   kv_row_bytes, k_src, kv_src_pitch, kv_row_bytes, rows,
                                   cudaMemcpyDeviceToDevice, compute_stream_));
      CUDA_CHECK(cudaMemcpy2DAsync(v_layer + static_cast<std::size_t>(base_pos) * kv_hidden,
                                   kv_row_bytes, v_src, kv_src_pitch, kv_row_bytes, rows,
                                   cudaMemcpyDeviceToDevice, compute_stream_));

      // Tensor cores first: Q.K^T and P.V as cuBLAS batched GEMMs. Falls back to the kernel if
      // the geometry is unsupported or the score matrix would be too big. This is plain causal
      // text prefill, no per-token limits, no sliding window, which is exactly the case the
      // GEMM path covers; the vision/bidirectional paths never reach here.
      //
      // On the in-place path Q lives in the fused buffer, so the stride is a whole QKV row.
      // The fallback kernel cannot take a stride; which is why inplace_qkv requires tc_ready.
      // q_hidden, always: Q keeps its contiguous copy so cuBLAS sees the same leading dimension
      // and picks the same algorithm; which is what keeps this bit-identical.
      if (!prefill_attention_tensorcore(d_prefill_q_, k_layer, v_layer, d_att_, rows, base_pos,
                                        cfg.num_heads, cfg.num_kv_heads, head_dim, q_hidden)) {
        kernels::launch_attention_prefill(static_cast<const __half*>(d_prefill_q_), k_layer,
                                          v_layer, static_cast<__half*>(d_att_), rows, base_pos,
                                          cfg.num_heads, cfg.num_kv_heads, head_dim,
                                          compute_stream_);
      }
    }

    const void* wo_dq = (lowbit_proj && wgemm_rows_ok && (wgemm_mask & 2))
                            ? dequant_weight_for_gemm(lw_i8->wo, lw_i8->s_wo, lw_i8->proj_int4,
                                                      hidden, q_hidden)
                            : nullptr;
    if (wo_dq != nullptr) {
      detail::dispatch_linear_rowmajor_weight(
          cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
          const_cast<void*>(wo_dq), d_att_, d_ff3_, hidden, q_hidden, rows, CUDA_R_16F);
    } else if (lowbit_proj && can_use_dp4a_proj) {
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

    const bool gate_gemm = lowbit_mlp_rowwise && wgemm_rows_ok && (wgemm_mask & 4);
    if (gate_gemm &&
        dequant_weight_for_gemm(lw_i8->w1, lw_i8->s_w1, lw_i8->mlp_int4, inter, hidden) !=
            nullptr) {
      // The scratch holds one matrix at a time, so each is dequantized right
      // before its own GEMM.
      detail::dispatch_linear_rowmajor_weight(
          cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
          d_prefill_wdq_, d_x_norm_, d_prefill_ff1_, inter, hidden, rows, CUDA_R_16F);
      dequant_weight_for_gemm(lw_i8->w3, lw_i8->s_w3, lw_i8->mlp_int4, inter, hidden);
      detail::dispatch_linear_rowmajor_weight(
          cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
          d_prefill_wdq_, d_x_norm_, d_prefill_ff2_, inter, hidden, rows, CUDA_R_16F);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && lw_i8->mlp_group > 0) {
      // Group-wise int4 MLP (w1/w3) via perm8 dp4a. Must match the grouped weight scales; a per-row
      // batched kernel here would misread them. w1/w3 share the one perm8 activation.
      kernels::launch_quantize_fp16_to_int8_perm8_g32_mt(
          static_cast<const __half*>(d_x_norm_), static_cast<std::int8_t*>(d_prefill_i8_),
          static_cast<float*>(d_prefill_perm8_scales_), hidden, rows, compute_stream_);
      const auto* xq = static_cast<const std::int8_t*>(d_prefill_i8_);
      const auto* xs = static_cast<const float*>(d_prefill_perm8_scales_);
      mlp_int4_grouped_dp4a(lw_i8->w1, lw_i8->s_w1, xq, xs, static_cast<__half*>(d_prefill_ff1_),
                            rows, inter, hidden, lw_i8->mlp_group);
      mlp_int4_grouped_dp4a(lw_i8->w3, lw_i8->s_w3, xq, xs, static_cast<__half*>(d_prefill_ff2_),
                            rows, inter, hidden, lw_i8->mlp_group);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && can_use_dp4a_prefill) {
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

      // Read gate/up straight off the fused [rows, 2*inter] w13 output rather than splitting it
      // into two buffers first: those copies only un-interleave data the GLU reads elementwise.
      kernels::launch_gated_glu_interleaved(ff13_base, static_cast<__half*>(d_prefill_ff2_), inter,
                                            rows, weights_.config().mlp_gelu, compute_stream_);
      fused_glu = true;
    }

    // not inside the else: the quantized branches above fill d_prefill_ff1_/ff2_ with separate
    // GEMMs and still need this. Folding it into the fp16 branch would compile, run, and
    // silently skip the activation for int8/int4.
    if (!fused_glu) {
      detail::launch_gated_glu(
          weights_.config().mlp_gelu, static_cast<const __half*>(d_prefill_ff1_),
          static_cast<const __half*>(d_prefill_ff2_), static_cast<__half*>(d_prefill_ff2_),
          rows * inter, compute_stream_);
    }

    const void* w2_dq = (lowbit_mlp_rowwise && wgemm_rows_ok && (wgemm_mask & 8))
                            ? dequant_weight_for_gemm(lw_i8->w2, lw_i8->s_w2, lw_i8->mlp_int4,
                                                      hidden, inter)
                            : nullptr;
    if (w2_dq != nullptr) {
      detail::dispatch_linear_rowmajor_weight(
          cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
          const_cast<void*>(w2_dq), d_prefill_ff2_, d_ff3_, hidden, inter, rows, CUDA_R_16F);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && lw_i8->mlp_group > 0) {
      // Group-wise int4 MLP (w2) via perm8 dp4a: post-GLU activation (d_prefill_ff2_).
      kernels::launch_quantize_fp16_to_int8_perm8_g32_mt(
          static_cast<const __half*>(d_prefill_ff2_), static_cast<std::int8_t*>(d_prefill_i8_),
          static_cast<float*>(d_prefill_perm8_scales_), inter, rows, compute_stream_);
      mlp_int4_grouped_dp4a(lw_i8->w2, lw_i8->s_w2, static_cast<const std::int8_t*>(d_prefill_i8_),
                            static_cast<const float*>(d_prefill_perm8_scales_),
                            static_cast<__half*>(d_ff3_), rows, hidden, inter, lw_i8->mlp_group);
    } else if (lw_i8 && lw_i8->w1 && lw_i8->w2 && lw_i8->w3 && can_use_dp4a_prefill) {
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
    // Same guard as the decode twin: fully-cached configs can still route here
    // (paged KV), and layers[num_layers] does not exist to pre-stream.
    if (cached_layer_count_ < cfg.num_layers) {
      CUDA_CHECK(cudaStreamWaitEvent(transfer_stream_, streaming_consumed_[0], 0));
      copy_layer_weights_to_device(
          cached_layer_count_, &streaming_layer_weights_[0],
          options_.int8_streaming ? &streaming_layer_weights_i8_[0] : nullptr, transfer_stream_);
      CUDA_CHECK(cudaEventRecord(streaming_ready_[0], transfer_stream_));
    }

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
    CPI_THROW("prompt token list is empty");
  }
  // Bounds guard: the KV cache and position tables are allocated for exactly
  // options_.max_context positions (see llama_engine_lifecycle.cpp). A prompt
  // that meets or exceeds the window would prefill KV out of bounds, corrupting
  // memory; observed as garbage output on some builds and a 0xC0000409
  // stack-buffer-overrun on others. Reject with a clear error before touching
  // any buffer, leaving at least one slot for a generated token.
  if (static_cast<int>(prompt_tokens.size()) >= options_.max_context) {
    CPI_THROW("prompt length " + std::to_string(prompt_tokens.size()) +
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
  // Lazy one-shot EAGLE head load (CPI_EAGLE=k); all target buffers exist here.
  if (!eagle_tried_) {
    eagle_tried_ = true;
    eagle_load();
  }
  // EAGLE needs the draft cache fed during prefill, which prefix reuse skips.
  const bool prefix_cacheable =
      !options_.disable_prefix_reuse &&                      // benchmarking wants a real prefill
      !options_.paged_kv_cache && !options_.paged_blocks &&  // paged: each request gets its own
      !kv_int4_enabled_ && !tq3_enabled_ &&             // block table, so prior KV isn't at the
      !cfg.is_moe() && !cfg.uses_non_full_attention() &&  // same physical blocks (paged
      !eagle_enabled_;                                    // shared-prefix is a later phase)
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
  // sizing + no-exhaustion + release across requests). Bookkeeping only in 2a;
  // the KV hot path still uses the flat cache, so output stays byte-identical.
  // Physical block ids are intentionally not contiguous/identity (the free-list
  // hands blocks back LIFO across requests); that permutation is exactly what
  // paging enables, and is consumed by the phase-2b gather kernel that will read
  // KV through this table instead of a flat offset.
  if (seq_blocks_) {
    // Phase 2d: allocate this sequence's blocks and publish its block table
    // (host + device). The allocator's free-list is LIFO, so requests after the
    // first get non-contiguous physical blocks, exercising real paging. Prefill
    // + decode KV writes and both attention reads all go through this table, so
    // byte-identical output proves non-contiguous paging works.
    seq_blocks_->clear();
    const int total = std::min(
        options_.max_context, static_cast<int>(prompt_tokens.size()) + std::max(0, max_new_tokens));
    if (!seq_blocks_->ensure_position(total - 1)) {
      CPI_THROW("paged KV pool exhausted for sequence length " + std::to_string(total));
    }
    const int bs = options_.paged_block_size > 0 ? options_.paged_block_size : 32;
    const int nblk = (total + bs - 1) / bs;
    block_table_host_.assign(static_cast<std::size_t>(nblk), 0);
    for (int c = 0; c < nblk; ++c) {
      const int blk = seq_blocks_->block_for(c * bs);
      if (blk == BlockAllocator::kInvalidBlock) {
        CPI_THROW("paged block table has no block for chunk " + std::to_string(c));
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

  // EAGLE chain-speculative decode: greedy only, no grammar/min-tokens, and the
  // same batched-path eligibility verify_tokens needs. The draft cache was fed
  // during prefill (see prefill_prompt); the loop verifies chains of k drafts.
  const bool eagle_eligible =
      eagle_enabled_ && temperature <= 0.0f && active_grammar_ == nullptr &&
      min_new_tokens == 0 && !options_.paged_kv_cache && prefill_chunk_size_ > 1 &&
      !(cached_int8_proj_enabled_ && cached_layer_count_ != cfg.num_layers) &&
      !kv_int4_enabled_ && !tq3_enabled_ && !cfg.is_moe() && !cfg.uses_non_full_attention() &&
      eagle_k_ + 1 <= prefill_chunk_size_ && reuse == 0;
  if (eagle_eligible) {
    const auto eagle_decode_start = std::chrono::steady_clock::now();
    const std::vector<int> gen = eagle_tree_
                                     ? eagle_tree_generate(prompt_tokens, max_new_tokens, on_token)
                                     : eagle_generate(prompt_tokens, max_new_tokens, on_token);
    out.insert(out.end(), gen.begin(), gen.end());
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    last_benchmark_stats_.decode_ms = std::chrono::duration<double, std::milli>(
                                          std::chrono::steady_clock::now() - eagle_decode_start)
                                          .count();
    last_benchmark_stats_.generated_tokens = static_cast<int>(gen.size());
    return out;
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
        CPI_THROW(oss.str());
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
    CPI_THROW("inspect_next_logits requires non-empty prompt");
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
    CPI_THROW("verify_tokens requires the batched full-attention path");
  }
  if (K > prefill_chunk_size_) {
    CPI_THROW("verify_tokens: K exceeds prefill_chunk_size_");
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
  // lm_head, reading its ~1 GB weight once; the per-position loop re-read it K
  // times (K x the lm_head bandwidth per verify). Consistent with the verify's
  // layer projections, which already run through the batched cuBLAS path. Then a
  // per-row device argmax (only K ints cross the bus).
  batched_lm_head(K, hidden, cfg.vocab_size);
  for (int i = 0; i < K; ++i) {
    kernels::launch_argmax_float(
        d_batch_logits_ + static_cast<std::size_t>(i) * static_cast<std::size_t>(cfg.vocab_size),
        cfg.vocab_size, d_argmax_, compute_stream_, d_argmax_part_val_, d_argmax_part_idx_,
        argmax_parts_);
    CUDA_CHECK(cudaMemcpyAsync(out_argmax.data() + i, d_argmax_, sizeof(int),
                               cudaMemcpyDeviceToHost, compute_stream_));
  }
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
}

}  // namespace engine
