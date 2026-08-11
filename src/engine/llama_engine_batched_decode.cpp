// P2: batched multi-sequence decode (one step over N sequences).
//
// This assembles the already-parity-verified batched primitives (per-position
// RoPE, batched paged KV scatter, batched paged split-K attention) around the
// existing batched projection GEMMs (dispatch_linear_rowmajor_weight, the same
// path prefill uses over `rows` tokens). Scope: fp16 resident weights
// (--gpu-cache-all), full-attention, paged KV. Greedy (argmax) per sequence.
//
// Correctness is proven by the independent-decode parity gate: decoding N
// sequences a step at a time through this method must produce the same tokens
// as decoding each sequence alone through the single-token path.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/llama_engine.hpp"
#include "engine/sampling.hpp"  // dispatch_sample_from_candidates (top-k verify closure)
#include "llama_engine_internal.hpp"
#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"

namespace engine {

// The batched decode path implements only the plain fp16 full-attention resident
// case (lw->wqkv/wo/w13/w2 fp16, non-paged single-token path otherwise). Reject
// anything else with a clear message instead of dereferencing null fp16 weights
// (int8/int4 keep their weights in layer_cache_i8_, MoE/TQ3 use other caches).
void LlamaEngine::require_batched_supported() const {
  const auto& cfg = weights_.config();
  const char* why = nullptr;
  if (!options_.paged_blocks)
    why = "requires --paged-blocks";
  else if (cached_layer_count_ != cfg.num_layers)
    why = "requires --gpu-cache-all (fully resident weights)";
  else if (cfg.is_moe())
    why = "MoE models are not supported by the batched path";
  else if (cfg.uses_non_full_attention())
    why = "sliding-window / non-full-attention models are not supported";
  else if (kv_int4_enabled_ &&
           !kernels::paged_quant_attention_supported(
               cfg.num_heads, cfg.num_kv_heads,
               attn_head_dim_ > 0 ? attn_head_dim_ : (cfg.hidden_size / cfg.num_heads),
               options_.paged_block_size > 0 ? options_.paged_block_size : 32))
    why = "quantized KV cache requires head_dim 128 and a GQA group for the batched path";
  else if (tq3_enabled_)
    why = "TurboQuant (TQ3) models are not supported by the batched path";
  else if (cached_int8_proj_enabled_ || cached_int8_mlp_enabled_) {
    why = "INT8/INT4-quantized models are not supported by the batched path (use the fp16 model)";
  }
  if (why) throw std::runtime_error(std::string("batched decode ") + why);
}

// Shared batched-decode forward: embed -> layers (paged batched attention) ->
// final norm. Leaves the per-row final hidden states in d_x_norm_ [batch][hidden]
// so the LM-head tail can produce either argmax (greedy) or per-row logits (for
// per-request sampling). Returns the batch size.
int LlamaEngine::decode_step_batched_forward(const std::vector<int>& tokens,
                                             const std::vector<int>& positions,
                                             const std::vector<int>& block_tables_flat,
                                             int max_blocks,
                                             const std::vector<int>& kv_slots) {
  const auto& cfg = weights_.config();
  const int batch = static_cast<int>(tokens.size());
  if (batch == 0) return 0;
  if (static_cast<int>(positions.size()) != batch ||
      static_cast<int>(block_tables_flat.size()) != batch * max_blocks) {
    throw std::runtime_error("decode_step_batched: ragged tokens/positions/block_tables");
  }
  if (!kv_slots.empty() && static_cast<int>(kv_slots.size()) != batch) {
    throw std::runtime_error("decode_step_batched: kv_slots must be empty or [batch]");
  }
  require_batched_supported();
  // Quality-slot ids for the quant sink/window tier: empty means all rows use
  // slot 0 (single-sequence and duplicate-row check callers). Grow the side
  // buffers before any kernel touches them; safe here, the previous step's
  // work was synced before its call returned.
  const bool kv_quality = kv_int4_enabled_ && (kv_quant_sink_ > 0 || kv_quant_win_ > 0) &&
                          (d_kv_sink_k_ != nullptr || d_kv_ring_k_ != nullptr);
  if (kv_quality) {
    int max_slot = 0;
    for (int s : kv_slots) max_slot = std::max(max_slot, s);
    ensure_kv_quality_slots(max_slot + 1);
  }

  const int hidden = cfg.hidden_size;
  const int inter = cfg.intermediate_size;
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : hidden;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (hidden / cfg.num_heads);
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const int bs = options_.paged_block_size > 0 ? options_.paged_block_size : 32;
  const std::size_t layer_stride =
      static_cast<std::size_t>(kv_capacity_tokens_) * static_cast<std::size_t>(kv_hidden);

  // Row-strided views into the fused QKV / gate-up buffers (per-row layout,
  // identical to run_batched_chunk).
  const std::size_t q_row_bytes = static_cast<std::size_t>(q_hidden) * sizeof(__half);
  const std::size_t kv_row_bytes = static_cast<std::size_t>(kv_hidden) * sizeof(__half);
  const std::size_t ff_row_bytes = static_cast<std::size_t>(inter) * sizeof(__half);
  const std::size_t qkv_stride_bytes =
      static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * sizeof(__half);
  const std::size_t ff13_stride_bytes = static_cast<std::size_t>(2 * inter) * sizeof(__half);
  auto* qkv_base = static_cast<const __half*>(d_qkv_);
  auto* ff13_base = static_cast<const __half*>(d_ff13_);

  // Per-sequence device state: seq_len[b] = positions[b] + 1; max over batch
  // sizes the attention's split-K scratch.
  std::vector<int> seq_lens(batch);
  int max_seq = 0;
  for (int b = 0; b < batch; ++b) {
    seq_lens[b] = positions[b] + 1;
    if (seq_lens[b] > max_seq) max_seq = seq_lens[b];
  }
  ensure_batch_state_buffers(batch, max_blocks);
  CUDA_CHECK(cudaMemcpyAsync(d_token_id_, tokens.data(), batch * sizeof(int),
                             cudaMemcpyHostToDevice, compute_stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_batch_positions_, positions.data(), batch * sizeof(int),
                             cudaMemcpyHostToDevice, compute_stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_batch_seq_lens_, seq_lens.data(), batch * sizeof(int),
                             cudaMemcpyHostToDevice, compute_stream_));
  CUDA_CHECK(cudaMemcpyAsync(d_batch_block_tables_, block_tables_flat.data(),
                             static_cast<std::size_t>(batch) * max_blocks * sizeof(int),
                             cudaMemcpyHostToDevice, compute_stream_));
  if (kv_quality) {
    if (kv_slots.empty()) {
      CUDA_CHECK(cudaMemsetAsync(d_batch_kv_slots_, 0, batch * sizeof(int), compute_stream_));
    } else {
      CUDA_CHECK(cudaMemcpyAsync(d_batch_kv_slots_, kv_slots.data(), batch * sizeof(int),
                                 cudaMemcpyHostToDevice, compute_stream_));
    }
  }

  // Split-K scratch for the batched attention: [batch*heads*chunks] stats and
  // [batch*heads*chunks*head_dim] partial outputs. Chunks bounded by max_seq.
  // Persistent (grown on demand); allocating/freeing per step would stall the
  // pipeline with synchronizing cudaMalloc/cudaFree on every decode step.
  const int chunks = (max_seq + bs - 1) / bs;
  const std::size_t stat_elems = static_cast<std::size_t>(batch) * cfg.num_heads * chunks;
  if (stat_elems > d_bs_stat_cap_) {
    if (d_bs_scratch_m_) cudaFree(d_bs_scratch_m_);
    if (d_bs_scratch_l_) cudaFree(d_bs_scratch_l_);
    if (d_bs_scratch_o_) cudaFree(d_bs_scratch_o_);
    CUDA_CHECK(cudaMalloc(&d_bs_scratch_m_, stat_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bs_scratch_l_, stat_elems * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bs_scratch_o_, stat_elems * head_dim * sizeof(float)));
    d_bs_stat_cap_ = stat_elems;
  }
  float* scratch_m = d_bs_scratch_m_;
  float* scratch_l = d_bs_scratch_l_;
  float* scratch_o = d_bs_scratch_o_;

  kernels::launch_embedding_lookup(static_cast<const __half*>(d_tok_embeddings_), d_token_id_,
                                   static_cast<__half*>(d_x_), batch, hidden, compute_stream_);
  if (cfg.scale_embeddings)  // Gemma: scale token embeddings by sqrt(hidden)
    kernels::launch_scale_copy(static_cast<__half*>(d_x_), static_cast<const __half*>(d_x_),
                               batch * hidden, std::sqrt(static_cast<float>(hidden)),
                               compute_stream_);

  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    const LayerDeviceWeights* lw = &layer_cache_[static_cast<std::size_t>(layer)];

    // --- Attention block ---
    launch_norm(d_x_, lw->norm_att, lw->norm_att_bias, d_x_norm_, batch, hidden);

    // Straight off the packed blocks when the batch is small enough to be worth
    // it; otherwise expand once and let cuBLAS have it.
    if (!packed_qkv_matmul(*lw, d_x_norm_, d_qkv_, batch, q_hidden + 2 * kv_hidden,
                           compute_stream_)) {
      const void* qkv_src = dequant_packed_qkv_for_gemm(*lw, compute_stream_);
      if (qkv_src == nullptr) qkv_src = lw->wqkv;
      detail::dispatch_linear_rowmajor_weight(
          cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
          const_cast<void*>(qkv_src), d_x_norm_, d_qkv_, q_hidden + 2 * kv_hidden, hidden, batch,
          CUDA_R_16F);
    }

    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_q_, q_row_bytes, qkv_base, qkv_stride_bytes, q_row_bytes,
                                 batch, cudaMemcpyDeviceToDevice, compute_stream_));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_k_, kv_row_bytes, qkv_base + q_hidden, qkv_stride_bytes,
                                 kv_row_bytes, batch, cudaMemcpyDeviceToDevice, compute_stream_));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_v_, kv_row_bytes, qkv_base + q_hidden + kv_hidden,
                                 qkv_stride_bytes, kv_row_bytes, batch, cudaMemcpyDeviceToDevice,
                                 compute_stream_));

    if (lw->bqkv) {
      const auto* bqkv_half = static_cast<const __half*>(lw->bqkv);
      kernels::launch_add_bias_broadcast(static_cast<__half*>(d_prefill_q_), bqkv_half, batch,
                                         q_hidden, compute_stream_);
      kernels::launch_add_bias_broadcast(static_cast<__half*>(d_prefill_k_), bqkv_half + q_hidden,
                                         batch, kv_hidden, compute_stream_);
      kernels::launch_add_bias_broadcast(static_cast<__half*>(d_prefill_v_),
                                         bqkv_half + q_hidden + kv_hidden, batch, kv_hidden,
                                         compute_stream_);
    }

    if (lw->q_norm && lw->k_norm) {
      // Qwen3 per-head RMSNorm on Q and K (batch rows × heads), pre-RoPE.
      kernels::launch_rmsnorm(static_cast<const __half*>(d_prefill_q_),
                              static_cast<const __half*>(lw->q_norm),
                              static_cast<__half*>(d_prefill_q_), batch * cfg.num_heads, head_dim,
                              cfg.norm_eps, compute_stream_);
      kernels::launch_rmsnorm(static_cast<const __half*>(d_prefill_k_),
                              static_cast<const __half*>(lw->k_norm),
                              static_cast<__half*>(d_prefill_k_), batch * cfg.num_kv_heads,
                              head_dim, cfg.norm_eps, compute_stream_);
    }

    kernels::launch_rope_inplace_perpos(static_cast<__half*>(d_prefill_q_),
                                        static_cast<__half*>(d_prefill_k_), batch, cfg.num_heads,
                                        cfg.num_kv_heads, head_dim, d_batch_positions_, d_rope_cos_,
                                        d_rope_sin_, compute_stream_);

    if (kv_int4_enabled_) {
      const std::size_t k_row = static_cast<std::size_t>(head_dim * kv_quant_kbits_ / 8);
      const std::size_t v_row = static_cast<std::size_t>(head_dim * kv_quant_vbits_ / 8);
      const std::size_t token_heads = static_cast<std::size_t>(kv_capacity_tokens_) *
                                      static_cast<std::size_t>(cfg.num_kv_heads);
      auto* kq = d_k_cache_i4_ + static_cast<std::size_t>(layer) * token_heads * k_row;
      auto* vq = d_v_cache_i4_ + static_cast<std::size_t>(layer) * token_heads * v_row;
      auto* ksc = d_k_scales_ + static_cast<std::size_t>(layer) * token_heads;
      auto* vsc = d_v_scales_ + static_cast<std::size_t>(layer) * token_heads;
      // Layer bases of the per-slot fp16 sink/ring quality buffers (kernels add
      // each row's slot offset from d_batch_kv_slots_).
      const std::size_t sink_lstride = static_cast<std::size_t>(kv_quality_slots_) *
                                       kv_quant_sink_ * cfg.num_kv_heads * head_dim;
      const std::size_t ring_lstride = static_cast<std::size_t>(kv_quality_slots_) *
                                       kv_quant_win_ * cfg.num_kv_heads * head_dim;
      auto* sink_k = (kv_quality && d_kv_sink_k_) ? d_kv_sink_k_ + layer * sink_lstride : nullptr;
      auto* sink_v = (kv_quality && d_kv_sink_v_) ? d_kv_sink_v_ + layer * sink_lstride : nullptr;
      auto* ring_k = (kv_quality && d_kv_ring_k_) ? d_kv_ring_k_ + layer * ring_lstride : nullptr;
      auto* ring_v = (kv_quality && d_kv_ring_v_) ? d_kv_ring_v_ + layer * ring_lstride : nullptr;
      const int* slot_ids = kv_quality ? d_batch_kv_slots_ : nullptr;
      kernels::launch_store_kv_batched_paged_quant(
          static_cast<const __half*>(d_prefill_k_), static_cast<const __half*>(d_prefill_v_), kq,
          vq, ksc, vsc, d_batch_block_tables_, d_batch_positions_, max_blocks, batch,
          cfg.num_kv_heads, head_dim, bs, kv_quant_kbits_, kv_quant_vbits_, kv_quant_rot_,
          compute_stream_, slot_ids, sink_k, sink_v, ring_k, ring_v, kv_quant_sink_,
          kv_quant_win_);
      kernels::launch_attention_step_batched_paged_quant(
          static_cast<const __half*>(d_prefill_q_), kq, vq, ksc, vsc, d_batch_block_tables_,
          d_batch_seq_lens_, max_blocks, max_seq, static_cast<__half*>(d_att_), batch,
          cfg.num_heads, cfg.num_kv_heads, head_dim, bs, kv_quant_kbits_, kv_quant_vbits_,
          kv_quant_rot_, compute_stream_, scratch_m, scratch_l, scratch_o, chunks, slot_ids,
          sink_k, sink_v, ring_k, ring_v, kv_quant_sink_, kv_quant_win_);
    } else {
      auto* k_pool =
          static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride;
      auto* v_pool =
          static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride;
      kernels::launch_store_kv_batched_paged(
          k_pool, v_pool, static_cast<const __half*>(d_prefill_k_),
          static_cast<const __half*>(d_prefill_v_), d_batch_block_tables_, d_batch_positions_,
          max_blocks, batch, kv_hidden, bs, compute_stream_);

      kernels::launch_attention_step_batched_paged(
          static_cast<const __half*>(d_prefill_q_), k_pool, v_pool, d_batch_block_tables_,
          d_batch_seq_lens_, max_blocks, max_seq, static_cast<__half*>(d_att_), batch,
          cfg.num_heads, cfg.num_kv_heads, head_dim, bs, compute_stream_, scratch_m, scratch_l,
          scratch_o, chunks);
    }

    // A packed k-quant weight has no fp16 copy for cuBLAS; expand it into the
    // shared prefill scratch. Same stream as the GEMM, so the expansion is
    // ordered before the read and before the next weight overwrites it.
    if (!packed_matmul(lw->wo_packed, d_att_, d_ff3_, batch, hidden, 0, compute_stream_)) {
      const void* wo_src = lw->wo_packed.active()
                               ? dequant_packed_for_gemm(lw->wo_packed, compute_stream_)
                               : static_cast<const void*>(lw->wo);
      detail::dispatch_linear_rowmajor_weight(
          cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
          const_cast<void*>(wo_src), d_att_, d_ff3_, hidden, q_hidden, batch, CUDA_R_16F);
    }
    maybe_add_half_bias(d_ff3_, lw->bo, batch, hidden);
    kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                batch * hidden, compute_stream_);

    // --- FFN block ---
    launch_norm(d_x_, lw->norm_ffn, lw->norm_ffn_bias, d_x_norm_, batch, hidden);

    // A packed k-quant weight has no fp16 copy for cuBLAS; expand it into the
    // shared prefill scratch. Same stream as the GEMM, so the expansion is
    // ordered before the read and before the next weight overwrites it.
    if (!packed_matmul(lw->w13_packed, d_x_norm_, d_ff13_, batch, 2 * inter, 0, compute_stream_)) {
      const void* w13_src = lw->w13_packed.active()
                                ? dequant_packed_for_gemm(lw->w13_packed, compute_stream_)
                                : static_cast<const void*>(lw->w13);
      detail::dispatch_linear_rowmajor_weight(
          cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
          const_cast<void*>(w13_src), d_x_norm_, d_ff13_, 2 * inter, hidden, batch, CUDA_R_16F);
    }
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_ff1_, ff_row_bytes, ff13_base, ff13_stride_bytes,
                                 ff_row_bytes, batch, cudaMemcpyDeviceToDevice, compute_stream_));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_ff2_, ff_row_bytes, ff13_base + inter, ff13_stride_bytes,
                                 ff_row_bytes, batch, cudaMemcpyDeviceToDevice, compute_stream_));
    detail::launch_gated_glu(weights_.config().mlp_gelu, static_cast<const __half*>(d_prefill_ff1_),
                             static_cast<const __half*>(d_prefill_ff2_),
                             static_cast<__half*>(d_prefill_ff2_), batch * inter, compute_stream_);
    // A packed k-quant weight has no fp16 copy for cuBLAS; expand it into the
    // shared prefill scratch. Same stream as the GEMM, so the expansion is
    // ordered before the read and before the next weight overwrites it.
    if (!packed_matmul(lw->w2_packed, d_prefill_ff2_, d_ff3_, batch, hidden, 0, compute_stream_)) {
      const void* w2_src = lw->w2_packed.active()
                               ? dequant_packed_for_gemm(lw->w2_packed, compute_stream_)
                               : static_cast<const void*>(lw->w2);
      detail::dispatch_linear_rowmajor_weight(
          cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
          const_cast<void*>(w2_src), d_prefill_ff2_, d_ff3_, hidden, inter, batch, CUDA_R_16F);
    }
    kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                batch * hidden, compute_stream_);
  }

  launch_norm(d_x_, d_norm_out_, d_norm_out_bias_, d_x_norm_, batch, hidden);
  return batch;
}

// Project all `batch` rows of d_x_norm_ through the LM head into d_batch_logits_
// ([batch][vocab] float) with a single GEMM (half in, float out via cublasGemmEx),
// replacing a per-row loop of GEMV + sync + big D2H copy.
void LlamaEngine::batched_lm_head(int batch, int hidden, int vocab) {
  if (batch > d_batch_logits_cap_) {
    if (d_batch_logits_) cudaFree(d_batch_logits_);
    CUDA_CHECK(
        cudaMalloc(&d_batch_logits_, static_cast<std::size_t>(batch) * vocab * sizeof(float)));
    d_batch_logits_cap_ = batch;
  }
    const void* head_src = lm_head_packed_.active()
                               ? dequant_packed_for_gemm(lm_head_packed_, compute_stream_)
                               : static_cast<const void*>(d_lm_head_);
  detail::dispatch_linear_rowmajor_weight(
      cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, compute_stream_,
      const_cast<void*>(head_src), d_x_norm_, d_batch_logits_, vocab, hidden, batch, CUDA_R_32F);
  if (d_lm_head_bias_) {
    for (int b = 0; b < batch; ++b) {
      kernels::launch_add_bias_inplace_float_from_half(
          d_batch_logits_ + static_cast<std::size_t>(b) * vocab,
          static_cast<const __half*>(d_lm_head_bias_), vocab, compute_stream_);
    }
  }
}

std::vector<int> LlamaEngine::decode_step_batched(const std::vector<int>& tokens,
                                                  const std::vector<int>& positions,
                                                  const std::vector<int>& block_tables_flat,
                                                  int max_blocks,
                                                  const std::vector<int>& kv_slots) {
  const int batch =
      decode_step_batched_forward(tokens, positions, block_tables_flat, max_blocks, kv_slots);
  if (batch == 0) return {};
  const int hidden = weights_.config().hidden_size;
  const int vocab = weights_.config().vocab_size;
  batched_lm_head(batch, hidden, vocab);
  // Per-row device argmax (only B ints cross the bus, not B*vocab floats).
  std::vector<int> next(batch);
  for (int b = 0; b < batch; ++b) {
    kernels::launch_argmax_float(d_batch_logits_ + static_cast<std::size_t>(b) * vocab, vocab,
                                 d_argmax_, compute_stream_);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    CUDA_CHECK(cudaMemcpy(&next[b], d_argmax_, sizeof(int), cudaMemcpyDeviceToHost));
  }
  return next;
}

void LlamaEngine::decode_step_batched_logits(const std::vector<int>& tokens,
                                             const std::vector<int>& positions,
                                             const std::vector<int>& block_tables_flat,
                                             int max_blocks,
                                             std::vector<std::vector<float>>& out_logits,
                                             const std::vector<int>& kv_slots) {
  static const bool prof = std::getenv("CPI_BATCH_PROFILE") != nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  const int batch =
      decode_step_batched_forward(tokens, positions, block_tables_flat, max_blocks, kv_slots);
  // resize (not assign) so rows kept from a previous step retain their vocab-sized capacity;
  // the copy loop below reuses it instead of reallocating every row every step.
  out_logits.resize(static_cast<std::size_t>(batch));
  if (batch == 0) return;
  const int hidden = weights_.config().hidden_size;
  const int vocab = weights_.config().vocab_size;
  if (prof) {
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    prof_fwd_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  }
  const auto t1 = std::chrono::steady_clock::now();
  // One batched GEMM, then one coalesced D2H copy of the whole [batch][vocab] block into a
  // persistent pinned buffer; pinned dst copies at ~full PCIe bandwidth (a pageable dst
  // stages through a driver bounce buffer at ~half) and the copy can be issued async, so the
  // single stream sync covers both the GEMM and the transfer. The buffer is reused across
  // steps rather than freshly allocating ~batch*vocab floats every decode.
  batched_lm_head(batch, hidden, vocab);
  const std::size_t need = static_cast<std::size_t>(batch) * static_cast<std::size_t>(vocab);
  if (batch > h_batch_logits_cap_) {
    if (h_batch_logits_) cudaFreeHost(h_batch_logits_);
    CUDA_CHECK(cudaHostAlloc(&h_batch_logits_, need * sizeof(float), cudaHostAllocDefault));
    h_batch_logits_cap_ = batch;
  }
  CUDA_CHECK(cudaMemcpyAsync(h_batch_logits_, d_batch_logits_, need * sizeof(float),
                             cudaMemcpyDeviceToHost, compute_stream_));
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  // Copy into out_logits reusing each row's existing capacity (resize on a same-size row is a
  // no-op, and out_logits is a scheduler-owned scratch reused across steps), so there is no
  // per-row reallocation. The remaining copy is a plain pinned->host memcpy.
  for (int b = 0; b < batch; ++b) {
    const float* src = h_batch_logits_ + static_cast<std::size_t>(b) * vocab;
    out_logits[b].resize(static_cast<std::size_t>(vocab));
    std::memcpy(out_logits[b].data(), src, static_cast<std::size_t>(vocab) * sizeof(float));
  }
  if (prof)
    prof_head_s_ += std::chrono::duration<double>(std::chrono::steady_clock::now() - t1).count();
}

bool LlamaEngine::decode_step_batched_topk(
    const std::vector<int>& tokens, const std::vector<int>& positions,
    const std::vector<int>& block_tables_flat, int max_blocks, const BatchTopkParams& sp,
    std::vector<std::vector<detail::SampleCandidate>>& out_cand,
    const std::vector<int>& kv_slots) {
  // must match llama_engine_graph.cpp: the shared d_topk_/d_cand_ buffers are sized to these.
  constexpr int kMaxDeviceTopK = 1024;
  constexpr int kCandCapacity = 4096;
  const int vocab = weights_.config().vocab_size;
  // Every row's k must be usable by the device top-k (the caller falls back to the logits path
  // for the whole batch otherwise; a per-row split is a later refinement).
  for (int kb : sp.k)
    if (kb <= 0 || kb > kMaxDeviceTopK || kb >= vocab) return false;

  const int batch =
      decode_step_batched_forward(tokens, positions, block_tables_flat, max_blocks, kv_slots);
  if (static_cast<int>(sp.k.size()) != batch) return false;
  out_cand.resize(static_cast<std::size_t>(batch));
  if (batch == 0) return true;
  const int hidden = weights_.config().hidden_size;
  batched_lm_head(batch, hidden, vocab);
  ensure_device_topk_buffers();

  // Repetition penalty lives before the top-k, on device, so the candidate set matches the host
  // slow path without shipping the vocab. In verify mode, snapshot the raw logits first so the
  // host reference can reproduce sanitize+penalty independently.
  static const bool verify = std::getenv("CPI_BATCH_TOPK_VERIFY") != nullptr;
  std::vector<float> raw;
  if (verify) {
    // batched_lm_head wrote d_batch_logits_ on compute_stream_; a default-stream copy would race
    // it, so sync first and snapshot the final pre-penalty logits the reference reproduces from.
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    raw.resize(static_cast<std::size_t>(batch) * vocab);
    CUDA_CHECK(cudaMemcpy(raw.data(), d_batch_logits_, raw.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }
  bool any_penalty = false;
  for (float p : sp.penalty)
    if (p > 1.0f) {
      any_penalty = true;
      break;
    }
  if (any_penalty) {
    if (batch > d_penalty_cap_) {
      // Reset the pointer/cap before the malloc: if it throws (OOM), the next grow must not
      // cudaFree the already-freed pointer.
      if (d_penalty_) {
        cudaFree(d_penalty_);
        d_penalty_ = nullptr;
        d_penalty_cap_ = 0;
      }
      CUDA_CHECK(cudaMalloc(&d_penalty_, static_cast<std::size_t>(batch) * sizeof(float)));
      d_penalty_cap_ = batch;
    }
    CUDA_CHECK(cudaMemcpyAsync(d_penalty_, sp.penalty.data(),
                               static_cast<std::size_t>(batch) * sizeof(float),
                               cudaMemcpyHostToDevice, compute_stream_));
    kernels::launch_sanitize_penalty_rows(d_batch_logits_, vocab, d_penalty_, batch,
                                          compute_stream_);
    const int total = static_cast<int>(sp.penalty_ids.size());
    if (total > 0) {
      if (total > d_seen_cap_) {
        // Same OOM safety as above: null both and drop the cap before either malloc, so a throw
        // can't leave a freed pointer that the next grow double-frees.
        if (d_seen_ids_) {
          cudaFree(d_seen_ids_);
          d_seen_ids_ = nullptr;
        }
        if (d_seen_rows_) {
          cudaFree(d_seen_rows_);
          d_seen_rows_ = nullptr;
        }
        d_seen_cap_ = 0;
        CUDA_CHECK(cudaMalloc(&d_seen_ids_, static_cast<std::size_t>(total) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_seen_rows_, static_cast<std::size_t>(total) * sizeof(int)));
        d_seen_cap_ = total;
      }
      CUDA_CHECK(cudaMemcpyAsync(d_seen_ids_, sp.penalty_ids.data(),
                                 static_cast<std::size_t>(total) * sizeof(int),
                                 cudaMemcpyHostToDevice, compute_stream_));
      CUDA_CHECK(cudaMemcpyAsync(d_seen_rows_, sp.penalty_rows.data(),
                                 static_cast<std::size_t>(total) * sizeof(int),
                                 cudaMemcpyHostToDevice, compute_stream_));
      kernels::launch_repetition_penalty(d_batch_logits_, vocab, d_seen_ids_, d_seen_rows_,
                                         d_penalty_, total, compute_stream_);
    }
  }

  // Per row: device top-k (the row's k-th value is the threshold) then a gather of every finite
  // logit >= threshold; exactly the candidate set sample_from_logits_topk builds on the host, so
  // only those (a few dozen) cross the bus instead of the full vocab. Same logic as the
  // single-sequence graph path, looped over the batch's rows.
  for (int b = 0; b < batch; ++b) {
    const int kb = sp.k[static_cast<std::size_t>(b)];
    const float* row = d_batch_logits_ + static_cast<std::size_t>(b) * vocab;
    CUDA_CHECK(cudaMemsetAsync(d_cand_count_, 0, sizeof(int), compute_stream_));
    kernels::launch_topk_float(row, vocab, kb, d_topk_part_val_, d_topk_part_idx_, d_topk_val_,
                               d_topk_idx_, compute_stream_);
    kernels::launch_gather_ge_threshold(row, vocab, d_topk_val_ + (kb - 1), d_cand_idx_,
                                        d_cand_val_, d_cand_count_, kCandCapacity, compute_stream_);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));

    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, d_cand_count_, sizeof(int), cudaMemcpyDeviceToHost));
    if (count <= 0 || count > kCandCapacity) return false;  // pathological tie -> caller falls back
    std::vector<int> idx(static_cast<std::size_t>(count));
    std::vector<float> val(static_cast<std::size_t>(count));
    CUDA_CHECK(cudaMemcpy(idx.data(), d_cand_idx_, static_cast<std::size_t>(count) * sizeof(int),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(val.data(), d_cand_val_, static_cast<std::size_t>(count) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    auto& cand = out_cand[static_cast<std::size_t>(b)];
    cand.resize(static_cast<std::size_t>(count));
    for (int i = 0; i < count; ++i) {
      cand[static_cast<std::size_t>(i)] = {idx[static_cast<std::size_t>(i)],
                                           val[static_cast<std::size_t>(i)]};
    }
    // The gather appends atomically (arbitrary order); the host builds in index order. Sort to
    // match so a seeded run produces the same token as the host path.
    std::sort(cand.begin(), cand.end(),
              [](const detail::SampleCandidate& a, const detail::SampleCandidate& b) {
                return a.id < b.id;
              });

    // CPI_BATCH_TOPK_VERIFY=1: rebuild the candidate set the host slow path would produce from the
    // raw logits (sanitize, then repetition penalty, then the k-th-largest threshold and gather of
    // every finite logit >= it) and compare id-sets. If they ever differ, the fast path could
    // sample a different token than the full path.
    if (verify) {
      std::vector<float> host(raw.begin() + static_cast<std::ptrdiff_t>(b) * vocab,
                              raw.begin() + static_cast<std::ptrdiff_t>(b + 1) * vocab);
      const float p = sp.penalty[static_cast<std::size_t>(b)];
      if (p > 1.0f) {
        for (float& v : host)
          v = std::isfinite(v) ? std::min(80.0f, std::max(-80.0f, v)) : -INFINITY;
        for (std::size_t i = 0; i < sp.penalty_ids.size(); ++i) {
          if (sp.penalty_rows[i] != b) continue;
          const int id = sp.penalty_ids[i];
          if (id >= 0 && id < vocab)
            host[static_cast<std::size_t>(id)] = host[static_cast<std::size_t>(id)] > 0.0f
                                                     ? host[static_cast<std::size_t>(id)] / p
                                                     : host[static_cast<std::size_t>(id)] * p;
        }
      }
      std::vector<float> part(host);
      std::nth_element(part.begin(), part.begin() + (kb - 1), part.end(), std::greater<float>());
      const float kth = part[static_cast<std::size_t>(kb - 1)];
      std::vector<int> ref;
      for (int i = 0; i < vocab; ++i)
        if (std::isfinite(host[static_cast<std::size_t>(i)]) &&
            host[static_cast<std::size_t>(i)] >= kth)
          ref.push_back(i);
      std::vector<int> got;
      got.reserve(cand.size());
      for (const auto& c : cand) got.push_back(c.id);
      std::sort(ref.begin(), ref.end());
      if (ref != got) {
        std::fprintf(stderr, "[topk-verify] MISMATCH(set) row=%d device=%zu host=%zu\n", b,
                     got.size(), ref.size());
      }

      // Closure over the id-set check: run the actual shared sampler
      // (dispatch_sample_from_candidates; the exact code the scheduler uses to turn candidates
      // into a token) on both the device candidate set and the host reference set, seeded
      // identically, and compare the drawn token. Because both go through the same traversal, an
      // agreeing token proves the sets and their values agree; the id-set check above ignores the
      // candidate values, so this catches a penalty/sanitize value bug that leaves the ids intact.
      // We deliberately do not compare against the full sample_from_logits path: it walks the vocab
      // in index order while the candidate path walks in probability order, so the two map the same
      // RNG draw to different tokens by construction (distributionally equal, not token-identical).
      // top_p = 1 keeps the whole candidate set (most value-sensitive); several seeds guard against
      // a coincidental agreement masking a value difference.
      std::vector<detail::SampleCandidate> host_cand;
      host_cand.reserve(ref.size());
      for (int id : ref) host_cand.push_back({id, host[static_cast<std::size_t>(id)]});
      for (unsigned seed : {1u, 7u, 99u}) {
        std::vector<detail::SampleCandidate> dc(cand), hc(host_cand);
        detail::dispatch_seed_sampler_rng(seed);
        const int td = detail::dispatch_sample_from_candidates(dc, 1.0f, 1.0f);
        detail::dispatch_seed_sampler_rng(seed);
        const int th = detail::dispatch_sample_from_candidates(hc, 1.0f, 1.0f);
        if (td != th) {
          std::fprintf(stderr, "[topk-verify] MISMATCH(token) row=%d seed=%u device=%d host=%d\n",
                       b, seed, td, th);
        }
      }
    }
  }
  return true;
}

bool LlamaEngine::decode_step_batched_argmax(const std::vector<int>& tokens,
                                             const std::vector<int>& positions,
                                             const std::vector<int>& block_tables_flat,
                                             int max_blocks, const BatchArgmaxParams& ap,
                                             std::vector<int>& out_ids,
                                             const std::vector<int>& kv_slots) {
  const int vocab = weights_.config().vocab_size;
  const int batch =
      decode_step_batched_forward(tokens, positions, block_tables_flat, max_blocks, kv_slots);
  out_ids.resize(static_cast<std::size_t>(batch));
  if (batch == 0) return false;
  if (static_cast<int>(ap.penalty.size()) != batch ||
      static_cast<int>(ap.blocked.size()) != batch) {
    return false;
  }
  const int hidden = weights_.config().hidden_size;
  batched_lm_head(batch, hidden, vocab);

  // In verify mode, snapshot the raw (pre-penalty) logits so the host reference can reproduce the
  // whole greedy path independently. batched_lm_head wrote on compute_stream_, so sync first.
  static const bool verify = std::getenv("CPI_BATCH_TOPK_VERIFY") != nullptr;
  std::vector<float> raw;
  if (verify) {
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    raw.resize(static_cast<std::size_t>(batch) * vocab);
    CUDA_CHECK(cudaMemcpy(raw.data(), d_batch_logits_, raw.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
  }

  // Repetition penalty on-device for penalty rows: sanitize the row, then scale each unique seen
  // id; identical to the top-k path, so the argmax runs on the same logits the host would. The
  // argmax kernel also sanitizes inline, so non-penalty rows need no separate sanitize here.
  bool any_penalty = false;
  for (float p : ap.penalty)
    if (p > 1.0f) {
      any_penalty = true;
      break;
    }
  if (any_penalty) {
    if (batch > d_penalty_cap_) {
      if (d_penalty_) {
        cudaFree(d_penalty_);
        d_penalty_ = nullptr;
        d_penalty_cap_ = 0;
      }
      CUDA_CHECK(cudaMalloc(&d_penalty_, static_cast<std::size_t>(batch) * sizeof(float)));
      d_penalty_cap_ = batch;
    }
    CUDA_CHECK(cudaMemcpyAsync(d_penalty_, ap.penalty.data(),
                               static_cast<std::size_t>(batch) * sizeof(float),
                               cudaMemcpyHostToDevice, compute_stream_));
    kernels::launch_sanitize_penalty_rows(d_batch_logits_, vocab, d_penalty_, batch,
                                          compute_stream_);
    const int total = static_cast<int>(ap.penalty_ids.size());
    if (total > 0) {
      if (total > d_seen_cap_) {
        if (d_seen_ids_) {
          cudaFree(d_seen_ids_);
          d_seen_ids_ = nullptr;
        }
        if (d_seen_rows_) {
          cudaFree(d_seen_rows_);
          d_seen_rows_ = nullptr;
        }
        d_seen_cap_ = 0;
        CUDA_CHECK(cudaMalloc(&d_seen_ids_, static_cast<std::size_t>(total) * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_seen_rows_, static_cast<std::size_t>(total) * sizeof(int)));
        d_seen_cap_ = total;
      }
      CUDA_CHECK(cudaMemcpyAsync(d_seen_ids_, ap.penalty_ids.data(),
                                 static_cast<std::size_t>(total) * sizeof(int),
                                 cudaMemcpyHostToDevice, compute_stream_));
      CUDA_CHECK(cudaMemcpyAsync(d_seen_rows_, ap.penalty_rows.data(),
                                 static_cast<std::size_t>(total) * sizeof(int),
                                 cudaMemcpyHostToDevice, compute_stream_));
      kernels::launch_repetition_penalty(d_batch_logits_, vocab, d_seen_ids_, d_seen_rows_,
                                         d_penalty_, total, compute_stream_);
    }
  }

  // Grow the argmax buffers and upload the per-row blocked id, then reduce.
  if (batch > d_argmax_cap_) {
    if (d_argmax_blocked_) {
      cudaFree(d_argmax_blocked_);
      d_argmax_blocked_ = nullptr;
    }
    if (d_argmax_out_) {
      cudaFree(d_argmax_out_);
      d_argmax_out_ = nullptr;
    }
    d_argmax_cap_ = 0;
    CUDA_CHECK(cudaMalloc(&d_argmax_blocked_, static_cast<std::size_t>(batch) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_argmax_out_, static_cast<std::size_t>(batch) * sizeof(int)));
    d_argmax_cap_ = batch;
  }
  CUDA_CHECK(cudaMemcpyAsync(d_argmax_blocked_, ap.blocked.data(),
                             static_cast<std::size_t>(batch) * sizeof(int), cudaMemcpyHostToDevice,
                             compute_stream_));
  kernels::launch_batched_argmax(d_batch_logits_, vocab, d_argmax_blocked_, d_argmax_out_, batch,
                                 compute_stream_);
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  CUDA_CHECK(cudaMemcpy(out_ids.data(), d_argmax_out_,
                        static_cast<std::size_t>(batch) * sizeof(int), cudaMemcpyDeviceToHost));

  // CPI_BATCH_TOPK_VERIFY=1: rebuild the winner the host greedy path would pick from the raw logits
  // (sanitize all -> repetition penalty -> block EOS -> argmax) and compare. The inline re-clamp in
  // the kernel only touches values already <= -80 or a shrunk positive, so it never changes which
  // id wins; the reference deliberately does not re-clamp after the penalty, matching the host.
  if (verify) {
    for (int b = 0; b < batch; ++b) {
      std::vector<float> host(raw.begin() + static_cast<std::ptrdiff_t>(b) * vocab,
                              raw.begin() + static_cast<std::ptrdiff_t>(b + 1) * vocab);
      for (float& v : host) v = std::isfinite(v) ? std::min(80.0f, std::max(-80.0f, v)) : -INFINITY;
      const float p = ap.penalty[static_cast<std::size_t>(b)];
      if (p > 1.0f) {
        for (std::size_t i = 0; i < ap.penalty_ids.size(); ++i) {
          if (ap.penalty_rows[i] != b) continue;
          const int id = ap.penalty_ids[i];
          if (id >= 0 && id < vocab)
            host[static_cast<std::size_t>(id)] = host[static_cast<std::size_t>(id)] > 0.0f
                                                     ? host[static_cast<std::size_t>(id)] / p
                                                     : host[static_cast<std::size_t>(id)] * p;
        }
      }
      const int blk = ap.blocked[static_cast<std::size_t>(b)];
      if (blk >= 0 && blk < vocab) host[static_cast<std::size_t>(blk)] = -INFINITY;
      const int ref = static_cast<int>(std::max_element(host.begin(), host.end()) - host.begin());
      if (ref != out_ids[static_cast<std::size_t>(b)]) {
        std::fprintf(stderr, "[argmax-verify] MISMATCH row=%d device=%d host=%d\n", b,
                     out_ids[static_cast<std::size_t>(b)], ref);
      }
    }
  }
  return true;
}

void LlamaEngine::run_batched_decode_check(const std::vector<int>& prompt_tokens, int num_steps) {
  if (!options_.paged_blocks || !seq_blocks_) {
    throw std::runtime_error("batched-decode check requires --paged-blocks");
  }
  if (cached_layer_count_ != weights_.config().num_layers) {
    throw std::runtime_error("batched-decode check requires --gpu-cache-all (fully resident)");
  }
  const int seq_len = static_cast<int>(prompt_tokens.size());
  if (seq_len == 0) throw std::runtime_error("batched-decode check requires a non-empty prompt");
  const int total = std::min(options_.max_context, seq_len + std::max(1, num_steps));
  const int steps = total - seq_len;
  const int bs = options_.paged_block_size > 0 ? options_.paged_block_size : 32;

  // Build this sequence's paged block table for the whole run (host + device),
  // exactly as generate_stream does, then prefill the prompt through it.
  reset_kv_cache();
  seq_blocks_->clear();
  if (!seq_blocks_->ensure_position(total - 1)) {
    throw std::runtime_error("batched-decode check: paged pool exhausted");
  }
  const int nblk = (total + bs - 1) / bs;
  block_table_host_.assign(static_cast<std::size_t>(nblk), 0);
  for (int c = 0; c < nblk; ++c) {
    block_table_host_[static_cast<std::size_t>(c)] = seq_blocks_->block_for(c * bs);
  }
  CUDA_CHECK(cudaMemcpy(d_block_table_, block_table_host_.data(),
                        static_cast<std::size_t>(nblk) * sizeof(int), cudaMemcpyHostToDevice));
  prefill_prompt(prompt_tokens);

  // Step-by-step: single-token path (reference) vs decode_step_batched at N=1 and
  // N=2 (duplicate rows). KV writes at the current position are idempotent (same
  // K/V to the same slot), so running both at the same state is safe; identical
  // logits => identical argmax. Advance by the reference token.
  std::printf("batched-decode check: prompt=%d steps=%d block_size=%d blocks=%d\n", seq_len, steps,
              bs, nblk);
  std::vector<int> dup_table = block_table_host_;
  dup_table.insert(dup_table.end(), block_table_host_.begin(),
                   block_table_host_.end());  // [2][nblk]

  int cur = prompt_tokens.back();
  int pos = seq_len - 1;
  int mism1 = 0, mism2 = 0;
  for (int s = 0; s < steps; ++s) {
    int single_arg = -1;
    forward_token_logits(cur, pos, nullptr, &single_arg);  // argmax branch writes KV + greedy id

    const auto n1 = decode_step_batched({cur}, {pos}, block_table_host_, nblk);
    const auto n2 = decode_step_batched({cur, cur}, {pos, pos}, dup_table, nblk);

    const bool ok1 = (n1[0] == single_arg);
    const bool ok2 = (n2[0] == single_arg && n2[1] == single_arg);
    if (!ok1) ++mism1;
    if (!ok2) ++mism2;
    if (s < 12 || !ok1 || !ok2) {
      std::printf("  step %2d pos %4d  single=%d  batched[N=1]=%d %s  batched[N=2]=%d,%d %s\n", s,
                  pos, single_arg, n1[0], ok1 ? "ok" : "MISMATCH", n2[0], n2[1],
                  ok2 ? "ok" : "MISMATCH");
    }
    cur = single_arg;
    ++pos;
  }
  std::printf("batched-decode check: %s  (N=1 mismatches=%d/%d, N=2 mismatches=%d/%d)\n",
              (mism1 == 0 && mism2 == 0) ? "PASS" : "FAIL", mism1, steps, mism2, steps);
}

std::vector<int> LlamaEngine::greedy_generate_single(const std::vector<int>& prompt, int max_new,
                                                     int eos_id) {
  const int seq_len = static_cast<int>(prompt.size());
  const int total = std::min(options_.max_context, seq_len + std::max(1, max_new));
  const int bs = options_.paged_block_size > 0 ? options_.paged_block_size : 32;

  reset_kv_cache();
  seq_blocks_->clear();
  if (!seq_blocks_->ensure_position(total - 1)) {
    throw std::runtime_error("scheduler check (reference): paged pool exhausted");
  }
  const int nblk = (total + bs - 1) / bs;
  block_table_host_.assign(static_cast<std::size_t>(nblk), 0);
  for (int c = 0; c < nblk; ++c)
    block_table_host_[static_cast<std::size_t>(c)] = seq_blocks_->block_for(c * bs);
  CUDA_CHECK(cudaMemcpy(d_block_table_, block_table_host_.data(),
                        static_cast<std::size_t>(nblk) * sizeof(int), cudaMemcpyHostToDevice));
  prefill_prompt(prompt);

  std::vector<int> out;
  int cur = prompt.back();
  int pos = seq_len - 1;
  for (int s = 0; s < max_new && pos < options_.max_context; ++s) {
    int arg = -1;
    forward_token_logits(cur, pos, nullptr, &arg);
    out.push_back(arg);
    if (arg == eos_id) break;
    cur = arg;
    ++pos;
  }
  seq_blocks_->clear();
  return out;
}

// The backend half of continuous batching. Everything else the scheduler does; admission,
// preemption, block growth, the shared-prefix LRU, sampling, retirement; is host logic and
// now lives in engine::BatchScheduler, shared with Metal. These are the only two operations
// that ever needed a GPU, which is why keeping the rest in this class made batching CUDA-only.
class LlamaEngine::BatchAdapter final : public BatchBackend {
public:
  explicit BatchAdapter(LlamaEngine* e) : e_(e) {}

  void prefill_suffix(const std::vector<int>& prompt_tokens, int shared_tokens,
                      const std::vector<int>& block_table, int kv_slot) override {
    // d_block_table_ is shared engine state and a fully-cached prefill runs async on
    // compute_stream_, so this must be settled before returning: a later admit or the decode
    // loop would otherwise clobber it. That is the BatchBackend contract, not an internal
    // detail.
    e_->block_table_host_ = block_table;
    CUDA_CHECK(cudaMemcpy(e_->d_block_table_, block_table.data(), block_table.size() * sizeof(int),
                          cudaMemcpyHostToDevice));
    // Route the prefill's paged quant KV stores into this sequence's quality slot
    // (sink + recent-window fp16 side buffers). Restored to the single-sequence
    // default before returning, per the "settled on return" contract above.
    e_->ensure_kv_quality_slots(kv_slot + 1);
    e_->prefill_kv_slot_ = kv_slot;
    e_->prefill_prompt(prompt_tokens, shared_tokens);
    CUDA_CHECK(cudaStreamSynchronize(e_->compute_stream_));
    e_->prefill_kv_slot_ = 0;
  }

  void decode_batched_logits(const std::vector<int>& tokens, const std::vector<int>& positions,
                             const std::vector<int>& block_tables_flat, int max_blocks,
                             const std::vector<int>& kv_slots,
                             std::vector<std::vector<float>>& out_logits) override {
    e_->decode_step_batched_logits(tokens, positions, block_tables_flat, max_blocks, out_logits,
                                   kv_slots);
  }

  bool decode_batched_topk(const std::vector<int>& tokens, const std::vector<int>& positions,
                           const std::vector<int>& block_tables_flat, int max_blocks,
                           const std::vector<int>& kv_slots, const BatchTopkParams& sp,
                           std::vector<std::vector<detail::SampleCandidate>>& out_cand) override {
    return e_->decode_step_batched_topk(tokens, positions, block_tables_flat, max_blocks, sp,
                                        out_cand, kv_slots);
  }

  bool decode_batched_argmax(const std::vector<int>& tokens, const std::vector<int>& positions,
                             const std::vector<int>& block_tables_flat, int max_blocks,
                             const std::vector<int>& kv_slots, const BatchArgmaxParams& ap,
                             std::vector<int>& out_ids) override {
    return e_->decode_step_batched_argmax(tokens, positions, block_tables_flat, max_blocks, ap,
                                          out_ids, kv_slots);
  }

private:
  LlamaEngine* e_;
};

BatchScheduler& LlamaEngine::ensure_scheduler() {
  if (!scheduler_) {
    if (!block_alloc_) throw std::runtime_error("stream_admit requires --paged-blocks");
    require_batched_supported();
    batch_adapter_ = std::make_unique<BatchAdapter>(this);
    BatchSchedulerOptions o;
    o.paged_block_size = options_.paged_block_size > 0 ? options_.paged_block_size : 32;
    o.max_context = options_.max_context;
    o.eos_token_id = options_.eos_token_id;
    o.verbose = options_.verbose;
    // The quant sink/window quality tier keeps per-sequence fp16 side buffers that only
    // this sequence's own prefill + decode write. An adopted shared prefix would leave
    // holes in them (its positions were prefilled by another sequence into another
    // slot), so prefix reuse is off under this tier. Backfill-on-adopt is a possible
    // follow-up.
    o.disable_prefix_reuse = kv_int4_enabled_ && (kv_quant_sink_ > 0 || kv_quant_win_ > 0);
    // Pre-size the quality-slot pool for a full batch up front: growing it
    // mid-serving means GB-scale cudaMallocs in the admission path (measured
    // as a multi-second stall the first time a batch crossed 32 concurrent
    // sequences). 64 slots ~= 1.2GB for the 8B; beyond that the on-demand
    // growth still works, it is just no longer the common case.
    if (o.disable_prefix_reuse) ensure_kv_quality_slots(64);
    scheduler_ = std::make_unique<BatchScheduler>(batch_adapter_.get(), block_alloc_.get(), o);
  }
  return *scheduler_;
}

int LlamaEngine::stream_active() const {
  return scheduler_ ? scheduler_->active() : 0;
}

void LlamaEngine::stream_admit(const std::string& id, const std::vector<int>& prompt_tokens,
                               const StreamParams& params) {
  ensure_scheduler().admit(id, prompt_tokens, params);
}

bool LlamaEngine::stream_step(std::vector<StreamEvent>& events) {
  if (!scheduler_) {
    events.clear();
    return false;
  }
  return scheduler_->step(events);
}

bool LlamaEngine::stream_cancel(const std::string& id) {
  return scheduler_ && scheduler_->cancel(id);
}

std::vector<std::vector<int>> LlamaEngine::run_batch(const std::vector<BatchRequest>& requests) {
  if (!options_.paged_blocks || !block_alloc_) {
    throw std::runtime_error("run_batch requires --paged-blocks");
  }
  const int N = static_cast<int>(requests.size());
  reset_kv_cache();  // release any prior blocks before we admit

  std::unordered_map<std::string, int> id_to_index;
  std::vector<std::vector<int>> outs(N);
  for (int i = 0; i < N; ++i) {
    StreamParams p;
    p.max_new_tokens = requests[i].max_new_tokens;
    p.temperature = requests[i].temperature;
    // Sampling shape is now per-request; the bench/single-flight caller fills it from the
    // engine-wide options so behaviour is unchanged.
    p.top_k = options_.top_k;
    p.top_p = options_.top_p;
    p.repetition_penalty = options_.repetition_penalty;
    p.no_repeat_ngram_size = options_.no_repeat_ngram_size;
    if (requests[i].eos_id >= 0) p.stop_ids.push_back(requests[i].eos_id);
    const std::string id = std::to_string(i);
    id_to_index[id] = i;
    stream_admit(id, requests[i].prompt, p);
  }

  std::vector<StreamEvent> events;
  while (stream_step(events)) {
    for (const auto& e : events)
      if (e.token >= 0) outs[id_to_index[e.id]].push_back(e.token);  // skip preempt sentinel
  }
  return outs;
}

// Per-box search over the k-quant kernel knobs.
//
// Every parameter in these kernels that was chosen by reasoning turned out
// wrong on measurement -- warps per row, the batched cutoff, the mma tile width,
// twice. What settled each of them was a paired comparison in one process. This
// automates exactly that, and the shape is lifted from PlanCudaEngine::autotune,
// which already learned the same lessons:
//
//   - a throwaway run first, because the first timing at a new shape lands on
//     cold clocks and a cold incumbent loses to any warm candidate;
//   - the incumbent re-timed in the same pass rather than trusting an earlier
//     number, because drift between runs is larger than most of these effects;
//   - hysteresis, so a candidate has to win by more than the noise to displace
//     a default.
//
// Timed on real batched decode, not an isolated kernel benchmark: an isolated
// per-shape sweep picked the wrong warps-per-row three times running, because a
// matvec measured alone owns the whole GPU and a warm L2.
void LlamaEngine::tune_kquant_knobs(int batch, int steps) {
  if (cached_layer_count_ != weights_.config().num_layers) {
    std::fprintf(stderr, "[tune] k-quant tuning needs a fully resident model; skipping\n");
    return;
  }
  const std::vector<int> prompt(8, 1);
  // batch==1 tunes the single-stream path, which goes through the matvec and the
  // decode graph rather than run_batch -- the batched harness never touches the
  // matvec, so tuning its warps-per-row through it would measure nothing.
  const bool single = batch <= 1;
  auto time_batch = [&](int reps) {
    const auto t0 = std::chrono::steady_clock::now();
    for (int r = 0; r < reps; ++r) {
      if (single) {
        greedy_generate_single(prompt, steps, -1);
      } else {
        std::vector<BatchRequest> reqs(static_cast<std::size_t>(batch),
                                       BatchRequest{prompt, steps, -1});
        run_batch(reqs);
      }
    }
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  };

  kernels::KQuantTuning t = kernels::get_kquant_tuning();
  struct Knob {
    const char* name;
    int* value;
    std::vector<int> candidates;
  };
  const std::vector<Knob> batched_knobs = {
      {"mmq_async", &t.mmq_async, {0, 1}},
      {"mmq_async_nt", &t.mmq_async_nt, {1, 2}},
      {"mm_max", &t.mm_max, {0, 4, 8, 16}},
      {"mmq_nt", &t.mmq_nt, {2, 4}},
  };
  // Warps cooperating on one row only matters to the matvec, i.e. single-stream.
  const std::vector<Knob> single_knobs = {
      {"matvec_wpr", &t.matvec_wpr, {1, 2, 4, 8}},
  };
  const std::vector<Knob>& knobs = single ? single_knobs : batched_knobs;

  std::fprintf(stderr, "[tune] k-quant knobs, %s, %d tokens per sequence\n",
               single ? "single stream" : "batched", steps);
  kernels::set_kquant_tuning(t);
  time_batch(1);  // discard: cold clocks
  for (const auto& k : knobs) {
    const int incumbent = *k.value;
    int best = incumbent;
    // Re-time the incumbent in this pass; an earlier number is a different
    // thermal state.
    kernels::set_kquant_tuning(t);
    double best_s = time_batch(2);
    for (int cand : k.candidates) {
      if (cand == incumbent) continue;
      *k.value = cand;
      kernels::set_kquant_tuning(t);
      const double s = time_batch(2);
      if (s < best_s * 0.98) {  // must beat the incumbent by more than drift
        best = cand;
        best_s = s;
      }
    }
    *k.value = best;
    kernels::set_kquant_tuning(t);
    std::fprintf(stderr, "[tune]   %-14s %d%s\n", k.name, best,
                 best == incumbent ? " (kept)" : " (changed)");
  }
  std::fprintf(stderr,
               "[tune] winners: CPI_KQUANT_MMQ_ASYNC=%d CPI_KQUANT_MMQ_ANT=%d "
               "CPI_KQUANT_MM_MAX=%d CPI_KQUANT_MMQ_NT=%d CPI_KQUANT_WPR=%d\n",
               t.mmq_async, t.mmq_async_nt, t.mm_max, t.mmq_nt, t.matvec_wpr);
}

void LlamaEngine::run_batch_bench(const std::vector<int>& prompt, int max_new) {
  if (!options_.paged_blocks) throw std::runtime_error("batch bench requires --paged-blocks");
  if (cached_layer_count_ != weights_.config().num_layers) {
    throw std::runtime_error("batch bench requires --gpu-cache-all (fully resident)");
  }
  using clock = std::chrono::steady_clock;
  const auto secs = [](clock::duration d) { return std::chrono::duration<double>(d).count(); };

  // Batch sizes to sweep; skip any whose worst-case KV would exceed the budget.
  std::vector<int> sizes = {1, 2, 4, 8, 16, 32, 48, 64};
  const int per_seq = static_cast<int>(prompt.size()) + max_new;
  const float bench_temp =
      std::getenv("CPI_BENCH_TEMP") ? std::atof(std::getenv("CPI_BENCH_TEMP")) : 0.0f;

  // Warmup (kernel autotune / first-launch costs shouldn't skew the first row).
  {
    std::vector<BatchRequest> w(1, BatchRequest{prompt, std::min(8, max_new), -1});
    run_batch(w);
  }

  std::printf(
      // The argument to --batch-bench is max_new, not a batch size: the batch
      // sizes are the fixed sweep below. At a small max_new every row is
      // dominated by its prefill and the decode numbers are meaningless -- 4
      // tokens per sequence reports roughly a quarter of what 32 does. Use 32
      // or more, and mind that B*(prompt+max_new) has to fit max_context or the
      // larger batches are silently skipped.
      "batch throughput bench: prompt=%zu max_new=%d temp=%.2f (decode tokens/sec, eos disabled)\n",
      prompt.size(), max_new, bench_temp);
  std::printf("  %4s  %14s  %14s  %8s\n", "B", "serial_tok/s", "batched_tok/s", "speedup");
  // CPI_BATCH_ONLY=N runs one batch size and skips the serial baseline. That
  // baseline is B independent single-sequence generations, so it dominates any
  // profile of this bench: a trace of the default sweep spends 40% in the
  // batch-1 matvec and 10.9% in the single-sequence paged attention, and the
  // batched kernels do not reach the top ten. Anything measuring the batched
  // step -- nsys, ncu, a debugger -- wants this on, or it is measuring the
  // baseline instead.
  static const int only_b = [] {
    const char* e = std::getenv("CPI_BATCH_ONLY");
    return e ? atoi(e) : 0;
  }();
  for (int B : sizes) {
    if (static_cast<long long>(B) * per_seq > options_.max_context) continue;
    if (only_b > 0 && B != only_b) continue;

    // Serial baseline: B independent single-sequence generations, back to back.
    double serial_s = 0.0;
    long long serial_tokens = 0;
    if (only_b <= 0) {
      const auto s0 = clock::now();
      for (int i = 0; i < B; ++i) {
        const auto out = greedy_generate_single(prompt, max_new, -1);
        serial_tokens += static_cast<long long>(out.size());
      }
      serial_s = secs(clock::now() - s0);
    }

    // Batched: all B sequences concurrently through the scheduler.
    std::vector<BatchRequest> reqs(B, BatchRequest{prompt, max_new, -1, bench_temp});
    const auto b0 = clock::now();
    const auto outs = run_batch(reqs);
    const double batched_s = secs(clock::now() - b0);
    long long batched_tokens = 0;
    for (const auto& o : outs) batched_tokens += static_cast<long long>(o.size());

    const double serial_tps = serial_s > 0 ? serial_tokens / serial_s : 0.0;
    const double batched_tps = batched_s > 0 ? batched_tokens / batched_s : 0.0;
    std::printf("  %4d  %14.1f  %14.1f  %7.2fx", B, serial_tps, batched_tps,
                serial_tps > 0 ? batched_tps / serial_tps : 0.0);
    if (std::getenv("CPI_BATCH_PROFILE")) {
      const double tot = prof_fwd_s_ + prof_head_s_;
      std::printf("   [fwd %.0f%% head %.0f%%]", tot > 0 ? 100.0 * prof_fwd_s_ / tot : 0.0,
                  tot > 0 ? 100.0 * prof_head_s_ / tot : 0.0);
      prof_fwd_s_ = 0.0;
      prof_head_s_ = 0.0;
    }
    std::printf("\n");
  }
  if (packed_matvec_calls_ > 0 || packed_matmul_calls_ > 0 || packed_matmul_declined_ > 0) {
    std::printf("  [packed] matvecs=%llu matmuls=%llu matmul_declined=%llu\n",
                static_cast<unsigned long long>(packed_matvec_calls_),
                static_cast<unsigned long long>(packed_matmul_calls_),
                static_cast<unsigned long long>(packed_matmul_declined_));
  }
}

void LlamaEngine::run_scheduler_check(const std::vector<int>& base_prompt, int max_new,
                                      int eos_id) {
  if (base_prompt.size() < 4)
    throw std::runtime_error("scheduler check needs a prompt of >= 4 tokens");
  const int L = static_cast<int>(base_prompt.size());

  // Build distinct sequences of different lengths/content so prefill lands in
  // non-contiguous blocks and sequences finish on different steps (ragged batch
  // + free-on-finish). Each gets a different max_new so completions stagger.
  std::vector<std::vector<int>> prompts = {
      base_prompt,                                                             // full
      std::vector<int>(base_prompt.begin(), base_prompt.begin() + L / 2 + 1),  // shorter
      std::vector<int>(base_prompt.begin() + 1, base_prompt.end()),            // shifted content
  };
  {
    std::vector<int> longer = base_prompt;  // longer than max prefill? no
    longer.insert(longer.end(), base_prompt.begin(), base_prompt.begin() + 3);
    prompts.push_back(std::move(longer));
  }
  std::vector<int> budgets;
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    budgets.push_back(std::max(6, max_new - static_cast<int>(i) * 7));  // staggered finishes
  }

  // Reference: each sequence alone.
  std::vector<std::vector<int>> ref(prompts.size());
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    ref[i] = greedy_generate_single(prompts[i], budgets[i], eos_id);
  }

  // Candidate: all sequences concurrently through the scheduler.
  std::vector<BatchRequest> reqs;
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    reqs.push_back(BatchRequest{prompts[i], budgets[i], eos_id});
  }
  const auto got = run_batch(reqs);

  std::printf("scheduler check: %d concurrent sequences (block_size=%d, max_context=%d)\n",
              static_cast<int>(prompts.size()),
              (options_.paged_block_size > 0 ? options_.paged_block_size : 32),
              options_.max_context);
  int fails = 0;
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    bool ok = (got[i] == ref[i]);
    // Report first divergence if any.
    int diverge = -1;
    const std::size_t n = std::min(got[i].size(), ref[i].size());
    for (std::size_t k = 0; k < n; ++k) {
      if (got[i][k] != ref[i][k]) {
        diverge = static_cast<int>(k);
        break;
      }
    }
    if (got[i].size() != ref[i].size() && diverge < 0) diverge = static_cast<int>(n);
    if (!ok) ++fails;
    std::printf("  seq %zu: prompt_len=%zu budget=%d ref_tokens=%zu batch_tokens=%zu  %s%s\n", i,
                prompts[i].size(), budgets[i], ref[i].size(), got[i].size(),
                ok ? "MATCH" : "MISMATCH",
                diverge >= 0 ? (" @" + std::to_string(diverge)).c_str() : "");
  }
  std::printf("scheduler check: %s  (%d/%zu sequences mismatched)\n", fails == 0 ? "PASS" : "FAIL",
              fails, prompts.size());
}

void LlamaEngine::ensure_batch_state_buffers(int batch, int max_blocks) {
  if (batch > batch_buffers_max_seqs_) {
    if (d_batch_positions_) cudaFree(d_batch_positions_);
    if (d_batch_seq_lens_) cudaFree(d_batch_seq_lens_);
    if (d_batch_kv_slots_) cudaFree(d_batch_kv_slots_);
    CUDA_CHECK(cudaMalloc(&d_batch_positions_, static_cast<std::size_t>(batch) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_batch_seq_lens_, static_cast<std::size_t>(batch) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_batch_kv_slots_, static_cast<std::size_t>(batch) * sizeof(int)));
    batch_buffers_max_seqs_ = batch;
  }
  // The block table is batch*max_blocks ints. Both factors vary independently and
  // the batch can regrow after shrinking, so size against the real capacity: the
  // previous product of high-watermarks over-reported it and let a larger request
  // overflow the buffer (device-side memcpy past the end).
  const std::size_t need = static_cast<std::size_t>(batch) * static_cast<std::size_t>(max_blocks);
  if (need > batch_buffers_block_table_cap_) {
    if (d_batch_block_tables_) cudaFree(d_batch_block_tables_);
    CUDA_CHECK(cudaMalloc(&d_batch_block_tables_, need * sizeof(int)));
    batch_buffers_block_table_cap_ = need;
    if (max_blocks > batch_buffers_max_blocks_) batch_buffers_max_blocks_ = max_blocks;
  }
}

// Grow the fp16 sink/ring quality-slot side buffers to at least `min_slots`
// per-sequence slots, preserving live slots' contents. Layout is
// [layers][slots][sink_n | win_n][kv_heads][head_dim], so growth is a per-layer
// strided copy (old slots block is a prefix of each layer's new region). Only
// ever called between settled steps (the previous decode/prefill synced before
// returning), so no in-flight kernel can hold the old pointers.
void LlamaEngine::ensure_kv_quality_slots(int min_slots) {
  if (min_slots <= kv_quality_slots_) return;
  if (!kv_int4_enabled_ || (d_kv_sink_k_ == nullptr && d_kv_ring_k_ == nullptr)) return;
  const auto& cfg = weights_.config();
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (cfg.hidden_size / cfg.num_heads);
  const std::size_t kv_hidden =
      static_cast<std::size_t>(cfg.num_kv_heads) * static_cast<std::size_t>(head_dim);
  int new_slots = kv_quality_slots_;
  while (new_slots < min_slots) new_slots *= 2;

  const auto grow = [&](__half*& buf, int n_tokens) {
    if (buf == nullptr || n_tokens <= 0) return;
    const std::size_t slot_elems = static_cast<std::size_t>(n_tokens) * kv_hidden;
    const std::size_t old_layer = static_cast<std::size_t>(kv_quality_slots_) * slot_elems;
    const std::size_t new_layer = static_cast<std::size_t>(new_slots) * slot_elems;
    __half* nb = nullptr;
    CUDA_CHECK(cudaMalloc(&nb, static_cast<std::size_t>(cfg.num_layers) * new_layer *
                                   sizeof(__half)));
    // No memset of the new region: reads never touch a slot position the owning
    // sequence didn't itself write (see the reset_kv_cache note in lifecycle.cpp).
    for (int l = 0; l < cfg.num_layers; ++l) {
      CUDA_CHECK(cudaMemcpy(nb + l * new_layer, buf + l * old_layer, old_layer * sizeof(__half),
                            cudaMemcpyDeviceToDevice));
    }
    cudaFree(buf);
    buf = nb;
  };
  grow(d_kv_sink_k_, kv_quant_sink_);
  grow(d_kv_sink_v_, kv_quant_sink_);
  grow(d_kv_ring_k_, kv_quant_win_);
  grow(d_kv_ring_v_, kv_quant_win_);
  if (options_.verbose) {
    std::printf("[engine] kv quality slots %d -> %d (%.1f MB)\n", kv_quality_slots_, new_slots,
                static_cast<double>(cfg.num_layers) * new_slots *
                    (kv_quant_sink_ + kv_quant_win_) * kv_hidden * 2 * sizeof(__half) /
                    (1024.0 * 1024.0));
  }
  kv_quality_slots_ = new_slots;
}

}  // namespace engine
