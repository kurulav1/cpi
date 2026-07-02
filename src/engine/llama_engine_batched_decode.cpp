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
#include "engine/llama_engine.hpp"
#include "llama_engine_internal.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"

#include "grammar/grammar_sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace engine {

// Shared batched-decode forward: embed -> layers (paged batched attention) ->
// final norm. Leaves the per-row final hidden states in d_x_norm_ [batch][hidden]
// so the LM-head tail can produce either argmax (greedy) or per-row logits (for
// per-request sampling). Returns the batch size.
int LlamaEngine::decode_step_batched_forward(const std::vector<int>& tokens,
                                             const std::vector<int>& positions,
                                             const std::vector<int>& block_tables_flat,
                                             int max_blocks) {
  const auto& cfg = weights_.config();
  const int batch = static_cast<int>(tokens.size());
  if (batch == 0) return 0;
  if (static_cast<int>(positions.size()) != batch ||
      static_cast<int>(block_tables_flat.size()) != batch * max_blocks) {
    throw std::runtime_error("decode_step_batched: ragged tokens/positions/block_tables");
  }
  if (cached_layer_count_ != cfg.num_layers) {
    throw std::runtime_error("decode_step_batched requires fully resident weights (--gpu-cache-all)");
  }
  if (!options_.paged_blocks) {
    throw std::runtime_error("decode_step_batched requires --paged-blocks");
  }

  const int hidden = cfg.hidden_size;
  const int inter = cfg.intermediate_size;
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : hidden;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (hidden / cfg.num_heads);
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const int bs = options_.paged_block_size > 0 ? options_.paged_block_size : 32;
  const std::size_t layer_stride = static_cast<std::size_t>(options_.max_context) *
                                   static_cast<std::size_t>(kv_hidden);

  // Row-strided views into the fused QKV / gate-up buffers (per-row layout,
  // identical to run_batched_chunk).
  const std::size_t q_row_bytes = static_cast<std::size_t>(q_hidden) * sizeof(__half);
  const std::size_t kv_row_bytes = static_cast<std::size_t>(kv_hidden) * sizeof(__half);
  const std::size_t ff_row_bytes = static_cast<std::size_t>(inter) * sizeof(__half);
  const std::size_t qkv_stride_bytes = static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * sizeof(__half);
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

  // Split-K scratch for the batched attention: [batch*heads*chunks] stats and
  // [batch*heads*chunks*head_dim] partial outputs. Chunks bounded by max_seq.
  const int chunks = (max_seq + bs - 1) / bs;
  const std::size_t stat_elems =
      static_cast<std::size_t>(batch) * cfg.num_heads * chunks;
  float *scratch_m = nullptr, *scratch_l = nullptr, *scratch_o = nullptr;
  CUDA_CHECK(cudaMalloc(&scratch_m, stat_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&scratch_l, stat_elems * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&scratch_o, stat_elems * head_dim * sizeof(float)));

  kernels::launch_embedding_lookup(static_cast<const __half*>(d_tok_embeddings_),
                                   d_token_id_, static_cast<__half*>(d_x_), batch, hidden,
                                   compute_stream_);

  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    const LayerDeviceWeights* lw = &layer_cache_[static_cast<std::size_t>(layer)];

    // --- Attention block ---
    launch_norm(d_x_, lw->norm_att, lw->norm_att_bias, d_x_norm_, batch, hidden);

    detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_,
                                            lt_workspace_, lt_workspace_bytes_, compute_stream_,
                                            lw->wqkv, d_x_norm_, d_qkv_,
                                            q_hidden + 2 * kv_hidden, hidden, batch, CUDA_R_16F);

    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_q_, q_row_bytes, qkv_base, qkv_stride_bytes,
                                 q_row_bytes, batch, cudaMemcpyDeviceToDevice, compute_stream_));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_k_, kv_row_bytes, qkv_base + q_hidden, qkv_stride_bytes,
                                 kv_row_bytes, batch, cudaMemcpyDeviceToDevice, compute_stream_));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_v_, kv_row_bytes, qkv_base + q_hidden + kv_hidden,
                                 qkv_stride_bytes, kv_row_bytes, batch,
                                 cudaMemcpyDeviceToDevice, compute_stream_));

    if (lw->bqkv) {
      const auto* bqkv_half = static_cast<const __half*>(lw->bqkv);
      kernels::launch_add_bias_broadcast(static_cast<__half*>(d_prefill_q_), bqkv_half,
                                         batch, q_hidden, compute_stream_);
      kernels::launch_add_bias_broadcast(static_cast<__half*>(d_prefill_k_), bqkv_half + q_hidden,
                                         batch, kv_hidden, compute_stream_);
      kernels::launch_add_bias_broadcast(static_cast<__half*>(d_prefill_v_),
                                         bqkv_half + q_hidden + kv_hidden, batch, kv_hidden,
                                         compute_stream_);
    }

    kernels::launch_rope_inplace_perpos(static_cast<__half*>(d_prefill_q_),
                                        static_cast<__half*>(d_prefill_k_), batch,
                                        cfg.num_heads, cfg.num_kv_heads, head_dim,
                                        d_batch_positions_, d_rope_cos_, d_rope_sin_,
                                        compute_stream_);

    auto* k_pool = static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride;
    auto* v_pool = static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride;
    kernels::launch_store_kv_batched_paged(k_pool, v_pool,
                                           static_cast<const __half*>(d_prefill_k_),
                                           static_cast<const __half*>(d_prefill_v_),
                                           d_batch_block_tables_, d_batch_positions_,
                                           max_blocks, batch, kv_hidden, bs, compute_stream_);

    kernels::launch_attention_step_batched_paged(static_cast<const __half*>(d_prefill_q_),
                                                 k_pool, v_pool, d_batch_block_tables_,
                                                 d_batch_seq_lens_, max_blocks, max_seq,
                                                 static_cast<__half*>(d_att_), batch,
                                                 cfg.num_heads, cfg.num_kv_heads, head_dim, bs,
                                                 compute_stream_, scratch_m, scratch_l, scratch_o,
                                                 chunks);

    detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_,
                                            lt_workspace_, lt_workspace_bytes_, compute_stream_,
                                            lw->wo, d_att_, d_ff3_, hidden, q_hidden, batch,
                                            CUDA_R_16F);
    maybe_add_half_bias(d_ff3_, lw->bo, batch, hidden);
    kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                batch * hidden, compute_stream_);

    // --- FFN block ---
    launch_norm(d_x_, lw->norm_ffn, lw->norm_ffn_bias, d_x_norm_, batch, hidden);

    detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_,
                                            lt_workspace_, lt_workspace_bytes_, compute_stream_,
                                            lw->w13, d_x_norm_, d_ff13_, 2 * inter, hidden, batch,
                                            CUDA_R_16F);
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_ff1_, ff_row_bytes, ff13_base, ff13_stride_bytes,
                                 ff_row_bytes, batch, cudaMemcpyDeviceToDevice, compute_stream_));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_ff2_, ff_row_bytes, ff13_base + inter, ff13_stride_bytes,
                                 ff_row_bytes, batch, cudaMemcpyDeviceToDevice, compute_stream_));
    kernels::launch_silu_mul(static_cast<const __half*>(d_prefill_ff1_),
                             static_cast<const __half*>(d_prefill_ff2_),
                             static_cast<__half*>(d_prefill_ff2_), batch * inter, compute_stream_);
    detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_,
                                            lt_workspace_, lt_workspace_bytes_, compute_stream_,
                                            lw->w2, d_prefill_ff2_, d_ff3_, hidden, inter, batch,
                                            CUDA_R_16F);
    kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                batch * hidden, compute_stream_);
  }

  launch_norm(d_x_, d_norm_out_, d_norm_out_bias_, d_x_norm_, batch, hidden);

  cudaFree(scratch_m);
  cudaFree(scratch_l);
  cudaFree(scratch_o);
  return batch;
}

// Project one row's final hidden state (d_x_norm_ + b*hidden) through the LM head
// into d_logits_ (float, +optional bias). Caller has run the forward.
void LlamaEngine::batched_lm_head_row(int b, int hidden) {
  const auto& cfg = weights_.config();
  const auto* x_row = static_cast<const __half*>(d_x_norm_) + static_cast<std::size_t>(b) * hidden;
  resident_projection_float(d_lm_head_, x_row, d_logits_, cfg.vocab_size, hidden,
                            resident_lm_head_warps_, resident_lm_head_tile_pairs_,
                            resident_lm_head_rows_per_warp_);
  if (d_lm_head_bias_) {
    kernels::launch_add_bias_inplace_float_from_half(static_cast<float*>(d_logits_),
                                                     static_cast<const __half*>(d_lm_head_bias_),
                                                     cfg.vocab_size, compute_stream_);
  }
}

std::vector<int> LlamaEngine::decode_step_batched(const std::vector<int>& tokens,
                                                  const std::vector<int>& positions,
                                                  const std::vector<int>& block_tables_flat,
                                                  int max_blocks) {
  const int batch = decode_step_batched_forward(tokens, positions, block_tables_flat, max_blocks);
  if (batch == 0) return {};
  const int hidden = weights_.config().hidden_size;
  const int vocab = weights_.config().vocab_size;
  std::vector<int> next(batch);
  for (int b = 0; b < batch; ++b) {
    batched_lm_head_row(b, hidden);
    kernels::launch_argmax_float(static_cast<const float*>(d_logits_), vocab, d_argmax_, compute_stream_);
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    CUDA_CHECK(cudaMemcpy(&next[b], d_argmax_, sizeof(int), cudaMemcpyDeviceToHost));
  }
  return next;
}

void LlamaEngine::decode_step_batched_logits(const std::vector<int>& tokens,
                                             const std::vector<int>& positions,
                                             const std::vector<int>& block_tables_flat,
                                             int max_blocks,
                                             std::vector<std::vector<float>>& out_logits) {
  const int batch = decode_step_batched_forward(tokens, positions, block_tables_flat, max_blocks);
  out_logits.assign(static_cast<std::size_t>(batch), {});
  if (batch == 0) return;
  const int hidden = weights_.config().hidden_size;
  const int vocab = weights_.config().vocab_size;
  for (int b = 0; b < batch; ++b) {
    batched_lm_head_row(b, hidden);
    out_logits[b].resize(static_cast<std::size_t>(vocab));
    CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
    CUDA_CHECK(cudaMemcpy(out_logits[b].data(), d_logits_,
                          static_cast<std::size_t>(vocab) * sizeof(float), cudaMemcpyDeviceToHost));
  }
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
  std::printf("batched-decode check: prompt=%d steps=%d block_size=%d blocks=%d\n",
              seq_len, steps, bs, nblk);
  std::vector<int> dup_table = block_table_host_;
  dup_table.insert(dup_table.end(), block_table_host_.begin(), block_table_host_.end());  // [2][nblk]

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
      std::printf("  step %2d pos %4d  single=%d  batched[N=1]=%d %s  batched[N=2]=%d,%d %s\n",
                  s, pos, single_arg, n1[0], ok1 ? "ok" : "MISMATCH", n2[0], n2[1],
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
  for (int c = 0; c < nblk; ++c) block_table_host_[static_cast<std::size_t>(c)] = seq_blocks_->block_for(c * bs);
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

void LlamaEngine::stream_grow_table(StreamSeq& s, int upto_pos) {
  const int bs = options_.paged_block_size > 0 ? options_.paged_block_size : 32;
  if (!s.blocks->ensure_position(upto_pos)) {
    throw std::runtime_error("stream scheduler: paged KV pool exhausted (concurrent length > max_context budget)");
  }
  const int nblk = upto_pos / bs + 1;
  if (static_cast<int>(s.table.size()) != nblk) {
    s.table.resize(static_cast<std::size_t>(nblk));
    for (int c = 0; c < nblk; ++c) s.table[static_cast<std::size_t>(c)] = s.blocks->block_for(c * bs);
  }
}

int LlamaEngine::stream_active() const { return static_cast<int>(stream_seqs_.size()); }

void LlamaEngine::stream_admit(const std::string& id, const std::vector<int>& prompt_tokens,
                               const StreamParams& params) {
  if (!options_.paged_blocks || !block_alloc_) {
    throw std::runtime_error("stream_admit requires --paged-blocks");
  }
  if (cached_layer_count_ != weights_.config().num_layers) {
    throw std::runtime_error("stream_admit requires --gpu-cache-all (fully resident)");
  }
  if (prompt_tokens.empty()) throw std::runtime_error("stream_admit: empty prompt");
  const int bs = options_.paged_block_size > 0 ? options_.paged_block_size : 32;

  StreamSeq s;
  s.id = id;
  s.blocks = std::make_unique<SequenceBlockTable>(block_alloc_.get(), bs);
  s.history = prompt_tokens;
  s.last_token = prompt_tokens.back();
  s.pos = static_cast<int>(prompt_tokens.size()) - 1;
  s.params = params;
  stream_grow_table(s, s.pos);

  // Prefill this request's prompt into its own blocks. d_block_table_ is shared
  // engine state and fully-cached prefill runs async on compute_stream_, so sync
  // before returning (a later admit / the decode loop must not clobber it).
  block_table_host_ = s.table;
  CUDA_CHECK(cudaMemcpy(d_block_table_, s.table.data(), s.table.size() * sizeof(int),
                        cudaMemcpyHostToDevice));
  prefill_prompt(prompt_tokens);
  CUDA_CHECK(cudaStreamSynchronize(compute_stream_));
  stream_seqs_.push_back(std::move(s));
}

bool LlamaEngine::stream_step(std::vector<StreamEvent>& events) {
  events.clear();
  const int B = static_cast<int>(stream_seqs_.size());
  if (B == 0) return false;

  // Assemble the running batch (grow each table to cover the position we write).
  int max_blocks = 0;
  for (auto& s : stream_seqs_) {
    stream_grow_table(s, s.pos);
    max_blocks = std::max(max_blocks, static_cast<int>(s.table.size()));
  }
  std::vector<int> toks(B), poss(B), flat(static_cast<std::size_t>(B) * max_blocks, 0);
  for (int b = 0; b < B; ++b) {
    toks[b] = stream_seqs_[b].last_token;
    poss[b] = stream_seqs_[b].pos;
    const auto& t = stream_seqs_[b].table;
    for (std::size_t c = 0; c < t.size(); ++c) flat[static_cast<std::size_t>(b) * max_blocks + c] = t[c];
  }

  std::vector<std::vector<float>> logits;
  decode_step_batched_logits(toks, poss, flat, max_blocks, logits);

  // Sample each row with its own params/grammar, then retire finished requests.
  std::vector<int> finished;  // indices into stream_seqs_
  for (int b = 0; b < B; ++b) {
    StreamSeq& s = stream_seqs_[b];
    std::vector<float>& lg = logits[b];

    if (s.params.grammar) s.params.grammar->apply_mask(lg);
    // Suppress EOS until min_new_tokens (skip when a grammar drives termination).
    if (s.generated < s.params.min_new_tokens && !s.params.grammar &&
        options_.eos_token_id >= 0 &&
        static_cast<std::size_t>(options_.eos_token_id) < lg.size()) {
      lg[static_cast<std::size_t>(options_.eos_token_id)] = -std::numeric_limits<float>::infinity();
    }

    const int tok = detail::dispatch_sample_from_logits(lg, s.params.temperature, options_.top_k,
                                                        options_.top_p, options_.repetition_penalty,
                                                        options_.no_repeat_ngram_size, s.history);
    if (s.params.grammar) s.params.grammar->accept(tok);
    s.history.push_back(tok);
    s.last_token = tok;
    ++s.pos;
    ++s.generated;

    const bool is_stop =
        std::find(s.params.stop_ids.begin(), s.params.stop_ids.end(), tok) != s.params.stop_ids.end();
    const char* reason = "";
    bool fin = false;
    if (is_stop) {
      fin = true;
      reason = (tok == options_.eos_token_id) ? "eos" : "stop";
    } else if (s.generated >= s.params.max_new_tokens || s.pos >= options_.max_context) {
      fin = true;
      reason = "length";
    }
    events.push_back(StreamEvent{s.id, tok, fin, reason});
    if (fin) finished.push_back(b);
  }

  // Free finished requests' blocks and remove them (iterate high->low to keep
  // indices valid). RAII on the unique_ptr releases blocks to the pool.
  for (auto it = finished.rbegin(); it != finished.rend(); ++it) {
    stream_seqs_.erase(stream_seqs_.begin() + *it);
  }
  return true;
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
    p.temperature = 0.0f;  // greedy
    if (requests[i].eos_id >= 0) p.stop_ids.push_back(requests[i].eos_id);
    const std::string id = std::to_string(i);
    id_to_index[id] = i;
    stream_admit(id, requests[i].prompt, p);
  }

  std::vector<StreamEvent> events;
  while (stream_step(events)) {
    for (const auto& e : events) outs[id_to_index[e.id]].push_back(e.token);
  }
  return outs;
}

void LlamaEngine::run_batch_bench(const std::vector<int>& prompt, int max_new) {
  if (!options_.paged_blocks) throw std::runtime_error("batch bench requires --paged-blocks");
  if (cached_layer_count_ != weights_.config().num_layers) {
    throw std::runtime_error("batch bench requires --gpu-cache-all (fully resident)");
  }
  using clock = std::chrono::steady_clock;
  const auto secs = [](clock::duration d) {
    return std::chrono::duration<double>(d).count();
  };

  // Batch sizes to sweep; skip any whose worst-case KV would exceed the budget.
  std::vector<int> sizes = {1, 2, 4, 8, 16};
  const int per_seq = static_cast<int>(prompt.size()) + max_new;

  // Warmup (kernel autotune / first-launch costs shouldn't skew the first row).
  { std::vector<BatchRequest> w(1, BatchRequest{prompt, std::min(8, max_new), -1}); run_batch(w); }

  std::printf("batch throughput bench: prompt=%zu max_new=%d (decode tokens/sec, eos disabled)\n",
              prompt.size(), max_new);
  std::printf("  %4s  %14s  %14s  %8s\n", "B", "serial_tok/s", "batched_tok/s", "speedup");
  for (int B : sizes) {
    if (static_cast<long long>(B) * per_seq > options_.max_context) continue;

    // Serial baseline: B independent single-sequence generations, back to back.
    const auto s0 = clock::now();
    long long serial_tokens = 0;
    for (int i = 0; i < B; ++i) {
      const auto out = greedy_generate_single(prompt, max_new, -1);
      serial_tokens += static_cast<long long>(out.size());
    }
    const double serial_s = secs(clock::now() - s0);

    // Batched: all B sequences concurrently through the scheduler.
    std::vector<BatchRequest> reqs(B, BatchRequest{prompt, max_new, -1});
    const auto b0 = clock::now();
    const auto outs = run_batch(reqs);
    const double batched_s = secs(clock::now() - b0);
    long long batched_tokens = 0;
    for (const auto& o : outs) batched_tokens += static_cast<long long>(o.size());

    const double serial_tps = serial_s > 0 ? serial_tokens / serial_s : 0.0;
    const double batched_tps = batched_s > 0 ? batched_tokens / batched_s : 0.0;
    std::printf("  %4d  %14.1f  %14.1f  %7.2fx\n", B, serial_tps, batched_tps,
                serial_tps > 0 ? batched_tps / serial_tps : 0.0);
  }
}

void LlamaEngine::run_scheduler_check(const std::vector<int>& base_prompt, int max_new, int eos_id) {
  if (base_prompt.size() < 4) throw std::runtime_error("scheduler check needs a prompt of >= 4 tokens");
  const int L = static_cast<int>(base_prompt.size());

  // Build distinct sequences of different lengths/content so prefill lands in
  // non-contiguous blocks and sequences finish on different steps (ragged batch
  // + free-on-finish). Each gets a different max_new so completions stagger.
  std::vector<std::vector<int>> prompts = {
      base_prompt,                                                  // full
      std::vector<int>(base_prompt.begin(), base_prompt.begin() + L / 2 + 1),  // shorter
      std::vector<int>(base_prompt.begin() + 1, base_prompt.end()),           // shifted content
  };
  {
    std::vector<int> longer = base_prompt;                          // longer than max prefill? no
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
              static_cast<int>(prompts.size()), (options_.paged_block_size > 0 ? options_.paged_block_size : 32),
              options_.max_context);
  int fails = 0;
  for (std::size_t i = 0; i < prompts.size(); ++i) {
    bool ok = (got[i] == ref[i]);
    // Report first divergence if any.
    int diverge = -1;
    const std::size_t n = std::min(got[i].size(), ref[i].size());
    for (std::size_t k = 0; k < n; ++k) {
      if (got[i][k] != ref[i][k]) { diverge = static_cast<int>(k); break; }
    }
    if (got[i].size() != ref[i].size() && diverge < 0) diverge = static_cast<int>(n);
    if (!ok) ++fails;
    std::printf("  seq %zu: prompt_len=%zu budget=%d ref_tokens=%zu batch_tokens=%zu  %s%s\n",
                i, prompts[i].size(), budgets[i], ref[i].size(), got[i].size(),
                ok ? "MATCH" : "MISMATCH",
                diverge >= 0 ? (" @" + std::to_string(diverge)).c_str() : "");
  }
  std::printf("scheduler check: %s  (%d/%zu sequences mismatched)\n",
              fails == 0 ? "PASS" : "FAIL", fails, prompts.size());
}

void LlamaEngine::ensure_batch_state_buffers(int batch, int max_blocks) {
  if (batch > batch_buffers_max_seqs_) {
    if (d_batch_positions_) cudaFree(d_batch_positions_);
    if (d_batch_seq_lens_) cudaFree(d_batch_seq_lens_);
    CUDA_CHECK(cudaMalloc(&d_batch_positions_, static_cast<std::size_t>(batch) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_batch_seq_lens_, static_cast<std::size_t>(batch) * sizeof(int)));
    batch_buffers_max_seqs_ = batch;
    batch_buffers_max_blocks_ = 0;  // force block-table realloc below
  }
  if (batch * max_blocks > batch_buffers_max_seqs_ * batch_buffers_max_blocks_ ||
      max_blocks > batch_buffers_max_blocks_) {
    if (d_batch_block_tables_) cudaFree(d_batch_block_tables_);
    CUDA_CHECK(cudaMalloc(&d_batch_block_tables_,
                          static_cast<std::size_t>(batch) * max_blocks * sizeof(int)));
    batch_buffers_max_blocks_ = max_blocks;
  }
}

}  // namespace engine
