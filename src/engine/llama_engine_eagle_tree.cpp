// EAGLE tree drafting (CPI_EAGLE_TREE=1 on top of CPI_EAGLE=k).
//
// The chain line is capped at ~2.2 tokens/round by acceptance decay
// (69/43/48/39% down a single path); a static draft tree spends the same
// verify weight-read on branches so a wrong top-1 guess no longer ends the
// round. Tree shape (10 nodes, from the measured per-depth acceptance):
//
//   root ── n0 ── n4 ── n7 ── n9        depth: n0..n3=1, n4..n6=2,
//     │      └─ n5 ── n8                       n7,n8=3, n9=4
//     ├─ n1 ── n6
//     ├─ n2
//     └─ n3
//
// Verify row r>=1 carries node r-1; row 0 carries cur. Sibling rows share a
// rope position (base + depth), so verify K/V cannot go through the
// sequential cache store: rows land in per-layer scratch, attention takes an
// ancestor bitmask per row (launch_attention_tree_masked), and the accepted
// path's rows are scattered into the real cache after the verdict walk.
//
// Draft-side DFS needs no new attention: expanding depth-first and writing
// each node's K/V at draft-cache slot (pos-1)+depth keeps exactly the
// ancestor path live below every node's attention length; stale sibling
// entries sit beyond it and are never read. Each forward's output hidden is
// stashed per node so a later sibling can branch from its parent's feature.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "engine/llama_engine.hpp"
#include "llama_engine_internal.hpp"
#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"

namespace engine {

namespace {

constexpr int kRows = 11;   // cur + 10 tree nodes
constexpr int kNodes = 10;

// Per-row depth (row 0 = cur) and ancestor bitmask (bit j = verify row j
// visible; own bit set; row 0 visible to all).
constexpr int kRowDepth[kRows] = {0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4};
constexpr unsigned int kAncMask[kRows] = {1u,   3u,   5u,  9u,  17u, 35u,
                                          67u,  133u, 291u, 579u, 1315u};

// Verdict walk: children (verify rows) of each verify row.
constexpr int kChildRows[kRows][4] = {{1, 2, 3, 4}, {5, 6}, {7}, {}, {}, {8},
                                      {9},          {},     {},  {10}, {}};
constexpr int kChildCount[kRows] = {4, 2, 1, 0, 0, 1, 1, 0, 0, 1, 0};

// DFS draft schedule. Each forward runs one node (or the root catch-up pair)
// through the draft layer and extracts its top-n_child tokens into the
// consecutive d_eagle_dtoks_ slots first_child.. (argmax, then mask+argmax).
// stash slots: 0 = root's output hidden, node i's output at slot i+1.
struct DraftFwd {
  int stash_src;    // -1: root catch-up (feature = cur's true feature)
  int tok_node;     // node whose drafted token is the input (-1: cur)
  int depth;        // draft-cache slot = (pos-1) + depth
  int first_child;  // first child node index (consecutive)
  int n_child;
  int stash_dst;
};
constexpr DraftFwd kDraft[6] = {
    {-1, -1, 0, 0, 4, 0},  // root: top-4 -> n0..n3
    {0, 0, 1, 4, 2, 1},    // n0: top-2 -> n4, n5
    {1, 4, 2, 7, 1, 5},    // n4 -> n7
    {5, 7, 3, 9, 1, 8},    // n7 -> n9
    {1, 5, 2, 8, 1, 6},    // n5 -> n8
    {0, 1, 1, 6, 1, 2},    // n1 -> n6
};

}  // namespace

// Verify body: eagle_verify_forward with three swaps -- per-row depth rope
// offsets, K/V to per-layer scratch instead of the sequential cache store,
// and ancestor-masked attention. Same fixed shapes, graph-safe by
// construction (not captured yet; measured eager first).
void LlamaEngine::eagle_tree_verify_forward(int K) {
  const auto& cfg = weights_.config();
  const int hidden = cfg.hidden_size;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (hidden / cfg.num_heads);
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : hidden;
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const int inter = cfg.intermediate_size;
  auto s = compute_stream_;
  const std::size_t q_row_bytes = static_cast<std::size_t>(q_hidden) * sizeof(__half);
  const std::size_t kv_row_bytes = static_cast<std::size_t>(kv_hidden) * sizeof(__half);
  const std::size_t qkv_stride_bytes =
      static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * sizeof(__half);
  const std::size_t ff_row_bytes = static_cast<std::size_t>(inter) * sizeof(__half);
  const std::size_t ff13_stride_bytes = static_cast<std::size_t>(2 * inter) * sizeof(__half);
  const std::size_t scratch_layer =
      static_cast<std::size_t>(kRows) * static_cast<std::size_t>(kv_hidden);
  auto* qkv_base = static_cast<const __half*>(d_qkv_);
  auto* ff13_base = static_cast<const __half*>(d_ff13_);

  kernels::launch_embedding_lookup(static_cast<const __half*>(d_tok_embeddings_), d_token_id_,
                                   static_cast<__half*>(d_x_), K, hidden, s);
  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    const LayerDeviceWeights* lw = &layer_cache_[static_cast<std::size_t>(layer)];
    launch_norm(d_x_, lw->norm_att, lw->norm_att_bias, d_x_norm_, K, hidden);
    detail::dispatch_linear_rowmajor_weight(
        cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, s, lw->wqkv,
        d_x_norm_, d_qkv_, q_hidden + 2 * kv_hidden, hidden, K, CUDA_R_16F);
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_q_, q_row_bytes, qkv_base, qkv_stride_bytes,
                                 q_row_bytes, K, cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_k_, kv_row_bytes, qkv_base + q_hidden,
                                 qkv_stride_bytes, kv_row_bytes, K, cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_v_, kv_row_bytes, qkv_base + q_hidden + kv_hidden,
                                 qkv_stride_bytes, kv_row_bytes, K, cudaMemcpyDeviceToDevice, s));
    kernels::launch_rope_inplace_batched_offsets_device_pos(
        static_cast<__half*>(d_prefill_q_), static_cast<__half*>(d_prefill_k_), K, cfg.num_heads,
        cfg.num_kv_heads, head_dim, d_eagle_pos_, d_eagle_row_off_, d_rope_cos_, d_rope_sin_,
        q_hidden, kv_hidden, s);
    __half* k_scr = d_eagle_tree_k_ + static_cast<std::size_t>(layer) * scratch_layer;
    __half* v_scr = d_eagle_tree_v_ + static_cast<std::size_t>(layer) * scratch_layer;
    CUDA_CHECK(cudaMemcpyAsync(k_scr, d_prefill_k_, static_cast<std::size_t>(K) * kv_row_bytes,
                               cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(v_scr, d_prefill_v_, static_cast<std::size_t>(K) * kv_row_bytes,
                               cudaMemcpyDeviceToDevice, s));
    const std::size_t layer_stride =
        static_cast<std::size_t>(kv_capacity_tokens_) * static_cast<std::size_t>(kv_hidden);
    auto* k_layer =
        static_cast<const __half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride;
    auto* v_layer =
        static_cast<const __half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride;
    kernels::launch_attention_tree_split(
        static_cast<const __half*>(d_prefill_q_), k_layer, v_layer, k_scr, v_scr,
        static_cast<__half*>(d_att_), d_eagle_pos_, d_eagle_anc_mask_, K, cfg.num_heads,
        cfg.num_kv_heads, head_dim, d_eagle_mt_m_, d_eagle_mt_l_, d_eagle_mt_o_, 32,
        eagle_mt_chunks_, s);
    // A packed k-quant weight has no fp16 copy for cuBLAS; expand it into the
    // shared prefill scratch. Same stream as the GEMM, so the expansion is
    // ordered before the read and before the next weight overwrites it.
    const void* wo_src = lw->wo_packed.active()
                              ? dequant_packed_for_gemm(lw->wo_packed, s)
                              : static_cast<const void*>(lw->wo);
    detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                            lt_workspace_bytes_, s, const_cast<void*>(wo_src), d_att_, d_ff3_, hidden,
                                            q_hidden, K, CUDA_R_16F);
    kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                K * hidden, s);
    launch_norm(d_x_, lw->norm_ffn, lw->norm_ffn_bias, d_x_norm_, K, hidden);
    // A packed k-quant weight has no fp16 copy for cuBLAS; expand it into the
    // shared prefill scratch. Same stream as the GEMM, so the expansion is
    // ordered before the read and before the next weight overwrites it.
    const void* w13_src = lw->w13_packed.active()
                              ? dequant_packed_for_gemm(lw->w13_packed, s)
                              : static_cast<const void*>(lw->w13);
    detail::dispatch_linear_rowmajor_weight(
        cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, s, const_cast<void*>(w13_src),
        d_x_norm_, d_ff13_, 2 * inter, hidden, K, CUDA_R_16F);
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_ff1_, ff_row_bytes, ff13_base, ff13_stride_bytes,
                                 ff_row_bytes, K, cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_ff2_, ff_row_bytes, ff13_base + inter,
                                 ff13_stride_bytes, ff_row_bytes, K, cudaMemcpyDeviceToDevice, s));
    detail::launch_gated_glu(cfg.mlp_gelu, static_cast<const __half*>(d_prefill_ff1_),
                             static_cast<const __half*>(d_prefill_ff2_),
                             static_cast<__half*>(d_prefill_ff2_), K * inter, s);
    // A packed k-quant weight has no fp16 copy for cuBLAS; expand it into the
    // shared prefill scratch. Same stream as the GEMM, so the expansion is
    // ordered before the read and before the next weight overwrites it.
    const void* w2_src = lw->w2_packed.active()
                              ? dequant_packed_for_gemm(lw->w2_packed, s)
                              : static_cast<const void*>(lw->w2);
    detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                            lt_workspace_bytes_, s, const_cast<void*>(w2_src), d_prefill_ff2_, d_ff3_,
                                            hidden, inter, K, CUDA_R_16F);
    kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                K * hidden, s);
  }
  launch_norm(d_x_, d_norm_out_, d_norm_out_bias_, d_x_norm_, K, hidden);
  CUDA_CHECK(cudaMemcpyAsync(d_eagle_feats_, d_x_norm_,
                             static_cast<std::size_t>(K) * hidden * sizeof(__half),
                             cudaMemcpyDeviceToDevice, s));
  batched_lm_head(K, hidden, cfg.vocab_size);
  for (int i = 0; i < K; ++i) {
    kernels::launch_argmax_float(
        d_batch_logits_ + static_cast<std::size_t>(i) * static_cast<std::size_t>(cfg.vocab_size),
        cfg.vocab_size, d_eagle_verdict_ + i, s, d_argmax_part_val_, d_argmax_part_idx_,
        argmax_parts_);
  }
}

std::vector<int> LlamaEngine::eagle_tree_generate(const std::vector<int>& prompt_tokens,
                                                  int max_new_tokens,
                                                  const std::function<bool(int)>& on_token) {
  const auto& cfg = weights_.config();
  // The masked verify has no eager verify_tokens fallback; outside the
  // fp16-resident shape the chain path (which has one) takes over.
  if (cached_layer_count_ != cfg.num_layers || cached_int8_proj_enabled_ || kv_int4_enabled_ ||
      layer_cache_.empty() || layer_cache_[0].bqkv != nullptr ||
      layer_cache_[0].q_norm != nullptr) {
    return eagle_generate(prompt_tokens, max_new_tokens, on_token);
  }
  const int H = cfg.hidden_size;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (H / cfg.num_heads);
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const int max_ctx = options_.max_context;
  auto s = compute_stream_;
  std::vector<int> out;
  out.reserve(static_cast<std::size_t>(max_new_tokens));

  CUDA_CHECK(cudaMemcpyAsync(d_eagle_row_off_, kRowDepth, kRows * sizeof(int),
                             cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d_eagle_anc_mask_, kAncMask, kRows * sizeof(unsigned int),
                             cudaMemcpyHostToDevice, s));

  int pos = static_cast<int>(prompt_tokens.size()) - 1;
  int cur = prompt_tokens.back();
  std::vector<int> dummy_hist;
  int verifies = 0, drafted = 0, accepted = 0;
  int depth_try[8] = {0}, depth_ok[8] = {0};

  static const bool prof = std::getenv("CPI_EAGLE_PROF") != nullptr;
  double t_heal = 0.0, t_draft = 0.0, t_verify = 0.0, t_round = 0.0;
  double t_vfwd = 0.0, t_scatter = 0.0;  // verify sub-buckets (prof only)
  using pclock = std::chrono::steady_clock;
  const auto psec = [](pclock::time_point a, pclock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  int next = decode_next_token(cur, pos, 0.0f, dummy_hist);
  CUDA_CHECK(cudaMemcpyAsync(d_eagle_feats_, d_x_norm_,
                             static_cast<std::size_t>(H) * sizeof(__half),
                             cudaMemcpyDeviceToDevice, s));
  out.push_back(next);
  bool stop = false;
  if (on_token && !on_token(next)) stop = true;
  if (!stop && options_.eos_token_id >= 0 && next == options_.eos_token_id) stop = true;
  cur = next;
  ++pos;
  int heal_count = 0;
  int heal_tokens[8];
  int heal_feat_rows[8];  // verify row whose feature produced each healed pair
  int cur_feat_row = 0;
  int scatter_rows[8];    // function-scope: async H2D source must outlive the copy

  // Room guard: the round writes draft-cache pairs up to (pos-1)+4 and cache
  // slots up to pos+5 (acc <= 5 plus bonus).
  while (!stop && static_cast<int>(out.size()) < max_new_tokens && pos + 6 < max_ctx) {
    const auto tr0 = pclock::now();
    const int base = pos - 1 - heal_count;
    for (int j = 0; j < heal_count; ++j) {
      eagle_step(d_eagle_feats_ + static_cast<std::size_t>(heal_feat_rows[j]) * H, nullptr,
                 heal_tokens[j], base + j, /*want_token=*/false, nullptr);
    }
    auto tp = pclock::now();
    if (prof) {
      CUDA_CHECK(cudaStreamSynchronize(s));
      tp = pclock::now();
      t_heal += psec(tr0, tp);
    }

    // DFS tree draft: each forward stashes its output hidden, then top-n
    // tokens chain device-side into consecutive d_eagle_dtoks_ slots.
    for (const DraftFwd& f : kDraft) {
      const __half* feat =
          (f.stash_src < 0)
              ? d_eagle_feats_ + static_cast<std::size_t>(cur_feat_row) * H
              : d_eagle_tree_h_ + static_cast<std::size_t>(f.stash_src) * H;
      const int* tok_dev = (f.tok_node < 0) ? nullptr : d_eagle_dtoks_ + f.tok_node;
      eagle_step(feat, tok_dev, cur, (pos - 1) + f.depth, /*want_token=*/true,
                 d_eagle_dtoks_ + f.first_child);
      CUDA_CHECK(cudaMemcpyAsync(d_eagle_tree_h_ + static_cast<std::size_t>(f.stash_dst) * H,
                                 d_eagle_x_, static_cast<std::size_t>(H) * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, s));
      for (int r = 1; r < f.n_child; ++r) {
        kernels::launch_mask_logit(d_eagle_logits_, d_eagle_dtoks_ + f.first_child + r - 1, s);
        kernels::launch_argmax_float(d_eagle_logits_, cfg.vocab_size,
                                     d_eagle_dtoks_ + f.first_child + r, s, d_argmax_part_val_,
                                     d_argmax_part_idx_, argmax_parts_);
      }
    }
    int drafts[kNodes];
    CUDA_CHECK(cudaMemcpyAsync(drafts, d_eagle_dtoks_, kNodes * sizeof(int),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    drafted += kNodes;
    auto td = pclock::now();
    if (prof) t_draft += psec(tp, td);

    // Verify: row 0 = cur (host), rows 1..10 = the drafted tree (device).
    CUDA_CHECK(cudaMemcpyAsync(d_token_id_, &cur, sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_token_id_ + 1, d_eagle_dtoks_, kNodes * sizeof(int),
                               cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_eagle_pos_, &pos, sizeof(int), cudaMemcpyHostToDevice, s));
    // Capture once, replay per round (same trick as the chain's graphed
    // verify; the tree body is fixed-shape with all round-varying state read
    // from device memory, and the eligibility check at function entry already
    // guaranteed the graph-safe configuration).
    static const bool nograph = std::getenv("CPI_EAGLE_TREE_NOGRAPH") != nullptr;
    if (nograph) {
      eagle_tree_verify_forward(kRows);
    } else if (eagle_vgraph_k_ != kRows) {
      if (eagle_vgraph_exec_) {
        cudaGraphExecDestroy(eagle_vgraph_exec_);
        eagle_vgraph_exec_ = nullptr;
      }
      if (eagle_vgraph_) {
        cudaGraphDestroy(eagle_vgraph_);
        eagle_vgraph_ = nullptr;
      }
      eagle_tree_verify_forward(kRows);  // warm run (cublasLt plan cache)
      CUDA_CHECK(cudaStreamSynchronize(s));
      CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
      eagle_tree_verify_forward(kRows);
      CUDA_CHECK(cudaStreamEndCapture(s, &eagle_vgraph_));
      CUDA_CHECK(cudaGraphInstantiate(&eagle_vgraph_exec_, eagle_vgraph_, nullptr, nullptr, 0));
      eagle_vgraph_k_ = kRows;
      if (options_.verbose) {
        std::fprintf(stderr, "[eagle] tree verify graph captured (rows=%d)\n", kRows);
      }
    } else {
      CUDA_CHECK(cudaGraphLaunch(eagle_vgraph_exec_, s));
    }
    int verdict[kRows];
    CUDA_CHECK(cudaMemcpyAsync(verdict, d_eagle_verdict_, kRows * sizeof(int),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    ++verifies;
    if (prof) t_vfwd += psec(td, pclock::now());

    // Walk the tree along the target's argmaxes.
    int path[8];
    int acc = 0;
    int row = 0;
    while (acc < 8) {
      const int want = verdict[row];
      int next_row = -1;
      for (int c = 0; c < kChildCount[row]; ++c) {
        const int cr = kChildRows[row][c];
        if (drafts[cr - 1] == want) {
          next_row = cr;
          break;
        }
      }
      ++depth_try[kRowDepth[row]];
      if (next_row < 0) break;
      ++depth_ok[kRowDepth[row]];
      path[acc++] = next_row;
      row = next_row;
    }
    accepted += acc;

    // Scatter the accepted rows' K/V into the real cache at base..base+acc
    // (one launch across all layers).
    {
      scatter_rows[0] = 0;
      for (int j = 0; j < acc; ++j) scatter_rows[j + 1] = path[j];
      CUDA_CHECK(cudaMemcpyAsync(d_eagle_scatter_, scatter_rows,
                                 static_cast<std::size_t>(acc + 1) * sizeof(int),
                                 cudaMemcpyHostToDevice, s));
      const std::size_t layer_stride =
          static_cast<std::size_t>(kv_capacity_tokens_) * static_cast<std::size_t>(kv_hidden);
      kernels::launch_eagle_tree_scatter(
          d_eagle_tree_k_, d_eagle_tree_v_, static_cast<__half*>(d_k_cache_),
          static_cast<__half*>(d_v_cache_), d_eagle_scatter_, acc + 1, pos, kv_hidden, kRows,
          cfg.num_layers, layer_stride, s);
    }
    if (prof) {
      CUDA_CHECK(cudaStreamSynchronize(s));
      const auto tv = pclock::now();
      t_verify += psec(td, tv);
      t_scatter = t_verify - t_vfwd;
      t_round += psec(tr0, tv);
    }

    for (int i = 0; i < acc && !stop; ++i) {
      out.push_back(drafts[path[i] - 1]);
      if (static_cast<int>(out.size()) >= max_new_tokens) stop = true;
      if (!stop && on_token && !on_token(drafts[path[i] - 1])) stop = true;
      if (!stop && options_.eos_token_id >= 0 && drafts[path[i] - 1] == options_.eos_token_id) {
        stop = true;
      }
    }
    if (stop) break;
    const int bonus = verdict[row];
    out.push_back(bonus);
    if (static_cast<int>(out.size()) >= max_new_tokens) stop = true;
    if (!stop && on_token && !on_token(bonus)) stop = true;
    if (!stop && options_.eos_token_id >= 0 && bonus == options_.eos_token_id) stop = true;

    for (int j = 0; j < acc; ++j) {
      heal_tokens[j] = drafts[path[j] - 1];
      heal_feat_rows[j] = (j == 0) ? 0 : path[j - 1];
    }
    heal_count = acc;
    cur_feat_row = (acc > 0) ? path[acc - 1] : 0;
    cur = bonus;
    pos += acc + 1;
  }
  if (prof && verifies > 0) {
    const double n = static_cast<double>(verifies);
    std::fprintf(stderr,
                 "[eagle-prof] tree rounds=%d avg_ms: heal=%.2f draft=%.2f verify=%.2f "
                 "(fwd=%.2f scatter=%.2f) round=%.2f (tokens/round=%.2f -> %.1f tok/s in-loop)\n",
                 verifies, t_heal / n, t_draft / n, t_verify / n, t_vfwd / n, t_scatter / n,
                 t_round / n, (accepted + verifies) / n,
                 1000.0 * (accepted + verifies) / t_round);
  }
  if (std::getenv("CPI_EAGLE_STATS")) {
    std::fprintf(stderr, "[eagle] tree verifies=%d drafted=%d accepted=%d tokens=%zu\n", verifies,
                 drafted, accepted, out.size());
    for (int d = 0; d < 8; ++d) {
      if (depth_try[d]) {
        std::fprintf(stderr, "[eagle]   depth %d continue: %d/%d (%.1f%%)\n", d + 1, depth_ok[d],
                     depth_try[d], 100.0 * depth_ok[d] / depth_try[d]);
      }
    }
  }
  return out;
}

}  // namespace engine
