// EAGLE tree drafting (CPI_EAGLE_TREE=1 on top of CPI_EAGLE=k).
//
// The chain line is capped at ~2.2 tokens/round by acceptance decay
// (69/43/48/39% down a single path); a static draft tree spends the same
// verify weight-read on branches so a wrong top-1 guess no longer ends the
// round.
//
// Draft side (v2): one BATCHED forward per tree depth instead of one DFS
// forward per node. Level rows share a rope position (base + depth), their
// K/V go to a per-node scratch in forward order, and attention is the same
// ancestor-bitmask tree kernel the verify uses (cache prefix at constant
// length base, in-batch scratch columns as one virtual chunk). That drops
// the draft from one weight-read per expanded node to one per depth, which
// is what makes wider trees affordable.
//
// Verify side: rows land in per-layer scratch (siblings share a position so
// the sequential cache store cannot hold them), attention takes an ancestor
// bitmask per row, and the accepted path's rows are scattered into the real
// cache after the verdict walk.
//
// Two static shapes, chosen by CPI_EAGLE_TREE_SHAPE (16 default, 10 = the v1
// shape). Shape 16 (from the measured per-depth continue rates 90/66/54%):
//
//   root ── n0 ── n4 ── n11 ── n15      root top-4, n0 top-3, n1 top-2,
//     │      │      └─ n12              n2/n3 top-1, n4 top-2, n5/n7 top-1,
//     │      ├─ n5 ── n13              n11 top-1. 16 nodes, 17 verify rows,
//     │      └─ n6                      level widths 4/3/1.
//     ├─ n1 ── n7 ── n14
//     │      └─ n8
//     ├─ n2 ── n9
//     └─ n3 ── n10
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

// A static tree shape: verify tables (rows = cur + nodes, node i on row i+1)
// plus the batched draft-level tables. Level rows are concatenated across
// levels in forward order; a row's draft-scratch column and its stash slot
// (1 + index; slot 0 is the root's output) both equal its concatenated index.
struct TreeShape {
  int rows;  // verify rows (1 + nodes)
  int nodes;
  int root_children;             // root's top-n (dtok slots 0..n-1)
  const int* row_depth;          // [rows] verify row depth (row 0 = 0)
  const unsigned int* anc_mask;  // [rows] verify ancestor bitmask (self incl)
  const int (*child_rows)[4];    // [rows] verdict-walk children (verify rows)
  const int* child_count;        // [rows]
  int n_levels;                  // batched draft levels (depth 1..n_levels)
  const int* lvl_b;              // [n_levels] level width
  int total_lvl_rows;
  const int* lvl_tok;            // [total] dtok slot of the row's input token
  const int* lvl_feat;           // [total] stash row of the row's input feature
  const unsigned int* lvl_mask;  // [total] draft-scratch ancestor mask (self incl)
  const int* lvl_dep;            // [total] row depth (rope offset)
  const int (*lvl_child)[4];     // [total] children dtok slots
  const int* lvl_child_n;        // [total]
};

// Shape 10 (v1): root top-4, n0 top-2, n1/n4/n5/n7 top-1. Levels 2/2/1.
constexpr int k10RowDepth[11] = {0, 1, 1, 1, 1, 2, 2, 2, 3, 3, 4};
constexpr unsigned int k10AncMask[11] = {1u, 3u, 5u, 9u, 17u, 35u, 67u, 133u, 291u, 579u, 1315u};
constexpr int k10ChildRows[11][4] = {{1, 2, 3, 4}, {5, 6}, {7}, {}, {}, {8}, {9}, {}, {}, {10}, {}};
constexpr int k10ChildCount[11] = {4, 2, 1, 0, 0, 1, 1, 0, 0, 1, 0};
constexpr int k10LvlB[3] = {2, 2, 1};
// Level rows: n0, n1 | n4, n5 | n7.
constexpr int k10LvlTok[5] = {0, 1, 4, 5, 7};
constexpr int k10LvlFeat[5] = {0, 0, 1, 1, 3};
constexpr unsigned int k10LvlMask[5] = {1u, 2u, 5u, 9u, 21u};
constexpr int k10LvlDep[5] = {1, 1, 2, 2, 3};
constexpr int k10LvlChild[5][4] = {{4, 5}, {6}, {7}, {8}, {9}};
constexpr int k10LvlChildN[5] = {2, 1, 1, 1, 1};

// Shape 16: node order n0..n3 (depth 1), n4..n6 from n0, n7/n8 from n1, n9
// from n2, n10 from n3 (depth 2), n11/n12 from n4, n13 from n5, n14 from n7
// (depth 3), n15 from n11 (depth 4).
constexpr int k16RowDepth[17] = {0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 4};
constexpr unsigned int k16AncMask[17] = {1u,    3u,    5u,     9u,     17u,   35u,
                                         67u,   131u,  261u,   517u,   1033u, 2065u,
                                         4131u, 8227u, 16451u, 33029u, 69667u};
constexpr int k16ChildRows[17][4] = {{1, 2, 3, 4}, {5, 6, 7}, {8, 9}, {10}, {11}, {12, 13},
                                     {14},         {},        {15},   {},   {},   {},
                                     {16},         {},        {},     {},   {}};
constexpr int k16ChildCount[17] = {4, 3, 2, 1, 1, 2, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0};
constexpr int k16LvlB[3] = {4, 3, 1};
// Level rows: n0, n1, n2, n3 | n4, n5, n7 | n11.
constexpr int k16LvlTok[8] = {0, 1, 2, 3, 4, 5, 7, 11};
constexpr int k16LvlFeat[8] = {0, 0, 0, 0, 1, 1, 2, 5};
constexpr unsigned int k16LvlMask[8] = {1u, 2u, 4u, 8u, 17u, 33u, 66u, 145u};
constexpr int k16LvlDep[8] = {1, 1, 1, 1, 2, 2, 2, 3};
constexpr int k16LvlChild[8][4] = {{4, 5, 6}, {7, 8}, {9}, {10}, {11, 12}, {13}, {14}, {15}};
constexpr int k16LvlChildN[8] = {3, 2, 1, 1, 2, 1, 1, 1};

constexpr TreeShape kShape10 = {
    11,      10, 4,         k10RowDepth, k10AncMask, k10ChildRows, k10ChildCount, 3,
    k10LvlB, 5,  k10LvlTok, k10LvlFeat,  k10LvlMask, k10LvlDep,    k10LvlChild,   k10LvlChildN};
constexpr TreeShape kShape16 = {
    17,      16, 4,         k16RowDepth, k16AncMask, k16ChildRows, k16ChildCount, 3,
    k16LvlB, 8,  k16LvlTok, k16LvlFeat,  k16LvlMask, k16LvlDep,    k16LvlChild,   k16LvlChildN};

const TreeShape& tree_shape() {
  static const TreeShape* shape = [] {
    const char* env = std::getenv("CPI_EAGLE_TREE_SHAPE");
    if (env && std::atoi(env) == 10) return &kShape10;
    return &kShape16;
  }();
  return *shape;
}

}  // namespace

// One batched draft-level forward: rows row0..row0+B-1 of the shape's
// concatenated level tables, all at the same depth. Input tokens come from
// d_eagle_dtoks_ via lvl_tok, features from the stash via lvl_feat; K/V land
// at draft-scratch columns row0..row0+B-1 (their concatenated indices), and
// each row's top-n children go to their static dtok slots (the caller drives
// that part; this function leaves per-row logits in d_batch_logits_ and the
// output hiddens in the stash at rows 1+row0..).
void LlamaEngine::eagle_tree_level(int B, int row0, int n_scr) {
  const auto& cfg = weights_.config();
  const int H = cfg.hidden_size;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (H / cfg.num_heads);
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : H;
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const int inter = cfg.intermediate_size;
  auto s = compute_stream_;

  kernels::launch_eagle_cat_gather(static_cast<const __half*>(d_tok_embeddings_), d_eagle_dtoks_,
                                   d_eagle_lvl_tok_ + row0, d_eagle_lvl_feat_ + row0,
                                   d_eagle_tree_h_, d_eagle_bcat_, B, H, s);
  detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                          lt_workspace_bytes_, s, d_eagle_fc_, d_eagle_bcat_,
                                          d_eagle_bx_, H, 2 * H, B, CUDA_R_16F);
  // No pre-attention norm (EAGLE removes layer 0's input_layernorm).
  detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                          lt_workspace_bytes_, s, d_eagle_wq_, d_eagle_bx_,
                                          d_eagle_bq_, q_hidden, H, B, CUDA_R_16F);
  __half* k_dst = d_eagle_scrk_ + static_cast<std::size_t>(row0) * kv_hidden;
  __half* v_dst = d_eagle_scrv_ + static_cast<std::size_t>(row0) * kv_hidden;
  detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                          lt_workspace_bytes_, s, d_eagle_wk_, d_eagle_bx_, k_dst,
                                          kv_hidden, H, B, CUDA_R_16F);
  detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                          lt_workspace_bytes_, s, d_eagle_wv_, d_eagle_bx_, v_dst,
                                          kv_hidden, H, B, CUDA_R_16F);
  kernels::launch_rope_inplace_batched_offsets_device_pos(
      d_eagle_bq_, k_dst, B, cfg.num_heads, cfg.num_kv_heads, head_dim, d_eagle_pos_,
      d_eagle_lvl_dep_ + row0, d_rope_cos_, d_rope_sin_, q_hidden, kv_hidden, s);
  kernels::launch_attention_tree_split(
      d_eagle_bq_, d_eagle_kcache_, d_eagle_vcache_, d_eagle_scrk_, d_eagle_scrv_, d_eagle_batt_,
      d_eagle_pos_, d_eagle_lvl_mask_ + row0, B, n_scr, cfg.num_heads, cfg.num_kv_heads, head_dim,
      d_eagle_mt_m_, d_eagle_mt_l_, d_eagle_mt_o_, 32, eagle_mt_chunks_, s);
  detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                          lt_workspace_bytes_, s, d_eagle_wo_, d_eagle_batt_,
                                          d_eagle_btmp_, H, q_hidden, B, CUDA_R_16F);
  kernels::launch_add_inplace(d_eagle_bx_, d_eagle_btmp_, B * H, s);
  kernels::launch_rmsnorm(d_eagle_bx_, static_cast<const __half*>(d_eagle_pnorm_), d_eagle_bnorm_,
                          B, H, cfg.norm_eps, s);
  detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                          lt_workspace_bytes_, s, d_eagle_w1_, d_eagle_bnorm_,
                                          d_eagle_bgate_, inter, H, B, CUDA_R_16F);
  detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                          lt_workspace_bytes_, s, d_eagle_w3_, d_eagle_bnorm_,
                                          d_eagle_bup_, inter, H, B, CUDA_R_16F);
  kernels::launch_silu_mul(d_eagle_bgate_, d_eagle_bup_, d_eagle_bgate_, B * inter, s);
  detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                          lt_workspace_bytes_, s, d_eagle_w2_, d_eagle_bgate_,
                                          d_eagle_btmp_, H, inter, B, CUDA_R_16F);
  kernels::launch_add_inplace(d_eagle_bx_, d_eagle_btmp_, B * H, s);
  // Stash the level's output hiddens (slot = 1 + concatenated row index).
  CUDA_CHECK(cudaMemcpyAsync(d_eagle_tree_h_ + static_cast<std::size_t>(1 + row0) * H, d_eagle_bx_,
                             static_cast<std::size_t>(B) * H * sizeof(__half),
                             cudaMemcpyDeviceToDevice, s));
  // Shared lm_head over the RAW draft hiddens, one batched pass; per-row
  // logits land in d_batch_logits_ for the caller's top-n extraction.
  if (B > d_batch_logits_cap_) {
    if (d_batch_logits_) cudaFree(d_batch_logits_);
    CUDA_CHECK(
        cudaMalloc(&d_batch_logits_, static_cast<std::size_t>(B) * cfg.vocab_size * sizeof(float)));
    d_batch_logits_cap_ = B;
  }
  const void* head_src = lm_head_packed_.active() ? dequant_packed_for_gemm(lm_head_packed_, s)
                                                  : static_cast<const void*>(d_lm_head_);
  detail::dispatch_linear_rowmajor_weight(
      cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, s,
      const_cast<void*>(head_src), d_eagle_bx_, d_batch_logits_, cfg.vocab_size, H, B, CUDA_R_32F);
}

// Verify body: eagle_verify_forward with three swaps (per-row depth rope
// offsets, K/V to per-layer scratch instead of the sequential cache store,
// and ancestor-masked attention). Same fixed shapes, graph-safe by
// construction.
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
      static_cast<std::size_t>(K) * static_cast<std::size_t>(kv_hidden);
  auto* qkv_base = static_cast<const __half*>(d_qkv_);
  auto* ff13_base = static_cast<const __half*>(d_ff13_);

  kernels::launch_embedding_lookup(static_cast<const __half*>(d_tok_embeddings_), d_token_id_,
                                   static_cast<__half*>(d_x_), K, hidden, s);
  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    const LayerDeviceWeights* lw = &layer_cache_[static_cast<std::size_t>(layer)];
    launch_norm(d_x_, lw->norm_att, lw->norm_att_bias, d_x_norm_, K, hidden);
    const void* qkv_src = dequant_packed_qkv_for_gemm(*lw, s);
    if (qkv_src == nullptr) qkv_src = lw->wqkv;
    detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                            lt_workspace_bytes_, s, const_cast<void*>(qkv_src),
                                            d_x_norm_, d_qkv_, q_hidden + 2 * kv_hidden, hidden, K,
                                            CUDA_R_16F);
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_q_, q_row_bytes, qkv_base, qkv_stride_bytes, q_row_bytes,
                                 K, cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_k_, kv_row_bytes, qkv_base + q_hidden, qkv_stride_bytes,
                                 kv_row_bytes, K, cudaMemcpyDeviceToDevice, s));
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
    kernels::launch_attention_tree_split(static_cast<const __half*>(d_prefill_q_), k_layer, v_layer,
                                         k_scr, v_scr, static_cast<__half*>(d_att_), d_eagle_pos_,
                                         d_eagle_anc_mask_, K, K, cfg.num_heads, cfg.num_kv_heads,
                                         head_dim, d_eagle_mt_m_, d_eagle_mt_l_, d_eagle_mt_o_, 32,
                                         eagle_mt_chunks_, s);
    // A packed k-quant weight has no fp16 copy for cuBLAS; expand it into the
    // shared prefill scratch. Same stream as the GEMM, so the expansion is
    // ordered before the read and before the next weight overwrites it.
    const void* wo_src = lw->wo_packed.active() ? dequant_packed_for_gemm(lw->wo_packed, s)
                                                : static_cast<const void*>(lw->wo);
    detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                            lt_workspace_bytes_, s, const_cast<void*>(wo_src),
                                            d_att_, d_ff3_, hidden, q_hidden, K, CUDA_R_16F);
    kernels::launch_add_inplace(static_cast<__half*>(d_x_), static_cast<const __half*>(d_ff3_),
                                K * hidden, s);
    launch_norm(d_x_, lw->norm_ffn, lw->norm_ffn_bias, d_x_norm_, K, hidden);
    // A packed k-quant weight has no fp16 copy for cuBLAS; expand it into the
    // shared prefill scratch. Same stream as the GEMM, so the expansion is
    // ordered before the read and before the next weight overwrites it.
    const void* w13_src = lw->w13_packed.active() ? dequant_packed_for_gemm(lw->w13_packed, s)
                                                  : static_cast<const void*>(lw->w13);
    detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                            lt_workspace_bytes_, s, const_cast<void*>(w13_src),
                                            d_x_norm_, d_ff13_, 2 * inter, hidden, K, CUDA_R_16F);
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_ff1_, ff_row_bytes, ff13_base, ff13_stride_bytes,
                                 ff_row_bytes, K, cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_ff2_, ff_row_bytes, ff13_base + inter, ff13_stride_bytes,
                                 ff_row_bytes, K, cudaMemcpyDeviceToDevice, s));
    detail::launch_gated_glu(cfg.mlp_gelu, static_cast<const __half*>(d_prefill_ff1_),
                             static_cast<const __half*>(d_prefill_ff2_),
                             static_cast<__half*>(d_prefill_ff2_), K * inter, s);
    // A packed k-quant weight has no fp16 copy for cuBLAS; expand it into the
    // shared prefill scratch. Same stream as the GEMM, so the expansion is
    // ordered before the read and before the next weight overwrites it.
    const void* w2_src = lw->w2_packed.active() ? dequant_packed_for_gemm(lw->w2_packed, s)
                                                : static_cast<const void*>(lw->w2);
    detail::dispatch_linear_rowmajor_weight(cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_,
                                            lt_workspace_bytes_, s, const_cast<void*>(w2_src),
                                            d_prefill_ff2_, d_ff3_, hidden, inter, K, CUDA_R_16F);
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
  const TreeShape& shape = tree_shape();
  const int kRows = shape.rows;
  const int kNodes = shape.nodes;
  const int H = cfg.hidden_size;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (H / cfg.num_heads);
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const int max_ctx = options_.max_context;
  auto s = compute_stream_;
  std::vector<int> out;
  out.reserve(static_cast<std::size_t>(max_new_tokens));

  CUDA_CHECK(cudaMemcpyAsync(d_eagle_row_off_, shape.row_depth, kRows * sizeof(int),
                             cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d_eagle_anc_mask_, shape.anc_mask, kRows * sizeof(unsigned int),
                             cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d_eagle_lvl_tok_, shape.lvl_tok, shape.total_lvl_rows * sizeof(int),
                             cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d_eagle_lvl_feat_, shape.lvl_feat, shape.total_lvl_rows * sizeof(int),
                             cudaMemcpyHostToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d_eagle_lvl_mask_, shape.lvl_mask,
                             shape.total_lvl_rows * sizeof(unsigned int), cudaMemcpyHostToDevice,
                             s));
  CUDA_CHECK(cudaMemcpyAsync(d_eagle_lvl_dep_, shape.lvl_dep, shape.total_lvl_rows * sizeof(int),
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
                             static_cast<std::size_t>(H) * sizeof(__half), cudaMemcpyDeviceToDevice,
                             s));
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
  int scatter_rows[8];  // function-scope: async H2D source must outlive the copy

  // Room guard: the round writes the root catch-up pair at pos-1 and cache
  // slots up to pos + acc + 1 (acc <= max depth plus bonus).
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

    // Both the draft levels and the verify read the base position from
    // d_eagle_pos_ (draft-cache prefix length == committed cache length).
    CUDA_CHECK(cudaMemcpyAsync(d_eagle_pos_, &pos, sizeof(int), cudaMemcpyHostToDevice, s));

    // Root catch-up: cur's pair through the draft layer (into the draft
    // cache at pos-1), output hidden to stash slot 0, top-n to dtoks 0..n-1.
    eagle_step(d_eagle_feats_ + static_cast<std::size_t>(cur_feat_row) * H, nullptr, cur, pos - 1,
               /*want_token=*/true, d_eagle_dtoks_);
    CUDA_CHECK(cudaMemcpyAsync(d_eagle_tree_h_, d_eagle_x_,
                               static_cast<std::size_t>(H) * sizeof(__half),
                               cudaMemcpyDeviceToDevice, s));
    for (int r = 1; r < shape.root_children; ++r) {
      kernels::launch_mask_logit(d_eagle_logits_, d_eagle_dtoks_ + r - 1, s);
      kernels::launch_argmax_float(d_eagle_logits_, cfg.vocab_size, d_eagle_dtoks_ + r, s,
                                   d_argmax_part_val_, d_argmax_part_idx_, argmax_parts_);
    }
    // Batched levels: one forward per depth; each row's top-n children are
    // extracted from its d_batch_logits_ row into their static dtok slots.
    {
      int row0 = 0;
      for (int lvl = 0; lvl < shape.n_levels; ++lvl) {
        const int B = shape.lvl_b[lvl];
        eagle_tree_level(B, row0, shape.total_lvl_rows);
        for (int r = 0; r < B; ++r) {
          float* logits = d_batch_logits_ +
                          static_cast<std::size_t>(r) * static_cast<std::size_t>(cfg.vocab_size);
          const int nc = shape.lvl_child_n[row0 + r];
          for (int c = 0; c < nc; ++c) {
            if (c > 0) {
              kernels::launch_mask_logit(logits, d_eagle_dtoks_ + shape.lvl_child[row0 + r][c - 1],
                                         s);
            }
            kernels::launch_argmax_float(logits, cfg.vocab_size,
                                         d_eagle_dtoks_ + shape.lvl_child[row0 + r][c], s,
                                         d_argmax_part_val_, d_argmax_part_idx_, argmax_parts_);
          }
        }
        row0 += B;
      }
    }
    int drafts[16];
    CUDA_CHECK(
        cudaMemcpyAsync(drafts, d_eagle_dtoks_, kNodes * sizeof(int), cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    drafted += kNodes;
    auto td = pclock::now();
    if (prof) t_draft += psec(tp, td);

    // Verify: row 0 = cur (host), rows 1..kNodes = the drafted tree (device).
    CUDA_CHECK(cudaMemcpyAsync(d_token_id_, &cur, sizeof(int), cudaMemcpyHostToDevice, s));
    CUDA_CHECK(cudaMemcpyAsync(d_token_id_ + 1, d_eagle_dtoks_, kNodes * sizeof(int),
                               cudaMemcpyDeviceToDevice, s));
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
    int verdict[17];
    CUDA_CHECK(
        cudaMemcpyAsync(verdict, d_eagle_verdict_, kRows * sizeof(int), cudaMemcpyDeviceToHost, s));
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
      for (int c = 0; c < shape.child_count[row]; ++c) {
        const int cr = shape.child_rows[row][c];
        if (drafts[cr - 1] == want) {
          next_row = cr;
          break;
        }
      }
      ++depth_try[shape.row_depth[row]];
      if (next_row < 0) break;
      ++depth_ok[shape.row_depth[row]];
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
                 t_round / n, (accepted + verifies) / n, 1000.0 * (accepted + verifies) / t_round);
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
