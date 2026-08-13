// EAGLE-1 speculative decoding for the LlamaEngine fp16 path (CPI_EAGLE=k).
//
// A released single-layer draft head (yuhuili/EAGLE-LLaMA3.1-Instruct-8B,
// verified to match the target checkpoint's geometry and tokenizer) predicts
// chains of k tokens from [embed(token) | target post-final-norm feature]
// pairs; the existing verify_tokens batched pass accepts the longest matching
// prefix plus a bonus token. Greedy-only, and it inherits verify_tokens'
// caveat: verify and single-token decode kernels round differently on
// near-ties, so output can diverge from plain greedy on tie tokens.
//
// Wiring (validated against the numpy oracle at 68.6% on-distribution chain
// acceptance): fc(cat(embed, feature)) with no pre-attention norm, standard
// residual + post_attention_layernorm + SwiGLU, RAW draft hidden through the
// target's lm_head, plain-theta rope (the draft head is trained without the
// target's llama3 rope scaling; the engine's tables are plain already).
//
// Draft cache rows are PAIR-indexed: pair p = (feature of target position p,
// token at position p+1), roped at position p+1. Prefill feeds pairs
// 0..P-2; each verify supplies true features to heal the provisionally
// drafted rows before the next chain.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "engine/llama_engine.hpp"
#include "llama_engine_internal.hpp"
#include "runtime/cuda_utils.cuh"
#include "runtime/kernels.cuh"

namespace engine {

namespace {
bool read_fp16_file(const std::string& path, std::size_t elems, std::vector<__half>& out) {
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f) return false;
  const std::size_t bytes = static_cast<std::size_t>(f.tellg());
  if (bytes != elems * sizeof(__half)) return false;
  out.resize(elems);
  f.seekg(0);
  f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
  return static_cast<bool>(f);
}
}  // namespace

bool LlamaEngine::eagle_load() {
  const char* kenv = std::getenv("CPI_EAGLE");
  if (!kenv) return false;
  eagle_k_ = std::max(1, std::min(15, std::atoi(kenv)));
  const char* dir_env = std::getenv("CPI_EAGLE_DIR");
  const std::string dir = dir_env ? dir_env : "artifacts/hub/eagle-llama31-8b";

  const auto& cfg = weights_.config();
  const int H = cfg.hidden_size;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (H / cfg.num_heads);
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : H;
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const int inter = cfg.intermediate_size;
  (void)head_dim;

  struct Spec {
    const char* file;
    void** dst;
    std::size_t elems;
  };
  const Spec specs[] = {
      {"fc_weight.bin", &d_eagle_fc_, static_cast<std::size_t>(H) * 2 * H},
      {"layers_0_self_attn_q_proj_weight.bin", &d_eagle_wq_,
       static_cast<std::size_t>(q_hidden) * H},
      {"layers_0_self_attn_k_proj_weight.bin", &d_eagle_wk_,
       static_cast<std::size_t>(kv_hidden) * H},
      {"layers_0_self_attn_v_proj_weight.bin", &d_eagle_wv_,
       static_cast<std::size_t>(kv_hidden) * H},
      {"layers_0_self_attn_o_proj_weight.bin", &d_eagle_wo_,
       static_cast<std::size_t>(H) * q_hidden},
      {"layers_0_mlp_gate_proj_weight.bin", &d_eagle_w1_, static_cast<std::size_t>(inter) * H},
      {"layers_0_mlp_up_proj_weight.bin", &d_eagle_w3_, static_cast<std::size_t>(inter) * H},
      {"layers_0_mlp_down_proj_weight.bin", &d_eagle_w2_, static_cast<std::size_t>(H) * inter},
      {"layers_0_post_attention_layernorm_weight.bin", &d_eagle_pnorm_,
       static_cast<std::size_t>(H)},
  };
  std::vector<__half> host;
  for (const Spec& s : specs) {
    if (!read_fp16_file(dir + "/" + s.file, s.elems, host)) {
      std::fprintf(stderr, "[eagle] failed to load %s/%s (size mismatch or missing); disabled\n",
                   dir.c_str(), s.file);
      eagle_free();
      return false;
    }
    CUDA_CHECK(cudaMalloc(s.dst, s.elems * sizeof(__half)));
    CUDA_CHECK(cudaMemcpy(*s.dst, host.data(), s.elems * sizeof(__half), cudaMemcpyHostToDevice));
  }

  const std::size_t kv_bytes =
      static_cast<std::size_t>(options_.max_context) * kv_hidden * sizeof(__half);
  CUDA_CHECK(cudaMalloc(&d_eagle_kcache_, kv_bytes));
  CUDA_CHECK(cudaMalloc(&d_eagle_vcache_, kv_bytes));
  CUDA_CHECK(cudaMemset(d_eagle_kcache_, 0, kv_bytes));
  CUDA_CHECK(cudaMemset(d_eagle_vcache_, 0, kv_bytes));
  CUDA_CHECK(cudaMalloc(&d_eagle_cat_, static_cast<std::size_t>(2) * H * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_eagle_x_, static_cast<std::size_t>(H) * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_eagle_tmp_, static_cast<std::size_t>(H) * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_eagle_norm_, static_cast<std::size_t>(H) * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_eagle_q_, static_cast<std::size_t>(q_hidden) * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_eagle_kv_, static_cast<std::size_t>(2) * kv_hidden * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_eagle_att_, static_cast<std::size_t>(q_hidden) * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_eagle_gate_, static_cast<std::size_t>(inter) * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_eagle_up_, static_cast<std::size_t>(inter) * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_eagle_feats_,
                        static_cast<std::size_t>(17) * H * sizeof(__half)));
  CUDA_CHECK(cudaMalloc(&d_eagle_logits_,
                        static_cast<std::size_t>(cfg.vocab_size) * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_eagle_tok_, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_eagle_dtoks_, 16 * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_eagle_pos_, sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_eagle_verdict_, 17 * sizeof(int)));
  // mt split-K verify scratch: fixed max-context chunk grid with per-chunk
  // guards (the graph-capturable attention the plan engine's spec verify uses).
  // +1: the tree verify parks its masked in-batch columns in a virtual chunk
  // slot after the last cache chunk (launch_attention_tree_split).
  eagle_mt_chunks_ = std::max(1, (options_.max_context + 31) / 32) + 1;
  {
    const std::size_t cells = static_cast<std::size_t>(17) * cfg.num_heads * eagle_mt_chunks_;
    CUDA_CHECK(cudaMalloc(&d_eagle_mt_m_, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_eagle_mt_l_, cells * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_eagle_mt_o_, cells * head_dim * sizeof(float)));
  }
  eagle_tree_ = std::getenv("CPI_EAGLE_TREE") != nullptr;
  if (eagle_tree_) {
    // Static tree shapes up to 16 nodes / 17 verify rows / 4-wide levels (see
    // llama_engine_eagle_tree.cpp); buffers sized for the largest shape.
    constexpr int kMaxRows = 17;
    constexpr int kMaxLvlRows = 16;
    constexpr int kMaxB = 4;
    CUDA_CHECK(cudaMalloc(&d_eagle_tree_h_,
                          static_cast<std::size_t>(kMaxRows) * H * sizeof(__half)));
    const std::size_t scratch =
        static_cast<std::size_t>(cfg.num_layers) * kMaxRows * kv_hidden * sizeof(__half);
    CUDA_CHECK(cudaMalloc(&d_eagle_tree_k_, scratch));
    CUDA_CHECK(cudaMalloc(&d_eagle_tree_v_, scratch));
    CUDA_CHECK(cudaMalloc(&d_eagle_row_off_, kMaxRows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_eagle_anc_mask_, kMaxRows * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_eagle_scatter_, 8 * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_eagle_bcat_,
                          static_cast<std::size_t>(kMaxB) * 2 * H * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_eagle_bx_, static_cast<std::size_t>(kMaxB) * H * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_eagle_btmp_, static_cast<std::size_t>(kMaxB) * H * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_eagle_bnorm_, static_cast<std::size_t>(kMaxB) * H * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_eagle_bq_,
                          static_cast<std::size_t>(kMaxB) * q_hidden * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_eagle_batt_,
                          static_cast<std::size_t>(kMaxB) * q_hidden * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_eagle_bgate_,
                          static_cast<std::size_t>(kMaxB) * inter * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_eagle_bup_,
                          static_cast<std::size_t>(kMaxB) * inter * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_eagle_scrk_,
                          static_cast<std::size_t>(kMaxLvlRows) * kv_hidden * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_eagle_scrv_,
                          static_cast<std::size_t>(kMaxLvlRows) * kv_hidden * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_eagle_lvl_tok_, kMaxLvlRows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_eagle_lvl_feat_, kMaxLvlRows * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_eagle_lvl_mask_, kMaxLvlRows * sizeof(unsigned int)));
    CUDA_CHECK(cudaMalloc(&d_eagle_lvl_dep_, kMaxLvlRows * sizeof(int)));
  }
  eagle_enabled_ = true;
  if (options_.verbose) {
    std::cout << "[engine] eagle=on k=" << eagle_k_ << (eagle_tree_ ? " tree" : " chain")
              << " dir=" << dir << "\n";
  }
  return true;
}

void LlamaEngine::eagle_free() {
  for (void** p : {&d_eagle_fc_, &d_eagle_wq_, &d_eagle_wk_, &d_eagle_wv_, &d_eagle_wo_,
                   &d_eagle_w1_, &d_eagle_w3_, &d_eagle_w2_, &d_eagle_pnorm_}) {
    if (*p) cudaFree(*p);
    *p = nullptr;
  }
  auto freep = [](auto*& p) {
    if (p) cudaFree(p);
    p = nullptr;
  };
  freep(d_eagle_kcache_);
  freep(d_eagle_vcache_);
  freep(d_eagle_cat_);
  freep(d_eagle_x_);
  freep(d_eagle_tmp_);
  freep(d_eagle_norm_);
  freep(d_eagle_q_);
  freep(d_eagle_kv_);
  freep(d_eagle_att_);
  freep(d_eagle_gate_);
  freep(d_eagle_up_);
  freep(d_eagle_feats_);
  freep(d_eagle_logits_);
  freep(d_eagle_tok_);
  freep(d_eagle_dtoks_);
  freep(d_eagle_pos_);
  freep(d_eagle_verdict_);
  freep(d_eagle_mt_m_);
  freep(d_eagle_mt_l_);
  freep(d_eagle_mt_o_);
  freep(d_eagle_tree_h_);
  freep(d_eagle_tree_k_);
  freep(d_eagle_tree_v_);
  freep(d_eagle_row_off_);
  freep(d_eagle_anc_mask_);
  freep(d_eagle_scatter_);
  freep(d_eagle_bcat_);
  freep(d_eagle_bx_);
  freep(d_eagle_btmp_);
  freep(d_eagle_bnorm_);
  freep(d_eagle_bq_);
  freep(d_eagle_batt_);
  freep(d_eagle_bgate_);
  freep(d_eagle_bup_);
  freep(d_eagle_scrk_);
  freep(d_eagle_scrv_);
  freep(d_eagle_lvl_tok_);
  freep(d_eagle_lvl_feat_);
  freep(d_eagle_lvl_mask_);
  freep(d_eagle_lvl_dep_);
  if (eagle_vgraph_exec_) {
    cudaGraphExecDestroy(eagle_vgraph_exec_);
    eagle_vgraph_exec_ = nullptr;
  }
  if (eagle_vgraph_) {
    cudaGraphDestroy(eagle_vgraph_);
    eagle_vgraph_ = nullptr;
  }
  eagle_vgraph_k_ = 0;
  eagle_enabled_ = false;
}

// Runs one (feature, token) pair through the draft layer at pair row pair_idx
// (rope position pair_idx + 1). want_token=false skips the lm_head/argmax
// (used when healing accepted rows with true features). The output feature is
// left in d_eagle_x_.
void LlamaEngine::eagle_step(const __half* feature, const int* token_dev, int token_host,
                             int pair_idx, bool want_token, int* dtok_out) {
  const auto& cfg = weights_.config();
  const int H = cfg.hidden_size;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (H / cfg.num_heads);
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : H;
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const int inter = cfg.intermediate_size;
  auto s = compute_stream_;

  const int* tok_ptr = token_dev;
  if (tok_ptr == nullptr) {
    CUDA_CHECK(cudaMemcpyAsync(d_eagle_tok_, &token_host, sizeof(int), cudaMemcpyHostToDevice, s));
    tok_ptr = d_eagle_tok_;
  }
  kernels::launch_embedding_lookup(static_cast<const __half*>(d_tok_embeddings_), tok_ptr,
                                   d_eagle_cat_, 1, H, s);
  CUDA_CHECK(cudaMemcpyAsync(d_eagle_cat_ + H, feature, static_cast<std::size_t>(H) * sizeof(__half),
                             cudaMemcpyDeviceToDevice, s));
  kernels::launch_gemv_splitk_f16(static_cast<const __half*>(d_eagle_fc_), d_eagle_cat_,
                                  d_eagle_x_, H, 2 * H, s);
  // No pre-attention norm (EAGLE removes layer 0's input_layernorm).
  kernels::launch_gemv_splitk_f16(static_cast<const __half*>(d_eagle_wq_), d_eagle_x_, d_eagle_q_,
                                  q_hidden, H, s);
  kernels::launch_gemv_splitk_f16(static_cast<const __half*>(d_eagle_wk_), d_eagle_x_, d_eagle_kv_,
                                  kv_hidden, H, s);
  kernels::launch_gemv_splitk_f16(static_cast<const __half*>(d_eagle_wv_), d_eagle_x_,
                                  d_eagle_kv_ + kv_hidden, kv_hidden, H, s);
  kernels::launch_rope_inplace_table(d_eagle_q_, d_eagle_kv_, cfg.num_heads, cfg.num_kv_heads,
                                     head_dim, pair_idx + 1, d_rope_cos_, d_rope_sin_, s);
  CUDA_CHECK(cudaMemcpyAsync(d_eagle_kcache_ + static_cast<std::size_t>(pair_idx) * kv_hidden,
                             d_eagle_kv_, static_cast<std::size_t>(kv_hidden) * sizeof(__half),
                             cudaMemcpyDeviceToDevice, s));
  CUDA_CHECK(cudaMemcpyAsync(d_eagle_vcache_ + static_cast<std::size_t>(pair_idx) * kv_hidden,
                             d_eagle_kv_ + kv_hidden,
                             static_cast<std::size_t>(kv_hidden) * sizeof(__half),
                             cudaMemcpyDeviceToDevice, s));
  kernels::launch_attention_step(d_eagle_q_, d_eagle_kcache_, d_eagle_vcache_, d_eagle_att_,
                                 pair_idx + 1, cfg.num_heads, cfg.num_kv_heads, head_dim, s,
                                 d_attn_chunk_m_, d_attn_chunk_l_, d_attn_chunk_o_,
                                 attn_chunk_capacity_, !options_.disable_split_attention);
  kernels::launch_gemv_splitk_f16(static_cast<const __half*>(d_eagle_wo_), d_eagle_att_,
                                  d_eagle_tmp_, H, q_hidden, s);
  kernels::launch_add_inplace(d_eagle_x_, d_eagle_tmp_, H, s);
  kernels::launch_rmsnorm(d_eagle_x_, static_cast<const __half*>(d_eagle_pnorm_), d_eagle_norm_, 1,
                          H, cfg.norm_eps, s);
  kernels::launch_gemv_splitk_f16(static_cast<const __half*>(d_eagle_w1_), d_eagle_norm_,
                                  d_eagle_gate_, inter, H, s);
  kernels::launch_gemv_splitk_f16(static_cast<const __half*>(d_eagle_w3_), d_eagle_norm_,
                                  d_eagle_up_, inter, H, s);
  kernels::launch_silu_mul(d_eagle_gate_, d_eagle_up_, d_eagle_gate_, inter, s);
  kernels::launch_gemv_splitk_f16(static_cast<const __half*>(d_eagle_w2_), d_eagle_gate_,
                                  d_eagle_tmp_, H, inter, s);
  kernels::launch_add_inplace(d_eagle_x_, d_eagle_tmp_, H, s);

  if (!want_token) return;
  // Shared lm_head over the RAW draft hidden, then partitioned device argmax
  // straight into the chained-token slot; no host sync.
  const void* head_src = lm_head_packed_.active()
                             ? dequant_packed_for_gemm(lm_head_packed_, s)
                             : static_cast<const void*>(d_lm_head_);
  kernels::launch_rowmajor_half_gemv_f32(static_cast<const __half*>(head_src), d_eagle_x_,
                                         d_eagle_logits_, cfg.vocab_size, H, s);
  kernels::launch_argmax_float(d_eagle_logits_, cfg.vocab_size, dtok_out, s, d_argmax_part_val_,
                               d_argmax_part_idx_, argmax_parts_);
}

// Feed a prefill chunk's (feature, token) pairs into the draft cache. Features
// for the chunk's rows are expected in d_x_norm_ [count, H] (post final norm);
// tokens[r] is the token at position chunk_start + r + 1.
void LlamaEngine::eagle_prefill_pairs(const int* tokens, int count, int chunk_start) {
  const int H = weights_.config().hidden_size;
  for (int r = 0; r < count; ++r) {
    const __half* feat = static_cast<const __half*>(d_x_norm_) + static_cast<std::size_t>(r) * H;
    eagle_step(feat, nullptr, tokens[r], chunk_start + r, /*want_token=*/false, nullptr);
  }
}

// Graph-safe verify body: K token rows (already in d_token_id_) forwarded at
// device base position d_eagle_pos_. Fixed shapes throughout: fp16-resident
// GEMMs (fixed M=K), device-position rope/store, and the mt split-K attention
// whose fixed max-context chunk grid guards off dead chunks. Leaves verdicts
// in d_eagle_verdict_[0..K) and the post-norm features in d_eagle_feats_.
void LlamaEngine::eagle_verify_forward(int K) {
  const auto& cfg = weights_.config();
  const int hidden = cfg.hidden_size;
  const int head_dim = attn_head_dim_ > 0 ? attn_head_dim_ : (hidden / cfg.num_heads);
  const int q_hidden = attn_q_hidden_ > 0 ? attn_q_hidden_ : hidden;
  const int kv_hidden = attn_kv_hidden_ > 0 ? attn_kv_hidden_ : (cfg.num_kv_heads * head_dim);
  const int inter = cfg.intermediate_size;
  auto s = compute_stream_;
  const std::size_t layer_stride =
      static_cast<std::size_t>(kv_capacity_tokens_) * static_cast<std::size_t>(kv_hidden);
  const std::size_t q_row_bytes = static_cast<std::size_t>(q_hidden) * sizeof(__half);
  const std::size_t kv_row_bytes = static_cast<std::size_t>(kv_hidden) * sizeof(__half);
  const std::size_t qkv_stride_bytes =
      static_cast<std::size_t>(q_hidden + 2 * kv_hidden) * sizeof(__half);
  const std::size_t ff_row_bytes = static_cast<std::size_t>(inter) * sizeof(__half);
  const std::size_t ff13_stride_bytes = static_cast<std::size_t>(2 * inter) * sizeof(__half);
  auto* qkv_base = static_cast<const __half*>(d_qkv_);
  auto* ff13_base = static_cast<const __half*>(d_ff13_);

  kernels::launch_embedding_lookup(static_cast<const __half*>(d_tok_embeddings_), d_token_id_,
                                   static_cast<__half*>(d_x_), K, hidden, s);
  for (int layer = 0; layer < cfg.num_layers; ++layer) {
    const LayerDeviceWeights* lw = &layer_cache_[static_cast<std::size_t>(layer)];
    launch_norm(d_x_, lw->norm_att, lw->norm_att_bias, d_x_norm_, K, hidden);
    const void* qkv_src = dequant_packed_qkv_for_gemm(*lw, s);
    if (qkv_src == nullptr) qkv_src = lw->wqkv;
    detail::dispatch_linear_rowmajor_weight(
        cublas_, cublas_lt_, &lt_plan_cache_, lt_workspace_, lt_workspace_bytes_, s,
        const_cast<void*>(qkv_src),
        d_x_norm_, d_qkv_, q_hidden + 2 * kv_hidden, hidden, K, CUDA_R_16F);
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_q_, q_row_bytes, qkv_base, qkv_stride_bytes,
                                 q_row_bytes, K, cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_k_, kv_row_bytes, qkv_base + q_hidden,
                                 qkv_stride_bytes, kv_row_bytes, K, cudaMemcpyDeviceToDevice, s));
    CUDA_CHECK(cudaMemcpy2DAsync(d_prefill_v_, kv_row_bytes, qkv_base + q_hidden + kv_hidden,
                                 qkv_stride_bytes, kv_row_bytes, K, cudaMemcpyDeviceToDevice, s));
    kernels::launch_rope_inplace_batched_strided_device_pos(
        static_cast<__half*>(d_prefill_q_), static_cast<__half*>(d_prefill_k_), K, cfg.num_heads,
        cfg.num_kv_heads, head_dim, d_eagle_pos_, d_rope_cos_, d_rope_sin_, q_hidden, kv_hidden,
        s);
    auto* k_layer =
        static_cast<__half*>(d_k_cache_) + static_cast<std::size_t>(layer) * layer_stride;
    auto* v_layer =
        static_cast<__half*>(d_v_cache_) + static_cast<std::size_t>(layer) * layer_stride;
    kernels::launch_store_kv_seq_device_pos(static_cast<const __half*>(d_prefill_k_),
                                            static_cast<const __half*>(d_prefill_v_), k_layer,
                                            v_layer, d_eagle_pos_, kv_hidden, K,
                                            kv_capacity_tokens_, s);
    kernels::launch_attention_split_any_mt_device_pos(
        static_cast<const __half*>(d_prefill_q_), k_layer, v_layer, static_cast<__half*>(d_att_),
        d_eagle_pos_, K, /*window=*/0, cfg.num_heads, cfg.num_kv_heads, head_dim, d_eagle_mt_m_,
        d_eagle_mt_l_, d_eagle_mt_o_, 32, eagle_mt_chunks_, s);
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

// Capture-once, replay-per-round wrapper. Returns false when the graphed path
// is unavailable (caller falls back to eager verify_tokens).
bool LlamaEngine::eagle_verify_graphed(const std::vector<int>& batch, int start_pos,
                                       std::vector<int>& out_argmax) {
  const auto& cfg = weights_.config();
  const int K = static_cast<int>(batch.size());
  // Graph eligibility: pure fp16-resident weights, no qkv bias / qk-norm
  // stages (the forward above omits them), contiguous fp16 KV.
  if (cached_layer_count_ != cfg.num_layers || cached_int8_proj_enabled_ || kv_int4_enabled_ ||
      layer_cache_.empty() || layer_cache_[0].bqkv != nullptr || layer_cache_[0].q_norm != nullptr) {
    return false;
  }
  auto s = compute_stream_;
  CUDA_CHECK(cudaMemcpyAsync(d_token_id_, batch.data(), static_cast<std::size_t>(K) * sizeof(int),
                             cudaMemcpyHostToDevice, s));
  CUDA_CHECK(
      cudaMemcpyAsync(d_eagle_pos_, &start_pos, sizeof(int), cudaMemcpyHostToDevice, s));
  if (eagle_vgraph_k_ != K) {
    if (eagle_vgraph_exec_) {
      cudaGraphExecDestroy(eagle_vgraph_exec_);
      eagle_vgraph_exec_ = nullptr;
    }
    if (eagle_vgraph_) {
      cudaGraphDestroy(eagle_vgraph_);
      eagle_vgraph_ = nullptr;
    }
    // Warm run (also warms the cublasLt plan cache), then capture.
    eagle_verify_forward(K);
    CUDA_CHECK(cudaStreamSynchronize(s));
    CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    eagle_verify_forward(K);
    CUDA_CHECK(cudaStreamEndCapture(s, &eagle_vgraph_));
    CUDA_CHECK(cudaGraphInstantiate(&eagle_vgraph_exec_, eagle_vgraph_, nullptr, nullptr, 0));
    eagle_vgraph_k_ = K;
    if (options_.verbose) {
      std::cout << "[eagle] verify graph captured (K=" << K << ")\n";
    }
  }
  CUDA_CHECK(cudaGraphLaunch(eagle_vgraph_exec_, s));
  out_argmax.resize(static_cast<std::size_t>(K));
  CUDA_CHECK(cudaMemcpyAsync(out_argmax.data(), d_eagle_verdict_,
                             static_cast<std::size_t>(K) * sizeof(int), cudaMemcpyDeviceToHost,
                             s));
  CUDA_CHECK(cudaStreamSynchronize(s));
  return true;
}

std::vector<int> LlamaEngine::eagle_generate(const std::vector<int>& prompt_tokens,
                                             int max_new_tokens,
                                             const std::function<bool(int)>& on_token) {
  const auto& cfg = weights_.config();
  const int H = cfg.hidden_size;
  const int max_ctx = options_.max_context;
  auto s = compute_stream_;
  std::vector<int> out;
  out.reserve(static_cast<std::size_t>(max_new_tokens));

  int pos = static_cast<int>(prompt_tokens.size()) - 1;
  int cur = prompt_tokens.back();
  std::vector<int> dummy_hist;  // greedy: sampler history unused
  int verifies = 0, drafted = 0, accepted = 0;
  int pos_try[16] = {0}, pos_ok[16] = {0};  // per-chain-position accept stats

  // CPI_EAGLE_PROF=1: wall-clock the round's phases (each phase boundary gets a
  // stream sync while profiling, so the numbers are attribution, not exact
  // production timings -- the perturbation is a few syncs/round).
  static const bool prof = std::getenv("CPI_EAGLE_PROF") != nullptr;
  double t_heal = 0.0, t_draft = 0.0, t_verify = 0.0, t_round = 0.0;
  using pclock = std::chrono::steady_clock;
  const auto psec = [](pclock::time_point a, pclock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  // First token through the normal single-token path; its feature (the post-
  // final-norm hidden of position pos) lands in d_x_norm_.
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
  int heal_count = 0;           // accepted pairs to re-run with true features
  int heal_tokens[17];          // their input tokens (batch[j+1])
  int cur_feat_row = 0;         // d_eagle_feats_ row holding cur's producing feature

  while (!stop && static_cast<int>(out.size()) < max_new_tokens && pos < max_ctx - 1) {
    const auto tr0 = pclock::now();
    // Heal previously accepted pairs with their true features, then the
    // catch-up pair for cur (which yields the first draft).
    const int base = pos - 1 - heal_count;  // pair index of the first heal row
    for (int j = 0; j < heal_count; ++j) {
      eagle_step(d_eagle_feats_ + static_cast<std::size_t>(j) * H, nullptr, heal_tokens[j],
                 base + j, /*want_token=*/false, nullptr);
    }
    auto tp = pclock::now();
    if (prof) {
      CUDA_CHECK(cudaStreamSynchronize(s));
      tp = pclock::now();
      t_heal += psec(tr0, tp);
    }
    // Chain entirely on the device: each step's argmax lands in a
    // d_eagle_dtoks_ slot which feeds the next step's embedding lookup; the
    // drafts cross to the host once, after the chain.
    eagle_step(d_eagle_feats_ + static_cast<std::size_t>(cur_feat_row) * H, nullptr, cur, pos - 1,
               /*want_token=*/true, d_eagle_dtoks_);
    int nd = 1;
    const int room = max_ctx - pos - 2;
    while (nd < eagle_k_ && nd < room) {
      eagle_step(d_eagle_x_, d_eagle_dtoks_ + (nd - 1), 0, pos - 1 + nd, /*want_token=*/true,
                 d_eagle_dtoks_ + nd);
      ++nd;
    }
    int drafts[16];
    CUDA_CHECK(cudaMemcpyAsync(drafts, d_eagle_dtoks_, static_cast<std::size_t>(nd) * sizeof(int),
                               cudaMemcpyDeviceToHost, s));
    CUDA_CHECK(cudaStreamSynchronize(s));
    drafted += nd;
    auto td = pclock::now();
    if (prof) t_draft += psec(tp, td);

    std::vector<int> batch(static_cast<std::size_t>(nd) + 1);
    batch[0] = cur;
    for (int i = 0; i < nd; ++i) batch[static_cast<std::size_t>(i) + 1] = drafts[i];
    std::vector<int> verdict;
    if (!eagle_verify_graphed(batch, pos, verdict)) {
      verify_tokens(batch, pos, verdict);
      // True features for all verified rows (the graphed path copies them
      // inside the graph; the eager path leaves them in d_x_norm_).
      CUDA_CHECK(cudaMemcpyAsync(d_eagle_feats_, d_x_norm_,
                                 static_cast<std::size_t>(nd + 1) * H * sizeof(__half),
                                 cudaMemcpyDeviceToDevice, s));
    }
    ++verifies;
    if (prof) {
      CUDA_CHECK(cudaStreamSynchronize(s));
      const auto tv = pclock::now();
      t_verify += psec(td, tv);
      t_round += psec(tr0, tv);  // host tail after verify is negligible; counted next round
    }

    int acc = 0;
    while (acc < nd && verdict[static_cast<std::size_t>(acc)] == drafts[acc]) ++acc;
    accepted += acc;
    for (int i = 0; i < nd; ++i) {
      ++pos_try[i];
      if (verdict[static_cast<std::size_t>(i)] == drafts[i]) ++pos_ok[i];
      if (i >= acc) break;  // positions after the first miss are conditional noise
    }
    for (int i = 0; i < acc && !stop; ++i) {
      out.push_back(drafts[i]);
      if (static_cast<int>(out.size()) >= max_new_tokens) stop = true;
      if (!stop && on_token && !on_token(drafts[i])) stop = true;
      if (!stop && options_.eos_token_id >= 0 && drafts[i] == options_.eos_token_id) stop = true;
    }
    if (stop) break;
    const int bonus = verdict[static_cast<std::size_t>(acc)];
    out.push_back(bonus);
    if (static_cast<int>(out.size()) >= max_new_tokens) stop = true;
    if (!stop && on_token && !on_token(bonus)) stop = true;
    if (!stop && options_.eos_token_id >= 0 && bonus == options_.eos_token_id) stop = true;

    for (int j = 0; j < acc; ++j) heal_tokens[j] = batch[static_cast<std::size_t>(j) + 1];
    heal_count = acc;
    cur_feat_row = acc;
    cur = bonus;
    pos += acc + 1;
  }
  if (prof && verifies > 0) {
    const double n = static_cast<double>(verifies);
    std::fprintf(stderr,
                 "[eagle-prof] rounds=%d avg_ms: heal=%.2f draft=%.2f verify=%.2f round=%.2f "
                 "(tokens/round=%.2f -> %.1f tok/s in-loop)\n",
                 verifies, t_heal / n, t_draft / n, t_verify / n, t_round / n,
                 (accepted + verifies) / n, 1000.0 * (accepted + verifies) / t_round);
  }
  if (std::getenv("CPI_EAGLE_STATS")) {
    std::fprintf(stderr, "[eagle] verifies=%d drafted=%d accepted=%d (%.1f%%) tokens=%zu\n",
                 verifies, drafted, accepted, drafted ? 100.0 * accepted / drafted : 0.0,
                 out.size());
    for (int i = 0; i < eagle_k_; ++i) {
      if (pos_try[i]) {
        std::fprintf(stderr, "[eagle]   chain pos %d: %d/%d (%.1f%%)\n", i + 1, pos_ok[i],
                     pos_try[i], 100.0 * pos_ok[i] / pos_try[i]);
      }
    }
  }
  return out;
}

}  // namespace engine
