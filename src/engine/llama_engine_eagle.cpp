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
  eagle_enabled_ = true;
  if (options_.verbose) {
    std::cout << "[engine] eagle=on k=" << eagle_k_ << " dir=" << dir << "\n";
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
  kernels::launch_rowmajor_half_gemv_f32(static_cast<const __half*>(d_lm_head_), d_eagle_x_,
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
    // Heal previously accepted pairs with their true features, then the
    // catch-up pair for cur (which yields the first draft).
    const int base = pos - 1 - heal_count;  // pair index of the first heal row
    for (int j = 0; j < heal_count; ++j) {
      eagle_step(d_eagle_feats_ + static_cast<std::size_t>(j) * H, nullptr, heal_tokens[j],
                 base + j, /*want_token=*/false, nullptr);
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

    std::vector<int> batch(static_cast<std::size_t>(nd) + 1);
    batch[0] = cur;
    for (int i = 0; i < nd; ++i) batch[static_cast<std::size_t>(i) + 1] = drafts[i];
    std::vector<int> verdict;
    verify_tokens(batch, pos, verdict);
    ++verifies;
    // True features for all verified rows (before anything clobbers d_x_norm_).
    CUDA_CHECK(cudaMemcpyAsync(d_eagle_feats_, d_x_norm_,
                               static_cast<std::size_t>(nd + 1) * H * sizeof(__half),
                               cudaMemcpyDeviceToDevice, s));

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
