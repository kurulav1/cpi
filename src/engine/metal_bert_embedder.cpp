// Metal BERT embedding encoder. See include/engine/metal_bert_embedder.hpp for why this mirrors
// BertEmbedder rather than generalising it.
//
// The arithmetic is a transcription of src/engine/bert_embedder.cu's embed(), op for op, so the
// two backends can be diffed against each other and against HuggingFace rather than compared
// approximately.

#include "engine/metal_bert_embedder.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/fp16.hpp"

namespace engine {

namespace {

struct GemvParams {
  std::uint32_t out_dim, in_dim, tokens, has_bias;
};
struct ElemParams {
  std::uint32_t n;
  float scale;
};
struct LayerNormParams {
  std::uint32_t rows, cols;
  float eps;
  std::uint32_t has_bias;
};
struct BiAttnParams {
  std::uint32_t tokens, heads, head_dim;
  float scale;
  std::uint32_t row_stride;
};
struct BertEmbedParams {
  std::uint32_t tokens, hidden, type_id;
};
struct BertPoolParams {
  std::uint32_t tokens, hidden, mean_pool, normalize;
};

// Must match plan_metal_engine.cpp and the shader. Restated here rather than shared because the
// shader owns them; if they drift, the GEMM covers the wrong rows silently, which is why the
// out_dim check below refuses instead of trusting.
constexpr int kGemmBN = 32;
constexpr int kGemmFBM = 64;
constexpr int kGemmRF = 4;
constexpr int kGemmCF = 4;
constexpr int kGemmTG = 32 * (kGemmFBM / (8 * kGemmRF)) * (kGemmBN / (8 * kGemmCF));

}  // namespace

runtime::MetalBuffer MetalBertEmbedder::upload_fp16(const std::string& name,
                                                    std::size_t expected_elems) {
  if (!weights_.has_tensor(name)) {
    throw std::runtime_error("metal bert: missing tensor " + name);
  }
  const std::size_t bytes = weights_.tensor_bytes(name);
  if (bytes != expected_elems * sizeof(float)) {
    throw std::runtime_error("metal bert: tensor " + name + " expected " +
                             std::to_string(expected_elems) + " fp32 elems, got " +
                             std::to_string(bytes / sizeof(float)));
  }
  const auto* src = reinterpret_cast<const float*>(weights_.tensor_ptr(name));
  std::vector<std::uint16_t> host(expected_elems);
  for (std::size_t i = 0; i < expected_elems; ++i) host[i] = cpi::f32_to_f16(src[i]);
  return ctx_.alloc_from(host.data(), host.size() * sizeof(std::uint16_t));
}

void MetalBertEmbedder::linear(const runtime::MetalBuffer& w, const runtime::MetalBuffer& bias,
                               const runtime::MetalBuffer& x, runtime::MetalBuffer& y, int out,
                               int in, int rows) {
  GemvParams p{static_cast<std::uint32_t>(out), static_cast<std::uint32_t>(in),
               static_cast<std::uint32_t>(rows), 1u};
  const void* bufs[] = {w.handle(), x.handle(), y.handle(), bias.handle()};
  const std::size_t tiles = static_cast<std::size_t>((rows + kGemmBN - 1) / kGemmBN);
  const std::size_t groups = (static_cast<std::size_t>(out) / kGemmFBM) * tiles;
  ctx_.dispatch("cpi_gemm_f16", runtime::MetalContext::Grid::Groups, groups, kGemmTG, bufs, nullptr,
                4, &p, sizeof(p));
}

void MetalBertEmbedder::layernorm_residual(runtime::MetalBuffer& src,
                                           runtime::MetalBuffer& residual,
                                           const runtime::MetalBuffer& w,
                                           const runtime::MetalBuffer& b, runtime::MetalBuffer& out,
                                           int rows, int cols) {
  const std::size_t n = static_cast<std::size_t>(rows) * cols;
  // cpi_add_inplace is (in, out) and writes buffer 1, so the accumulator is second: this makes
  // src += residual, leaving the sum in src. The CUDA kernel fuses the add into layernorm; doing
  // it in two dispatches is the same arithmetic because the add is exact in fp32 either way.
  {
    ElemParams p{static_cast<std::uint32_t>(n), 1.0f};
    const void* bufs[] = {residual.handle(), src.handle()};
    ctx_.dispatch("cpi_add_inplace", runtime::MetalContext::Grid::Threads, n, 256, bufs, nullptr, 2,
                  &p, sizeof(p));
  }
  {
    LayerNormParams p{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(cols),
                      cfg_.layer_norm_eps, 1u};
    const void* bufs[] = {src.handle(), w.handle(), b.handle(), out.handle()};
    ctx_.dispatch("cpi_layernorm", runtime::MetalContext::Grid::Groups,
                  static_cast<std::size_t>(rows), 256, bufs, nullptr, 4, &p, sizeof(p));
  }
}

void MetalBertEmbedder::initialize(const std::string& model_dir) {
  if (!ctx_.available()) {
    throw std::runtime_error("metal bert: no Metal GPU (" + ctx_.last_error() + ")");
  }
  if (!ctx_.load_library()) {
    // A silent no-op dispatch is the failure mode here: MetalContext::dispatch does nothing
    // without a library, which produces plausible zeros rather than an error.
    throw std::runtime_error("metal bert: no shader library (" + ctx_.last_error() + ")");
  }
  cfg_ = EmbeddingConfig::load(model_dir);
  weights_.open(model_dir);

  const int H = cfg_.hidden_size, I = cfg_.intermediate_size;
  // cpi_gemm_f16 maps one threadgroup to a kGemmFBM-row block, so an out_dim that is not a whole
  // number of them would silently skip the remainder rather than fail.
  if (H % kGemmFBM != 0 || I % kGemmFBM != 0) {
    throw std::runtime_error("metal bert: hidden_size (" + std::to_string(H) +
                             ") and intermediate_size (" + std::to_string(I) +
                             ") must both be multiples of " + std::to_string(kGemmFBM) +
                             " for the fp16 GEMM tiling");
  }
  if (H % cfg_.num_heads != 0) {
    throw std::runtime_error("metal bert: hidden_size is not divisible by num_attention_heads");
  }

  word_emb_ = upload_fp16("embeddings.word_embeddings.weight",
                          static_cast<std::size_t>(cfg_.vocab_size) * H);
  pos_emb_ = upload_fp16("embeddings.position_embeddings.weight",
                         static_cast<std::size_t>(cfg_.max_position_embeddings) * H);
  type_emb_ = upload_fp16("embeddings.token_type_embeddings.weight",
                          static_cast<std::size_t>(cfg_.type_vocab_size) * H);
  emb_ln_w_ = upload_fp16("embeddings.LayerNorm.weight", H);
  emb_ln_b_ = upload_fp16("embeddings.LayerNorm.bias", H);

  layers_.resize(static_cast<std::size_t>(cfg_.num_layers));
  for (int l = 0; l < cfg_.num_layers; ++l) {
    const std::string p = "encoder.layer." + std::to_string(l) + ".";
    LayerWeights& w = layers_[static_cast<std::size_t>(l)];
    const std::size_t HH = static_cast<std::size_t>(H) * H;
    w.q_w = upload_fp16(p + "attention.self.query.weight", HH);
    w.q_b = upload_fp16(p + "attention.self.query.bias", H);
    w.k_w = upload_fp16(p + "attention.self.key.weight", HH);
    w.k_b = upload_fp16(p + "attention.self.key.bias", H);
    w.v_w = upload_fp16(p + "attention.self.value.weight", HH);
    w.v_b = upload_fp16(p + "attention.self.value.bias", H);
    w.o_w = upload_fp16(p + "attention.output.dense.weight", HH);
    w.o_b = upload_fp16(p + "attention.output.dense.bias", H);
    w.attn_ln_w = upload_fp16(p + "attention.output.LayerNorm.weight", H);
    w.attn_ln_b = upload_fp16(p + "attention.output.LayerNorm.bias", H);
    w.inter_w = upload_fp16(p + "intermediate.dense.weight", static_cast<std::size_t>(I) * H);
    w.inter_b = upload_fp16(p + "intermediate.dense.bias", I);
    w.out_w = upload_fp16(p + "output.dense.weight", static_cast<std::size_t>(H) * I);
    w.out_b = upload_fp16(p + "output.dense.bias", H);
    w.out_ln_w = upload_fp16(p + "output.LayerNorm.weight", H);
    w.out_ln_b = upload_fp16(p + "output.LayerNorm.bias", H);
  }

  // Scratch, sized once for the longest sequence the config allows.
  const std::size_t L = static_cast<std::size_t>(cfg_.max_tokens);
  tokens_ = ctx_.alloc(L * sizeof(std::int32_t));
  x_ = ctx_.alloc(L * H * 2);
  tmp_ = ctx_.alloc(L * H * 2);
  q_ = ctx_.alloc(L * H * 2);
  k_ = ctx_.alloc(L * H * 2);
  v_ = ctx_.alloc(L * H * 2);
  att_ = ctx_.alloc(L * H * 2);
  inter_ = ctx_.alloc(L * static_cast<std::size_t>(I) * 2);
  pooled_ = ctx_.alloc(static_cast<std::size_t>(H) * sizeof(float));
  ready_ = true;
}

std::vector<float> MetalBertEmbedder::embed(const std::vector<int>& token_ids) {
  if (!ready_) throw std::runtime_error("metal bert: initialize() was not called");
  const int H = cfg_.hidden_size, I = cfg_.intermediate_size;
  const int heads = cfg_.num_heads;
  const int head_dim = H / heads;

  int L = static_cast<int>(token_ids.size());
  if (L < 1) L = 1;
  if (L > cfg_.max_tokens) L = cfg_.max_tokens;

  {
    auto* dst = static_cast<std::int32_t*>(tokens_.contents());
    for (int i = 0; i < L; ++i)
      dst[i] = static_cast<std::int32_t>(token_ids[static_cast<std::size_t>(i)]);
  }

  // ---- embeddings: word + position + token_type, then LayerNorm ----
  {
    BertEmbedParams p{static_cast<std::uint32_t>(L), static_cast<std::uint32_t>(H), 0u};
    const void* bufs[] = {word_emb_.handle(), pos_emb_.handle(), type_emb_.handle(),
                          tokens_.handle(), tmp_.handle()};
    ctx_.dispatch("cpi_bert_embed", runtime::MetalContext::Grid::Threads,
                  static_cast<std::size_t>(L) * H, 256, bufs, nullptr, 5, &p, sizeof(p));
  }
  {
    LayerNormParams p{static_cast<std::uint32_t>(L), static_cast<std::uint32_t>(H),
                      cfg_.layer_norm_eps, 1u};
    const void* bufs[] = {tmp_.handle(), emb_ln_w_.handle(), emb_ln_b_.handle(), x_.handle()};
    ctx_.dispatch("cpi_layernorm", runtime::MetalContext::Grid::Groups, static_cast<std::size_t>(L),
                  256, bufs, nullptr, 4, &p, sizeof(p));
  }

  for (int l = 0; l < cfg_.num_layers; ++l) {
    const LayerWeights& w = layers_[static_cast<std::size_t>(l)];

    linear(w.q_w, w.q_b, x_, q_, H, H, L);
    linear(w.k_w, w.k_b, x_, k_, H, H, L);
    linear(w.v_w, w.v_b, x_, v_, H, H, L);
    {
      // row_stride 0 means "packed", i.e. heads*head_dim; q, k and v are separate buffers here,
      // not the fused block the vision tower feeds this kernel. Passing the vision path's
      // 3*hidden would read each row from two thirds of the way into the wrong tensor.
      BiAttnParams p{static_cast<std::uint32_t>(L), static_cast<std::uint32_t>(heads),
                     static_cast<std::uint32_t>(head_dim),
                     1.0f / std::sqrt(static_cast<float>(head_dim)), 0u};
      const void* bufs[] = {q_.handle(), k_.handle(), v_.handle(), att_.handle()};
      ctx_.dispatch("cpi_attention_bidirectional", runtime::MetalContext::Grid::Groups,
                    static_cast<std::size_t>(L) * heads, 64, bufs, nullptr, 4, &p, sizeof(p));
    }
    linear(w.o_w, w.o_b, att_, tmp_, H, H, L);
    layernorm_residual(tmp_, x_, w.attn_ln_w, w.attn_ln_b, x_, L, H);

    linear(w.inter_w, w.inter_b, x_, inter_, I, H, L);
    {
      // Exact erf GELU, not the tanh approximation. BERT's hidden_act is "gelu", which in
      // transformers is the erf form; cpi_gelu is the tanh one and the two differ by ~4e-4.
      ElemParams p{static_cast<std::uint32_t>(L) * static_cast<std::uint32_t>(I), 1.0f};
      const void* bufs[] = {inter_.handle()};
      ctx_.dispatch("cpi_gelu_erf", runtime::MetalContext::Grid::Threads,
                    static_cast<std::size_t>(L) * I, 256, bufs, nullptr, 1, &p, sizeof(p));
    }
    linear(w.out_w, w.out_b, inter_, tmp_, H, I, L);
    layernorm_residual(tmp_, x_, w.out_ln_w, w.out_ln_b, x_, L, H);
  }

  // ---- pool + L2-normalise ----
  {
    BertPoolParams p{static_cast<std::uint32_t>(L), static_cast<std::uint32_t>(H),
                     cfg_.pooling == PoolingMode::Mean ? 1u : 0u, cfg_.normalize ? 1u : 0u};
    const void* bufs[] = {x_.handle(), pooled_.handle()};
    // one threadgroup of 256: the kernel's tree reduction halves from nthr and assumes a power
    // of two, and the whole output vector must be visible to the normalising pass.
    //
    // The buffer count is 2 and must equal the array length. A 3 here reads one element past
    // `bufs` and binds a garbage pointer: a SIGBUS on the first embed() call, after initialize()
    // had already reported success.
    ctx_.dispatch("cpi_bert_pool_normalize", runtime::MetalContext::Grid::Groups, 1, 256, bufs,
                  nullptr, 2, &p, sizeof(p));
  }
  ctx_.commit_and_wait();

  const auto* out = static_cast<const float*>(pooled_.contents());
  const int d = (cfg_.dimension > 0 && cfg_.dimension <= H) ? cfg_.dimension : H;
  return std::vector<float>(out, out + d);
}

}  // namespace engine
