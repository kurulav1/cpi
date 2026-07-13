// Metal executor for the op-plan IR. See plan_metal_engine.hpp.

#include "engine/plan_metal_engine.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace engine {

namespace {

// Parameter blocks. These MUST match the struct layouts in cpi_kernels.metal.
// They are duplicated rather than shared through a header because MSL and C++ do
// not share a compilation unit; tools/check_metal_bindings.py enforces the other
// half of the contract (the params block is always the LAST buffer binding).
struct NormParams {
  std::uint32_t rows, cols;
  float eps;
  std::uint32_t weight_offset, has_weight;
};
struct GemvParams {
  std::uint32_t out_dim, in_dim, tokens, has_bias;
};
struct RopeParams {
  std::uint32_t heads, head_dim, position, tokens;
  float theta;
  std::uint32_t use_position_buffer, row_stride;
};
struct ElemParams {
  std::uint32_t n;
  float scale;
};
struct KvParams {
  std::uint32_t kv_heads, head_dim, position, max_context, use_position_buffer;
};
struct AttnParams {
  std::uint32_t heads, kv_heads, head_dim, position, max_context, window;
  float scale;
  std::uint32_t use_position_buffer;
};
struct EmbedParams {
  std::uint32_t hidden, tokens;
};

constexpr int kTG = 256;               // threads per group
constexpr int kSimdsPerTG = kTG / 32;  // = rows per threadgroup in the GEMV
constexpr int kArgmaxParts = 256;

std::size_t groups_for_rows(std::size_t rows) {
  return (rows + kSimdsPerTG - 1) / kSimdsPerTG;
}

}  // namespace

// Resolves a tensor name to its MTLBuffer handle. The plan carries these as opaque
// void*, and execute_ops passes them straight back to Metal as buffer bindings --
// they are never dereferenced as pointers.
class PlanMetalEngine::MetalWeights : public opplan::WeightSource {
public:
  MetalWeights(runtime::MetalContext& ctx, model::WeightLoader& wl,
               std::unordered_map<std::string, runtime::MetalBuffer>& bufs)
      : ctx_(ctx), wl_(wl), bufs_(bufs) {}

  const void* fp16(const std::string& name) const override {
    auto it = bufs_.find(name);
    if (it != bufs_.end()) return it->second.handle();
    if (!wl_.has_tensor(name)) {
      throw std::runtime_error("weight not in the model file: " + name);
    }
    // Unified memory: this "upload" is a memcpy into a buffer the GPU can read.
    const std::size_t bytes = wl_.tensor_bytes(name);
    runtime::MetalBuffer b = ctx_.alloc_from(wl_.tensor_data(name), bytes);
    if (!b.valid()) throw std::runtime_error("failed to allocate a device buffer for " + name);
    const void* h = b.handle();
    bufs_.emplace(name, std::move(b));
    return h;
  }

  bool has(const std::string& name) const override {
    return wl_.has_tensor(name);
  }

private:
  runtime::MetalContext& ctx_;
  model::WeightLoader& wl_;
  std::unordered_map<std::string, runtime::MetalBuffer>& bufs_;
};

PlanMetalEngine::PlanMetalEngine() = default;
PlanMetalEngine::~PlanMetalEngine() = default;

bool PlanMetalEngine::available() const {
  return ctx_.available();
}

std::string PlanMetalEngine::device_name() const {
  return ctx_.device_name();
}

void* PlanMetalEngine::slot(opplan::Slot s) const {
  return slots_[static_cast<int>(s)].handle();
}

void PlanMetalEngine::open(const std::string& weights_path, int max_context) {
  if (!ctx_.available()) {
    throw std::runtime_error("no Metal GPU: " + ctx_.last_error());
  }
  if (!ctx_.load_library()) {
    throw std::runtime_error("could not load the shader library: " + ctx_.last_error());
  }

  weights_.open(weights_path);
  cfg_ = weights_.config();
  max_context_ = max_context;

  if (cfg_.is_moe() || cfg_.mlp_gelu) {
    throw std::runtime_error(
        "PlanMetalEngine handles dense SwiGLU decoders only (no MoE / GeGLU yet)");
  }

  const int H = cfg_.hidden_size;

  // head_dim is NOT hidden/heads in general. Qwen3-0.6B has hidden=1024, heads=16,
  // head_dim=128 -- so q_dim (2048) is twice the hidden size. Derive it from the Q
  // projection's actual shape; assuming hidden/heads builds a wrong model silently.
  const std::size_t wq_bytes = weights_.tensor_bytes("layers.0.attention.wq");
  const int q_dim =
      static_cast<int>(wq_bytes / (static_cast<std::size_t>(H) * sizeof(std::uint16_t)));
  const int head_dim = q_dim / cfg_.num_heads;
  const int kv_dim = cfg_.num_kv_heads * head_dim;
  if (head_dim <= 0 || q_dim != cfg_.num_heads * head_dim) {
    throw std::runtime_error("could not derive head_dim from the Q projection's shape");
  }

  opplan::LlamaGeometry g;
  g.num_layers = cfg_.num_layers;
  g.hidden = H;
  g.inter = cfg_.intermediate_size;
  g.heads = cfg_.num_heads;
  g.kv_heads = cfg_.num_kv_heads;
  g.head_dim = head_dim;
  g.vocab = cfg_.vocab_size;
  g.rms_eps = cfg_.norm_eps;
  g.rope_theta = cfg_.effective_rope_theta();
  g.has_qkv_bias = cfg_.has_qkv_bias;
  g.has_qk_norm = cfg_.has_qk_norm;
  g.scale_embeddings = cfg_.scale_embeddings;

  wsrc_ = std::make_unique<MetalWeights>(ctx_, weights_, wbuf_);
  plan_ = opplan::build_llama_plan(g, *wsrc_);

  // Slots. Sized for one token; a slot is just scratch.
  auto slot_elems = [&](opplan::Slot s) -> std::size_t {
    switch (s) {
      case opplan::Slot::Q:
      case opplan::Slot::Att:
        return static_cast<std::size_t>(q_dim);
      case opplan::Slot::K:
      case opplan::Slot::V:
        return static_cast<std::size_t>(kv_dim);
      case opplan::Slot::Gate:
      case opplan::Slot::Up:
      case opplan::Slot::Inter:
        return static_cast<std::size_t>(cfg_.intermediate_size);
      default:
        return static_cast<std::size_t>(H);
    }
  };
  slots_.resize(static_cast<int>(opplan::Slot::Count));
  for (int i = 0; i < static_cast<int>(opplan::Slot::Count); ++i) {
    slots_[i] = ctx_.alloc(slot_elems(static_cast<opplan::Slot>(i)) * sizeof(std::uint16_t));
  }

  // KV cache: [max_context][kv_dim] fp16, per layer.
  const std::size_t cache_bytes =
      static_cast<std::size_t>(max_context_) * static_cast<std::size_t>(kv_dim) * 2;
  k_cache_.resize(static_cast<std::size_t>(cfg_.num_layers));
  v_cache_.resize(static_cast<std::size_t>(cfg_.num_layers));
  for (int L = 0; L < cfg_.num_layers; ++L) {
    k_cache_[static_cast<std::size_t>(L)] = ctx_.alloc(cache_bytes);
    v_cache_[static_cast<std::size_t>(L)] = ctx_.alloc(cache_bytes);
  }

  tok_buf_ = ctx_.alloc(sizeof(std::int32_t));
  pos_buf_ = ctx_.alloc(sizeof(std::int32_t));
  logits_buf_ = ctx_.alloc(static_cast<std::size_t>(cfg_.vocab_size) * sizeof(float));
  argmax_val_ = ctx_.alloc(kArgmaxParts * sizeof(float));
  argmax_idx_ = ctx_.alloc(kArgmaxParts * sizeof(std::int32_t));
  argmax_out_ = ctx_.alloc(sizeof(std::int32_t));

  logits_.resize(static_cast<std::size_t>(cfg_.vocab_size));
}

void PlanMetalEngine::execute_ops(const std::vector<opplan::Op>& ops, int layer, int position) {
  using opplan::OpKind;
  using G = runtime::MetalContext::Grid;

  for (const opplan::Op& op : ops) {
    switch (op.kind) {
      case OpKind::EmbeddingLookup: {
        EmbedParams p{static_cast<std::uint32_t>(op.cols), 1};
        const void* bufs[] = {op.weight, tok_buf_.handle(), slot(op.out)};
        ctx_.dispatch("cpi_embedding_lookup", G::Threads, static_cast<std::size_t>(op.cols), kTG,
                      bufs, nullptr, 3, &p, sizeof(p));
        break;
      }
      case OpKind::RmsNorm: {
        NormParams p{static_cast<std::uint32_t>(op.rows), static_cast<std::uint32_t>(op.cols),
                     op.eps, op.norm_offset ? 1u : 0u, op.weight != nullptr ? 1u : 0u};
        // A weightless norm still needs something bound at index 1; has_weight
        // being 0 means the shader never reads it.
        const void* wb = op.weight != nullptr ? op.weight : slot(op.in);
        const void* bufs[] = {slot(op.in), wb, slot(op.out)};
        ctx_.dispatch("cpi_rmsnorm", G::Groups, static_cast<std::size_t>(op.rows), kTG, bufs,
                      nullptr, 3, &p, sizeof(p));
        break;
      }
      case OpKind::Gemv: {
        if (op.qbits != 0) {
          throw std::runtime_error("PlanMetalEngine: quantized Gemv is not implemented");
        }
        GemvParams p{static_cast<std::uint32_t>(op.cols), static_cast<std::uint32_t>(op.in_dim), 1,
                     op.bias != nullptr ? 1u : 0u};
        const void* bb = op.bias != nullptr ? op.bias : op.weight;  // bound, unread when absent
        const void* bufs[] = {op.weight, slot(op.in), slot(op.out), bb};
        // Q/K/V share one fused bqkv tensor; bias_offset selects this op's slice.
        const std::size_t offs[] = {
            0, 0, 0, static_cast<std::size_t>(op.bias_offset) * sizeof(std::uint16_t)};
        ctx_.dispatch("cpi_gemv_f16", G::Groups, groups_for_rows(static_cast<std::size_t>(op.cols)),
                      kTG, bufs, offs, 4, &p, sizeof(p));
        break;
      }
      case OpKind::LmHead: {
        GemvParams p{static_cast<std::uint32_t>(op.cols), static_cast<std::uint32_t>(op.in_dim), 1,
                     0};
        const void* bufs[] = {op.weight, slot(op.in), logits_buf_.handle()};
        ctx_.dispatch("cpi_lm_head", G::Groups, groups_for_rows(static_cast<std::size_t>(op.cols)),
                      kTG, bufs, nullptr, 3, &p, sizeof(p));
        break;
      }
      case OpKind::Rope: {
        RopeParams p{static_cast<std::uint32_t>(op.heads),
                     static_cast<std::uint32_t>(op.head_dim),
                     static_cast<std::uint32_t>(position),
                     1,
                     op.scale,  // the builder folds rope_theta into scale
                     1,
                     0};
        const void* bufs[] = {slot(op.in), pos_buf_.handle()};
        const std::size_t total =
            static_cast<std::size_t>(op.heads) * static_cast<std::size_t>(op.head_dim / 2);
        ctx_.dispatch("cpi_rope", G::Threads, total, kTG, bufs, nullptr, 2, &p, sizeof(p));
        break;
      }
      case OpKind::KvStore: {
        KvParams p{static_cast<std::uint32_t>(op.kv_heads), static_cast<std::uint32_t>(op.head_dim),
                   static_cast<std::uint32_t>(position), static_cast<std::uint32_t>(max_context_),
                   1};
        const void* bufs[] = {
            slot(op.in), slot(op.in2), k_cache_[static_cast<std::size_t>(layer)].handle(),
            v_cache_[static_cast<std::size_t>(layer)].handle(), pos_buf_.handle()};
        const std::size_t total =
            static_cast<std::size_t>(op.kv_heads) * static_cast<std::size_t>(op.head_dim);
        ctx_.dispatch("cpi_kv_store", G::Threads, total, kTG, bufs, nullptr, 5, &p, sizeof(p));
        break;
      }
      case OpKind::Attention: {
        AttnParams p{static_cast<std::uint32_t>(op.heads),
                     static_cast<std::uint32_t>(op.kv_heads),
                     static_cast<std::uint32_t>(op.head_dim),
                     static_cast<std::uint32_t>(position),
                     static_cast<std::uint32_t>(max_context_),
                     static_cast<std::uint32_t>(op.full_attention ? 0 : op.sliding_window),
                     op.scale,
                     1};
        const void* bufs[] = {slot(op.in), k_cache_[static_cast<std::size_t>(layer)].handle(),
                              v_cache_[static_cast<std::size_t>(layer)].handle(), slot(op.out),
                              pos_buf_.handle()};
        ctx_.dispatch("cpi_attention_decode", G::Groups, static_cast<std::size_t>(op.heads), kTG,
                      bufs, nullptr, 5, &p, sizeof(p));
        break;
      }
      case OpKind::SiluMul:
      case OpKind::GeluMul: {
        ElemParams p{static_cast<std::uint32_t>(op.cols), 0.0f};
        const void* bufs[] = {slot(op.in), slot(op.in2), slot(op.out)};
        ctx_.dispatch(op.kind == OpKind::SiluMul ? "cpi_silu_mul" : "cpi_gelu_mul", G::Threads,
                      static_cast<std::size_t>(op.cols), kTG, bufs, nullptr, 3, &p, sizeof(p));
        break;
      }
      case OpKind::AddInplace: {
        ElemParams p{static_cast<std::uint32_t>(cfg_.hidden_size), 0.0f};
        const void* bufs[] = {slot(op.in), slot(op.out)};
        ctx_.dispatch("cpi_add_inplace", G::Threads, static_cast<std::size_t>(cfg_.hidden_size),
                      kTG, bufs, nullptr, 2, &p, sizeof(p));
        break;
      }
      case OpKind::ScaleCopy: {
        ElemParams p{static_cast<std::uint32_t>(op.cols), op.scale};
        const void* bufs[] = {slot(op.in), slot(op.out)};
        ctx_.dispatch("cpi_scale_copy", G::Threads, static_cast<std::size_t>(op.cols), kTG, bufs,
                      nullptr, 2, &p, sizeof(p));
        break;
      }
      case OpKind::CopySlot: {
        ElemParams p{static_cast<std::uint32_t>(op.cols), 0.0f};
        const void* bufs[] = {slot(op.in), slot(op.out)};
        ctx_.dispatch("cpi_copy", G::Threads, static_cast<std::size_t>(op.cols), kTG, bufs, nullptr,
                      2, &p, sizeof(p));
        break;
      }
      default:
        throw std::runtime_error("PlanMetalEngine: unimplemented op kind");
    }
  }
}

// Encodes a whole forward WITHOUT committing, so the caller can append more work
// (the argmax) to the same command buffer and sync once instead of twice.
void PlanMetalEngine::encode_forward(int token, int position) {
  // Unified memory: writing the token and the position IS the H2D transfer.
  *static_cast<std::int32_t*>(tok_buf_.contents()) = static_cast<std::int32_t>(token);
  *static_cast<std::int32_t*>(pos_buf_.contents()) = static_cast<std::int32_t>(position);

  execute_ops(plan_.prologue, -1, position);
  for (const opplan::LayerPlan& lp : plan_.layers) {
    execute_ops(lp.ops, lp.layer_index, position);
  }
  execute_ops(plan_.epilogue, -1, position);
}

const std::vector<float>& PlanMetalEngine::forward_token(int token, int position) {
  encode_forward(token, position);
  ctx_.commit_and_wait();

  if (!ctx_.last_error().empty()) {
    last_error_ = ctx_.last_error();
  }

  const float* src = static_cast<const float*>(logits_buf_.contents());
  std::memcpy(logits_.data(), src, logits_.size() * sizeof(float));
  return logits_;
}

std::vector<int> PlanMetalEngine::generate_greedy(const std::vector<int>& prompt, int max_new) {
  if (prompt.empty()) throw std::runtime_error("empty prompt");

  std::vector<int> out;
  int pos = 0;

  // Prefill runs the same decode path one token at a time. Slow, but this backend
  // exists to be correct first; a batched prefill is a later kernel.
  for (std::size_t i = 0; i + 1 < prompt.size(); ++i, ++pos) {
    forward_token(prompt[i], pos);
  }

  int next = prompt.back();
  for (int i = 0; i < max_new; ++i, ++pos) {
    // Forward + argmax in ONE command buffer: the logits never leave the GPU, so
    // there is no reason to sync between them. Two syncs per token was pure latency.
    encode_forward(next, pos);

    // Argmax on the GPU: the vocab never crosses to the host.
    ElemParams p1{static_cast<std::uint32_t>(cfg_.vocab_size), 0.0f};
    const void* b1[] = {logits_buf_.handle(), argmax_val_.handle(), argmax_idx_.handle()};
    ctx_.dispatch("cpi_argmax_partial", runtime::MetalContext::Grid::Groups, kArgmaxParts, kTG, b1,
                  nullptr, 3, &p1, sizeof(p1));

    ElemParams p2{static_cast<std::uint32_t>(kArgmaxParts), 0.0f};
    const void* b2[] = {argmax_val_.handle(), argmax_idx_.handle(), argmax_out_.handle()};
    ctx_.dispatch("cpi_argmax_reduce", runtime::MetalContext::Grid::Groups, 1, kTG, b2, nullptr, 3,
                  &p2, sizeof(p2));
    ctx_.commit_and_wait();

    next = static_cast<int>(*static_cast<const std::int32_t*>(argmax_out_.contents()));
    out.push_back(next);
    if (pos + 1 >= max_context_) break;
  }
  return out;
}

}  // namespace engine
