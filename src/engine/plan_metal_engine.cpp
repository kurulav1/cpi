// Metal executor for the op-plan IR. See plan_metal_engine.hpp.

#include "engine/plan_metal_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "engine/sampling.hpp"

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
  std::uint32_t kv_heads, head_dim, position, max_context, use_position_buffer, tokens;
};
struct AttnParams {
  std::uint32_t heads, kv_heads, head_dim, position, max_context, window;
  float scale;
  std::uint32_t use_position_buffer, tokens;
};
struct EmbedParams {
  std::uint32_t hidden, tokens;
};

// fp16 -> fp32, for the host-side quantizer. The weights are raw fp16 bytes in the
// mmap; nothing else in this file needs to interpret them.
//
// NOT in a namespace called `detail`: engine::detail is the shared sampler's namespace,
// and clang rightly calls the lookup ambiguous (MSVC silently picked one).
inline float fp16_to_f32(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h >> 15) << 31;
  const std::uint32_t exp = (h >> 10) & 0x1F;
  const std::uint32_t mant = h & 0x3FF;
  std::uint32_t bits;
  if (exp == 0) {
    bits = sign;  // denormals flush to zero; they are far below any quantization level
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112) << 23) | (mant << 13);
  }
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}

struct QuantParams {
  std::uint32_t out_dim, in_dim, tokens, bits, group, groups, has_bias;
};

constexpr int kTG = 256;               // threads per group
constexpr int kSimdsPerTG = kTG / 32;  // = rows per threadgroup in the GEMV
constexpr int kArgmaxParts = 256;
constexpr int kGemvTile = 8;  // MUST match GEMV_TILE in cpi_kernels.metal

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

  // Quantizes on the HOST at load. CUDA quantizes on the GPU, but that is only worth
  // two extra kernels when you already have them; this runs once per tensor at startup.
  //
  // The format must match kernels_weight_only_matvec.cu EXACTLY, or the model produces
  // fluent nonsense rather than failing:
  //   scale = group_max_abs / 7   (7 = int4's max level; 127 for int8), floored at 1e-8
  //   q     = round(w / scale), clamped to [-8, 7]
  //   nibble = q < 0 ? q + 16 : q, packed low-then-high within each byte
  opplan::QuantWeight quant(const std::string& name, int out_dim, int in_dim) const override {
    if (bits_ == 0 || !wl_.has_tensor(name)) return {};

    auto it = qbufs_.find(name);
    if (it == qbufs_.end()) {
      const auto* src = reinterpret_cast<const std::uint16_t*>(wl_.tensor_data(name));
      const int gsz = (group_ > 0) ? group_ : in_dim;
      const int groups = (in_dim + gsz - 1) / gsz;
      const float max_q = (bits_ == 4) ? 7.0f : 127.0f;

      const std::size_t packed_row = (bits_ == 4) ? static_cast<std::size_t>((in_dim + 1) / 2)
                                                  : static_cast<std::size_t>(in_dim);
      std::vector<std::uint8_t> packed(static_cast<std::size_t>(out_dim) * packed_row, 0);
      std::vector<float> scales(static_cast<std::size_t>(out_dim) * groups, 0.0f);

      for (int r = 0; r < out_dim; ++r) {
        const std::uint16_t* row = src + static_cast<std::size_t>(r) * in_dim;
        for (int g = 0; g < groups; ++g) {
          const int j0 = g * gsz;
          const int j1 = std::min(in_dim, j0 + gsz);

          float amax = 0.0f;
          for (int j = j0; j < j1; ++j) {
            amax = std::max(amax, std::fabs(fp16_to_f32(row[j])));
          }
          float scale = amax / max_q;
          if (scale < 1.0e-8f) scale = 1.0e-8f;
          scales[static_cast<std::size_t>(r) * groups + g] = scale;

          const float inv = 1.0f / scale;
          for (int j = j0; j < j1; ++j) {
            int q = static_cast<int>(std::lround(fp16_to_f32(row[j]) * inv));
            if (bits_ == 4) {
              q = std::max(-8, std::min(7, q));
              const std::uint8_t nib = static_cast<std::uint8_t>(q < 0 ? q + 16 : q);
              std::uint8_t& byte = packed[static_cast<std::size_t>(r) * packed_row +
                                          static_cast<std::size_t>(j / 2)];
              if ((j & 1) == 0) {
                byte = static_cast<std::uint8_t>((byte & 0xF0u) | nib);  // even column = low
              } else {
                byte = static_cast<std::uint8_t>((byte & 0x0Fu) | (nib << 4));  // odd = high
              }
            } else {
              q = std::max(-127, std::min(127, q));
              packed[static_cast<std::size_t>(r) * packed_row + static_cast<std::size_t>(j)] =
                  static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
            }
          }
        }
      }

      QBuf qb;
      qb.packed = ctx_.alloc_from(packed.data(), packed.size());
      qb.scales = ctx_.alloc_from(scales.data(), scales.size() * sizeof(float));
      qb.groups = groups;
      it = qbufs_.emplace(name, std::move(qb)).first;
    }

    opplan::QuantWeight q;
    q.packed = it->second.packed.handle();
    q.scales = it->second.scales.handle();
    q.bits = bits_;
    q.group = group_;
    return q;
  }

  void set_quant(int bits, int group) {
    bits_ = bits;
    group_ = group;
  }
  std::size_t bytes() const {
    std::size_t n = 0;
    for (const auto& kv : qbufs_) {
      n += kv.second.packed.size() + kv.second.scales.size();
    }
    return n;
  }

private:
  struct QBuf {
    runtime::MetalBuffer packed;
    runtime::MetalBuffer scales;
    int groups = 1;
  };

  runtime::MetalContext& ctx_;
  model::WeightLoader& wl_;
  std::unordered_map<std::string, runtime::MetalBuffer>& bufs_;
  mutable std::unordered_map<std::string, QBuf> qbufs_;
  int bits_ = 0;
  int group_ = 0;
};

PlanMetalEngine::PlanMetalEngine() = default;
PlanMetalEngine::~PlanMetalEngine() = default;

bool PlanMetalEngine::available() const {
  return ctx_.available();
}

std::string PlanMetalEngine::device_name() const {
  return ctx_.device_name();
}

std::size_t PlanMetalEngine::weight_bytes() const {
  std::size_t n = 0;
  for (const auto& kv : wbuf_) n += kv.second.size();  // fp16 (embeddings, norms, ...)
  if (wsrc_) n += wsrc_->bytes();                      // quantized projections
  return n;
}

void* PlanMetalEngine::slot(opplan::Slot s) const {
  return slots_[static_cast<int>(s)].handle();
}

void PlanMetalEngine::open(const std::string& weights_path, int max_context, int quant_bits,
                           int quant_group) {
  if (!ctx_.available()) {
    throw std::runtime_error("no Metal GPU: " + ctx_.last_error());
  }
  if (!ctx_.load_library()) {
    throw std::runtime_error("could not load the shader library: " + ctx_.last_error());
  }

  weights_.open(weights_path);
  cfg_ = weights_.config();
  max_context_ = max_context;
  // Slots must hold a whole prefill chunk. 512 tokens of the widest slot (the MLP
  // intermediate) is only a few MB, and unified memory means it costs nothing to move.
  max_prefill_ = std::min(512, max_context);

  if (cfg_.is_moe()) {
    throw std::runtime_error("PlanMetalEngine: MoE models are not supported yet");
  }
  if (cfg_.use_layernorm) {
    throw std::runtime_error("PlanMetalEngine: true LayerNorm (mean+variance) is not implemented");
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
  g.mlp_gelu = cfg_.mlp_gelu;
  // norm_offset stays FALSE for the .ll2c path. Gemma's HF checkpoint stores its RMSNorm
  // weights as (w - 1), but CPI's converter already folds the +1 in -- which is why
  // LlamaEngine runs gemma-2b through plain launch_rmsnorm, not launch_rmsnorm_offset.
  // Adding it again scales by ~2 at every norm; that compounds and overflows fp16 by
  // layer 2, which is exactly how this was found (NaN at layers=2, finite at layers=1).
  // The (1+w) form is still needed for Gemma 4, which comes from the .cpi container.
  g.norm_offset = false;

  wsrc_ = std::make_unique<MetalWeights>(ctx_, weights_, wbuf_);
  if (quant_bits == 4 || quant_bits == 8) {
    // A group size that is a multiple of 8 keeps each 8-weight chunk the int4 kernel
    // loads inside a single scale group.
    // The int4 kernel loads 32 weights per uint4, so the group must be a multiple of 32
    // for all 32 to share one scale.
    wsrc_->set_quant(quant_bits, quant_group > 0 ? quant_group : 64);
    quant_bits_ = quant_bits;
  }
  plan_ = opplan::build_llama_plan(g, *wsrc_);

  // Slots, sized for a whole prefill chunk: [tokens][dim] contiguous, which is the
  // layout every batched kernel assumes.
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
    slots_[i] = ctx_.alloc(slot_elems(static_cast<opplan::Slot>(i)) *
                           static_cast<std::size_t>(max_prefill_) * sizeof(std::uint16_t));
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
  seq_tok_buf_ = ctx_.alloc(static_cast<std::size_t>(max_prefill_) * sizeof(std::int32_t));
  pos_buf_ = ctx_.alloc(sizeof(std::int32_t));
  logits_buf_ = ctx_.alloc(static_cast<std::size_t>(cfg_.vocab_size) * sizeof(float));
  argmax_val_ = ctx_.alloc(kArgmaxParts * sizeof(float));
  argmax_idx_ = ctx_.alloc(kArgmaxParts * sizeof(std::int32_t));
  argmax_out_ = ctx_.alloc(sizeof(std::int32_t));

  logits_.resize(static_cast<std::size_t>(cfg_.vocab_size));
}

void PlanMetalEngine::execute_ops(const std::vector<opplan::Op>& ops, int layer, int position,
                                  int T) {
  using opplan::OpKind;
  using G = runtime::MetalContext::Grid;

  for (const opplan::Op& op : ops) {
    switch (op.kind) {
      case OpKind::EmbeddingLookup: {
        EmbedParams p{static_cast<std::uint32_t>(op.cols), static_cast<std::uint32_t>(T)};
        // A prefill chunk reads the whole prompt from the sequence buffer; decode reads
        // the single-token one.
        const void* tb = (T > 1) ? seq_tok_buf_.handle() : tok_buf_.handle();
        const void* bufs[] = {op.weight, tb, slot(op.out)};
        ctx_.dispatch("cpi_embedding_lookup", G::Threads,
                      static_cast<std::size_t>(op.cols) * static_cast<std::size_t>(T), kTG, bufs,
                      nullptr, 3, &p, sizeof(p));
        break;
      }
      case OpKind::RmsNorm: {
        // Slots are [token][rows*cols] contiguous, so N tokens is simply N times as many
        // normalisation groups -- no separate kernel.
        const int rows = op.rows * T;
        NormParams p{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(op.cols), op.eps,
                     op.norm_offset ? 1u : 0u, op.weight != nullptr ? 1u : 0u};
        // A weightless norm still needs something bound at index 1; has_weight
        // being 0 means the shader never reads it.
        const void* wb = op.weight != nullptr ? op.weight : slot(op.in);
        const void* bufs[] = {slot(op.in), wb, slot(op.out)};
        ctx_.dispatch("cpi_rmsnorm", G::Groups, static_cast<std::size_t>(rows), kTG, bufs, nullptr,
                      3, &p, sizeof(p));
        break;
      }
      case OpKind::Gemv: {
        if (op.qbits != 0) {
          const int gsz = (op.qgroup > 0) ? op.qgroup : op.in_dim;
          QuantParams p{static_cast<std::uint32_t>(op.cols),
                        static_cast<std::uint32_t>(op.in_dim),
                        static_cast<std::uint32_t>(T),
                        static_cast<std::uint32_t>(op.qbits),
                        static_cast<std::uint32_t>(op.qgroup),
                        static_cast<std::uint32_t>((op.in_dim + gsz - 1) / gsz),
                        op.bias != nullptr ? 1u : 0u};
          // Quantizing the WEIGHTS does not remove the bias. Qwen2's Q/K/V have one,
          // and dropping it yields fluent nonsense rather than an error.
          const void* bb = op.bias != nullptr ? op.bias : op.qweight;  // bound, unread if absent
          const void* bufs[] = {op.qweight, slot(op.in), slot(op.out), op.qscales, bb};
          const std::size_t offs[] = {
              0, 0, 0, 0, static_cast<std::size_t>(op.bias_offset) * sizeof(std::uint16_t)};
          const std::size_t tiles = static_cast<std::size_t>((T + kGemvTile - 1) / kGemvTile);
          ctx_.dispatch("cpi_gemv_quant", G::Groups,
                        groups_for_rows(static_cast<std::size_t>(op.cols)) * tiles, kTG, bufs, offs,
                        5, &p, sizeof(p));
          break;
        }
        const void* bb = op.bias != nullptr ? op.bias : op.weight;  // bound, unread when absent
        const void* bufs[] = {op.weight, slot(op.in), slot(op.out), bb};
        // Q/K/V share one fused bqkv tensor; bias_offset selects this op's slice.
        const std::size_t offs[] = {
            0, 0, 0, static_cast<std::size_t>(op.bias_offset) * sizeof(std::uint16_t)};

        // A prefill is a MATMUL. Send the full 8-token tiles to the simdgroup-matrix GEMM
        // (Metal's matrix units) and let the GEMV mop up the remainder. Decode is T=1 and
        // stays entirely on the GEMV -- a matrix unit cannot help when one operand is a
        // vector.
        const int gemm_tokens = (op.cols % 8 == 0 && op.in_dim % 8 == 0) ? (T / 8) * 8 : 0;

        if (gemm_tokens >= 8) {
          GemvParams gp{static_cast<std::uint32_t>(op.cols), static_cast<std::uint32_t>(op.in_dim),
                        static_cast<std::uint32_t>(gemm_tokens), op.bias != nullptr ? 1u : 0u};
          // Each simdgroup owns 8 rows x 32 tokens, so a weight tile serves four token
          // tiles. That reuse -- not the matrix units on their own -- is what beats the GEMV.
          const std::size_t tiles = (static_cast<std::size_t>(op.cols) / 8) *
                                    ((static_cast<std::size_t>(gemm_tokens) + 31) / 32);
          const std::size_t groups = (tiles + kSimdsPerTG - 1) / kSimdsPerTG;
          ctx_.dispatch("cpi_gemm_f16", G::Groups, groups, kTG, bufs, offs, 4, &gp, sizeof(gp));
        }

        const int rest = T - gemm_tokens;
        if (rest > 0) {
          GemvParams p{static_cast<std::uint32_t>(op.cols), static_cast<std::uint32_t>(op.in_dim),
                       static_cast<std::uint32_t>(rest), op.bias != nullptr ? 1u : 0u};
          // The remainder tokens start at gemm_tokens: offset both in and out.
          const std::size_t roffs[] = {
              0, static_cast<std::size_t>(gemm_tokens) * static_cast<std::size_t>(op.in_dim) * 2,
              static_cast<std::size_t>(gemm_tokens) * static_cast<std::size_t>(op.cols) * 2,
              static_cast<std::size_t>(op.bias_offset) * sizeof(std::uint16_t)};
          const std::size_t tiles = static_cast<std::size_t>((rest + kGemvTile - 1) / kGemvTile);
          ctx_.dispatch("cpi_gemv_f16", G::Groups,
                        groups_for_rows(static_cast<std::size_t>(op.cols)) * tiles, kTG, bufs,
                        roffs, 4, &p, sizeof(p));
        }
        break;
      }
      case OpKind::LmHead: {
        if (op.qbits != 0) {
          const int gsz = (op.qgroup > 0) ? op.qgroup : op.in_dim;
          QuantParams p{static_cast<std::uint32_t>(op.cols),
                        static_cast<std::uint32_t>(op.in_dim),
                        1,
                        static_cast<std::uint32_t>(op.qbits),
                        static_cast<std::uint32_t>(op.qgroup),
                        static_cast<std::uint32_t>((op.in_dim + gsz - 1) / gsz),
                        0};
          const void* bufs[] = {op.qweight, slot(op.in), logits_buf_.handle(), op.qscales};
          const std::size_t offs[] = {
              0, static_cast<std::size_t>(T - 1) * static_cast<std::size_t>(op.in_dim) * 2, 0, 0};
          ctx_.dispatch("cpi_lm_head_quant", G::Groups,
                        groups_for_rows(static_cast<std::size_t>(op.cols)), kTG, bufs, offs, 4, &p,
                        sizeof(p));
          break;
        }
        // Only the LAST token of a chunk needs logits -- the others exist only to fill the
        // KV cache. Running the vocab GEMV for all T would dominate a prefill.
        GemvParams p{static_cast<std::uint32_t>(op.cols), static_cast<std::uint32_t>(op.in_dim), 1,
                     0};
        const void* bufs[] = {op.weight, slot(op.in), logits_buf_.handle()};
        const std::size_t offs[] = {
            0, static_cast<std::size_t>(T - 1) * static_cast<std::size_t>(op.in_dim) * 2, 0};
        ctx_.dispatch("cpi_lm_head", G::Groups, groups_for_rows(static_cast<std::size_t>(op.cols)),
                      kTG, bufs, offs, 3, &p, sizeof(p));
        break;
      }
      case OpKind::Rope: {
        RopeParams p{static_cast<std::uint32_t>(op.heads),
                     static_cast<std::uint32_t>(op.head_dim),
                     static_cast<std::uint32_t>(position),
                     static_cast<std::uint32_t>(T),
                     op.scale,  // the builder folds rope_theta into scale
                     1,
                     0};
        const void* bufs[] = {slot(op.in), pos_buf_.handle()};
        const std::size_t total = static_cast<std::size_t>(op.heads) *
                                  static_cast<std::size_t>(op.head_dim / 2) *
                                  static_cast<std::size_t>(T);
        ctx_.dispatch("cpi_rope", G::Threads, total, kTG, bufs, nullptr, 2, &p, sizeof(p));
        break;
      }
      case OpKind::KvStore: {
        KvParams p{static_cast<std::uint32_t>(op.kv_heads),
                   static_cast<std::uint32_t>(op.head_dim),
                   static_cast<std::uint32_t>(position),
                   static_cast<std::uint32_t>(max_context_),
                   1,
                   static_cast<std::uint32_t>(T)};
        const void* bufs[] = {
            slot(op.in), slot(op.in2), k_cache_[static_cast<std::size_t>(layer)].handle(),
            v_cache_[static_cast<std::size_t>(layer)].handle(), pos_buf_.handle()};
        const std::size_t total = static_cast<std::size_t>(op.kv_heads) *
                                  static_cast<std::size_t>(op.head_dim) *
                                  static_cast<std::size_t>(T);
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
                     1,
                     static_cast<std::uint32_t>(T)};
        const void* bufs[] = {slot(op.in), k_cache_[static_cast<std::size_t>(layer)].handle(),
                              v_cache_[static_cast<std::size_t>(layer)].handle(), slot(op.out),
                              pos_buf_.handle()};
        ctx_.dispatch("cpi_attention_decode", G::Groups,
                      static_cast<std::size_t>(op.heads) * static_cast<std::size_t>(T), kTG, bufs,
                      nullptr, 5, &p, sizeof(p));
        break;
      }
      case OpKind::SiluMul:
      case OpKind::GeluMul: {
        const std::size_t n = static_cast<std::size_t>(op.cols) * static_cast<std::size_t>(T);
        ElemParams p{static_cast<std::uint32_t>(n), 0.0f};
        const void* bufs[] = {slot(op.in), slot(op.in2), slot(op.out)};
        ctx_.dispatch(op.kind == OpKind::SiluMul ? "cpi_silu_mul" : "cpi_gelu_mul", G::Threads, n,
                      kTG, bufs, nullptr, 3, &p, sizeof(p));
        break;
      }
      case OpKind::AddInplace: {
        const std::size_t n =
            static_cast<std::size_t>(cfg_.hidden_size) * static_cast<std::size_t>(T);
        ElemParams p{static_cast<std::uint32_t>(n), 0.0f};
        const void* bufs[] = {slot(op.in), slot(op.out)};
        ctx_.dispatch("cpi_add_inplace", G::Threads, n, kTG, bufs, nullptr, 2, &p, sizeof(p));
        break;
      }
      case OpKind::ScaleCopy: {
        const std::size_t n = static_cast<std::size_t>(op.cols) * static_cast<std::size_t>(T);
        ElemParams p{static_cast<std::uint32_t>(n), op.scale};
        const void* bufs[] = {slot(op.in), slot(op.out)};
        ctx_.dispatch("cpi_scale_copy", G::Threads, n, kTG, bufs, nullptr, 2, &p, sizeof(p));
        break;
      }
      case OpKind::CopySlot: {
        const std::size_t n = static_cast<std::size_t>(op.cols) * static_cast<std::size_t>(T);
        ElemParams p{static_cast<std::uint32_t>(n), 0.0f};
        const void* bufs[] = {slot(op.in), slot(op.out)};
        ctx_.dispatch("cpi_copy", G::Threads, n, kTG, bufs, nullptr, 2, &p, sizeof(p));
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
  // CPI_METAL_LAYERS=N runs only the first N layers. A NaN has to start SOMEWHERE, and
  // bisecting on the layer count finds where far faster than reasoning about which op
  // could overflow.
  static const int layer_limit = [] {
    const char* e = std::getenv("CPI_METAL_LAYERS");
    return e != nullptr ? std::atoi(e) : -1;
  }();

  // Unified memory: writing the token and the position IS the H2D transfer.
  *static_cast<std::int32_t*>(tok_buf_.contents()) = static_cast<std::int32_t>(token);
  *static_cast<std::int32_t*>(pos_buf_.contents()) = static_cast<std::int32_t>(position);

  // CPI_METAL_DEBUG prints max|X| after every layer. Unified memory makes this almost
  // free to do: the residual stream is host-addressable, so an explosion can simply be
  // WATCHED rather than deduced.
  static const bool dbg = std::getenv("CPI_METAL_DEBUG") != nullptr;
  auto peek = [&](const char* what) {
    if (!dbg) return;
    ctx_.commit_and_wait();
    const auto* x = static_cast<const std::uint16_t*>(slots_[0].contents());
    float m = 0.0f;
    bool nan = false;
    for (int i = 0; i < cfg_.hidden_size; ++i) {
      const float v = fp16_to_f32(x[i]);
      if (v != v) nan = true;
      m = std::max(m, std::fabs(v));
    }
    std::fprintf(stderr, "  [dbg] %-12s max|X|=%.1f%s\n", what, m, nan ? "  <-- NaN" : "");
  };

  // Peeks an arbitrary slot, so a NaN can be traced to the exact op that made it.
  auto peek_slot = [&](opplan::Slot sl, int n, const char* what) {
    if (!dbg) return;
    ctx_.commit_and_wait();
    const auto* p = static_cast<const std::uint16_t*>(slots_[static_cast<int>(sl)].contents());
    float m = 0.0f;
    bool nan = false;
    for (int i = 0; i < n; ++i) {
      const float v = fp16_to_f32(p[i]);
      if (v != v) nan = true;
      m = std::max(m, std::fabs(v));
    }
    std::fprintf(stderr, "  [dbg]   %-8s max=%.1f%s\n", what, m, nan ? "  <-- NaN" : "");

    // On the first NaN, print the index and the two INPUTS at that index. A NaN out of
    // finite inputs is impossible, so this says which assumption is wrong.
    if (nan && sl == opplan::Slot::Inter) {
      const auto* ga = static_cast<const std::uint16_t*>(
          slots_[static_cast<int>(opplan::Slot::Gate)].contents());
      const auto* up =
          static_cast<const std::uint16_t*>(slots_[static_cast<int>(opplan::Slot::Up)].contents());
      for (int i = 0; i < n; ++i) {
        const float v = fp16_to_f32(p[i]);
        if (v != v) {
          std::fprintf(stderr, "  [dbg]     first NaN at i=%d of %d: gate=%.4f up=%.4f\n", i, n,
                       fp16_to_f32(ga[i]), fp16_to_f32(up[i]));
          break;
        }
      }
    }
  };

  execute_ops(plan_.prologue, -1, position, 1);
  peek("prologue");
  int done = 0;
  for (const opplan::LayerPlan& lp : plan_.layers) {
    if (layer_limit >= 0 && done >= layer_limit) break;
    if (dbg && lp.layer_index == 0) {
      // Walk layer 0 op by op, so the exact op that produces the NaN is visible.
      for (const opplan::Op& op : lp.ops) {
        execute_ops({op}, lp.layer_index, position, 1);
        if (op.kind == opplan::OpKind::Gemv || op.kind == opplan::OpKind::Attention ||
            op.kind == opplan::OpKind::SiluMul || op.kind == opplan::OpKind::GeluMul ||
            op.kind == opplan::OpKind::RmsNorm) {
          const int n = (op.out == opplan::Slot::Inter || op.out == opplan::Slot::Gate ||
                         op.out == opplan::Slot::Up)
                            ? cfg_.intermediate_size
                            : cfg_.hidden_size;
          static const char* kn[] = {"RmsNorm",  "Gemv",    "Rope",   "ScaleCopy",
                                     "CopySlot", "KvStore", "Attn",   "GeluMul",
                                     "AddInpl",  "Embed",   "LmHead", "SiluMul"};
          peek_slot(op.out, n, kn[static_cast<int>(op.kind)]);
        }
      }
      ++done;
      continue;
    }
    execute_ops(lp.ops, lp.layer_index, position, 1);
    if (dbg) {
      char b[32];
      std::snprintf(b, sizeof(b), "layer %d", lp.layer_index);
      peek(b);
    }
    ++done;
  }
  execute_ops(plan_.epilogue, -1, position, 1);

  if (dbg) {
    // The epilogue (final norm -> LM head) is the only thing between a finite X and a
    // NaN logit, so look at both ends of it.
    ctx_.commit_and_wait();
    const auto* xn =
        static_cast<const std::uint16_t*>(slots_[static_cast<int>(opplan::Slot::XNorm)].contents());
    float mn = 0.0f;
    bool nn = false;
    for (int i = 0; i < cfg_.hidden_size; ++i) {
      const float v = fp16_to_f32(xn[i]);
      if (v != v) nn = true;
      mn = std::max(mn, std::fabs(v));
    }
    const auto* lg = static_cast<const float*>(logits_buf_.contents());
    float ml = 0.0f;
    bool nl = false;
    for (int i = 0; i < cfg_.vocab_size; ++i) {
      if (lg[i] != lg[i]) nl = true;
      ml = std::max(ml, std::fabs(lg[i]));
    }
    std::fprintf(stderr, "  [dbg] XNorm        max=%.2f%s\n", mn, nn ? "  <-- NaN" : "");
    std::fprintf(stderr, "  [dbg] logits       max=%.2f%s\n", ml, nl ? "  <-- NaN" : "");
  }
}

// Runs T tokens through the whole tower in ONE pass, instead of T passes of one token.
// Every op already knows its token count, so nothing here is prefill-specific except
// staging the ids and the count -- which is the point of having the op-plan.
//
// The win is not arithmetic (a batched GEMV still touches every weight) but TRAFFIC:
// one pass over the weights serves T tokens instead of one, so a 12-token prompt reads
// the model once rather than twelve times. On a bandwidth-bound machine that is the
// whole game.
void PlanMetalEngine::encode_prefill(const std::vector<int>& tokens, int start_position) {
  const int T = static_cast<int>(tokens.size());
  if (T <= 0) return;
  if (T > max_prefill_) {
    throw std::runtime_error("prefill chunk larger than the slots were sized for");
  }

  auto* ids = static_cast<std::int32_t*>(seq_tok_buf_.contents());
  for (int i = 0; i < T; ++i) {
    ids[i] = static_cast<std::int32_t>(tokens[static_cast<std::size_t>(i)]);
  }
  *static_cast<std::int32_t*>(pos_buf_.contents()) = static_cast<std::int32_t>(start_position);

  execute_ops(plan_.prologue, -1, start_position, T);
  for (const opplan::LayerPlan& lp : plan_.layers) {
    execute_ops(lp.ops, lp.layer_index, start_position, T);
  }
  execute_ops(plan_.epilogue, -1, start_position, T);
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

std::vector<int> PlanMetalEngine::generate(const std::vector<int>& prompt, int max_new,
                                           const Sampling& s) {
  // Greedy keeps the on-GPU argmax: the vocab never crosses to the host, which is
  // worth a lot at 152k tokens.
  if (s.temperature <= 0.0f && s.repetition_penalty <= 1.0f && s.no_repeat_ngram_size <= 1) {
    return generate_greedy(prompt, max_new);
  }
  if (prompt.empty()) throw std::runtime_error("empty prompt");

  detail::dispatch_seed_sampler_rng(s.seed);

  std::vector<int> out;
  std::vector<int> history = prompt;  // the sampler needs it for penalties / n-grams
  int pos = 0;

  const int n_pre = static_cast<int>(prompt.size()) - 1;
  const auto pre0 = std::chrono::steady_clock::now();
  for (int off = 0; off < n_pre; off += max_prefill_) {
    const int chunk = std::min(max_prefill_, n_pre - off);
    const std::vector<int> ids(prompt.begin() + off, prompt.begin() + off + chunk);
    encode_prefill(ids, pos);
    ctx_.commit_and_wait();
    pos += chunk;
  }
  prefill_ms_ =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pre0).count();
  prefill_tokens_ = n_pre;

  int next = prompt.back();
  for (int i = 0; i < max_new; ++i, ++pos) {
    std::vector<float> lg = forward_token(next, pos);  // a copy: the sampler edits in place
    next = detail::dispatch_sample_from_logits(
        lg, s.temperature, s.top_k, s.top_p, s.repetition_penalty, s.no_repeat_ngram_size, history);
    if (s.eos_id >= 0 && next == s.eos_id) break;
    out.push_back(next);
    history.push_back(next);
    if (pos + 1 >= max_context_) break;
  }
  return out;
}

std::vector<int> PlanMetalEngine::generate_greedy(const std::vector<int>& prompt, int max_new) {
  if (prompt.empty()) throw std::runtime_error("empty prompt");

  std::vector<int> out;
  int pos = 0;

  // Prefill the prompt in chunks: one pass over the weights serves the whole chunk.
  // The last prompt token is left for the decode loop, which needs its logits anyway.
  const int n_pre = static_cast<int>(prompt.size()) - 1;
  const auto pre0 = std::chrono::steady_clock::now();
  for (int off = 0; off < n_pre; off += max_prefill_) {
    const int chunk = std::min(max_prefill_, n_pre - off);
    const std::vector<int> ids(prompt.begin() + off, prompt.begin() + off + chunk);
    encode_prefill(ids, pos);
    ctx_.commit_and_wait();
    pos += chunk;
  }
  const auto pre1 = std::chrono::steady_clock::now();
  prefill_ms_ = std::chrono::duration<double, std::milli>(pre1 - pre0).count();
  prefill_tokens_ = n_pre;

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
