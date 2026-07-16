// Metal executor for the op-plan IR. See plan_metal_engine.hpp.

#include "engine/plan_metal_engine.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <map>
#include <stdexcept>
#include <utility>
#include <vector>

#include "engine/generation_constraints.hpp"
#include "engine/sampling.hpp"
#include "grammar/grammar_sampler.hpp"

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
  // 0 (the default every existing call gets) keeps the prefill meaning: row t is token
  // base+t of one sequence. Batched decode sets it, and then row t takes positions[t].
  std::uint32_t per_row_positions = 0;
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
// Batched paged decode. These mirror the structs in cpi_kernels.metal field for field --
// a mismatch here is silent and shows up as garbage numbers, not as a build error.
struct KvPagedParams {
  std::uint32_t kv_hidden, max_blocks, block_size, batch;
};
struct AttnPagedParams {
  std::uint32_t heads, kv_heads, head_dim, max_blocks, block_size, window;
  float scale;
  std::uint32_t batch;
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
struct QCatParams {  // fused GEMV over 2-3 concatenated matrices; MUST match cpi_kernels.metal
  std::uint32_t n0, n1, n2, in_dim, tokens, bits, group, groups, has_bias;
};

constexpr int kTG = 256;               // threads per group
constexpr int kSimdsPerTG = kTG / 32;  // = rows per threadgroup in the GEMV
// The blocked GEMMs tile 64 rows x 64 tokens: 4 simdgroups (128 threads), each owning a 32x32
// output tile of 4x4 fragments.
constexpr int kGemmBN = 64;    // MUST match GEMM_BN in the shader (fp16 tokens per tile)
constexpr int kGemmFBM = 64;  // MUST match GEMM_FBM in the shader (fp16 rows per tile)
// One simdgroup per 32x32 sub-tile of the FBM x BN output tile: at 64x64 that is 2 x 2 = 4
// simdgroups, 128 threads. DERIVED from the tile, never restated -- this line used to read
// `32 * (64 / 32) * (kGemmBN / 32)`, and when the row tile went 64 -> 128 the literal 64 stayed.
// The kernel takes its row block from GEMM_FBM itself, so it then ran half the simdgroups it
// needed and wrote only rows row0..row0+63 of every 128-row tile; the rest kept whatever was in
// the slot. Nothing failed: the GEMM only runs at T >= kGemmMinTokens and every golden prompt is
// ~10 tokens, so no gate executed it, and the prefill benchmarks that did only timed it -- half
// the writes read as a 37% speedup. metal_smoke now checks it against a CPU reference.
constexpr int kGemmTG = 32 * (kGemmFBM / 32) * (kGemmBN / 32);
// The quantized GEMM reads activations from device, not a staged tile, so a wider token tile
// buys weight reuse at no threadgroup-memory cost. It wants 128 where fp16 wants 64.
constexpr int kGemmQBN = 128;                              // MUST match GEMM_QBN in the shader
constexpr int kGemmQTG = 32 * (64 / 32) * (kGemmQBN / 32);
// Below this many tokens the GEMV wins: a GEMM tile is padded out to kGemmBN and the padding
// is wasted arithmetic. Above it the GEMV is a catastrophe -- see the dispatch below.
constexpr int kGemmMinTokens = 16;
constexpr int kGemmQBK = 32;  // MUST match GEMM_QBK in the shader (quantized)
constexpr int kQBlock = 8;  // MUST match Q_BLOCK in cpi_kernels.metal
constexpr int kArgmaxParts = 256;
constexpr int kGemvTile = 8;  // MUST match GEMV_TILE in cpi_kernels.metal

const char* op_kind_name(int k) {
  switch (static_cast<opplan::OpKind>(k)) {
    case opplan::OpKind::RmsNorm: return "RmsNorm";
    case opplan::OpKind::Gemv: return "Gemv/Gemm";
    case opplan::OpKind::Rope: return "Rope";
    case opplan::OpKind::ScaleCopy: return "ScaleCopy";
    case opplan::OpKind::CopySlot: return "CopySlot";
    case opplan::OpKind::KvStore: return "KvStore";
    case opplan::OpKind::Attention: return "Attention";
    case opplan::OpKind::GeluMul: return "GeluMul";
    case opplan::OpKind::SiluMul: return "SiluMul";
    case opplan::OpKind::AddInplace: return "AddInplace";
    case opplan::OpKind::EmbeddingLookup: return "Embedding";
    case opplan::OpKind::LmHead: return "LmHead";
    default: return "other";
  }
}

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

  // KV cache: [max_context][kv_dim] fp16, per layer -- or, in paged mode, one pool per layer
  // of num_blocks * block_size token slots shared by every sequence. The two are the same
  // bytes under different addressing: a paged slot is block * block_size + offset, which
  // equals the token position exactly when the block table is the identity map.
  const std::size_t cache_tokens =
      paged_blocks_ > 0
          ? static_cast<std::size_t>(paged_blocks_) * static_cast<std::size_t>(paged_block_size_)
          : static_cast<std::size_t>(max_context_);
  const std::size_t cache_bytes = cache_tokens * static_cast<std::size_t>(kv_dim) * 2;
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

  // CPI_METAL_PROFILE serialises the pass -- one commit per op -- and accumulates GPU time
  // by op kind. It makes the pass slower, so it reports SHARE, not speed. Two GEMM inner-loop
  // optimizations in a row bought nothing, which is what a wrong bottleneck feels like; the
  // only cure is to measure which op actually owns the seconds.
  static const bool profile = std::getenv("CPI_METAL_PROFILE") != nullptr;

  if (profile) profile_last_ = std::chrono::steady_clock::now();

  for (std::size_t oi = 0; oi < ops.size(); ++oi) {
    const opplan::Op& op = ops[oi];

    // PEEPHOLE FUSION: `X += delta` (AddInplace) immediately followed by `XNorm = rmsnorm(X)`
    // is one fused pass -- the residual is read and written once instead of twice. The two ops
    // are adjacent within a layer (post-attention: the residual add and the ffn_norm). Done
    // here rather than in the shared plan builder so the CUDA path is untouched.
    if (op.kind == OpKind::AddInplace && oi + 1 < ops.size() &&
        ops[oi + 1].kind == OpKind::RmsNorm && ops[oi + 1].in == op.out) {
      const opplan::Op& nrm = ops[oi + 1];
      const int rows = nrm.rows * T;
      NormParams p{static_cast<std::uint32_t>(rows), static_cast<std::uint32_t>(nrm.cols), nrm.eps,
                   nrm.norm_offset ? 1u : 0u, nrm.weight != nullptr ? 1u : 0u};
      const void* wb = nrm.weight != nullptr ? nrm.weight : slot(op.out);
      // x = residual (add.out == norm.in), delta = add.in, out = norm.out (XNorm).
      const void* bufs[] = {slot(op.out), slot(op.in), wb, slot(nrm.out)};
      ctx_.dispatch("cpi_add_rmsnorm", G::Groups, static_cast<std::size_t>(rows), kTG, bufs, nullptr,
                    4, &p, sizeof(p));
      ++oi;  // consume the fused RmsNorm
      if (profile) profile_tick("AddRmsNorm(fused)");
      continue;
    }

    // PEEPHOLE FUSION: consecutive quantized GEMVs that share an input become ONE dispatch.
    // Decode fires Q/K/V (all reading XNorm) and gate/up (both reading XNorm) as separate tiny
    // GEMVs, each launch/latency-bound. Only in the GEMV regime (small T / decode); prefill
    // uses the blocked GEMM per op. The fused kernel routes each output row to its matrix.
    if (op.kind == OpKind::Gemv && op.qbits != 0 && T < kGemmMinTokens) {
      auto is_qgemv = [&](std::size_t j, opplan::Slot out) {
        return j < ops.size() && ops[j].kind == OpKind::Gemv && ops[j].qbits != 0 &&
               ops[j].out == out && ops[j].in == op.in;
      };
      const bool qkv = op.out == opplan::Slot::Q && is_qgemv(oi + 1, opplan::Slot::K) &&
                       is_qgemv(oi + 2, opplan::Slot::V);
      const bool gu = op.out == opplan::Slot::Gate && is_qgemv(oi + 1, opplan::Slot::Up);
      if (qkv || gu) {
        const opplan::Op& a = op;
        const opplan::Op& b = ops[oi + 1];
        const opplan::Op& c = qkv ? ops[oi + 2] : op;  // c unused when n2 == 0
        const std::uint32_t n0 = static_cast<std::uint32_t>(a.cols);
        const std::uint32_t n1 = static_cast<std::uint32_t>(b.cols);
        const std::uint32_t n2 = qkv ? static_cast<std::uint32_t>(c.cols) : 0u;
        const int gsz = (a.qgroup > 0) ? a.qgroup : a.in_dim;
        QCatParams gp{n0,
                      n1,
                      n2,
                      static_cast<std::uint32_t>(a.in_dim),
                      static_cast<std::uint32_t>(T),
                      static_cast<std::uint32_t>(a.qbits),
                      static_cast<std::uint32_t>(a.qgroup),
                      static_cast<std::uint32_t>((a.in_dim + gsz - 1) / gsz),
                      a.bias != nullptr ? 1u : 0u};
        const void* bb = a.bias != nullptr ? a.bias : a.qweight;  // bound, unread if absent
        const void* bufs[] = {slot(a.in),  a.qweight, a.qscales, slot(a.out),
                              b.qweight,    b.qscales, slot(b.out), c.qweight,
                              c.qscales,    slot(c.out), bb};
        const std::size_t offs[] = {
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            static_cast<std::size_t>(a.bias_offset) * sizeof(std::uint16_t)};
        const std::size_t total = static_cast<std::size_t>(n0 + n1 + n2);
        const std::size_t rb = (total + kSimdsPerTG - 1) / kSimdsPerTG;
        const std::size_t tiles = static_cast<std::size_t>((T + kGemvTile - 1) / kGemvTile);
        ctx_.dispatch("cpi_gemv_quant_cat", G::Groups, rb * tiles, kTG, bufs, offs, 11, &gp,
                      sizeof(gp));
        oi += qkv ? 2 : 1;
        if (profile) profile_tick(qkv ? "Gemv(qkv fused)" : "Gemv(gate/up fused)");
        continue;
      }
    }

    switch (op.kind) {
      case OpKind::EmbeddingLookup: {
        EmbedParams p{static_cast<std::uint32_t>(op.cols), static_cast<std::uint32_t>(T)};
        // A prefill chunk reads the whole prompt from the sequence buffer; decode reads
        // the single-token one. Batched decode always uses the sequence buffer -- a batch
        // of ONE is still T == 1, but its token is staged there, not in tok_buf_.
        const void* tb = (T > 1 || batch_ != nullptr) ? seq_tok_buf_.handle() : tok_buf_.handle();
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

          // The 8B's prefill is quantized, and that is where the whole prefill gap lives:
          // an fp16-only GEMM does nothing for it. Send the full 32-token tiles to the
          // QUANTIZED blocked GEMM, which dequantizes each weight once into threadgroup
          // memory and then reuses it across all 32 tokens.
          // The GEMM takes ALL the tokens, not just the whole tiles: its store already
          // guards tok >= tokens, so a partial tail tile just masks off.
          //
          // Handing the tail to the GEMV instead was costing a THIRD of the 8B's prefill. A
          // 551-token prompt chunks into 512 + 39, and those 39 leftover tokens took 1696 ms
          // against the GEMM's 2755 ms for the other 512 -- the GEMV re-streams the whole
          // model once per 8-token tile, so the tail swept 4.9 GB of weights five times over.
          // Padding the tail out to one masked GEMM tile wastes some arithmetic and is still
          // far cheaper.
          // A K-block carries ONE scale per row, so it must sit inside one quantization
          // group. gsz is 64 by default and kGemmQBK is 32, so a K-block sits inside one
          // group; a smaller group would straddle, and those ops fall back to the GEMV rather
          // than reading a wrong scale.
          const bool qgemm_ok =
              op.cols % 64 == 0 && op.in_dim % kGemmQBK == 0 && gsz >= kGemmQBK;
          const int qgemm_tokens = (qgemm_ok && T >= kGemmMinTokens) ? T : 0;

          if (profile) profile_tick("(before gemm)");
          if (qgemm_tokens > 0) {
            QuantParams gp = p;
            gp.tokens = static_cast<std::uint32_t>(qgemm_tokens);
            const std::size_t tiles =
                static_cast<std::size_t>((qgemm_tokens + kGemmQBN - 1) / kGemmQBN);
            const std::size_t groups = (static_cast<std::size_t>(op.cols) / 64) * tiles;
            ctx_.dispatch("cpi_gemm_quant", G::Groups, groups, kGemmQTG, bufs, offs, 5, &gp,
                          sizeof(gp));
            if (profile) profile_tick("Gemm(quant)");
          }

          const int qrest = T - qgemm_tokens;
          if (qrest > 0) {
            QuantParams rp = p;
            rp.tokens = static_cast<std::uint32_t>(qrest);
            const std::size_t roffs[] = {
                0, static_cast<std::size_t>(qgemm_tokens) * static_cast<std::size_t>(op.in_dim) * 2,
                static_cast<std::size_t>(qgemm_tokens) * static_cast<std::size_t>(op.cols) * 2, 0,
                static_cast<std::size_t>(op.bias_offset) * sizeof(std::uint16_t)};
            const std::size_t tiles = static_cast<std::size_t>((qrest + kGemvTile - 1) / kGemvTile);
            ctx_.dispatch("cpi_gemv_quant", G::Groups,
                          groups_for_rows(static_cast<std::size_t>(op.cols)) * tiles, kTG, bufs,
                          roffs, 5, &rp, sizeof(rp));
            if (profile) profile_tick("Gemv(quant remainder)");
          }
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
        // BLOCKED GEMM for prefill. It stages a K-block of both operands into threadgroup
        // memory, so each weight byte loaded from device is reused by all 32 tokens in the
        // tile. The two earlier attempts streamed operands from device on every K-step and
        // LOST to the GEMV -- the matrix units were never the bottleneck, the traffic
        // feeding them was.
        //
        // Decode (T=1) stays on the GEMV: a matrix unit cannot help when one operand is a
        // vector. Token remainders below 32 do too.
        const bool gemm_ok = op.cols % 128 == 0 && op.in_dim % 32 == 0;
        const int gemm_tokens = (gemm_ok && T >= kGemmMinTokens) ? T : 0;

        if (gemm_tokens > 0) {
          GemvParams gp{static_cast<std::uint32_t>(op.cols), static_cast<std::uint32_t>(op.in_dim),
                        static_cast<std::uint32_t>(gemm_tokens), op.bias != nullptr ? 1u : 0u};
          const std::size_t tiles = static_cast<std::size_t>((gemm_tokens + kGemmBN - 1) / kGemmBN);
          // The kernel maps tgid -> (row block, token tile) using out_dim / GEMM_FBM row
          // blocks, so the grid must agree. This read `op.cols / 64`, which launched exactly
          // twice the threadgroups needed; the surplus all landed on tok0 >= tokens and
          // returned immediately, so it was pure waste rather than a wrong answer -- but it
          // hid the row-tile mismatch above by looking like it covered the rows.
          const std::size_t groups = (static_cast<std::size_t>(op.cols) / kGemmFBM) * tiles;
          ctx_.dispatch("cpi_gemm_f16", G::Groups, groups, kGemmTG, bufs, offs, 4, &gp, sizeof(gp));
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
        // Batched decode needs logits for EVERY row -- each row is a different sequence's
        // next token. One vocab GEMV per row: the rows are independent, so this is a GEMM's
        // worth of work either way. A paged PREFILL takes the last-row path below instead
        // (logits_last_only), exactly as a contiguous prefill does.
        if (batch_ != nullptr && !batch_->logits_last_only) {
          for (int b = 0; b < batch_->batch; ++b) {
            const std::size_t in_off =
                static_cast<std::size_t>(b) * static_cast<std::size_t>(op.in_dim) * 2;
            const std::size_t out_off = static_cast<std::size_t>(b) *
                                        static_cast<std::size_t>(cfg_.vocab_size) * sizeof(float);
            if (op.qbits != 0) {
              const int gsz = (op.qgroup > 0) ? op.qgroup : op.in_dim;
              QuantParams p{static_cast<std::uint32_t>(op.cols),
                            static_cast<std::uint32_t>(op.in_dim),
                            1,
                            static_cast<std::uint32_t>(op.qbits),
                            static_cast<std::uint32_t>(op.qgroup),
                            static_cast<std::uint32_t>((op.in_dim + gsz - 1) / gsz),
                            0};
              const void* bufs[] = {op.qweight, slot(op.in), batch_logits_buf_.handle(),
                                    op.qscales};
              const std::size_t offs[] = {0, in_off, out_off, 0};
              ctx_.dispatch("cpi_lm_head_quant", G::Groups,
                            groups_for_rows(static_cast<std::size_t>(op.cols)), kTG, bufs, offs, 4,
                            &p, sizeof(p));
            } else {
              GemvParams p{static_cast<std::uint32_t>(op.cols),
                           static_cast<std::uint32_t>(op.in_dim), 1, 0};
              const void* bufs[] = {op.weight, slot(op.in), batch_logits_buf_.handle()};
              const std::size_t offs[] = {0, in_off, out_off};
              ctx_.dispatch("cpi_lm_head", G::Groups,
                            groups_for_rows(static_cast<std::size_t>(op.cols)), kTG, bufs, offs, 3,
                            &p, sizeof(p));
            }
          }
          break;
        }
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
        // Batched decode: the rows are N different sequences, so each takes its own
        // position rather than base+t.
        if (batch_ != nullptr) p.per_row_positions = 1;
        const void* bufs[] = {slot(op.in),
                              batch_ != nullptr ? batch_pos_buf_.handle() : pos_buf_.handle()};
        const std::size_t total = static_cast<std::size_t>(op.heads) *
                                  static_cast<std::size_t>(op.head_dim / 2) *
                                  static_cast<std::size_t>(T);
        ctx_.dispatch("cpi_rope", G::Threads, total, kTG, bufs, nullptr, 2, &p, sizeof(p));
        break;
      }
      case OpKind::KvStore: {
        if (batch_ != nullptr) {
          // Each sequence scatters its one new row into its own block.
          KvPagedParams p{static_cast<std::uint32_t>(op.kv_heads * op.head_dim),
                          static_cast<std::uint32_t>(batch_->max_blocks),
                          static_cast<std::uint32_t>(batch_->block_size),
                          static_cast<std::uint32_t>(batch_->batch)};
          const void* bufs[] = {
              slot(op.in), slot(op.in2), k_cache_[static_cast<std::size_t>(layer)].handle(),
              v_cache_[static_cast<std::size_t>(layer)].handle(), batch_bt_buf_.handle(),
              batch_pos_buf_.handle()};
          ctx_.dispatch("cpi_kv_store_batched_paged", G::Groups,
                        static_cast<std::size_t>(batch_->batch), kTG, bufs, nullptr, 6, &p,
                        sizeof(p));
          break;
        }
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
        if (batch_ != nullptr) {
          // Every row is one query attending over its OWN length, gathered through its own
          // block table -- the ragged batch the scheduler hands us.
          AttnPagedParams p{static_cast<std::uint32_t>(op.heads),
                            static_cast<std::uint32_t>(op.kv_heads),
                            static_cast<std::uint32_t>(op.head_dim),
                            static_cast<std::uint32_t>(batch_->max_blocks),
                            static_cast<std::uint32_t>(batch_->block_size),
                            static_cast<std::uint32_t>(op.full_attention ? 0 : op.sliding_window),
                            op.scale,
                            static_cast<std::uint32_t>(batch_->batch)};
          const void* bufs[] = {slot(op.in), k_cache_[static_cast<std::size_t>(layer)].handle(),
                                v_cache_[static_cast<std::size_t>(layer)].handle(), slot(op.out),
                                batch_bt_buf_.handle(), batch_seqlen_buf_.handle()};
          ctx_.dispatch("cpi_attention_decode_batched_paged", G::Groups,
                        static_cast<std::size_t>(batch_->batch) *
                            static_cast<std::size_t>(op.heads),
                        kTG, bufs, nullptr, 6, &p, sizeof(p));
          break;
        }
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
        // A prefill's threadgroups each walk the whole KV cache, so per-token attention is
        // O(T^2) in DEVICE traffic and was 23% of the 8B's prefill. The prefill kernel gives
        // one threadgroup a BLOCK of queries, so a key block it pulls in serves kQBlock of
        // them. Decode is a single query and has nothing to block over -- it keeps the
        // per-token kernel.
        if (T >= kQBlock) {
          const std::size_t blocks = static_cast<std::size_t>((T + kQBlock - 1) / kQBlock);
          // head_dim <= 128 runs the matrix-unit kernel; Gemma's 256 keeps the scalar one.
          const char* kern =
              (op.head_dim <= 128) ? "cpi_attention_prefill_mm" : "cpi_attention_prefill";
          ctx_.dispatch(kern, G::Groups, static_cast<std::size_t>(op.heads) * blocks, kTG, bufs,
                        nullptr, 5, &p, sizeof(p));
        } else {
          ctx_.dispatch("cpi_attention_decode", G::Groups,
                        static_cast<std::size_t>(op.heads) * static_cast<std::size_t>(T), kTG,
                        bufs, nullptr, 5, &p, sizeof(p));
        }
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
    if (profile) profile_tick(op_kind_name(static_cast<int>(op.kind)));
  }
}


// Commits what is encoded, waits, and charges the elapsed time to `name`. Only ever called
// under CPI_METAL_PROFILE -- it serialises the pass.
void PlanMetalEngine::profile_tick(const char* name) {
  ctx_.commit_and_wait();
  const auto now = std::chrono::steady_clock::now();
  profile_ms_[name] += std::chrono::duration<double, std::milli>(now - profile_last_).count();
  profile_last_ = now;
}

void PlanMetalEngine::dump_profile() const {
  double total = 0.0;
  for (const auto& kv : profile_ms_) total += kv.second;
  if (total <= 0.0) return;
  std::fprintf(stderr, "[metal profile] %.0f ms of GPU work, by op:\n", total);
  std::vector<std::pair<std::string, double>> rows(profile_ms_.begin(), profile_ms_.end());
  std::sort(rows.begin(), rows.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
  for (const auto& r : rows) {
    std::fprintf(stderr, "  %-22s %8.0f ms  %5.1f%%\n", r.first.c_str(), r.second,
                 100.0 * r.second / total);
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

void PlanMetalEngine::set_paged_kv(int num_blocks, int block_size) {
  paged_blocks_ = num_blocks;
  paged_block_size_ = block_size;
}

void PlanMetalEngine::prefill_paged(const std::vector<int>& tokens, int start_position,
                                    const std::vector<int>& block_table) {
  const int T = static_cast<int>(tokens.size());
  if (T <= 0) return;
  if (paged_blocks_ <= 0) {
    throw std::runtime_error("prefill_paged needs a paged KV pool: call set_paged_kv() before open()");
  }
  if (T > max_prefill_) {
    throw std::runtime_error("prefill chunk larger than the slots were sized for");
  }
  const int max_blocks = static_cast<int>(block_table.size());
  const int last_pos = start_position + T - 1;
  if (max_blocks <= last_pos / paged_block_size_) {
    throw std::runtime_error("prefill_paged: block table does not cover the chunk");
  }

  if (T > batch_cap_) {
    batch_pos_buf_ = ctx_.alloc(static_cast<std::size_t>(T) * sizeof(std::int32_t));
    batch_seqlen_buf_ = ctx_.alloc(static_cast<std::size_t>(T) * sizeof(std::int32_t));
    batch_logits_buf_ = ctx_.alloc(static_cast<std::size_t>(T) *
                                   static_cast<std::size_t>(cfg_.vocab_size) * sizeof(float));
    batch_cap_ = T;
  }
  const int bt_elems = T * max_blocks;
  if (bt_elems > batch_bt_elems_) {
    batch_bt_buf_ = ctx_.alloc(static_cast<std::size_t>(bt_elems) * sizeof(std::int32_t));
    batch_bt_elems_ = bt_elems;
  }

  auto* ids = static_cast<std::int32_t*>(seq_tok_buf_.contents());
  auto* ps = static_cast<std::int32_t*>(batch_pos_buf_.contents());
  auto* sl = static_cast<std::int32_t*>(batch_seqlen_buf_.contents());
  auto* bt = static_cast<std::int32_t*>(batch_bt_buf_.contents());
  for (int t = 0; t < T; ++t) {
    ids[t] = static_cast<std::int32_t>(tokens[static_cast<std::size_t>(t)]);
    ps[t] = static_cast<std::int32_t>(start_position + t);
    // Row t's length is its own position + 1, which IS the causal mask: token t sees
    // 0..start+t and nothing later, including the chunk-mates stored beside it.
    sl[t] = static_cast<std::int32_t>(start_position + t + 1);
    // Every row is the same sequence, so every row gets the same table. Replicating it costs
    // a few KB and keeps the kernels unaware that this case exists at all.
    for (int j = 0; j < max_blocks; ++j) {
      bt[static_cast<std::size_t>(t) * max_blocks + j] =
          static_cast<std::int32_t>(block_table[static_cast<std::size_t>(j)]);
    }
  }

  const BatchCtx bc{T, max_blocks, paged_block_size_, /*logits_last_only=*/true};
  batch_ = &bc;
  execute_ops(plan_.prologue, -1, start_position, T);
  for (const opplan::LayerPlan& lp : plan_.layers) {
    execute_ops(lp.ops, lp.layer_index, start_position, T);
  }
  execute_ops(plan_.epilogue, -1, start_position, T);
  ctx_.commit_and_wait();
  batch_ = nullptr;

  if (!ctx_.last_error().empty()) last_error_ = ctx_.last_error();
}

// One decode step for N independent sequences.
//
// Almost nothing here is batching-specific, and that is the point of the op-plan: the
// embedding lookup, every projection, every norm and the activations are row-independent
// and already run N rows at once for prefill. Only rope, the KV scatter and attention care
// which sequence a row belongs to, and execute_ops swaps those three while batch_ is set.
void PlanMetalEngine::decode_step_batched_logits(const std::vector<int>& tokens,
                                                 const std::vector<int>& positions,
                                                 const std::vector<int>& block_tables_flat,
                                                 int max_blocks,
                                                 std::vector<std::vector<float>>& out_logits) {
  const int B = static_cast<int>(tokens.size());
  out_logits.clear();
  if (B <= 0) return;
  if (paged_blocks_ <= 0) {
    throw std::runtime_error("batched decode needs a paged KV pool: call set_paged_kv() before open()");
  }
  if (positions.size() != tokens.size()) {
    throw std::runtime_error("batched decode: tokens and positions differ in length");
  }
  if (max_blocks <= 0 ||
      static_cast<int>(block_tables_flat.size()) < B * max_blocks) {
    throw std::runtime_error("batched decode: block table is smaller than batch * max_blocks");
  }
  if (B > max_prefill_) {
    throw std::runtime_error("batch larger than the slots were sized for");
  }

  // Scratch grows on demand rather than being sized for a worst-case batch.
  if (B > batch_cap_) {
    batch_pos_buf_ = ctx_.alloc(static_cast<std::size_t>(B) * sizeof(std::int32_t));
    batch_seqlen_buf_ = ctx_.alloc(static_cast<std::size_t>(B) * sizeof(std::int32_t));
    batch_logits_buf_ = ctx_.alloc(static_cast<std::size_t>(B) *
                                   static_cast<std::size_t>(cfg_.vocab_size) * sizeof(float));
    batch_cap_ = B;
  }
  const int bt_elems = B * max_blocks;
  if (bt_elems > batch_bt_elems_) {
    batch_bt_buf_ = ctx_.alloc(static_cast<std::size_t>(bt_elems) * sizeof(std::int32_t));
    batch_bt_elems_ = bt_elems;
  }

  // Unified memory: writing these IS the upload.
  auto* ids = static_cast<std::int32_t*>(seq_tok_buf_.contents());
  auto* ps = static_cast<std::int32_t*>(batch_pos_buf_.contents());
  auto* sl = static_cast<std::int32_t*>(batch_seqlen_buf_.contents());
  auto* bt = static_cast<std::int32_t*>(batch_bt_buf_.contents());
  for (int b = 0; b < B; ++b) {
    ids[b] = static_cast<std::int32_t>(tokens[static_cast<std::size_t>(b)]);
    ps[b] = static_cast<std::int32_t>(positions[static_cast<std::size_t>(b)]);
    // The attention kernel takes a LENGTH, so it is the new token's position plus one.
    sl[b] = static_cast<std::int32_t>(positions[static_cast<std::size_t>(b)] + 1);
  }
  for (int i = 0; i < bt_elems; ++i) {
    bt[i] = static_cast<std::int32_t>(block_tables_flat[static_cast<std::size_t>(i)]);
  }

  const BatchCtx bc{B, max_blocks, paged_block_size_, /*logits_last_only=*/false};
  batch_ = &bc;
  // `position` is unused by every op that reads it while batched (rope and the paged ops
  // take per-row positions instead), so 0 is not a lie here -- it is simply not consulted.
  execute_ops(plan_.prologue, -1, 0, B);
  for (const opplan::LayerPlan& lp : plan_.layers) {
    execute_ops(lp.ops, lp.layer_index, 0, B);
  }
  execute_ops(plan_.epilogue, -1, 0, B);
  ctx_.commit_and_wait();
  batch_ = nullptr;

  if (!ctx_.last_error().empty()) last_error_ = ctx_.last_error();

  const float* src = static_cast<const float*>(batch_logits_buf_.contents());
  out_logits.resize(static_cast<std::size_t>(B));
  for (int b = 0; b < B; ++b) {
    out_logits[static_cast<std::size_t>(b)].assign(
        src + static_cast<std::size_t>(b) * static_cast<std::size_t>(cfg_.vocab_size),
        src + static_cast<std::size_t>(b + 1) * static_cast<std::size_t>(cfg_.vocab_size));
  }
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
  prev_seq_.clear();  // this path rewrites the KV cache untracked; drop any shared-prefix state

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
    if (detail::dispatch_has_degenerate_tail(out, prompt.size())) break;  // loop guard (parity)
    if (pos + 1 >= max_context_) break;
  }
  return out;
}

// Streaming generate: the same prefill + decode as generate(), but every token is handed to
// on_token so the serving layer can stream it. Always goes through the host-side sampler (even
// for greedy) because streaming needs each token on the host anyway.
std::vector<int> PlanMetalEngine::generate_stream(const std::vector<int>& prompt, int max_new,
                                                  const Sampling& s,
                                                  const std::function<bool(int)>& on_token,
                                                  const GenerationConstraints* constraints) {
  if (prompt.empty()) throw std::runtime_error("empty prompt");
  detail::dispatch_seed_sampler_rng(s.seed);

  // Grammar-constrained decode (JSON / structured output), at parity with the CUDA/CPU path:
  // mask logits for tokens that cannot continue the grammar BEFORE sampling, then advance the
  // grammar with the chosen token. min_new_tokens suppresses EOS so a collapsed distribution
  // cannot terminate early.
  grammar::GrammarSampler* grammar = constraints ? constraints->grammar : nullptr;
  const int min_new = constraints ? std::max(0, constraints->min_new_tokens) : 0;
  if (constraints != nullptr && constraints->seed >= 0) {
    detail::dispatch_seed_sampler_rng(static_cast<unsigned>(constraints->seed));
  }

  std::vector<int> out;
  std::vector<int> history = prompt;

  const int n_pre = static_cast<int>(prompt.size()) - 1;

  // SHARED-PREFIX REUSE. If this prompt extends the last sequence this engine ran (a multi-turn
  // chat, or many requests behind one system prompt), the KV cache still holds the common prefix
  // -- same tokens at the same positions -- so re-prefill only the new suffix. This is the Metal
  // sibling of the CUDA serving's prefix reuse; it cuts time-to-first-token on repeated prefixes.
  int shared = 0;
  while (shared < n_pre && shared < static_cast<int>(prev_seq_.size()) &&
         prompt[shared] == prev_seq_[shared]) {
    ++shared;
  }

  const auto pre0 = std::chrono::steady_clock::now();
  for (int off = shared; off < n_pre; off += max_prefill_) {
    const int chunk = std::min(max_prefill_, n_pre - off);
    const std::vector<int> ids(prompt.begin() + off, prompt.begin() + off + chunk);
    encode_prefill(ids, off);  // start position == off, so reused-prefix positions line up
    ctx_.commit_and_wait();
  }
  prefill_ms_ =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - pre0).count();
  prefill_tokens_ = n_pre - shared;

  const auto dec0 = std::chrono::steady_clock::now();
  int pos = n_pre;
  int next = prompt.back();
  for (int i = 0; i < max_new; ++i, ++pos) {
    std::vector<float> lg = forward_token(next, pos);
    if (grammar != nullptr) grammar->apply_mask(lg);
    if (i < min_new && s.eos_id >= 0 && s.eos_id < static_cast<int>(lg.size())) {
      lg[static_cast<std::size_t>(s.eos_id)] = -std::numeric_limits<float>::infinity();
    }
    // Grammar-constrained decode runs greedy: the mask already restricts to the valid
    // continuations, and sampling within them (at whatever temperature the request carried)
    // just picks a lower-probability valid token and wanders -- structured output wants the
    // most likely valid token. This matches how the CUDA/CPU JSON path behaves.
    const float temp = (grammar != nullptr) ? 0.0f : s.temperature;
    next = detail::dispatch_sample_from_logits(lg, temp, s.top_k, s.top_p, s.repetition_penalty,
                                               s.no_repeat_ngram_size, history);
    if (s.eos_id >= 0 && next == s.eos_id) break;
    if (grammar != nullptr) grammar->accept(next);  // advance grammar state by the chosen token
    out.push_back(next);
    history.push_back(next);
    if (!on_token(next)) break;
    // Loop guard (parity with the CUDA/CPU path, on by default): stop once the output has
    // collapsed into a repeated tail, so a degenerate distribution can't stream forever. A
    // grammar already bounds the output, so skip it there.
    if (grammar == nullptr && detail::dispatch_has_degenerate_tail(out, prompt.size())) break;
    if (pos + 1 >= max_context_) break;
  }
  const double decode_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - dec0).count();

  bench_stats_ = BenchmarkStats{};
  bench_stats_.prefill_ms = prefill_ms_;
  bench_stats_.decode_ms = decode_ms;
  bench_stats_.prompt_tokens = static_cast<int>(prompt.size());
  bench_stats_.generated_tokens = static_cast<int>(out.size());

  // Record the full sequence (prompt + generated) so the next request can reuse its prefix.
  // The KV cache now holds exactly these tokens at positions [0, prev_seq_.size()).
  prev_seq_ = prompt;
  prev_seq_.insert(prev_seq_.end(), out.begin(), out.end());
  return out;
}

std::vector<std::pair<int, float>> PlanMetalEngine::inspect_next_logits(
    const std::vector<int>& prompt, int top_k) {
  if (prompt.empty()) throw std::runtime_error("empty prompt");
  int pos = 0;
  const int n_pre = static_cast<int>(prompt.size()) - 1;
  for (int off = 0; off < n_pre; off += max_prefill_) {
    const int chunk = std::min(max_prefill_, n_pre - off);
    const std::vector<int> ids(prompt.begin() + off, prompt.begin() + off + chunk);
    encode_prefill(ids, pos);
    ctx_.commit_and_wait();
    pos += chunk;
  }
  std::vector<float> lg = forward_token(prompt.back(), pos);

  std::vector<std::pair<int, float>> ranked(lg.size());
  for (std::size_t i = 0; i < lg.size(); ++i) ranked[i] = {static_cast<int>(i), lg[i]};
  const int k = std::min<int>(top_k, static_cast<int>(ranked.size()));
  std::partial_sort(ranked.begin(), ranked.begin() + k, ranked.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });
  ranked.resize(k);
  return ranked;
}

std::vector<int> PlanMetalEngine::generate_greedy(const std::vector<int>& prompt, int max_new) {
  if (prompt.empty()) throw std::runtime_error("empty prompt");

  prev_seq_.clear();  // this path rewrites the KV cache untracked; drop any shared-prefix state

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
    if (detail::dispatch_has_degenerate_tail(out, prompt.size())) break;  // loop guard (parity)
    if (pos + 1 >= max_context_) break;
  }
  return out;
}

}  // namespace engine
