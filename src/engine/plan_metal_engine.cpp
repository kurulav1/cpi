// Metal executor for the op-plan IR. See plan_metal_engine.hpp.

#include "engine/plan_metal_engine.hpp"
#include "engine/plan_model_config.hpp"
#include "model/config_json.hpp"
#include <filesystem>
#include "runtime/fp16.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <limits>
#include <map>
#include <set>
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
  // Lanes rotated per head; 0 means the whole head. MUST match RopeParams in 00_common.metal --
  // this mirror going stale is how the kernel would read a field the host never wrote.
  std::uint32_t rotary_dim = 0;
  std::uint32_t mrope_t = 0, mrope_h = 0, mrope_w = 0;
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
  // Paged prefill; read only by cpi_attention_prefill_mm. Defaulted so every existing
  // construction keeps the contiguous meaning.
  std::uint32_t paged = 0, block_size = 0;
};
// MUST match AttnSplitParams in cpi_kernels.metal, field for field.
struct AttnSplitParams {
  std::uint32_t heads, kv_heads, head_dim, position, use_position_buffer, window;
  float scale;
  std::uint32_t chunk_size, chunks;
};
struct MoeRouterParams {
  std::uint32_t experts, top_k, has_expert_scale;
};
struct MoeExpertParams {
  std::uint32_t inter, hidden, top_k, use_gelu;
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
// The engine only ever read fp16 before this; the image splice has to WRITE it.
// These were local copies that truncated the mantissa and flushed subnormals to zero. The old
// comment here read "denormals flush to zero; they are far below any quantization level" -- true
// of their magnitude, but they are not below the level at which a systematic, one-directional
// bias accumulates. See include/runtime/fp16.hpp; gated by metal_fp16_test.
inline std::uint16_t f32_to_fp16(float f) { return cpi::f32_to_f16(f); }
inline float fp16_to_f32(std::uint16_t h) { return cpi::f16_to_f32(h); }

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
// The ceiling on a prefill chunk, and what its slots may cost. Past ~1-2k tokens a chunk is long
// since large enough to fill the GPU, so the ceiling is where the returns have gone flat rather
// than a hardware limit; the budget is what keeps a wide model's slots from quietly reaching
// hundreds of MB. See open() for why chunk size matters at all.
constexpr int kPrefillMaxTokens = 2048;
constexpr std::size_t kPrefillSlotBudget = 256u * 1024u * 1024u;

constexpr int kGemmBN = 32;    // MUST match GEMM_BN in the shader (fp16 tokens per tile)
constexpr int kGemmFBM = 64;  // MUST match GEMM_FBM in the shader (fp16 rows per tile)
constexpr int kGemmSplitK = 4;  // MUST match GEMM_SPLITK in the shader
// Below this many threadgroups the fp16 GEMM does not fill the GPU, and splitting its
// reduction across kGemmSplitK threadgroups each -- which costs a float partial buffer and a
// summing pass -- wins. Above it, the grid is already wide and the extra pass is pure loss.
//
// Measured per shape at the tile this kernel uses (T=64, one dispatch each, best of 30):
//   down_proj 896x4864   28 groups   0.867 -> 0.417 ms   (2.1x)
//   q_proj     896x896   28 groups   0.141 -> 0.076 ms
//   o_proj     896x896   28 groups   0.097 -> 0.051 ms
//   gate_proj 4864x896  152 groups   0.415 -> 0.452 ms   (LOSS -- already wide)
// By T=256 every one of these shapes is past the threshold and the split is off again, which
// is what the measurements say should happen: at T=256 splitting q/o/gate all lose.
constexpr std::size_t kGemmSplitKMaxGroups = 64;
// ...and only while the token count is small. A narrow-output projection (k/v are 128 rows, so
// two row blocks) keeps a small group count at EVERY prompt length, so a grid test alone splits
// them inside the 512-token chunks of a long prefill, where splitting loses: measured 580 -> 590 ms
// at T=2041 with the grid test alone. Both conditions together confine it to the regime the
// per-shape measurements actually cover.
constexpr int kGemmSplitKMaxTokens = 128;
// One simdgroup per 32x32 sub-tile of the FBM x BN output tile: at 64x64 that is 2 x 2 = 4
// simdgroups, 128 threads. DERIVED from the tile, never restated -- this line used to read
// `32 * (64 / 32) * (kGemmBN / 32)`, and when the row tile went 64 -> 128 the literal 64 stayed.
// The kernel takes its row block from GEMM_FBM itself, so it then ran half the simdgroups it
// needed and wrote only rows row0..row0+63 of every 128-row tile; the rest kept whatever was in
// the slot. Nothing failed: the GEMM only runs at T >= kGemmMinTokens and every golden prompt is
// ~10 tokens, so no gate executed it, and the prefill benchmarks that did only timed it -- half
// the writes read as a 37% speedup. metal_smoke now checks it against a CPU reference.
// MUST match GEMM_RF / GEMM_CF in the shader: fragments of 8x8 per simdgroup.
constexpr int kGemmRF = 4;
constexpr int kGemmCF = 4;
// One simdgroup per (8*RF rows x 8*CF tokens) sub-tile of the FBM x BN output tile. DERIVED,
// never restated -- the literal that used to live here is what wrote half the rows.
constexpr int kGemmTG = 32 * (kGemmFBM / (8 * kGemmRF)) * (kGemmBN / (8 * kGemmCF));
// The quantized GEMM reads activations from device, not a staged tile, so a wider token tile
// buys weight reuse at no threadgroup-memory cost. It wants 128 where fp16 wants 64.
constexpr int kGemmQBN = 128;                              // MUST match GEMM_QBN in the shader
constexpr int kGemmQTG = 32 * (64 / 32) * (kGemmQBN / 32);
// Below this many tokens the GEMV wins: a GEMM tile is padded out to kGemmBN and the padding
// is wasted arithmetic. Above it the GEMV is a catastrophe -- see the dispatch below.
constexpr int kGemmMinTokens = 16;
constexpr int kGemmQBK = 32;  // MUST match GEMM_QBK in the shader (quantized)
constexpr int kQBlock = 8;      // MUST match Q_BLOCK in cpi_kernels.metal (scalar prefill)
constexpr int kQMMBlock = 16;   // MUST match QMM_BLOCK in cpi_kernels.metal (matrix-unit prefill)
constexpr int kKeyBlock = 32;   // MUST match KEY_BLOCK in cpi_kernels.metal (scalar / decode)
constexpr int kMMKeyBlock = 128; // MUST match MM_KEY_BLOCK (matrix-unit prefill attention)
// MUST match QP_QBLK / (QP_NSG*32) in the query-partitioned attention kernel.
constexpr int kQPBlock = 32;
constexpr int kQPThreads = 128;

// Shape specialization for cpi_attention_prefill_mm. MUST stay in the order the kernel declares
// its [[function_constant(i)]] slots: head_dim, kv_dim, q_dim.
inline void specialize_mm_attn(runtime::MetalContext& ctx, const opplan::Op& op) {
  const std::uint32_t vals[3] = {static_cast<std::uint32_t>(op.head_dim),
                                 static_cast<std::uint32_t>(op.kv_heads * op.head_dim),
                                 static_cast<std::uint32_t>(op.heads * op.head_dim)};
  ctx.set_next_specialization(vals, 3);
}
// MUST match FA_QBLK (8 * FA_QT * FA_NSG) in the register-resident attention kernel.
constexpr int kFABlock = 32;
constexpr int kFAThreads = 128;

// Does this device lay out an 8x8 simdgroup fragment the way cpi_attention_prefill_fa assumes?
//
// That kernel keeps the score matrix in registers and reduces across the lanes of a row, which
// means it depends on WHICH lane holds which element -- a mapping MSL does not document and does
// not promise to keep. So it is not assumed: the probe kernel writes the mapping out, and this
// checks every one of the 64 elements against the same expression the kernel uses. A device that
// disagrees (a future GPU, a different simd width) simply does not get the kernel; the
// phase-partitioned one stays correct everywhere and is the fallback.
//
// Run once and cached -- it costs a dispatch, and the answer cannot change under us.
inline bool simdgroup_layout_matches(runtime::MetalContext& ctx) {
  runtime::MetalBuffer probe = ctx.alloc(64 * sizeof(float));
  if (!probe.valid()) return false;
  float* got = static_cast<float*>(probe.contents());
  for (int i = 0; i < 64; ++i) got[i] = -1.0f;
  const void* bufs[] = {probe.handle()};
  ctx.dispatch("cpi_simdgroup_layout_probe", runtime::MetalContext::Grid::Groups, 1, 32, bufs,
               nullptr, 1, nullptr, 0);
  ctx.commit_and_wait();
  for (unsigned lane = 0; lane < 32u; ++lane) {
    const unsigned row = ((lane & 16u) >> 2) | ((lane & 6u) >> 1);
    const unsigned col = ((lane & 8u) >> 1) | ((lane & 1u) << 1);
    for (unsigned j = 0; j < 2u; ++j) {
      if (got[row * 8u + col + j] != static_cast<float>(lane * 2u + j)) return false;
    }
  }
  return true;
}

// Split-KV decode attention. A decode's attention grid is one threadgroup per head -- 14 of
// them for Qwen2.5-0.5B, on a GPU sized for hundreds -- so past a few hundred keys the cache
// walk is a serial loop in an idle machine. Splitting the keys across threadgroups costs a
// second pass to merge them, which is why it is not worth it on a short context: below this
// many keys the single-threadgroup kernel already wins.
constexpr int kAttnSplitMinKeys = 256;
// MUST match DEC_KEY_BLOCK / DEC_TG in cpi_kernels.metal. The decode split kernel runs a NARROW
// threadgroup -- one thread per key, so the width has to match the keys in flight, not the 256
// the rest of the file uses.
constexpr int kDecKeyBlock = 64;
constexpr int kDecTG = 64;
// Enough (head, chunk) threadgroups to fill the GPU, with a ceiling so the partial buffers stay
// small and the merge's per-thread loop over chunks stays short. 512 is measured, not reasoned:
// decode at 2048 keys reads 61.7 / 64.4 / 66.5 / 65.9 / 66.0 tok/s at a target of 128 / 256 /
// 512 / 1024 / 2048, so it climbs well past "one threadgroup per core" and then flattens. Far
// more threadgroups than the GPU has cores is the point -- each one is a short serial walk, and
// oversubscribing is what hides its latency.
constexpr int kAttnSplitTargetGroups = 512;
constexpr int kAttnSplitMaxChunks = 64;
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
    // Not in kAblatableKinds: ablating a recurrent op does not skip work, it corrupts the state
    // every later token reads. These are here so errors and the profile name them.
    case opplan::OpKind::SplitHeadHalves: return "SplitHeadHalves";
    case opplan::OpKind::SigmoidGate: return "SigmoidGate";
    case opplan::OpKind::MulVec: return "MulVec";
    case opplan::OpKind::RepeatLinearHeads: return "RepeatLinearHeads";
    case opplan::OpKind::LinearConv1d: return "LinearConv1d";
    case opplan::OpKind::LinearAttentionStep: return "LinearAttentionStep";
    default: return "other";
  }
}

// Must match the layouts in 70_linear_attn.metal exactly.
struct LinSplitParams {
  std::uint32_t heads, head_dim;
};
struct LinGateParams {
  std::uint32_t n;
};
struct LinMulVecParams {
  std::uint32_t n;
  float scale;
};
struct LinRepeatParams {
  std::uint32_t num_key_heads, head_repeat, key_head_dim, value_head_dim;
};
struct LinConvParams {
  std::uint32_t channels, kernel_size;
};
struct LinAttnParams {
  std::uint32_t key_head_dim, value_head_dim;
  float rms_eps;
};

// The delta-net family is recurrent and has no token dimension. A batched chunk would read past
// a slot sized for one token and produce plausible numbers, so it is refused here by name.
inline void require_single_token(opplan::OpKind kind, int tokens) {
  if (tokens != 1) {
    throw std::runtime_error(std::string("Metal: ") + op_kind_name(static_cast<int>(kind)) +
                             " is a per-token recurrent op and cannot run on a chunk of " +
                             std::to_string(tokens) + " tokens");
  }
}

// Every name CPI_METAL_ABLATE accepts. MUST stay in step with op_kind_name above -- a name that
// drifts out of this list stops being ablatable, and one that never matched an op is the bug the
// validation below exists to catch.
constexpr const char* kAblatableKinds[] = {"RmsNorm",  "Gemv/Gemm",  "Rope",       "ScaleCopy",
                                           "CopySlot", "KvStore",    "Attention",  "GeluMul",
                                           "SiluMul",  "AddInplace", "Embedding",  "LmHead"};

std::size_t groups_for_rows(std::size_t rows) {
  return (rows + kSimdsPerTG - 1) / kSimdsPerTG;
}

// Cuts `n` tokens into chunks of at most `max_chunk`, each a whole number of `tile` rows.
//
// The obvious loop -- take max_chunk until the tokens run out -- leaves the remainder as the
// final chunk, and a small chunk is far more expensive than its token count suggests. A GEMM's
// cost follows its tile count and how much of the GPU its grid fills, and a narrow chunk fills
// almost none of it: on an M4 a 28-token chunk of Qwen2.5-0.5B runs its projections at 0.69
// TFLOP/s where a 512-token chunk reaches 2.96. Greedy chunking therefore spends 19% of a
// 540-token prefill on the 5% of tokens that spill past the slot, and crossing 512 by two
// tokens costs 19 ms.
//
// Splitting evenly keeps every chunk wide enough to fill the GPU, and rounding the split up to
// a tile means no chunk wastes rows within one. Chunk boundaries carry absolute positions and
// the GEMM reduces over in_dim (which no split touches), so this changes speed only -- the
// token stream is unchanged, which is what the goldens check.
std::vector<int> chunk_sizes(int n, int max_chunk, int tile) {
  std::vector<int> out;
  if (n <= 0) return out;
  if (n <= max_chunk) {
    out.push_back(n);
    return out;
  }
  const int chunks = (n + max_chunk - 1) / max_chunk;  // the fewest that fit
  int base = (n + chunks - 1) / chunks;                // an even split...
  base = ((base + tile - 1) / tile) * tile;            // ...rounded up to whole tiles
  base = std::min(base, max_chunk);
  for (int off = 0; off < n; off += base) out.push_back(std::min(base, n - off));
  return out;
}

}  // namespace

// Resolves a tensor name to its MTLBuffer handle. The plan carries these as opaque
// void*, and execute_ops passes them straight back to Metal as buffer bindings --
// they are never dereferenced as pointers.
// Templated on the LOADER, not written twice. model::WeightLoader (.ll2c) and
// model::SafetensorsLoader (.cpi / HF dir) expose the same has_tensor / tensor_bytes /
// tensor_data surface, so the upload logic -- including the MoE expert fusion below -- is
// identical for both container formats and cannot drift between them.
template <typename Loader>
class PlanMetalEngine::MetalWeightsT : public PlanMetalEngine::MetalWeightsBase {
public:
  MetalWeightsT(runtime::MetalContext& ctx, Loader& wl,
                std::unordered_map<std::string, runtime::MetalBuffer>& bufs)
      : ctx_(ctx), wl_(wl), bufs_(bufs) {}

  const void* fp16(const std::string& name) const override {
    auto it = bufs_.find(name);
    if (it != bufs_.end()) return it->second.handle();
    // The MoE expert kernels want ONE tensor per layer, but a .ll2c stores one per expert.
    // The builder asks for the fused name and this materialises it -- so the layout decision
    // lives next to the memory that holds it, and the kernels stay identical to CUDA's.
    const auto ends_with = [&name](const char* suf) {
      const std::size_t n = std::strlen(suf);
      return name.size() > n && name.compare(name.size() - n, n, suf) == 0;
    };
    if (ends_with(".experts.gate_up")) return fuse_experts(name, /*gate_up=*/true);
    if (ends_with(".experts.down")) return fuse_experts(name, /*gate_up=*/false);
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

  // Concatenate a layer's per-expert w1/w3 (or w2) into the one tensor the expert kernels
  // index, matching the layout kernels_moe_experts.cu documents:
  //   gate_up: [experts][2 * inter][hidden]   -- expert e's gate rows then its up rows
  //   down:    [experts][hidden][inter]
  //
  // `name` is "<prefix>feed_forward.experts.gate_up" / ".down"; the per-expert tensors are
  // "<prefix>feed_forward.experts.<e>.w1" / ".w3" / ".w2".
  const void* fuse_experts(const std::string& name, bool gate_up) const {
    const std::string base = name.substr(0, name.rfind('.') + 1);  // "...experts."
    std::vector<std::uint16_t> fused;
    for (int e = 0;; ++e) {
      const std::string pe = base + std::to_string(e) + ".";
      const std::string a = pe + (gate_up ? "w1" : "w2");
      if (!wl_.has_tensor(a)) {
        if (e == 0) throw std::runtime_error("MoE expert weights not in the model file: " + a);
        break;  // ran out of experts
      }
      auto append = [&](const std::string& t) {
        const std::size_t bytes = wl_.tensor_bytes(t);
        const auto* src = reinterpret_cast<const std::uint16_t*>(wl_.tensor_data(t));
        fused.insert(fused.end(), src, src + bytes / sizeof(std::uint16_t));
      };
      append(a);
      if (gate_up) append(pe + "w3");  // up rows follow this expert's gate rows
    }
    runtime::MetalBuffer b = ctx_.alloc_from(fused.data(), fused.size() * sizeof(std::uint16_t));
    if (!b.valid()) throw std::runtime_error("failed to allocate a device buffer for " + name);
    const void* h = b.handle();
    bufs_.emplace(name, std::move(b));
    return h;
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

  void set_quant(int bits, int group) override {
    bits_ = bits;
    group_ = group;
  }
  std::size_t bytes() const override {
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
  Loader& wl_;
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
                           int quant_group, float rope_theta_override) {
  if (!ctx_.available()) {
    throw std::runtime_error("no Metal GPU: " + ctx_.last_error());
  }
  if (!ctx_.load_library()) {
    throw std::runtime_error("could not load the shader library: " + ctx_.last_error());
  }

  // Route by container. A `.cpi` file (or a HuggingFace safetensors DIRECTORY) is safetensors
  // layout with its config in the JSON __metadata__; anything else is a .ll2c with a typed binary
  // header. Same tensor names either way -- a .cpi repacked by ll2c_to_cpi preserves them -- so
  // the plan and every kernel are untouched by this choice.
  {
    namespace fs = std::filesystem;
    const fs::path p(weights_path);
    const bool is_dir = fs::exists(p) && fs::is_directory(p);
    const bool is_cpi = p.extension() == ".cpi";
    from_safetensors_ = is_dir || is_cpi;
  }
  if (from_safetensors_) {
    st_.open(weights_path);
    // Gemma 4 ships as a HuggingFace DIRECTORY: config.json beside the weights, and no
    // __metadata__ block at all. Its config therefore has to come from config.json, parsed by the
    // one Gemma 4 parser (engine::parse_gemma4_text_config) rather than a second copy here.
    //
    // Checking config.json BEFORE __metadata__ is deliberate: a `.cpi` written by
    // convert_gemma4.py does carry a metadata block, but in GEMMA's own schema (family / hidden /
    // vocab), which model::config_from_json does not read -- it would hand back a fully defaulted
    // config and the model would load at the wrong shape. That is the same trap probe_model fell
    // into, and a defaulted parse is indistinguishable from a successful one.
    std::string gemma_cfg;
    {
      namespace fs = std::filesystem;
      const fs::path cj = fs::path(weights_path) / "config.json";
      std::error_code ec;
      if (fs::is_directory(weights_path, ec) && fs::exists(cj, ec)) {
        const std::string raw = engine::mini::read_text_file(cj);
        if (engine::config_json_is_gemma4(raw)) gemma_cfg = raw;
      }
    }
    if (!gemma_cfg.empty()) {
      cfg_ = engine::gemma4_to_llama_config(engine::parse_gemma4_text_config(gemma_cfg));
    } else if (st_.has_metadata()) {
      cfg_ = model::config_from_json(st_.metadata_json());
    } else {
      throw std::runtime_error(
          "safetensors container has no __metadata__ config block: " + weights_path +
          " -- repack it with ll2c_to_cpi, which writes one.");
    }
  } else {
    weights_.open(weights_path);
    cfg_ = weights_.config();
  }
  // Gemma 4 is the one family here whose tensors carry HuggingFace names and whose geometry is
  // per-layer rather than uniform, so the paths below branch on it rather than on the container.
  const bool is_gemma4 = cfg_.model_family == model::ModelFamily::Gemma4;

  // --rope-theta, applied BEFORE the plan is built so the builder folds the override into every
  // Rope op's scale. This used to be dropped on the floor: the flag parsed, the run succeeded, and
  // the model silently used the container's theta. That is the one ignored flag that produces a
  // WRONG answer rather than a slow one -- a long-context checkpoint needing a different base
  // decodes garbage past the original window, with nothing to indicate the override was skipped.
  // Same precedence as the CUDA path: CLI override wins, otherwise the model file.
  if (rope_theta_override > 0.0f) {
    cfg_.rope_theta = rope_theta_override;
  }
  max_context_ = max_context;
  // max_prefill_ is set once the slot widths are known -- see below, it is priced off them.

  if (cfg_.use_layernorm) {
    throw std::runtime_error("PlanMetalEngine: true LayerNorm (mean+variance) is not implemented");
  }

  const int H = cfg_.hidden_size;

  // head_dim is NOT hidden/heads in general. Qwen3-0.6B has hidden=1024, heads=16,
  // head_dim=128 -- so q_dim (2048) is twice the hidden size. Derive it from the Q
  // projection's actual shape; assuming hidden/heads builds a wrong model silently.
  //
  // Two Qwen3.5 wrinkles, both of which convert cleanly into a wrong model rather than an error:
  //   - layer 0 is usually a DELTA-NET layer and has no attention.wq at all, so the probe has to
  //     find the first full-attention layer instead of assuming layer 0 is one;
  //   - on that family wq projects [q | gate] per head, so its row count is 2*q_dim. Halving it
  //     is what attn_output_gate is recorded in the container for.
  int q_dim = 0;
  int head_dim = 0;
  int kv_dim = 0;
  if (is_gemma4) {
    // Gemma 4 states head_dim in its config, per LAYER TYPE, and its tensors use HuggingFace
    // names -- so there is no "layers.N.attention.wq" to measure and nothing to derive. The slots
    // below are sized for the WIDER of the two types, because one buffer serves both.
    head_dim = std::max(cfg_.head_dim_sliding, cfg_.head_dim_full);
    q_dim = cfg_.num_heads * head_dim;
    kv_dim = std::max(cfg_.num_kv_heads_sliding * cfg_.head_dim_sliding,
                      cfg_.num_kv_heads_full * cfg_.head_dim_full);
    if (head_dim <= 0 || kv_dim <= 0) {
      throw std::runtime_error("Gemma 4 config carries no per-layer-type head_dim / kv heads");
    }
  } else {
    int probe_layer = 0;
    while (probe_layer < cfg_.num_layers &&
           cfg_.attention_kind_for_layer(probe_layer) == model::AttentionKind::Linear) {
      ++probe_layer;
    }
    if (probe_layer >= cfg_.num_layers) {
      throw std::runtime_error("model has no full-attention layer to derive head_dim from");
    }
    const std::size_t wq_bytes =
        raw_bytes("layers." + std::to_string(probe_layer) + ".attention.wq");
    q_dim = static_cast<int>(wq_bytes / (static_cast<std::size_t>(H) * sizeof(std::uint16_t)));
    if (cfg_.attn_output_gate) {
      q_dim /= 2;
    }
    head_dim = q_dim / cfg_.num_heads;
    kv_dim = cfg_.num_kv_heads * head_dim;
    if (head_dim <= 0 || q_dim != cfg_.num_heads * head_dim) {
      throw std::runtime_error("could not derive head_dim from the Q projection's shape");
    }
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
  // MoE (Mixtral). 0 experts leaves the dense MLP exactly as it was.
  g.num_experts = cfg_.num_local_experts;
  g.experts_per_tok = cfg_.num_experts_per_tok;
  g.expert_inter = cfg_.expert_intermediate_size;
  // norm_offset stays FALSE for the .ll2c path. Gemma's HF checkpoint stores its RMSNorm
  // weights as (w - 1), but CPI's converter already folds the +1 in -- which is why
  // LlamaEngine runs gemma-2b through plain launch_rmsnorm, not launch_rmsnorm_offset.
  // Adding it again scales by ~2 at every norm; that compounds and overflows fp16 by
  // layer 2, which is exactly how this was found (NaN at layers=2, finite at layers=1).
  // The (1+w) form is still needed for Gemma 4, which comes from the .cpi container.
  g.norm_offset = false;

  if (std::getenv("CPI_METAL_GPUPROFILE") != nullptr) {
    ctx_.enable_gpu_profile(true);
    if (!ctx_.gpu_profile_enabled()) {
      std::fprintf(stderr, "[metal gpu-profile] unavailable: %s\n", ctx_.last_error().c_str());
    }
  }
  if (from_safetensors_) {
    wsrc_ = std::make_unique<MetalWeightsT<model::SafetensorsLoader>>(ctx_, st_, wbuf_);
  } else {
    wsrc_ = std::make_unique<MetalWeightsT<model::WeightLoader>>(ctx_, weights_, wbuf_);
  }
  if (quant_bits == 4 || quant_bits == 8) {
    // A group size that is a multiple of 8 keeps each 8-weight chunk the int4 kernel
    // loads inside a single scale group.
    // The int4 kernel loads 32 weights per uint4, so the group must be a multiple of 32
    // for all 32 to share one scale.
    wsrc_->set_quant(quant_bits, quant_group > 0 ? quant_group : 64);
    quant_bits_ = quant_bits;
  }
  if (cfg_.model_family == model::ModelFamily::Qwen3_5) {
    // Mixed geometry: a "uniform" LlamaGeometry cannot describe a model whose layers are not all
    // the same kind, so this family gets its own builder over the SAME op vocabulary -- every op
    // it emits already exists and every backend runs it with the shared kernels.
    opplan::Qwen35Geometry q;
    q.num_layers = cfg_.num_layers;
    q.hidden = H;
    q.inter = cfg_.intermediate_size;
    q.vocab = cfg_.vocab_size;
    q.rms_eps = cfg_.norm_eps;
    q.rope_theta = cfg_.effective_rope_theta();
    q.heads = cfg_.num_heads;
    q.kv_heads = cfg_.num_kv_heads;
    q.head_dim = head_dim;
    // Partial RoPE: rotate only the leading fraction of each head, and only an even number of
    // lanes -- an odd rotary_dim would leave a half-pair the kernel cannot rotate.
    q.rotary_dim = static_cast<int>(static_cast<float>(head_dim) * cfg_.partial_rotary_factor);
    q.rotary_dim -= q.rotary_dim % 2;
    if (q.rotary_dim <= 0) {
      q.rotary_dim = head_dim - (head_dim % 2);
    }
    q.lin_key_heads = cfg_.linear_num_key_heads;
    q.lin_value_heads = cfg_.linear_num_value_heads;
    q.lin_key_head_dim = cfg_.linear_key_head_dim;
    q.lin_value_head_dim = cfg_.linear_value_head_dim;
    q.lin_conv_kernel = cfg_.linear_conv_kernel_dim;
    q.layer_is_linear.assign(static_cast<std::size_t>(cfg_.num_layers), false);
    for (int L = 0; L < cfg_.num_layers; ++L) {
      q.layer_is_linear[static_cast<std::size_t>(L)] =
          cfg_.attention_kind_for_layer(L) == model::AttentionKind::Linear;
    }
    plan_ = opplan::build_qwen35_plan(q, *wsrc_);
  } else if (is_gemma4) {
    opplan::Gemma4Geometry gg;
    gg.num_layers = cfg_.num_layers;
    gg.hidden = H;
    gg.inter = cfg_.intermediate_size;
    gg.vocab = cfg_.vocab_size;
    gg.heads = cfg_.num_heads;
    gg.rms_eps = cfg_.norm_eps;
    gg.sliding_window = cfg_.sliding_window;
    gg.head_dim_sliding = cfg_.head_dim_sliding;
    gg.head_dim_full = cfg_.head_dim_full;
    gg.kv_heads_sliding = cfg_.num_kv_heads_sliding;
    gg.kv_heads_full = cfg_.num_kv_heads_full;
    gg.rope_theta_sliding = cfg_.rope_theta_sliding;
    gg.rope_theta_full = cfg_.rope_theta_full;
    gg.partial_rotary_full = cfg_.partial_rotary_full;
    gg.first_shared_layer = cfg_.first_shared_layer;
    gg.attention_k_eq_v = cfg_.attention_k_eq_v;
    gg.use_double_wide_mlp = cfg_.use_double_wide_mlp;
    gg.ple = cfg_.hidden_size_per_layer_input;
    gg.layer_full.assign(static_cast<std::size_t>(cfg_.num_layers), 0);
    for (int L = 0; L < cfg_.num_layers; ++L) {
      gg.layer_full[static_cast<std::size_t>(L)] =
          cfg_.attention_kind_for_layer(L) == model::AttentionKind::Full ? 1 : 0;
    }
    // The per-layer output gain is a NUMBER the plan multiplies in, not a weight the GPU reads, so
    // it has to be pulled host-side. wsrc_->fp16() would hand back an opaque Metal buffer here --
    // dereferencing one is a SIGBUS, then silent NaN. It is bf16 in the checkpoint, and a model
    // without the tensor means a no-op gain of 1.0.
    gg.layer_scalar.assign(static_cast<std::size_t>(cfg_.num_layers), 1.0f);
    for (int L = 0; L < cfg_.num_layers; ++L) {
      const std::string n =
          "model.language_model.layers." + std::to_string(L) + ".layer_scalar";
      if (!st_.has_tensor(n)) continue;
      const auto* raw = reinterpret_cast<const std::uint16_t*>(st_.tensor_data(n));
      gg.layer_scalar[static_cast<std::size_t>(L)] = engine::mini::bf16_to_float(raw[0]);
    }
    plan_ = opplan::build_gemma4_plan(gg, *wsrc_);
  } else {
    plan_ = opplan::build_llama_plan(g, *wsrc_);
  }
  // Read off the PLAN rather than the config: the plan is what the encoder walks, so a model that
  // gained or lost a recurrent op cannot disagree with this flag.
  for (const auto& lp : plan_.layers) {
    for (const auto& op : lp.ops) {
      if (op.kind == opplan::OpKind::LinearConv1d ||
          op.kind == opplan::OpKind::LinearAttentionStep) {
        has_recurrent_ops_ = true;
      }
      // An op that reads a WINDOW of another slot (Gemma 4's per-layer-input injection: layer L
      // multiplies by per_layer_inputs[L], at aux_offset = L*ple) has the same restriction as a
      // recurrent one, for a different reason. The offset is a flat displacement into in2, but
      // slots are [token][dim] and in2's row stride (num_layers*ple) differs from the op's own
      // (ple) -- so one displacement is only correct for a single token. Batched chunks would
      // silently pair the wrong rows. Force token-by-token, like the delta-net family, until the
      // kernel takes a per-operand stride.
      if (op.aux_offset != 0) {
        has_recurrent_ops_ = true;
      }
    }
  }

  // Delta-net state. Sized from the container's v6 geometry, so it is zero-sized for every model
  // that is not a linear-attention one. The conv window holds kernel-1 previous inputs per
  // channel; the recurrent matrix is [key_head_dim x value_head_dim] per value head.
  if (cfg_.linear_num_value_heads > 0 && cfg_.linear_conv_kernel_dim > 0) {
    const std::size_t lin_k = static_cast<std::size_t>(cfg_.linear_num_key_heads) *
                              static_cast<std::size_t>(cfg_.linear_key_head_dim);
    const std::size_t lin_v = static_cast<std::size_t>(cfg_.linear_num_value_heads) *
                              static_cast<std::size_t>(cfg_.linear_value_head_dim);
    const std::size_t conv_dim = lin_k * 2u + lin_v;
    lin_conv_stride_ = conv_dim * static_cast<std::size_t>(cfg_.linear_conv_kernel_dim - 1);
    lin_recurrent_stride_ = static_cast<std::size_t>(cfg_.linear_num_value_heads) *
                            static_cast<std::size_t>(cfg_.linear_key_head_dim) *
                            static_cast<std::size_t>(cfg_.linear_value_head_dim);
    const std::size_t layers = static_cast<std::size_t>(cfg_.num_layers);
    lin_conv_state_ = ctx_.alloc(std::max<std::size_t>(1, layers * lin_conv_stride_) * sizeof(float));
    lin_recurrent_state_ =
        ctx_.alloc(std::max<std::size_t>(1, layers * lin_recurrent_stride_) * sizeof(float));
    std::memset(lin_conv_state_.contents(), 0, lin_conv_state_.size());
    std::memset(lin_recurrent_state_.contents(), 0, lin_recurrent_state_.size());
  }

  // Slots, sized for a whole prefill chunk: [tokens][dim] contiguous, which is the
  // layout every batched kernel assumes.
  // Delta-net / gated-attention geometry. Zero for every model that is not Qwen3.5, which makes
  // the slots below zero-sized there rather than wrong-sized -- the default branch would hand
  // them `hidden`, and a slot that is silently too small is read out of bounds by the first
  // kernel that uses it.
  const int lin_k_dim = cfg_.linear_num_key_heads * cfg_.linear_key_head_dim;
  const int lin_v_dim = cfg_.linear_num_value_heads * cfg_.linear_value_head_dim;
  const int lin_conv_dim = lin_k_dim * 2 + lin_v_dim;  // in_proj_qkv emits q|k|v back to back
  const int lin_qk_dim = cfg_.linear_num_value_heads * cfg_.linear_key_head_dim;

  auto slot_elems = [&](opplan::Slot s) -> std::size_t {
    switch (s) {
      // Qwen3.5 full-attention layers project [q|gate] per head, so QPair is twice q_dim; the
      // split writes Q and QGate, each q_dim wide.
      case opplan::Slot::QPair:
        return static_cast<std::size_t>(q_dim) * 2u;
      case opplan::Slot::QGate:
        return static_cast<std::size_t>(q_dim);
      // Delta-net. LinQ/LinK are per VALUE head (RepeatLinearHeads fans the shared key heads
      // out), so they are num_value_heads * key_head_dim, NOT num_key_heads * key_head_dim.
      case opplan::Slot::LinMix:
        return static_cast<std::size_t>(lin_conv_dim);
      case opplan::Slot::LinZ:
        return static_cast<std::size_t>(lin_v_dim);
      case opplan::Slot::LinA:
      case opplan::Slot::LinB:
        return static_cast<std::size_t>(std::max(1, cfg_.linear_num_value_heads));
      case opplan::Slot::LinQ:
      case opplan::Slot::LinK:
        return static_cast<std::size_t>(lin_qk_dim);
      case opplan::Slot::LinV:
      case opplan::Slot::LinAtt:
        return static_cast<std::size_t>(lin_v_dim);
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
      case opplan::Slot::MoeLogits:
        return static_cast<std::size_t>(std::max(1, cfg_.num_local_experts));
      // Gemma 4 Per-Layer Embeddings. PleRaw and PleAll hold the WHOLE packed table --
      // [num_layers][ple] -- because the prologue builds every layer's window in one pass and each
      // layer then gates its own slice out of it (GeluMul with aux_offset = L*ple). Falling
      // through to `default` would hand them `hidden`: 1536 instead of 8960 on E2B, which the
      // comment above this switch describes exactly -- silently too small, read out of bounds by
      // the first kernel that touches it. PleGate is one layer's window only.
      case opplan::Slot::PleRaw:
      case opplan::Slot::PleAll:
        return static_cast<std::size_t>(cfg_.num_layers) *
               static_cast<std::size_t>(std::max(0, cfg_.hidden_size_per_layer_input));
      case opplan::Slot::PleGate:
        return static_cast<std::size_t>(std::max(0, cfg_.hidden_size_per_layer_input));
      case opplan::Slot::MoeInter: {
        // [top_k, expert_inter]: every selected expert's act(gate)*up lives here at once, so
        // MoeDownAccum walks them in one pass.
        const int ei = cfg_.expert_intermediate_size > 0 ? cfg_.expert_intermediate_size
                                                         : cfg_.intermediate_size;
        return static_cast<std::size_t>(std::max(1, cfg_.num_experts_per_tok)) *
               static_cast<std::size_t>(std::max(1, ei));
      }
      default:
        return static_cast<std::size_t>(H);
    }
  };
  // How many tokens a prefill chunk may hold. This is the single biggest lever on prefill
  // speed, because a GEMM's efficiency climbs with its token count: on an M4 this model's
  // projections run at 0.69 TFLOP/s over 28 tokens and 2.96 over 512, so a chunk that spills a
  // short remainder pays for it twice over. Bigger is simply better -- one 635-token chunk beats
  // 512+123 by 10% -- and the only thing stopping us is the slots, which is why the budget is
  // what gets bounded here and not some round token count. (512 used to be that round number,
  // picked for being "a few MB".)
  //
  // Slots cost max_prefill * (every slot's width summed) * 2 bytes: ~37 KB per token for a 0.5B,
  // ~123 KB for an 8B, so the cap only binds on the big models -- which is the point, since
  // those are the ones that would otherwise quietly take hundreds of MB.
  std::size_t slot_bytes_per_token = 0;
  for (int i = 0; i < static_cast<int>(opplan::Slot::Count); ++i) {
    slot_bytes_per_token += slot_elems(static_cast<opplan::Slot>(i)) * sizeof(std::uint16_t);
  }
  const int budget_tokens =
      static_cast<int>(std::max<std::size_t>(1, kPrefillSlotBudget / slot_bytes_per_token));
  max_prefill_ = std::max(1, std::min({kPrefillMaxTokens, max_context, budget_tokens}));

  if (cfg_.is_moe()) {
    // The router's selection crosses ops in DEVICE buffers, so a token never round-trips to the
    // host mid-layer.
    //
    // SIZED [max_prefill][top_k], NOT [top_k]. The router, the expert FFN and the accumulate
    // are separate OPS, so execute_ops runs the router's whole token loop before the expert
    // loop starts. One shared slot would therefore hold the LAST token's experts by the time
    // the FFN reads it, and every token in the chunk would run the last token's experts. That
    // produces confident wrong tokens rather than a crash, and it is exactly what the
    // tiny-mixtral golden caught: a T=1 forward was exact (max_abs_diff 0.0005) while the
    // T=12 prefill diverged.
    const int topk = std::max(1, cfg_.num_experts_per_tok);
    const std::size_t sel = static_cast<std::size_t>(topk) * static_cast<std::size_t>(max_prefill_);
    moe_idx_buf_ = ctx_.alloc(sel * sizeof(std::int32_t));
    moe_w_buf_ = ctx_.alloc(sel * sizeof(float));
  }

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
  k_cache_.resize(static_cast<std::size_t>(cfg_.num_layers));
  v_cache_.resize(static_cast<std::size_t>(cfg_.num_layers));
  // Who owns whose cache. Identity everywhere except Gemma 4's shared tail, which reads the cache
  // of the last non-shared layer OF THE SAME TYPE (kv_source, derived once in
  // parse_gemma4_text_config). A shared layer allocates NOTHING -- it emits no KvStore either, so
  // an own cache would be read-but-never-written, which is precisely the uninitialised memory that
  // made the first Gemma-4-on-Metal run return NaN.
  kv_owner_.resize(static_cast<std::size_t>(cfg_.num_layers));
  for (int L = 0; L < cfg_.num_layers; ++L) {
    const bool shared = is_gemma4 && cfg_.first_shared_layer > 0 && L >= cfg_.first_shared_layer &&
                        L < static_cast<int>(cfg_.kv_source.size());
    kv_owner_[static_cast<std::size_t>(L)] = shared ? cfg_.kv_source[static_cast<std::size_t>(L)] : L;
  }
  if (is_gemma4 && std::getenv("CPI_GEMMA4_TRACE") != nullptr) {
    std::fprintf(stderr,
                 "[g4] layers=%d hidden=%d inter=%d vocab=%d heads=%d ple=%d first_shared=%d\n"
                 "[g4] hd_slide=%d hd_full=%d kvh_slide=%d kvh_full=%d eps=%g swin=%d dwmlp=%d\n"
                 "[g4] kv_owner[0..7]=%d %d %d %d %d %d %d %d  kv_owner[last]=%d\n",
                 cfg_.num_layers, cfg_.hidden_size, cfg_.intermediate_size, cfg_.vocab_size,
                 cfg_.num_heads, cfg_.hidden_size_per_layer_input, cfg_.first_shared_layer,
                 cfg_.head_dim_sliding, cfg_.head_dim_full, cfg_.num_kv_heads_sliding,
                 cfg_.num_kv_heads_full, cfg_.norm_eps, cfg_.sliding_window,
                 static_cast<int>(cfg_.use_double_wide_mlp), kv_owner_[0], kv_owner_[1],
                 kv_owner_[2], kv_owner_[3], kv_owner_[4], kv_owner_[5], kv_owner_[6],
                 kv_owner_[7], kv_owner_.back());
    std::fflush(stderr);
  }
  for (int L = 0; L < cfg_.num_layers; ++L) {
    if (kv_owner_[static_cast<std::size_t>(L)] != L) continue;  // reads someone else's
    // Sized PER LAYER TYPE. Gemma 4's sliding and full layers differ in both head_dim and kv-head
    // count (E2B: 256x2 vs 512x1), so one kv_dim for the whole tower is wrong for one of them --
    // too small is an out-of-bounds read, too large silently wastes hundreds of MB.
    std::size_t layer_kv_dim = static_cast<std::size_t>(kv_dim);
    if (is_gemma4) {
      const bool full = cfg_.attention_kind_for_layer(L) == model::AttentionKind::Full;
      layer_kv_dim = static_cast<std::size_t>(full ? cfg_.num_kv_heads_full * cfg_.head_dim_full
                                                   : cfg_.num_kv_heads_sliding *
                                                         cfg_.head_dim_sliding);
    }
    const std::size_t cache_bytes = cache_tokens * layer_kv_dim * 2;
    k_cache_[static_cast<std::size_t>(L)] = ctx_.alloc(cache_bytes);
    v_cache_[static_cast<std::size_t>(L)] = ctx_.alloc(cache_bytes);
  }

  tok_buf_ = ctx_.alloc(sizeof(std::int32_t));
  seq_tok_buf_ = ctx_.alloc(static_cast<std::size_t>(max_prefill_) * sizeof(std::int32_t));

  // Split-KV decode scratch, sized for the worst case (every head, every chunk) so the decode
  // path never allocates. Small enough not to argue about: ~114 KB for a 0.5B, ~512 KB for an 8B.
  const std::size_t split_slots =
      static_cast<std::size_t>(cfg_.num_heads) * static_cast<std::size_t>(kAttnSplitMaxChunks);
  attn_part_m_ = ctx_.alloc(split_slots * sizeof(float));
  attn_part_l_ = ctx_.alloc(split_slots * sizeof(float));
  attn_part_o_ =
      ctx_.alloc(split_slots * static_cast<std::size_t>(head_dim) * sizeof(float));
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
  static const std::set<std::string> ablate_ = [] {
    std::set<std::string> out;
    const char* e = std::getenv("CPI_METAL_ABLATE");
    if (e == nullptr) return out;
    std::string cur;
    for (const char* c = e;; ++c) {
      if (*c == ',' || *c == 0) {
        if (!cur.empty()) out.insert(cur);
        cur.clear();
        if (*c == 0) break;
      } else {
        cur.push_back(*c);
      }
    }
    // A name matching no op ablates NOTHING while still printing ABLATION ACTIVE below, which is
    // far worse than not ablating: the run looks like the experiment you asked for and is the
    // experiment you did not. "Attn" for "Attention" is how a prefill got profiled with all of
    // its attention still in it, and the conclusions drawn off that trace were wrong for a week.
    // So refuse to run rather than quietly measure the wrong thing.
    for (const std::string& n : out) {
      bool known = false;
      for (const char* k : kAblatableKinds) {
        if (n == k) {
          known = true;
          break;
        }
      }
      if (!known) {
        std::string msg = "CPI_METAL_ABLATE: '" + n + "' is not an op kind. Valid:";
        for (const char* k : kAblatableKinds) msg += std::string(" ") + k;
        throw std::runtime_error(msg);
      }
    }
    std::fprintf(stderr, "[metal] ABLATION ACTIVE -- output is deliberately WRONG (%s)\n", e);
    return out;
  }();

  if (profile) profile_last_ = std::chrono::steady_clock::now();

  // Tracks the previously dispatched op, for the concurrent-overlap check below. Local to this
  // execute_ops call: a new call (the next layer) starts with no predecessor, so its first op
  // barriers after everything the previous call queued.
  bool prev_dispatch_valid = false;
  opplan::OpKind prev_dispatch_kind = opplan::OpKind::RmsNorm;
  opplan::Slot prev_dispatch_in = opplan::Slot::X;
  opplan::Slot prev_dispatch_out = opplan::Slot::X;

  for (std::size_t oi = 0; oi < ops.size(); ++oi) {
    const opplan::Op& op = ops[oi];

    // CPI_METAL_ABLATE=<kind>[,<kind>] SKIPS those ops entirely. The output becomes garbage --
    // that is the point: it answers "what does this op cost?" by removing it, which no profiler
    // distortion can confuse. CPI_METAL_PROFILE cannot answer it, because serialising the pass
    // gives every tiny dispatch its own command buffer and inflates exactly the small ops you
    // are asking about (it reported 551 ms of GPU work for a pass that really takes ~190).
    // Never set this outside an experiment.
    if (!ablate_.empty() && ablate_.count(op_kind_name(static_cast<int>(op.kind))) != 0) continue;

    // CONCURRENT-DISPATCH OVERLAP. The compute encoder is concurrent and every dispatch barriers
    // first by default (serial-equivalent). A Gemv that reads the SAME input slot as the
    // immediately preceding Gemv and writes a DIFFERENT output is independent of it: the three
    // Q/K/V projections all read the post-norm slot and write distinct slots, as do gate/up. They
    // are consecutive, so nothing writes the shared input between them -- no RAW, WAR or WAW -- and
    // the second/third may skip the barrier and overlap on the GPU. That is where the win is: the
    // 128-row k/v projections are grid-starved (18 threadgroups on 10 cores) and never fill the
    // machine alone. Everything else keeps the barrier, so multi-dispatch ops and the real
    // dependency chain stay correctly ordered. prev is reset each execute_ops call, so the first op
    // of a layer (its rmsnorm) always barriers after the previous layer.
    if (op.kind == OpKind::Gemv && prev_dispatch_valid && prev_dispatch_kind == OpKind::Gemv &&
        op.in == prev_dispatch_in && op.out != prev_dispatch_out && op.out != prev_dispatch_in) {
      ctx_.set_next_barrier(false);
    }
    prev_dispatch_valid = true;
    prev_dispatch_kind = op.kind;
    prev_dispatch_in = op.in;
    prev_dispatch_out = op.out;

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
        // WHICH BUFFER THE CALLER STAGED INTO, not how many tokens there are. These are not the
        // same question: a recurrent model prefills in chunks of exactly one token, which still
        // arrive in the sequence buffer. Keying this off `T > 1` made such a chunk read tok_buf_,
        // a buffer prefill never writes -- so every prefill position re-embedded whatever decode
        // had left there, identically, and the model saw one repeated token instead of the prompt.
        const void* tb = tokens_in_seq_buf_ ? seq_tok_buf_.handle() : tok_buf_.handle();
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
        // The in_proj_qkv gemv is the first consumer of the delta-net block's normalised input,
        // so dumping its INPUT here pins whether a wrong lin_mix came from the norm or the
        // projection. Identified by its output slot -- no other gemv writes LinMix.
        if (op.out == opplan::Slot::LinMix) {
          dump_named_slot("lin_xnorm", layer, position, op.in, cfg_.hidden_size);
        }
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

          // Split the reduction when the grid is too small to fill the GPU. Shrinking the row
          // tile to make more threadgroups was tried and lost, because it also halves how many
          // output rows a staged weight tile serves and so doubles the weight traffic; splitting
          // K does not, since every threadgroup reads a different slice of the weights.
          const std::size_t out_elems =
              static_cast<std::size_t>(gemm_tokens) * static_cast<std::size_t>(op.cols);
          if (groups < kGemmSplitKMaxGroups && gemm_tokens <= kGemmSplitKMaxTokens) {
            const std::size_t need = out_elems * kGemmSplitK * sizeof(float);
            if (gemm_partial_buf_.size() < need) gemm_partial_buf_ = ctx_.alloc(need);
            const void* sbufs[] = {bufs[0], bufs[1], gemm_partial_buf_.handle()};
            const std::size_t soffs[] = {offs[0], offs[1], 0};
            ctx_.dispatch("cpi_gemm_f16_splitk", G::Groups, groups * kGemmSplitK, kGemmTG, sbufs,
                          soffs, 3, &gp, sizeof(gp));
            const void* rbufs[] = {gemm_partial_buf_.handle(), bufs[2],
                                   op.bias != nullptr ? bufs[3] : bufs[2]};
            const std::size_t roffs2[] = {0, offs[2], op.bias != nullptr ? offs[3] : offs[2]};
            ctx_.dispatch("cpi_gemm_splitk_reduce", G::Threads, out_elems, 256, rbufs, roffs2, 3,
                          &gp, sizeof(gp));
          } else {
            ctx_.dispatch("cpi_gemm_f16", G::Groups, groups, kGemmTG, bufs, offs, 4, &gp,
                          sizeof(gp));
          }
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
        // Speculative verify: every position's logits, into batch_logits_buf_. This is the same
        // per-row loop the batched-decode path uses, but over consecutive tokens of ONE sequence
        // rather than N sequences -- verify_tokens sizes the buffer and reads K argmaxes back.
        // The extra T-1 vocab GEMVs are the cost of checking K drafts in one pass, which is still
        // far cheaper than K separate target decodes.
        if (prefill_all_logits_) {
          for (int t = 0; t < T; ++t) {
            const std::size_t in_off =
                static_cast<std::size_t>(t) * static_cast<std::size_t>(op.in_dim) * 2;
            const std::size_t out_off =
                static_cast<std::size_t>(t) * static_cast<std::size_t>(cfg_.vocab_size) *
                sizeof(float);
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
                     0,
                     0,
                     static_cast<std::uint32_t>(op.rotary_dim)};
        // Batched decode: the rows are N different sequences, so each takes its own
        // position rather than base+t.
        if (batch_ != nullptr) p.per_row_positions = 1;
        // M-RoPE: bind the [3][T] axis positions for THIS chunk and tell the kernel how the
        // rotary lanes divide between them. Off unless a multimodal prefill armed it, and for
        // pure text the three axes are equal, so the kernel's own 1-D result is reproduced
        // bit for bit (metal_smoke::mrope_reduces_to_1d).
        const bool use_mrope = mrope_active_ && mrope_pos_buf_.valid();
        if (use_mrope) {
          p.mrope_t = static_cast<std::uint32_t>(mrope_section_[0]);
          p.mrope_h = static_cast<std::uint32_t>(mrope_section_[1]);
          p.mrope_w = static_cast<std::uint32_t>(mrope_section_[2]);
        }
        const void* bufs[] = {slot(op.in),
                              use_mrope ? mrope_pos_buf_.handle()
                                        : (batch_ != nullptr ? batch_pos_buf_.handle()
                                                             : pos_buf_.handle())};
        // Threads cover the ROTATED lanes only. Launching head_dim/2 per head was not just
        // wasteful, it was wrong: those extra threads rotated lanes that must pass through.
        const int rot = op.rotary_dim > 0 ? op.rotary_dim : op.head_dim;
        const std::size_t total = static_cast<std::size_t>(op.heads) *
                                  static_cast<std::size_t>(rot / 2) *
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
              slot(op.in), slot(op.in2), kbuf(layer).handle(),
              vbuf(layer).handle(), batch_bt_buf_.handle(),
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
            slot(op.in), slot(op.in2), kbuf(layer).handle(),
            vbuf(layer).handle(), pos_buf_.handle()};
        const std::size_t total = static_cast<std::size_t>(op.kv_heads) *
                                  static_cast<std::size_t>(op.head_dim) *
                                  static_cast<std::size_t>(T);
        ctx_.dispatch("cpi_kv_store", G::Threads, total, kTG, bufs, nullptr, 5, &p, sizeof(p));
        break;
      }
      case OpKind::Attention: {
        // A paged PREFILL is one sequence's consecutive tokens sharing one block table, so it
        // can use the query-block matrix kernel instead of the decode kernel -- which matters:
        // the decode kernel makes every query re-walk the whole history, measured at 2.81x the
        // contiguous path's time on a 480-token prompt.
        //
        // The guards are what make the paged gather legal, not taste. block_size % KEY_BLOCK
        // keeps a 32-key run inside one physical block, and no-window keeps every kb
        // KEY_BLOCK-aligned (start == 0); together they mean a key block is contiguous and the
        // kernel needs one address per block rather than per key. Anything else -- a window,
        // an odd block size, Gemma's head_dim 256, a chunk below one query block -- falls
        // through to the decode kernel, which handles any geometry at O(T^2).
        if (batch_ != nullptr && batch_->logits_last_only && T >= kQBlock && op.head_dim <= 128 &&
            (op.full_attention || op.sliding_window == 0) &&
            batch_->block_size % kMMKeyBlock == 0) {
          AttnParams p{static_cast<std::uint32_t>(op.heads),
                       static_cast<std::uint32_t>(op.kv_heads),
                       static_cast<std::uint32_t>(op.head_dim),
                       static_cast<std::uint32_t>(position),
                       static_cast<std::uint32_t>(max_context_),
                       0,
                       op.scale,
                       1,
                       static_cast<std::uint32_t>(T),
                       1,
                       static_cast<std::uint32_t>(batch_->block_size)};
          // batch_bt_buf_ holds the sequence's table replicated per row; the mm kernel wants
          // just the table, and row 0 sits at offset 0, so it reads the right thing.
          const void* mmbufs[] = {slot(op.in), kbuf(layer).handle(),
                                  vbuf(layer).handle(), slot(op.out),
                                  pos_buf_.handle(), batch_bt_buf_.handle()};
          // Blocked by the kernel's OWN query block. This used to use kQBlock, which is half
          // kQMMBlock, so every query was covered twice and half the threadgroups launched only to
          // find t0 >= tokens and return. Harmless, and invisible in the output, which is why it
          // survived: the kernel bounds-checks correctly, it was just dispatched twice over.
          const std::size_t blocks = static_cast<std::size_t>((T + kQMMBlock - 1) / kQMMBlock);
          specialize_mm_attn(ctx_, op);
          ctx_.dispatch("cpi_attention_prefill_mm", G::Groups,
                        static_cast<std::size_t>(op.heads) * blocks, kTG, mmbufs, nullptr, 6, &p,
                        sizeof(p));
          break;
        }
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
          const void* bufs[] = {slot(op.in), kbuf(layer).handle(),
                                vbuf(layer).handle(), slot(op.out),
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
        const void* bufs[] = {slot(op.in), kbuf(layer).handle(),
                              vbuf(layer).handle(), slot(op.out),
                              pos_buf_.handle()};
        // A prefill's threadgroups each walk the whole KV cache, so per-token attention is
        // O(T^2) in DEVICE traffic and was 23% of the 8B's prefill. The prefill kernel gives
        // one threadgroup a BLOCK of queries, so a key block it pulls in serves kQBlock of
        // them. Decode is a single query and has nothing to block over -- it keeps the
        // per-token kernel.
        if (T >= kQBlock) {
          // head_dim <= 128 runs the matrix-unit kernel; Gemma's 256 keeps the scalar one. The
          // two block over queries differently, so each derives its own grid -- sharing one
          // `blocks` here would hand a kernel the other's tiling and quietly drop or double-run
          // queries.
          if (op.head_dim <= 128) {
            // p.paged stays 0, so block_tables is never read -- but the binding must exist,
            // because dispatch() puts the params block at index n_buffers.
            const void* mmbufs[] = {slot(op.in), kbuf(layer).handle(),
                                    vbuf(layer).handle(),
                                    slot(op.out), pos_buf_.handle(),
                                    kbuf(layer).handle()};
            // CPI_METAL_ATTN_QP=1 selects the query-partitioned research kernel (see the header
            // above it): 4 simdgroups x 8 queries, per-simdgroup scratch, simdgroup_barrier only.
            static const bool qp = [] {
              const char* e = std::getenv("CPI_METAL_ATTN_QP");
              return e != nullptr && e[0] == '1';
            }();
            // CPI_METAL_ATTN_FA=1 selects the register-resident research kernel (see the header
            // above it). Opt-in, because it measured slower -- the proven kernel stays the default.
            // The layout check still gates it: the kernel is only correct on a device whose
            // fragment layout matches what it assumes, and that is verified, never assumed.
            static const bool fa = [&] {
              const char* e = std::getenv("CPI_METAL_ATTN_FA");
              if (e == nullptr || e[0] != '1') return false;
              return simdgroup_layout_matches(ctx_);
            }();
            if (fa && op.head_dim <= 128 && op.head_dim % 8 == 0) {
              const std::size_t blocks = static_cast<std::size_t>((T + kFABlock - 1) / kFABlock);
              ctx_.dispatch("cpi_attention_prefill_fa", G::Groups,
                            static_cast<std::size_t>(op.heads) * blocks, kFAThreads, mmbufs,
                            nullptr, 6, &p, sizeof(p));
            } else if (qp) {
              const std::size_t blocks = static_cast<std::size_t>((T + kQPBlock - 1) / kQPBlock);
              ctx_.dispatch("cpi_attention_prefill_qp", G::Groups,
                            static_cast<std::size_t>(op.heads) * blocks, kQPThreads, mmbufs, nullptr,
                            6, &p, sizeof(p));
            } else {
              const std::size_t blocks = static_cast<std::size_t>((T + kQMMBlock - 1) / kQMMBlock);
              specialize_mm_attn(ctx_, op);
              ctx_.dispatch("cpi_attention_prefill_mm", G::Groups,
                            static_cast<std::size_t>(op.heads) * blocks, kTG, mmbufs, nullptr, 6, &p,
                            sizeof(p));
            }
          } else {
            const std::size_t blocks = static_cast<std::size_t>((T + kQBlock - 1) / kQBlock);
            ctx_.dispatch("cpi_attention_prefill", G::Groups,
                          static_cast<std::size_t>(op.heads) * blocks, kTG, bufs, nullptr, 5, &p,
                          sizeof(p));
          }
        } else if (T == 1 && attn_split_chunks(op, position) > 1) {
          // Split-KV: one threadgroup per (head, chunk), then a merge. See the kernels.
          const int window = op.full_attention ? 0 : op.sliding_window;
          const int seq = position + 1;
          const int start = (window > 0 && seq > window) ? seq - window : 0;
          const int chunks = attn_split_chunks(op, position);
          // Whole key blocks per chunk: the kernel's scoring loop is one thread per key, so a
          // chunk that is not a multiple of the block runs a short final pass with idle threads.
          // Rounding UP can leave the last chunks empty, which they handle (-INF max, zero sum,
          // and the merge skips them).
          int chunk_size = ((seq - start) + chunks - 1) / chunks;
          chunk_size = ((chunk_size + kDecKeyBlock - 1) / kDecKeyBlock) * kDecKeyBlock;

          AttnSplitParams sp{static_cast<std::uint32_t>(op.heads),
                             static_cast<std::uint32_t>(op.kv_heads),
                             static_cast<std::uint32_t>(op.head_dim),
                             static_cast<std::uint32_t>(position),
                             1,
                             static_cast<std::uint32_t>(window),
                             op.scale,
                             static_cast<std::uint32_t>(chunk_size),
                             static_cast<std::uint32_t>(chunks)};
          const void* sbufs[] = {slot(op.in),
                                 kbuf(layer).handle(),
                                 vbuf(layer).handle(),
                                 attn_part_m_.handle(),
                                 attn_part_l_.handle(),
                                 attn_part_o_.handle(),
                                 pos_buf_.handle()};
          ctx_.dispatch("cpi_attention_decode_split", G::Groups,
                        static_cast<std::size_t>(op.heads) * static_cast<std::size_t>(chunks),
                        kDecTG, sbufs, nullptr, 7, &sp, sizeof(sp));
          const void* mbufs[] = {attn_part_m_.handle(), attn_part_l_.handle(),
                                 attn_part_o_.handle(), slot(op.out)};
          ctx_.dispatch("cpi_attention_decode_merge", G::Groups,
                        static_cast<std::size_t>(op.heads), kDecTG, mbufs, nullptr, 4, &sp,
                        sizeof(sp));
        } else {
          ctx_.dispatch("cpi_attention_decode", G::Groups,
                        static_cast<std::size_t>(op.heads) * static_cast<std::size_t>(T), kTG,
                        bufs, nullptr, 5, &p, sizeof(p));
        }
        break;
      }
      // ---- Mixture of Experts ------------------------------------------------
      //
      // ONE TOKEN AT A TIME, deliberately: routing is a per-token decision, so each token
      // selects its own experts and there is no batched form to share. Every other op in this
      // plan is row-independent and runs T rows at once; these cannot, because moe_idx_buf_
      // holds ONE token's selection. A prefill chunk therefore walks these T times.
      //
      // Not doing that is a bug that hides well: with T rows in the slot and a kernel reading
      // row 0, a prefill gives every token the FIRST token's experts and its FFN output. It
      // does not crash, it produces confident wrong tokens -- which is exactly what the
      // tiny-mixtral golden caught (divergence at token 0, logit gap 0.19, no tie).
      case OpKind::MoeRouterTopk:
      case OpKind::MoeGateUpGeglu:
      case OpKind::MoeDownAccum: {
        if (op.qbits != 0) {
          throw std::runtime_error("PlanMetalEngine: quantized MoE experts are not implemented");
        }
        const std::size_t h2 = sizeof(std::uint16_t);  // fp16 element
        for (int t = 0; t < T; ++t) {
          if (op.kind == OpKind::MoeRouterTopk) {
            MoeRouterParams p{static_cast<std::uint32_t>(op.cols),
                              static_cast<std::uint32_t>(op.heads),
                              op.weight != nullptr ? 1u : 0u};
            // op.weight is the optional per-expert gain (Gemma 4 has one, Mixtral does not).
            // Bind the logits as a stand-in when absent: params must stay the LAST binding,
            // so the slot cannot simply be empty.
            const void* bufs[] = {slot(op.in), op.weight != nullptr ? op.weight : slot(op.in),
                                  moe_idx_buf_.handle(), moe_w_buf_.handle()};
            const std::size_t sel_i = static_cast<std::size_t>(t) *
                                      static_cast<std::size_t>(op.heads) * sizeof(std::int32_t);
            const std::size_t sel_f = static_cast<std::size_t>(t) *
                                      static_cast<std::size_t>(op.heads) * sizeof(float);
            const std::size_t offs[] = {
                static_cast<std::size_t>(t) * static_cast<std::size_t>(op.cols) * h2, 0, sel_i,
                sel_f};
            ctx_.dispatch("cpi_moe_router_topk", G::Threads, 1, 32, bufs, offs, 4, &p, sizeof(p));
          } else if (op.kind == OpKind::MoeGateUpGeglu) {
            MoeExpertParams p{static_cast<std::uint32_t>(op.cols),
                              static_cast<std::uint32_t>(op.in_dim),
                              static_cast<std::uint32_t>(op.heads), op.mlp_gelu ? 1u : 0u};
            const void* bufs[] = {op.weight, slot(op.in), moe_idx_buf_.handle(), slot(op.out)};
            const std::size_t offs[] = {
                0, static_cast<std::size_t>(t) * static_cast<std::size_t>(op.in_dim) * h2,
                static_cast<std::size_t>(t) * static_cast<std::size_t>(op.heads) *
                    sizeof(std::int32_t),
                static_cast<std::size_t>(t) * static_cast<std::size_t>(op.heads) *
                    static_cast<std::size_t>(op.cols) * h2};
            ctx_.dispatch("cpi_moe_gate_up", G::Groups,
                          static_cast<std::size_t>(op.cols) * static_cast<std::size_t>(op.heads),
                          kTG, bufs, offs, 4, &p, sizeof(p));
          } else {
            MoeExpertParams p{static_cast<std::uint32_t>(op.in_dim),
                              static_cast<std::uint32_t>(op.cols),
                              static_cast<std::uint32_t>(op.heads), 0u};
            const void* bufs[] = {op.weight, slot(op.in), moe_idx_buf_.handle(),
                                  moe_w_buf_.handle(), slot(op.out)};
            const std::size_t offs[] = {
                0,
                static_cast<std::size_t>(t) * static_cast<std::size_t>(op.heads) *
                    static_cast<std::size_t>(op.in_dim) * h2,
                static_cast<std::size_t>(t) * static_cast<std::size_t>(op.heads) *
                    sizeof(std::int32_t),
                static_cast<std::size_t>(t) * static_cast<std::size_t>(op.heads) * sizeof(float),
                static_cast<std::size_t>(t) * static_cast<std::size_t>(op.cols) * h2};
            ctx_.dispatch("cpi_moe_down_accum", G::Groups, static_cast<std::size_t>(op.cols), kTG,
                          bufs, offs, 5, &p, sizeof(p));
          }
        }
        break;
      }
      case OpKind::SiluMul:
      case OpKind::GeluMul: {
        const std::size_t n = static_cast<std::size_t>(op.cols) * static_cast<std::size_t>(T);
        ElemParams p{static_cast<std::uint32_t>(n), 0.0f};
        const void* bufs[] = {slot(op.in), slot(op.in2), slot(op.out)};
        // aux_offset selects a WINDOW of in2 rather than its start -- Gemma 4's per-layer-input
        // injection multiplies by per_layer_inputs[L], layer L's slice of one packed
        // [num_layers][ple] table. Dropping it (which this dispatch did) makes every layer
        // multiply by LAYER 0's window: correct at layer 0 by accident, wrong everywhere after.
        // CUDA does the same thing as pointer arithmetic (S(op.in2) + op.aux_offset); here it is a
        // buffer offset in BYTES, which is what the LmHead case below already does for its input.
        // Safe as a flat displacement only because an aux_offset plan runs token-by-token -- see
        // has_recurrent_ops_ above, which this op now sets.
        const std::size_t aux_bytes = static_cast<std::size_t>(op.aux_offset) * sizeof(std::uint16_t);
        const std::size_t offs[] = {0, aux_bytes, 0};
        ctx_.dispatch(op.kind == OpKind::SiluMul ? "cpi_silu_mul" : "cpi_gelu_mul", G::Threads, n,
                      kTG, bufs, op.aux_offset != 0 ? offs : nullptr, 3, &p, sizeof(p));
        break;
      }
      // ---- linear attention (delta-net) and gated attention -----------------
      //
      // These run ONE TOKEN AT A TIME. The delta-net kernels carry state across steps and have no
      // token dimension -- CUDA and the CPU reference both drive this family through
      // forward_one(token, position) for the same reason. Metal's batched prefill chunk does not
      // apply here, so every case below refuses T > 1 rather than quietly walking off the end of a
      // slot sized for one token.
      case OpKind::SplitHeadHalves: {
        require_single_token(op.kind, T);
        LinSplitParams p{static_cast<std::uint32_t>(op.heads),
                         static_cast<std::uint32_t>(op.head_dim)};
        const std::size_t n = static_cast<std::size_t>(op.heads) * op.head_dim;
        const void* bufs[] = {slot(op.in), slot(op.out), slot(op.out2)};
        ctx_.dispatch("cpi_split_head_halves", G::Threads, n, kTG, bufs, nullptr, 3, &p, sizeof(p));
        break;
      }
      case OpKind::SigmoidGate: {
        require_single_token(op.kind, T);
        LinGateParams p{static_cast<std::uint32_t>(op.cols)};
        const void* bufs[] = {slot(op.out), slot(op.in2)};
        ctx_.dispatch("cpi_sigmoid_gate_inplace", G::Threads, static_cast<std::size_t>(op.cols),
                      kTG, bufs, nullptr, 2, &p, sizeof(p));
        break;
      }
      case OpKind::MulVec: {
        require_single_token(op.kind, T);
        LinMulVecParams p{static_cast<std::uint32_t>(op.cols), op.scale};
        const void* bufs[] = {slot(op.in), op.weight, slot(op.out)};
        ctx_.dispatch("cpi_mul_vec", G::Threads, static_cast<std::size_t>(op.cols), kTG, bufs,
                      nullptr, 3, &p, sizeof(p));
        break;
      }
      case OpKind::RepeatLinearHeads: {
        require_single_token(op.kind, T);
        const int repeat = op.num_k_heads > 0 ? op.num_v_heads / op.num_k_heads : 1;
        LinRepeatParams p{static_cast<std::uint32_t>(op.num_k_heads),
                          static_cast<std::uint32_t>(repeat),
                          static_cast<std::uint32_t>(op.key_head_dim),
                          static_cast<std::uint32_t>(op.value_head_dim)};
        const std::size_t width =
            static_cast<std::size_t>(std::max(op.key_head_dim, op.value_head_dim));
        const void* bufs[] = {slot(op.in), slot(opplan::Slot::LinQ), slot(opplan::Slot::LinK),
                              slot(opplan::Slot::LinV)};
        ctx_.dispatch("cpi_repeat_linear_heads", G::Threads,
                      width * static_cast<std::size_t>(op.num_k_heads), 64, bufs, nullptr, 4, &p,
                      sizeof(p));
        break;
      }

      case OpKind::LinearConv1d: {
        require_single_token(op.kind, T);
        if (!lin_conv_state_.valid()) {
          throw std::runtime_error("Metal: LinearConv1d without delta-net state (bad geometry?)");
        }
        LinConvParams p{static_cast<std::uint32_t>(op.cols),
                        static_cast<std::uint32_t>(op.conv_kernel)};
        const std::size_t off =
            static_cast<std::size_t>(layer) * lin_conv_stride_ * sizeof(float);
        const void* bufs[] = {op.weight, lin_conv_state_.handle(), slot(op.in)};
        const std::size_t offs[] = {0, off, 0};
        ctx_.dispatch("cpi_linear_conv1d_silu", G::Threads, static_cast<std::size_t>(op.cols), kTG,
                      bufs, offs, 3, &p, sizeof(p));
        dump_named_slot("lin_mix", layer, position, opplan::Slot::LinMix, op.cols);
        break;
      }
      case OpKind::LinearAttentionStep: {
        require_single_token(op.kind, T);
        if (!lin_recurrent_state_.valid()) {
          throw std::runtime_error("Metal: LinearAttentionStep without delta-net state");
        }
        LinAttnParams p{static_cast<std::uint32_t>(op.key_head_dim),
                        static_cast<std::uint32_t>(op.value_head_dim), op.eps};
        const std::size_t off =
            static_cast<std::size_t>(layer) * lin_recurrent_stride_ * sizeof(float);
        // auxf_a/auxf_b are typed const float* by the shared plan, but on this backend every
        // weight handle is an opaque device buffer -- the same pun op.weight already relies on.
        // They are fp16 here (the container casts every weight), see 70_linear_attn.metal.
        const void* bufs[] = {slot(opplan::Slot::LinQ),
                              slot(opplan::Slot::LinK),
                              slot(opplan::Slot::LinV),
                              slot(opplan::Slot::LinZ),
                              slot(opplan::Slot::LinA),
                              slot(opplan::Slot::LinB),
                              static_cast<const void*>(op.auxf_a),
                              static_cast<const void*>(op.auxf_b),
                              op.aux_ptr,
                              lin_recurrent_state_.handle(),
                              slot(opplan::Slot::LinAtt)};
        const std::size_t offs[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, off, 0};
        ctx_.dispatch("cpi_linear_attention_step", G::Groups,
                      static_cast<std::size_t>(op.num_v_heads), 256, bufs, offs, 11, &p,
                      sizeof(p));
        dump_named_slot("lin_att", layer, position, opplan::Slot::LinAtt,
                        op.num_v_heads * op.value_head_dim);
        break;
      }

      // Kernels exist (80_vision.metal) but nothing here supplies pixels, patch positions or the
      // 2-D RoPE tables, and those four ops implement GEMMA 4's tower -- Qwen3.5's is a different
      // architecture (3-D conv patch embed, learned absolute positions, LayerNorm, spatial merge).
      case OpKind::PatchEmbed:
      case OpKind::Rope2D:
      case OpKind::AvgPoolPatches:
      case OpKind::Standardize:
        throw std::runtime_error(std::string("Metal: ") + op_kind_name(static_cast<int>(op.kind)) +
                                 " belongs to a vision tower that is not wired on this backend");

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
// ⚠ THIS MEASURES A COMMAND-BUFFER ROUND TRIP, NOT THE OP. Read dump_profile()'s banner before
// believing a number from here.
void PlanMetalEngine::profile_tick(const char* name) {
  ctx_.commit_and_wait();
  const auto now = std::chrono::steady_clock::now();
  profile_ms_[name] += std::chrono::duration<double, std::milli>(now - profile_last_).count();
  profile_last_ = now;
}

void PlanMetalEngine::dump_gpu_profile() {
  if (!ctx_.gpu_profile_enabled()) return;
  const auto& rows = ctx_.gpu_profile_results();
  if (rows.empty()) {
    std::fprintf(stderr, "[metal gpu-profile] no samples\n");
    return;
  }
  std::uint64_t total = 0;
  for (const auto& r : rows) total += r.second.first;
  // Say what these numbers are and are not, in the output rather than only in a header: the
  // serialisation is why the sum overshoots a real pass, and the missing limiters are a hardware
  // fact (this device exposes one counter set, timestamp), not an omission.
  std::fprintf(stderr,
               "[metal gpu-profile] true GPU ns per kernel, from the device timestamp counters.\n"
               "  Each dispatch gets its own encoder to bracket it, which SERIALISES work a real\n"
               "  pass overlaps -- rows are honest individually, the sum exceeds a real pass.\n"
               "  Limiters/occupancy are NOT reachable from the Metal API here; those need Xcode.\n");
  std::fprintf(stderr, "  %-34s %12s %10s %9s\n", "kernel", "gpu_ms", "calls", "share");
  for (const auto& r : rows) {
    const double ms = static_cast<double>(r.second.first) / 1.0e6;
    std::fprintf(stderr, "  %-34s %12.3f %10llu %8.1f%%\n", r.first.c_str(), ms,
                 static_cast<unsigned long long>(r.second.second),
                 100.0 * static_cast<double>(r.second.first) / static_cast<double>(total));
  }
  std::fprintf(stderr, "  %-34s %12.3f\n", "(sum, serialised)",
               static_cast<double>(total) / 1.0e6);
}

void PlanMetalEngine::dump_profile() const {
  double total = 0.0;
  for (const auto& kv : profile_ms_) total += kv.second;
  if (total <= 0.0) return;
  // The banner is not decoration. These numbers are SHARES OF A DISTORTED TOTAL, and taking
  // them at face value has already cost this project real time.
  std::fprintf(
      stderr,
      "[metal profile] ⚠ READ THIS BEFORE BELIEVING THE TABLE.\n"
      "  These are HOST times around a commit-and-wait PER OP, not GPU times. Every op pays a\n"
      "  full command-buffer round trip it does not pay in a real pass, which is a fixed cost,\n"
      "  so the SMALL ops are inflated most -- the exact ops one profiles to find. Below, the\n"
      "  total comes to ~%.0f ms for a pass that really runs in ~190: it is ~3x, concentrated\n"
      "  in the cheap rows. It once put the non-GEMM ops at 34%%; deleting them showed ~3.5%%,\n"
      "  and a fusion plan was built on the difference.\n"
      "  It cannot be fixed here: honest per-op GPU timing needs counter sampling at dispatch\n"
      "  boundaries, and Apple Silicon (M4/AGXG16G) does not support it -- asking crashes the\n"
      "  driver. Use this ONLY to compare an op against ITSELF across a change.\n"
      "  To ask what an op COSTS, delete it: CPI_METAL_ABLATE=<OpKind>[,<OpKind>] and time the\n"
      "  pass. No attribution error survives that.\n",
      total);
  std::fprintf(stderr, "[metal profile] %.0f ms (INFLATED, see above), by op:\n", total);
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
void PlanMetalEngine::encode_forward(int token, int position, int cache_position) {
  if (cache_position < 0) cache_position = position;
  // CPI_METAL_LAYERS=N runs only the first N layers. A NaN has to start SOMEWHERE, and
  // bisecting on the layer count finds where far faster than reasoning about which op
  // could overflow.
  static const int layer_limit = [] {
    const char* e = std::getenv("CPI_METAL_LAYERS");
    return e != nullptr ? std::atoi(e) : -1;
  }();

  // Unified memory: writing the token and the position IS the H2D transfer.
  //
  // pos_buf_ holds the CACHE position -- where K/V land and how far attention reaches (KvParams
  // and AttnParams both read it). The ROTARY position is `position`, which differs from the
  // cache position only under M-RoPE and is fed to the rope kernel separately, below.
  tokens_in_seq_buf_ = false;
  *static_cast<std::int32_t*>(tok_buf_.contents()) = static_cast<std::int32_t>(token);
  *static_cast<std::int32_t*>(pos_buf_.contents()) = static_cast<std::int32_t>(cache_position);

  // When the two diverge, the rope kernel must NOT read pos_buf_ (now the cache position). Arm a
  // one-token M-RoPE buffer with the rotary position on all three axes: t == h == w is exactly
  // 1-D rope, and the lane mapping collapses to a single value -- gated bit-for-bit by
  // metal_smoke::mrope_reduces_to_1d. This is why generated tokens, which are text, are correct
  // with a scalar rotary position even though the prompt used a 3-D one.
  if (cache_position != position) {
    const std::size_t need = static_cast<std::size_t>(3) * sizeof(std::int32_t);
    if (mrope_pos_buf_.size() < need) mrope_pos_buf_ = ctx_.alloc(need);
    auto* dst = static_cast<std::int32_t*>(mrope_pos_buf_.contents());
    dst[0] = dst[1] = dst[2] = static_cast<std::int32_t>(position);
    mrope_active_ = true;
  } else {
    // A pure-text decode has no reason to route through the M-RoPE path; leave it off so the
    // scalar pos_buf_ drives rope as it always has.
    mrope_active_ = false;
  }

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
  // The embedding, before any layer. Dumped on the decode path too, not just prefill -- a diff
  // tool cannot tell "these differ" from "one side never wrote this", and a missing file here
  // reads as a token mismatch that did not happen.
  dump_layer_state(0, position);
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
    // layer_00 is the embedding, before any layer ran -- so a layer's output is filed under N+1,
    // matching the CPU engine.
    dump_layer_state(lp.layer_index + 1, position);
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
int PlanMetalEngine::attn_split_chunks(const opplan::Op& op, int position) const {
  // CPI_METAL_ATTN_SPLIT_MIN overrides the key count at which splitting starts, and it exists
  // to make the split path TESTABLE rather than to tune it. The split is a pure optimization --
  // it must produce the same tokens as the single-threadgroup kernel -- but every golden is
  // short enough that no decode ever reaches the default threshold, so a green gate would say
  // nothing about these kernels. Setting it to 1 forces the split on at any depth, which points
  // the existing goldens straight at pass 1 and the merge; tools/metal_verify.sh does exactly
  // that. Setting it very high turns the split off, for bisecting a suspected merge bug.
  static const int min_keys = [] {
    const char* e = std::getenv("CPI_METAL_ATTN_SPLIT_MIN");
    if (e == nullptr) return kAttnSplitMinKeys;
    const int v = std::atoi(e);
    return v > 0 ? v : kAttnSplitMinKeys;
  }();
  if (position < 0 || op.heads <= 0) return 1;
  const int window = op.full_attention ? 0 : op.sliding_window;
  const int seq = position + 1;
  const int start = (window > 0 && seq > window) ? seq - window : 0;
  const int nkeys = seq - start;
  if (nkeys < min_keys) return 1;  // the merge pass would cost more than it saves
  int chunks = (kAttnSplitTargetGroups + op.heads - 1) / op.heads;
  // Never cut finer than a key block: a chunk with fewer keys than the threadgroup has threads
  // leaves threads idle, which is the bottleneck the one-key-per-thread scoring loop exists to
  // avoid.
  chunks = std::min(chunks, (nkeys + kDecKeyBlock - 1) / kDecKeyBlock);
  chunks = std::min(chunks, kAttnSplitMaxChunks);
  return std::max(1, chunks);
}

// CPI_Q35_DUMP=<dir>: same files, same layout, same names as the CPU engine's dump, so
// tools/layer_diff.py can point at one of each and name the first layer that disagrees. Writing
// fp32 of an fp16 state is deliberate -- the format has to match the reference, and the widening
// is lossless.
//
// Called from BOTH the prefill and decode paths. Dumping only decode would leave every position
// but the last missing, and a divergence at the last position cannot be told apart from one
// inherited from a prefill step that was never looked at.
//
// Costs a commit_and_wait per layer; only ever set this while debugging.
void PlanMetalEngine::dump_layer_state(int layer_index, int position) {
  static const char* dir = std::getenv("CPI_Q35_DUMP");
  if (dir == nullptr) return;
  ctx_.commit_and_wait();
  const auto* x = static_cast<const std::uint16_t*>(slots_[0].contents());
  char path[512];
  std::snprintf(path, sizeof(path), "%s/layer_%02d_pos_%02d.f32", dir, layer_index, position);
  std::FILE* f = std::fopen(path, "wb");
  if (f == nullptr) return;
  for (int i = 0; i < cfg_.hidden_size; ++i) {
    const float v = fp16_to_f32(x[i]);
    std::fwrite(&v, sizeof(float), 1, f);
  }
  std::fclose(f);
}

// Dumps a named intermediate slot, in the CPU engine's format and under the same names, so the
// two can be diffed buffer by buffer. Widening fp16 to fp32 keeps the format identical -- the
// difference in precision between the engines is real, but it is not what this is looking for.
void PlanMetalEngine::dump_named_slot(const char* name, int layer, int position,
                                      opplan::Slot sl, int n) {
  static const char* dir = std::getenv("CPI_Q35_DUMP");
  if (dir == nullptr) return;
  ctx_.commit_and_wait();
  const auto* x = static_cast<const std::uint16_t*>(slots_[static_cast<int>(sl)].contents());
  char path[512];
  std::snprintf(path, sizeof(path), "%s/%s_L%02d_pos_%02d.f32", dir, name, layer, position);
  std::FILE* f = std::fopen(path, "wb");
  if (f == nullptr) return;
  for (int i = 0; i < n; ++i) {
    const float v = fp16_to_f32(x[i]);
    std::fwrite(&v, sizeof(float), 1, f);
  }
  std::fclose(f);
}

opplan::WeightSource& PlanMetalEngine::weight_source() {
  if (wsrc_ == nullptr) {
    throw std::runtime_error("weight source requested before open()");
  }
  return *wsrc_;
}

// Prefills a prompt in which some positions are IMAGE tokens: `embeds` is indexed by absolute
// position, and an empty row means "use the token's own embedding".
//
// Chunking is left to prefill_chunks, so this works unchanged on a recurrent model (which
// prefills one token at a time) -- the splice is keyed on absolute position, not chunk offset.
void PlanMetalEngine::prefill_multimodal(const std::vector<int>& tokens,
                                         const std::vector<std::vector<float>>& embeds,
                                         const std::vector<std::int32_t>& mrope_positions) {
  embeds_ = &embeds;
  const int n = static_cast<int>(tokens.size());
  const bool want_mrope = !mrope_positions.empty();
  if (want_mrope && static_cast<int>(mrope_positions.size()) != 3 * n) {
    embeds_ = nullptr;
    throw std::runtime_error("mrope positions must be 3 x tokens, got " +
                             std::to_string(mrope_positions.size()) + " for " +
                             std::to_string(n) + " tokens");
  }
  mrope_active_ = want_mrope;
  int pos = 0;
  try {
    for (const int chunk : prefill_chunks(n)) {
      const std::vector<int> ids(tokens.begin() + pos, tokens.begin() + pos + chunk);
      if (want_mrope) {
        // The kernel indexes positions[axis * p.tokens + token] with p.tokens == this chunk's
        // width, so each chunk needs its own [3][chunk] slice rather than the whole array.
        const std::size_t need = static_cast<std::size_t>(3) * chunk * sizeof(std::int32_t);
        if (mrope_pos_buf_.size() < need) mrope_pos_buf_ = ctx_.alloc(need);
        auto* dst = static_cast<std::int32_t*>(mrope_pos_buf_.contents());
        for (int axis = 0; axis < 3; ++axis) {
          for (int t = 0; t < chunk; ++t) {
            dst[axis * chunk + t] = mrope_positions[static_cast<std::size_t>(axis) * n + pos + t];
          }
        }
      }
      encode_prefill(ids, pos);
      ctx_.commit_and_wait();
      pos += chunk;
    }
  } catch (...) {
    embeds_ = nullptr;  // a throw must not leave a dangling pointer for the next prefill
    mrope_active_ = false;
    throw;
  }
  embeds_ = nullptr;
  // Decode continues on the plain 1-D path: generated tokens are text, so their three axes are
  // equal and the caller passes the M-RoPE counter as a scalar position. Leaving this armed
  // would make forward_token read a buffer sized for the last prefill chunk.
  mrope_active_ = false;
}

std::vector<int> PlanMetalEngine::prefill_chunks(int n) const {
  // A recurrent model has no token dimension to batch over: its state must advance one token at a
  // time, in order, so "prefill" here is just decode without sampling. CUDA and the CPU reference
  // drive this family the same way, through forward_one(token, position).
  //
  // This costs the whole batched-prefill speedup on those models, which is a real loss and not a
  // temporary one -- it is what the architecture allows. Chunked delta-net prefill exists in the
  // literature and would be a separate kernel, not a tuning of this path.
  if (has_recurrent_ops_) {
    return std::vector<int>(static_cast<std::size_t>(n), 1);
  }
  // The quantized GEMM tiles 128 tokens where the fp16 one tiles 64.
  return chunk_sizes(n, max_prefill_, quant_bits_ != 0 ? kGemmQBN : kGemmBN);
}

void PlanMetalEngine::encode_prefill(const std::vector<int>& tokens, int start_position) {
  const int T = static_cast<int>(tokens.size());
  if (T <= 0) return;
  if (T > max_prefill_) {
    throw std::runtime_error("prefill chunk larger than the slots were sized for");
  }

  tokens_in_seq_buf_ = true;
  auto* ids = static_cast<std::int32_t*>(seq_tok_buf_.contents());
  for (int i = 0; i < T; ++i) {
    ids[i] = static_cast<std::int32_t>(tokens[static_cast<std::size_t>(i)]);
  }
  *static_cast<std::int32_t*>(pos_buf_.contents()) = static_cast<std::int32_t>(start_position);

  execute_ops(plan_.prologue, -1, start_position, T);

  // Image soft tokens go in HERE: after the embedding lookup, before any layer reads X. Used
  // AS-IS -- unlike text embeddings they get no sqrt(hidden) scale -- matching the CUDA path
  // and what the reference projects from.
  if (embeds_ != nullptr) {
    bool any = false;
    for (int t = 0; t < T; ++t) {
      const std::size_t idx = static_cast<std::size_t>(start_position + t);
      if (idx < embeds_->size() && !(*embeds_)[idx].empty()) any = true;
    }
    if (any) {
      ctx_.commit_and_wait();  // the lookup must have landed before it is overwritten
      auto* xw = static_cast<std::uint16_t*>(slots_[0].contents());
      const int H2 = cfg_.hidden_size;
      for (int t = 0; t < T; ++t) {
        const std::size_t idx = static_cast<std::size_t>(start_position + t);
        if (idx >= embeds_->size() || (*embeds_)[idx].empty()) continue;
        const std::vector<float>& row = (*embeds_)[idx];
        if (static_cast<int>(row.size()) != H2) {
          throw std::runtime_error("image embedding at position " + std::to_string(idx) +
                                   " is " + std::to_string(row.size()) + " wide, expected " +
                                   std::to_string(H2));
        }
        for (int j = 0; j < H2; ++j) {
          xw[static_cast<std::size_t>(t) * H2 + j] = f32_to_fp16(row[static_cast<std::size_t>(j)]);
        }
      }
    }
  }

  // layer_00 is the embedding, before any layer -- the same file the CPU engine writes. Lets a
  // prefill divergence be split into "the token never got in" and "a layer computed it wrong".
  if (T == 1) dump_layer_state(0, start_position);
  for (const opplan::LayerPlan& lp : plan_.layers) {
    execute_ops(lp.ops, lp.layer_index, start_position, T);
    // Only when the chunk IS one token: with T > 1 slot X holds T states and "the state at
    // start_position" is not what the first hidden_size values are.
    if (T == 1) dump_layer_state(lp.layer_index + 1, start_position);
  }
  execute_ops(plan_.epilogue, -1, start_position, T);
}

void PlanMetalEngine::set_paged_kv(int num_blocks, int block_size) {
  paged_blocks_ = num_blocks;
  paged_block_size_ = block_size;
}

// This engine's half of continuous batching: the only two operations BatchScheduler needs
// from a GPU. Everything else -- who gets admitted, who gets preempted, which prefix is
// reused -- is the shared scheduler's, so Metal cannot drift from CUDA on any of it.
class PlanMetalEngine::BatchAdapter final : public BatchBackend {
public:
  explicit BatchAdapter(PlanMetalEngine* e) : e_(e) {}

  void prefill_suffix(const std::vector<int>& prompt_tokens, int shared_tokens,
                      const std::vector<int>& block_table) override {
    // [0, shared) is already in the pool through adopted blocks. Chunk the rest to the slot
    // size; each chunk lands at its own absolute position, so a chunk boundary is not a
    // sequence boundary. prefill_paged commits and waits, which is the contract.
    const int n = static_cast<int>(prompt_tokens.size());
    int off = shared_tokens;
    for (const int chunk : e_->prefill_chunks(n - shared_tokens)) {
      const std::vector<int> ids(prompt_tokens.begin() + off, prompt_tokens.begin() + off + chunk);
      e_->prefill_paged(ids, off, block_table);
      off += chunk;
    }
  }

  void decode_batched_logits(const std::vector<int>& tokens, const std::vector<int>& positions,
                             const std::vector<int>& block_tables_flat, int max_blocks,
                             std::vector<std::vector<float>>& out_logits) override {
    e_->decode_step_batched_logits(tokens, positions, block_tables_flat, max_blocks, out_logits);
  }

private:
  PlanMetalEngine* e_;
};

BatchScheduler& PlanMetalEngine::batch_scheduler(const BatchSchedulerOptions& opts) {
  if (!scheduler_) {
    if (paged_blocks_ <= 0) {
      throw std::runtime_error(
          "batch_scheduler requires a paged KV pool: call set_paged_kv() before open()");
    }
    block_alloc_ = std::make_unique<BlockAllocator>(paged_blocks_);
    batch_adapter_ = std::make_unique<BatchAdapter>(this);
    BatchSchedulerOptions o = opts;
    o.paged_block_size = paged_block_size_;  // must match the pool this engine allocated
    scheduler_ = std::make_unique<BatchScheduler>(batch_adapter_.get(), block_alloc_.get(), o);
    // Single-sequence prefix reuse tracks its own state and would now be wrong: the batched
    // path rewrites KV through block tables it knows nothing about.
    prev_seq_.clear();
  }
  return *scheduler_;
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

  tokens_in_seq_buf_ = true;
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

  // The mm path reads the base position from pos_buf_[0] (per-row positions are only needed by
  // the decode kernel's ragged batch), so stage it for whichever path the dispatch picks.
  *static_cast<std::int32_t*>(pos_buf_.contents()) = static_cast<std::int32_t>(start_position);

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
  tokens_in_seq_buf_ = true;
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

const std::vector<float>& PlanMetalEngine::forward_token(int token, int position,
                                                         int cache_position) {
  encode_forward(token, position, cache_position);
  ctx_.commit_and_wait();

  if (!ctx_.last_error().empty()) {
    last_error_ = ctx_.last_error();
  }

  const float* src = static_cast<const float*>(logits_buf_.contents());
  std::memcpy(logits_.data(), src, logits_.size() * sizeof(float));
  return logits_;
}

// ---- Speculative decoding ----

void PlanMetalEngine::reset_kv_cache() {
  // Nothing to zero: the KV cache is addressed by absolute position and overwritten in place. The
  // only per-sequence state is the prefix-reuse record, which a fresh sequence must not inherit.
  prev_seq_.clear();
}

void PlanMetalEngine::prefill_prompt(const std::vector<int>& prompt_tokens, int start_pos) {
  // Process every token EXCEPT the last, so the last is consumed by the first decode/verify step
  // at position P-1 -- matching LlamaEngine and generate_stream. Chunked at max_prefill_.
  const int P = static_cast<int>(prompt_tokens.size());
  if (P <= 1) return;
  prev_seq_.clear();  // this rewrites the cache directly; drop any shared-prefix state
  int pos = start_pos;
  int i = 0;
  const int last_idx = P - 1;
  while (i < last_idx) {
    const int chunk = std::min(max_prefill_, last_idx - i);
    const std::vector<int> ids(prompt_tokens.begin() + i, prompt_tokens.begin() + i + chunk);
    encode_prefill(ids, pos);
    ctx_.commit_and_wait();
    pos += chunk;
    i += chunk;
  }
  if (!ctx_.last_error().empty()) last_error_ = ctx_.last_error();
}

int PlanMetalEngine::decode_next_token(int token, int position, float /*temperature*/,
                                       const std::vector<int>& /*history*/) {
  const std::vector<float>& lg = forward_token(token, position);
  int best = 0;
  for (std::size_t j = 1; j < lg.size(); ++j) {
    if (lg[j] > lg[static_cast<std::size_t>(best)]) best = static_cast<int>(j);
  }
  return best;
}

int PlanMetalEngine::decode_next_token2(int token, int position, int* second) {
  const std::vector<float>& lg = forward_token(token, position);
  int best = 0;
  for (std::size_t j = 1; j < lg.size(); ++j) {
    if (lg[j] > lg[static_cast<std::size_t>(best)]) best = static_cast<int>(j);
  }
  if (second != nullptr) {
    int s = (best == 0) ? 1 : 0;
    for (std::size_t j = 0; j < lg.size(); ++j) {
      if (static_cast<int>(j) == best) continue;
      if (lg[j] > lg[static_cast<std::size_t>(s)]) s = static_cast<int>(j);
    }
    *second = s;
  }
  return best;
}

void PlanMetalEngine::verify_tokens(const std::vector<int>& tokens, int start_pos,
                                    std::vector<int>& out_argmax) {
  const int K = static_cast<int>(tokens.size());
  out_argmax.assign(static_cast<std::size_t>(K), 0);
  if (K == 0) return;
  if (K > max_prefill_) {
    throw std::runtime_error("verify_tokens: K=" + std::to_string(K) + " exceeds max_prefill_=" +
                             std::to_string(max_prefill_));
  }
  if (has_recurrent_ops_) {
    throw std::runtime_error(
        "verify_tokens: speculative decoding needs a parallel verify pass, which a recurrent "
        "(delta-net) model has no token dimension for");
  }

  // batch_logits_buf_ is the batched-decode scratch; verify borrows it for K rows. Grow if a
  // previous batched call sized it smaller.
  const std::size_t need =
      static_cast<std::size_t>(K) * static_cast<std::size_t>(cfg_.vocab_size) * sizeof(float);
  if (batch_logits_buf_.size() < need) batch_logits_buf_ = ctx_.alloc(need);

  // One causal chunk at [start_pos, start_pos+K-1], with logits for every position.
  prefill_all_logits_ = true;
  encode_prefill(tokens, start_pos);
  ctx_.commit_and_wait();
  prefill_all_logits_ = false;
  if (!ctx_.last_error().empty()) last_error_ = ctx_.last_error();

  const float* src = static_cast<const float*>(batch_logits_buf_.contents());
  for (int t = 0; t < K; ++t) {
    const float* row = src + static_cast<std::size_t>(t) * static_cast<std::size_t>(cfg_.vocab_size);
    int best = 0;
    for (int j = 1; j < cfg_.vocab_size; ++j) {
      if (row[j] > row[best]) best = j;
    }
    out_argmax[static_cast<std::size_t>(t)] = best;
  }
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
  for (const int chunk : prefill_chunks(n_pre)) {
    const std::vector<int> ids(prompt.begin() + pos, prompt.begin() + pos + chunk);
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
  int off = shared;
  for (const int chunk : prefill_chunks(n_pre - shared)) {
    const std::vector<int> ids(prompt.begin() + off, prompt.begin() + off + chunk);
    encode_prefill(ids, off);  // start position == off, so reused-prefix positions line up
    ctx_.commit_and_wait();
    off += chunk;
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
  for (const int chunk : prefill_chunks(n_pre)) {
    const std::vector<int> ids(prompt.begin() + pos, prompt.begin() + pos + chunk);
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
  for (const int chunk : prefill_chunks(n_pre)) {
    const std::vector<int> ids(prompt.begin() + pos, prompt.begin() + pos + chunk);
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
    encode_forward(next, pos, pos);  // text decode: rotary and cache positions coincide

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
