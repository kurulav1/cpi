#pragma once

#include <cuda_fp16.h>

#include <cstdint>
#include <vector>

// Op-plan IR for the generic per-layer decode executor.
//
// A model's forward is, per layer, a fixed ordered sequence of GPU kernel
// launches whose only variation is per-layer geometry + capability flags
// (head_dim, kv_heads, sliding vs full, KV-share, k==v, PLE, layer scalar…).
// Today that variation is resolved with `if`s inside the hot loop, once per
// token. The op-plan resolves it ONCE at load: the builder walks the model
// descriptor and emits a `std::vector<Op>` per layer with every conditional
// already decided and every weight pointer already bound. The hot loop then
// just iterates ops — branch-free over identity, and a fixed sequence that a
// CUDA graph can capture.
//
// This is the seam that lets Gemma 4 (per-layer varying geometry, KV sharing,
// PLE) run on a shared executor instead of a fork engine, while a uniform-
// geometry model can still take its specialized fast path (tiering by
// capability, chosen on the built plan — never by model name).
namespace engine {
namespace opplan {

// Named intermediate buffers the executor owns (the residual stream + scratch).
// Ops read/write these slots; the executor maps a slot to its device pointer.
// Kept small and explicit — this is the whole working set of a decode step.
enum class Slot : std::uint8_t {
  X,         // residual stream (persists across ops/layers)
  XNorm,     // normalized input to a projection
  Q, K, V,   // attention projections
  Att,       // attention output (pre o_proj)
  Tmp,       // block output staged before the residual add
  Gate, Up,  // MLP gate/up projections
  Inter,     // MLP intermediate (post GeGLU / SwiGLU)
  PleGate,   // per-layer-input gate scratch
  PleRaw,    // per-layer-input raw embedding (prologue scratch)
  PleAll,    // all layers' per-layer inputs [num_layers * ple] (prologue output)
  // Gated-attention / linear-attention (delta-net) working set. Models that don't
  // use these simply never emit ops touching them.
  QPair,     // fused q|gate projection, before it is split into Q and QGate
  QGate,     // sigmoid gate applied to the attention output
  LinMix,    // fused linear-attention qkv mixture (post conv1d+silu)
  LinZ, LinA, LinB,      // delta-net gate / A / B projections
  LinQ, LinK, LinV,      // delta-net q/k/v after head expansion
  LinAtt,    // delta-net output (pre out-projection)
  Count
};

// Which RoPE / attention table + geometry an attention op uses. Resolved per
// layer at build time (sliding layers use a different theta table than full).
enum class RopeTable : std::uint8_t { Sliding, Full };

// Where a scalar multiplier comes from (resolved to a concrete value at build,
// except SqrtHeadDim which is a geometry constant folded in at build too).
enum class ScaleKind : std::uint8_t { Constant };

enum class OpKind : std::uint8_t {
  RmsNorm,      // rows×cols groups; weight==nullptr ⇒ weightless (ones)
  Gemv,         // rowmajor half GEMV: out[out_dim] = W[out_dim×in_dim] · in
  Rope,         // in-place RoPE over `heads` heads of `head_dim`, at position
  ScaleCopy,    // out = in * scale
  CopySlot,     // out = in (device-to-device; e.g. V shares raw k_proj output)
  KvStore,      // append K,V slots to this layer's cache at `position`
  Attention,    // single-query attention over the cache (sliding window aware)
  GeluMul,      // out = gelu(a) * b   (GeGLU / PLE gate)
  AddInplace,   // out += in           (residual; out defaults to X)
  EmbeddingLookup,  // out = embed_table[token]  (token from the executor's device buffer)
  LmHead,       // logits[vocab] (float) = W[vocab×in_dim] · in   (writes the executor's logits)

  // ── extensions for gated / linear-attention ("delta-net") architectures ──
  // These are general OPS, not a model: any architecture that needs them declares
  // them in its plan. The kernels already exist in the shared kernel library.
  SiluMul,          // out = silu(a) * b   (SwiGLU; the SiLU sibling of GeluMul)
  SplitHeadHalves,  // fused q|gate -> out (Q) + out2 (QGate), interleaved per head
  SigmoidGate,      // in-place: out *= sigmoid(aux slot)   (gated attention output)
  LinearConv1d,     // short causal depthwise conv + SiLU over the fused qkv mixture;
                    // reads/writes this layer's rolling CONV STATE
  RepeatLinearHeads,// expand the fused mixture into per-head q/k/v (GQA-style repeat)
  LinearAttentionStep,  // gated delta-net recurrence; reads/writes this layer's
                        // RECURRENT STATE. The linear-attention analogue of Attention.
};

// One resolved op. Fields are a union-by-convention keyed on `kind`; only the
// members relevant to `kind` are set (kept as a flat struct for cache-friendly
// iteration and trivial copyability — no virtuals in the hot loop).
struct Op {
  OpKind kind;
  Slot in = Slot::X;
  Slot out = Slot::X;
  Slot in2 = Slot::X;          // GeluMul's second operand (when aux_ptr is null)
  const __half* weight = nullptr;  // bound at build; nullptr = weightless norm
  const __half* aux_ptr = nullptr; // GeluMul raw 2nd operand (PLE per-layer input); overrides in2
  // Quantized Gemv: when qweight is set the op runs a weight-only matvec instead
  // of the fp16 one. qbits picks the encoding: 8 = int8, 4 = int4 (packed
  // two-per-byte). qgroup picks the scale granularity: 0 = one scale per row,
  // >0 = one scale per that many contiguous input features (a power of two), with
  // `qscales` laid out [out_dim, quant_group_count(in_dim, qgroup)] row-major.
  // Resolved at LOAD like everything else, so the hot loop stays branch-free over
  // model identity — the op just carries a different weight encoding.
  const std::int8_t* qweight = nullptr;
  const float* qscales = nullptr;
  int qbits = 0;
  int qgroup = 0;
  int rows = 1;                // RmsNorm groups (heads for q/k norm, else 1)
  int cols = 0;                // RmsNorm width / Gemv out_dim / GeluMul length
  int in_dim = 0;              // Gemv in_dim
  int heads = 0;               // Rope/Attention query heads
  int kv_heads = 0;            // Attention kv heads
  int head_dim = 0;            // Rope/Attention head_dim
  RopeTable rope_table = RopeTable::Full;
  bool full_attention = false; // Attention: full (no window) vs sliding
  int sliding_window = 0;      // Attention: window size (0 = unbounded)
  float scale = 1.0f;          // ScaleCopy multiplier (SqrtHeadDim/layer scalar folded in)

  // ── extension fields (gated / linear-attention) ──
  Slot out2 = Slot::X;         // SplitHeadHalves' second output (the gate half)
  bool norm_offset = false;    // RmsNorm uses (1 + weight) instead of weight
  int rotary_dim = 0;          // Rope: >0 rotates only the first rotary_dim of each
                               // head (partial RoPE) and pairs Q (in) with K (in2)
  // Delta-net geometry + its extra weights. The recurrence needs a float RMS-norm
  // weight, a float A_log, and an fp16 dt_bias alongside the usual projections.
  int num_k_heads = 0;
  int num_v_heads = 0;
  int key_head_dim = 0;
  int value_head_dim = 0;
  int conv_kernel = 0;         // LinearConv1d: causal kernel width
  float eps = 1e-6f;           // RmsNorm / LinearAttentionStep epsilon
  const float* auxf_a = nullptr;  // delta-net RMS-norm weight (float)
  const float* auxf_b = nullptr;  // delta-net A_log (float)
};

// The forward as data: one op list per layer (conditionals already resolved —
// shared layers simply omit the KV-compute ops, non-PLE models omit PLE ops).
struct LayerPlan {
  std::vector<Op> ops;
  int layer_index = 0;
};

struct ModelPlan {
  std::vector<Op> prologue;        // token → embeddings (+ scale, PLE build)
  std::vector<LayerPlan> layers;   // the tower
  std::vector<Op> epilogue;        // final norm → LM head (logits); softcap is host-side
};

}  // namespace opplan
}  // namespace engine
