#pragma once

#include <cstdint>
#include <limits>
#include <vector>

// Op-plan IR for the generic per-layer decode executor.
//
// A model's forward is, per layer, a fixed ordered sequence of GPU kernel
// launches whose only variation is per-layer geometry + capability flags
// (head_dim, kv_heads, sliding vs full, KV-share, k==v, PLE, layer scalar…).
// Today that variation is resolved with `if`s inside the hot loop, once per
// token. The op-plan resolves it once at load: the builder walks the model
// descriptor and emits a `std::vector<Op>` per layer with every conditional
// already decided and every weight pointer already bound. The hot loop then
// just iterates ops: branch-free over identity, and a fixed sequence that a
// CUDA graph can capture.
//
// This is the seam that lets Gemma 4 (per-layer varying geometry, KV sharing,
// PLE) run on a shared executor instead of a fork engine, while a uniform-
// geometry model can still take its specialized fast path (tiering by
// capability, chosen on the built plan, never by model name).
namespace engine {
namespace opplan {

// Named intermediate buffers the executor owns (the residual stream + scratch).
// Ops read/write these slots; the executor maps a slot to its device pointer.
// Kept small and explicit: this is the whole working set of a decode step.
enum class Slot : std::uint8_t {
  X,      // residual stream (persists across ops/layers)
  XNorm,  // normalized input to a projection
  Q,
  K,
  V,    // attention projections
  Att,  // attention output (pre o_proj)
  Tmp,  // block output staged before the residual add
  Gate,
  Up,       // MLP gate/up projections
  Inter,    // MLP intermediate (post GeGLU / SwiGLU)
  PleGate,  // per-layer-input gate scratch
  PleRaw,   // per-layer-input raw embedding (prologue scratch)
  PleAll,   // all layers' per-layer inputs [num_layers * ple] (prologue output)
  // Gated-attention / linear-attention (delta-net) working set. Models that don't
  // use these simply never emit ops touching them.
  QPair,   // fused q|gate projection, before it is split into Q and QGate
  QGate,   // sigmoid gate applied to the attention output
  LinMix,  // fused linear-attention qkv mixture (post conv1d+silu)
  LinZ,
  LinA,
  LinB,  // delta-net gate / A / B projections
  LinQ,
  LinK,
  LinV,    // delta-net q/k/v after head expansion
  LinAtt,  // delta-net output (pre out-projection)
  // Mixture-of-Experts working set.
  MoeRouterIn,  // hidden after the router's weightless norm + learned scale
  MoeLogits,    // router logits [num_experts]
  MoeInter,     // [top_k, moe_inter] gelu(gate)*up, per selected expert
  MoeOut,       // [hidden] routing-weighted sum over the selected experts
  // DeepSeek-V2 MLA working set (latent attention). MlaCkv = kv_a_proj output [kv_lora+qk_rope],
  // MlaLatent = kv_a_layernorm(latent) [kv_lora], MlaKvb = kv_b_proj output [nh*(qk_nope+v_head)].
  MlaCkv,
  MlaLatent,
  MlaKvb,
  Count
};

// Which RoPE / attention table + geometry an attention op uses. Resolved per
// layer at build time (sliding layers use a different theta table than full).
enum class RopeTable : std::uint8_t { Sliding, Full };

// Where a scalar multiplier comes from (resolved to a concrete value at build,
// except SqrtHeadDim which is a geometry constant folded in at build too).
enum class ScaleKind : std::uint8_t { Constant };

enum class OpKind : std::uint8_t {
  RmsNorm,          // rows×cols groups; weight==nullptr ⇒ weightless (ones)
  Gemv,             // rowmajor half GEMV: out[out_dim] = W[out_dim×in_dim] · in
  Rope,             // in-place RoPE over `heads` heads of `head_dim`, at position
  ScaleCopy,        // out = in * scale
  CopySlot,         // out = in (device-to-device; e.g. V shares raw k_proj output)
  KvStore,          // append K,V slots to this layer's cache at `position`
  Attention,        // single-query attention over the cache (sliding window aware)
  GeluMul,          // out = gelu(a) * b   (GeGLU / PLE gate)
  AddInplace,       // out += in           (residual; out defaults to X)
  AddRmsNorm,       // fused residual add + RMSNorm, never emitted by the builder: the Metal
                    // engine folds an AddInplace(delta -> X) immediately followed by
                    // RmsNorm(X -> XNorm) into one pass. in = delta, aux slot = X (residual,
                    // updated in place), out = XNorm. Saves a full read+write of the residual.
  EmbeddingLookup,  // out = embed_table[token]  (token from the executor's device buffer)
  LmHead,           // logits[vocab] (float) = W[vocab×in_dim] · in   (writes the executor's logits)

  // ── extensions for gated / linear-attention ("delta-net") architectures ──
  // These are general ops, not a model: any architecture that needs them declares
  // them in its plan. The kernels already exist in the shared kernel library.
  SiluMul,              // out = silu(a) * b   (SwiGLU; the SiLU sibling of GeluMul)
  SplitHeadHalves,      // fused q|gate -> out (Q) + out2 (QGate), interleaved per head
  SigmoidGate,          // in-place: out *= sigmoid(aux slot)   (gated attention output)
  LinearConv1d,         // short causal depthwise conv + SiLU over the fused qkv mixture;
                        // reads/writes this layer's rolling conv state
  RepeatLinearHeads,    // expand the fused mixture into per-head q/k/v (GQA-style repeat)
  LinearAttentionStep,  // gated delta-net recurrence; reads/writes this layer's
                        // recurrent state. The linear-attention analogue of Attention.

  // ── extensions for Mixture-of-Experts ──
  // Also general ops. The selected experts stay on the device end to end: the
  // router writes indices/weights to device buffers and the expert ops read them
  // there, so a token never round-trips to the host mid-layer and the decode graph
  // remains capturable.
  MulVec,          // out = in * weight_vector * scale   (router pre-projection gain)
  MoeRouterTopk,   // logits -> softmax -> top-k -> renormalise -> *per_expert_scale;
                   // writes the executor's device topk index/weight buffers.
                   // `weight` = optional per-expert gain. cols = experts, heads = top_k.
  MoeGateUpGeglu,  // per selected expert: gelu(gate)*up  -> MoeInter
                   // cols = moe_inter, in_dim = hidden, heads = top_k
  MoeDownAccum,    // sum_k topk_weight[k] * down[expert_k] . MoeInter[k]  -> MoeOut
                   // cols = hidden, in_dim = moe_inter, heads = top_k

  // ── DeepSeek-V2 MLA (Multi-head Latent Attention) ──
  MlaAssembleRope,  // assemble per-head K=[k_nope|roped shared k_pe] (from MlaKvb + MlaCkv's k_pe) and
                    // V=[v|zeros] into Slot::K/Slot::V, and interleaved-rope Slot::Q's q_pe in place.
                    // heads=nh, head_dim=qk_nope+qk_rope, key_head_dim=qk_nope, value_head_dim=v_head,
                    // rotary_dim=qk_rope, in_dim=kv_lora, scale=attention_scaling, aux_ptr=inv_freq.

  // ── extensions for vision encoders ──
  // A vision tower differs from a text decoder in only a few places: positions are
  // 2-D, the input is pixels not token ids, and the patch grid is pooled to a fixed
  // number of soft tokens. Each is one op; norms/GEMM/GeGLU/attention are shared.
  PatchEmbed,      // pixels -> embeddings (+ two-axis learned position table)
                   // cols = hidden, in_dim = patch_dim, rows = pos_table_size,
                   // weight = input_proj, aux_ptr = position table
  Rope2D,          // in-place 2-D RoPE: first half of each head rotates by the patch's
                   // x coord, second half by its y. heads, head_dim.
  AvgPoolPatches,  // k x k spatial average pool + gain
                   // cols = hidden, heads = k, kv_heads = cells_x, rows = out_tokens,
                   // scale = gain
  Standardize,     // x = (x - weight) * aux_ptr, broadcast over tokens. cols = hidden.
};

// One resolved op. Fields are a union-by-convention keyed on `kind`; only the
// members relevant to `kind` are set (kept as a flat struct for cache-friendly
// iteration and trivial copyability; no virtuals in the hot loop).
struct Op {
  OpKind kind;
  Slot in = Slot::X;
  Slot out = Slot::X;
  Slot in2 = Slot::X;  // GeluMul's second operand (when aux_ptr is null)
  // Opaque device handles to fp16 weights. Deliberately void*, not __half*: this
  // IR is the backend seam, and a Metal executor must be able to include it with
  // no CUDA toolkit present. Each backend casts to its own half type on read;
  // writes need no cast (any object pointer converts to void* implicitly).
  const void* weight = nullptr;   // bound at build; nullptr = weightless norm
  const void* aux_ptr = nullptr;  // GeluMul raw 2nd operand (PLE per-layer input); overrides in2
  // Gemv: optional additive bias over out_dim (Qwen2's Q/K/V projections have one;
  // Llama's do not). nullptr = no bias, which is every model the CUDA executor
  // currently plans, so its behaviour is unchanged.
  //
  // The .ll2c stores Q/K/V's biases fused as one `attention.bqkv` tensor laid out
  // [bq | bk | bv], so all three ops point at the same handle and are distinguished
  // by bias_offset (in elements, not bytes).
  const void* bias = nullptr;
  // Optional rowwise-int8 PREFILL form of a quantized weight (CUDA sequence GEMMs);
  // executors that don't sequence-prefill simply ignore these.
  const void* pf_qweight = nullptr;
  const void* pf_qscales = nullptr;
  int bias_offset = 0;
  // Quantized Gemv: when qweight is set the op runs a weight-only matvec instead
  // of the fp16 one. qbits picks the encoding: 8 = int8, 4 = int4 (packed
  // two-per-byte). qgroup picks the scale granularity: 0 = one scale per row,
  // >0 = one scale per that many contiguous input features (a power of two), with
  // `qscales` laid out [out_dim, quant_group_count(in_dim, qgroup)] row-major.
  // Resolved at load like everything else, so the hot loop stays branch-free over
  // model identity; the op just carries a different weight encoding.
  // Opaque, for the same reason `weight` is: on Metal a quantized weight is an
  // MTLBuffer object, not an address, so the IR cannot name its element type.
  const void* qweight = nullptr;  // packed int4 (two per byte) or int8
  const void* qscales = nullptr;  // float[out_dim, quant_group_count(in_dim, qgroup)]
  int qbits = 0;
  int qgroup = 0;
  int rows = 1;      // RmsNorm groups (heads for q/k norm, else 1)
  int cols = 0;      // RmsNorm width / Gemv out_dim / GeluMul length
  int in_dim = 0;    // Gemv in_dim
  int heads = 0;     // Rope/Attention query heads
  int kv_heads = 0;  // Attention kv heads
  int head_dim = 0;  // Rope/Attention head_dim
  RopeTable rope_table = RopeTable::Full;
  bool full_attention = false;  // Attention: full (no window) vs sliding
  int sliding_window = 0;       // Attention: window size (0 = unbounded)
  float scale = 1.0f;           // ScaleCopy multiplier (SqrtHeadDim/layer scalar folded in)

  // ── extension fields (gated / linear-attention) ──
  Slot out2 = Slot::X;       // SplitHeadHalves' second output (the gate half)
  bool norm_offset = false;  // RmsNorm uses (1 + weight) instead of weight
  // MoeGateUpGeglu: the expert FFN's gate activation. The op's name says Geglu because Gemma 4
  //, its first tenant, is GeGLU, but Mixtral's experts are SwiGLU, so the activation has to
  // travel with the op rather than with its name.
  //
  // defaults true, which is deliberate: Gemma 4's builder predates this field and does not set
  // it, and the CUDA kernel hardcodes tanh-GELU. Defaulting false would make an unset field mean
  // "SiLU", so the day someone wires this flag into that kernel, Gemma 4 would silently change
  // activation and produce fluent nonsense. True means "unset == what it does today".
  bool mlp_gelu = true;
  // MoeRouterTopk: renormalise the selected experts' softmax probs to sum 1. defaults true (Gemma /
  // Mixtral). DeepSeek-V2-Lite sets norm_topk_prob=false, so it emits this false to keep the raw
  // top-k softmax weights.
  bool moe_renorm = true;
  int rotary_dim = 0;        // Rope: >0 rotates only the first rotary_dim of each
                             // head (partial RoPE) and pairs Q (in) with K (in2)
  // Delta-net geometry + its extra weights. The recurrence needs a float RMS-norm
  // weight, a float A_log, and an fp16 dt_bias alongside the usual projections.
  int num_k_heads = 0;
  int num_v_heads = 0;
  int key_head_dim = 0;
  int value_head_dim = 0;
  int conv_kernel = 0;  // LinearConv1d: causal kernel width
  // Offset into the aux slot (not a raw pointer): the per-layer-input gate reads layer
  // L's window of the wide PLE tensor. A baked raw pointer works at one token and reads
  // the wrong rows over a sequence, so the slot + offset is the addressable form.
  int aux_offset = 0;
  float eps = 1e-6f;  // RmsNorm / LinearAttentionStep epsilon
  // Clipped projections (Gemma 4 E2B's vision tower ships per-projection activation
  // bounds and clamps both the input and the output of every linear). Infinite by
  // default, so every other model's Gemv is untouched. The input clamp is the one
  // that matters numerically; it changes the dot product, not just its range.
  float clip_in_min = -std::numeric_limits<float>::infinity();
  float clip_in_max = std::numeric_limits<float>::infinity();
  float clip_out_min = -std::numeric_limits<float>::infinity();
  float clip_out_max = std::numeric_limits<float>::infinity();
  const float* auxf_a = nullptr;  // delta-net RMS-norm weight (float)
  const float* auxf_b = nullptr;  // delta-net A_log (float)
};

// The forward as data: one op list per layer (conditionals already resolved:
// shared layers simply omit the KV-compute ops, non-PLE models omit PLE ops).
struct LayerPlan {
  std::vector<Op> ops;
  int layer_index = 0;
};

struct ModelPlan {
  // Index into `prologue` at which the token embeddings are final; i.e. straight after
  // the embedding lookup and its scale, and before anything that reads them. Multimodal
  // prefill splices image embeddings in here. It matters: Gemma E2B projects its
  // per-layer inputs from the embeddings, so splicing after the prologue would compute
  // them from the placeholder token instead of the image (HF projects them post-scatter).
  std::size_t embed_ready = 0;
  std::vector<Op> prologue;       // token → embeddings (+ scale, PLE build)
  std::vector<LayerPlan> layers;  // the tower
  std::vector<Op> epilogue;       // final norm → LM head (logits); softcap is host-side
};

}  // namespace opplan
}  // namespace engine
