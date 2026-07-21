#pragma once

// Backend-neutral plan builders.
//
// A plan is just data (op_plan.hpp): which ops, in what order, over what geometry,
// with weights bound as opaque handles. Nothing about that is CUDA-specific, so the
// builder does not belong inside a CUDA engine -- it belongs here, where a Metal
// executor can call it too.
//
// Weights are resolved through WeightSource rather than a device-pointer map, which
// is the one thing that was backend-specific: each backend hands the builder its own
// notion of "a device handle for the tensor called X", and the plan just carries it.
//
// This file must compile with no CUDA and no Metal toolkit present.

#include <stdexcept>
#include <string>
#include <vector>

#include "engine/op_plan.hpp"

namespace engine {
namespace opplan {

// A quantized weight: the packed values plus their scales. bits == 0 means the
// backend is not quantizing this tensor and the fp16 handle should be used instead.
struct QuantWeight {
  const void* packed = nullptr;  // int4 (two per byte) or int8
  const void* scales = nullptr;  // float[out_dim, groups]
  int bits = 0;                  // 0 = not quantized, 4, or 8
  int group = 0;                 // 0 = one scale per row
};

// How the builder asks a backend for a weight. The returned handle is opaque and
// is only ever dereferenced by that backend's executor.
class WeightSource {
public:
  virtual ~WeightSource() = default;

  // Device handle for an fp16 tensor. Throws if it is not loaded.
  virtual const void* fp16(const std::string& name) const = 0;

  // True when the tensor exists at all -- used for optional weights (a model may
  // or may not have QK-norm, a bias, a tied LM head...).
  virtual bool has(const std::string& name) const = 0;

  // Quantized form of a projection, when the backend wants one. The default is "no",
  // which keeps every existing caller on the fp16 path.
  //
  // The backend decides, not the builder: whether to quantize is a memory/bandwidth
  // policy, and only the backend knows what it can execute.
  virtual QuantWeight quant(const std::string& name, int out_dim, int in_dim) const {
    (void)name;
    (void)out_dim;
    (void)in_dim;
    return {};
  }

  // HOST value of a one-element tensor. The vision builder needs the clip BOUNDS as numbers
  // (they go into Op::clip_*, not into a buffer binding), and only the backend knows how to
  // read bytes out of its container. Default throws: a builder that needs scalars on a backend
  // that cannot supply them is a wiring error, not a silent 0.
  virtual float scalar(const std::string& name) const {
    throw std::runtime_error("WeightSource::scalar not implemented for " + name);
  }
};

// Geometry of a uniform-geometry, Llama-style decoder: every layer identical,
// SwiGLU MLP, RMSNorm, full causal attention, half-split RoPE. Llama 2/3, Qwen2.5,
// Mistral. (Gemma and Qwen3.5 are NOT this -- their per-layer geometry varies, and
// they keep their own builders.)
struct LlamaGeometry {
  int num_layers = 0;
  int hidden = 0;
  int inter = 0;  // MLP intermediate
  int heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int vocab = 0;
  float rms_eps = 1e-5f;
  float rope_theta = 10000.0f;

  // Qwen2's Q/K/V projections carry an additive bias; Llama's and Mistral's do not.
  // Getting this wrong does not crash -- it produces a model that generates fluent
  // nonsense -- so it is explicit rather than inferred.
  bool has_qkv_bias = false;

  // Qwen3 applies a per-head RMSNorm to Q and to K after projection, before RoPE.
  // The weight is one [head_dim] vector shared across heads, so it is a plain
  // RmsNorm with rows = heads rather than a new op.
  bool has_qk_norm = false;

  // NOTE head_dim is NOT hidden/heads in general. Qwen3-0.6B has hidden=1024,
  // heads=16 and head_dim=128 (so q_dim=2048 != hidden). Deriving it from hidden
  // silently builds the wrong model -- callers must set it from the weights.

  // Gemma's MLP is GeGLU (tanh-GELU), not SwiGLU, and its RMSNorm weights are stored
  // as (w - 1) so the norm scales by (1 + w). Both are ops that already exist -- a
  // capability flag, not a fork.
  bool mlp_gelu = false;
  bool norm_offset = false;

  // Some models (Gemma-style) scale the token embedding by sqrt(hidden). Llama and
  // Qwen do not. Kept here because it is geometry, not identity.
  bool scale_embeddings = false;

  // Weight names follow the .ll2c convention (layers.N.attention.wq, ...). A model
  // whose tensors are named differently supplies its own prefix mapping.
  std::string layer_prefix = "layers.";

  // ---- Mixture of Experts (Mixtral) ---------------------------------------
  //
  // num_experts == 0 is a dense SwiGLU/GeGLU MLP, i.e. everything above. Non-zero swaps the
  // MLP block for router -> top-k -> per-expert FFN -> routing-weighted sum. Same attention,
  // same norms, same everything else: MoE is a capability of this geometry, not a fork.
  //
  // NOTE this is MIXTRAL's MoE. Gemma 4's is a different animal (a weightless router norm, a
  // learned per-expert gain, fused gate_up in the checkpoint) and keeps its own builder inside
  // PlanCudaEngine. Sharing them would mean pretending two different models are one.
  int num_experts = 0;
  int experts_per_tok = 0;
  int expert_inter = 0;  // per-expert FFN width; falls back to `inter` when 0

  // The expert kernels want ONE tensor per layer, not one per expert:
  //   <prefix>feed_forward.experts.gate_up  [num_experts * 2 * expert_inter, hidden]
  //   <prefix>feed_forward.experts.down     [num_experts * hidden, expert_inter]
  // A .ll2c stores them per expert (experts.<e>.w1 / .w3 / .w2), so a backend's WeightSource
  // is what concatenates them -- the builder names what it needs and the backend materialises
  // it, which keeps the layout decision next to the memory that holds it.
};

// Emits the decode plan: a prologue (embedding lookup), one op list per layer, and
// an epilogue (final norm + LM head).
//
// The ops emitted are exactly the eleven that a dense fp16 decode needs:
//   EmbeddingLookup, RmsNorm, Gemv, Rope, KvStore, Attention, AddInplace,
//   SiluMul, ScaleCopy, CopySlot, LmHead
// which is the same set the Metal backend implements. Any backend that can execute
// those eleven can run this plan.
ModelPlan build_llama_plan(const LlamaGeometry& g, const WeightSource& w);

// Geometry of a Qwen3.5-style MIXED-attention decoder: most layers are gated delta-net (a linear
// recurrence), a few are full attention whose q projection emits [q | gate] per head. Which is
// which comes from `layer_is_linear`, per layer, because it is not derivable from anything else.
//
// This is a separate builder rather than a flag on LlamaGeometry because the two layer kinds have
// disjoint tensor sets and disjoint op sequences -- a "uniform geometry" struct cannot describe a
// model whose geometry is not uniform. It is still the SHARED core: every op below already exists
// and every backend executes them with the same kernels.
//
// Names are the canonical .ll2c ones the converter emits, reached through WeightSource. CUDA's own
// Qwen3.5 path predates this and reads HF names straight from safetensors, so adopting this there
// needs a name-mapping WeightSource -- see the note at PlanCudaEngine::build_qwen35_plan.
struct Qwen35Geometry {
  int num_layers = 0;
  int hidden = 0;
  int inter = 0;
  int vocab = 0;
  float rms_eps = 1e-6f;
  float rope_theta = 10000000.0f;

  // Full-attention layers.
  int heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int rotary_dim = 0;  // partial RoPE: rotate only the first rotary_dim of each head

  // Delta-net layers.
  int lin_key_heads = 0;
  int lin_value_heads = 0;
  int lin_key_head_dim = 0;
  int lin_value_head_dim = 0;
  int lin_conv_kernel = 0;

  // Per layer: true = gated delta-net, false = full attention. MUST be num_layers long.
  std::vector<bool> layer_is_linear;
};

ModelPlan build_qwen35_plan(const Qwen35Geometry& g, const WeightSource& w);

// Geometry of a Gemma 4 decoder. A third sibling rather than a flag on LlamaGeometry, for the
// reason given at num_experts above: this family's per-layer geometry is not uniform, and its MoE
// is not Mixtral's. Layers alternate SLIDING and FULL attention with a DIFFERENT head_dim, kv-head
// count and RoPE base each (E2B: 256/512), the last `num_kv_shared_layers` reuse an earlier
// layer's K/V instead of projecting their own, and E2B additionally carries Per-Layer Embeddings
// that the prologue folds in.
//
// Every op this needs already exists and every backend executes them with the same kernels -- this
// is the shared core, not a fork. Confirmed by counting: all 26 OpKinds are implemented on both
// the CUDA and Metal executors.
struct Gemma4Geometry {
  int num_layers = 0;
  int hidden = 0;
  int inter = 0;  // MLP intermediate, DOUBLED on shared layers when use_double_wide_mlp
  int vocab = 0;
  int heads = 0;
  float rms_eps = 1e-6f;
  int sliding_window = 0;

  // Per layer TYPE, because one head_dim cannot describe this model.
  int head_dim_sliding = 0, head_dim_full = 0;
  int kv_heads_sliding = 0, kv_heads_full = 0;

  // ...and a different RoPE base per layer type, with PARTIAL rotary on the full layers only.
  // These must be set: Op::scale carries theta on backends that compute RoPE in-shader rather
  // than from a table, and it defaults to 1.0f -- a theta of 1.0 makes every lane rotate at the
  // same frequency, which is a wrong model rather than an error.
  float rope_theta_sliding = 10000.0f;
  float rope_theta_full = 1000000.0f;
  float partial_rotary_full = 1.0f;  // fraction of head_dim that rotates on FULL layers

  // Per layer: 1 = full attention, 0 = sliding. MUST be num_layers long.
  std::vector<int> layer_full;

  // Layers at or after this index reuse another layer's K/V and project none of their own.
  // 0 means no sharing at all (the 12B and the MoE).
  int first_shared_layer = 0;
  bool attention_k_eq_v = false;   // full layers: V reuses the raw k_proj output, there is no v_proj
  bool use_double_wide_mlp = false;

  // Per-Layer Embeddings width. 0 = this checkpoint has none (12B, MoE), and the whole PLE
  // prologue and per-layer injection drop out.
  int ple = 0;

  // Per-layer output gain. HOST floats, not weight handles: the value is multiplied into a
  // ScaleCopy op, so the builder needs the NUMBER, and only a backend knows how to read bytes out
  // of its container. Empty means "all 1.0", which is what a checkpoint without layer_scalar means.
  std::vector<float> layer_scalar;

  // Gemma 4 ships HuggingFace tensor names, so both prefixes are explicit rather than assumed.
  std::string weight_prefix = "model.language_model.";

  [[nodiscard]] bool has_ple() const { return ple > 0; }
  [[nodiscard]] int head_dim_of(int layer) const {
    return layer_full[layer] ? head_dim_full : head_dim_sliding;
  }
  [[nodiscard]] int kv_heads_of(int layer) const {
    return layer_full[layer] ? kv_heads_full : kv_heads_sliding;
  }
  // Only the FULL layers of a k_eq_v checkpoint share V with K.
  [[nodiscard]] bool k_eq_v(int layer) const { return attention_k_eq_v && layer_full[layer] != 0; }
  [[nodiscard]] bool is_shared(int layer) const {
    return first_shared_layer > 0 && layer >= first_shared_layer;
  }
};

// NOTE on RMSNorm: Gemma 4 uses PLAIN rmsnorm, NOT the (1 + w) form. Gemma 1/2 store their norm
// weights as (w - 1); Gemma 4 does not -- measured on the E2B checkpoint, whose input_layernorm
// values are 9.375 / 7.97 / 10.69, i.e. raw multiplicative gains rather than offsets around zero.
// Applying the offset form here scales by ~2 at every norm, which compounds and overflows fp16 by
// about layer 2. So no op below sets norm_offset.
ModelPlan build_gemma4_plan(const Gemma4Geometry& g, const WeightSource& w);

// Geometry of Gemma 4's VISION tower (`vision_config` in the checkpoint). The tower is a
// text-shaped sandwich-norm encoder with three differences, each already an op: positions are
// 2-D (Rope2D), the input is pixels (PatchEmbed), and attention is bidirectional (the executor's
// cacheless mode). Pooling and projection into text space are NOT in the plan -- pooling changes
// the token count, so the executor is re-entered with the new count; see the engines'
// encode_image.
struct Gemma4VisionGeometry {
  bool present = false;
  int hidden = 0;
  int layers = 0;
  int heads = 0;
  int kv_heads = 0;
  int head_dim = 0;
  int intermediate = 0;
  int patch_size = 0;
  int pos_table_size = 0;
  int pooling_kernel = 1;
  float rms_eps = 1e-6f;
  float rope_theta = 100.0f;
  bool standardize = false;
  // E2B ships per-projection activation bounds and HF CLAMPS with them; the 26B sets this false
  // and ships none. Skipping them does not crash -- it silently changes the numbers.
  bool clipped_linears = false;
};

// The vision plan, previously built privately inside PlanCudaEngine::build_vision_plan. One
// description, two executors -- the same argument that moved the text builders here. Clip bounds
// are read through WeightSource::scalar (only when g.clipped_linears).
ModelPlan build_gemma4_vision_plan(const Gemma4VisionGeometry& g, const WeightSource& w);

}  // namespace opplan
}  // namespace engine
