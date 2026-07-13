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

#include <string>
#include <vector>

#include "engine/op_plan.hpp"

namespace engine {
namespace opplan {

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

  // Some models (Gemma-style) scale the token embedding by sqrt(hidden). Llama and
  // Qwen do not. Kept here because it is geometry, not identity.
  bool scale_embeddings = false;

  // Weight names follow the .ll2c convention (layers.N.attention.wq, ...). A model
  // whose tensors are named differently supplies its own prefix mapping.
  std::string layer_prefix = "layers.";
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

}  // namespace opplan
}  // namespace engine
