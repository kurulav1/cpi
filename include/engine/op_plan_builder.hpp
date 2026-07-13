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
