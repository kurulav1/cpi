#pragma once

// The Metal executor for the op-plan IR -- the Apple Silicon sibling of
// PlanCudaEngine.
//
// Same plan, different backend: build_llama_plan() produces the op list, and this
// walks it. The IR carries weights as opaque handles, which here are MTLBuffer
// objects rather than raw device pointers -- that is exactly why op_plan.hpp types
// them as void* and not __half*.
//
// Unified memory removes most of what the CUDA engine spends code on: a weight is
// "uploaded" by memcpy into a shared buffer, and logits are read back by simply
// looking at the buffer. No streams, no staging, no async copies.
//
// Scope: fp16, dense, uniform-geometry decode (Llama/Qwen2). No quantization -- Metal
// has no __dp4a equivalent, so the int4/int8 paths need a different kernel and are
// deliberately out of scope here.

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "engine/op_plan.hpp"
#include "engine/op_plan_builder.hpp"
#include "model/llama_config.hpp"
#include "model/weight_loader.hpp"
#include "runtime/metal_context.hpp"

namespace engine {

class PlanMetalEngine {
public:
  PlanMetalEngine();
  ~PlanMetalEngine();

  // False when there is no Metal GPU (any VM without a paravirtual device). The
  // caller must check rather than assume a Mac has one.
  bool available() const;
  std::string device_name() const;
  const std::string& last_error() const {
    return last_error_;
  }

  // Loads a .ll2c model, uploads its weights, and builds the plan.
  void open(const std::string& weights_path, int max_context = 2048);

  const model::LlamaConfig& config() const {
    return cfg_;
  }

  // Runs one decode step and returns the logits (a view into the shared buffer,
  // valid until the next step).
  const std::vector<float>& forward_token(int token, int position);

  // Greedy decode. Argmax runs on the GPU so the vocab never crosses to the host.
  std::vector<int> generate_greedy(const std::vector<int>& prompt, int max_new);

  // Wall time of the last prompt prefill, and how many tokens it covered.
  double last_prefill_ms() const {
    return prefill_ms_;
  }
  int last_prefill_tokens() const {
    return prefill_tokens_;
  }

private:
  void execute_ops(const std::vector<opplan::Op>& ops, int layer, int position, int tokens);
  void encode_forward(int token, int position);
  // Encodes a whole prefill chunk: T tokens through the tower in one pass.
  void encode_prefill(const std::vector<int>& tokens, int start_position);
  void* slot(opplan::Slot s) const;

  runtime::MetalContext ctx_;
  model::WeightLoader weights_;
  model::LlamaConfig cfg_{};
  opplan::ModelPlan plan_;
  int max_context_ = 0;
  int max_prefill_ = 0;  // slots are sized for this many tokens at once

  // name -> device buffer. Owns every weight for the model's lifetime.
  std::unordered_map<std::string, runtime::MetalBuffer> wbuf_;

  std::vector<runtime::MetalBuffer> slots_;  // indexed by opplan::Slot
  std::vector<runtime::MetalBuffer> k_cache_;
  std::vector<runtime::MetalBuffer> v_cache_;

  runtime::MetalBuffer tok_buf_;      // int32[1]
  runtime::MetalBuffer seq_tok_buf_;  // int32[max_prefill] -- a whole prompt chunk
  runtime::MetalBuffer pos_buf_;      // int32[1]
  runtime::MetalBuffer logits_buf_;   // float[vocab]
  runtime::MetalBuffer argmax_val_;   // float[parts]
  runtime::MetalBuffer argmax_idx_;   // int32[parts]
  runtime::MetalBuffer argmax_out_;   // int32[1]

  std::vector<float> logits_;
  double prefill_ms_ = 0.0;
  int prefill_tokens_ = 0;
  std::string last_error_;

  class MetalWeights;
  std::unique_ptr<MetalWeights> wsrc_;
};

}  // namespace engine
