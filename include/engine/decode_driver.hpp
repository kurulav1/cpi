// Shared decode driver — the prefill + autoregressive-decode + sample + stop
// loop, extracted so fork engines (Gemma4, and future exotic architectures)
// don't each re-implement generate/generate_stream. An engine provides only a
// minimal SequenceModel (its forward + a few scalars); it gets the loop, the
// shared sampler (temperature/top-k/top-p/repetition penalty), EOS/callback
// stopping, and benchmark stats for free.
//
// This is the "reuse without touching perf" seam: LlamaEngine keeps its own
// optimized generate (CUDA-graph decode, batching, paging); the driver serves
// the single-flight fork engines. The per-token virtual dispatch is negligible
// against a memory-bound decode step.
#pragma once

#include <functional>
#include <vector>

#include "engine/engine_types.hpp"  // BenchmarkStats

namespace engine {
namespace runtime {

// The contract a decoder-only engine exposes to the driver. Positions are
// presented strictly increasing (prefill 0..P-1, then decode P, P+1, ...), so an
// engine may reuse KV state across calls.
class SequenceModel {
 public:
  virtual ~SequenceModel() = default;
  virtual int vocab() const = 0;
  virtual int eos_id() const = 0;
  virtual int max_context() const = 0;
  // Process one token at `position`; when want_logits, refresh logits() with the
  // host-side logits for that position (size == vocab()).
  virtual void step(int token, int position, bool want_logits) = 0;
  virtual std::vector<float>& logits() = 0;
};

struct DecodeParams {
  int max_new_tokens = 256;
  float temperature = 0.0f;      // <= 0 => greedy
  int top_k = 0;
  float top_p = 1.0f;
  float repetition_penalty = 1.0f;
  int no_repeat_ngram_size = 0;
  int seed = -1;                 // >= 0 reseeds the sampler RNG
};

// Prefill `prompt` (only the last token computes logits), then decode up to
// max_new_tokens: sample each step via the shared sampler, invoke on_token
// (return false to stop), stop on EOS or when the context is full. Returns the
// newly generated token ids (prompt excluded). Fills *stats when non-null.
std::vector<int> run_decode(SequenceModel& model, const std::vector<int>& prompt,
                            const DecodeParams& params,
                            const std::function<bool(int)>& on_token, BenchmarkStats* stats);

}  // namespace runtime
}  // namespace engine
