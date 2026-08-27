// Shared decode driver: the prefill + autoregressive-decode + sample + stop
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

struct DecodeParams {
  int max_new_tokens = 256;
  float temperature = 0.0f;  // <= 0 => greedy
  int top_k = 0;
  float top_p = 1.0f;
  float repetition_penalty = 1.0f;
  int no_repeat_ngram_size = 0;
  int seed = -1;  // >= 0 reseeds the sampler RNG
  // Below this many generated tokens, neither a stop token nor the on_token
  // callback may end the loop. Masking the EOS logit alone is not enough: a chat
  // template's turn marker is a different id, is detected by the transport rather
  // than by is_stop, and ends generation through on_token returning false.
  int min_new_tokens = 0;
  bool include_prompt = false;  // return prompt+generated (vs generated only)
  // Optional grammar hooks (kept as std::function so the driver doesn't depend on
  // the grammar module). mask: set disallowed logits to -inf; accept: advance the
  // grammar state by the chosen token.
  std::function<void(std::vector<float>&)> grammar_mask;
  std::function<void(int)> grammar_accept;
};

// The contract a decoder-only engine exposes to the driver. Positions are
// presented strictly increasing (prefill 0..P-1, then decode P, P+1, ...), so an
// engine may reuse KV state across calls. Only the pure-virtuals are mandatory;
// the rest have defaults matching the simplest engine (single EOS, host-side
// sampling, no persistent state).
class SequenceModel {
public:
  virtual ~SequenceModel() = default;
  virtual int vocab() const = 0;
  virtual int eos_id() const = 0;
  virtual int max_context() const = 0;
  // Process one token at `position`; when want_logits, make logits() valid for
  // that position (size == vocab()). Engines with a device-side sample() may keep
  // logits on device and only fill logits() lazily.
  virtual void step(int token, int position, bool want_logits) = 0;
  virtual std::vector<float>& logits() = 0;

  // --- optional hooks (defaults keep the simple single-flight behavior) ---
  virtual void reset_state() {}  // clear KV/recurrent state
  virtual void synchronize() {}  // block until pending work done
  virtual bool is_stop(int token) const {
    return token == eos_id();
  }  // dual-EOS engines override
  // Sample the next token from the current step's logits. Default: shared host
  // sampler (+ optional grammar mask). Engines with a device-argmax greedy fast
  // path override this to keep it.
  virtual int sample(const DecodeParams& params, const std::vector<int>& history);
  // Ingest the whole prompt (only the last token needs logits). Default: the per-token
  // step loop. Engines with a batched prefill override this; the plan engine's
  // sequence prefill is ~100x the per-token rate on long prompts.
  virtual void prefill(const std::vector<int>& prompt) {
    const int P = static_cast<int>(prompt.size());
    for (int i = 0; i < P; ++i) step(prompt[i], i, i == P - 1);
  }
};

// Prefill `prompt` (only the last token computes logits), then decode up to
// max_new_tokens: sample each step via the shared sampler, invoke on_token
// (return false to stop), stop on EOS or when the context is full. Returns the
// newly generated token ids (prompt excluded). Fills *stats when non-null.
std::vector<int> run_decode(SequenceModel& model, const std::vector<int>& prompt,
                            const DecodeParams& params, const std::function<bool(int)>& on_token,
                            BenchmarkStats* stats);

}  // namespace runtime
}  // namespace engine
