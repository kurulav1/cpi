#pragma once

// Per-request decoding constraints threaded into the engines' generate_stream.
// Kept as a lightweight header (GrammarSampler is only forward-declared) so the
// engine headers (including the CUDA ones) stay cheap to include. Engines that
// actually apply the grammar include "grammar/grammar_sampler.hpp" in their .cpp.

namespace grammar {
class GrammarSampler;
}

namespace engine {

struct GenerationConstraints {
  // When non-null, decoding is constrained to this grammar: the sampler masks
  // logits for tokens that cannot continue the grammar before argmax/sampling,
  // and advances the grammar with each accepted token. Owned by the caller for
  // the duration of the generate_stream call.
  grammar::GrammarSampler* grammar = nullptr;

  // Sampling RNG seed (>= 0 to fix; -1 leaves the engine default). Only affects
  // temperature > 0 decoding; JSON generation runs at temperature 0 (greedy).
  int seed = -1;

  // Minimum number of tokens to generate before EOS may be sampled. While fewer
  // than this many tokens have been produced, the EOS logit is masked to -inf so
  // greedy (temperature 0) decoding cannot terminate early; it fixes the
  // failure where the model emits EOS the moment its distribution collapses on a
  // repeated/round-number run (e.g. "revenue":4000000) and truncates the output.
  // 0 disables (default).
  int min_new_tokens = 0;
};

}  // namespace engine
