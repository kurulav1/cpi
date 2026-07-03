#pragma once

#include <functional>
#include <vector>

namespace engine {

class LlamaEngine;

// Aggregate stats for a speculative-decoding run (for logging / benchmarking).
struct SpeculativeStats {
  int rounds = 0;    // verify rounds (= target forward passes)
  int drafted = 0;   // draft tokens proposed
  int accepted = 0;  // draft tokens accepted (excludes correction/bonus tokens)
  int emitted = 0;   // total tokens emitted

  double accept_rate() const {
    return drafted > 0 ? static_cast<double>(accepted) / drafted : 0.0;
  }
  double tokens_per_round() const {
    return rounds > 0 ? static_cast<double>(emitted) / rounds : 0.0;
  }
};

// Greedy speculative decoding over a draft + target engine pair that share a
// tokenizer/vocabulary. Lossless w.r.t. pure-greedy (temperature 0, no
// repetition penalty) decoding of the target: the target's argmax is always the
// emitted token, the draft only proposes candidates the target then confirms.
class SpeculativeDecoder {
public:
  // `draft` and `target` must already be initialized with the same tokenizer
  // and a compatible max_context. `spec_tokens` (K) is the number of tokens the
  // draft proposes per round; clamped to >= 1.
  SpeculativeDecoder(LlamaEngine& draft, LlamaEngine& target, int spec_tokens);

  // Generates up to `max_new_tokens` greedily. `on_token` is invoked for each
  // emitted token id; returning false stops generation. Returns the generated
  // token ids (excluding the prompt). Stops on `eos_token_id` (>= 0).
  std::vector<int> generate(const std::vector<int>& prompt_tokens, int max_new_tokens,
                            int eos_token_id, const std::function<bool(int)>& on_token);

  const SpeculativeStats& stats() const {
    return stats_;
  }

private:
  LlamaEngine& draft_;
  LlamaEngine& target_;
  int k_;
  SpeculativeStats stats_;
};

}  // namespace engine
