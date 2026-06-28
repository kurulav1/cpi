#pragma once

// GrammarSampler: bridges a compiled grammar to token-level decoding. It owns a
// Grammar and a live GrammarState and exposes the two operations the engine
// sampling loop needs:
//   - apply_mask():  set logits for tokens that cannot continue the grammar to
//                    -inf, before argmax/sampling.
//   - accept():      advance grammar state by the chosen token; report whether
//                    the grammar has reached a complete value (clean stop).
//
// The token->bytes table is supplied by the caller (see Tokenizer::token_pieces).
// Non-copyable / non-movable: GrammarState holds pointers into the owned Grammar.

#include <cstdint>
#include <string>
#include <vector>

#include "grammar/grammar.hpp"

namespace grammar {

class GrammarSampler {
 public:
  // `token_pieces[id]` is the raw bytes token `id` emits; an empty entry marks a
  // special/control token the grammar must not consume. `eos_token_id` may be -1.
  GrammarSampler(Grammar grammar, const std::vector<std::string>& token_pieces, int eos_token_id);

  GrammarSampler(const GrammarSampler&) = delete;
  GrammarSampler& operator=(const GrammarSampler&) = delete;

  // Sets logits[t] = -inf for every token that cannot legally continue the
  // current grammar state. The EOS token is kept iff the grammar can terminate
  // here; special tokens (empty piece) are always masked except EOS.
  void apply_mask(std::vector<float>& logits) const;

  // Advances the grammar by `token_id`. Returns true when the grammar has
  // reached a complete value after this token (the caller may stop). Accepting
  // EOS returns true without consuming bytes.
  bool accept(int token_id);

  // True when the grammar has matched a complete value and may legally stop.
  bool can_terminate() const { return state_.can_terminate(); }

 private:
  Grammar grammar_;
  const std::vector<std::string>& token_pieces_;
  int eos_token_id_;
  GrammarState state_;

  // Per-token precomputation for the fast masking path (built once at
  // construction): token_cps_[id] is the token's codepoints and token_simple_[id]
  // is 1 when the token decodes to whole codepoints (eligible for the fast path).
  // Non-simple tokens (mid-codepoint / invalid bytes) fall back to would_accept.
  std::vector<std::vector<std::uint32_t>> token_cps_;
  std::vector<char> token_simple_;
};

}  // namespace grammar
