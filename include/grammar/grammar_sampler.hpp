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
#include <memory>
#include <string>
#include <vector>

#include "grammar/grammar.hpp"

namespace grammar {

// Holds the lazy DFA and the per-state allowed-token bitmasks across calls.
// Defined in the .cpp; GrammarSampler's destructor lives there too, which is what
// lets a unique_ptr to an incomplete type work here.
class MaskCache;

// Per-tokenizer precomputation for the masking fast path. Depends only on the
// piece table, not on the grammar, so it can be built once per tokenizer and
// shared by every request. Building it decodes the whole vocabulary: measured at
// ~26 ms for Gemma's 262144 tokens, which every schema request used to pay in the
// GrammarSampler constructor before generating a token.
//
// cps[id] is the token's codepoints; simple[id] is 1 when the token decodes to
// whole codepoints and is eligible for the fast path.
struct TokenTables {
  std::vector<std::vector<std::uint32_t>> cps;
  std::vector<char> simple;

  // Simple tokens grouped by the codepoint they start with, so a state that
  // forbids that codepoint skips every token beginning with it in one lookup
  // instead of walking each in turn.
  //
  // This is what a mask costs. Building one used to test all 128256 tokens
  // against the DFA even though nearly all of them die on their first codepoint,
  // and a vocabulary that size has only a few thousand distinct first
  // codepoints. In a restrictive state, inside a literal like "add", exactly one
  // bucket survives.
  std::vector<std::uint32_t> first_cps;    // distinct first codepoints
  std::vector<std::vector<int>> buckets;   // token ids, parallel to first_cps
  std::vector<int> non_simple;             // ids that need the would_accept path
};

// Builds the tables for `token_pieces`. Callers that serve many requests from one
// tokenizer should call this once and pass the result to every GrammarSampler.
std::shared_ptr<const TokenTables> build_token_tables(const std::vector<std::string>& token_pieces);

class GrammarSampler {
public:
  // `token_pieces[id]` is the raw bytes token `id` emits; an empty entry marks a
  // special/control token the grammar must not consume. `eos_token_id` may be -1.
  // `tables` may be shared across requests; when null the sampler builds its own,
  // which is the right thing for a one-shot CLI run and the wrong thing for a
  // server.
  GrammarSampler(Grammar grammar, const std::vector<std::string>& token_pieces, int eos_token_id,
                 std::shared_ptr<const TokenTables> tables = nullptr);

  GrammarSampler(const GrammarSampler&) = delete;
  GrammarSampler& operator=(const GrammarSampler&) = delete;

  // Reports the timings below when CPI_GRAMMAR_PROFILE is set. Masking is the
  // suspected cost centre of constrained decoding: apply_mask rebuilds its lazy
  // DFA on every call and walks the whole vocabulary, so it is measured before
  // anything is optimised around it.
  ~GrammarSampler();

  // Sets logits[t] = -inf for every token that cannot legally continue the
  // current grammar state. The EOS token is kept iff the grammar can terminate
  // here; special tokens (empty piece) are always masked except EOS.
  void apply_mask(std::vector<float>& logits) const;

  // Same, over a raw row. Speculative verification masks rows of a [K][vocab]
  // block in place, and copying each row into a vector first would cost more than
  // the masking does.
  void apply_mask(float* logits, std::size_t n) const;

  // Advances the grammar by `token_id`. Returns true when the grammar has
  // reached a complete value after this token (the caller may stop). Accepting
  // EOS returns true without consuming bytes.
  bool accept(int token_id);

  // True when the grammar has matched a complete value and may legally stop.
  bool can_terminate() const {
    return state_.can_terminate();
  }

private:
  // True when token `t` may legally continue from interned DFA state `start`.
  // This is the per-token decision apply_mask used to inline; it is factored out
  // so the cached bitmask build and the uncached walk cannot drift apart.
  bool token_allowed(int start, bool terminable, bool partial_pending, std::size_t t) const;

  // Fills `out` with ceil(vocab/32) words, bit set meaning the token is allowed.
  void build_bitmask(int start, bool terminable, std::size_t vocab,
                     std::vector<std::uint32_t>& out) const;

  Grammar grammar_;
  const std::vector<std::string>& token_pieces_;
  int eos_token_id_;
  GrammarState state_;

  // Shared when the caller supplied it, otherwise built for this sampler alone.
  std::shared_ptr<const TokenTables> tables_;

  // The DFA and the mask cache persist for the sampler's lifetime. Before this,
  // apply_mask built a fresh TransitionMemo on every call and threw away every
  // memoized transition, then walked the whole vocabulary again. Measured at 4.72
  // ms per call on Gemma's 262144-token vocabulary, which was most of a decode
  // step. mutable because apply_mask is const and stays that way.
  mutable std::unique_ptr<MaskCache> cache_;

  // Profiling counters, active only under CPI_GRAMMAR_PROFILE. mutable because
  // apply_mask is const and must stay so.
  bool profile_ = false;
  bool mask_cache_enabled_ = true;
  double ctor_ms_ = 0.0;
  mutable std::uint64_t mask_calls_ = 0;
  mutable double mask_ms_ = 0.0;
  mutable double mask_ms_max_ = 0.0;
  // Split out so the phases of a mask build can be attributed instead of guessed.
  mutable double slow_ms_ = 0.0;
  mutable std::uint64_t builds_ = 0;
};

}  // namespace grammar
