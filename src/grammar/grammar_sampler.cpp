#include "grammar/grammar_sampler.hpp"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>

namespace grammar {
namespace {

// Per-step lazy DFA over grammar states. A "state" is a canonicalized stack-set;
// the JSON grammar has very few distinct states, so memoizing (state, codepoint)
// transitions runs the expensive grammar walk a few thousand times per step
// instead of once per (token x codepoint) across the whole vocabulary.
class TransitionMemo {
 public:
  explicit TransitionMemo(const Grammar& grammar) : grammar_(grammar) {}

  int intern(std::vector<Stack> stacks) {
    canonicalize(stacks);
    const std::string key = key_of(stacks);
    const auto it = ids_.find(key);
    if (it != ids_.end()) {
      return it->second;
    }
    const int id = static_cast<int>(states_.size());
    states_.push_back(std::move(stacks));
    transitions_.emplace_back();
    ids_.emplace(key, id);
    return id;
  }

  // Returns the next state id for consuming `cp` from state `id`, or -1 if the
  // codepoint is rejected. Computed once per (id, cp) and cached.
  int transition(int id, std::uint32_t cp) {
    const auto it = transitions_[static_cast<std::size_t>(id)].find(cp);
    if (it != transitions_[static_cast<std::size_t>(id)].end()) {
      return it->second;
    }
    std::vector<Stack> next = grammar_accept_codepoint(grammar_, states_[static_cast<std::size_t>(id)], cp);
    const int nid = next.empty() ? -1 : intern(std::move(next));
    transitions_[static_cast<std::size_t>(id)][cp] = nid;  // re-index: `id` is stable
    return nid;
  }

 private:
  static void canonicalize(std::vector<Stack>& stacks) {
    std::sort(stacks.begin(), stacks.end());
    stacks.erase(std::unique(stacks.begin(), stacks.end()), stacks.end());
  }

  static std::string key_of(const std::vector<Stack>& stacks) {
    std::string key;
    const auto put = [&](std::size_t v) {
      key.append(reinterpret_cast<const char*>(&v), sizeof(v));
    };
    put(stacks.size());
    for (const Stack& stack : stacks) {
      put(stack.size());
      for (const Element* e : stack) {
        const std::uintptr_t p = reinterpret_cast<std::uintptr_t>(e);
        key.append(reinterpret_cast<const char*>(&p), sizeof(p));
      }
    }
    return key;
  }

  const Grammar& grammar_;
  std::vector<std::vector<Stack>> states_;
  std::vector<std::unordered_map<std::uint32_t, int>> transitions_;
  std::unordered_map<std::string, int> ids_;
};

}  // namespace

GrammarSampler::GrammarSampler(Grammar grammar,
                               const std::vector<std::string>& token_pieces,
                               int eos_token_id)
    : grammar_(std::move(grammar)),
      token_pieces_(token_pieces),
      eos_token_id_(eos_token_id),
      state_(grammar_) {
  // Precompute each token's codepoints once so masking avoids re-decoding the
  // whole vocabulary on every decode step.
  const std::size_t vocab = token_pieces_.size();
  token_cps_.resize(vocab);
  token_simple_.assign(vocab, 0);
  for (std::size_t t = 0; t < vocab; ++t) {
    if (token_pieces_[t].empty()) {
      continue;  // special tokens are handled without codepoints
    }
    if (grammar::decode_simple_codepoints(token_pieces_[t], token_cps_[t])) {
      token_simple_[t] = 1;
    } else {
      token_cps_[t].clear();
    }
  }
}

void GrammarSampler::apply_mask(std::vector<float>& logits) const {
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const bool terminable = state_.can_terminate();
  const bool partial_pending = state_.has_partial();

  // Lazy DFA: intern the current stack-set as the start state; (state, codepoint)
  // transitions are computed once and reused across every token this step.
  TransitionMemo memo(grammar_);
  const int start = memo.intern(state_.current_stacks());

  const std::size_t vocab = logits.size();
  for (std::size_t t = 0; t < vocab; ++t) {
    if (static_cast<int>(t) == eos_token_id_) {
      // EOS is permitted only at a complete value.
      if (!terminable) {
        logits[t] = neg_inf;
      }
      continue;
    }
    // Tokens beyond the piece table, or special tokens with no emitted bytes,
    // are not part of the JSON surface and are masked out.
    if (t >= token_pieces_.size() || token_pieces_[t].empty()) {
      logits[t] = neg_inf;
      continue;
    }

    // Fast path: simple (whole-codepoint) token and no pending partial UTF-8 —
    // walk its codepoints through the memoized transitions.
    if (!partial_pending && token_simple_[t] && !token_cps_[t].empty()) {
      int s = start;
      for (const std::uint32_t cp : token_cps_[t]) {
        if (s < 0) {
          break;
        }
        s = memo.transition(s, cp);
      }
      if (s < 0) {
        logits[t] = neg_inf;
      }
      continue;
    }

    // Slow path: tokens that span a partial UTF-8 boundary or invalid bytes.
    if (!state_.would_accept(token_pieces_[t])) {
      logits[t] = neg_inf;
    }
  }
}

bool GrammarSampler::accept(int token_id) {
  if (token_id == eos_token_id_) {
    return true;
  }
  if (token_id < 0 || static_cast<std::size_t>(token_id) >= token_pieces_.size()) {
    return state_.can_terminate();
  }
  const std::string& piece = token_pieces_[static_cast<std::size_t>(token_id)];
  if (piece.empty()) {
    // Special token with no bytes: leave grammar state unchanged.
    return state_.can_terminate();
  }
  state_.accept_bytes(piece);
  return state_.can_terminate();
}

}  // namespace grammar
