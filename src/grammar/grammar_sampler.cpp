#include "grammar/grammar_sampler.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

namespace grammar {
namespace detail {

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
    std::vector<Stack> next =
        grammar_accept_codepoint(grammar_, states_[static_cast<std::size_t>(id)], cp);
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

}  // namespace detail

// Persistent DFA plus one allowed-token bitmask per (state, terminable) pair.
//
// A grammar visits very few distinct DFA states (a JSON schema cycles through a
// handful), so after the first few tokens nearly every apply_mask is a hash
// lookup and a word-wise scan instead of a full-vocabulary grammar walk.
//
// `terminable` is part of the key because it decides only whether EOS survives,
// and it is a function of the same stack-set the state id is interned from, so
// the pair determines the mask exactly.
class MaskCache {
public:
  explicit MaskCache(const Grammar& grammar) : memo(grammar) {}

  detail::TransitionMemo memo;
  std::unordered_map<std::uint64_t, std::vector<std::uint32_t>> masks;
  std::size_t vocab = 0;
};

namespace {

// Bounds the cache at 64 masks. Gemma's 262144-token vocabulary is 32 KB per
// mask, so this caps a sampler at 2 MB, and samplers are per request. A grammar
// with more live states than this still works: the overflow states fall back to
// the uncached walk rather than evicting.
constexpr std::size_t kMaxCachedMasks = 64;

}  // namespace

std::shared_ptr<const TokenTables> build_token_tables(
    const std::vector<std::string>& token_pieces) {
  auto tables = std::make_shared<TokenTables>();
  const std::size_t vocab = token_pieces.size();
  tables->cps.resize(vocab);
  tables->simple.assign(vocab, 0);
  for (std::size_t t = 0; t < vocab; ++t) {
    if (token_pieces[t].empty()) {
      continue;  // special tokens are handled without codepoints
    }
    if (grammar::decode_simple_codepoints(token_pieces[t], tables->cps[t])) {
      tables->simple[t] = 1;
    } else {
      tables->cps[t].clear();
    }
  }
  return tables;
}

GrammarSampler::GrammarSampler(Grammar grammar, const std::vector<std::string>& token_pieces,
                               int eos_token_id, std::shared_ptr<const TokenTables> tables)
    : grammar_(std::move(grammar)),
      token_pieces_(token_pieces),
      eos_token_id_(eos_token_id),
      state_(grammar_) {
  profile_ = std::getenv("CPI_GRAMMAR_PROFILE") != nullptr;
  // CPI_GRAMMAR_MASK_CACHE=0 forces the uncached walk on every call. Kept so the
  // cache can be A/B'd against the old behaviour on the same binary and the same
  // request, rather than comparing two different requests and calling it a
  // speedup.
  if (const char* mc = std::getenv("CPI_GRAMMAR_MASK_CACHE")) {
    mask_cache_enabled_ = !(mc[0] == '0' && mc[1] == 0);
  }
  const auto ctor_t0 = std::chrono::steady_clock::now();
  tables_ = tables ? std::move(tables) : build_token_tables(token_pieces_);
  cache_ = std::make_unique<MaskCache>(grammar_);
  if (profile_) {
    ctor_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - ctor_t0)
                   .count();
  }
}

GrammarSampler::~GrammarSampler() {
  if (!profile_) return;
  // One line per sampler. The sampler is constructed per request, so ctor_ms is
  // a fixed cost every schema request pays, separate from the per-token masking.
  std::fprintf(stderr,
               "[grammar] vocab=%zu ctor_ms=%.2f mask_calls=%llu mask_ms_total=%.2f "
               "mask_ms_avg=%.3f mask_ms_max=%.3f\n",
               token_pieces_.size(), ctor_ms_, static_cast<unsigned long long>(mask_calls_),
               mask_ms_, mask_calls_ ? mask_ms_ / static_cast<double>(mask_calls_) : 0.0,
               mask_ms_max_);
}

bool GrammarSampler::token_allowed(int start, bool terminable, bool partial_pending,
                                   std::size_t t) const {
  if (static_cast<int>(t) == eos_token_id_) {
    // EOS is permitted only at a complete value.
    return terminable;
  }
  // Tokens beyond the piece table, or special tokens with no emitted bytes, are
  // not part of the grammar's surface.
  if (t >= token_pieces_.size() || token_pieces_[t].empty()) {
    return false;
  }
  // Fast path: simple (whole-codepoint) token and no pending partial UTF-8; walk
  // its codepoints through the memoized transitions.
  if (!partial_pending && tables_->simple[t] && !tables_->cps[t].empty()) {
    int s = start;
    for (const std::uint32_t cp : tables_->cps[t]) {
      if (s < 0) {
        break;
      }
      s = cache_->memo.transition(s, cp);
    }
    return s >= 0;
  }
  // Slow path: tokens that span a partial UTF-8 boundary or invalid bytes.
  return state_.would_accept(token_pieces_[t]);
}

void GrammarSampler::build_bitmask(int start, bool terminable, std::size_t vocab,
                                   std::vector<std::uint32_t>& out) const {
  out.assign((vocab + 31) / 32, 0u);
  for (std::size_t t = 0; t < vocab; ++t) {
    if (token_allowed(start, terminable, /*partial_pending=*/false, t)) {
      out[t >> 5] |= (1u << (t & 31));
    }
  }
}

void GrammarSampler::apply_mask(std::vector<float>& logits) const {
  const auto mask_t0 =
      profile_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const bool terminable = state_.can_terminate();
  const bool partial_pending = state_.has_partial();
  const std::size_t vocab = logits.size();

  // Cached path. A pending partial UTF-8 sequence is excluded: the legal set then
  // depends on the carried bytes as well as the stack-set, which the state id does
  // not capture. It is rare (mid-codepoint only) and falls through to the walk.
  if (!partial_pending && mask_cache_enabled_) {
    if (cache_->vocab == 0) cache_->vocab = vocab;
    if (cache_->vocab == vocab) {
      const int start = cache_->memo.intern(state_.current_stacks());
      const std::uint64_t key =
          (static_cast<std::uint64_t>(start) << 1) | (terminable ? 1ull : 0ull);
      auto it = cache_->masks.find(key);
      if (it == cache_->masks.end() && cache_->masks.size() < kMaxCachedMasks) {
        std::vector<std::uint32_t> bits;
        build_bitmask(start, terminable, vocab, bits);
        it = cache_->masks.emplace(key, std::move(bits)).first;
      }
      if (it != cache_->masks.end()) {
        const std::vector<std::uint32_t>& bits = it->second;
        const std::size_t words = bits.size();
        for (std::size_t w = 0; w < words; ++w) {
          const std::uint32_t word = bits[w];
          if (word == 0xFFFFFFFFu) continue;  // every token in this word is legal
          const std::size_t base = w * 32;
          const std::size_t upto = std::min(base + 32, vocab);
          if (word == 0u) {
            for (std::size_t t = base; t < upto; ++t) logits[t] = neg_inf;
            continue;
          }
          for (std::size_t t = base; t < upto; ++t) {
            if (((word >> (t - base)) & 1u) == 0u) logits[t] = neg_inf;
          }
        }
        if (profile_) {
          const double ms =
              std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mask_t0)
                  .count();
          ++mask_calls_;
          mask_ms_ += ms;
          if (ms > mask_ms_max_) mask_ms_max_ = ms;
        }
        return;
      }
    }
  }

  // Uncached walk: partial UTF-8 pending, or the cache is full.
  const int start = cache_->memo.intern(state_.current_stacks());
  for (std::size_t t = 0; t < vocab; ++t) {
    if (!token_allowed(start, terminable, partial_pending, t)) logits[t] = neg_inf;
  }
  if (profile_) {
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - mask_t0)
            .count();
    ++mask_calls_;
    mask_ms_ += ms;
    if (ms > mask_ms_max_) mask_ms_max_ = ms;
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
