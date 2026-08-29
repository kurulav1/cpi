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

  // Whether `cp` is acceptable from state `id`, without working out which state
  // it leads to.
  //
  // That distinction is most of the cost of a mask. transition() has to walk the
  // grammar, canonicalize the resulting stack set, build its key and intern it;
  // this only tests the elements on top of each stack. Since a mask rejects the
  // large majority of the codepoints it tries, testing first and computing the
  // successor only for survivors avoids nearly all of that work.
  bool accepts(int id, std::uint32_t cp) const {
    for (const Stack& stack : states_[static_cast<std::size_t>(id)]) {
      if (!stack.empty() && char_set_accepts(stack.back(), cp)) {
        return true;
      }
    }
    return false;
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
  // Reused when a state cannot be cached, so the bucketed build still runs
  // instead of falling back to the whole-vocabulary walk.
  std::vector<std::uint32_t> scratch;
  std::size_t mask_bytes = 0;  // what `masks` holds, against kMaxCachedMaskBytes
  std::size_t vocab = 0;
};

namespace {

// Bound the cache by memory rather than by a mask count, because the two differ
// by 16x across the vocabularies in use: a mask is 16 KB at Llama-3's 128256
// tokens and 32 KB at Gemma's 262144.
//
// 64 masks was far too few. A tool-calling request made 154 mask calls and 150 of
// them had to rebuild, because once the cache filled nothing could enter it, so
// states that recurred paid full price every time. 16 MB holds a thousand masks
// at the smaller vocabulary, which covers a schema's live states with room spare,
// and it is per request, so it is released when the request ends.
constexpr std::size_t kMaxCachedMaskBytes = 16u * 1024u * 1024u;

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

  // Group the simple tokens by first codepoint. A mask can then reject a whole
  // bucket with one DFA transition rather than testing each token in it, which is
  // most of the work: a 128k vocabulary has only a few thousand distinct first
  // codepoints, and a state inside a string literal admits one of them.
  std::unordered_map<std::uint32_t, std::size_t> slot;
  for (std::size_t t = 0; t < vocab; ++t) {
    if (token_pieces[t].empty()) {
      continue;
    }
    if (!tables->simple[t] || tables->cps[t].empty()) {
      tables->non_simple.push_back(static_cast<int>(t));
      continue;
    }
    const std::uint32_t head = tables->cps[t][0];
    auto it = slot.find(head);
    if (it == slot.end()) {
      it = slot.emplace(head, tables->first_cps.size()).first;
      tables->first_cps.push_back(head);
      tables->buckets.emplace_back();
    }
    tables->buckets[it->second].push_back(static_cast<int>(t));
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
               "mask_ms_avg=%.3f mask_ms_max=%.3f builds=%llu slow_ms=%.2f\n",
               token_pieces_.size(), ctor_ms_, static_cast<unsigned long long>(mask_calls_),
               mask_ms_, mask_calls_ ? mask_ms_ / static_cast<double>(mask_calls_) : 0.0,
               mask_ms_max_, static_cast<unsigned long long>(builds_), slow_ms_);
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
  const auto allow = [&out, vocab](std::size_t t) {
    if (t < vocab) {
      out[t >> 5] |= (1u << (t & 31));
    }
  };

  // EOS is legal exactly where the grammar may stop.
  if (eos_token_id_ >= 0) {
    if (terminable) {
      allow(static_cast<std::size_t>(eos_token_id_));
    }
  }

  // One transition per distinct first codepoint decides a whole bucket. This is
  // the difference between testing 128256 tokens and testing the few thousand
  // that could possibly continue: in a restrictive state nearly every bucket dies
  // on that single lookup, and the tokens inside it are never touched.
  for (std::size_t b = 0; b < tables_->first_cps.size(); ++b) {
    // Cheap test first. Most buckets are rejected, and asking transition()
    // directly would build and intern a successor state for each one before
    // discovering it was dead.
    if (!cache_->memo.accepts(start, tables_->first_cps[b])) {
      continue;
    }
    const int after_first = cache_->memo.transition(start, tables_->first_cps[b]);
    if (after_first < 0) {
      continue;  // no token starting with this codepoint can continue
    }
    for (const int id : tables_->buckets[b]) {
      const std::vector<std::uint32_t>& cps = tables_->cps[static_cast<std::size_t>(id)];
      int s = after_first;
      for (std::size_t k = 1; k < cps.size() && s >= 0; ++k) {
        s = cache_->memo.transition(s, cps[k]);
      }
      if (s >= 0) {
        allow(static_cast<std::size_t>(id));
      }
    }
  }

  // Tokens that are not whole codepoints (mid-sequence or invalid bytes) cannot
  // be bucketed by a first codepoint, so they keep the byte-level check. There
  // are a thousand or so of them against a 128k vocabulary.
  const auto slow_t0 =
      profile_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  for (const int id : tables_->non_simple) {
    if (static_cast<std::size_t>(id) < token_pieces_.size() &&
        state_.would_accept(token_pieces_[static_cast<std::size_t>(id)])) {
      allow(static_cast<std::size_t>(id));
    }
  }
  if (profile_) {
    slow_ms_ +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - slow_t0)
            .count();
    ++builds_;
  }
}

void GrammarSampler::apply_mask(std::vector<float>& logits) const {
  apply_mask(logits.data(), logits.size());
}

void GrammarSampler::apply_mask(float* logits, std::size_t vocab_n) const {
  const auto mask_t0 =
      profile_ ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
  const float neg_inf = -std::numeric_limits<float>::infinity();
  const bool terminable = state_.can_terminate();
  const bool partial_pending = state_.has_partial();
  const std::size_t vocab = vocab_n;

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
      const std::size_t mask_bytes = ((vocab + 31) / 32) * sizeof(std::uint32_t);
      if (it == cache_->masks.end() && cache_->mask_bytes + mask_bytes <= kMaxCachedMaskBytes) {
        std::vector<std::uint32_t> bits;
        build_bitmask(start, terminable, vocab, bits);
        cache_->mask_bytes += mask_bytes;
        it = cache_->masks.emplace(key, std::move(bits)).first;
      }
      // Past the cache limit, build into scratch and apply it without storing.
      //
      // This used to fall through to the whole-vocabulary walk, which is the
      // slowest path there is, so exceeding the cap did not merely stop saving
      // work: it switched algorithms. Measured on a tool-calling request, 165
      // mask calls produced 64 builds, the cap exactly, and the remaining calls
      // took the walk. The bucketed build is the same result at a fraction of the
      // cost, so it is what runs whether or not the mask can be kept.
      const std::vector<std::uint32_t>* bits_ptr = nullptr;
      if (it != cache_->masks.end()) {
        bits_ptr = &it->second;
      } else {
        build_bitmask(start, terminable, vocab, cache_->scratch);
        bits_ptr = &cache_->scratch;
      }
      {
        const std::vector<std::uint32_t>& bits = *bits_ptr;
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
