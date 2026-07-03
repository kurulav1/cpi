#pragma once

// GBNF grammar engine: a self-contained, CUDA-free port of the grammar core
// used by llama.cpp (grammar-parser + llama_grammar). It compiles a GBNF
// grammar string into rules, then drives a stack-based nondeterministic matcher
// over UTF-8 bytes. Used to constrain decoding to schema-valid JSON.
//
// Two pieces:
//   - Grammar:      immutable compiled rules + root rule id (parse() from GBNF).
//   - GrammarState: mutable parse position (a set of stacks + partial-UTF-8
//                   carry) that advances as tokens are accepted and can report
//                   whether a candidate byte string keeps the grammar alive.
//
// Everything operates on raw bytes; callers map token ids to their emitted
// bytes (see GrammarSampler).

#include <cstdint>
#include <string>
#include <vector>

namespace grammar {

// Element of a grammar rule. A rule is a flat vector of Elements: alternates are
// separated by Alt and the rule ends with End. A character set is a Char (or
// CharNot for a negated set) optionally followed by a CharRngUpper (range upper
// bound) and any number of CharAlt (further chars/ranges in the same set).
enum class ElemType : std::uint8_t {
  End = 0,       // end of rule (terminates the final alternate)
  Alt,           // alternation separator '|'
  RuleRef,       // reference to another rule; value = rule id
  Char,          // literal codepoint, or start of a positive char set; value = codepoint
  CharNot,       // start of a negated char set [^...]; value = codepoint
  CharRngUpper,  // upper bound of a range whose lower bound is the preceding Char/CharAlt
  CharAlt,       // a further char/range alternative within the current set
};

struct Element {
  ElemType type;
  std::uint32_t value;  // codepoint for Char*/range, rule id for RuleRef
};

using Rule = std::vector<Element>;

// Carry state for a UTF-8 sequence split across byte chunks (token boundaries).
// n_remain < 0 means "no partial sequence in progress".
struct PartialUtf8 {
  std::uint32_t value = 0;
  int n_remain = -1;
};

// Immutable compiled grammar.
class Grammar {
public:
  Grammar() = default;
  Grammar(std::vector<Rule> rules, std::size_t root_rule_id)
      : rules_(std::move(rules)), root_rule_(root_rule_id) {}

  // Compiles a GBNF grammar string. `root_rule_name` selects the entry rule.
  // Throws std::runtime_error on a parse error or a missing root rule.
  static Grammar parse(const std::string& gbnf, const std::string& root_rule_name = "root");

  const std::vector<Rule>& rules() const {
    return rules_;
  }
  std::size_t root_rule_id() const {
    return root_rule_;
  }
  bool empty() const {
    return rules_.empty();
  }

private:
  std::vector<Rule> rules_;
  std::size_t root_rule_ = 0;
};

// A single stack: pointers into the grammar's rule element arrays. back() is the
// next element to process. An empty stack denotes an accepting (complete) state.
using Stack = std::vector<const Element*>;

// True if the character set beginning at `set` (a Char or CharNot element, as
// produced by the parser) accepts codepoint `cp`. Used for cheap first-codepoint
// rejection during token masking without walking the grammar.
bool char_set_accepts(const Element* set, std::uint32_t cp);

// Decodes `bytes` into codepoints assuming no pending partial sequence. Returns
// true and fills `out` when the bytes form a clean, complete UTF-8 sequence
// (a "simple" token, eligible for the fast masking path); returns false for
// bytes that end mid-codepoint or are invalid (the caller uses the slow path).
bool decode_simple_codepoints(const std::string& bytes, std::vector<std::uint32_t>& out);

// Advances a stack-set by one codepoint, returning the surviving stacks (an
// empty result means the codepoint is rejected). Exposed so token masking can
// memoize transitions per grammar state instead of re-walking the grammar for
// every token in the vocabulary.
std::vector<Stack> grammar_accept_codepoint(const Grammar& grammar,
                                            const std::vector<Stack>& stacks, std::uint32_t cp);

// Mutable matcher state. Holds pointers into the Grammar it was constructed
// from, so that Grammar must outlive the state and must not be moved/reallocated
// while the state is in use.
class GrammarState {
public:
  explicit GrammarState(const Grammar& grammar);

  // Advances the state by every byte in `piece`. Returns false if the bytes
  // cannot continue the grammar (the state is then dead / empty()).
  bool accept_bytes(const std::string& piece);

  // True when the grammar has matched a complete value here (an accepting stack
  // is present and no partial UTF-8 sequence is pending) — a legal stop point.
  bool can_terminate() const;

  // Would accepting `piece` keep the grammar alive? Does not mutate the state.
  bool would_accept(const std::string& piece) const;

  // Fast-path variant of would_accept for a precomputed codepoint sequence,
  // valid only when no partial UTF-8 sequence is pending (see has_partial()).
  // Skips byte decoding; behaviour matches would_accept for simple tokens.
  bool would_accept_codepoints(const std::vector<std::uint32_t>& cps) const;

  // Appends the current next-character set of every live (non-empty) stack to
  // `out`. Each entry is a Char/CharNot element usable with char_set_accepts to
  // test whether a codepoint could be consumed next.
  void collect_stack_tops(std::vector<const Element*>& out) const;

  // True when a partial UTF-8 sequence is pending from a prior accepted token
  // (the fast codepoint path is invalid in that case).
  bool has_partial() const {
    return partial_utf8_.n_remain > 0;
  }

  // The current live stack-set (starting point for memoized token masking).
  const std::vector<Stack>& current_stacks() const {
    return stacks_;
  }

  // True when no stacks remain (grammar is dead and cannot continue).
  bool empty() const {
    return stacks_.empty();
  }

private:
  const Grammar* grammar_ = nullptr;
  std::vector<Stack> stacks_;
  PartialUtf8 partial_utf8_{};
};

}  // namespace grammar
