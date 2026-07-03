// Unit tests for the grammar module (GBNF engine, JSON-Schema compiler,
// GrammarSampler). Pure C++ / CUDA-free; runs under ctest without a GPU.
//
// Exit code 0 = all passed, 1 = a failure.

#include "grammar/grammar.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "grammar/grammar_sampler.hpp"
#include "grammar/json_schema_to_grammar.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool cond, const std::string& msg) {
  ++g_checks;
  if (!cond) {
    ++g_failures;
    std::cerr << "FAIL: " << msg << "\n";
  }
}

// Feeds `text` into a fresh state of `grammar`, one byte at a time, to exercise
// token-boundary handling. Returns true if every byte was accepted.
bool feed_bytes(const grammar::Grammar& g, const std::string& text) {
  grammar::GrammarState st(g);
  for (char c : text) {
    if (!st.accept_bytes(std::string(1, c))) {
      return false;
    }
  }
  return true;
}

// Feeds `text` then reports whether the grammar can legally terminate.
bool feed_and_complete(const grammar::Grammar& g, const std::string& text) {
  grammar::GrammarState st(g);
  for (char c : text) {
    if (!st.accept_bytes(std::string(1, c))) {
      return false;
    }
  }
  return st.can_terminate();
}

void test_handwritten_gbnf() {
  // A minimal object grammar written by hand.
  const std::string gbnf =
      "root ::= \"{\" ws \"\\\"n\\\"\" ws \":\" ws number ws \"}\"\n"
      "ws ::= [ \\t\\n]*\n"
      "number ::= \"-\"? [0-9]+\n";
  grammar::Grammar g = grammar::Grammar::parse(gbnf);

  check(feed_and_complete(g, "{\"n\":42}"), "handwritten: accepts {\"n\":42}");
  check(feed_and_complete(g, "{ \"n\" : -7 }"), "handwritten: accepts spaced/negative");
  check(!feed_bytes(g, "{\"n\":x}"), "handwritten: rejects non-number value");
  check(!feed_bytes(g, "{\"x\":1}"), "handwritten: rejects wrong key");
  // Incomplete input parses so far but must not be able to terminate.
  grammar::GrammarState st(g);
  for (char c : std::string("{\"n\":4")) {
    st.accept_bytes(std::string(1, c));
  }
  check(!st.can_terminate(), "handwritten: incomplete object cannot terminate");
}

void test_schema_object() {
  const std::string schema = R"({
    "type": "object",
    "properties": {
      "title":  {"type": "string"},
      "count":  {"type": "integer"},
      "tags":   {"type": "array", "items": {"type": "string"}},
      "active": {"type": "boolean"}
    },
    "required": ["title", "count"]
  })";
  const std::string gbnf = grammar::json_schema_to_grammar(schema);
  check(gbnf.find("root ::=") != std::string::npos, "schema: emits a root rule");
  check(gbnf.find("integer ::=") != std::string::npos, "schema: emits integer primitive");

  grammar::Grammar g = grammar::Grammar::parse(gbnf);

  check(feed_and_complete(g, R"({"title":"hi","count":3})"),
        "schema: required-only object accepts");
  check(feed_and_complete(g, R"({"title":"hi","count":3,"tags":["a","b"],"active":true})"),
        "schema: full object accepts");
  check(feed_and_complete(g, R"({"title":"hi","count":3,"active":false})"),
        "schema: skips an optional, keeps a later one");
  check(!feed_bytes(g, R"({"count":3,"title":"hi"})"),
        "schema: rejects required props out of order (documented v1 strictness)");
  check(!feed_bytes(g, R"({"title":"hi","count":"x"})"),
        "schema: rejects wrong-typed value (string where integer required)");
  check(!feed_bytes(g, R"({"title":"hi"})"), "schema: rejects missing required prop (no count)");
}

void test_schema_enum() {
  const std::string schema = R"({"enum": ["red", "green", "blue"]})";
  grammar::Grammar g = grammar::Grammar::parse(grammar::json_schema_to_grammar(schema));
  check(feed_and_complete(g, "\"green\""), "enum: accepts a member");
  check(!feed_bytes(g, "\"purple\""), "enum: rejects a non-member");
}

void test_partial_utf8() {
  // A bare string schema; feed a 2-byte UTF-8 codepoint (U+00E9 'é' = C3 A9)
  // split across two accept_bytes calls to exercise partial-UTF-8 carry.
  grammar::Grammar g =
      grammar::Grammar::parse(grammar::json_schema_to_grammar(R"({"type":"string"})"));
  grammar::GrammarState st(g);
  check(st.accept_bytes("\""), "utf8: opening quote");
  check(st.accept_bytes(std::string(1, static_cast<char>(0xC3))), "utf8: lead byte alive");
  check(!st.can_terminate(), "utf8: cannot terminate mid-codepoint");
  check(st.accept_bytes(std::string(1, static_cast<char>(0xA9))), "utf8: continuation completes");
  check(st.accept_bytes("\""), "utf8: closing quote");
  check(st.can_terminate(), "utf8: complete string can terminate");
}

void test_grammar_sampler_mask() {
  grammar::Grammar g =
      grammar::Grammar::parse(grammar::json_schema_to_grammar(R"({"type":"integer"})"));

  // Tiny synthetic vocab. id 0 is EOS (empty piece).
  const std::vector<std::string> pieces = {"", "-", "0", "1", "9", "a", "{", " "};
  const int eos_id = 0;
  grammar::GrammarSampler sampler(std::move(g), pieces, eos_id);

  auto masked = [&]() {
    std::vector<float> logits(pieces.size(), 0.0f);
    sampler.apply_mask(logits);
    std::vector<bool> allowed(pieces.size());
    for (std::size_t i = 0; i < logits.size(); ++i) {
      allowed[i] = std::isfinite(logits[i]);
    }
    return allowed;
  };

  std::vector<bool> a0 = masked();
  check(!a0[0], "sampler: EOS masked before any digit (cannot terminate)");
  check(a0[1], "sampler: '-' allowed at start");
  check(a0[3], "sampler: digit '1' allowed at start");
  check(a0[7], "sampler: leading whitespace allowed");
  check(!a0[5], "sampler: letter 'a' masked");
  check(!a0[6], "sampler: '{' masked for integer schema");

  check(sampler.accept(3), "sampler: integer is satisfiable after one digit");
  // After one digit the integer is satisfiable -> EOS now allowed, and the
  // grammar still permits more digits (it does not force a stop).
  std::vector<bool> a1 = masked();
  check(a1[0], "sampler: EOS allowed after a complete integer");
  check(a1[4], "sampler: more digits still allowed");
  check(!a1[5], "sampler: letter still masked after digit");
}

// The optimized apply_mask (first-codepoint rejection + precomputed codepoints)
// must produce exactly the same mask as a brute-force reference that runs
// would_accept on every token. Drive a valid JSON token sequence and compare the
// two masks at every step, including mid-structure states.
void test_mask_fastpath_parity() {
  grammar::Grammar g = grammar::Grammar::parse(grammar::json_schema_to_grammar(R"({
    "type": "object",
    "properties": { "title": {"type":"string"}, "count": {"type":"integer"} },
    "required": ["title", "count"]
  })"));

  // Synthetic vocab: id 0 is EOS; includes single-char structural tokens,
  // multi-char key/value tokens, a filler that should usually be rejected.
  const std::vector<std::string> pieces = {
      "",       // 0 EOS
      "{",      // 1
      "\"",     // 2
      "title",  // 3
      "count",  // 4
      ":",      // 5
      ",",      // 6
      "hi",     // 7
      "3",      // 8
      "}",      // 9
      " ",      // 10
      "zzz",    // 11 filler
      "\":\""   // 12 multi-char
  };
  const int eos_id = 0;

  // Reference mask: would_accept per token, EOS gated on can_terminate.
  auto reference_mask = [&](const grammar::GrammarState& st) {
    std::vector<bool> allowed(pieces.size());
    const bool terminable = st.can_terminate();
    for (std::size_t t = 0; t < pieces.size(); ++t) {
      if (static_cast<int>(t) == eos_id) {
        allowed[t] = terminable;
      } else if (pieces[t].empty()) {
        allowed[t] = false;
      } else {
        allowed[t] = st.would_accept(pieces[t]);
      }
    }
    return allowed;
  };

  grammar::GrammarSampler sampler(std::move(g), pieces, eos_id);
  grammar::Grammar g_ref = grammar::Grammar::parse(grammar::json_schema_to_grammar(R"({
        "type": "object",
        "properties": { "title": {"type":"string"}, "count": {"type":"integer"} },
        "required": ["title", "count"]
      })"));
  grammar::GrammarState ref(g_ref);

  // Valid token path for {"title":"hi","count":3}.
  const std::vector<int> path = {1, 2, 3, 2, 5, 2, 7, 2, 6, 2, 4, 2, 5, 8, 9};

  bool all_match = true;
  for (std::size_t step = 0; step <= path.size(); ++step) {
    std::vector<float> logits(pieces.size(), 0.0f);
    sampler.apply_mask(logits);
    const std::vector<bool> ref_allowed = reference_mask(ref);
    for (std::size_t t = 0; t < pieces.size(); ++t) {
      const bool opt = std::isfinite(logits[t]);
      if (opt != ref_allowed[t]) {
        all_match = false;
      }
    }
    if (step < path.size()) {
      const int tok = path[step];
      // The chosen token must be allowed by the (optimized) mask at this step.
      check(std::isfinite(logits[static_cast<std::size_t>(tok)]),
            "fastpath parity: path token allowed at step " + std::to_string(step));
      sampler.accept(tok);
      ref.accept_bytes(pieces[static_cast<std::size_t>(tok)]);
    }
  }
  check(all_match, "fastpath parity: optimized mask == brute-force reference at every step");
  check(ref.can_terminate(), "fastpath parity: sequence reaches a complete value");
}

}  // namespace

int main() {
  test_handwritten_gbnf();
  test_schema_object();
  test_schema_enum();
  test_partial_utf8();
  test_grammar_sampler_mask();
  test_mask_fastpath_parity();

  std::cout << "grammar_test: " << (g_checks - g_failures) << "/" << g_checks << " checks passed\n";
  if (g_failures != 0) {
    std::cerr << "grammar_test: " << g_failures << " failure(s)\n";
    return 1;
  }
  return 0;
}
