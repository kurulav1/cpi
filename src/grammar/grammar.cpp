#include "grammar/grammar.hpp"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <utility>

// GBNF grammar engine. The parser and the stack matcher follow the design used
// by llama.cpp (grammar-parser.cpp / llama-grammar.cpp), reimplemented here as a
// standalone, dependency-free component.

namespace grammar {
namespace {

// ----------------------------------------------------------------------------
// UTF-8 decoding
// ----------------------------------------------------------------------------

// Decodes the bytes in `src`, continuing any partial sequence described by
// `partial_start`. Returns the completed codepoints and the trailing partial
// state (n_remain > 0 if `src` ended mid-codepoint). On an invalid encoding the
// codepoint list is {0} and the returned partial has n_remain == -1, which the
// caller treats as a hard reject.
std::pair<std::vector<std::uint32_t>, PartialUtf8> decode_utf8(const std::string& src,
                                                               PartialUtf8 partial_start) {
  // Number of bytes in a sequence keyed by the top 4 bits of the lead byte.
  // 0 entries mark continuation bytes (invalid as a lead byte).
  static const int lookup[16] = {1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 2, 2, 3, 4};

  // A scalar must arrive in its shortest form. Without this, an overlong sequence
  // decodes to a legal codepoint and satisfies the grammar, while the bytes that
  // actually reach the output are invalid UTF-8. That is how a JSON enum limited
  // to "add" produced "aÁ¤d": 0xC1 0xA4 is an overlong 'd', accepted as
  // U+0064 by the matcher and written to the wire as two bytes no parser accepts.
  // Surrogates and anything past U+10FFFF are not scalars either.
  const auto valid_scalar = [](std::uint32_t v, int len) {
    if (v > 0x10FFFFu) return false;
    if (v >= 0xD800u && v <= 0xDFFFu) return false;
    if (len == 2) return v >= 0x80u;
    if (len == 3) return v >= 0x800u;
    if (len == 4) return v >= 0x10000u;
    return true;
  };

  std::vector<std::uint32_t> code_points;
  code_points.reserve(src.size() + 1);

  std::uint32_t value = partial_start.value;
  int n_remain = partial_start.n_remain;
  std::size_t i = 0;

  // Finish a sequence carried over from a previous chunk.
  while (i < src.size() && n_remain > 0) {
    const std::uint8_t next_byte = static_cast<std::uint8_t>(src[i]);
    if ((next_byte >> 6) != 2) {  // not a continuation byte
      return {{0}, PartialUtf8{0, -1, 0}};
    }
    value = (value << 6) + (next_byte & 0x3F);
    ++i;
    --n_remain;
  }
  if (partial_start.n_remain > 0 && n_remain == 0) {
    if (!valid_scalar(value, partial_start.n_total)) {
      return {{0}, PartialUtf8{0, -1, 0}};
    }
    code_points.push_back(value);
  }

  // Decode whole sequences; the final one may be incomplete.
  while (i < src.size()) {
    const std::uint8_t first_byte = static_cast<std::uint8_t>(src[i]);
    const std::uint8_t highbits = first_byte >> 4;
    n_remain = lookup[highbits] - 1;
    if (n_remain < 0) {  // invalid lead byte
      return {{0}, PartialUtf8{0, -1, 0}};
    }
    // 0xC0 and 0xC1 can only ever begin an overlong two-byte form, and anything
    // above 0xF4 exceeds U+10FFFF. Neither can start a valid sequence.
    if (first_byte == 0xC0u || first_byte == 0xC1u || first_byte > 0xF4u) {
      return {{0}, PartialUtf8{0, -1, 0}};
    }
    const int seq_len = lookup[highbits];
    const std::uint8_t mask = static_cast<std::uint8_t>((1 << (7 - n_remain)) - 1);
    value = first_byte & mask;
    ++i;
    while (i < src.size() && n_remain > 0) {
      const std::uint8_t next_byte = static_cast<std::uint8_t>(src[i]);
      if ((next_byte >> 6) != 2) {
        return {{0}, PartialUtf8{0, -1, 0}};
      }
      value = (value << 6) + (next_byte & 0x3F);
      ++i;
      --n_remain;
    }
    if (n_remain == 0) {
      if (!valid_scalar(value, seq_len)) {
        return {{0}, PartialUtf8{0, -1, 0}};
      }
      code_points.push_back(value);
    } else {
      return {code_points, PartialUtf8{value, n_remain, seq_len}};
    }
  }

  return {code_points, PartialUtf8{value, n_remain, 0}};
}

// ----------------------------------------------------------------------------
// Matcher helpers
// ----------------------------------------------------------------------------

bool is_end_of_sequence(const Element* pos) {
  return pos->type == ElemType::End || pos->type == ElemType::Alt;
}

// Matches a codepoint against the char set beginning at `pos` (a Char or
// CharNot, with optional CharRngUpper / CharAlt continuation). Returns whether
// the codepoint is accepted by the set, and the position just past the set.
std::pair<bool, const Element*> match_char(const Element* pos, std::uint32_t chr) {
  const bool is_positive = pos->type == ElemType::Char;  // else CharNot
  bool found = false;
  do {
    if (pos[1].type == ElemType::CharRngUpper) {
      found = found || (pos->value <= chr && chr <= pos[1].value);
      pos += 2;
    } else {
      found = found || pos->value == chr;
      pos += 1;
    }
  } while (pos->type == ElemType::CharAlt);
  return {found == is_positive, pos};
}

// Whether a still-incomplete UTF-8 sequence could possibly match the char set at
// `pos`. Used when a candidate token ends mid-codepoint: we keep it alive only
// if some completion of the partial sequence is acceptable.
bool match_partial_char(const Element* pos, const PartialUtf8& partial) {
  const int n_remain = partial.n_remain;
  // Range of codepoints the partial value could complete to.
  std::uint32_t low = partial.value << (n_remain * 6);
  std::uint32_t high = low | ((1u << (n_remain * 6)) - 1);
  if (low == 0) {
    if (n_remain == 2) {
      low = 1u << 11;
    } else if (n_remain == 3) {
      low = 1u << 16;
    }
  }

  const bool is_positive = pos->type == ElemType::Char;
  do {
    if (pos[1].type == ElemType::CharRngUpper) {
      if (pos->value <= high && low <= pos[1].value) {
        return is_positive;
      }
      pos += 2;
    } else {
      if (low <= pos->value && pos->value <= high) {
        return is_positive;
      }
      pos += 1;
    }
  } while (pos->type == ElemType::CharAlt);
  return !is_positive;
}

void stacks_push_unique(std::vector<Stack>& stacks, Stack stack) {
  if (std::find(stacks.begin(), stacks.end(), stack) == stacks.end()) {
    stacks.push_back(std::move(stack));
  }
}

// Expands a stack until its top element is a terminal (Char/CharNot) or the
// stack is empty, pushing every resulting stack into `out`. RuleRef tops are
// replaced by each alternate of the referenced rule.
void advance_stack(const std::vector<Rule>& rules, const Stack& stack, std::vector<Stack>& out) {
  if (stack.empty()) {
    stacks_push_unique(out, stack);
    return;
  }

  const Element* pos = stack.back();
  switch (pos->type) {
    case ElemType::RuleRef: {
      const std::size_t rule_id = static_cast<std::size_t>(pos->value);
      const Element* subpos = rules[rule_id].data();
      do {
        // The stack without its RuleRef top.
        Stack new_stack(stack.begin(), stack.end() - 1);
        // Continuation after the rule reference, if any.
        if (!is_end_of_sequence(pos + 1)) {
          new_stack.push_back(pos + 1);
        }
        // Body of this alternate, if non-empty.
        if (!is_end_of_sequence(subpos)) {
          new_stack.push_back(subpos);
        }
        advance_stack(rules, new_stack, out);
        // Skip to the next alternate of the referenced rule.
        while (!is_end_of_sequence(subpos)) {
          ++subpos;
        }
        if (subpos->type == ElemType::Alt) {
          ++subpos;  // move past the '|' into the next alternate
        } else {
          break;
        }
      } while (true);
      break;
    }
    case ElemType::Char:
    case ElemType::CharNot:
      stacks_push_unique(out, stack);
      break;
    default:
      // End/Alt/CharRngUpper/CharAlt should never be a stack top.
      throw std::runtime_error("grammar: malformed stack top");
  }
}

// Builds the set of initial stacks from the alternates of the root rule.
std::vector<Stack> init_stacks(const Grammar& grammar) {
  std::vector<Stack> stacks;
  if (grammar.rules().empty()) {
    return stacks;
  }
  const Element* pos = grammar.rules()[grammar.root_rule_id()].data();
  do {
    Stack stack;
    if (!is_end_of_sequence(pos)) {
      stack.push_back(pos);
    }
    advance_stack(grammar.rules(), stack, stacks);
    while (!is_end_of_sequence(pos)) {
      ++pos;
    }
    if (pos->type == ElemType::Alt) {
      ++pos;
    } else {
      break;
    }
  } while (true);
  return stacks;
}

// Advances every stack by one codepoint, returning the surviving stacks.
std::vector<Stack> accept_codepoint(const std::vector<Rule>& rules,
                                    const std::vector<Stack>& stacks, std::uint32_t chr) {
  std::vector<Stack> next;
  for (const Stack& stack : stacks) {
    if (stack.empty()) {
      continue;
    }
    const auto [matched, pos] = match_char(stack.back(), chr);
    if (matched) {
      Stack new_stack(stack.begin(), stack.end() - 1);
      if (!is_end_of_sequence(pos)) {
        new_stack.push_back(pos);
      }
      advance_stack(rules, new_stack, next);
    }
  }
  return next;
}

bool any_stack_matches_partial(const std::vector<Stack>& stacks, const PartialUtf8& partial) {
  for (const Stack& stack : stacks) {
    if (!stack.empty() && match_partial_char(stack.back(), partial)) {
      return true;
    }
  }
  return false;
}

// ----------------------------------------------------------------------------
// GBNF parser
// ----------------------------------------------------------------------------

struct ParseState {
  std::map<std::string, std::uint32_t> symbol_ids;
  std::vector<Rule> rules;
};

bool is_word_char(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' ||
         c == '_';
}

std::uint32_t get_symbol_id(ParseState& state, const std::string& name) {
  const std::uint32_t next_id = static_cast<std::uint32_t>(state.symbol_ids.size());
  auto [it, inserted] = state.symbol_ids.emplace(name, next_id);
  if (inserted) {
    state.rules.emplace_back();  // reserve a slot; filled in by add_rule
  }
  return it->second;
}

std::uint32_t generate_symbol_id(ParseState& state, const std::string& base_name) {
  const std::uint32_t id = static_cast<std::uint32_t>(state.symbol_ids.size());
  state.symbol_ids[base_name + "_" + std::to_string(id)] = id;
  state.rules.emplace_back();
  return id;
}

void add_rule(ParseState& state, std::uint32_t rule_id, Rule rule) {
  if (state.rules.size() <= rule_id) {
    state.rules.resize(rule_id + 1);
  }
  state.rules[rule_id] = std::move(rule);
}

const char* parse_space(const char* src, bool newline_ok) {
  while (*src == ' ' || *src == '\t' || *src == '#' ||
         (newline_ok && (*src == '\r' || *src == '\n'))) {
    if (*src == '#') {
      while (*src != 0 && *src != '\r' && *src != '\n') {
        ++src;
      }
    } else {
      ++src;
    }
  }
  return src;
}

const char* parse_name(const char* src, std::string& out) {
  const char* pos = src;
  while (is_word_char(*pos)) {
    ++pos;
  }
  if (pos == src) {
    throw std::runtime_error(std::string("grammar: expecting name at ") + src);
  }
  out.assign(src, pos);
  return pos;
}

std::uint32_t parse_hex(const char*& pos, int n) {
  std::uint32_t value = 0;
  for (int i = 0; i < n; ++i) {
    value <<= 4;
    const char c = *pos;
    if (c >= '0' && c <= '9') {
      value += static_cast<std::uint32_t>(c - '0');
    } else if (c >= 'a' && c <= 'f') {
      value += static_cast<std::uint32_t>(c - 'a' + 10);
    } else if (c >= 'A' && c <= 'F') {
      value += static_cast<std::uint32_t>(c - 'A' + 10);
    } else {
      throw std::runtime_error("grammar: expecting hex digit");
    }
    ++pos;
  }
  return value;
}

// Decodes one (possibly escaped, possibly multi-byte UTF-8) character literal,
// advancing `pos` past it and returning its codepoint.
std::uint32_t parse_char(const char*& pos) {
  if (*pos == '\\') {
    const char esc = pos[1];
    switch (esc) {
      case 'x':
        pos += 2;
        return parse_hex(pos, 2);
      case 'u':
        pos += 2;
        return parse_hex(pos, 4);
      case 'U':
        pos += 2;
        return parse_hex(pos, 8);
      case 't':
        pos += 2;
        return '\t';
      case 'r':
        pos += 2;
        return '\r';
      case 'n':
        pos += 2;
        return '\n';
      case '\\':
      case '"':
      case '[':
      case ']':
      case '\'': {
        const std::uint32_t v = static_cast<std::uint8_t>(esc);
        pos += 2;
        return v;
      }
      default:
        throw std::runtime_error(std::string("grammar: unknown escape \\") + esc);
    }
  }
  if (*pos == 0) {
    throw std::runtime_error("grammar: unexpected end of input");
  }
  // UTF-8 decode a single character.
  const std::uint8_t first = static_cast<std::uint8_t>(*pos);
  if (first < 0x80) {
    ++pos;
    return first;
  }
  int n_remain = 0;
  std::uint32_t value = 0;
  if ((first & 0xE0) == 0xC0) {
    n_remain = 1;
    value = first & 0x1F;
  } else if ((first & 0xF0) == 0xE0) {
    n_remain = 2;
    value = first & 0x0F;
  } else if ((first & 0xF8) == 0xF0) {
    n_remain = 3;
    value = first & 0x07;
  } else {
    throw std::runtime_error("grammar: invalid UTF-8 lead byte in literal");
  }
  ++pos;
  for (int i = 0; i < n_remain; ++i) {
    const std::uint8_t b = static_cast<std::uint8_t>(*pos);
    if ((b >> 6) != 2) {
      throw std::runtime_error("grammar: invalid UTF-8 continuation in literal");
    }
    value = (value << 6) + (b & 0x3F);
    ++pos;
  }
  return value;
}

const char* parse_sequence(ParseState& state, const char* src, const std::string& rule_name,
                           Rule& out, bool is_nested);

const char* parse_alternates(ParseState& state, const char* src, const std::string& rule_name,
                             std::uint32_t rule_id, bool is_nested) {
  Rule rule;
  const char* pos = parse_sequence(state, src, rule_name, rule, is_nested);
  while (*pos == '|') {
    rule.push_back({ElemType::Alt, 0});
    pos = parse_space(pos + 1, true);
    pos = parse_sequence(state, pos, rule_name, rule, is_nested);
  }
  rule.push_back({ElemType::End, 0});
  add_rule(state, rule_id, std::move(rule));
  return pos;
}

const char* parse_sequence(ParseState& state, const char* src, const std::string& rule_name,
                           Rule& out, bool is_nested) {
  const char* pos = src;
  std::size_t last_sym_start = out.size();

  while (*pos != 0) {
    if (*pos == '"') {  // literal string
      ++pos;
      last_sym_start = out.size();
      while (*pos != '"') {
        if (*pos == 0) {
          throw std::runtime_error("grammar: unterminated string literal");
        }
        const std::uint32_t cp = parse_char(pos);
        out.push_back({ElemType::Char, cp});
      }
      pos = parse_space(pos + 1, is_nested);
    } else if (*pos == '[') {  // char class
      ++pos;
      ElemType start_type = ElemType::Char;
      if (*pos == '^') {
        ++pos;
        start_type = ElemType::CharNot;
      }
      last_sym_start = out.size();
      bool first = true;
      while (*pos != ']') {
        if (*pos == 0) {
          throw std::runtime_error("grammar: unterminated character class");
        }
        const std::uint32_t cp = parse_char(pos);
        const ElemType t = first ? start_type : ElemType::CharAlt;
        out.push_back({t, cp});
        first = false;
        if (*pos == '-' && pos[1] != ']' && pos[1] != 0) {
          ++pos;
          const std::uint32_t hi = parse_char(pos);
          out.push_back({ElemType::CharRngUpper, hi});
        }
      }
      pos = parse_space(pos + 1, is_nested);
    } else if (is_word_char(*pos)) {  // rule reference
      std::string name;
      const char* name_end = parse_name(pos, name);
      pos = parse_space(name_end, is_nested);
      last_sym_start = out.size();
      out.push_back({ElemType::RuleRef, get_symbol_id(state, name)});
    } else if (*pos == '(') {  // grouping -> sub-rule
      pos = parse_space(pos + 1, true);
      const std::uint32_t sub_rule_id = generate_symbol_id(state, rule_name);
      pos = parse_alternates(state, pos, rule_name, sub_rule_id, true);
      last_sym_start = out.size();
      out.push_back({ElemType::RuleRef, sub_rule_id});
      if (*pos != ')') {
        throw std::runtime_error("grammar: expecting ')'");
      }
      pos = parse_space(pos + 1, is_nested);
    } else if (*pos == '*' || *pos == '+' || *pos == '?') {  // repetition
      if (last_sym_start == out.size()) {
        throw std::runtime_error("grammar: repetition operator without preceding symbol");
      }
      const char op = *pos;
      const std::uint32_t sub_rule_id = generate_symbol_id(state, rule_name);
      Rule sub_rule(out.begin() + static_cast<std::ptrdiff_t>(last_sym_start), out.end());
      if (op == '*' || op == '+') {
        sub_rule.push_back({ElemType::RuleRef, sub_rule_id});  // recurse
      }
      sub_rule.push_back({ElemType::Alt, 0});
      if (op == '+') {
        sub_rule.insert(sub_rule.end(), out.begin() + static_cast<std::ptrdiff_t>(last_sym_start),
                        out.end());
      }
      sub_rule.push_back({ElemType::End, 0});
      add_rule(state, sub_rule_id, std::move(sub_rule));
      out.resize(last_sym_start);
      out.push_back({ElemType::RuleRef, sub_rule_id});
      pos = parse_space(pos + 1, is_nested);
    } else {
      break;  // end of sequence ('|', ')', newline, or EOF)
    }
  }
  return pos;
}

const char* parse_rule(ParseState& state, const char* src) {
  std::string name;
  const char* pos = parse_name(src, name);
  pos = parse_space(pos, false);
  const std::uint32_t rule_id = get_symbol_id(state, name);

  if (!(pos[0] == ':' && pos[1] == ':' && pos[2] == '=')) {
    throw std::runtime_error(std::string("grammar: expecting '::=' after ") + name);
  }
  pos = parse_space(pos + 3, true);

  pos = parse_alternates(state, pos, name, rule_id, false);

  if (*pos == '\r') {
    pos += (pos[1] == '\n') ? 2 : 1;
  } else if (*pos == '\n') {
    ++pos;
  } else if (*pos != 0) {
    throw std::runtime_error("grammar: expecting newline or end after rule");
  }
  return parse_space(pos, true);
}

}  // namespace

Grammar Grammar::parse(const std::string& gbnf, const std::string& root_rule_name) {
  ParseState state;
  const char* pos = parse_space(gbnf.c_str(), true);
  while (*pos != 0) {
    pos = parse_rule(state, pos);
  }

  const auto it = state.symbol_ids.find(root_rule_name);
  if (it == state.symbol_ids.end()) {
    throw std::runtime_error("grammar: missing root rule '" + root_rule_name + "'");
  }
  // Validate that every referenced rule was actually defined.
  for (const Rule& rule : state.rules) {
    for (const Element& el : rule) {
      if (el.type == ElemType::RuleRef &&
          (el.value >= state.rules.size() || state.rules[el.value].empty())) {
        throw std::runtime_error("grammar: reference to undefined rule");
      }
    }
  }
  return Grammar(std::move(state.rules), it->second);
}

bool char_set_accepts(const Element* set, std::uint32_t cp) {
  return match_char(set, cp).first;
}

std::vector<Stack> grammar_accept_codepoint(const Grammar& grammar,
                                            const std::vector<Stack>& stacks, std::uint32_t cp) {
  return accept_codepoint(grammar.rules(), stacks, cp);
}

bool decode_simple_codepoints(const std::string& bytes, std::vector<std::uint32_t>& out) {
  const auto [cps, partial] = decode_utf8(bytes, PartialUtf8{});
  // n_remain == 0 means the bytes decoded into whole codepoints with nothing
  // pending. n_remain > 0 (ends mid-codepoint) or < 0 (invalid encoding) means
  // the token needs the slow path that tracks partial UTF-8 across tokens.
  if (partial.n_remain != 0) {
    return false;
  }
  out = cps;
  return true;
}

GrammarState::GrammarState(const Grammar& grammar)
    : grammar_(&grammar), stacks_(init_stacks(grammar)) {}

bool GrammarState::accept_bytes(const std::string& piece) {
  const auto [code_points, new_partial] = decode_utf8(piece, partial_utf8_);
  if (new_partial.n_remain < 0) {  // invalid encoding
    stacks_.clear();
    return false;
  }
  std::vector<Stack> stacks = stacks_;
  for (const std::uint32_t cp : code_points) {
    stacks = accept_codepoint(grammar_->rules(), stacks, cp);
    if (stacks.empty()) {
      stacks_.clear();
      return false;
    }
  }
  if (new_partial.n_remain > 0 && !any_stack_matches_partial(stacks, new_partial)) {
    stacks_.clear();
    return false;
  }
  stacks_ = std::move(stacks);
  partial_utf8_ = new_partial;
  return !stacks_.empty();
}

bool GrammarState::would_accept(const std::string& piece) const {
  const auto [code_points, new_partial] = decode_utf8(piece, partial_utf8_);
  if (new_partial.n_remain < 0) {
    return false;
  }
  std::vector<Stack> stacks = stacks_;
  for (const std::uint32_t cp : code_points) {
    stacks = accept_codepoint(grammar_->rules(), stacks, cp);
    if (stacks.empty()) {
      return false;
    }
  }
  if (new_partial.n_remain > 0 && !any_stack_matches_partial(stacks, new_partial)) {
    return false;
  }
  return !stacks.empty();
}

bool GrammarState::would_accept_codepoints(const std::vector<std::uint32_t>& cps) const {
  std::vector<Stack> stacks = stacks_;
  for (const std::uint32_t cp : cps) {
    stacks = accept_codepoint(grammar_->rules(), stacks, cp);
    if (stacks.empty()) {
      return false;
    }
  }
  return !stacks.empty();
}

void GrammarState::collect_stack_tops(std::vector<const Element*>& out) const {
  out.clear();
  for (const Stack& stack : stacks_) {
    if (!stack.empty()) {
      out.push_back(stack.back());
    }
  }
}

bool GrammarState::can_terminate() const {
  if (partial_utf8_.n_remain > 0) {
    return false;
  }
  for (const Stack& stack : stacks_) {
    if (stack.empty()) {
      return true;
    }
  }
  return false;
}

}  // namespace grammar
