#include "grammar/json_schema_to_grammar.hpp"

#include <cctype>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace grammar {
namespace {

// ----------------------------------------------------------------------------
// Minimal JSON DOM + parser (self-contained; the project has no JSON library).
// ----------------------------------------------------------------------------

struct JsonValue {
  enum class Type { Null, Bool, Number, String, Array, Object };
  Type type = Type::Null;
  bool bool_val = false;
  std::string text;  // string contents, or raw numeric token text
  std::vector<JsonValue> arr;
  std::vector<std::pair<std::string, JsonValue>> obj;  // declaration order preserved

  const JsonValue* find(const std::string& key) const {
    for (const auto& kv : obj) {
      if (kv.first == key) {
        return &kv.second;
      }
    }
    return nullptr;
  }
};

class JsonParser {
public:
  explicit JsonParser(const std::string& src) : s_(src) {}

  JsonValue parse() {
    skip_ws();
    JsonValue v = parse_value();
    skip_ws();
    if (i_ != s_.size()) {
      throw std::runtime_error("json schema: trailing content after value");
    }
    return v;
  }

private:
  const std::string& s_;
  std::size_t i_ = 0;

  [[noreturn]] void fail(const char* msg) const {
    throw std::runtime_error(std::string("json schema parse error: ") + msg);
  }

  char peek() const {
    return i_ < s_.size() ? s_[i_] : '\0';
  }

  void skip_ws() {
    while (i_ < s_.size()) {
      const char c = s_[i_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++i_;
      } else {
        break;
      }
    }
  }

  JsonValue parse_value() {
    const char c = peek();
    switch (c) {
      case '{':
        return parse_object();
      case '[':
        return parse_array();
      case '"': {
        JsonValue v;
        v.type = JsonValue::Type::String;
        v.text = parse_string();
        return v;
      }
      case 't':
      case 'f':
        return parse_bool();
      case 'n':
        return parse_null();
      default:
        if (c == '-' || (c >= '0' && c <= '9')) {
          return parse_number();
        }
        fail("unexpected character");
    }
  }

  JsonValue parse_object() {
    JsonValue v;
    v.type = JsonValue::Type::Object;
    ++i_;  // '{'
    skip_ws();
    if (peek() == '}') {
      ++i_;
      return v;
    }
    while (true) {
      skip_ws();
      if (peek() != '"') {
        fail("expecting object key string");
      }
      const std::string key = parse_string();
      skip_ws();
      if (peek() != ':') {
        fail("expecting ':' after object key");
      }
      ++i_;
      skip_ws();
      v.obj.emplace_back(key, parse_value());
      skip_ws();
      const char c = peek();
      if (c == ',') {
        ++i_;
        continue;
      }
      if (c == '}') {
        ++i_;
        break;
      }
      fail("expecting ',' or '}' in object");
    }
    return v;
  }

  JsonValue parse_array() {
    JsonValue v;
    v.type = JsonValue::Type::Array;
    ++i_;  // '['
    skip_ws();
    if (peek() == ']') {
      ++i_;
      return v;
    }
    while (true) {
      skip_ws();
      v.arr.push_back(parse_value());
      skip_ws();
      const char c = peek();
      if (c == ',') {
        ++i_;
        continue;
      }
      if (c == ']') {
        ++i_;
        break;
      }
      fail("expecting ',' or ']' in array");
    }
    return v;
  }

  std::string parse_string() {
    ++i_;  // opening quote
    std::string out;
    while (i_ < s_.size()) {
      const char c = s_[i_++];
      if (c == '"') {
        return out;
      }
      if (c == '\\') {
        if (i_ >= s_.size()) {
          fail("truncated escape in string");
        }
        const char e = s_[i_++];
        switch (e) {
          case '"':
            out.push_back('"');
            break;
          case '\\':
            out.push_back('\\');
            break;
          case '/':
            out.push_back('/');
            break;
          case 'b':
            out.push_back('\b');
            break;
          case 'f':
            out.push_back('\f');
            break;
          case 'n':
            out.push_back('\n');
            break;
          case 'r':
            out.push_back('\r');
            break;
          case 't':
            out.push_back('\t');
            break;
          case 'u': {
            // Keep the escape verbatim; schema strings used here (keys, enum
            // members) are ASCII in practice, and we only need round-trippable
            // text for grammar literals.
            out.push_back('\\');
            out.push_back('u');
            for (int k = 0; k < 4 && i_ < s_.size(); ++k) {
              out.push_back(s_[i_++]);
            }
            break;
          }
          default:
            fail("invalid escape in string");
        }
      } else {
        out.push_back(c);
      }
    }
    fail("unterminated string");
  }

  JsonValue parse_number() {
    const std::size_t start = i_;
    if (peek() == '-') {
      ++i_;
    }
    while (i_ < s_.size()) {
      const char c = s_[i_];
      if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') {
        ++i_;
      } else {
        break;
      }
    }
    JsonValue v;
    v.type = JsonValue::Type::Number;
    v.text = s_.substr(start, i_ - start);
    return v;
  }

  JsonValue parse_bool() {
    JsonValue v;
    v.type = JsonValue::Type::Bool;
    if (s_.compare(i_, 4, "true") == 0) {
      v.bool_val = true;
      i_ += 4;
    } else if (s_.compare(i_, 5, "false") == 0) {
      v.bool_val = false;
      i_ += 5;
    } else {
      fail("invalid literal");
    }
    return v;
  }

  JsonValue parse_null() {
    if (s_.compare(i_, 4, "null") != 0) {
      fail("invalid literal");
    }
    i_ += 4;
    JsonValue v;
    v.type = JsonValue::Type::Null;
    return v;
  }
};

// ----------------------------------------------------------------------------
// GBNF builder
// ----------------------------------------------------------------------------

// Wraps raw text as a GBNF double-quoted string literal, escaping '\' and '"'.
std::string gbnf_literal(const std::string& raw) {
  std::string out = "\"";
  for (const char c : raw) {
    if (c == '\\' || c == '"') {
      out.push_back('\\');
    }
    out.push_back(c);
  }
  out.push_back('"');
  return out;
}

// Serializes a JSON value back to canonical JSON text (used for enum literals).
std::string json_serialize(const JsonValue& v) {
  switch (v.type) {
    case JsonValue::Type::Null:
      return "null";
    case JsonValue::Type::Bool:
      return v.bool_val ? "true" : "false";
    case JsonValue::Type::Number:
      return v.text;
    case JsonValue::Type::String: {
      std::string out = "\"";
      for (const char c : v.text) {
        if (c == '"' || c == '\\') {
          out.push_back('\\');
        }
        out.push_back(c);
      }
      out.push_back('"');
      return out;
    }
    default:
      throw std::runtime_error("json schema: enum members must be scalars");
  }
}

class GrammarBuilder {
public:
  std::string build(const JsonValue& schema) {
    const std::string root_ref = visit(schema, "root");
    // Leading whitespace is tolerated, but there is deliberately no trailing
    // whitespace rule: once the top-level value closes, the only grammar-legal
    // continuation is EOS, which forces a clean stop rather than letting the
    // model ramble past the closing brace.
    ensure_ws();
    std::ostringstream out;
    out << "root ::= ws " << root_ref << "\n";
    for (const auto& [name, def] : rules_) {
      out << name << " ::= " << def << "\n";
    }
    return out.str();
  }

private:
  std::vector<std::pair<std::string, std::string>> rules_;
  std::set<std::string> defined_;
  int counter_ = 0;

  std::string fresh(const std::string& base) {
    // Sanitize to a valid GBNF identifier: rule names allow [A-Za-z0-9_-].
    std::string clean;
    clean.reserve(base.size());
    for (const char c : base) {
      const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') ||
                      c == '-' || c == '_';
      clean.push_back(ok ? c : '_');
    }
    return clean + "-" + std::to_string(counter_++);
  }

  // Adds a rule if its name is new; returns the name for use as a reference.
  std::string define(const std::string& name, const std::string& def) {
    if (defined_.insert(name).second) {
      rules_.emplace_back(name, def);
    }
    return name;
  }

  // --- primitive rules (added on demand) ---
  void ensure_ws() {
    define("ws", "[ \\t\\n]*");
  }
  std::string ensure_boolean() {
    return define("boolean", "\"true\" | \"false\"");
  }
  std::string ensure_null() {
    return define("null", "\"null\"");
  }
  std::string ensure_integer() {
    return define("integer", "\"-\"? [0-9]+");
  }
  std::string ensure_number() {
    return define("number", "\"-\"? [0-9]+ (\".\" [0-9]+)? ([eE] [-+]? [0-9]+)?");
  }
  std::string ensure_string() {
    define("hex", "[0-9a-fA-F]");
    define("char", "[^\"\\\\] | \"\\\\\" ([\"\\\\/bfnrt] | \"u\" hex hex hex hex)");
    return define("string", "\"\\\"\" char* \"\\\"\"");
  }
  std::string ensure_value() {
    ensure_ws();
    const std::string s = ensure_string();
    const std::string n = ensure_number();
    const std::string b = ensure_boolean();
    const std::string nl = ensure_null();
    define("object-any", "\"{\" ws ( " + s + " ws \":\" ws value ( ws \",\" ws " + s +
                             " ws \":\" ws value )* )? ws \"}\"");
    define("array-any", "\"[\" ws ( value ( ws \",\" ws value )* )? ws \"]\"");
    return define("value", "object-any | array-any | " + s + " | " + n + " | " + b + " | " + nl);
  }

  // Generates a rule for `schema` and returns a GBNF reference (rule name).
  // `hint` seeds generated rule names for readability.
  std::string visit(const JsonValue& schema, const std::string& hint) {
    if (schema.type != JsonValue::Type::Object) {
      // A bare `true`/empty schema means "any value".
      return ensure_value();
    }

    // enum: alternation of literal members.
    if (const JsonValue* en = schema.find("enum")) {
      if (en->type != JsonValue::Type::Array || en->arr.empty()) {
        throw std::runtime_error("json schema: 'enum' must be a non-empty array");
      }
      std::string def;
      for (std::size_t k = 0; k < en->arr.size(); ++k) {
        if (k) {
          def += " | ";
        }
        def += gbnf_literal(json_serialize(en->arr[k]));
      }
      return define(fresh(hint + "-enum"), def);
    }

    const JsonValue* type = schema.find("type");
    if (type == nullptr) {
      return ensure_value();
    }
    if (type->type == JsonValue::Type::Array) {
      // Union of simple types, e.g. ["string","null"].
      std::string def;
      for (std::size_t k = 0; k < type->arr.size(); ++k) {
        if (type->arr[k].type != JsonValue::Type::String) {
          throw std::runtime_error("json schema: 'type' array members must be strings");
        }
        if (k) {
          def += " | ";
        }
        def += ref_for_simple_type(type->arr[k].text, schema, hint);
      }
      return define(fresh(hint + "-union"), def);
    }
    if (type->type != JsonValue::Type::String) {
      throw std::runtime_error("json schema: 'type' must be a string or array");
    }
    return ref_for_simple_type(type->text, schema, hint);
  }

  std::string ref_for_simple_type(const std::string& type, const JsonValue& schema,
                                  const std::string& hint) {
    if (type == "object") {
      return visit_object(schema, hint);
    }
    if (type == "array") {
      return visit_array(schema, hint);
    }
    if (type == "string") {
      return ensure_string();
    }
    if (type == "integer") {
      return ensure_integer();
    }
    if (type == "number") {
      return ensure_number();
    }
    if (type == "boolean") {
      return ensure_boolean();
    }
    if (type == "null") {
      return ensure_null();
    }
    throw std::runtime_error("json schema: unsupported type '" + type + "'");
  }

  std::string visit_array(const JsonValue& schema, const std::string& hint) {
    ensure_ws();
    std::string item_ref;
    if (const JsonValue* items = schema.find("items")) {
      item_ref = visit(*items, hint + "-item");
    } else {
      item_ref = ensure_value();
    }
    const std::string def =
        "\"[\" ws ( " + item_ref + " ( ws \",\" ws " + item_ref + " )* )? ws \"]\"";
    return define(fresh(hint + "-array"), def);
  }

  std::string visit_object(const JsonValue& schema, const std::string& hint) {
    ensure_ws();
    const JsonValue* props = schema.find("properties");
    if (props == nullptr || props->type != JsonValue::Type::Object || props->obj.empty()) {
      // No declared properties -> accept any JSON object.
      ensure_value();
      return define(fresh(hint + "-object"),
                    "\"{\" ws ( string ws \":\" ws value ( ws \",\" ws string ws \":\" ws value )* "
                    ")? ws \"}\"");
    }

    // Determine which properties are required.
    std::set<std::string> required;
    if (const JsonValue* req = schema.find("required")) {
      if (req->type == JsonValue::Type::Array) {
        for (const JsonValue& r : req->arr) {
          if (r.type == JsonValue::Type::String) {
            required.insert(r.text);
          }
        }
      }
    }

    // v1 property model:
    //   - If any properties are required: required ones are mandatory in
    //     declaration order, then optional ones follow as trailing optional
    //     groups (fixed required-before-optional ordering).
    //   - If none are required: all properties are treated as mandatory in
    //     declaration order. Any-order / fully-optional support is a follow-up;
    //     a CFG cannot express "any subset in any order" without enumerating
    //     permutations.
    const bool any_required = !required.empty();

    std::vector<std::size_t> mandatory;  // indices into props->obj
    std::vector<std::size_t> optional;
    for (std::size_t k = 0; k < props->obj.size(); ++k) {
      const bool is_req = !any_required || required.count(props->obj[k].first) > 0;
      if (is_req) {
        mandatory.push_back(k);
      } else {
        optional.push_back(k);
      }
    }

    auto clause = [&](std::size_t idx) {
      const auto& [name, subschema] = props->obj[idx];
      const std::string val_ref = visit(subschema, hint + "-" + name);
      return gbnf_literal("\"" + name + "\"") + " ws \":\" ws " + val_ref;
    };

    std::string def = "\"{\" ws ";
    for (std::size_t m = 0; m < mandatory.size(); ++m) {
      if (m) {
        def += " ws \",\" ws ";
      }
      def += clause(mandatory[m]);
    }
    for (const std::size_t idx : optional) {
      def += " ( ws \",\" ws " + clause(idx) + " )?";
    }
    def += " ws \"}\"";
    return define(fresh(hint + "-object"), def);
  }
};

}  // namespace

std::string json_schema_to_grammar(const std::string& schema_json) {
  JsonParser parser(schema_json);
  const JsonValue schema = parser.parse();
  GrammarBuilder builder;
  return builder.build(schema);
}

}  // namespace grammar
