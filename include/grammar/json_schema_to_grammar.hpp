#pragma once

// Compiles a JSON Schema into a GBNF grammar string suitable for grammar::Grammar.
// A subset of JSON Schema is supported, covering what typical tool schemas use:
//   - "type": object, array, string, integer, number, boolean, null
//   - object: "properties", "required"
//   - array:  "items"
//   - "enum"  (string / number / boolean / null members)
//   - nested objects and arrays
// Unknown or absent "type" falls back to a permissive "any JSON value" rule.
//
// The generated grammar's entry rule is named "root", matching the default
// grammar::Grammar::parse() root.

#include <string>

namespace grammar {

// Converts a JSON Schema document (raw JSON text) into a GBNF grammar string.
// Throws std::runtime_error on a JSON parse error or an unsupported construct.
std::string json_schema_to_grammar(const std::string& schema_json);

}  // namespace grammar
