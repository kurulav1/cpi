// Guards the response_format.json_schema unwrap.
//
// The bug this exists for: OpenAI nests the schema one level below
// response_format, as {"name":..., "strict":..., "schema":{...}}, and both HTTP
// request paths passed that wrapper straight to the grammar converter. A wrapper
// has neither "type" nor "properties", so it compiled to a permissive any-JSON
// grammar and a request for an object with two required fields was answered with
// "[1]". Valid JSON, conforming to nothing, and no error on any path.
//
// It stayed invisible because a second bug upstream was dropping the grammar
// entirely, so the output was unconstrained prose either way and every reading
// pointed at the wiring instead. Only after that was fixed did "[1]" appear.

#include <cstdio>
#include <string>

#include "app/main_helpers.hpp"

namespace {

int failures = 0;

void check(const char* what, bool ok, const std::string& detail = "") {
  if (!ok) {
    std::printf("  %-44s FAIL %s\n", what, detail.c_str());
    ++failures;
  }
}

bool has(const std::string& hay, const char* needle) {
  return hay.find(needle) != std::string::npos;
}

}  // namespace

int main() {
  using app::main_helpers::unwrap_json_schema;

  // The OpenAI form: unwrap to the inner schema.
  {
    const std::string wrapped =
        R"({"name":"weather","strict":true,"schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}})";
    const std::string got = unwrap_json_schema(wrapped);
    check("openai wrapper unwraps to the schema", has(got, "\"type\"") && has(got, "properties"),
          got);
    check("openai wrapper drops the envelope", !has(got, "\"strict\"") && !has(got, "\"name\""),
          got);
  }

  // The bare CPI form: unchanged. This is what the docs and the batched path use.
  {
    const std::string bare = R"({"type":"object","properties":{"city":{"type":"string"}}})";
    check("bare schema is returned unchanged", unwrap_json_schema(bare) == bare);
  }

  // "$schema" is JSON Schema's own metadata key and must not trigger the unwrap.
  // Matching it would silently replace the schema with a URL string.
  {
    const std::string with_meta =
        R"({"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object"})";
    check("$schema is not mistaken for the wrapper", unwrap_json_schema(with_meta) == with_meta,
          unwrap_json_schema(with_meta));
  }

  // Empty in, empty out: no schema requested means no grammar, not an error.
  check("empty stays empty", unwrap_json_schema("").empty());

  // A nested "schema" property inside "properties" is not a top-level key, so an
  // object that describes a field called "schema" must survive untouched.
  {
    const std::string nested =
        R"({"type":"object","properties":{"schema":{"type":"string"}},"required":["schema"]})";
    check("a property named schema is not unwrapped", unwrap_json_schema(nested) == nested,
          unwrap_json_schema(nested));
  }

  if (failures == 0) {
    std::printf("json_schema_unwrap_test: all checks passed\n");
    return 0;
  }
  std::printf("json_schema_unwrap_test: %d check(s) failed\n", failures);
  return 1;
}
