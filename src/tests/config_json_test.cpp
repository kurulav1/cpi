// Round-trips LlamaConfig through the .cpi container's JSON metadata.
//
// This is the gate that makes the v7 whitelist bug impossible to repeat. That bug shipped a
// container with all-zero vision geometry because a field existed upstream and not in the
// packer's whitelist, and nothing compared the two. Here the writer, the reader AND this
// comparison are all generated from the one CPI_CONFIG_FIELDS list, so:
//
//   - a field added to the list is written, read and checked at once;
//   - a field LEFT OUT of the list fails this test, because it is filled with a distinct value
//     below and comes back as its default.
//
// Every scalar gets a DISTINCT value. Filling them with 1s would let a writer that emits the
// wrong field pass, since every wrong answer would still be 1.

#include <cstdio>
#include <string>

#include "model/config_json.hpp"

namespace {

int failures = 0;

void check(const char* what, bool ok, const std::string& detail = "") {
  if (!ok) {
    std::printf("  %-34s FAIL %s\n", what, detail.c_str());
    ++failures;
  }
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::printf("[config_json] LlamaConfig <-> .cpi __metadata__ round trip\n");

  model::LlamaConfig in;
  // Distinct values, assigned through the same list the serializer uses.
  int seed = 101;
#define CPI_FILL_I(member, key, kind) in.member = seed++;
#define CPI_FILL_F(member, key, kind) in.member = static_cast<float>(seed++) + 0.5f;
#define CPI_FILL_B(member, key, kind) in.member = ((seed++) % 2) == 0;
#define CPI_FILL(member, key, kind) CPI_FILL_##kind(member, key, kind)
  CPI_CONFIG_FIELDS(CPI_FILL)
#undef CPI_FILL
#undef CPI_FILL_I
#undef CPI_FILL_F
#undef CPI_FILL_B

  in.dtype = "int4";
  in.model_family = model::ModelFamily::Qwen3_5;
  in.layer_attention_kinds = {model::AttentionKind::Linear, model::AttentionKind::Full,
                              model::AttentionKind::SlidingWindow, model::AttentionKind::Linear};
  // Gemma 4 KV sharing. Deliberately includes indices ABOVE 9: layer_attention_kinds encodes one
  // digit per entry, and reusing that scheme here would silently turn 13 into 1 and 3. A 35-layer
  // E2B shares 20 layers onto sources in the teens, so this is the real case, not a corner one.
  in.kv_source = {0, 7, 13, 13, 14, 27};

  const std::string json = model::config_to_json(in);
  const model::LlamaConfig out = model::config_from_json(json);

  int checked = 0;
#define CPI_CMP_I(member, key, kind)                                              \
  check(key, out.member == in.member,                                             \
        "got " + std::to_string(out.member) + " want " + std::to_string(in.member)); \
  ++checked;
#define CPI_CMP_F(member, key, kind)                                              \
  check(key, out.member == in.member,                                             \
        "got " + std::to_string(out.member) + " want " + std::to_string(in.member)); \
  ++checked;
#define CPI_CMP_B(member, key, kind)                                              \
  check(key, out.member == in.member,                                             \
        std::string("got ") + (out.member ? "1" : "0") + " want " + (in.member ? "1" : "0")); \
  ++checked;
#define CPI_CMP(member, key, kind) CPI_CMP_##kind(member, key, kind)
  CPI_CONFIG_FIELDS(CPI_CMP)
#undef CPI_CMP
#undef CPI_CMP_I
#undef CPI_CMP_F
#undef CPI_CMP_B

  check("dtype", out.dtype == in.dtype, "got '" + out.dtype + "'");
  check("model_family", out.model_family == in.model_family);
  check("layer_attention_kinds.size", out.layer_attention_kinds.size() ==
                                          in.layer_attention_kinds.size());
  bool kinds_ok = out.layer_attention_kinds.size() == in.layer_attention_kinds.size();
  for (std::size_t i = 0; kinds_ok && i < in.layer_attention_kinds.size(); ++i) {
    kinds_ok = out.layer_attention_kinds[i] == in.layer_attention_kinds[i];
  }
  check("layer_attention_kinds", kinds_ok);
  bool kv_ok = out.kv_source.size() == in.kv_source.size();
  for (std::size_t i = 0; kv_ok && i < in.kv_source.size(); ++i) {
    kv_ok = out.kv_source[i] == in.kv_source[i];
  }
  check("kv_source", kv_ok,
        "size " + std::to_string(out.kv_source.size()) + " want " +
            std::to_string(in.kv_source.size()));
  checked += 4;

  // THE RESIDUAL HOLE, and why this count is pinned.
  //
  // The X-macro guarantees the writer and the reader agree -- a field in the list is in both. It
  // does NOT guarantee the list covers LlamaConfig: add a member to the struct and forget this
  // file, and everything above still passes while the new field silently rides its default. That
  // is the v7 whitelist bug wearing a different hat.
  //
  // Pinning the count is the cheap half of the guard: REMOVING a field from the list trips it
  // immediately. For ADDING one, LlamaConfig's declaration carries a pointer back here -- a
  // sizeof() static_assert would be stronger but is not portable, because std::string/std::vector
  // differ in size between MSVC and libc++ and the number would be wrong on one of them.
  check("field count (update when CPI_CONFIG_FIELDS changes)", checked == 55,
        "got " + std::to_string(checked) + " want 55");
  std::printf("  %d fields round-tripped\n", checked);
  std::printf("[config_json] %s\n", failures == 0 ? "ALL PASS" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
