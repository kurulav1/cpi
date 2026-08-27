// Gates the CONTAINER-IDENTITY probe: which engine a `.cpi` resolves to.
//
// This exists because a `.cpi` says nothing about the model inside it; it is a container format,
// not a family; so probe_model looks at the safetensors `__metadata__` block. The trap it walked
// into, and the reason for this file: a metadata block is not proof the block is OURS.
//
//   * OUR containers (ll2c_to_cpi) write config_to_json's schema:  model_family, hidden_size, ...
//   * Gemma 4's converter writes its OWN schema:                   family, hidden, vocab, ...
//
// config_from_json reads only the first, so handed the second it returns a fully DEFAULTED
// LlamaConfig; and LlamaConfig::hidden_size defaults to 4096, not 0. The original predicate
// `c.hidden_size > 0` was therefore true for every container carrying any metadata, which routed
// Gemma 4 `.cpi` files to LlamaEngine (a `.ll2c`-only reader). It failed with
// "unsupported weights format. expected LL2CUDA manifest"; an error naming a format nobody asked
// for, from an engine nobody selected.
//
// control, at the bottom: this file asserts that a defaulted parse really does yield hidden_size
// 4096, so the fix is pinned to the actual mechanism rather than to a symptom. If someone changes
// that default to 0 the control fails loudly and this comment stops being true.

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

#include "app/main_helpers.hpp"
#include "model/config_json.hpp"

namespace {

int g_failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("%-58s %s\n", what.c_str(), ok ? "PASS" : "FAIL");
  if (!ok) ++g_failures;
}

// Minimal safetensors container: 8-byte LE header length, JSON directory, then raw data. One
// tiny F16 tensor so the file is structurally valid; the probe only reads the header.
void write_container(const std::filesystem::path& p, const std::string& metadata_json) {
  std::string header = "{";
  if (!metadata_json.empty()) header += "\"__metadata__\":" + metadata_json + ",";
  header += "\"t\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[0,4]}}";

  std::ofstream out(p, std::ios::binary);
  const std::uint64_t n = header.size();
  for (int i = 0; i < 8; ++i) out.put(static_cast<char>((n >> (8 * i)) & 0xFF));
  out.write(header.data(), static_cast<std::streamsize>(header.size()));
  const char payload[4] = {0, 0, 0, 0};
  out.write(payload, 4);
}

// Exactly the shape tools/convert_gemma4.py emits: its own vocabulary, sharing not one key with
// config_to_json's. Trimmed to the fields that matter for the collision.
const char* kGemma4Metadata =
    "{\"family\":\"gemma4\",\"hidden\":\"1536\",\"vocab\":\"262144\",\"num_layers\":\"35\","
    "\"head_dim\":\"256\",\"hidden_size_per_layer_input\":\"256\",\"sliding_window\":\"512\"}";

}  // namespace

int main() {
  namespace mh = app::main_helpers;
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "cpi_probe_model_test";
  std::filesystem::create_directories(dir);

  // 1. A Gemma 4 container. Has metadata, but not OUR metadata -> must stay Gemma 4.
  //    This is the regression: it resolved to Llama and hit a .ll2c-only loader.
  const auto gemma = dir / "gemma4.cpi";
  write_container(gemma, kGemma4Metadata);
  check(mh::probe_model(gemma.string()).kind == mh::ModelFamilyKind::Gemma4,
        "gemma4 .cpi (foreign metadata schema) -> Gemma4");

  // 2. No metadata at all; the pre-config_to_json Gemma 4 containers.
  const auto bare = dir / "bare.cpi";
  write_container(bare, "");
  check(mh::probe_model(bare.string()).kind == mh::ModelFamilyKind::Gemma4,
        "bare .cpi (no metadata) -> Gemma4");

  // 3. OUR container, a uniform-geometry model (what ll2c_to_cpi writes for Qwen2.5/Llama).
  model::LlamaConfig llama;
  llama.model_family = model::ModelFamily::Qwen2;
  llama.hidden_size = 896;
  llama.num_layers = 24;
  const auto repacked = dir / "repacked.cpi";
  write_container(repacked, model::config_to_json(llama));
  check(mh::probe_model(repacked.string()).kind == mh::ModelFamilyKind::Llama,
        "repacked .cpi (config_to_json schema) -> Llama");

  // 4. OUR container, Qwen3.5; the family that needs its own engine.
  model::LlamaConfig q35;
  q35.model_family = model::ModelFamily::Qwen3_5;
  q35.hidden_size = 1024;
  const auto q35_path = dir / "qwen35.cpi";
  write_container(q35_path, model::config_to_json(q35));
  check(mh::probe_model(q35_path.string()).kind == mh::ModelFamilyKind::Qwen35,
        "qwen3.5 .cpi (config_to_json schema) -> Qwen35");

  // 5. control. Without this the four checks above could all pass while testing nothing, because
  //    they never state why the old predicate was wrong. Parsing foreign metadata must yield a
  //    fully defaulted config; and that default must be the non-zero value that made
  //    `hidden_size > 0` vacuous. Both halves are asserted.
  const model::LlamaConfig defaulted = model::config_from_json(kGemma4Metadata);
  check(defaulted.hidden_size == 4096,
        "CONTROL: foreign metadata parses to DEFAULT hidden_size 4096");
  check(defaulted.hidden_size > 0, "CONTROL: so the old `hidden_size > 0` test was always true");
  check(defaulted.model_family != model::ModelFamily::Qwen3_5,
        "CONTROL: and model_family stayed unset");

  std::filesystem::remove_all(dir);
  if (g_failures == 0) {
    std::printf("\nprobe_model_test: ALL PASS\n");
    return 0;
  }
  std::printf("\nprobe_model_test: %d FAILED\n", g_failures);
  return 1;
}
