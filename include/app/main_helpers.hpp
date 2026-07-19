#pragma once

#include <cstddef>
#include <string>
#include <vector>

namespace app::main_helpers {

class SingleInstanceGuard {
public:
  SingleInstanceGuard() = default;
  ~SingleInstanceGuard();

  SingleInstanceGuard(const SingleInstanceGuard&) = delete;
  SingleInstanceGuard& operator=(const SingleInstanceGuard&) = delete;

  bool acquire();

private:
  void release();

#ifdef _WIN32
  void* mutex_ = nullptr;
#else
  int fd_ = -1;
#endif
};

std::vector<int> parse_tokens(const std::string& csv);
std::string join_ints(const std::vector<int>& values, std::size_t limit = 0);

std::string json_get_string(const std::string& json, const std::string& key);
// Returns the raw JSON text of a key's value (object, array, string, or scalar),
// brace/bracket-balanced and string-aware. Use for nested values like
// `json_schema` that json_get_string (scalars only) cannot extract. Empty if absent.
std::string json_get_raw_value(const std::string& json, const std::string& key);
int json_get_int(const std::string& json, const std::string& key, int def);
float json_get_float(const std::string& json, const std::string& key, float def);
bool json_get_bool(const std::string& json, const std::string& key, bool def);
std::vector<std::string> json_get_string_array(const std::string& json, const std::string& key);
std::string json_escape(const std::string& s);

std::string build_chat_prompt(const std::string& chat_template, const std::string& prompt_text,
                              bool tinyllama_plain_fallback);
std::vector<std::string> default_stop_texts_for_template(const std::string& chat_template);

std::string sanitize_stream_text(std::string s);
std::string normalize_final_response_text(const std::string& text);
std::size_t find_first_stop_pos(const std::string& text, const std::vector<std::string>& stops);
bool has_complete_sentence(const std::string& text);

bool is_safetensors_model_dir(const std::string& path);
std::string infer_safetensors_model_family(const std::string& path);
std::string guess_chat_template_from_model_path(const std::string& model_path);
std::string auto_detect_tokenizer_path(const std::string& model_path);

// Which engine family a model maps to. The CPU-vs-CUDA choice is a separate,
// runtime concern (force_cpu / device availability), not a model property.
enum class ModelFamilyKind { Llama, Gemma4, Qwen35, Llama4 };

// Centralized model-identity classification: the single place that inspects a
// model path and reports its engine family + the raw facts dispatch keys off.
// This is the capability seam — as the engines converge onto one config-driven
// executor, this grows into a full capability descriptor and the caller stops
// switching on identity. For now it just de-scatters the detection.
struct ModelProbe {
  ModelFamilyKind kind = ModelFamilyKind::Llama;
  bool is_cpi = false;              // Gemma 4 fork container (.cpi)
  bool is_safetensors_dir = false;  // Llama4-style safetensors directory
  std::string safetensors_family;   // e.g. "qwen3_5" for a Qwen3.5 safetensors dir; else ""
};

ModelProbe probe_model(const std::string& model_path);

// The concrete engine to construct for a resolved (model, device) combination.
// Dispatch resolves this once and switches on it, replacing the former scattered
// is_X/use_X boolean cascade. CPU-vs-CUDA is a runtime decision captured here, not
// a model property (see ModelFamilyKind).
enum class EngineChoice {
  LlamaCuda,   // default fast path (.ll2c and anything not otherwise classified)
  LlamaMetal,  // PlanMetalEngine on Apple Silicon (no CUDA, Metal GPU present)
  LlamaCpu,    // CpuLlamaEngine (force_cpu / no CUDA device / no-CUDA build)
  PlanCuda,    // generic per-layer op-plan executor (Gemma 4 .cpi today) — CUDA only
  Qwen35Cuda,
  Qwen35Metal,  // PlanMetalEngine running the shared Qwen3.5 op plan (Apple Silicon, no CUDA)
  Qwen35Cpu,
  Llama4Cuda,
  Llama4Cpu,
};

// Resolves the model family (from a probe) plus the runtime device situation into
// a single engine choice — the whole dispatch decision in one place. Throws if the
// model cannot run on the available device (Gemma 4 currently requires CUDA).
EngineChoice resolve_engine(const ModelProbe& probe, bool cuda_available, bool metal_available,
                            bool force_cpu);

}  // namespace app::main_helpers
