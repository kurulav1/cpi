#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include "engine/engine_types.hpp"

namespace app::main_cli {

struct ParsedArgs {
  engine::EngineOptions opts{};
  std::string prompt_text;
  std::string image_path;  // --image: a PNG to place before the prompt text
  std::string chat_template;
  std::string tokenizer_path;
  // Speculative decoding: a small draft model (same tokenizer/vocab as the
  // target) proposes spec_tokens tokens per round that the target verifies in
  // one batched forward pass. Empty draft_model_path disables it.
  std::string draft_model_path;
  int spec_tokens = 5;
  std::string token_csv = "1,2,3";
  int max_new = 16;
  float temp = 0.8f;
  bool parity_check = false;
  int batched_check = 0;           // >0: run decode_step_batched parity gate for this many steps
  int scheduler_check = 0;         // >0: run the batch-scheduler parity gate with this max_new
  bool interactive_batch = false;  // multiplexed continuous-batching interactive worker
  bool serve_http = false;         // --serve: in-binary OpenAI-compatible HTTP server
  int serve_port = 8080;           // --port
  std::string serve_host = "127.0.0.1";  // --host (0.0.0.0 to expose beyond loopback)
  std::string serve_api_key;             // --api-key / CPI_API_KEY: bearer token for /v1/*
  std::string serve_embed_model;         // --embed-model <dir>: enables /v1/embeddings
  int batch_bench = 0;             // >0: run the batch throughput benchmark with this max_new
  int tune_kquant = 0;             // >0: search the k-quant kernel knobs at this batch size
  bool dump_tokenizer_meta = false;
  bool dump_prompt_tokens = false;
  bool allow_legacy_chat_tokenizer = false;
  bool force_no_bos = false;
  bool sentence_stop = false;
  bool benchmark_mode = false;
  bool benchmark_phases = false;
  bool runtime_metrics = false;
  bool web_mode = false;
  bool interactive_mode = false;
  bool simple_mode = false;
  bool force_cpu = false;
  int benchmark_reps = 1;
  int benchmark_warmup = 0;
  int inspect_next_topk = 0;
  int trace_steps = 0;
  std::vector<std::string> stop_texts;
  bool max_new_set = false;
  bool temp_set = false;
  bool chat_template_set = false;
  bool tokenizer_set = false;
  bool cache_mode_set = false;
  bool weight_quant_set = false;
};

void print_usage(std::ostream& os);
ParsedArgs parse_args(int argc, char** argv);
void apply_simple_mode_defaults(ParsedArgs* args);
void validate_args(const ParsedArgs& args);
// Backward-compat: map any set LLAMA_INFER_* env var onto its CPI_* successor (unless the new
// name is already set), with a one-line deprecation notice. Call once at startup, before any
// getenv. Removed in a future release.
void apply_legacy_env_aliases();

}  // namespace app::main_cli
