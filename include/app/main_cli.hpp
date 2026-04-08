#pragma once

#include <iosfwd>
#include <string>
#include <vector>

#include "engine/llama_engine.hpp"

namespace app::main_cli {

struct ParsedArgs {
  engine::EngineOptions opts{};
  std::string prompt_text;
  std::string chat_template;
  std::string tokenizer_path;
  std::string token_csv = "1,2,3";
  int max_new = 16;
  float temp = 0.8f;
  bool parity_check = false;
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

}  // namespace app::main_cli
