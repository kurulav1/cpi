#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "engine/engine_types.hpp"
#include "engine/generation_constraints.hpp"

namespace model {
class Tokenizer;
}
namespace engine {
class LlamaEngine;
}

namespace app::main_modes {

struct RunExecutionOptions {
  bool interactive_mode = false;
  bool quiet_output = false;
  bool use_tokenizer = false;
  bool sentence_stop = false;
  bool benchmark_mode = false;
  bool benchmark_phases = false;
  bool simple_mode = false;
  bool force_no_bos = false;
  int max_new = 16;
  float temp = 0.8f;
  int inspect_next_topk = 0;
  int trace_steps = 0;
  int benchmark_reps = 1;
  int benchmark_warmup = 0;
};

using GenerateFn = std::function<std::vector<int>(const std::vector<int>& prompt_tokens,
                                                  int max_new_tokens, float temperature)>;
using GenerateStreamFn = std::function<std::vector<int>(
    const std::vector<int>& prompt_tokens, int max_new_tokens, float temperature,
    const std::function<bool(int)>& on_token, const engine::GenerationConstraints* constraints)>;
using InspectNextLogitsFn = std::function<std::vector<std::pair<int, float>>(
    const std::vector<int>& prompt_tokens, int top_k)>;
using LastBenchmarkStatsFn = std::function<const engine::BenchmarkStats&()>;

void execute_engine_modes(const RunExecutionOptions& options, const std::vector<int>& prompt_tokens,
                          const std::vector<int>& stop_token_ids,
                          const std::vector<std::string>& stop_texts, model::Tokenizer* tokenizer,
                          const GenerateFn& generate, const GenerateStreamFn& generate_stream,
                          const InspectNextLogitsFn& inspect_next_logits,
                          const LastBenchmarkStatsFn& last_benchmark_stats);

#if LLAMA_ENGINE_HAS_CUDA
// Multiplexed continuous-batching interactive worker (opt-in, --interactive-batch).
// Drives the engine's streaming batch scheduler; see main_interactive_batch.cpp.
void run_interactive_batch(engine::LlamaEngine& eng, model::Tokenizer& tokenizer,
                           const std::vector<std::string>& default_stop_texts, bool default_add_bos,
                           int default_max_new, float default_temp);
#endif

}  // namespace app::main_modes
