#pragma once

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "engine/batch_scheduler.hpp"
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
// Streams a reply for a prompt that contains an <|image|> placeholder. Null for engines
// with no vision tower -- the request then fails cleanly instead of silently ignoring the
// picture.
using GenerateMultimodalFn = std::function<std::vector<int>(
    const std::vector<int>& base_tokens, const std::string& image_path, int max_new_tokens,
    float temperature, const std::function<bool(int)>& on_token)>;
using InspectNextLogitsFn = std::function<std::vector<std::pair<int, float>>(
    const std::vector<int>& prompt_tokens, int top_k)>;
using LastBenchmarkStatsFn = std::function<const engine::BenchmarkStats&()>;

void execute_engine_modes(const RunExecutionOptions& options, const std::vector<int>& prompt_tokens,
                          const std::vector<int>& stop_token_ids,
                          const std::vector<std::string>& stop_texts, model::Tokenizer* tokenizer,
                          const GenerateFn& generate, const GenerateStreamFn& generate_stream,
                          const InspectNextLogitsFn& inspect_next_logits,
                          const LastBenchmarkStatsFn& last_benchmark_stats,
                          const GenerateMultimodalFn& generate_multimodal = nullptr);

// Multiplexed continuous-batching interactive worker (opt-in, --interactive-batch).
// See main_interactive_batch.cpp.
//
// Takes the scheduler rather than an engine, and is NOT gated on CUDA: engine::BatchScheduler
// is backend-free, so whichever engine built it (LlamaEngine or PlanMetalEngine) is invisible
// here. The gate used to exist because the scheduler lived inside LlamaEngine.
void run_interactive_batch(engine::BatchScheduler& sched, model::Tokenizer& tokenizer,
                           const std::vector<std::string>& default_stop_texts, bool default_add_bos,
                           int default_max_new, float default_temp);

}  // namespace app::main_modes
