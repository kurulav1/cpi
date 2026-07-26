#include "engine/decode_driver.hpp"

#include <chrono>
#include <stdexcept>

#include "engine/sampling.hpp"

namespace engine {
namespace runtime {

// Default sampling: apply the optional grammar mask, then the shared host sampler
// (temperature/top-k/top-p/repetition penalty/no-repeat-ngram). Engines with a
// device-side greedy fast path override SequenceModel::sample.
int SequenceModel::sample(const DecodeParams& params, const std::vector<int>& history) {
  std::vector<float>& lg = logits();
  if (params.grammar_mask) params.grammar_mask(lg);
  return detail::dispatch_sample_from_logits(lg, params.temperature, params.top_k, params.top_p,
                                             params.repetition_penalty, params.no_repeat_ngram_size,
                                             history);
}

std::vector<int> run_decode(SequenceModel& model, const std::vector<int>& prompt,
                            const DecodeParams& params,
                            const std::function<bool(int)>& on_token, BenchmarkStats* stats) {
  using clock = std::chrono::steady_clock;
  const auto ms = [](clock::time_point a, clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
  };

  std::vector<int> out;
  const int P = static_cast<int>(prompt.size());
  if (P == 0) return out;
  // A prompt at or past the context limit used to overrun the KV cache and surface as a
  // cryptic CUDA invalid-argument (found via a 2049-token prompt against the 2048
  // default). Fail with the actual reason instead.
  if (P >= model.max_context())
    throw std::runtime_error("prompt (" + std::to_string(P) + " tokens) does not fit the context (" +
                             std::to_string(model.max_context()) + "); raise --max-context");
  model.reset_state();
  if (stats) {
    *stats = BenchmarkStats{};
    stats->prompt_tokens = P;
  }
  if (params.seed >= 0) detail::dispatch_seed_sampler_rng(static_cast<unsigned>(params.seed));

  // Prefill: batched when the engine supports it (see SequenceModel::prefill).
  const auto prefill_start = clock::now();
  model.prefill(prompt);
  model.synchronize();
  if (stats) stats->prefill_ms = ms(prefill_start, clock::now());

  // history feeds the repetition penalty / no-repeat-ngram; seed it with the prompt.
  std::vector<int> history = prompt;
  const int max_ctx = model.max_context();

  const auto decode_start = clock::now();
  int pos = P;
  for (int s = 0; s < params.max_new_tokens && pos < max_ctx; ++s) {
    const int next = model.sample(params, history);
    if (params.grammar_accept) params.grammar_accept(next);
    out.push_back(next);
    history.push_back(next);
    if (stats) stats->generated_tokens++;
    if (on_token && !on_token(next)) break;
    if (model.is_stop(next)) break;
    model.step(next, pos, /*want_logits=*/true);
    ++pos;
  }
  model.synchronize();
  if (stats) stats->decode_ms = ms(decode_start, clock::now());

  if (params.include_prompt) {
    std::vector<int> full = prompt;
    full.insert(full.end(), out.begin(), out.end());
    return full;
  }
  return out;
}

}  // namespace runtime
}  // namespace engine
