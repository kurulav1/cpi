#include "engine/decode_driver.hpp"

#include "engine/sampling.hpp"

namespace engine {
namespace runtime {

std::vector<int> run_decode(SequenceModel& model, const std::vector<int>& prompt,
                            const DecodeParams& params,
                            const std::function<bool(int)>& on_token, BenchmarkStats* stats) {
  std::vector<int> out;
  const int P = static_cast<int>(prompt.size());
  if (P == 0) return out;
  if (stats) {
    *stats = BenchmarkStats{};
    stats->prompt_tokens = P;
  }
  if (params.seed >= 0) detail::dispatch_seed_sampler_rng(static_cast<unsigned>(params.seed));

  // Prefill: run every prompt token; only the last computes logits.
  for (int i = 0; i < P; ++i) model.step(prompt[i], i, i == P - 1);

  // history feeds the repetition penalty / no-repeat-ngram; seed it with the prompt.
  std::vector<int> history = prompt;
  const int max_ctx = model.max_context();

  int pos = P;
  for (int s = 0; s < params.max_new_tokens && pos < max_ctx; ++s) {
    const int next = detail::dispatch_sample_from_logits(
        model.logits(), params.temperature, params.top_k, params.top_p,
        params.repetition_penalty, params.no_repeat_ngram_size, history);
    out.push_back(next);
    history.push_back(next);
    if (stats) stats->generated_tokens++;
    if (on_token && !on_token(next)) break;
    if (next == model.eos_id()) break;
    model.step(next, pos, /*want_logits=*/true);
    ++pos;
  }
  return out;
}

}  // namespace runtime
}  // namespace engine
