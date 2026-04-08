#include "engine/llama_engine.hpp"
#include "llama_engine_internal.hpp"

namespace engine {

int LlamaEngine::decode_next_token(int token,
                                   int position,
                                   float temperature,
                                   const std::vector<int>& history) {
  const bool greedy_fast_path =
      temperature <= 0.0f && options_.repetition_penalty <= 1.0f && options_.no_repeat_ngram_size <= 1;
  if (greedy_fast_path) {
    if (can_use_greedy_decode_graph()) {
      return decode_next_token_graph(token, position);
    }
    int next = 0;
    forward_token_logits(token, position, nullptr, &next);
    return next;
  }

  std::vector<float> h_logits;
  if (can_use_greedy_decode_graph()) {
    decode_next_token_logits_graph(token, position, h_logits);
  } else {
    forward_token_logits(token, position, &h_logits, nullptr);
  }
  return detail::dispatch_sample_from_logits(
      h_logits,
      temperature,
      options_.top_k,
      options_.top_p,
      options_.repetition_penalty,
      options_.no_repeat_ngram_size,
      history);
}


}  // namespace engine
