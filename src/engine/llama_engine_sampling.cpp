#include <limits>

#include "engine/llama_engine.hpp"
#include "grammar/grammar_sampler.hpp"
#include "llama_engine_internal.hpp"

namespace engine {

int LlamaEngine::decode_next_token(int token, int position, float temperature,
                                   const std::vector<int>& history) {
  // Host-logits path is required whenever we must edit the logits before
  // sampling: a grammar mask, or EOS suppression for min_new_tokens. The greedy
  // fast-path and the greedy CUDA graph both argmax on-device and never surface
  // host logits, so we force the host-logits path when either is active.
  const bool need_logit_edit = active_grammar_ != nullptr || suppress_eos_;
  if (need_logit_edit) {
    std::vector<float> h_logits;
    if (can_use_greedy_decode_graph()) {
      decode_next_token_logits_graph(token, position, h_logits);
    } else {
      forward_token_logits(token, position, &h_logits, nullptr);
    }
    if (active_grammar_ != nullptr) {
      active_grammar_->apply_mask(h_logits);
    }
    // Block EOS so greedy/sampling cannot terminate before min_new_tokens. Skip
    // when a grammar is active: a completed grammar may permit ONLY EOS, and also
    // masking it would leave no legal token (deadlock). Grammar manages its own
    // termination, so it takes precedence over min_new_tokens.
    if (suppress_eos_ && active_grammar_ == nullptr && options_.eos_token_id >= 0 &&
        static_cast<std::size_t>(options_.eos_token_id) < h_logits.size()) {
      h_logits[static_cast<std::size_t>(options_.eos_token_id)] =
          -std::numeric_limits<float>::infinity();
    }
    return detail::dispatch_sample_from_logits(h_logits, temperature, options_.top_k,
                                               options_.top_p, options_.repetition_penalty,
                                               options_.no_repeat_ngram_size, history);
  }

  const bool greedy_fast_path = temperature <= 0.0f && options_.repetition_penalty <= 1.0f &&
                                options_.no_repeat_ngram_size <= 1;
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
  return detail::dispatch_sample_from_logits(h_logits, temperature, options_.top_k, options_.top_p,
                                             options_.repetition_penalty,
                                             options_.no_repeat_ngram_size, history);
}

}  // namespace engine
