#include "engine/speculative_decoder.hpp"

#include <algorithm>

#include "engine/llama_engine.hpp"

namespace engine {

SpeculativeDecoder::SpeculativeDecoder(LlamaEngine& draft, LlamaEngine& target, int spec_tokens)
    : draft_(draft), target_(target), k_(std::max(1, spec_tokens)) {}

std::vector<int> SpeculativeDecoder::generate(const std::vector<int>& prompt_tokens,
                                              int max_new_tokens, int eos_token_id,
                                              const std::function<bool(int)>& on_token) {
  // Return the full sequence (prompt + generated), matching generate_stream's
  // contract: callers strip the first prompt_tokens.size() entries.
  std::vector<int> out(prompt_tokens.begin(), prompt_tokens.end());
  stats_ = SpeculativeStats{};
  if (prompt_tokens.empty() || max_new_tokens <= 0) {
    return out;
  }
  out.reserve(prompt_tokens.size() + static_cast<std::size_t>(max_new_tokens));
  int generated = 0;

  // Prime both caches with the prompt. prefill_prompt processes the first P-1
  // tokens (positions [0, P-2]); the last prompt token is consumed at position
  // P-1 by the first decode/verify step, exactly mirroring generate_stream.
  draft_.reset_kv_cache();
  target_.reset_kv_cache();
  draft_.prefill_prompt(prompt_tokens);
  target_.prefill_prompt(prompt_tokens);

  int pos = static_cast<int>(prompt_tokens.size()) - 1;  // position of `last`
  int last = prompt_tokens.back();                       // token consumed at `pos`
  bool stop = false;

  const std::vector<int> empty_history;  // pure greedy: no repetition history

  auto emit = [&](int token) -> bool {
    out.push_back(token);
    ++generated;
    stats_.emitted += 1;
    if (on_token && !on_token(token)) {
      return false;  // caller requested stop
    }
    if (eos_token_id >= 0 && token == eos_token_id) {
      return false;
    }
    return generated < max_new_tokens;
  };

  while (!stop && generated < max_new_tokens) {
    const int K = k_;

    // 1. Draft K tokens greedily. Consumes last@pos, draft[0]@pos+1, ...,
    //    draft[K-2]@pos+K-1, producing draft[0..K-1].
    std::vector<int> draft_tokens;
    draft_tokens.reserve(static_cast<std::size_t>(K));
    int cur = last;
    int dpos = pos;
    for (int j = 0; j < K; ++j) {
      const int t = draft_.decode_next_token(cur, dpos, 0.0f, empty_history);
      draft_tokens.push_back(t);
      cur = t;
      ++dpos;
    }

    // 2. Build the verify input: verify_in[0]=last, verify_in[i]=draft[i-1].
    //    targ[i] (target argmax after consuming verify_in[i]) is the target's
    //    token for position pos+i+1, i.e. the check for draft[i].
    std::vector<int> verify_in(static_cast<std::size_t>(K));
    verify_in[0] = last;
    for (int i = 1; i < K; ++i) {
      verify_in[static_cast<std::size_t>(i)] = draft_tokens[static_cast<std::size_t>(i - 1)];
    }

    std::vector<int> targ;
    target_.verify_tokens(verify_in, pos, targ);

    // 3. Accept the longest prefix where the draft matched the target argmax.
    int n_accept = 0;
    while (n_accept < K && targ[static_cast<std::size_t>(n_accept)] ==
                               draft_tokens[static_cast<std::size_t>(n_accept)]) {
      ++n_accept;
    }

    stats_.rounds += 1;
    stats_.drafted += K;
    stats_.accepted += n_accept;

    // 4. Emit accepted drafts, then advance the frontier.
    for (int i = 0; i < n_accept; ++i) {
      if (!emit(draft_tokens[static_cast<std::size_t>(i)])) {
        stop = true;
        break;
      }
    }
    if (stop) {
      break;
    }

    if (n_accept < K) {
      // First mismatch: targ[n_accept] is the target's own (correct) token for
      // position pos+n_accept+1. Emit it as the correction; its K/V is written
      // next round when it is consumed as `last`.
      const int correction = targ[static_cast<std::size_t>(n_accept)];
      if (!emit(correction)) {
        break;
      }
      last = correction;
      pos += n_accept + 1;
    } else {
      // All K drafts accepted. draft[K-1] is committed but its K/V is not yet in
      // the cache (verify consumed up to draft[K-2]); it becomes `last` and is
      // consumed next round.
      last = draft_tokens[static_cast<std::size_t>(K - 1)];
      pos += K;
    }
  }

  return out;
}

}  // namespace engine
