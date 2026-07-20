#pragma once

#include <algorithm>
#include <cstdlib>
#include <functional>
#include <vector>

namespace engine {

// Aggregate stats for a speculative-decoding run (for logging / benchmarking).
struct SpeculativeStats {
  int rounds = 0;    // verify rounds (= target forward passes)
  int drafted = 0;   // draft tokens proposed
  int accepted = 0;  // draft tokens accepted (excludes correction/bonus tokens)
  int emitted = 0;   // total tokens emitted
  // Tree-speculation opportunity probe: rounds that rejected before K, and of
  // those, how many where the draft's SECOND choice at the rejection position
  // equalled the target's token (a width-2 tree would have recovered there).
  int mismatch_rounds = 0;
  int tree2_recoveries = 0;

  double accept_rate() const {
    return drafted > 0 ? static_cast<double>(accepted) / drafted : 0.0;
  }
  double tokens_per_round() const {
    return rounds > 0 ? static_cast<double>(emitted) / rounds : 0.0;
  }
  double tree2_recovery_rate() const {
    return mismatch_rounds > 0 ? static_cast<double>(tree2_recoveries) / mismatch_rounds : 0.0;
  }
};

// Greedy speculative decoding over a draft + target engine pair that share a
// tokenizer/vocabulary. Lossless w.r.t. pure-greedy (temperature 0, no
// repetition penalty) decoding of the target: the target's argmax is always the
// emitted token, the draft only proposes candidates the target then confirms.
//
// A TEMPLATE over the engine type rather than a class bound to one engine: the algorithm is
// backend-free, and both LlamaEngine (CUDA) and PlanMetalEngine expose the same five methods it
// needs -- reset_kv_cache, prefill_prompt, decode_next_token, decode_next_token2, verify_tokens.
// Class template argument deduction makes `SpeculativeDecoder spec(draft, target, k)` pick the
// engine type from its arguments, so callers read identically on either backend. Header-only for
// the same reason: it now compiles into the Metal build, which has no speculative_decoder.cpp.
template <typename EngineT>
class SpeculativeDecoder {
public:
  // `draft` and `target` must already be initialized with the same tokenizer
  // and a compatible max_context. `spec_tokens` (K) is the number of tokens the
  // draft proposes per round; clamped to >= 1.
  SpeculativeDecoder(EngineT& draft, EngineT& target, int spec_tokens)
      : draft_(draft), target_(target), k_(std::max(1, spec_tokens)) {}

  // Generates up to `max_new_tokens` greedily. `on_token` is invoked for each
  // emitted token id; returning false stops generation. Returns the generated
  // token ids (excluding the prompt). Stops on `eos_token_id` (>= 0).
  std::vector<int> generate(const std::vector<int>& prompt_tokens, int max_new_tokens,
                            int eos_token_id, const std::function<bool(int)>& on_token) {
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
    const bool tree_probe_ = std::getenv("LLAMA_INFER_SPEC_TREE_PROBE") != nullptr;
    const std::vector<int> empty_history_;  // pure greedy: no repetition history

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

      // 1. Draft K tokens greedily. When the tree-opportunity probe is on
      //    (LLAMA_INFER_SPEC_TREE_PROBE), also record each position's SECOND choice
      //    via the slower host-logits path; production uses the fast argmax graph.
      std::vector<int> draft_tokens;
      std::vector<int> draft2;
      draft_tokens.reserve(static_cast<std::size_t>(K));
      int cur = last;
      int dpos = pos;
      for (int j = 0; j < K; ++j) {
        int t;
        if (tree_probe_) {
          int second = 0;
          t = draft_.decode_next_token2(cur, dpos, &second);
          draft2.push_back(second);
        } else {
          t = draft_.decode_next_token(cur, dpos, 0.0f, empty_history_);
        }
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

      // Tree-opportunity probe: at the rejection position, would the draft's second
      // choice have matched the target's token (a width-2 tree recovers there)?
      if (tree_probe_ && n_accept < K) {
        ++stats_.mismatch_rounds;
        if (draft2[static_cast<std::size_t>(n_accept)] ==
            targ[static_cast<std::size_t>(n_accept)]) {
          ++stats_.tree2_recoveries;
        }
      }

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

  const SpeculativeStats& stats() const {
    return stats_;
  }

private:
  EngineT& draft_;
  EngineT& target_;
  int k_;
  SpeculativeStats stats_;
};

}  // namespace engine
