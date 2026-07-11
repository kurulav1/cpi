// Shared token sampler — the canonical logits->token routine (temperature,
// top-k, top-p, repetition penalty, no-repeat-ngram), defined once in
// llama_engine_sampling_utils.cpp. Exposed here so every engine (LlamaEngine and
// the fork engines) samples through one implementation instead of copying it
// (llama4 previously carried its own copy; Gemma4 had none and only did argmax).
#pragma once

#include <cstddef>
#include <vector>

namespace engine {
namespace detail {

// Returns the sampled token id. temperature <= 0 is greedy (argmax); the other
// knobs shape the temperature>0 distribution. `history` feeds the repetition
// penalty / no-repeat-ngram checks. `logits` may be modified in place.
int dispatch_sample_from_logits(std::vector<float>& logits, float temperature, int top_k,
                                float top_p, float repetition_penalty, int no_repeat_ngram_size,
                                const std::vector<int>& history);

// True when the tail of `ids` past `prompt_size` has collapsed into a repeated
// run (a loop guard for greedy decoding).
bool dispatch_has_degenerate_tail(const std::vector<int>& ids, std::size_t prompt_size);

// Reseeds the shared sampling RNG for reproducible temperature>0 sampling.
void dispatch_seed_sampler_rng(unsigned seed);

// One candidate token: its id and (in turn) its logit, then its probability.
struct SampleCandidate {
  int id;
  float value;
};

// Draws a token from an already-selected candidate set (temperature scaling →
// softmax → top-p truncation → RNG draw). This is the second half of the top-k
// sampler, factored out so a device-side top-k path can reuse the EXACT same math
// and RNG: given the same candidates it returns the same token, so a seeded run is
// byte-identical whether the candidates were selected on host or on GPU.
// `cand` is modified in place; values are clamped to [-80, 80] internally.
int dispatch_sample_from_candidates(std::vector<SampleCandidate>& cand, float temperature,
                                    float top_p);

}  // namespace detail
}  // namespace engine
