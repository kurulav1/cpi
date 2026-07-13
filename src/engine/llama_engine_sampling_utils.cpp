#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

#include "engine/sampling.hpp"  // shared decls — compile-time check the sampler signatures match
namespace engine {
namespace {

// Shared per-thread sampling RNG. Seeded to 42 by default for reproducible
// behaviour; reseeded via detail::dispatch_seed_sampler_rng when a request
// supplies a seed. Only the temperature>0 paths draw from it.
std::mt19937& sampler_rng() {
  thread_local std::mt19937 rng(42);
  return rng;
}

// Fast sampling path for the common chat configuration (temperature > 0,
// 0 < top_k < vocab, no repetition penalty, no n-gram blocking).
//
// The full sampler below scales, softmaxes and — for top_p — std::sorts the
// entire vocabulary (~128k entries) on every decoded token, even though only
// the top_k (default 40) candidates can ever be chosen. This collapses that to
// the top_k set first, so the per-token work drops from an O(vocab log vocab)
// sort to an O(top_k log top_k) one. The resulting distribution is identical to
// the full path: top_k keeps exactly the entries >= the k-th largest logit
// (ties included), and the nucleus for top_p is always a subset of that set.
// Draws a token from an already-selected candidate set. Factored out of the top-k
// sampler so the device-side top-k path can reuse the identical math + RNG: given
// the same candidates it returns the same token (see sampling.hpp).
int sample_from_candidates(std::vector<engine::detail::SampleCandidate>& cand, float temperature,
                           float top_p) {
  using Candidate = engine::detail::SampleCandidate;
  if (cand.empty()) {
    return 0;
  }
  // Same [-80, 80] clamp the full path uses, to keep exp() well-behaved. Applied
  // to the VALUES only — candidate selection already happened on the raw logits.
  for (Candidate& c : cand) {
    if (c.value > 80.0f)
      c.value = 80.0f;
    else if (c.value < -80.0f)
      c.value = -80.0f;
  }

  const float inv_temp = 1.0f / temperature;
  float max_logit = -std::numeric_limits<float>::infinity();
  for (const Candidate& c : cand) {
    max_logit = std::max(max_logit, c.value * inv_temp);
  }

  float sum = 0.0f;
  for (Candidate& c : cand) {
    c.value = std::exp(c.value * inv_temp - max_logit);
    sum += c.value;
  }
  if (sum <= 0.0f) {
    return cand.front().id;
  }
  for (Candidate& c : cand) {
    c.value /= sum;
  }

  // Sort by probability descending; used for both nucleus truncation and a
  // stable, high-mass-first sampling traversal.
  std::sort(cand.begin(), cand.end(),
            [](const Candidate& a, const Candidate& b) { return a.value > b.value; });

  std::size_t keep = cand.size();
  if (top_p > 0.0f && top_p < 1.0f) {
    float csum = 0.0f;
    keep = 0;
    for (std::size_t i = 0; i < cand.size(); ++i) {
      csum += cand[i].value;
      ++keep;
      if (csum >= top_p) {
        break;
      }
    }
    float renorm = 0.0f;
    for (std::size_t i = 0; i < keep; ++i) {
      renorm += cand[i].value;
    }
    if (renorm > 0.0f) {
      for (std::size_t i = 0; i < keep; ++i) {
        cand[i].value /= renorm;
      }
    }
  }

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float r = dist(sampler_rng());

  float acc = 0.0f;
  for (std::size_t i = 0; i < keep; ++i) {
    acc += cand[i].value;
    if (r <= acc) {
      return cand[i].id;
    }
  }
  return cand[keep - 1].id;
}

int sample_from_logits_topk(const std::vector<float>& logits, float temperature, int top_k,
                            float top_p) {
  // Threshold = the k-th largest logit. nth_element is O(vocab) (no full sort).
  std::vector<float> partitioned(logits);
  std::nth_element(partitioned.begin(), partitioned.begin() + (top_k - 1), partitioned.end(),
                   std::greater<float>());
  const float kth = partitioned[static_cast<std::size_t>(top_k - 1)];

  // Every finite logit at or above the threshold — ties at kth included, which is
  // why this can yield more than top_k candidates (the device path mirrors this).
  std::vector<engine::detail::SampleCandidate> cand;
  cand.reserve(static_cast<std::size_t>(top_k) + 8);
  for (std::size_t i = 0; i < logits.size(); ++i) {
    const float v = logits[i];
    if (!std::isfinite(v) || v < kth) {
      continue;
    }
    cand.push_back({static_cast<int>(i), v});
  }
  return sample_from_candidates(cand, temperature, top_p);
}

int sample_from_logits(std::vector<float>& logits, float temperature, int top_k, float top_p,
                       float repetition_penalty, int no_repeat_ngram_size,
                       const std::vector<int>& history) {
  if (logits.empty()) {
    return 0;
  }

  // Common chat path: reduce to the top_k candidate set before scaling/sorting.
  // Equivalent in distribution to the full path below, but avoids the
  // full-vocabulary softmax + sort on every token. Only valid when the
  // full-vocabulary features (repetition penalty, n-gram blocking) are off.
  if (temperature > 0.0f && top_k > 0 && top_k < static_cast<int>(logits.size()) &&
      repetition_penalty <= 1.0f && no_repeat_ngram_size <= 1) {
    return sample_from_logits_topk(logits, temperature, top_k, top_p);
  }

  /*
   * Defensive sanitization: fp16 decode paths can occasionally emit non-finite
   * logits; keep sampling stable by dropping invalid values and clamping range.
   */
  for (float& v : logits) {
    if (!std::isfinite(v)) {
      v = -std::numeric_limits<float>::infinity();
      continue;
    }
    if (v > 80.0f) {
      v = 80.0f;
    } else if (v < -80.0f) {
      v = -80.0f;
    }
  }

  if (repetition_penalty > 1.0f && !history.empty()) {
    std::unordered_set<int> seen(history.begin(), history.end());
    for (int id : seen) {
      if (id < 0 || id >= static_cast<int>(logits.size())) {
        continue;
      }
      if (logits[id] > 0.0f) {
        logits[id] /= repetition_penalty;
      } else {
        logits[id] *= repetition_penalty;
      }
    }
  }

  /*
   * No-repeat n-gram blocking:
   * If the current (n-1)-token suffix has appeared before, ban tokens that
   * previously followed that suffix.
   */
  if (no_repeat_ngram_size > 1 &&
      history.size() + 1 >= static_cast<std::size_t>(no_repeat_ngram_size)) {
    const int n = no_repeat_ngram_size;
    const int prefix_len = n - 1;
    const int hist_size = static_cast<int>(history.size());
    std::vector<int> prefix(prefix_len);
    for (int i = 0; i < prefix_len; ++i) {
      prefix[static_cast<std::size_t>(i)] =
          history[static_cast<std::size_t>(hist_size - prefix_len + i)];
    }

    std::vector<char> banned(logits.size(), 0);
    for (int i = 0; i + n <= hist_size; ++i) {
      bool match = true;
      for (int j = 0; j < prefix_len; ++j) {
        if (history[static_cast<std::size_t>(i + j)] != prefix[static_cast<std::size_t>(j)]) {
          match = false;
          break;
        }
      }
      if (match) {
        const int next_id = history[static_cast<std::size_t>(i + prefix_len)];
        if (next_id >= 0 && next_id < static_cast<int>(banned.size())) {
          banned[static_cast<std::size_t>(next_id)] = 1;
        }
      }
    }

    bool has_candidate = false;
    for (std::size_t i = 0; i < logits.size(); ++i) {
      if (banned[i]) {
        logits[i] = -std::numeric_limits<float>::infinity();
      } else if (std::isfinite(logits[i])) {
        has_candidate = true;
      }
    }
    if (!has_candidate) {
      return 0;
    }
  }

  if (temperature <= 0.0f) {
    return static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
  }

  const float inv_temp = 1.0f / temperature;
  float max_logit = -std::numeric_limits<float>::infinity();
  for (float v : logits) {
    if (std::isfinite(v)) {
      max_logit = std::max(max_logit, v * inv_temp);
    }
  }
  if (!std::isfinite(max_logit)) {
    return 0;
  }

  float sum = 0.0f;
  for (float& v : logits) {
    v = v * inv_temp;
  }

  if (top_k > 0 && top_k < static_cast<int>(logits.size())) {
    std::vector<float> copy = logits;
    std::nth_element(copy.begin(), copy.begin() + (top_k - 1), copy.end(), std::greater<float>());
    const float kth = copy[top_k - 1];
    for (float& v : logits) {
      if (v < kth) {
        v = -std::numeric_limits<float>::infinity();
      }
    }
  }

  std::vector<float> probs(logits.size(), 0.0f);
  sum = 0.0f;
  for (std::size_t i = 0; i < logits.size(); ++i) {
    if (!std::isfinite(logits[i])) {
      continue;
    }
    probs[i] = std::exp(logits[i] - max_logit);
    sum += probs[i];
  }
  if (sum <= 0.0f) {
    return 0;
  }
  for (float& p : probs) {
    p /= sum;
  }

  if (top_p > 0.0f && top_p < 1.0f) {
    std::vector<int> idx(probs.size());
    for (std::size_t i = 0; i < idx.size(); ++i) {
      idx[i] = static_cast<int>(i);
    }
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });

    float csum = 0.0f;
    std::vector<char> keep(probs.size(), 0);
    for (int id : idx) {
      if (probs[id] <= 0.0f) {
        continue;
      }
      keep[id] = 1;
      csum += probs[id];
      if (csum >= top_p) {
        break;
      }
    }

    float renorm = 0.0f;
    for (std::size_t i = 0; i < probs.size(); ++i) {
      if (!keep[i]) {
        probs[i] = 0.0f;
      }
      renorm += probs[i];
    }
    if (renorm > 0.0f) {
      for (float& p : probs) {
        p /= renorm;
      }
    }
  }

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float r = dist(sampler_rng());

  float acc = 0.0f;
  for (std::size_t i = 0; i < probs.size(); ++i) {
    acc += probs[i];
    if (r <= acc) {
      return static_cast<int>(i);
    }
  }

  return static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
}

bool has_degenerate_tail(const std::vector<int>& ids, std::size_t prompt_size) {
  if (ids.size() <= prompt_size + 8) {
    return false;
  }
  const std::size_t gen_size = ids.size() - prompt_size;

  // Hard repeat of a single token in the recent tail.
  if (gen_size >= 6) {
    const int t = ids[ids.size() - 1];
    bool all_same = true;
    for (std::size_t i = 0; i < 6; ++i) {
      if (ids[ids.size() - 1 - i] != t) {
        all_same = false;
        break;
      }
    }
    if (all_same) {
      return true;
    }
  }

  // Repeated 2-token cycle, e.g. A B A B A B.
  if (gen_size >= 12) {
    bool cycle2 = true;
    for (std::size_t i = 0; i < 6; ++i) {
      if (ids[ids.size() - 1 - i] != ids[ids.size() - 1 - i - 2]) {
        cycle2 = false;
        break;
      }
    }
    if (cycle2) {
      return true;
    }
  }

  // Repeated 4-token phrase copied immediately twice.
  if (gen_size >= 16) {
    bool cycle4 = true;
    for (std::size_t i = 0; i < 4; ++i) {
      if (ids[ids.size() - 1 - i] != ids[ids.size() - 1 - i - 4]) {
        cycle4 = false;
        break;
      }
    }
    if (cycle4) {
      return true;
    }
  }

  return false;
}
}  // namespace
namespace detail {
int dispatch_sample_from_logits(std::vector<float>& logits, float temperature, int top_k,
                                float top_p, float repetition_penalty, int no_repeat_ngram_size,
                                const std::vector<int>& history) {
  return sample_from_logits(logits, temperature, top_k, top_p, repetition_penalty,
                            no_repeat_ngram_size, history);
}

bool dispatch_has_degenerate_tail(const std::vector<int>& ids, std::size_t prompt_size) {
  return has_degenerate_tail(ids, prompt_size);
}

void dispatch_seed_sampler_rng(unsigned seed) {
  sampler_rng().seed(seed);
}

int dispatch_sample_from_candidates(std::vector<SampleCandidate>& cand, float temperature,
                                    float top_p) {
  return sample_from_candidates(cand, temperature, top_p);
}
}  // namespace detail
}  // namespace engine