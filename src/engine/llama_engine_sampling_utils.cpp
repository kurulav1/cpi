#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

#include "engine/sampling.hpp"  // shared decls; compile-time check the sampler signatures match
namespace engine {
namespace {

// Shared per-thread sampling RNG. Seeded to 42 by default for reproducible
// behaviour; reseeded via detail::dispatch_seed_sampler_rng when a request
// supplies a seed. Only the temperature>0 paths draw from it.
std::mt19937& sampler_rng() {
  thread_local std::mt19937 rng(42);
  return rng;
}

// The single sampling traversal. Every temperature > 0 path (the fast top-k path below, the
// full sample_from_logits path, and the device top-k path) turns its logits into a candidate set
// and draws through here: softmax over the candidates, top_p nucleus, one uniform draw walked in
// probability order. Having exactly one traversal is what makes those paths agree token-for-token
// (not merely in distribution) for a given seed, so which route a request takes (host vs device,
// fast vs full) never changes the sampled token. The candidate set is O(top_k), so this also
// avoids the O(vocab log vocab) full-vocabulary sort the sampler once did on every decoded token.
int sample_from_candidates(std::vector<engine::detail::SampleCandidate>& cand, float temperature,
                           float top_p) {
  using Candidate = engine::detail::SampleCandidate;
  if (cand.empty()) {
    return 0;
  }
  // Same [-80, 80] clamp the full path uses, to keep exp() well-behaved. Applied
  // to the values only; candidate selection already happened on the raw logits.
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

  // Every finite logit at or above the threshold; ties at kth included, which is
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

  // Build the same top_k candidate set sample_from_logits_topk does, but on these (sanitized,
  // penalized, n-gram-banned) logits, and delegate the softmax / nucleus / draw to the shared
  // sample_from_candidates. A single traversal is the point: the device top-k path already samples
  // through sample_from_candidates, so with an identical candidate set and seed this fallback
  // returns the identical token, not merely the same distribution. This path used to walk the
  // vocab in index order while the candidate path walks in probability order, so which route a
  // request took (decided by whether the whole batch was fast-path eligible, i.e. by other clients'
  // concurrent requests) changed the sampled token for a fixed seed, and made CPI_BATCH_TOPK=0 an
  // unreliable A/B for penalty rows. top_k selects the same set on the unscaled logits (inv_temp is
  // a positive, order-preserving scale), matching the fast path's threshold.
  std::vector<engine::detail::SampleCandidate> cand;
  if (top_k > 0 && top_k < static_cast<int>(logits.size())) {
    std::vector<float> part(logits);
    std::nth_element(part.begin(), part.begin() + (top_k - 1), part.end(), std::greater<float>());
    const float kth = part[static_cast<std::size_t>(top_k - 1)];
    for (std::size_t i = 0; i < logits.size(); ++i) {
      const float v = logits[i];
      if (std::isfinite(v) && v >= kth) cand.push_back({static_cast<int>(i), v});
    }
  } else {
    for (std::size_t i = 0; i < logits.size(); ++i) {
      const float v = logits[i];
      if (std::isfinite(v)) cand.push_back({static_cast<int>(i), v});
    }
  }
  return sample_from_candidates(cand, temperature, top_p);
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