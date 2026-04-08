#include "llama_engine_internal.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>
namespace engine {
namespace {
int sample_from_logits(std::vector<float>& logits,
                       float temperature,
                       int top_k,
                       float top_p,
                       float repetition_penalty,
                       int no_repeat_ngram_size,
                       const std::vector<int>& history) {
  if (logits.empty()) {
    return 0;
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
  if (no_repeat_ngram_size > 1 && history.size() + 1 >= static_cast<std::size_t>(no_repeat_ngram_size)) {
    const int n = no_repeat_ngram_size;
    const int prefix_len = n - 1;
    const int hist_size = static_cast<int>(history.size());
    std::vector<int> prefix(prefix_len);
    for (int i = 0; i < prefix_len; ++i) {
      prefix[static_cast<std::size_t>(i)] = history[static_cast<std::size_t>(hist_size - prefix_len + i)];
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

  thread_local std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float r = dist(rng);

  float acc = 0.0f;
  for (std::size_t i = 0; i < probs.size(); ++i) {
    acc += probs[i];
    if (r <= acc) {
      return static_cast<int>(i);
    }
  }

  return static_cast<int>(std::max_element(probs.begin(), probs.end()) - probs.begin());
}

bool has_degenerate_tail(const std::vector<int>& ids,
                         std::size_t prompt_size) {
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
int dispatch_sample_from_logits(std::vector<float>& logits,
                                float temperature,
                                int top_k,
                                float top_p,
                                float repetition_penalty,
                                int no_repeat_ngram_size,
                                const std::vector<int>& history) {
  return sample_from_logits(logits,
                            temperature,
                            top_k,
                            top_p,
                            repetition_penalty,
                            no_repeat_ngram_size,
                            history);
}

bool dispatch_has_degenerate_tail(const std::vector<int>& ids, std::size_t prompt_size) {
  return has_degenerate_tail(ids, prompt_size);
}
}  // namespace detail
}  // namespace engine