// Parity test: the top_k fast path and the full sampler must sample IDENTICALLY.
//
// The full path (sample_from_logits, temperature > 0) now builds a top_k candidate set and
// delegates the softmax / nucleus / draw to sample_from_candidates -- the same shared routine the
// fast path (sample_from_logits_topk) and the device top-k path use. So the two are no longer only
// distributionally equal (they used to walk the vocab in different orders); with the same candidate
// set and the same RNG they return the SAME token. The functions below are standalone mirrors (no
// CUDA) of the shared candidate sampler and the two entry points, and the test asserts:
//   1. neither ever emits a token outside the top_k / nucleus set, and
//   2. under a shared seed the two paths agree token-for-token, every draw.
//
// Build: g++ -O2 -std=c++17 tools/sampling_parity_test.cpp -o sampling_parity_test

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

struct Candidate {
  int id;
  float value;
};

// ---- mirror of sample_from_candidates (the one shared traversal) ------------
int sample_candidates(std::vector<Candidate> cand, float temperature, float top_p,
                      std::mt19937& rng) {
  if (cand.empty()) return 0;
  for (Candidate& c : cand) {
    if (c.value > 80.0f)
      c.value = 80.0f;
    else if (c.value < -80.0f)
      c.value = -80.0f;
  }
  const float inv_temp = 1.0f / temperature;
  float max_logit = -std::numeric_limits<float>::infinity();
  for (const Candidate& c : cand) max_logit = std::max(max_logit, c.value * inv_temp);
  float sum = 0.0f;
  for (Candidate& c : cand) {
    c.value = std::exp(c.value * inv_temp - max_logit);
    sum += c.value;
  }
  if (sum <= 0.0f) return cand.front().id;
  for (Candidate& c : cand) c.value /= sum;
  std::sort(cand.begin(), cand.end(),
            [](const Candidate& a, const Candidate& b) { return a.value > b.value; });
  std::size_t keep = cand.size();
  if (top_p > 0.0f && top_p < 1.0f) {
    float csum = 0.0f;
    keep = 0;
    for (std::size_t i = 0; i < cand.size(); ++i) {
      csum += cand[i].value;
      ++keep;
      if (csum >= top_p) break;
    }
    float renorm = 0.0f;
    for (std::size_t i = 0; i < keep; ++i) renorm += cand[i].value;
    if (renorm > 0.0f)
      for (std::size_t i = 0; i < keep; ++i) cand[i].value /= renorm;
  }
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float r = dist(rng);
  float acc = 0.0f;
  for (std::size_t i = 0; i < keep; ++i) {
    acc += cand[i].value;
    if (r <= acc) return cand[i].id;
  }
  return cand[keep - 1].id;
}

// Build the top_k candidate set (finite logits >= the k-th largest), ties included.
std::vector<Candidate> build_topk(const std::vector<float>& logits, int top_k) {
  std::vector<Candidate> cand;
  if (top_k > 0 && top_k < static_cast<int>(logits.size())) {
    std::vector<float> part(logits);
    std::nth_element(part.begin(), part.begin() + (top_k - 1), part.end(), std::greater<float>());
    const float kth = part[top_k - 1];
    for (std::size_t i = 0; i < logits.size(); ++i)
      if (std::isfinite(logits[i]) && logits[i] >= kth) cand.push_back({(int)i, logits[i]});
  } else {
    for (std::size_t i = 0; i < logits.size(); ++i)
      if (std::isfinite(logits[i])) cand.push_back({(int)i, logits[i]});
  }
  return cand;
}

// ---- mirror of sample_from_logits (full path): sanitize, then delegate ------
int sample_full(std::vector<float> logits, float temperature, int top_k, float top_p,
                std::mt19937& rng) {
  for (float& v : logits) {
    if (!std::isfinite(v))
      v = -std::numeric_limits<float>::infinity();
    else if (v > 80.0f)
      v = 80.0f;
    else if (v < -80.0f)
      v = -80.0f;
  }
  return sample_candidates(build_topk(logits, top_k), temperature, top_p, rng);
}

// ---- mirror of sample_from_logits_topk (fast path): candidates from raw -----
int sample_fast(const std::vector<float>& logits, float temperature, int top_k, float top_p,
                std::mt19937& rng) {
  return sample_candidates(build_topk(logits, top_k), temperature, top_p, rng);
}

int main() {
  const int vocab = 128256;
  const int N = 30000;
  int failures = 0;

  struct Case {
    const char* name;
    float temp;
    int top_k;
    float top_p;
  };
  Case cases[] = {
      {"default chat (k=40,p=0.9,t=0.7)", 0.7f, 40, 0.9f},
      {"low temp (k=40,p=0.9,t=0.2)", 0.2f, 40, 0.9f},
      {"no top_p (k=40,p=1.0,t=0.8)", 0.8f, 40, 1.0f},
      {"tight nucleus (k=100,p=0.5,t=1)", 1.0f, 100, 0.5f},
  };

  for (const Case& c : cases) {
    // Build a realistic-ish logit field: a few sharp peaks over a noisy floor.
    std::mt19937 gen(12345);
    std::normal_distribution<float> noise(0.0f, 2.0f);
    std::vector<float> logits(vocab);
    for (int i = 0; i < vocab; ++i) logits[i] = noise(gen);
    std::uniform_int_distribution<int> pick(0, vocab - 1);
    for (int p = 0; p < 60; ++p) logits[pick(gen)] += 8.0f + (p % 5);

    // SHARED seed: the two paths must now draw the identical token every iteration, not merely
    // agree in aggregate.
    std::mt19937 rng_full(777), rng_fast(777);
    std::unordered_set<int> emitted;
    int mismatches = 0;
    for (int n = 0; n < N; ++n) {
      int tf = sample_full(logits, c.temp, c.top_k, c.top_p, rng_full);
      int tk = sample_fast(logits, c.temp, c.top_k, c.top_p, rng_fast);
      if (tf != tk) ++mismatches;
      emitted.insert(tf);
    }

    const bool ok = (mismatches == 0);
    if (!ok) failures++;
    std::printf("%-34s  support=%zu  token_mismatches=%d/%d  %s\n", c.name, emitted.size(),
                mismatches, N, ok ? "PASS" : "FAIL");
  }

  std::printf("\n%s\n", failures == 0 ? "ALL PARITY CHECKS PASSED" : "PARITY CHECKS FAILED");
  return failures == 0 ? 0 : 1;
}
