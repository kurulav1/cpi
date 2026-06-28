// Statistical parity test: the new top_k fast path must sample from the same
// distribution as the original full-vocabulary sampler.
//
// The two functions below are verbatim mirrors of sample_from_logits (full,
// pre-existing) and sample_from_logits_topk (new) in
// src/engine/llama_engine_sampling_utils.cpp. This test reimplements them
// standalone (no CUDA) so it can run anywhere, then checks that:
//   1. both only ever emit tokens from the expected top_k / nucleus set, and
//   2. their empirical sampling frequencies agree within statistical noise.
//
// Build: g++ -O2 -std=c++17 tools/sampling_parity_test.cpp -o sampling_parity_test

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

// ---- mirror of the original full-vocabulary sampler (top_k/top_p only) ------
int sample_full(std::vector<float> logits, float temperature, int top_k, float top_p,
                std::mt19937& rng) {
  for (float& v : logits) {
    if (!std::isfinite(v)) { v = -std::numeric_limits<float>::infinity(); continue; }
    if (v > 80.0f) v = 80.0f; else if (v < -80.0f) v = -80.0f;
  }
  const float inv_temp = 1.0f / temperature;
  float max_logit = -std::numeric_limits<float>::infinity();
  for (float v : logits) if (std::isfinite(v)) max_logit = std::max(max_logit, v * inv_temp);
  for (float& v : logits) v = v * inv_temp;

  if (top_k > 0 && top_k < (int)logits.size()) {
    std::vector<float> copy = logits;
    std::nth_element(copy.begin(), copy.begin() + (top_k - 1), copy.end(), std::greater<float>());
    const float kth = copy[top_k - 1];
    for (float& v : logits) if (v < kth) v = -std::numeric_limits<float>::infinity();
  }
  std::vector<float> probs(logits.size(), 0.0f);
  float sum = 0.0f;
  for (size_t i = 0; i < logits.size(); ++i) {
    if (!std::isfinite(logits[i])) continue;
    probs[i] = std::exp(logits[i] - max_logit);
    sum += probs[i];
  }
  for (float& p : probs) p /= sum;

  if (top_p > 0.0f && top_p < 1.0f) {
    std::vector<int> idx(probs.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int)i;
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[a] > probs[b]; });
    float csum = 0.0f;
    std::vector<char> keep(probs.size(), 0);
    for (int id : idx) {
      if (probs[id] <= 0.0f) continue;
      keep[id] = 1; csum += probs[id];
      if (csum >= top_p) break;
    }
    float renorm = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) { if (!keep[i]) probs[i] = 0.0f; renorm += probs[i]; }
    if (renorm > 0.0f) for (float& p : probs) p /= renorm;
  }
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float r = dist(rng);
  float acc = 0.0f;
  for (size_t i = 0; i < probs.size(); ++i) { acc += probs[i]; if (r <= acc) return (int)i; }
  return (int)(std::max_element(probs.begin(), probs.end()) - probs.begin());
}

// ---- mirror of the new fast top_k sampler ---------------------------------
int sample_fast(const std::vector<float>& logits, float temperature, int top_k, float top_p,
                std::mt19937& rng) {
  struct Candidate { int id; float value; };
  std::vector<float> partitioned(logits);
  std::nth_element(partitioned.begin(), partitioned.begin() + (top_k - 1), partitioned.end(),
                   std::greater<float>());
  const float kth = partitioned[top_k - 1];
  std::vector<Candidate> cand; cand.reserve(top_k + 8);
  for (size_t i = 0; i < logits.size(); ++i) {
    float v = logits[i];
    if (!std::isfinite(v) || v < kth) continue;
    if (v > 80.0f) v = 80.0f; else if (v < -80.0f) v = -80.0f;
    cand.push_back({(int)i, v});
  }
  if (cand.empty()) return 0;
  const float inv_temp = 1.0f / temperature;
  float max_logit = -std::numeric_limits<float>::infinity();
  for (auto& c : cand) max_logit = std::max(max_logit, c.value * inv_temp);
  float sum = 0.0f;
  for (auto& c : cand) { c.value = std::exp(c.value * inv_temp - max_logit); sum += c.value; }
  for (auto& c : cand) c.value /= sum;
  std::sort(cand.begin(), cand.end(), [](const Candidate& a, const Candidate& b) { return a.value > b.value; });
  size_t keep = cand.size();
  if (top_p > 0.0f && top_p < 1.0f) {
    float csum = 0.0f; keep = 0;
    for (size_t i = 0; i < cand.size(); ++i) { csum += cand[i].value; ++keep; if (csum >= top_p) break; }
    float renorm = 0.0f;
    for (size_t i = 0; i < keep; ++i) renorm += cand[i].value;
    if (renorm > 0.0f) for (size_t i = 0; i < keep; ++i) cand[i].value /= renorm;
  }
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  const float r = dist(rng);
  float acc = 0.0f;
  for (size_t i = 0; i < keep; ++i) { acc += cand[i].value; if (r <= acc) return cand[i].id; }
  return cand[keep - 1].id;
}

int main() {
  const int vocab = 128256;
  const int N = 30000;
  int failures = 0;

  struct Case { const char* name; float temp; int top_k; float top_p; };
  Case cases[] = {
    {"default chat (k=40,p=0.9,t=0.7)", 0.7f, 40, 0.9f},
    {"low temp (k=40,p=0.9,t=0.2)",     0.2f, 40, 0.9f},
    {"no top_p (k=40,p=1.0,t=0.8)",     0.8f, 40, 1.0f},
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

    std::mt19937 rng_full(777), rng_fast(777);
    std::unordered_set<int> full_tokens, fast_tokens;
    std::vector<int> freq_full(vocab, 0), freq_fast(vocab, 0);
    for (int n = 0; n < N; ++n) {
      int tf = sample_full(logits, c.temp, c.top_k, c.top_p, rng_full);
      int tk = sample_fast(logits, c.temp, c.top_k, c.top_p, rng_fast);
      freq_full[tf]++; freq_fast[tk]++;
      full_tokens.insert(tf); fast_tokens.insert(tk);
    }

    // The fast path must never emit a token the full path never emits.
    int out_of_set = 0;
    for (int t : fast_tokens) if (!full_tokens.count(t)) out_of_set++;

    // Total-variation distance between the two empirical distributions.
    double tv = 0.0;
    for (int i = 0; i < vocab; ++i)
      tv += std::abs((double)freq_full[i] - freq_fast[i]);
    tv /= (2.0 * N);

    const bool ok = (out_of_set == 0) && (tv < 0.01);
    if (!ok) failures++;
    std::printf("%-34s  support_full=%zu support_fast=%zu out_of_set=%d  TV=%.5f  %s\n",
                c.name, full_tokens.size(), fast_tokens.size(), out_of_set, tv,
                ok ? "PASS" : "FAIL");
  }

  std::printf("\n%s\n", failures == 0 ? "ALL PARITY CHECKS PASSED" : "PARITY CHECKS FAILED");
  return failures == 0 ? 0 : 1;
}
