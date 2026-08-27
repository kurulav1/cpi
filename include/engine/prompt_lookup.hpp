#pragma once

// Prompt-lookup drafting: propose the continuation that followed the last time
// this n-gram appeared. No draft model and no extra weights, so a miss costs only
// the verify it triggered.
//
// Shared rather than copied. PlanMetalEngine and PlanCudaEngine each carried an
// identical private copy, which meant the tuning below had to be rediscovered per
// backend and could drift silently between them.

#include <vector>

namespace engine {

// Scans backwards for the most recent earlier occurrence of the trailing `ng`
// tokens of `hist` and writes up to `k` following tokens into `out`, returning
// how many were written (0 when there is no match).
//
// `ng` wants to be larger than it looks. A 3-gram matches spuriously on common
// short sequences, and a wrong draft costs a full verify, which is several decode
// steps. Metal measured 6 as the point where precision starts paying: it moved
// Gemma from a 17% regression to neutral while leaving Qwen's 1.76x untouched.
// When a miss is this expensive, precision beats recall.
//
// `out` must have room for `k` ints.
inline int prompt_lookup_draft(const std::vector<int>& hist, int ng, int k, int* out) {
  const int n = static_cast<int>(hist.size());
  if (n < ng + 1 || k <= 0) return 0;
  for (int start = n - ng - 1; start >= 0; --start) {
    bool match = true;
    for (int j = 0; j < ng; ++j) {
      if (hist[start + j] != hist[n - ng + j]) {
        match = false;
        break;
      }
    }
    if (match) {
      int c = 0;
      for (int j = start + ng; j < n && c < k; ++j) out[c++] = hist[j];
      return c;
    }
  }
  return 0;
}

}  // namespace engine
