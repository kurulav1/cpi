// Sequence-prefill parity: the batched prompt path must agree with token-by-token
// stepping. This test exists because four stacked prefill defects (driver never batching,
// the .cpi route never enabling sequencing, context overflow crashing as a cryptic CUDA
// error, and quant sequence mode computing only token 0's K/V) all lived behind a suite
// that stayed green; nothing compared the two prefill paths.
//
// For each precision (fp16, int4, int8):
//   1. per-token prefill of a fixed prompt -> greedy next token + logits
//   2. fresh engine, batched prefill (chunked path included via a >1024-token prompt)
//      -> greedy next token + logits
//   3. assert: same argmax, finite logits, max |logit diff| within tolerance (the GEMM
//      and dp4a paths legitimately round differently; garbage differs by orders of
//      magnitude, not tenths).
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine/plan_cuda_engine.hpp"

namespace {

std::vector<int> make_prompt(int n) {
  std::vector<int> p;
  p.reserve(n);
  p.push_back(2);  // BOS
  for (int i = 1; i < n; ++i) p.push_back(1000 + (i * 37) % 5000);
  return p;
}

struct Result {
  int argmax = -1;
  std::vector<float> logits;
};

Result run(const std::string& model, int quant_bits, const std::vector<int>& prompt, bool batched) {
  engine::PlanCudaEngine eng;
  if (quant_bits != 0) {
    engine::EngineOptions opt;
    opt.model_path = model;
    opt.max_context = 4096;
    opt.int8_streaming = true;
    opt.streaming_quant_bits = quant_bits;
    eng.initialize(opt);
  } else {
    eng.open(model, 4096);
  }
  eng.reset_state();
  if (batched) {
    eng.prefill(prompt);
  } else {
    const int P = static_cast<int>(prompt.size());
    for (int i = 0; i < P; ++i) eng.step(prompt[i], i, i == P - 1);
  }
  eng.synchronize();
  Result r;
  engine::runtime::DecodeParams greedy;
  greedy.temperature = 0.0f;
  r.argmax = eng.sample(greedy, prompt);
  r.logits = eng.logits();
  return r;
}

bool check(const char* tag, const Result& a, const Result& b, float tol) {
  if (a.argmax != b.argmax) {
    std::printf("FAIL %s: argmax %d (per-token) vs %d (batched)\n", tag, a.argmax, b.argmax);
    return false;
  }
  if (a.logits.size() != b.logits.size() || a.logits.empty()) {
    std::printf("FAIL %s: logits size %zu vs %zu\n", tag, a.logits.size(), b.logits.size());
    return false;
  }
  float max_diff = 0.0f;
  for (std::size_t i = 0; i < a.logits.size(); ++i) {
    if (!std::isfinite(a.logits[i]) || !std::isfinite(b.logits[i])) {
      std::printf("FAIL %s: non-finite logit at %zu\n", tag, i);
      return false;
    }
    max_diff = std::max(max_diff, std::fabs(a.logits[i] - b.logits[i]));
  }
  if (max_diff > tol) {
    std::printf("FAIL %s: max |logit diff| %.4f > tol %.4f\n", tag, max_diff, tol);
    return false;
  }
  std::printf("PASS %s: argmax %d, max |logit diff| %.4f\n", tag, a.argmax, max_diff);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model = "artifacts/hub/google__gemma-4-E2B-it/gemma4-e2b.cpi";
  if (const char* e = std::getenv("CPI_TEST_MODEL")) model = e;
  for (int i = 1; i < argc; ++i) {
    if (std::string(argv[i]) == "--model" && i + 1 < argc) model = argv[++i];
  }
  {
    std::FILE* f = std::fopen(model.c_str(), "rb");
    if (!f) {
      std::printf("SKIP: model not found at %s (set CPI_TEST_MODEL)\n", model.c_str());
      return 0;  // graceful skip on boxes without artifacts
    }
    std::fclose(f);
  }

  bool ok = true;
  try {
    // Short prompt exercises the single-chunk path; long crosses the chunk boundary and
    // would have caught the stale-device-position tail and the context-overflow class.
    const std::vector<int> short_p = make_prompt(300);
    const std::vector<int> long_p = make_prompt(1500);
    struct Cfg {
      const char* tag;
      int bits;
      float tol;
    };
    // Quant tolerances are wider: the batched path dequantizes to fp16 GEMMs while the
    // per-token path quantizes ACTIVATIONS through dp4a; int4's act-quant noise
    // compounds over 35 layers and measures ~2.3 max on healthy runs (int8 ~0.2, fp16
    // ~0.05). Garbage KV, the bug class this test exists for, differs by ~30+ and
    // flips the argmax, which stays the hard gate.
    const Cfg cfgs[] = {{"fp16", 0, 0.5f}, {"int4", 4, 4.0f}, {"int8", 8, 1.0f}};
    for (const Cfg& c : cfgs) {
      ok &= check((std::string(c.tag) + "/short").c_str(), run(model, c.bits, short_p, false),
                  run(model, c.bits, short_p, true), c.tol);
      ok &= check((std::string(c.tag) + "/long").c_str(), run(model, c.bits, long_p, false),
                  run(model, c.bits, long_p, true), c.tol);
    }
  } catch (const std::exception& e) {
    std::printf("FAIL: exception: %s\n", e.what());
    return 1;
  }
  return ok ? 0 : 1;
}
