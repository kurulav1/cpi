// Gate for sequence-mode text prefill.
//
// Prefilling a prompt in one pass must agree with feeding it token by token: same KV
// cache, same logits, same continuation. This is the check that catches a wrong RoPE
// position, an off-by-one in the KV append, or a missing sliding-window k_start.
//
// The sliding-window case is the one worth insisting on: Gemma windows its sliding
// layers at 512, and the prefill kernel had no window support (Llama never needed it),
// so a prompt longer than the window would have been silently wrong. The long-prompt
// case below is there precisely to fail if that regresses.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/plan_cuda_engine.hpp"

namespace {

// greedy continuation from an already-prefilled state
std::vector<int> decode_greedy(engine::PlanCudaEngine& eng, int start_pos, int n) {
  std::vector<int> out;
  int pos = start_pos;
  int tok = 0;
  {
    const std::vector<float>& lg = eng.logits();
    tok = static_cast<int>(std::max_element(lg.begin(), lg.end()) - lg.begin());
  }
  for (int i = 0; i < n; ++i) {
    out.push_back(tok);
    eng.step(tok, pos++, true);
    const std::vector<float>& lg = eng.logits();
    tok = static_cast<int>(std::max_element(lg.begin(), lg.end()) - lg.begin());
  }
  return out;
}

double cosine(const std::vector<float>& a, const std::vector<float>& b) {
  double dot = 0, na = 0, nb = 0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    dot += static_cast<double>(a[i]) * b[i];
    na += static_cast<double>(a[i]) * a[i];
    nb += static_cast<double>(b[i]) * b[i];
  }
  return dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
}

int run_case(const std::string& model, const char* label, int prompt_len) {
  engine::PlanCudaEngine a;
  a.open(model, 4096);
  if (!a.can_sequence_prefill()) {
    std::printf("  [skip] %-18s plan cannot sequence-prefill\n", label);
    return 0;
  }

  // A deterministic prompt of the requested length (content does not matter; positions,
  // windows and cache offsets do).
  std::vector<int> prompt;
  prompt.push_back(2);  // BOS
  for (int i = 1; i < prompt_len; ++i) prompt.push_back(1000 + (i * 37) % 20000);

  // --- token by token (the existing path) ---
  for (int i = 0; i < static_cast<int>(prompt.size()); ++i)
    a.step(prompt[i], i, i == static_cast<int>(prompt.size()) - 1);
  const std::vector<float> logits_tbt = a.logits();
  const std::vector<int> cont_tbt = decode_greedy(a, static_cast<int>(prompt.size()), 8);

  // --- one sequence pass ---
  engine::PlanCudaEngine b;
  b.open(model, 4096);
  b.prefill_sequence(prompt, 0, {}, {});
  const std::vector<float> logits_seq = b.logits();
  const std::vector<int> cont_seq = decode_greedy(b, static_cast<int>(prompt.size()), 8);

  const int argmax_tbt =
      static_cast<int>(std::max_element(logits_tbt.begin(), logits_tbt.end()) - logits_tbt.begin());
  const int argmax_seq =
      static_cast<int>(std::max_element(logits_seq.begin(), logits_seq.end()) - logits_seq.begin());
  const double cos = cosine(logits_tbt, logits_seq);
  const bool same_cont = cont_tbt == cont_seq;
  const bool ok = argmax_tbt == argmax_seq && cos > 0.9999 && same_cont;

  std::printf("  [%s] %-18s len=%-5d argmax %d vs %d | logit cosine %.6f | continuation %s\n",
              ok ? "ok" : "FAIL", label, prompt_len, argmax_tbt, argmax_seq, cos,
              same_cont ? "identical" : "DIFFERS");
  return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model = argc > 1 ? argv[1] : "artifacts/hub/google__gemma-4-E2B-it/hf";
  int failures = 0;
  failures += run_case(model, "short", 12);
  failures += run_case(model, "medium", 200);
  // Longer than Gemma's 512 sliding window; this is the case that fails if the
  // prefill kernel forgets its per-token k_start.
  failures += run_case(model, "beyond window", 700);

  std::printf("\n%s\n", failures == 0
                            ? "PARITY OK (sequence prefill == token-by-token prefill)"
                            : "PARITY FAILED");
  return failures == 0 ? 0 : 1;
}
