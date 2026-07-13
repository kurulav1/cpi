// Metal decode vs the CPU engine, on the same weights.
//
// The CPU engine is the oracle: it is CUDA-free, it runs on Apple Silicon, and it
// is the same reference the CUDA backend is checked against. So this compares the
// two engines that are actually present on the Mac -- no CUDA needed, and no need
// to trust numbers carried over from another machine.
//
// Numerics will NOT match bit-for-bit: the shaders accumulate fp32 in a different
// order than the CPU does. What must match is the ARGMAX -- the token the model
// actually picks. A wrong stride or a dropped bias moves logits far more than
// accumulation order ever does.
//
// Usage:  metal_decode_test <model.ll2c> [num_tokens]
// Skips (exit 0) when no model path is given, so ctest stays green without weights.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine/cpu_engine.hpp"
#include "engine/engine_types.hpp"
#include "engine/plan_metal_engine.hpp"

namespace {

int top1(const std::vector<float>& v) {
  return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("[metal_decode] SKIP: no model given (usage: metal_decode_test <model.ll2c>)\n");
    return 0;
  }
  const std::string model = argv[1];
  const int n_new = (argc > 2) ? std::atoi(argv[2]) : 8;

  engine::PlanMetalEngine metal;
  if (!metal.available()) {
    std::printf("[metal_decode] SKIP: no Metal GPU (%s)\n", metal.last_error().c_str());
    return 0;
  }
  std::printf("[metal_decode] device: %s\n", metal.device_name().c_str());

  metal.open(model, /*max_context=*/512);
  const auto& cfg = metal.config();
  std::printf("[metal_decode] layers=%d hidden=%d heads=%d kv_heads=%d vocab=%d qkv_bias=%d\n",
              cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.vocab_size,
              cfg.has_qkv_bias ? 1 : 0);

  // A short, fixed prompt. Token ids, so no tokenizer is needed here.
  const std::vector<int> prompt = {1, 2, 3, 4, 5};

  // ---- one forward, compared against the CPU engine ------------------------
  int pos = 0;
  std::vector<float> mlogits;
  for (std::size_t i = 0; i < prompt.size(); ++i, ++pos) {
    mlogits = metal.forward_token(prompt[i], pos);
  }

  engine::CpuLlamaEngine cpu;
  engine::EngineOptions opt;
  opt.model_path = model;
  cpu.initialize(opt);
  const auto cpu_top = cpu.inspect_next_logits(prompt, 5);

  const int m_top = top1(mlogits);
  std::printf("\n  CPU  top-5:");
  for (const auto& p : cpu_top) std::printf("  %d(%.3f)", p.first, p.second);
  std::printf("\n  METAL top-1: %d(%.3f)\n", m_top, mlogits[static_cast<std::size_t>(m_top)]);

  const bool argmax_ok = !cpu_top.empty() && cpu_top[0].first == m_top;
  std::printf("\n  argmax agreement: %s\n", argmax_ok ? "PASS" : "FAIL");

  // How far apart are the logits the CPU reported, on the values Metal produced?
  double max_abs = 0.0;
  for (const auto& p : cpu_top) {
    const double d = std::fabs(static_cast<double>(mlogits[static_cast<std::size_t>(p.first)]) -
                               static_cast<double>(p.second));
    max_abs = std::max(max_abs, d);
  }
  std::printf("  max_abs_diff over the CPU's top-5: %.4f\n", max_abs);

  if (!argmax_ok) {
    std::printf("\n[metal_decode] FAIL: the two engines disagree on the next token.\n");
    return 1;
  }

  // ---- greedy decode, just to see it run ----------------------------------
  const std::vector<int> out = metal.generate_greedy(prompt, n_new);
  std::printf("\n  greedy(%d):", n_new);
  for (int t : out) std::printf(" %d", t);
  std::printf("\n\n[metal_decode] PASS\n");
  return 0;
}
