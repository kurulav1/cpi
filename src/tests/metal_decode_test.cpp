// Metal decode, checked on real weights.
//
// TWO ORACLES:
//
//   1. The CPU engine (fp32 end to end) on the first forward -- argmax and top-5.
//   2. A golden token stream from the CUDA backend for the whole greedy sequence.
//
// A CAUTIONARY NOTE, because it nearly cost us a real bug. When Metal first diverged
// from the CPU engine at token 11 with a 0.55 logit gap, that was diagnosed as "the
// fp16/fp32 activation gap -- expected". It was not. The CPU engine had rope_theta
// hardcoded to 10000 and was rotating every Q and K by the wrong angle for every
// model that does not use that base (Qwen2.5, Qwen3, Llama 3). Metal was right and
// the oracle was broken. With that fixed, all three engines agree token-for-token.
//
// The lesson: when a new backend disagrees with a reference, the reference is a
// suspect too. Do not reach for "expected numerical drift" -- it is the explanation
// that makes a bug invisible.
//
// Usage:  metal_decode_test <model.ll2c> [num_tokens] [golden.txt]
// Skips (exit 0) when no model path is given, so ctest stays green without weights.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include "engine/cpu_engine.hpp"
#include "engine/engine_types.hpp"
#include "engine/plan_metal_engine.hpp"

namespace {

int top1(const std::vector<float>& v) {
  return static_cast<int>(std::max_element(v.begin(), v.end()) - v.begin());
}

std::vector<int> parse_csv(const std::string& s) {
  std::vector<int> out;
  std::string cur;
  for (char c : s) {
    if (c == ',') {
      if (!cur.empty()) out.push_back(std::atoi(cur.c_str()));
      cur.clear();
    } else if (!std::isspace(static_cast<unsigned char>(c))) {
      cur += c;
    }
  }
  if (!cur.empty()) out.push_back(std::atoi(cur.c_str()));
  return out;
}

// Reads "prompt:" and "expect:" lines; '#' starts a comment.
bool read_golden(const std::string& path, std::vector<int>* prompt, std::vector<int>* expect) {
  std::ifstream f(path);
  if (!f) return false;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    const auto colon = line.find(':');
    if (colon == std::string::npos) continue;
    const std::string key = line.substr(0, colon);
    const std::string val = line.substr(colon + 1);
    if (key == "prompt") *prompt = parse_csv(val);
    if (key == "expect") *expect = parse_csv(val);
  }
  return !expect->empty();
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("[metal_decode] SKIP: no model given (usage: metal_decode_test <model.ll2c>)\n");
    return 0;
  }
  const std::string model = argv[1];
  int n_new = (argc > 2) ? std::atoi(argv[2]) : 8;

  // CPI_METAL_QUANT=4|8 runs the same checks on a quantized model. The CPU engine
  // still uses the fp16 weights, so it is the RIGHT oracle: it says what the model
  // should do, and quantization error is measured AGAINST that rather than hidden.
  const char* qenv = std::getenv("CPI_METAL_QUANT");
  const int quant = qenv != nullptr ? std::atoi(qenv) : 0;
  const char* genv = std::getenv("CPI_METAL_QGROUP");
  const int qgroup = genv != nullptr ? std::atoi(genv) : 0;

  // Default prompt (token ids, so no tokenizer is needed). A golden file overrides it.
  std::vector<int> prompt = {1, 2, 3, 4, 5};
  std::vector<int> golden;
  if (argc > 3) {
    if (!read_golden(argv[3], &prompt, &golden)) {
      std::printf("[metal_decode] FAIL: could not read the golden file %s\n", argv[3]);
      return 1;
    }
    n_new = static_cast<int>(golden.size());
    std::printf("[metal_decode] golden: %s (%zu tokens)\n", argv[3], golden.size());
  }

  engine::PlanMetalEngine metal;
  if (!metal.available()) {
    std::printf("[metal_decode] SKIP: no Metal GPU (%s)\n", metal.last_error().c_str());
    return 0;
  }
  std::printf("[metal_decode] device: %s\n", metal.device_name().c_str());

  metal.open(model, /*max_context=*/512, quant, qgroup);
  const auto& cfg = metal.config();
  std::printf("[metal_decode] layers=%d hidden=%d heads=%d kv_heads=%d vocab=%d qkv_bias=%d\n",
              cfg.num_layers, cfg.hidden_size, cfg.num_heads, cfg.num_kv_heads, cfg.vocab_size,
              cfg.has_qkv_bias ? 1 : 0);

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

  // Metal's own top-5, so the RANKING can be compared -- not just the winner. An
  // argmax can agree by luck on an easy token while the distribution underneath is
  // quietly wrong; the ordering of the runners-up is far more sensitive.
  std::vector<int> idx(mlogits.size());
  for (std::size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int>(i);
  std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(), [&](int a, int b) {
    return mlogits[static_cast<std::size_t>(a)] > mlogits[static_cast<std::size_t>(b)];
  });
  std::printf("\n  METAL top-5:");
  for (int i = 0; i < 5; ++i) {
    std::printf("  %d(%.3f)", idx[static_cast<std::size_t>(i)],
                mlogits[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])]);
  }
  std::printf("\n");

  int rank_matches = 0;
  for (int i = 0; i < 5 && i < static_cast<int>(cpu_top.size()); ++i) {
    if (cpu_top[static_cast<std::size_t>(i)].first == idx[static_cast<std::size_t>(i)]) {
      ++rank_matches;
    }
  }
  std::printf("  top-5 ranking agreement: %d/5\n", rank_matches);

  const bool argmax_ok = !cpu_top.empty() && cpu_top[0].first == m_top;
  std::printf("  argmax agreement: %s\n", argmax_ok ? "PASS" : "FAIL");

  double max_abs = 0.0;
  for (const auto& p : cpu_top) {
    const double d = std::fabs(static_cast<double>(mlogits[static_cast<std::size_t>(p.first)]) -
                               static_cast<double>(p.second));
    max_abs = std::max(max_abs, d);
  }
  std::printf("  max_abs_diff over the CPU's top-5: %.4f\n", max_abs);

  // ---- the gate: does Metal reproduce CUDA's greedy stream? ----------------
  //
  // Exact token equality is ALMOST the right gate, but not quite: the GEMV sums fp32
  // in a different order than CUDA does, and reordering an fp32 sum changes its
  // rounding by ~5e-3. Greedy decoding turns that into a discrete choice, so on a
  // near-tie the two backends can legitimately pick different tokens while both being
  // correct -- and they rejoin immediately after.
  //
  // So a divergence is only forgiven when it happens at a genuine tie: the top-2
  // logits within the noise floor. A divergence at a CONFIDENT token is a real bug and
  // still fails. This keeps the gate honest instead of merely lenient.
  constexpr double kTieTol = 0.05;  // ~4x the observed fp16 noise floor

  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<int> m_out = metal.generate_greedy(prompt, n_new);
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  std::printf("\n  metal decode: %.1f tok/s (%d tokens in %.0f ms)\n",
              static_cast<double>(m_out.size()) / (ms / 1000.0), static_cast<int>(m_out.size()),
              ms);

  std::vector<int> c_out = cpu.generate(prompt, n_new, /*temperature=*/0.0f);

  // CpuLlamaEngine::generate returns PROMPT + continuation; PlanMetalEngine returns
  // only the new tokens.
  if (c_out.size() >= prompt.size() && std::equal(prompt.begin(), prompt.end(), c_out.begin())) {
    c_out.erase(c_out.begin(), c_out.begin() + static_cast<long>(prompt.size()));
  }

  std::printf("\n  METAL greedy:");
  for (int t : m_out) std::printf(" %d", t);
  std::printf("\n  CPU   greedy:");
  for (int t : c_out) std::printf(" %d", t);
  std::size_t cpu_agree = 0;
  while (cpu_agree < m_out.size() && cpu_agree < c_out.size() &&
         m_out[cpu_agree] == c_out[cpu_agree]) {
    ++cpu_agree;
  }
  std::printf("\n  agrees with the CPU engine for %zu/%zu tokens\n", cpu_agree, c_out.size());

  bool golden_ok = true;
  if (golden.empty()) {
    std::printf("\n  no golden stream given; gating on argmax only\n");
  } else {
    const std::size_t n = std::min(golden.size(), m_out.size());
    std::size_t agree = 0;
    while (agree < n && m_out[agree] == golden[agree]) ++agree;

    if (agree == n) {
      std::printf("\n  Metal reproduces the CUDA stream (%zu tokens): PASS\n", n);
    } else {
      // Replay to the divergence and ask how confident the model was there. A tie
      // means the two backends were choosing between near-equal candidates; a
      // confident token means one of them is wrong.
      std::vector<int> ctx = prompt;
      ctx.insert(ctx.end(), m_out.begin(), m_out.begin() + static_cast<long>(agree));
      std::vector<float> lg;
      for (std::size_t i = 0; i < ctx.size(); ++i) {
        lg = metal.forward_token(ctx[i], static_cast<int>(i));
      }
      std::vector<float> sorted = lg;
      std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(), std::greater<float>());
      const double gap = static_cast<double>(sorted[0]) - static_cast<double>(sorted[1]);

      const bool tie = gap < kTieTol;
      golden_ok = tie;
      std::printf("\n  Metal diverges from CUDA at token %zu of %zu (metal=%d cuda=%d)\n", agree, n,
                  m_out[agree], golden[agree]);
      std::printf("    top-2 logit gap there: %.4f  (tie tolerance %.2f)\n", gap, kTieTol);
      std::printf("    -> %s\n", tie ? "A TIE: fp32 sum ordering differs between backends, so a "
                                       "near-equal choice can fall either way. PASS."
                                     : "NOT A TIE: the model was confident, so one backend is "
                                       "WRONG. FAIL.");
    }
  }

  if (!argmax_ok || !golden_ok) {
    std::printf("\n[metal_decode] FAIL\n");
    return 1;
  }
  std::printf("\n[metal_decode] PASS\n");
  return 0;
}
