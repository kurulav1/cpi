// Gemma 4 forward parity driver. Loads a .cpi, runs a fixed token sequence, and
// prints per-layer output RMS + top-10 logits so they can be eyeballed against
// the HF reference oracle (tools/gemma4_reference_oracle.py). For the prompt
// "The capital of France is" (ids 2,714,6037,576,6081,603) the oracle argmax is
// 1390 with logits rms ~15.70.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "engine/plan_cuda_engine.hpp"

int main(int argc, char** argv) {
  std::string cpi = "artifacts/hub/google__gemma-4-E2B-it/gemma4-e2b.cpi";
  // Default: "The capital of France is" (Gemma-4 tokenizer). HF reference argmax
  // (use_cache=True, KV sharing active) is 7001 (' France').
  std::vector<int> tokens = {818, 5279, 529, 7001, 563};
  int expect = 7001;
  int gen = 0;
  int graph_bench = 0;  // >0: run the CUDA-graph decode A/B benchmark for N iters
  int graph_pos = 0;    // decode position for --graph-bench (pads prefill; exercises the window)
  int weight_quant = 0;  // 0 fp16, 4 or 8: on-load weight-only quant (llama-bench-comparable)
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc) cpi = argv[++i];
    else if (a == "--expect" && i + 1 < argc) expect = std::stoi(argv[++i]);
    else if (a == "--gen" && i + 1 < argc) gen = std::stoi(argv[++i]);
    else if (a == "--graph-bench" && i + 1 < argc) graph_bench = std::stoi(argv[++i]);
    else if (a == "--weight-quant" && i + 1 < argc) weight_quant = std::stoi(argv[++i]);
    else if (a == "--graph-pos" && i + 1 < argc) graph_pos = std::stoi(argv[++i]);
    else if (a == "--tokens" && i + 1 < argc) {
      tokens.clear();
      std::string s = argv[++i], cur;
      for (char c : s) { if (c == ',') { tokens.push_back(std::stoi(cur)); cur.clear(); } else cur += c; }
      if (!cur.empty()) tokens.push_back(std::stoi(cur));
    }
  }

  try {
    engine::PlanCudaEngine eng;
    std::printf("loading %s ...\n", cpi.c_str());
    if (weight_quant == 4 || weight_quant == 8) {
      engine::EngineOptions opt;
      opt.model_path = cpi;
      opt.max_context = 4096;
      opt.int8_streaming = true;
      opt.streaming_quant_bits = weight_quant;
      eng.initialize(opt);
    } else {
      eng.open(cpi);
    }
    std::printf("loaded. vocab=%d\n", eng.vocab());

    if (graph_bench > 0) {
      eng.benchmark_graph_decode(tokens, graph_bench, graph_pos);
      return 0;
    }

    if (gen > 0) {
      auto out = eng.generate(tokens, gen, 0.0f);
      std::printf("generated ids:");
      for (int t : out) std::printf(" %d", t);
      std::printf("\n");
      return 0;
    }

    std::vector<float> rms;
    auto logits = eng.forward_logits(tokens, &rms);

    std::printf("per-layer output rms (last token):\n");
    for (std::size_t L = 0; L < rms.size(); ++L) std::printf("  L%2zu rms=%.4f\n", L, rms[L]);

    // logits stats + top-10
    double ss = 0;
    int argmax = 0;
    for (int i = 0; i < (int)logits.size(); ++i) {
      ss += (double)logits[i] * logits[i];
      if (logits[i] > logits[argmax]) argmax = i;
    }
    std::printf("logits rms=%.4f  argmax=%d  (expected=%d)\n",
                std::sqrt(ss / logits.size()), argmax, expect);
    std::vector<int> idx(logits.size());
    for (int i = 0; i < (int)idx.size(); ++i) idx[i] = i;
    std::partial_sort(idx.begin(), idx.begin() + 10, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::printf("top-10:\n");
    for (int i = 0; i < 10; ++i) std::printf("  %7d  %9.4f\n", idx[i], logits[idx[i]]);
    const bool ok = expect < 0 || argmax == expect;
    std::printf(ok ? "\nPARITY OK (argmax matches HF reference)\n" : "\nMISMATCH\n");
    return ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("ERROR: %s\n", e.what());
    return 2;
  }
}
