// Gemma 4 forward parity driver. Loads a .cpi, runs a fixed token sequence, and
// prints per-layer output RMS + top-10 logits so they can be eyeballed against
// the HF reference oracle (tools/gemma4_reference_oracle.py). For the prompt
// "The capital of France is" (ids 2,714,6037,576,6081,603) the oracle argmax is
// 1390 with logits rms ~15.70.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
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
  int graph_bench = 0;   // >0: run the CUDA-graph decode A/B benchmark for N iters
  int graph_pos = 0;     // decode position for --graph-bench (pads prefill; exercises the window)
  int weight_quant = 0;  // 0 fp16, 4 or 8: on-load weight-only quant (llama-bench-comparable)
  std::vector<int> ppl_tokens;  // --ppl-tokens: teacher-forced perplexity over this id sequence
  int ppl_stride = 1;    // --ppl-stride K: score every K-th position (subsample a big corpus)
  int ppl_warm = 1;      // --ppl-warm W: skip the first W positions (so each has real context)
  int ppl_win = 0;       // --ppl-win W: cap context to BOS + last W tokens (0 = full prefix)
  std::string ppl_file;  // --ppl-file: read comma/whitespace-separated ids from a file
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--model" && i + 1 < argc)
      cpi = argv[++i];
    else if (a == "--ppl-stride" && i + 1 < argc)
      ppl_stride = std::max(1, std::stoi(argv[++i]));
    else if (a == "--ppl-warm" && i + 1 < argc)
      ppl_warm = std::max(1, std::stoi(argv[++i]));
    else if (a == "--ppl-win" && i + 1 < argc)
      ppl_win = std::stoi(argv[++i]);
    else if (a == "--ppl-file" && i + 1 < argc)
      ppl_file = argv[++i];
    else if (a == "--ppl-tokens" && i + 1 < argc) {
      std::string s = argv[++i], cur;
      for (char c : s) {
        if (c == ',') {
          ppl_tokens.push_back(std::stoi(cur));
          cur.clear();
        } else
          cur += c;
      }
      if (!cur.empty()) ppl_tokens.push_back(std::stoi(cur));
    } else if (a == "--expect" && i + 1 < argc)
      expect = std::stoi(argv[++i]);
    else if (a == "--gen" && i + 1 < argc)
      gen = std::stoi(argv[++i]);
    else if (a == "--graph-bench" && i + 1 < argc)
      graph_bench = std::stoi(argv[++i]);
    else if (a == "--weight-quant" && i + 1 < argc)
      weight_quant = std::stoi(argv[++i]);
    else if (a == "--graph-pos" && i + 1 < argc)
      graph_pos = std::stoi(argv[++i]);
    else if (a == "--tokens" && i + 1 < argc) {
      tokens.clear();
      std::string s = argv[++i], cur;
      for (char c : s) {
        if (c == ',') {
          tokens.push_back(std::stoi(cur));
          cur.clear();
        } else
          cur += c;
      }
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

    if (!ppl_file.empty()) {
      std::ifstream f(ppl_file);
      std::string cur;
      char c;
      while (f.get(c)) {
        if (c == ',' || c == ' ' || c == '\n' || c == '\r' || c == '\t') {
          if (!cur.empty()) {
            ppl_tokens.push_back(std::stoi(cur));
            cur.clear();
          }
        } else {
          cur += c;
        }
      }
      if (!cur.empty()) ppl_tokens.push_back(std::stoi(cur));
    }

    if (!ppl_tokens.empty()) {
      // Teacher-forced perplexity: for each scored position i, forward the FULL prefix
      // tokens[0..i-1] (BOS-anchored, real left context) and score the ground-truth next token
      // tokens[i] via nll = logsumexp(logits) - logits[tokens[i]]. --ppl-stride subsamples which
      // positions are scored so a large corpus stays affordable; --ppl-warm skips the first W
      // positions so every scored one has non-trivial context. Output-space quality metric.
      double nll = 0.0;
      int cnt = 0;
      int argmatch = 0;  // greedy top-1 == ground-truth next token (teacher-forced accuracy)
      const int n = static_cast<int>(ppl_tokens.size());
      for (int i = ppl_warm; i < n; i += ppl_stride) {
        // Full prefix by default; with --ppl-win W, cap context to BOS + the last W tokens so each
        // prefill is bounded (O(N*W) instead of O(N^2)). BOS (ppl_tokens[0]) is kept so the window
        // is anchored -- dropping it makes Gemma's logits garbage.
        std::vector<int> pre;
        if (ppl_win > 0 && i > ppl_win) {
          pre.reserve(static_cast<std::size_t>(ppl_win) + 1);
          pre.push_back(ppl_tokens[0]);
          pre.insert(pre.end(), ppl_tokens.begin() + (i - ppl_win), ppl_tokens.begin() + i);
        } else {
          pre.assign(ppl_tokens.begin(), ppl_tokens.begin() + i);
        }
        auto lg = eng.forward_logits(pre, nullptr);
        const int tgt = ppl_tokens[static_cast<std::size_t>(i)];
        if (tgt < 0 || tgt >= static_cast<int>(lg.size())) continue;
        double mx = -1e30;
        int am = 0;
        for (int v = 0; v < static_cast<int>(lg.size()); ++v) {
          if (lg[v] > mx) {
            mx = lg[v];
            am = v;
          }
        }
        double se = 0.0;
        for (float v : lg) se += std::exp(static_cast<double>(v) - mx);
        const double lse = mx + std::log(se);
        nll += lse - static_cast<double>(lg[tgt]);
        if (am == tgt) ++argmatch;
        ++cnt;
      }
      const double mean_nll = cnt > 0 ? nll / cnt : 0.0;
      std::printf("[ppl] weight_quant=%d tokens_scored=%d ppl=%.5f nll/tok=%.5f top1_acc=%.4f\n",
                  weight_quant, cnt, std::exp(mean_nll), mean_nll,
                  cnt > 0 ? static_cast<double>(argmatch) / cnt : 0.0);
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
    std::printf("logits rms=%.4f  argmax=%d  (expected=%d)\n", std::sqrt(ss / logits.size()),
                argmax, expect);
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
