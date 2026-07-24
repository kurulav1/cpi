// Perplexity, the metric quantization schemes are ranked by. Per-tensor reconstruction error
// (CPI_METAL_QUANT_STATS) ranks tensors, not models, and greedy-prefix agreement against a
// higher-precision reference is useless -- one early flip destroys the prefix, so it measures
// when the first divergence landed (measured across promotion budgets: 26/26/3/1/74 chars,
// non-monotonic). Perplexity averages over every position instead.
//
// Teacher forcing: feed the corpus one token at a time through the decode path generation uses
// and accumulate -log P(actual next token). ppl = exp(mean NLL). Comparable only across runs
// with the same corpus and tokenizer.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "engine/plan_metal_engine.hpp"
#include "model/tokenizer.hpp"

namespace {

void usage() {
  std::fprintf(stderr,
               "usage: metal_ppl <model> --tokenizer <tokenizer.json> --corpus <text file>\n"
               "                 [--quant 4|8] [--quant-group N] [--max-tokens N]\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 2;
  }
  std::string model = argv[1], tokenizer_path, corpus_path;
  int quant_bits = 0, quant_group = 0, max_tokens = 2048;

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "missing value for %s\n", what);
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--tokenizer") {
      tokenizer_path = val("--tokenizer");
    } else if (a == "--corpus") {
      corpus_path = val("--corpus");
    } else if (a == "--quant") {
      quant_bits = std::atoi(val("--quant").c_str());
    } else if (a == "--quant-group") {
      quant_group = std::atoi(val("--quant-group").c_str());
    } else if (a == "--max-tokens") {
      max_tokens = std::atoi(val("--max-tokens").c_str());
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
      usage();
      return 2;
    }
  }
  if (tokenizer_path.empty() || corpus_path.empty()) {
    usage();
    return 2;
  }

  std::ifstream f(corpus_path);
  if (!f) {
    std::fprintf(stderr, "cannot open corpus: %s\n", corpus_path.c_str());
    return 2;
  }
  std::stringstream ss;
  ss << f.rdbuf();
  const std::string text = ss.str();

  model::Tokenizer tok;
  tok.load(tokenizer_path);
  std::vector<int> ids = tok.encode(text, /*add_bos=*/true);
  if (static_cast<int>(ids.size()) > max_tokens) ids.resize(static_cast<std::size_t>(max_tokens));
  if (ids.size() < 2) {
    std::fprintf(stderr, "corpus too short after tokenization (%zu tokens)\n", ids.size());
    return 2;
  }

  engine::PlanMetalEngine eng;
  // Context must cover the whole corpus: a wrapped position scores later tokens against the
  // wrong history.
  eng.open(model, static_cast<int>(ids.size()) + 8, quant_bits, quant_group);
  eng.reset_kv_cache();

  double nll = 0.0;
  int counted = 0;
  for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
    const std::vector<float>& logits = eng.forward_token(ids[i], static_cast<int>(i));
    const int target = ids[i + 1];
    if (target < 0 || target >= static_cast<int>(logits.size())) continue;

    // Subtract the max before exponentiating: a vocab-sized sum of exp(logit) overflows to
    // inf and every perplexity comes out nan.
    const float m = *std::max_element(logits.begin(), logits.end());
    double sum = 0.0;
    for (const float l : logits) sum += std::exp(static_cast<double>(l - m));
    const double logZ = static_cast<double>(m) + std::log(sum);
    nll += logZ - static_cast<double>(logits[static_cast<std::size_t>(target)]);
    ++counted;

    if (counted % 256 == 0) {
      std::fprintf(stderr, "  ... %d/%zu tokens, running ppl %.4f\n", counted, ids.size() - 1,
                   std::exp(nll / counted));
    }
  }

  if (counted == 0) {
    std::fprintf(stderr, "no tokens scored\n");
    return 1;
  }
  const double mean = nll / counted;
  std::printf("[ppl] tokens=%d quant=%d group=%d mean_nll=%.6f ppl=%.4f\n", counted, quant_bits,
              quant_group, mean, std::exp(mean));
  return 0;
}
