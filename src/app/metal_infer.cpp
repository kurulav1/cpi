// Text generation on Apple Silicon: tokenizer -> Metal decode -> text.
//
// A separate entry point from llama_infer rather than a flag on it. llama_infer's
// engine dispatch is templated over the CUDA engines and pulls in the whole CUDA
// tree; on a Mac none of that exists. Keeping this small and standalone means the
// Metal path has no CUDA in its dependency graph at all, which is the property that
// makes it buildable on a stock Mac.
//
// Scope matches PlanMetalEngine: fp16, dense, uniform geometry (Llama / Qwen2),
// greedy decode.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine/plan_metal_engine.hpp"
#include "model/tokenizer.hpp"

namespace {

void usage() {
  std::printf(
      "usage: metal_infer <model.ll2c> --tokenizer <tokenizer.json> --prompt <text>\n"
      "                   [--max-new N] [--max-context N]\n");
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    usage();
    return 1;
  }

  std::string model = argv[1];
  std::string tokenizer_path;
  std::string prompt;
  int max_new = 64;
  int max_context = 2048;
  int quant_bits = 0;
  int quant_group = 0;
  engine::PlanMetalEngine::Sampling samp;

  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    auto val = [&](const char* what) -> std::string {
      if (i + 1 >= argc) {
        std::printf("missing value for %s\n", what);
        std::exit(1);
      }
      return argv[++i];
    };
    if (a == "--tokenizer") {
      tokenizer_path = val("--tokenizer");
    } else if (a == "--prompt") {
      prompt = val("--prompt");
    } else if (a == "--max-new") {
      max_new = std::atoi(val("--max-new").c_str());
    } else if (a == "--max-context") {
      max_context = std::atoi(val("--max-context").c_str());
    } else if (a == "--quant") {
      quant_bits = std::atoi(val("--quant").c_str());
    } else if (a == "--quant-group") {
      quant_group = std::atoi(val("--quant-group").c_str());
    } else if (a == "--temp") {
      samp.temperature = std::stof(val("--temp"));
    } else if (a == "--top-k") {
      samp.top_k = std::atoi(val("--top-k").c_str());
    } else if (a == "--top-p") {
      samp.top_p = std::stof(val("--top-p"));
    } else if (a == "--repeat-penalty") {
      samp.repetition_penalty = std::stof(val("--repeat-penalty"));
    } else if (a == "--seed") {
      samp.seed = static_cast<unsigned>(std::atoi(val("--seed").c_str()));
    } else {
      std::printf("unknown argument: %s\n", a.c_str());
      usage();
      return 1;
    }
  }

  if (tokenizer_path.empty() || prompt.empty()) {
    usage();
    return 1;
  }

  engine::PlanMetalEngine eng;
  if (!eng.available()) {
    std::printf("no Metal GPU: %s\n", eng.last_error().c_str());
    return 1;
  }

  model::Tokenizer tok;
  tok.load(tokenizer_path);

  eng.open(model, max_context, quant_bits, quant_group);
  const auto& cfg = eng.config();
  std::fprintf(stderr, "[metal] %s | layers=%d hidden=%d heads=%d kv_heads=%d vocab=%d | %s\n",
               eng.device_name().c_str(), cfg.num_layers, cfg.hidden_size, cfg.num_heads,
               cfg.num_kv_heads, cfg.vocab_size,
               eng.quant_bits() == 0 ? "fp16" : (eng.quant_bits() == 4 ? "int4" : "int8"));
  std::fprintf(stderr, "[metal] weights on GPU: %.2f GB\n",
               static_cast<double>(eng.weight_bytes()) / 1073741824.0);

  const std::vector<int> ids = tok.encode(prompt, /*add_bos=*/true);
  std::fprintf(stderr, "[metal] prompt: %zu tokens\n", ids.size());

  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<int> out = eng.generate(ids, max_new, samp);
  const auto t1 = std::chrono::steady_clock::now();
  const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  // Decode the whole sequence at once. Decoding token by token loses the word
  // boundaries and runs every word together.
  std::printf("%s\n", tok.decode(out).c_str());
  const double pms = eng.last_prefill_ms();
  const int ptok = eng.last_prefill_tokens();
  if (ptok > 0) {
    std::fprintf(stderr, "\n[perf] prefill: %d tokens in %.0f ms = %.0f tok/s\n", ptok, pms,
                 static_cast<double>(ptok) / (pms / 1000.0));
  }
  std::fprintf(stderr, "[perf] decode:  %zu tokens in %.0f ms = %.1f tok/s\n", out.size(), ms - pms,
               static_cast<double>(out.size()) / ((ms - pms) / 1000.0));
  eng.dump_profile();  // no-op unless CPI_METAL_PROFILE is set
  return 0;
}
