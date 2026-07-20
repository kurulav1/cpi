// Qwen3.5's TEXT stack against a HuggingFace logits oracle.
//
//   ./qwen35_text_test <model.ll2c> <oracle_dir>     (oracle_dir = tools/qwen35_text_oracle.py --out)
//
// This exists because Qwen3.5 had NO text gate. "Text is done and verified" was a comparison
// somebody ran by hand once; nothing in src/tests would catch a regression, and metal_decode_test
// cannot help -- it compares against LlamaEngine's native CPU runtime, which throws on
// linear-attention layers, so it dies on this model before measuring anything.
//
// It is also the control for FIRST_STEP_LOGITS in metal_vision_test, which found the multimodal
// logits drifting from the oracle by mean_abs 0.64 at the first generated step. Four things sit
// upstream of that number -- tower, splice, M-RoPE, text stack -- and this removes the first
// three: no image, no splice, plain 1-D rope. Same engine, same prefill entry point, same
// forward_token loop, so a drift that shows up HERE is the text stack's and a clean run here puts
// the defect in the multimodal-specific path.
//
// The token ids are hardcoded and match tools/qwen35_text_oracle.py exactly. Deliberately not
// tokenized: a tokenizer disagreement would otherwise be indistinguishable from an engine one.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "engine/plan_metal_engine.hpp"
#include "model/json_mini.hpp"

namespace {

int failures = 0;

// MUST match TOKENS in tools/qwen35_text_oracle.py.
const std::vector<int> kTokens = {3838, 374, 419, 30, 358, 1079, 264,
                                  4128, 1614, 13, 6771, 752, 3291, 498};

std::vector<float> read_f32(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::fseek(f, 0, SEEK_END);
  const long bytes = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<float> v(static_cast<std::size_t>(bytes) / sizeof(float));
  if (std::fread(v.data(), 1, static_cast<std::size_t>(bytes), f) !=
      static_cast<std::size_t>(bytes)) {
    std::fprintf(stderr, "short read on %s\n", path.c_str());
    std::exit(2);
  }
  std::fclose(f);
  return v;
}

std::string read_text(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return std::string();
  std::string s;
  char buf[4096];
  std::size_t n;
  while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0) s.append(buf, n);
  std::fclose(f);
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 3) {
    std::printf("[qwen35_text] SKIP: usage: qwen35_text_test <model.ll2c> <oracle_dir>\n");
    return 0;
  }
  const std::string model = argv[1];
  const std::string dir = argv[2];

  const std::vector<float> ref = read_f32(dir + "/text_first_step_logits.f32");
  const std::string manifest = read_text(dir + "/manifest.json");

  engine::PlanMetalEngine eng;
  eng.open(model, 512);
  std::printf("[qwen35_text] %zu prompt tokens, oracle vocab %zu\n", kTokens.size(), ref.size());

  // Text prefill through the SAME entry point the multimodal path uses: no spliced embeddings
  // and no M-RoPE array means the plain 1-D rope path, which is what makes this a clean control
  // rather than a different code path that happens to also do text.
  //
  // Prefill everything EXCEPT the last token, then run that one through forward_token to get the
  // logits. prefill_multimodal consumes every token it is given (prefill_chunks(n) covers all n)
  // and there is no accessor for the logits it leaves behind, so prefilling all n and then
  // calling forward_token(tokens.back()) would feed the final token TWICE -- once at n-1 and
  // again at n -- corrupting both the KV cache and every position after it.
  //
  // That is not hypothetical. This test did exactly that on its first run and produced token 498
  // eight times in a row: 498 is kTokens.back(), and the model, shown "498 498", duly predicted
  // 498 forever. It read like a broken text stack. metal_vision_test's MULTIMODAL_stream loop
  // has the same double-feed.
  const std::vector<int> head(kTokens.begin(), kTokens.end() - 1);
  const std::vector<std::vector<float>> no_embeds(head.size());
  eng.prefill_multimodal(head, no_embeds);

  const std::vector<float>& lg =
      eng.forward_token(kTokens.back(), static_cast<int>(head.size()));

  if (lg.size() != ref.size()) {
    std::printf("  %-22s size %zu vs oracle %zu  FAIL\n", "TEXT_LOGITS", lg.size(), ref.size());
    return 1;
  }

  // Mean-centre both: a constant offset across every logit changes neither softmax nor argmax,
  // so counting it as error would report a difference that cannot affect any output.
  double mg = 0.0, mr = 0.0;
  for (std::size_t j = 0; j < lg.size(); ++j) { mg += lg[j]; mr += ref[j]; }
  mg /= static_cast<double>(lg.size());
  mr /= static_cast<double>(ref.size());

  float mx = 0.0f;
  double sum = 0.0;
  std::size_t nonfinite = 0;
  int arg_g = 0, arg_r = 0;
  for (std::size_t j = 0; j < lg.size(); ++j) {
    if (!std::isfinite(lg[j])) ++nonfinite;
    const float d =
        std::fabs((lg[j] - static_cast<float>(mg)) - (ref[j] - static_cast<float>(mr)));
    mx = std::max(mx, d);
    sum += d;
    if (lg[j] > lg[static_cast<std::size_t>(arg_g)]) arg_g = static_cast<int>(j);
    if (ref[j] > ref[static_cast<std::size_t>(arg_r)]) arg_r = static_cast<int>(j);
  }
  // NaN counted explicitly: std::max(x, NaN) returns x, so an all-NaN run accumulates max 0 and
  // would otherwise PASS every tolerance in this file.
  if (nonfinite != 0) {
    std::printf("  %-22s %zu/%zu non-finite logits  FAIL\n", "TEXT_LOGITS", nonfinite, lg.size());
    return 1;
  }

  // Tolerance 0.15, and the reasoning matters because a number picked to make the run green is
  // worthless. The BEHAVIOURAL gate here is TEXT_stream below, which is token-identical to the
  // fp32 reference over 8 tokens -- that is what says the text stack is right. This check is a
  // regression tripwire on top of it, anchored just above the measured 0.12 so that drift GROWTH
  // trips it. Logits here run 15-24 in magnitude, so 0.12 is roughly half a percent, which is
  // what fp16 weights against an fp32 reference cost over 24 layers.
  //
  // It is also the yardstick for metal_vision_test's FIRST_STEP_LOGITS: the same measurement on
  // the multimodal path reads 0.484, four times this, and THAT is the multimodal-specific defect.
  const double kTol = 0.15;
  const double mean = sum / static_cast<double>(lg.size());
  std::printf("      argmax ours=%d oracle=%d\n", arg_g, arg_r);
  std::printf("  %-22s max_abs=%.4f  mean_abs=%.4f  tol_mean=%.2f  %s\n", "TEXT_LOGITS", mx, mean,
              kTol, mean <= kTol ? "PASS" : "FAIL");
  if (mean > kTol) ++failures;

  // The greedy stream, for the same reason metal_vision_test prints one: logits say how far off
  // we are, tokens say whether it matters.
  // The last prompt token occupied position head.size(); the first generated one follows it.
  std::vector<int> got;
  int next = arg_g;
  const int pos = static_cast<int>(head.size()) + 1;
  got.push_back(arg_g);
  for (int i = 0; i < 7; ++i) {
    const std::vector<float>& s = eng.forward_token(next, pos + i);
    int best = 0;
    for (std::size_t j = 1; j < s.size(); ++j) {
      if (s[j] > s[static_cast<std::size_t>(best)]) best = static_cast<int>(j);
    }
    got.push_back(best);
    next = best;
  }
  // From tools/qwen35_text_oracle.py on this prompt, fp32. Hardcoded like the vision test's
  // reference stream: reading it back out of the manifest would just be trusting the same file
  // twice, and json_mini has no array accessor.
  const std::vector<int> want = {15, 60, 283, 220, 15, 26, 198, 262};
  int match = 0;
  for (std::size_t i = 0; i < want.size() && i < got.size(); ++i) {
    if (got[i] == want[i]) ++match; else break;
  }
  std::printf("      metal : ");
  for (int v : got) std::printf("%d ", v);
  std::printf("\n      oracle: ");
  for (int v : want) std::printf("%d ", v);
  std::printf("\n  %-22s %d/8 leading tokens match  %s\n", "TEXT_stream", match,
              match == 8 ? "PASS" : "FAIL");
  if (match != 8) ++failures;

  std::printf("[qwen35_text] %s\n", failures == 0 ? "ALL PASS" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
