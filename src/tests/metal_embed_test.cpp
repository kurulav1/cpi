// MetalBertEmbedder against a HuggingFace fp32 reference.
//
//   ./metal_embed_test <model_dir> <oracle_dir>    (oracle_dir = tools/embed_oracle.py --out)
//
// cpi_embed emits 384 plausible-looking floats whether or not the encoder is correct, and the
// vector is never read by a human -- it goes straight into a cosine ranking. So "it produced
// numbers" is worth nothing here and this compares against a reference.
//
// WHAT THIS GATE CAN AND CANNOT SEE, both established by deliberately breaking the encoder:
//
//   * Wrong attention row_stride (passing the vision tower's fused-qkv 3*hidden where BERT has
//     separate q/k/v buffers): per-sentence cosine collapses to 0.17-0.68. CAUGHT, decisively.
//   * Tanh GELU instead of the exact erf form: cosine stays 1.00000 and max_abs moves from
//     ~1.9e-4 to ~3.7e-4. NOT CAUGHT. That difference is ~4e-4 per element and six layers is not
//     enough depth for it to surface above fp16 noise. The erf choice is therefore asserted from
//     bert_embedder.cu and the model's hidden_act, NOT proven here -- do not read a pass as
//     evidence about the activation.
//
// The tolerance is on the COSINE, because that is what an embedding is for: a port can differ
// element-wise and still rank identically, or match element-wise and rank wrongly.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "engine/metal_bert_embedder.hpp"
#include "model/json_mini.hpp"
#include "model/wordpiece_tokenizer.hpp"

namespace {

int failures = 0;

// MUST match SENTENCES in tools/embed_oracle.py, in order.
const char* kSentences[] = {
    "a photo of a cat",
    "a photo of a dog",
    "quantum chromodynamics",
    "hello",
    "The quick brown fox jumps over the lazy dog near the river bank at dawn.",
};
constexpr int kN = 5;

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

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 3) {
    std::printf("[metal_embed] SKIP: usage: metal_embed_test <model_dir> <oracle_dir>\n");
    return 0;
  }
  const std::string model_dir = argv[1];
  const std::string oracle_dir = argv[2];

  const std::vector<float> ref = read_f32(oracle_dir + "/embeddings.f32");

  engine::MetalBertEmbedder emb;
  model::WordPieceTokenizer tok;
  try {
    emb.initialize(model_dir);
    tok.load(model_dir, emb.config().lowercase, emb.config().strip_accents);
  } catch (const std::exception& e) {
    std::printf("[metal_embed] SKIP: %s\n", e.what());
    return 0;
  }

  const int H = emb.dim();
  if (static_cast<int>(ref.size()) != kN * H) {
    std::printf("  %-24s oracle has %zu floats, expected %d x %d  FAIL\n", "EMBED_SHAPE",
                ref.size(), kN, H);
    return 1;
  }
  std::printf("[metal_embed] dim=%d pool=%s\n", H,
              emb.config().pooling == engine::PoolingMode::Mean ? "mean" : "cls");

  std::vector<std::vector<float>> ours;
  double worst_cos = 1.0;
  float worst_abs = 0.0f;
  std::size_t nonfinite = 0;

  for (int i = 0; i < kN; ++i) {
    const std::vector<int> ids = tok.encode(kSentences[i], emb.max_tokens());
    const std::vector<float> v = emb.embed(ids);
    ours.push_back(v);
    const float* r = ref.data() + static_cast<std::size_t>(i) * H;

    double dot = 0.0;
    float mx = 0.0f;
    for (int d = 0; d < H; ++d) {
      // NaN counted explicitly: it compares false against everything, so a max-error loop over
      // an all-NaN vector reports 0 and passes every tolerance in this file.
      if (!std::isfinite(v[static_cast<std::size_t>(d)])) ++nonfinite;
      dot += static_cast<double>(v[static_cast<std::size_t>(d)]) * r[d];
      mx = std::max(mx, std::fabs(v[static_cast<std::size_t>(d)] - r[d]));
    }
    worst_cos = std::min(worst_cos, dot);
    worst_abs = std::max(worst_abs, mx);
    std::printf("      %-56s cos=%.5f  max_abs=%.5f  n=%zu\n", kSentences[i], dot, mx, ids.size());
  }

  if (nonfinite != 0) {
    std::printf("  %-24s %zu non-finite components  FAIL\n", "EMBED_FINITE", nonfinite);
    ++failures;
  }
  // 0.9995: the measured value is 1.00000 and the wrong-stride control scored 0.17, so anything
  // in between is a defect rather than drift. Not tightened to 0.9999 because the reference is
  // fp32 and this is fp16 -- leave room for the arithmetic, not for a bug.
  const bool ok = worst_cos >= 0.9995;
  std::printf("  %-24s worst_cos=%.5f  worst_max_abs=%.5f  tol_cos=0.9995  %s\n", "EMBED_VS_ORACLE",
              worst_cos, worst_abs, ok ? "PASS" : "FAIL");
  if (!ok) ++failures;

  // Semantic sanity, independent of the reference vectors: an encoder can match a reference that
  // is itself being read wrongly (wrong pooling, wrong normalisation) and still be useless for
  // retrieval. cat/dog must rank far above cat/quantum-chromodynamics.
  auto cos = [&](int a, int b) {
    double d = 0.0;
    for (int i = 0; i < H; ++i) {
      d += static_cast<double>(ours[static_cast<std::size_t>(a)][static_cast<std::size_t>(i)]) *
           ours[static_cast<std::size_t>(b)][static_cast<std::size_t>(i)];
    }
    return d;
  };
  const double near = cos(0, 1), far = cos(0, 2);
  const bool sane = near > 0.5 && far < 0.2 && near > far + 0.3;
  std::printf("  %-24s cos(cat,dog)=%.4f  cos(cat,qcd)=%.4f  %s\n", "EMBED_RANKS", near, far,
              sane ? "PASS" : "FAIL");
  if (!sane) ++failures;

  std::printf("[metal_embed] %s\n", failures == 0 ? "ALL PASS" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
