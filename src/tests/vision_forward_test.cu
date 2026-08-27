// Parity gate for the CUDA vision tower.
//
// Reads the blob written by tools/gemma4_vision_oracle.py (a fixed synthetic image and
// the soft tokens HuggingFace produces for it), runs the same input through
// PlanCudaEngine::encode_image, and compares.
//
// The comparison is cosine similarity plus a relative error, not bit-equality: we run
// fp16 and the oracle runs fp32. A structural bug (wrong RoPE axis, wrong pooling cell,
// a transposed position table) destroys cosine similarity, so this catches the things
// that matter while tolerating fp16 rounding.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

#include "engine/plan_cuda_engine.hpp"

namespace {

template <typename T>
std::vector<T> read_n(std::ifstream& f, std::size_t n) {
  std::vector<T> v(n);
  f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * sizeof(T)));
  return v;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string model = argc > 1 ? argv[1] : "artifacts/hub/google__gemma-4-E2B-it/hf";
  const std::string blob = argc > 2 ? argv[2] : "artifacts/vision_oracle_e2b.bin";

  std::ifstream f(blob, std::ios::binary);
  if (!f) {
    std::printf("cannot open %s: run tools/gemma4_vision_oracle.py first\n", blob.c_str());
    return 2;
  }
  int header[5];
  f.read(reinterpret_cast<char*>(header), sizeof(header));
  const int patches = header[0], patch_dim = header[1], out_tokens = header[2],
            text_hidden = header[3];
  std::printf("oracle: patches=%d patch_dim=%d soft_tokens=%d text_hidden=%d\n", patches, patch_dim,
              out_tokens, text_hidden);

  const auto pixels = read_n<float>(f, static_cast<std::size_t>(patches) * patch_dim);
  const auto pos_x = read_n<int>(f, patches);
  const auto pos_y = read_n<int>(f, patches);
  const auto expect = read_n<float>(f, static_cast<std::size_t>(out_tokens) * text_hidden);

  engine::PlanCudaEngine eng;
  eng.open(model, 4096);
  if (!eng.has_vision()) {
    std::printf("FAIL: model reports no vision tower\n");
    return 1;
  }
  const auto& v = eng.vision_config();
  std::printf("engine: vision hidden=%d layers=%d heads=%d head_dim=%d pool=%d\n", v.hidden,
              v.layers, v.heads, v.head_dim, v.pooling_kernel);

  // Stage taps first: localise a divergence instead of guessing at it.
  const std::size_t vn = static_cast<std::size_t>(patches) * v.hidden;
  const auto exp_patch = read_n<float>(f, vn);
  const auto exp_enc = read_n<float>(f, vn);
  auto cosine = [](const std::vector<float>& a, const std::vector<float>& b, int rows, int cols) {
    double worst = 1.0;
    for (int t = 0; t < rows; ++t) {
      double dot = 0, na = 0, nb = 0;
      for (int i = 0; i < cols; ++i) {
        const double x = a[static_cast<std::size_t>(t) * cols + i];
        const double y = b[static_cast<std::size_t>(t) * cols + i];
        dot += x * y;
        na += x * x;
        nb += y * y;
      }
      worst = std::min(worst, dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12));
    }
    return worst;
  };
  const auto exp_l1 = read_n<float>(f, vn);
  // stage = how many encoder layers to run. One layer separates a STRUCTURAL bug
  // (cosine collapses immediately) from a compounding one (drifts over 16 layers).
  const auto got_patch = eng.encode_image_stage(pixels, pos_x, pos_y, 0);
  const auto got_l1 = eng.encode_image_stage(pixels, pos_x, pos_y, 1);
  const auto got_enc = eng.encode_image_stage(pixels, pos_x, pos_y, v.layers);
  std::printf("  patch_embed only  : worst cosine %.6f\n",
              cosine(got_patch, exp_patch, patches, v.hidden));
  std::printf("  after 1 layer     : worst cosine %.6f\n",
              cosine(got_l1, exp_l1, patches, v.hidden));
  std::printf("  after all layers  : worst cosine %.6f\n",
              cosine(got_enc, exp_enc, patches, v.hidden));

  // Optional intra-layer probe: Q straight after the 2-D RoPE in layer 0 (ops 0..7),
  // versus HF's. Isolates the RoPE kernel from everything downstream of it.
  if (const char* qfile = std::getenv("CPI_VISION_QROPE")) {
    std::ifstream qf(qfile, std::ios::binary);
    if (qf) {
      int qh[2];
      qf.read(reinterpret_cast<char*>(qh), sizeof(qh));
      const auto exp_q = read_n<float>(qf, static_cast<std::size_t>(qh[0]) * qh[1]);
      const int qstage =
          std::getenv("CPI_VISION_QSTAGE") ? std::atoi(std::getenv("CPI_VISION_QSTAGE")) : -8;
      const auto got_q = eng.encode_image_stage(pixels, pos_x, pos_y, qstage);
      std::printf(
          "  Q after 2-D RoPE  : worst cosine %.6f  (ours %.4f %.4f %.4f | hf %.4f %.4f %.4f)\n",
          cosine(got_q, exp_q, qh[0], qh[1]), got_q[0], got_q[1], got_q[2], exp_q[0], exp_q[1],
          exp_q[2]);
    }
  }

  const std::vector<float> got = eng.encode_image(pixels, pos_x, pos_y, out_tokens);
  if (got.size() != expect.size()) {
    std::printf("FAIL: size %zu vs expected %zu\n", got.size(), expect.size());
    return 1;
  }

  // Per-soft-token cosine similarity, plus the worst relative error overall.
  double worst_cos = 1.0, worst_rel = 0.0;
  for (int t = 0; t < out_tokens; ++t) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < text_hidden; ++i) {
      const double a = got[static_cast<std::size_t>(t) * text_hidden + i];
      const double b = expect[static_cast<std::size_t>(t) * text_hidden + i];
      dot += a * b;
      na += a * a;
      nb += b * b;
      const double denom = std::max(1.0, std::fabs(b));
      worst_rel = std::max(worst_rel, std::fabs(a - b) / denom);
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    worst_cos = std::min(worst_cos, cos);
  }

  std::printf("\nworst cosine similarity: %.6f   worst relative error: %.4f\n", worst_cos,
              worst_rel);
  std::printf("ours     [0,:4] = %.4f %.4f %.4f %.4f\n", got[0], got[1], got[2], got[3]);
  std::printf("oracle   [0,:4] = %.4f %.4f %.4f %.4f\n", expect[0], expect[1], expect[2],
              expect[3]);

  const bool ok = worst_cos > 0.999 && worst_rel < 0.05;
  std::printf("\n%s\n", ok ? "PARITY OK (vision soft tokens match the HF reference)"
                           : "PARITY FAILED (vision features diverge from the HF reference)");
  return ok ? 0 : 1;
}
