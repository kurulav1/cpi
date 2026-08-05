// Gates model::image::qwen2vl_preprocess against the HuggingFace Qwen2VLImageProcessor.
//
//   ./qwen2vl_preproc_test <oracle_dir>   (oracle_dir = tools/qwen35_preproc_oracle.py --out)
//
// The oracle wrote the exact uint8 image it fed the processor and the [patches, 1536] tensor the
// processor produced. This starts from those same bytes; so a mismatch is two implementations
// disagreeing, not two different images. Pure CPU: no Metal, so it runs on CI and Windows.
//
// The resize is the one step where a small deviation quietly costs accuracy (it is bicubic, and
// PIL's filter differs from a naive one), so the tolerance is reported, not just pass/fail.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "model/image_preprocess.hpp"
#include "model/json_mini.hpp"

namespace {

std::vector<std::uint8_t> read_u8(const std::string& path) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  std::fseek(f, 0, SEEK_END);
  const long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  std::vector<std::uint8_t> v(static_cast<std::size_t>(n));
  if (std::fread(v.data(), 1, static_cast<std::size_t>(n), f) != static_cast<std::size_t>(n)) {
    std::exit(2);
  }
  std::fclose(f);
  return v;
}

std::vector<float> read_f32(const std::string& path) {
  const std::vector<std::uint8_t> raw = read_u8(path);
  std::vector<float> v(raw.size() / sizeof(float));
  std::memcpy(v.data(), raw.data(), v.size() * sizeof(float));
  return v;
}

std::string read_text(const std::string& path) {
  const std::vector<std::uint8_t> raw = read_u8(path);
  return std::string(raw.begin(), raw.end());
}

}  // namespace

int main(int argc, char** argv) {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 2) {
    std::printf("[qwen2vl_preproc] SKIP: usage: qwen2vl_preproc_test <oracle_dir>\n");
    return 0;
  }
  const std::string dir = argv[1];
  const std::string man = read_text(dir + "/manifest.json");

  const std::string src = engine::mini::json_extract_object(man, "source_image");
  const int W = engine::mini::json_get_int(src, "width");
  const int H = engine::mini::json_get_int(src, "height");
  const std::string img_file = engine::mini::json_get_string(src, "file");

  const std::string proc = engine::mini::json_extract_object(man, "processor");
  const int patch = engine::mini::json_get_int(proc, "patch_size");
  const int temporal = engine::mini::json_get_int(proc, "temporal_patch_size");
  const int merge = engine::mini::json_get_int(proc, "merge_size");
  const std::string size = engine::mini::json_extract_object(proc, "size");
  const long min_px = static_cast<long>(engine::mini::json_get_int(size, "shortest_edge"));
  const long max_px = static_cast<long>(engine::mini::json_get_int(size, "longest_edge"));

  model::image::Image img;
  img.width = W;
  img.height = H;
  img.rgb = read_u8(dir + "/" + img_file);
  if (static_cast<int>(img.rgb.size()) != W * H * 3) {
    std::printf("  %-22s image is %zu bytes, expected %d  FAIL\n", "IMAGE_BYTES", img.rgb.size(),
                W * H * 3);
    return 1;
  }

  const float mean[3] = {0.5f, 0.5f, 0.5f};
  const float stdv[3] = {0.5f, 0.5f, 0.5f};
  const model::image::Qwen2VLPatches got =
      model::image::qwen2vl_preprocess(img, patch, temporal, merge, mean, stdv, min_px, max_px);

  int failures = 0;

  std::printf("[qwen2vl_preproc] %dx%d -> grid %dx%d, %zu patches x %d\n", W, H, got.grid_h,
              got.grid_w, got.pixels.size() / (3 * temporal * patch * patch),
              3 * temporal * patch * patch);

  const std::vector<float> ref = read_f32(dir + "/" +
                                          engine::mini::json_get_string(
                                              engine::mini::json_extract_object(man, "pixel_values"),
                                              "file"));
  if (ref.size() != got.pixels.size()) {
    std::printf("  %-22s ours %zu vs oracle %zu floats  FAIL\n", "PREPROC_SHAPE", got.pixels.size(),
                ref.size());
    return 1;
  }

  float mx = 0.0f;
  double sum = 0.0;
  std::size_t nonfinite = 0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    if (!std::isfinite(got.pixels[i])) ++nonfinite;
    const float d = std::fabs(got.pixels[i] - ref[i]);
    mx = std::max(mx, d);
    sum += d;
  }
  if (nonfinite != 0) {
    std::printf("  %-22s %zu non-finite values  FAIL\n", "PREPROC_FINITE", nonfinite);
    ++failures;
  }
  const double mean_abs = sum / static_cast<double>(ref.size());
  // Tolerance on the max. The normalise and patchify are exact integer/affine ops, so the only
  // source of error is the bicubic resample vs PIL's. 0.02 on a [-1,1] tensor is ~1%; tight
  // enough to catch a wrong filter or a transposed axis, loose enough for last-bit resample
  // differences. Both controls (wrong patch order, wrong normalise) blow past it.
  const bool ok = mx <= 0.02f;
  std::printf("  %-22s max_abs=%.5f  mean_abs=%.5f  tol=0.020  %s\n", "PREPROC_VS_ORACLE", mx,
              mean_abs, ok ? "PASS" : "FAIL");
  if (!ok) ++failures;

  std::printf("[qwen2vl_preproc] %s\n", failures == 0 ? "ALL PASS" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
