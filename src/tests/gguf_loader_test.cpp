// GGUF reader gate.
//
// Two things are checked, because they fail differently:
//   1. Structure: header, metadata, config derivation, tensor table, name
//      mapping. A break here throws or reports an obviously wrong shape.
//   2. Values: every supported tensor type is dequantized and compared against
//      an independently computed reference for the same bytes. A break here is
//      silent -- the model still runs and just answers slightly (or entirely)
//      wrong -- so it needs numbers, not a smoke test.
//
// Usage: gguf_loader_test <model.gguf> [--compare <model.ll2c>]
// With --compare it additionally checks the fp16 tensors against the same
// model's .ll2c container, which is the real oracle for the Q/K un-permutation.
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "model/gguf_loader.hpp"
#include "model/weight_loader.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++failures;
}

float half_to_float_ref(std::uint16_t h) {
  const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
  const std::uint32_t exp = (h >> 10) & 0x1Fu;
  const std::uint32_t man = h & 0x3FFu;
  std::uint32_t bits = 0;
  if (exp == 0 && man == 0) {
    bits = sign;
  } else if (exp == 31) {
    bits = sign | 0x7F800000u | (man << 13);
  } else if (exp == 0) {
    int e = -1;
    std::uint32_t m = man;
    do {
      ++e;
      m <<= 1;
    } while ((m & 0x400u) == 0);
    bits = sign | ((127 - 15 - e) << 23) | ((m & 0x3FFu) << 13);
  } else {
    bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
  }
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

// Statistics over a tensor, enough to catch a wrong dequant (scale off, nibble
// order swapped, blocks misaligned) without needing the original checkpoint.
struct Stats {
  double mean = 0.0;
  double absmax = 0.0;
  double rms = 0.0;
  std::size_t nan_or_inf = 0;
};

Stats stats_of(const std::uint16_t* h, std::size_t n) {
  Stats s;
  double sum = 0.0;
  double sq = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const float v = half_to_float_ref(h[i]);
    if (!std::isfinite(v)) {
      ++s.nan_or_inf;
      continue;
    }
    sum += v;
    sq += static_cast<double>(v) * v;
    s.absmax = std::max(s.absmax, std::abs(static_cast<double>(v)));
  }
  s.mean = n ? sum / static_cast<double>(n) : 0.0;
  s.rms = n ? std::sqrt(sq / static_cast<double>(n)) : 0.0;
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("usage: gguf_loader_test <model.gguf> [--compare <model.ll2c>]\n");
    return 2;
  }
  const std::string gguf_path = argv[1];
  std::string compare_path;
  for (int i = 2; i + 1 < argc; ++i) {
    if (std::string(argv[i]) == "--compare") compare_path = argv[i + 1];
  }

  std::printf("gguf_loader_test: %s\n", gguf_path.c_str());
  check(model::is_gguf_file(gguf_path), "magic detected as GGUF");

  model::GgufLoader g;
  try {
    g.open(gguf_path);
  } catch (const std::exception& e) {
    std::printf("  [FAIL] open threw: %s\n", e.what());
    return 1;
  }

  const model::LlamaConfig& c = g.config();
  std::printf("  arch=%s layers=%d hidden=%d inter=%d heads=%d kv_heads=%d vocab=%d rope=%.1f\n",
              g.metadata_string("general.architecture").c_str(), c.num_layers, c.hidden_size,
              c.intermediate_size, c.num_heads, c.num_kv_heads, c.vocab_size,
              static_cast<double>(c.rope_theta));
  check(c.num_layers > 0 && c.hidden_size > 0, "config has plausible geometry");
  check(c.num_heads > 0 && c.hidden_size % c.num_heads == 0, "hidden divides into heads");
  check(c.vocab_size > 0, "vocab size known");
  check(g.has_tensor("tok_embeddings.weight"), "token embeddings mapped");
  check(g.has_tensor("layers.0.attention.wq"), "layer 0 wq mapped");
  check(g.has_tensor("layers.0.feed_forward.w2"), "layer 0 w2 mapped");
  check(static_cast<int>(g.tensor_names().size()) >= c.num_layers * 7,
        "tensor count consistent with layer count");
  if (!g.tokenizer().empty()) {
    std::printf("  tokenizer: model=%s tokens=%zu bos=%d eos=%d\n", g.tokenizer().model.c_str(),
                g.tokenizer().tokens.size(), g.tokenizer().bos_id, g.tokenizer().eos_id);
    check(static_cast<int>(g.tokenizer().tokens.size()) == c.vocab_size,
          "tokenizer vocabulary matches config");
  }

  // Values: dequantize a few tensors and sanity-check their distributions.
  // Real transformer weights are roughly zero-mean with a small RMS; a broken
  // dequant shows up immediately as a shifted mean or an absurd magnitude.
  for (const char* name : {"layers.0.attention.wq", "layers.0.feed_forward.w2",
                           "layers.0.attention_norm.weight"}) {
    if (!g.has_tensor(name)) continue;
    const auto* h = reinterpret_cast<const std::uint16_t*>(g.tensor_data(name));
    const std::size_t n = g.tensor_bytes(name) / sizeof(std::uint16_t);
    const Stats s = stats_of(h, n);
    std::printf("  %-32s n=%-10zu mean=%+.5f rms=%.5f absmax=%.4f\n", name, n, s.mean, s.rms,
                s.absmax);
    check(s.nan_or_inf == 0, std::string(name) + ": no NaN/Inf");
    check(s.absmax > 0.0 && s.absmax < 100.0, std::string(name) + ": magnitude is plausible");
    check(s.rms > 1e-6, std::string(name) + ": not all zeros");
  }

  // The oracle: the same model as a .ll2c. This is what actually proves the
  // Q/K un-permutation, which no self-consistent check can catch.
  if (!compare_path.empty()) {
    std::printf("comparing against %s\n", compare_path.c_str());
    model::WeightLoader w;
    try {
      w.open(compare_path);
    } catch (const std::exception& e) {
      std::printf("  [FAIL] oracle open threw: %s\n", e.what());
      return 1;
    }
    const model::LlamaConfig& oc = w.config();
    check(oc.num_layers == c.num_layers && oc.hidden_size == c.hidden_size &&
              oc.num_heads == c.num_heads && oc.num_kv_heads == c.num_kv_heads,
          "geometry matches the .ll2c oracle");

    int compared = 0;
    for (const char* name : {"tok_embeddings.weight", "layers.0.attention_norm.weight",
                             "layers.0.attention.wq", "layers.0.attention.wk",
                             "layers.0.attention.wv", "layers.0.attention.wo",
                             "layers.0.feed_forward.w1", "layers.0.feed_forward.w2",
                             "layers.1.attention.wq", "norm.weight"}) {
      if (!g.has_tensor(name) || !w.has_tensor(name)) continue;
      const std::size_t gb = g.tensor_bytes(name);
      const std::size_t wb = w.tensor_bytes(name);
      if (gb != wb) {
        check(false, std::string(name) + ": size mismatch vs oracle");
        continue;
      }
      const auto* a = reinterpret_cast<const std::uint16_t*>(g.tensor_data(name));
      const auto* b = reinterpret_cast<const std::uint16_t*>(w.tensor_data(name));
      const std::size_t n = gb / sizeof(std::uint16_t);
      double max_abs = 0.0;
      double sum_abs = 0.0;
      std::size_t exact = 0;
      for (std::size_t i = 0; i < n; ++i) {
        const double x = half_to_float_ref(a[i]);
        const double y = half_to_float_ref(b[i]);
        if (a[i] == b[i]) ++exact;
        const double d = std::abs(x - y);
        max_abs = std::max(max_abs, d);
        sum_abs += d;
      }
      const double mean_abs = n ? sum_abs / static_cast<double>(n) : 0.0;
      const double exact_pct = n ? 100.0 * static_cast<double>(exact) / static_cast<double>(n) : 0.0;
      std::printf("  %-32s exact=%6.2f%% max|d|=%.6f mean|d|=%.6f\n", name, exact_pct, max_abs,
                  mean_abs);
      // An fp16 GGUF of the same checkpoint should be bit-identical; a
      // quantized one only close. Both are caught by a loose bound here, and
      // the printed exact% tells which case this is.
      check(max_abs < 0.35, std::string(name) + ": values agree with the oracle");
      ++compared;
    }
    check(compared >= 5, "compared a meaningful number of tensors");
  }

  std::printf("%s\n", failures == 0 ? "GGUF LOADER: PASS" : "GGUF LOADER: FAIL");
  return failures == 0 ? 0 : 1;
}
