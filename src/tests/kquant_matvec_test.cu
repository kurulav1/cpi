// Packed-weight k-quant matvec vs a host reference.
//
// The kernel multiplies without ever materializing the weight as fp16, which is
// the whole point of native k-quant support. The reference here dequantizes on
// the host (that implementation is gated against an fp16 oracle elsewhere) and
// accumulates in fp32, so a disagreement means the in-kernel unpack or the
// accumulation is wrong -- not that the weights differ.
//
// Tolerance rather than bit-exactness: the kernel accumulates in a different
// order (block reduction) than a serial host dot, so the two agree to fp32
// rounding, not exactly. A real unpacking error is orders of magnitude larger
// than that, which is what makes a loose bound sufficient here.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "model/gguf_kquant.hpp"
#include "model/gguf_loader.hpp"
#include "runtime/kernels.cuh"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++failures;
}

float h2f(std::uint16_t h) {
  __half v;
  std::memcpy(&v, &h, sizeof(v));
  return __half2float(v);
}

struct Case {
  const char* name;
  kernels::KQuantType type;
  std::size_t block_bytes;
  void (*host_dequant)(const std::uint8_t*, std::size_t, std::uint16_t*);
};

void run_case(const Case& c, int rows, int cols) {
  const int blocks_per_row = cols / static_cast<int>(model::kquant::kSuperBlock);
  const std::size_t row_bytes = static_cast<std::size_t>(blocks_per_row) * c.block_bytes;
  std::vector<std::uint8_t> w(static_cast<std::size_t>(rows) * row_bytes);
  std::mt19937 rng(987);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  for (auto& b : w) b = static_cast<std::uint8_t>(byte_dist(rng));

  // Random bytes are fine for the bit-exact dequant gate, but not for a dot
  // product: a random fp16 scale field is frequently huge, inf or NaN, and the
  // accumulation then overflows and compares nothing. Real blocks have small
  // scales, so the scale fields are overwritten with a plausible one and only
  // the quantized payload stays random.
  const __half small = __float2half(0.01f);
  std::uint16_t small_bits;
  std::memcpy(&small_bits, &small, sizeof(small_bits));
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < blocks_per_row; ++b) {
      std::uint8_t* p = w.data() + static_cast<std::size_t>(r) * row_bytes +
                        static_cast<std::size_t>(b) * c.block_bytes;
      if (c.type == kernels::KQuantType::Q6_K) {
        std::memcpy(p + 208, &small_bits, 2);  // d
      } else {
        std::memcpy(p, &small_bits, 2);      // d
        std::memcpy(p + 2, &small_bits, 2);  // dmin
      }
    }
  }

  std::vector<__half> x(cols);
  std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
  for (int i = 0; i < cols; ++i) x[i] = __float2half(xd(rng));

  // Host reference: dequantize each row, dot in fp32.
  std::vector<float> ref(rows, 0.0f);
  std::vector<std::uint16_t> row_fp16(static_cast<std::size_t>(cols));
  for (int r = 0; r < rows; ++r) {
    c.host_dequant(w.data() + static_cast<std::size_t>(r) * row_bytes,
                   static_cast<std::size_t>(blocks_per_row), row_fp16.data());
    double acc = 0.0;
    for (int i = 0; i < cols; ++i) {
      acc += static_cast<double>(h2f(row_fp16[i])) * __half2float(x[i]);
    }
    ref[r] = static_cast<float>(acc);
  }

  std::uint8_t* d_w = nullptr;
  __half* d_x = nullptr;
  __half* d_y = nullptr;
  cudaMalloc(&d_w, w.size());
  cudaMalloc(&d_x, x.size() * sizeof(__half));
  cudaMalloc(&d_y, static_cast<std::size_t>(rows) * sizeof(__half));
  cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
  kernels::launch_kquant_matvec(d_w, c.type, d_x, d_y, rows, cols, nullptr);
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    check(false, std::string(c.name) + ": kernel error " + cudaGetErrorString(err));
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y);
    return;
  }
  std::vector<__half> y(rows);
  cudaMemcpy(y.data(), d_y, y.size() * sizeof(__half), cudaMemcpyDeviceToHost);
  cudaFree(d_w);
  cudaFree(d_x);
  cudaFree(d_y);

  double worst_rel = 0.0;
  double scale = 0.0;
  for (int r = 0; r < rows; ++r) scale = std::max(scale, std::abs(static_cast<double>(ref[r])));
  for (int r = 0; r < rows; ++r) {
    const double got = __half2float(y[r]);
    // Relative to the largest output, so a near-zero row does not dominate.
    worst_rel = std::max(worst_rel, std::abs(got - ref[r]) / (scale > 0.0 ? scale : 1.0));
  }
  std::printf("  %-6s rows=%-4d cols=%-5d worst_rel=%.5f (scale %.3f)\n", c.name, rows, cols,
              worst_rel, scale);
  // fp16 output rounding alone is ~1e-3 relative; anything structural is far larger.
  check(worst_rel < 0.01, std::string(c.name) + ": matches the host reference");
}

}  // namespace

// Real weights: the same projection multiplied two ways -- once from the packed
// blocks by the kernel, once from the fp16 the loader produces. This is the last
// unknown before residency: synthetic blocks prove the arithmetic, only a real
// tensor proves the layout assumptions (row/col order, block-per-row stride)
// against a file a quantizer actually wrote.
void run_real(const std::string& gguf_path) {
  model::GgufLoader g;
  try {
    g.open(gguf_path);
  } catch (const std::exception& e) {
    std::printf("  [FAIL] open threw: %s\n", e.what());
    ++failures;
    return;
  }
  int checked = 0;
  for (const auto& name : g.tensor_names()) {
    const auto pk = g.packed_kquant(name);
    if (!pk.valid() || pk.rows < 2 || pk.cols % 256 != 0) continue;

    const int rows = std::min(pk.rows, 32);  // a slice is enough; rows are independent
    const std::size_t row_bytes = pk.bytes / static_cast<std::size_t>(pk.rows);
    std::vector<__half> x(pk.cols);
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    for (int i = 0; i < pk.cols; ++i) x[i] = __float2half(xd(rng));

    // Reference from the loader's fp16 expansion of the same tensor.
    const auto* fp16 = reinterpret_cast<const std::uint16_t*>(g.tensor_data(name));
    std::vector<float> ref(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
      double acc = 0.0;
      for (int i = 0; i < pk.cols; ++i) {
        acc += static_cast<double>(h2f(fp16[static_cast<std::size_t>(r) * pk.cols + i])) *
               __half2float(x[i]);
      }
      ref[r] = static_cast<float>(acc);
    }

    std::uint8_t* d_w = nullptr;
    __half* d_x = nullptr;
    __half* d_y = nullptr;
    cudaMalloc(&d_w, row_bytes * rows);
    cudaMalloc(&d_x, x.size() * sizeof(__half));
    cudaMalloc(&d_y, static_cast<std::size_t>(rows) * sizeof(__half));
    cudaMemcpy(d_w, pk.data, row_bytes * rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    kernels::launch_kquant_matvec(d_w, static_cast<kernels::KQuantType>(pk.kind), d_x, d_y, rows,
                                  pk.cols, nullptr);
    cudaDeviceSynchronize();
    std::vector<__half> y(rows);
    cudaMemcpy(y.data(), d_y, y.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y);

    double scale = 0.0;
    for (int r = 0; r < rows; ++r) scale = std::max(scale, std::abs(static_cast<double>(ref[r])));
    double worst = 0.0;
    for (int r = 0; r < rows; ++r) {
      worst = std::max(worst, std::abs(__half2float(y[r]) - ref[r]) / (scale > 0.0 ? scale : 1.0));
    }
    std::printf("  %-28s kind=%d rows=%-5d cols=%-5d worst_rel=%.5f\n", name.c_str(), pk.kind,
                pk.rows, pk.cols, worst);
    check(worst < 0.01, name + ": packed matvec matches the fp16 expansion");
    if (++checked >= 4) break;
  }
  check(checked > 0, "found packed k-quant tensors to check");
}

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("kquant_matvec_test: no CUDA device; skipping\n");
    return 0;
  }
  const Case cases[] = {
      {"Q4_K", kernels::KQuantType::Q4_K, model::kquant::kQ4KBlockBytes,
       &model::kquant::dequant_q4_k},
      {"Q5_K", kernels::KQuantType::Q5_K, model::kquant::kQ5KBlockBytes,
       &model::kquant::dequant_q5_k},
      {"Q6_K", kernels::KQuantType::Q6_K, model::kquant::kQ6KBlockBytes,
       &model::kquant::dequant_q6_k},
  };
  std::printf("kquant_matvec_test: packed matvec vs host dequant + fp32 dot\n");
  for (const Case& c : cases) {
    run_case(c, 8, 512);      // several rows, two super-blocks each
    run_case(c, 64, 4096);    // a realistic projection shape
  }
  if (argc > 1) {
    std::printf("kquant_matvec_test: real weights from %s\n", argv[1]);
    run_real(argv[1]);
  }
  std::printf("%s\n", failures == 0 ? "KQUANT MATVEC: PASS" : "KQUANT MATVEC: FAIL");
  return failures == 0 ? 0 : 1;
}
