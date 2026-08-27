// Device k-quant dequantization vs the host reference.
//
// The host implementation is already gated against an fp16 oracle (a Q4_K_M file
// compared tensor-by-tensor against the same checkpoint's .ll2c), so it is the
// right reference for the kernels. What this catches is the class of mistake
// that cost two reverted attempts elsewhere in this codebase: a bit-unpacking
// error produces plausible weights, never a crash, so only an exact comparison
// finds it.
//
// Runs on synthetic blocks always (deterministic, no fixture needed) and on real
// blocks when given a .gguf path.
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

struct TypeInfo {
  const char* name;
  kernels::KQuantType type;
  std::size_t block_bytes;
  void (*host_fn)(const std::uint8_t*, std::size_t, std::uint16_t*);
};

// Compares device against host for one type over `blocks` super-blocks of bytes.
void compare(const TypeInfo& t, const std::vector<std::uint8_t>& bytes, std::size_t blocks,
             const std::string& label) {
  const std::size_t n = blocks * model::kquant::kSuperBlock;
  std::vector<std::uint16_t> host(n, 0);
  t.host_fn(bytes.data(), blocks, host.data());

  std::uint8_t* d_in = nullptr;
  __half* d_out = nullptr;
  if (cudaMalloc(&d_in, bytes.size()) != cudaSuccess ||
      cudaMalloc(&d_out, n * sizeof(__half)) != cudaSuccess) {
    check(false, std::string(t.name) + " " + label + ": cudaMalloc failed");
    return;
  }
  cudaMemcpy(d_in, bytes.data(), bytes.size(), cudaMemcpyHostToDevice);
  kernels::launch_dequant_kquant(d_in, t.type, blocks, d_out, nullptr);
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    check(false, std::string(t.name) + " " + label + ": kernel error " + cudaGetErrorString(err));
    cudaFree(d_in);
    cudaFree(d_out);
    return;
  }
  std::vector<std::uint16_t> dev(n, 0);
  cudaMemcpy(dev.data(), d_out, n * sizeof(__half), cudaMemcpyDeviceToHost);
  cudaFree(d_in);
  cudaFree(d_out);

  std::size_t mismatches = 0;
  double max_abs = 0.0;
  std::size_t first_bad = 0;
  for (std::size_t i = 0; i < n; ++i) {
    if (dev[i] != host[i]) {
      if (mismatches == 0) first_bad = i;
      ++mismatches;
      max_abs = std::max(max_abs, std::abs(static_cast<double>(h2f(dev[i])) - h2f(host[i])));
    }
  }
  std::printf("  %-6s %-10s blocks=%-5zu mismatches=%zu max|d|=%.6f\n", t.name, label.c_str(),
              blocks, mismatches, max_abs);
  if (mismatches != 0) {
    std::printf("         first at %zu: device %.6f vs host %.6f\n", first_bad,
                static_cast<double>(h2f(dev[first_bad])),
                static_cast<double>(h2f(host[first_bad])));
  }
  // Bit-exact is the requirement: both sides run the same arithmetic in fp32 and
  // round once at the end, so anything else means the unpacking differs.
  check(mismatches == 0, std::string(t.name) + " " + label + ": device == host");
}

}  // namespace

int main(int argc, char** argv) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::printf("kquant_dequant_test: no CUDA device; skipping\n");
    return 0;
  }

  const TypeInfo types[] = {
      {"Q4_K", kernels::KQuantType::Q4_K, model::kquant::kQ4KBlockBytes,
       &model::kquant::dequant_q4_k},
      {"Q5_K", kernels::KQuantType::Q5_K, model::kquant::kQ5KBlockBytes,
       &model::kquant::dequant_q5_k},
      {"Q6_K", kernels::KQuantType::Q6_K, model::kquant::kQ6KResidentBytes,
       &model::kquant::dequant_q6_k},
  };

  // Synthetic: random bytes are a fine stress for the unpacking, since every bit
  // pattern in a block is legal (the scales are quantized fields, not validated).
  std::printf("kquant_dequant_test: synthetic blocks\n");
  std::mt19937 rng(12345);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  for (const TypeInfo& t : types) {
    constexpr std::size_t kBlocks = 64;
    std::vector<std::uint8_t> bytes(kBlocks * t.block_bytes);
    for (auto& b : bytes) b = static_cast<std::uint8_t>(byte_dist(rng));
    compare(t, bytes, kBlocks, "random");
  }

  // Real blocks, when a GGUF is supplied: exercises the actual scale ranges a
  // quantizer produces rather than uniform noise.
  if (argc > 1) {
    const std::string path = argv[1];
    std::printf("kquant_dequant_test: real blocks from %s\n", path.c_str());
    model::GgufLoader g;
    try {
      g.open(path);
    } catch (const std::exception& e) {
      std::printf("  [FAIL] open threw: %s\n", e.what());
      return 1;
    }
    int checked = 0;
    for (const auto& rt : g.raw_tensors()) {
      const TypeInfo* t = nullptr;
      // ggml type ids: 12 Q4_K, 13 Q5_K, 14 Q6_K.
      if (rt.type == 12) t = &types[0];
      if (rt.type == 13) t = &types[1];
      if (rt.type == 14) t = &types[2];
      if (t == nullptr || rt.elements % model::kquant::kSuperBlock != 0) continue;
      // A prefix is enough: the kernel is per-block, so block 0..N exercises it
      // as well as the whole tensor and keeps the test quick.
      const std::size_t blocks =
          std::min<std::size_t>(32, rt.elements / model::kquant::kSuperBlock);
      const std::byte* raw = g.raw_tensor_bytes(rt.name);
      if (raw == nullptr) continue;
      std::vector<std::uint8_t> bytes(blocks * t->block_bytes);
      std::memcpy(bytes.data(), raw, bytes.size());
      compare(*t, bytes, blocks, rt.name.substr(0, 10));
      if (++checked >= 6) break;
    }
    check(checked > 0, "found k-quant tensors to check");
  }

  std::printf("%s\n", failures == 0 ? "KQUANT DEQUANT: PASS" : "KQUANT DEQUANT: FAIL");
  return failures == 0 ? 0 : 1;
}
