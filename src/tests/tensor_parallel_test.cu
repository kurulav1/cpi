// Verifies TensorParallelLinear's sharding math ON A SINGLE GPU: split a weight across N simulated
// ranks (all mapped to device 0) and check the concatenated output matches the unsharded (world=1)
// forward. This is the first brick of multi-GPU prep; the sharding/combine logic is provable here;
// only the cross-device transport (NCCL) then needs a real cluster. Decode case (batch=1).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "engine/tensor_parallel.hpp"

namespace {

// Greedy row split identical to TensorParallelLinear::initialize.
void split(int out, int ws, std::vector<int>& off, std::vector<int>& rows) {
  off.resize(ws);
  rows.resize(ws);
  int rem = out;
  for (int r = 0; r < ws; ++r) {
    rows[r] = rem / (ws - r);
    off[r] = out - rem;
    rem -= rows[r];
  }
}

// Run the TP linear with `ws` ranks, all on device 0. W is the full column-major [out, in] weight
// (ld = out); each shard is repacked to its own column-major [rows_r, in] (ld = rows_r).
std::vector<half> run(const std::vector<half>& W, int out, int in, const half* dX, int batch,
                      int ws) {
  std::vector<int> off, rows;
  split(out, ws, off, rows);
  std::vector<std::vector<half>> shard(ws);
  std::vector<const void*> ptrs(ws);
  for (int r = 0; r < ws; ++r) {
    shard[r].resize((size_t)rows[r] * in);
    for (int k = 0; k < in; ++k)
      for (int i = 0; i < rows[r]; ++i)
        shard[r][(size_t)k * rows[r] + i] = W[(size_t)k * out + (off[r] + i)];
    ptrs[r] = shard[r].data();
  }
  engine::TensorParallelLinear tp;
  tp.initialize(ws, in, out, ptrs, std::vector<int>(ws, 0));  // every rank -> device 0
  half* dY = nullptr;
  cudaMalloc(&dY, (size_t)out * batch * sizeof(half));
  tp.forward(dX, batch, dY, 0);
  cudaStreamSynchronize(0);
  std::vector<half> y((size_t)out * batch);
  cudaMemcpy(y.data(), dY, y.size() * sizeof(half), cudaMemcpyDeviceToHost);
  cudaFree(dY);
  return y;
}

int case_ws(const std::vector<half>& W, int out, int in, const half* dX, int batch, int ws,
            const std::vector<half>& ref) {
  std::vector<half> y = run(W, out, in, dX, batch, ws);
  int exact = 0;
  float maxrel = 0.0f, denom = 1e-6f;
  for (size_t i = 0; i < ref.size(); ++i) {
    const std::uint16_t a = *reinterpret_cast<const std::uint16_t*>(&ref[i]);
    const std::uint16_t b = *reinterpret_cast<const std::uint16_t*>(&y[i]);
    if (a == b) exact++;
    const float fa = __half2float(ref[i]), fb = __half2float(y[i]);
    maxrel = std::max(maxrel, std::fabs(fa - fb));
    denom = std::max(denom, std::fabs(fa));
  }
  const float rel = maxrel / denom;
  const bool pass = rel < 1e-2f;  // sharding is exact math; any delta is cuBLAS tiling noise
  std::printf("%s[world=%d batch=%d]: %d/%zu bit-exact vs unsharded, max rel diff %.2e\n",
              pass ? "PASS" : "FAIL", ws, batch, exact, ref.size(), rel);
  return pass ? 0 : 1;
}

}  // namespace

int main() {
  const int out = 512, in = 256;
  std::mt19937 rng(3);
  std::normal_distribution<float> nd(0, 0.3f);
  std::vector<half> W((size_t)out * in);
  for (auto& v : W) v = __float2half(nd(rng));

  int fail = 0;
  for (int batch : {1, 4}) {  // batch=4 exercises the strided concat (prefill layout)
    std::vector<half> X((size_t)in * batch);
    for (auto& v : X) v = __float2half(nd(rng));
    half* dX = nullptr;
    cudaMalloc(&dX, X.size() * sizeof(half));
    cudaMemcpy(dX, X.data(), X.size() * sizeof(half), cudaMemcpyHostToDevice);

    const std::vector<half> ref = run(W, out, in, dX, batch, 1);  // unsharded reference
    fail |= case_ws(W, out, in, dX, batch, 2, ref);
    fail |= case_ws(W, out, in, dX, batch, 4, ref);
    fail |= case_ws(W, out, in, dX, batch, 7, ref);  // uneven split (512 not divisible by 7)
    cudaFree(dX);
  }
  return fail;
}
