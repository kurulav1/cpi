// Verifies RowParallelLinear on a single GPU: split a weight by INPUT columns across N simulated
// ranks (all on device 0), run each rank's partial GEMM over its input slice, all-reduce (sum) the
// partials, and check the result matches the unsharded (world=1) forward. Brick 2 of multi-GPU prep;
// only the cross-device all-reduce (ncclAllReduce) then needs a real cluster. Decode case (batch=1).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "engine/tensor_parallel.hpp"

namespace {

void split(int in, int ws, std::vector<int>& off, std::vector<int>& rows) {
  off.resize(ws);
  rows.resize(ws);
  int rem = in;
  for (int r = 0; r < ws; ++r) {
    rows[r] = rem / (ws - r);
    off[r] = in - rem;
    rem -= rows[r];
  }
}

// W is column-major [out, in] (ld = out), so an input-column slice is CONTIGUOUS (W + off*out); the
// input x is [in] so its slice is contiguous too (dX + off). No repack needed.
std::vector<half> run(const std::vector<half>& W, int out, int in, const half* dX, int batch,
                      int ws) {
  std::vector<int> off, rows;
  split(in, ws, off, rows);
  std::vector<const void*> wptrs(ws), xptrs(ws);
  for (int r = 0; r < ws; ++r) {
    wptrs[r] = W.data() + (size_t)off[r] * out;
    xptrs[r] = dX + off[r];
  }
  engine::RowParallelLinear rp;
  rp.initialize(ws, in, out, wptrs, std::vector<int>(ws, 0));  // every rank -> device 0
  half* dY = nullptr;
  cudaMalloc(&dY, (size_t)out * batch * sizeof(half));
  rp.forward(xptrs, batch, dY, 0);
  cudaStreamSynchronize(0);
  std::vector<half> y((size_t)out * batch);
  cudaMemcpy(y.data(), dY, y.size() * sizeof(half), cudaMemcpyDeviceToHost);
  cudaFree(dY);
  return y;
}

int case_ws(const std::vector<half>& W, int out, int in, const half* dX, int batch, int ws,
            const std::vector<half>& ref) {
  std::vector<half> y = run(W, out, in, dX, batch, ws);
  float maxabs = 0.0f, denom = 1e-6f;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float fa = __half2float(ref[i]), fb = __half2float(y[i]);
    maxabs = std::max(maxabs, std::fabs(fa - fb));
    denom = std::max(denom, std::fabs(fa));
  }
  const float rel = maxabs / denom;
  // Row-parallel sums fp16 partials, so it is NOT bit-exact vs the single fp32-accumulated GEMM;
  // the delta is fp16 partial-rounding. A tight fp16 tolerance still catches any real sharding bug.
  const bool pass = rel < 2e-2f;
  std::printf("%s[world=%d]: row-parallel all-reduce vs unsharded, max rel diff %.2e\n",
              pass ? "PASS" : "FAIL", ws, rel);
  return pass ? 0 : 1;
}

}  // namespace

int main() {
  const int out = 512, in = 256, batch = 1;
  std::mt19937 rng(5);
  std::normal_distribution<float> nd(0, 0.3f);
  std::vector<half> W((size_t)out * in);
  for (auto& v : W) v = __float2half(nd(rng));
  std::vector<half> X((size_t)in * batch);
  for (auto& v : X) v = __float2half(nd(rng));

  half* dX = nullptr;
  cudaMalloc(&dX, X.size() * sizeof(half));
  cudaMemcpy(dX, X.data(), X.size() * sizeof(half), cudaMemcpyHostToDevice);

  const std::vector<half> ref = run(W, out, in, dX, batch, 1);  // unsharded reference
  int fail = 0;
  fail |= case_ws(W, out, in, dX, batch, 2, ref);
  fail |= case_ws(W, out, in, dX, batch, 4, ref);
  fail |= case_ws(W, out, in, dX, batch, 7, ref);  // uneven input split (256 not divisible by 7)
  cudaFree(dX);
  return fail;
}
