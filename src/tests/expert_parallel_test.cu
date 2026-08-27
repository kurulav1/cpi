// Verifies expert-parallel MoE on a single GPU: shard the experts across N ranks, run the real MoE
// kernels (gate_up_geglu + down_accum) on each rank's expert sub-matrix with dispatched local
// indices
// + masked weights, sum the per-rank outputs, and check the result matches the all-local MoE. Brick
// 3 of multi-GPU prep; the per-rank sum is the all_to_all-combine seam that becomes a collective on
// a real cluster. Single token, fp16 experts.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "engine/expert_parallel.hpp"
#include "runtime/kernels.cuh"

namespace {

constexpr int E = 8, H = 16, MI = 8, TK = 4;

// gate_up_geglu + down_accum over `n_experts` fp16 experts. gate_up = [n*2*MI, H], down = [n*H,
// MI].
void run_moe(const half* gate_up, const half* down, const half* x, const int* idx, const float* w,
             half* inter, half* y) {
  kernels::launch_moe_gate_up_geglu(gate_up, nullptr, 0, 0, x, idx, inter, MI, H, TK, 0);
  kernels::launch_moe_down_accum(down, nullptr, 0, 0, inter, idx, w, y, H, MI, TK, 0);
}

}  // namespace

int main() {
  std::mt19937 rng(9);
  std::normal_distribution<float> nd(0, 0.3f);
  std::vector<half> gate_up((size_t)E * 2 * MI * H), down((size_t)E * H * MI), x(H);
  for (auto& v : gate_up) v = __float2half(nd(rng));
  for (auto& v : down) v = __float2half(nd(rng));
  for (auto& v : x) v = __float2half(nd(rng));
  const std::vector<int> idx = {1, 3, 5, 7};  // spans multiple ranks
  const std::vector<float> w = {0.4f, 0.3f, 0.2f, 0.1f};

  half *dGU, *dD, *dX, *dInter, *dY;
  int* dIdx;
  float* dW;
  cudaMalloc(&dGU, gate_up.size() * sizeof(half));
  cudaMemcpy(dGU, gate_up.data(), gate_up.size() * sizeof(half), cudaMemcpyHostToDevice);
  cudaMalloc(&dD, down.size() * sizeof(half));
  cudaMemcpy(dD, down.data(), down.size() * sizeof(half), cudaMemcpyHostToDevice);
  cudaMalloc(&dX, H * sizeof(half));
  cudaMemcpy(dX, x.data(), H * sizeof(half), cudaMemcpyHostToDevice);
  cudaMalloc(&dInter, (size_t)TK * MI * sizeof(half));
  cudaMalloc(&dY, H * sizeof(half));
  cudaMalloc(&dIdx, TK * sizeof(int));
  cudaMalloc(&dW, TK * sizeof(float));

  // Reference: all experts local.
  cudaMemcpy(dIdx, idx.data(), TK * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dW, w.data(), TK * sizeof(float), cudaMemcpyHostToDevice);
  run_moe(dGU, dD, dX, dIdx, dW, dInter, dY);
  cudaDeviceSynchronize();
  std::vector<half> y_full(H);
  cudaMemcpy(y_full.data(), dY, H * sizeof(half), cudaMemcpyDeviceToHost);

  int fail = 0;
  // ws=3 uneven split; ws=8 is one expert per rank (finest split); ws=12 > E exercises the
  // degenerate path where the trailing ranks own an empty range and must contribute exactly zero.
  for (int ws : {2, 3, 4, 8, 12}) {
    std::vector<float> y_ep(H, 0.0f);
    for (int r = 0; r < ws; ++r) {
      int lo = 0, hi = 0;
      engine::expert_parallel_range(E, ws, r, &lo, &hi);
      if (lo >= hi) continue;  // empty stage (world_size > experts): owns no experts, contributes 0
      std::vector<int> lidx(TK);
      std::vector<float> mw(TK);
      engine::expert_parallel_dispatch(E, ws, r, idx.data(), w.data(), TK, lidx.data(), mw.data());
      cudaMemcpy(dIdx, lidx.data(), TK * sizeof(int), cudaMemcpyHostToDevice);
      cudaMemcpy(dW, mw.data(), TK * sizeof(float), cudaMemcpyHostToDevice);
      run_moe(dGU + (size_t)lo * 2 * MI * H, dD + (size_t)lo * H * MI, dX, dIdx, dW, dInter, dY);
      cudaDeviceSynchronize();
      std::vector<half> y_r(H);
      cudaMemcpy(y_r.data(), dY, H * sizeof(half), cudaMemcpyDeviceToHost);
      for (int i = 0; i < H; ++i)
        y_ep[i] += __half2float(y_r[i]);  // all_to_all combine (local sum)
    }
    float maxabs = 0, denom = 1e-6f;
    for (int i = 0; i < H; ++i) {
      const float fa = __half2float(y_full[i]);
      maxabs = std::max(maxabs, std::fabs(fa - y_ep[i]));
      denom = std::max(denom, std::fabs(fa));
    }
    const float rel = maxabs / denom;
    const bool pass = rel < 1e-2f;
    std::printf("%s[world=%d]: expert-parallel MoE vs all-local, max rel diff %.2e\n",
                pass ? "PASS" : "FAIL", ws, rel);
    fail |= pass ? 0 : 1;
  }
  return fail;
}
