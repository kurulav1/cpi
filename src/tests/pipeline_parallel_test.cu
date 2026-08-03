// Verifies pipeline-parallel layer partitioning ON A SINGLE GPU: split a stack of L transformer-ish
// layers into P contiguous stages (all on device 0), run them stage by stage passing the activation
// across each stage boundary through a FRESH device buffer (the local stand-in for the cross-rank
// point-to-point send), and check the final activation matches the unsharded (P=1) forward. Brick 4 of
// multi-GPU prep; the stage handoff is the seam that becomes a P2P transfer on a real cluster. The
// per-layer op is a distinct fp16 GEMV (matmuls don't commute, so a dropped/reordered layer or a
// mis-wired handoff changes the result). Decode case (batch=1).
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "engine/pipeline_parallel.hpp"

namespace {

constexpr int H = 64;  // hidden size
constexpr int L = 6;   // layers in the stack

// y[H] = W[H,H] @ x[H], column-major W (ld = H), fp16 in/out, fp32 accumulate.
void gemv(cublasHandle_t handle, const half* dW, const half* dX, half* dY) {
  const float alpha = 1.0f, beta = 0.0f;
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, H, 1, H, &alpha, dW, CUDA_R_16F, H, dX, CUDA_R_16F, H,
               &beta, dY, CUDA_R_16F, H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

// Run layers [lo, hi) in order, reading dIn and writing dOut (may alias distinct buffers). Ping-pongs
// through dTmp so each layer's output feeds the next. lo == hi copies dIn -> dOut unchanged (identity
// stage). Returns nothing; dOut holds the stage output.
void run_layers(cublasHandle_t handle, const std::vector<half*>& dW, int lo, int hi, const half* dIn,
                half* dOut, half* dTmp) {
  if (lo >= hi) {  // empty stage: pass the activation through untouched
    cudaMemcpy(dOut, dIn, H * sizeof(half), cudaMemcpyDeviceToDevice);
    return;
  }
  const half* src = dIn;
  for (int l = lo; l < hi; ++l) {
    half* dst = (l == hi - 1) ? dOut : dTmp;
    gemv(handle, dW[l], src, dst);
    cudaDeviceSynchronize();
    src = dst;
  }
}

}  // namespace

int main() {
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.0f, 0.15f);  // small so a 6-deep fp16 chain stays finite

  // L distinct random layer weights + an input vector.
  std::vector<std::vector<half>> W(L, std::vector<half>((size_t)H * H));
  for (auto& w : W)
    for (auto& v : w) v = __float2half(nd(rng));
  std::vector<half> x(H);
  for (auto& v : x) v = __float2half(nd(rng));

  cublasHandle_t handle;
  cublasCreate(&handle);

  std::vector<half*> dW(L);
  for (int l = 0; l < L; ++l) {
    cudaMalloc(&dW[l], (size_t)H * H * sizeof(half));
    cudaMemcpy(dW[l], W[l].data(), (size_t)H * H * sizeof(half), cudaMemcpyHostToDevice);
  }
  half *dX, *dA, *dB, *dTmp;
  cudaMalloc(&dX, H * sizeof(half));
  cudaMalloc(&dA, H * sizeof(half));
  cudaMalloc(&dB, H * sizeof(half));
  cudaMalloc(&dTmp, H * sizeof(half));
  cudaMemcpy(dX, x.data(), H * sizeof(half), cudaMemcpyHostToDevice);

  // Reference: whole stack on one rank.
  run_layers(handle, dW, 0, L, dX, dA, dTmp);
  cudaDeviceSynchronize();
  std::vector<half> ref(H);
  cudaMemcpy(ref.data(), dA, H * sizeof(half), cudaMemcpyDeviceToHost);

  int fail = 0;
  // P=6 -> one layer per stage; P=7 -> a stage with no layers (identity handoff); P=4 -> uneven split.
  for (int P : {2, 3, 4, 6, 7}) {
    // Activation flows rank 0 -> 1 -> ... -> P-1. Each boundary copies into a FRESH buffer (dA<->dB
    // ping-pong) to model the cross-rank send and catch any pointer/size handoff bug.
    cudaMemcpy(dA, dX, H * sizeof(half), cudaMemcpyDeviceToDevice);
    half* cur = dA;
    half* nxt = dB;
    for (int r = 0; r < P; ++r) {
      int lo = 0, hi = 0;
      engine::pipeline_parallel_stage_range(L, P, r, &lo, &hi);
      run_layers(handle, dW, lo, hi, cur, nxt, dTmp);
      cudaDeviceSynchronize();
      std::swap(cur, nxt);  // `cur` now holds this stage's output = next stage's input
    }
    std::vector<half> y(H);
    cudaMemcpy(y.data(), cur, H * sizeof(half), cudaMemcpyDeviceToHost);

    float maxabs = 0.0f, denom = 1e-6f;
    for (int i = 0; i < H; ++i) {
      const float fa = __half2float(ref[i]), fb = __half2float(y[i]);
      maxabs = std::max(maxabs, std::fabs(fa - fb));
      denom = std::max(denom, std::fabs(fa));
    }
    const float rel = maxabs / denom;
    // Same device ops in the same order as the reference, so this is bit-exact; a tiny epsilon guards
    // only against a stray non-determinism. Any real partition/handoff bug moves rel far past this.
    const bool pass = rel < 1e-3f;
    // Verify the owner lookup is the exact inverse of the stage split for every layer.
    int owner_ok = 1;
    for (int l = 0; l < L; ++l) {
      int lo = 0, hi = 0;
      const int owner = engine::pipeline_parallel_owner(L, P, l);
      engine::pipeline_parallel_stage_range(L, P, owner, &lo, &hi);
      if (l < lo || l >= hi) owner_ok = 0;
    }
    const bool ok = pass && owner_ok;
    std::printf("%s[world=%d]: pipeline vs unsharded max rel diff %.2e, owner-map %s\n",
                ok ? "PASS" : "FAIL", P, rel, owner_ok ? "consistent" : "BROKEN");
    fail |= ok ? 0 : 1;
  }

  cublasDestroy(handle);
  return fail;
}
