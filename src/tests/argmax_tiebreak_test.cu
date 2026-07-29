// Regression test for the batched greedy argmax tie-break.
//
// warp_argmax reduces with __shfl_down_sync; a plain `>` keeps the lower LANE on a tie, but the
// strided per-thread scan means the lower lane does not hold the lower index. The host greedy path
// (std::max_element) resolves ties to the lowest INDEX, and the sanitize clamp to +/-80
// manufactures exact ties out of distinct raw logits -- so a lowest-lane rule makes device greedy
// pick a different token than the host. Each row below places the lower index at a higher lane so
// the two rules disagree; the test passes only when the device matches std::max_element (lowest
// index wins).
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

#include "runtime/kernels.cuh"

#define CK(x)                                                                          \
  do {                                                                                 \
    cudaError_t e = (x);                                                               \
    if (e != cudaSuccess) {                                                            \
      std::printf("CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
      return 2;                                                                        \
    }                                                                                  \
  } while (0)

int main() {
  const int batch = 4;
  const int vocab = 512;  // launch uses 256 threads/block; index i is scanned by thread i%256
  std::vector<float> h(static_cast<std::size_t>(batch) * vocab, 0.0f);
  auto at = [&](int b, int i) -> float& { return h[static_cast<std::size_t>(b) * vocab + i]; };

  // Row 0: clamp-manufactured tie. Both raw logits exceed 80 and saturate to 80.0f. Index 5 is at
  // lane 5; index 257 is at lane 1 (257 % 256 = 1). A lowest-lane rule would pick 257; correct = 5.
  at(0, 5) = 100.0f;
  at(0, 257) = 100.0f;
  // Row 1: unique max, sanity that the reduction still finds it.
  at(1, 10) = 50.0f;
  // Row 2: genuine (unclamped) equal values across threads. Index 6 at lane 6, index 257 at lane 1.
  // Lowest-lane rule picks 257; correct = 6.
  at(2, 6) = 60.0f;
  at(2, 257) = 60.0f;
  // Row 3: EOS suppression. The clamped max (index 4) is blocked, so the winner is the next (9).
  at(3, 4) = 100.0f;
  at(3, 9) = 70.0f;

  std::vector<int> blocked = {-1, -1, -1, 4};
  const int expected[batch] = {5, 10, 6, 9};

  float* d_logits = nullptr;
  int* d_blocked = nullptr;
  int* d_out = nullptr;
  CK(cudaMalloc(&d_logits, h.size() * sizeof(float)));
  CK(cudaMalloc(&d_blocked, batch * sizeof(int)));
  CK(cudaMalloc(&d_out, batch * sizeof(int)));
  CK(cudaMemcpy(d_logits, h.data(), h.size() * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_blocked, blocked.data(), batch * sizeof(int), cudaMemcpyHostToDevice));
  kernels::launch_batched_argmax(d_logits, vocab, d_blocked, d_out, batch, 0);
  CK(cudaDeviceSynchronize());
  std::vector<int> out(batch, -1);
  CK(cudaMemcpy(out.data(), d_out, batch * sizeof(int), cudaMemcpyDeviceToHost));

  int fails = 0;
  for (int b = 0; b < batch; ++b) {
    const bool ok = out[b] == expected[b];
    std::printf("row %d: got %d expected %d  %s\n", b, out[b], expected[b], ok ? "OK" : "FAIL");
    if (!ok) ++fails;
  }
  cudaFree(d_logits);
  cudaFree(d_blocked);
  cudaFree(d_out);
  std::printf("argmax_tiebreak_test: %s\n", fails == 0 ? "PASS" : "FAIL");
  return fails == 0 ? 0 : 1;
}
