// Verifies LocalCollective::all_reduce_sum_fp16 on a single GPU: N per-rank buffers (rank r holds
// the constant r+1) reduce so every rank ends with sum(1..N). This is the collective seam the
// tensor- and expert-parallel combines route through; the LocalCollective is verifiable here,
// NcclCollective is the deferred cluster impl.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>

#include "engine/collective.hpp"

int main() {
  const int count = 1024;
  int fail = 0;
  for (int ws : {1, 2, 4, 8}) {
    std::vector<half*> send(ws), recv(ws);
    std::vector<const void*> sp(ws);
    std::vector<void*> rp(ws);
    for (int r = 0; r < ws; ++r) {
      cudaMalloc(&send[r], count * sizeof(half));
      cudaMalloc(&recv[r], count * sizeof(half));
      std::vector<half> h(count, __float2half(static_cast<float>(r + 1)));
      cudaMemcpy(send[r], h.data(), count * sizeof(half), cudaMemcpyHostToDevice);
      sp[r] = send[r];
      rp[r] = recv[r];
    }
    engine::LocalCollective col(ws);
    col.all_reduce_sum_fp16(sp, rp, count, 0);
    cudaDeviceSynchronize();

    const float expect = static_cast<float>(ws * (ws + 1) / 2);  // sum 1..ws
    int bad = 0;
    for (int r = 0; r < ws; ++r) {
      std::vector<half> h(count);
      cudaMemcpy(h.data(), recv[r], count * sizeof(half), cudaMemcpyDeviceToHost);
      for (int i = 0; i < count; ++i)
        if (std::fabs(__half2float(h[i]) - expect) > 1e-3f) {
          bad++;
          break;
        }
      cudaFree(send[r]);
      cudaFree(recv[r]);
    }
    const bool pass = (bad == 0);
    std::printf("%s[world=%d]: all_reduce_sum -> %.0f on every rank (%d wrong)\n",
                pass ? "PASS" : "FAIL", ws, expect, bad);
    fail |= pass ? 0 : 1;
  }
  return fail;
}
