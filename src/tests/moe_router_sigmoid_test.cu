// Unit test for kernels::launch_moe_router_sigmoid_topk (the Kimi-K3-style MoE router).
// Verifies device == host reference for: (1) flat sigmoid top-16 (above the softmax router's top-8
// cap), (2) grouped node-limited routing (top-2-per-group score -> top groups -> top-k within),
// (3) selected weights are normalised sigmoid gates summing to 1. K3 uses 896 experts / top-16 /
// sigmoid+grouped; this gates the routing math in isolation ahead of wiring it to a model.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "runtime/kernels.cuh"

namespace {

// Host mirror of moe_router_sigmoid_topk_kernel. Takes the same half-rounded logits the device
// sees.
void ref_router(const std::vector<float>& g_in, int experts, int top_k, int n_group, int topk_group,
                std::vector<int>& idx, std::vector<float>& w) {
  std::vector<float> gate(experts);
  for (int e = 0; e < experts; ++e) gate[e] = 1.0f / (1.0f + std::exp(-g_in[e]));
  const int kMaxTopK = 32, kMaxGroups = 64;
  const int kk = std::min(top_k, kMaxTopK);
  const bool grouped = (n_group > 1 && topk_group > 0 && topk_group < n_group &&
                        n_group <= kMaxGroups && (experts % n_group) == 0);
  const int gsize = grouped ? experts / n_group : experts;
  std::vector<int> sel;
  if (grouped) {
    std::vector<float> gscore(n_group);
    for (int g = 0; g < n_group; ++g) {
      float b0 = -1, b1 = -1;
      for (int i = 0; i < gsize; ++i) {
        float v = gate[g * gsize + i];
        if (v > b0) {
          b1 = b0;
          b0 = v;
        } else if (v > b1) {
          b1 = v;
        }
      }
      gscore[g] = b0 + (b1 > 0 ? b1 : 0);
    }
    std::vector<char> gused(n_group, 0);
    for (int s = 0; s < topk_group; ++s) {
      int bg = -1;
      float bv = -1;
      for (int g = 0; g < n_group; ++g) {
        if (gused[g]) continue;
        if (gscore[g] > bv) {
          bv = gscore[g];
          bg = g;
        }
      }
      if (bg < 0) break;
      gused[bg] = 1;
      sel.push_back(bg);
    }
  }
  std::vector<int> picked(kk, -1);
  std::vector<float> pg(kk, 0);
  for (int k = 0; k < kk; ++k) {
    int be = -1;
    float bv = -1;
    for (int e = 0; e < experts; ++e) {
      if (grouped) {
        int g = e / gsize;
        bool ok = false;
        for (int x : sel)
          if (x == g) {
            ok = true;
            break;
          }
        if (!ok) continue;
      }
      bool used = false;
      for (int p = 0; p < k; ++p)
        if (picked[p] == e) {
          used = true;
          break;
        }
      if (used) continue;
      if (gate[e] > bv) {
        bv = gate[e];
        be = e;
      }
    }
    picked[k] = be;
    pg[k] = (be >= 0) ? gate[be] : 0;
  }
  float sum = 0;
  for (int k = 0; k < kk; ++k)
    if (picked[k] >= 0) sum += pg[k];
  const float inv = 1.0f / std::max(sum, 1e-8f);
  idx.assign(top_k, 0);
  w.assign(top_k, 0);
  for (int k = 0; k < top_k; ++k)
    if (k < kk && picked[k] >= 0) {
      idx[k] = picked[k];
      w[k] = pg[k] * inv;
    }
}

int run_case(const char* name, int experts, int top_k, int n_group, int topk_group,
             std::mt19937& rng) {
  // Well-separated logits (no near-ties at the selection boundary) -> unambiguous top-k.
  std::uniform_real_distribution<float> ud(-6.0f, 6.0f);
  std::vector<float> lf(experts);
  for (auto& v : lf) v = ud(rng);
  std::vector<half> lh(experts);
  std::vector<float> lhf(experts);
  for (int e = 0; e < experts; ++e) {
    lh[e] = __float2half(lf[e]);
    lhf[e] = __half2float(lh[e]);  // what both device and host reference consume
  }

  std::vector<int> ridx;
  std::vector<float> rw;
  ref_router(lhf, experts, top_k, n_group, topk_group, ridx, rw);

  half* dl;
  int* didx;
  float* dw;
  cudaMalloc(&dl, experts * sizeof(half));
  cudaMalloc(&didx, top_k * sizeof(int));
  cudaMalloc(&dw, top_k * sizeof(float));
  cudaMemcpy(dl, lh.data(), experts * sizeof(half), cudaMemcpyHostToDevice);
  kernels::launch_moe_router_sigmoid_topk(dl, experts, top_k, n_group, topk_group, didx, dw,
                                          nullptr);
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::printf("FAIL[%s]: kernel launch\n", name);
    return 1;
  }
  std::vector<int> hidx(top_k);
  std::vector<float> hw(top_k);
  cudaMemcpy(hidx.data(), didx, top_k * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(hw.data(), dw, top_k * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(dl);
  cudaFree(didx);
  cudaFree(dw);

  int fail = 0;
  float wsum = 0;
  for (int k = 0; k < top_k; ++k) {
    if (hidx[k] != ridx[k]) {
      std::printf("FAIL[%s]: idx[%d] device=%d host=%d\n", name, k, hidx[k], ridx[k]);
      fail = 1;
    }
    if (std::fabs(hw[k] - rw[k]) > 1e-4f) {
      std::printf("FAIL[%s]: w[%d] device=%.6f host=%.6f\n", name, k, hw[k], rw[k]);
      fail = 1;
    }
    wsum += hw[k];
  }
  if (std::fabs(wsum - 1.0f) > 1e-3f) {
    std::printf("FAIL[%s]: weights sum to %.5f, not 1\n", name, wsum);
    fail = 1;
  }
  // Grouped: every picked expert must live in a selected group.
  if (n_group > 1 && (experts % n_group) == 0) {
    const int gsize = experts / n_group;
    std::vector<float> gate(experts), gscore(n_group);
    for (int e = 0; e < experts; ++e) gate[e] = 1.0f / (1.0f + std::exp(-lhf[e]));
    for (int g = 0; g < n_group; ++g) {
      float b0 = -1, b1 = -1;
      for (int i = 0; i < gsize; ++i) {
        float v = gate[g * gsize + i];
        if (v > b0) {
          b1 = b0;
          b0 = v;
        } else if (v > b1)
          b1 = v;
      }
      gscore[g] = b0 + (b1 > 0 ? b1 : 0);
    }
    std::vector<float> sorted = gscore;
    std::sort(sorted.rbegin(), sorted.rend());
    const float cutoff = sorted[std::min(topk_group, n_group) - 1];
    for (int k = 0; k < top_k; ++k)
      if (hw[k] > 0 && gscore[hidx[k] / gsize] < cutoff) {
        std::printf("FAIL[%s]: picked expert %d in a non-top group\n", name, hidx[k]);
        fail = 1;
      }
  }
  if (!fail)
    std::printf("PASS[%s]: E=%d k=%d n_group=%d topk_group=%d\n", name, experts, top_k, n_group,
                topk_group);
  return fail;
}

}  // namespace

int main() {
  std::mt19937 rng(11);
  int fail = 0;
  fail |= run_case("flat-top16", 256, 16, 1, 0, rng);  // sigmoid, flat, above top-8 cap
  fail |= run_case("k3-grouped", 256, 16, 8, 4, rng);  // grouped node-limited (8 groups, top 4)
  fail |= run_case("k3-896", 896, 16, 8, 4, rng);      // K3 scale (896 experts / top-16)
  return fail;
}
