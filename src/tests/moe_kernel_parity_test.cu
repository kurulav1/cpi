#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/kernels.cuh"

namespace {

void require_cuda(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
  }
}

struct RouterRef {
  std::vector<int> idx;
  std::vector<float> prob;
};

RouterRef cpu_router_topk_softmax(const std::vector<float>& logits, int top_k) {
  const int experts = static_cast<int>(logits.size());
  const float max_logit = *std::max_element(logits.begin(), logits.end());
  std::vector<float> probs(static_cast<std::size_t>(experts), 0.0f);
  float sum = 0.0f;
  for (int i = 0; i < experts; ++i) {
    const float v = std::exp(logits[static_cast<std::size_t>(i)] - max_logit);
    probs[static_cast<std::size_t>(i)] = v;
    sum += v;
  }
  if (sum <= 0.0f) {
    sum = 1.0f;
  }
  for (float& p : probs) {
    p /= sum;
  }

  RouterRef out{};
  out.idx.assign(static_cast<std::size_t>(top_k), -1);
  out.prob.assign(static_cast<std::size_t>(top_k), 0.0f);
  std::vector<char> used(static_cast<std::size_t>(experts), 0);
  float picked = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    int best_i = -1;
    float best_v = -std::numeric_limits<float>::infinity();
    for (int i = 0; i < experts; ++i) {
      if (used[static_cast<std::size_t>(i)] != 0) {
        continue;
      }
      const float v = probs[static_cast<std::size_t>(i)];
      if (v > best_v) {
        best_v = v;
        best_i = i;
      }
    }
    if (best_i < 0) {
      break;
    }
    used[static_cast<std::size_t>(best_i)] = 1;
    out.idx[static_cast<std::size_t>(k)] = best_i;
    out.prob[static_cast<std::size_t>(k)] = best_v;
    picked += best_v;
  }
  if (picked > 0.0f) {
    for (float& p : out.prob) {
      p /= picked;
    }
  }
  return out;
}

void test_router_topk_softmax(cudaStream_t stream) {
  const std::vector<float> logits = {1.2f, -0.7f, 3.5f, 2.1f, -1.0f, 0.4f};
  const int experts = static_cast<int>(logits.size());
  const int top_k = 2;
  std::vector<__half> h_logits(static_cast<std::size_t>(experts));
  for (int i = 0; i < experts; ++i) {
    h_logits[static_cast<std::size_t>(i)] = __float2half(logits[static_cast<std::size_t>(i)]);
  }

  __half* d_logits = nullptr;
  int* d_idx = nullptr;
  float* d_prob = nullptr;
  require_cuda(cudaMalloc(&d_logits, static_cast<std::size_t>(experts) * sizeof(__half)), "cudaMalloc d_logits");
  require_cuda(cudaMalloc(&d_idx, static_cast<std::size_t>(top_k) * sizeof(int)), "cudaMalloc d_idx");
  require_cuda(cudaMalloc(&d_prob, static_cast<std::size_t>(top_k) * sizeof(float)), "cudaMalloc d_prob");
  require_cuda(cudaMemcpyAsync(d_logits,
                               h_logits.data(),
                               static_cast<std::size_t>(experts) * sizeof(__half),
                               cudaMemcpyHostToDevice,
                               stream),
               "cudaMemcpyAsync logits");

  kernels::launch_moe_router_topk_softmax(d_logits, experts, top_k, d_idx, d_prob, stream);
  require_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize router");

  std::vector<int> out_idx(static_cast<std::size_t>(top_k), -1);
  std::vector<float> out_prob(static_cast<std::size_t>(top_k), 0.0f);
  require_cuda(cudaMemcpy(out_idx.data(),
                          d_idx,
                          static_cast<std::size_t>(top_k) * sizeof(int),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy idx");
  require_cuda(cudaMemcpy(out_prob.data(),
                          d_prob,
                          static_cast<std::size_t>(top_k) * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy prob");

  const RouterRef ref = cpu_router_topk_softmax(logits, top_k);
  for (int k = 0; k < top_k; ++k) {
    if (out_idx[static_cast<std::size_t>(k)] != ref.idx[static_cast<std::size_t>(k)]) {
      throw std::runtime_error("router top-k index mismatch at k=" + std::to_string(k));
    }
    const float diff = std::abs(out_prob[static_cast<std::size_t>(k)] - ref.prob[static_cast<std::size_t>(k)]);
    if (diff > 5e-3f) {
      throw std::runtime_error("router top-k probability mismatch at k=" + std::to_string(k));
    }
  }

  require_cuda(cudaFree(d_prob), "cudaFree d_prob");
  require_cuda(cudaFree(d_idx), "cudaFree d_idx");
  require_cuda(cudaFree(d_logits), "cudaFree d_logits");
}

void test_weighted_merge(cudaStream_t stream) {
  const int n = 64;
  const float a = 0.35f;
  const float b = 0.65f;
  std::vector<float> x1(static_cast<std::size_t>(n), 0.0f);
  std::vector<float> x2(static_cast<std::size_t>(n), 0.0f);
  for (int i = 0; i < n; ++i) {
    x1[static_cast<std::size_t>(i)] = std::sin(static_cast<float>(i) * 0.13f);
    x2[static_cast<std::size_t>(i)] = std::cos(static_cast<float>(i) * 0.09f);
  }
  std::vector<__half> h1(static_cast<std::size_t>(n));
  std::vector<__half> h2(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    h1[static_cast<std::size_t>(i)] = __float2half(x1[static_cast<std::size_t>(i)]);
    h2[static_cast<std::size_t>(i)] = __float2half(x2[static_cast<std::size_t>(i)]);
  }

  __half* d1 = nullptr;
  __half* d2 = nullptr;
  __half* dout = nullptr;
  require_cuda(cudaMalloc(&d1, static_cast<std::size_t>(n) * sizeof(__half)), "cudaMalloc d1");
  require_cuda(cudaMalloc(&d2, static_cast<std::size_t>(n) * sizeof(__half)), "cudaMalloc d2");
  require_cuda(cudaMalloc(&dout, static_cast<std::size_t>(n) * sizeof(__half)), "cudaMalloc dout");
  require_cuda(cudaMemcpyAsync(d1,
                               h1.data(),
                               static_cast<std::size_t>(n) * sizeof(__half),
                               cudaMemcpyHostToDevice,
                               stream),
               "cudaMemcpyAsync d1");
  require_cuda(cudaMemcpyAsync(d2,
                               h2.data(),
                               static_cast<std::size_t>(n) * sizeof(__half),
                               cudaMemcpyHostToDevice,
                               stream),
               "cudaMemcpyAsync d2");

  kernels::launch_scale_copy(dout, d1, n, a, stream);
  kernels::launch_scale_add_inplace(dout, d2, n, b, stream);
  require_cuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize merge");

  std::vector<__half> out(static_cast<std::size_t>(n));
  require_cuda(cudaMemcpy(out.data(),
                          dout,
                          static_cast<std::size_t>(n) * sizeof(__half),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy dout");
  for (int i = 0; i < n; ++i) {
    const float got = __half2float(out[static_cast<std::size_t>(i)]);
    const float expect = a * x1[static_cast<std::size_t>(i)] + b * x2[static_cast<std::size_t>(i)];
    const float diff = std::abs(got - expect);
    if (diff > 2e-3f) {
      throw std::runtime_error("weighted merge mismatch at i=" + std::to_string(i));
    }
  }

  require_cuda(cudaFree(dout), "cudaFree dout");
  require_cuda(cudaFree(d2), "cudaFree d2");
  require_cuda(cudaFree(d1), "cudaFree d1");
}

}  // namespace

int main() {
  try {
    int device_count = 0;
    require_cuda(cudaGetDeviceCount(&device_count), "cudaGetDeviceCount");
    if (device_count <= 0) {
      std::cout << "[moe-kernel-test] no CUDA device available; skipping.\n";
      return 0;
    }
    require_cuda(cudaSetDevice(0), "cudaSetDevice");
    cudaStream_t stream = nullptr;
    require_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    test_router_topk_softmax(stream);
    test_weighted_merge(stream);
    require_cuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    std::cout << "[moe-kernel-test] PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "[moe-kernel-test] FAIL: " << e.what() << "\n";
    return 1;
  }
}

