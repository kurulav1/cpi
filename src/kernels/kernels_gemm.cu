// Batched fp16 GEMM: the sequence-mode counterpart of the row-major GEMV.
//
//   y[t, m] = sum_k w[m, k] * x[t, k]
//
// Same weight layout as launch_rowmajor_half_gemv_f16 ([out_features, in_features],
// row-major), so a plan can bind one weight and run it either per-token or over a
// whole sequence.
//
// Why this exists: text prefill sidesteps fp16 GEMM entirely by quantising the
// activations and using the int8 dp4a path. The vision tower runs fp16 weights over
// a few hundred patches at once, so it needs a real fp16 GEMM.
//
// NOTE ON NUMERICS: a tiled GEMM sums K in a different order than the GEMV, so the
// two do NOT agree bit-for-bit. The executor therefore keeps dispatching single-token
// work to the GEMV -- sequence mode must not perturb decode output.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "runtime/kernels.cuh"

namespace kernels {
namespace {

constexpr int kTile = 32;

__global__ void rowmajor_half_gemm_f16_kernel(const half* __restrict__ w, const half* __restrict__ x,
                                             half* __restrict__ y, int out_features, int in_features,
                                             int tokens, float in_min, float in_max, float out_min,
                                             float out_max) {
  // +1 column of padding: the compute step reads Ws down a column, which would
  // otherwise hit the same shared-memory bank for every thread in a warp.
  __shared__ half Xs[kTile][kTile + 1];
  __shared__ half Ws[kTile][kTile + 1];

  const int tx = threadIdx.x;  // output feature within the tile
  const int ty = threadIdx.y;  // token within the tile
  const int m = blockIdx.x * kTile + tx;
  const int t = blockIdx.y * kTile + ty;

  float acc = 0.0f;
  for (int k0 = 0; k0 < in_features; k0 += kTile) {
    const int kx = k0 + tx;
    // Each thread stages one element of X and one of W. Out-of-range lanes stage
    // zeros so the multiply-accumulate below needs no bounds check.
    const int xt = blockIdx.y * kTile + ty;
    // Clipped projections clamp the ACTIVATION as it is read. Defaults are +-inf, so
    // this is a no-op for every model that does not use them.
    Xs[ty][tx] = (xt < tokens && kx < in_features)
                     ? __float2half(fminf(fmaxf(__half2float(
                                              x[static_cast<std::size_t>(xt) * in_features + kx]),
                                          in_min),
                                      in_max))
                     : __float2half(0.0f);
    const int wm = blockIdx.x * kTile + ty;
    Ws[ty][tx] = (wm < out_features && kx < in_features)
                     ? w[static_cast<std::size_t>(wm) * in_features + kx]
                     : __float2half(0.0f);
    __syncthreads();

    for (int kk = 0; kk < kTile; ++kk) {
      acc += __half2float(Xs[ty][kk]) * __half2float(Ws[tx][kk]);
    }
    __syncthreads();
  }

  if (t < tokens && m < out_features) {
    y[static_cast<std::size_t>(t) * out_features + m] =
        __float2half(fminf(fmaxf(acc, out_min), out_max));
  }
}

}  // namespace

void launch_rowmajor_half_gemm_f16(const half* w, const half* x, half* y, int out_features,
                                   int in_features, int tokens, cudaStream_t stream, float in_min,
                                   float in_max, float out_min, float out_max) {
  if (out_features <= 0 || in_features <= 0 || tokens <= 0) {
    return;
  }
  const dim3 threads(kTile, kTile);
  const dim3 grid(static_cast<unsigned>((out_features + kTile - 1) / kTile),
                  static_cast<unsigned>((tokens + kTile - 1) / kTile));
  rowmajor_half_gemm_f16_kernel<<<grid, threads, 0, stream>>>(
      w, x, y, out_features, in_features, tokens, in_min, in_max, out_min, out_max);
}

}  // namespace kernels
