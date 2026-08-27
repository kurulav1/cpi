// Vision-encoder kernels: 2-D RoPE, patch embedding, and spatial average pooling.
//
// These are general ops, not a model. A vision tower differs from a text decoder in
// only a few places; positions are 2-D, the input is pixels rather than token ids,
// and the patch grid is pooled down to a fixed number of soft tokens; and each of
// those is one op here. Everything else (norms, GEMM, GeGLU, attention) is shared.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "runtime/kernels.cuh"

namespace kernels {
namespace {

// 2-D RoPE. The head's channels are split into two halves: the first half rotates by
// the patch's x coordinate, the second by its y coordinate. Within each half the
// rotation is rotate_half, channel j pairs with j + (half/2), the same convention
// the 1-D table kernel already uses, so cos/sin tables are built the same way.
//
// Grid: (heads, tokens). Threads: head_dim/2, one per rotated pair across both halves.
__global__ void rope_2d_inplace_kernel(half* x, const int* pos_x, const int* pos_y, int num_heads,
                                       int head_dim, const float* cos_table, const float* sin_table,
                                       int pairs_per_half) {
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int t = threadIdx.x;
  const int half_pairs = pairs_per_half;  // = head_dim / 4
  if (t >= 2 * half_pairs) {
    return;
  }

  const int axis = t / half_pairs;  // 0 = x, 1 = y
  const int j = t - axis * half_pairs;
  const int spatial_half = head_dim / 2;
  const int base = axis * spatial_half;

  const int pos = axis == 0 ? pos_x[token] : pos_y[token];
  const int tab = pos * half_pairs + j;
  const float c = cos_table[tab];
  const float s = sin_table[tab];

  const std::size_t row = (static_cast<std::size_t>(token) * num_heads + head) * head_dim;
  const int i0 = base + j;
  const int i1 = base + j + half_pairs;
  const float v0 = __half2float(x[row + i0]);
  const float v1 = __half2float(x[row + i1]);
  x[row + i0] = __float2half(v0 * c - v1 * s);
  x[row + i1] = __float2half(v1 * c + v0 * s);
}

// Patch embedding: out[t] = input_proj . (2*(pixels[t] - 0.5)) + pos_embed[x] + pos_embed[y].
//
// The position table is [2, position_embedding_size, hidden]: the x coordinate indexes
// plane 0, the y coordinate plane 1, and the two are SUMMED (HF one-hots the coords
// into the table and sums; a gather is the same thing without materialising the
// one-hot). Padding patches (pos < 0) are zeroed, contributing nothing downstream.
//
// Grid: (hidden/threads, tokens) is awkward for a dot product, so: one block per
// (token, out-channel-tile), each thread reducing over patch_dim.
__global__ void patch_embed_kernel(const half* __restrict__ proj, const float* __restrict__ pixels,
                                   const half* __restrict__ pos_table,
                                   const int* __restrict__ pos_x, const int* __restrict__ pos_y,
                                   half* __restrict__ out, int hidden, int patch_dim,
                                   int pos_table_size) {
  const int token = blockIdx.y;
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= hidden) {
    return;
  }

  const int px = pos_x[token];
  const int py = pos_y[token];
  const std::size_t o = static_cast<std::size_t>(token) * hidden + h;
  if (px < 0 || py < 0) {  // padding patch
    out[o] = __float2half(0.0f);
    return;
  }

  const float* pix = pixels + static_cast<std::size_t>(token) * patch_dim;
  const half* w = proj + static_cast<std::size_t>(h) * patch_dim;
  float acc = 0.0f;
  for (int k = 0; k < patch_dim; ++k) {
    // Gemma applies no mean/std normalisation; it rescales [0,1] -> [-1,1] here.
    acc += __half2float(w[k]) * (2.0f * (pix[k] - 0.5f));
  }

  const std::size_t plane = static_cast<std::size_t>(pos_table_size) * hidden;
  acc += __half2float(pos_table[static_cast<std::size_t>(px) * hidden + h]);
  acc += __half2float(pos_table[plane + static_cast<std::size_t>(py) * hidden + h]);
  out[o] = __float2half(acc);
}

// 2-D average pooling over a k x k grid of patches, then a sqrt(hidden) gain.
//
// Each output soft token averages the k*k patches whose (x/k, y/k) cell it is. Padding
// patches were zeroed upstream, so they contribute nothing to the sum; HF divides by
// k*k regardless, so we do too (dividing by the live count instead would change the
// numbers).
__global__ void avg_pool_patches_kernel(const half* __restrict__ in, const int* __restrict__ pos_x,
                                        const int* __restrict__ pos_y, half* __restrict__ out,
                                        int tokens, int hidden, int k, int cells_x, int out_tokens,
                                        float gain) {
  const int cell = blockIdx.y;
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= hidden || cell >= out_tokens) {
    return;
  }
  float acc = 0.0f;
  for (int t = 0; t < tokens; ++t) {
    const int px = pos_x[t];
    const int py = pos_y[t];
    if (px < 0 || py < 0) {
      continue;
    }
    const int c = (px / k) + cells_x * (py / k);
    if (c == cell) {
      acc += __half2float(in[static_cast<std::size_t>(t) * hidden + h]);
    }
  }
  out[static_cast<std::size_t>(cell) * hidden + h] =
      __float2half(acc / static_cast<float>(k * k) * gain);
}

// out = (in - bias) * scale, elementwise over the hidden dim, broadcast across tokens.
__global__ void standardize_kernel(half* x, const half* bias, const half* scale, int tokens,
                                   int hidden) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const std::size_t n = static_cast<std::size_t>(tokens) * hidden;
  if (i >= n) {
    return;
  }
  const int h = i % hidden;
  x[i] = __float2half((__half2float(x[i]) - __half2float(bias[h])) * __half2float(scale[h]));
}

}  // namespace

void launch_rope_2d_inplace(half* x, const int* pos_x, const int* pos_y, int num_heads,
                            int head_dim, int tokens, const float* cos_table,
                            const float* sin_table, cudaStream_t stream) {
  if (head_dim % 4 != 0) {
    return;  // needs two halves, each an even number of rotated pairs
  }
  const int pairs_per_half = head_dim / 4;
  const dim3 grid(static_cast<unsigned>(num_heads), static_cast<unsigned>(tokens));
  rope_2d_inplace_kernel<<<grid, 2 * pairs_per_half, 0, stream>>>(
      x, pos_x, pos_y, num_heads, head_dim, cos_table, sin_table, pairs_per_half);
}

void launch_patch_embed(const half* proj, const float* pixels, const half* pos_table,
                        const int* pos_x, const int* pos_y, half* out, int tokens, int hidden,
                        int patch_dim, int pos_table_size, cudaStream_t stream) {
  constexpr int kThreads = 128;
  const dim3 grid(static_cast<unsigned>((hidden + kThreads - 1) / kThreads),
                  static_cast<unsigned>(tokens));
  patch_embed_kernel<<<grid, kThreads, 0, stream>>>(proj, pixels, pos_table, pos_x, pos_y, out,
                                                    hidden, patch_dim, pos_table_size);
}

void launch_avg_pool_patches(const half* in, const int* pos_x, const int* pos_y, half* out,
                             int tokens, int hidden, int k, int cells_x, int out_tokens, float gain,
                             cudaStream_t stream) {
  constexpr int kThreads = 128;
  const dim3 grid(static_cast<unsigned>((hidden + kThreads - 1) / kThreads),
                  static_cast<unsigned>(out_tokens));
  avg_pool_patches_kernel<<<grid, kThreads, 0, stream>>>(in, pos_x, pos_y, out, tokens, hidden, k,
                                                         cells_x, out_tokens, gain);
}

void launch_standardize(half* x, const half* bias, const half* scale, int tokens, int hidden,
                        cudaStream_t stream) {
  constexpr int kThreads = 256;
  const std::size_t n = static_cast<std::size_t>(tokens) * hidden;
  const int blocks = static_cast<int>((n + kThreads - 1) / kThreads);
  standardize_kernel<<<blocks, kThreads, 0, stream>>>(x, bias, scale, tokens, hidden);
}

// Sequence-mode RoPE over a whole prompt chunk: token t sits at position
// start_position + t. Same rotate_half table convention as the single-token kernel, so
// prefilling a chunk and decoding it one token at a time agree exactly.
__global__ void rope_seq_table_kernel(half* x, int num_heads, int head_dim, int start_position,
                                      const float* cos_table, const float* sin_table,
                                      int rotary_dim) {
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int pair = threadIdx.x;
  const int rot = rotary_dim > 0 ? rotary_dim : head_dim;
  const int half_rot = rot / 2;
  if (pair >= half_rot) {
    return;
  }
  const int position = start_position + token;
  const int half_dim = head_dim / 2;
  const float c = cos_table[static_cast<std::size_t>(position) * half_dim + pair];
  const float s = sin_table[static_cast<std::size_t>(position) * half_dim + pair];

  const std::size_t row = (static_cast<std::size_t>(token) * num_heads + head) * head_dim;
  const int i0 = pair;
  const int i1 = pair + half_rot;
  const float v0 = __half2float(x[row + i0]);
  const float v1 = __half2float(x[row + i1]);
  x[row + i0] = __float2half(v0 * c - v1 * s);
  x[row + i1] = __float2half(v1 * c + v0 * s);
}

void launch_rope_seq_table(half* x, int num_heads, int head_dim, int start_position, int tokens,
                           const float* cos_table, const float* sin_table, int rotary_dim,
                           cudaStream_t stream) {
  const int rot = rotary_dim > 0 ? rotary_dim : head_dim;
  const dim3 grid(static_cast<unsigned>(num_heads), static_cast<unsigned>(tokens));
  rope_seq_table_kernel<<<grid, rot / 2, 0, stream>>>(x, num_heads, head_dim, start_position,
                                                      cos_table, sin_table, rotary_dim);
}

// Device-position twin: the chunk's base position is read from device memory, so a
// captured graph can replay the same sequence RoPE at any position. Rotation math and
// table layout are identical to rope_seq_table_kernel.
__global__ void rope_seq_table_device_pos_kernel(half* x, int num_heads, int head_dim,
                                                 const int* position_ptr, const float* cos_table,
                                                 const float* sin_table, int rotary_dim) {
  const int head = blockIdx.x;
  const int token = blockIdx.y;
  const int pair = threadIdx.x;
  const int rot = rotary_dim > 0 ? rotary_dim : head_dim;
  const int half_rot = rot / 2;
  if (pair >= half_rot) {
    return;
  }
  const int position = position_ptr[0] + token;
  const int half_dim = head_dim / 2;
  const float c = cos_table[static_cast<std::size_t>(position) * half_dim + pair];
  const float s = sin_table[static_cast<std::size_t>(position) * half_dim + pair];

  const std::size_t row = (static_cast<std::size_t>(token) * num_heads + head) * head_dim;
  const int i0 = pair;
  const int i1 = pair + half_rot;
  const float v0 = __half2float(x[row + i0]);
  const float v1 = __half2float(x[row + i1]);
  x[row + i0] = __float2half(v0 * c - v1 * s);
  x[row + i1] = __float2half(v1 * c + v0 * s);
}

void launch_rope_seq_table_device_pos(half* x, int num_heads, int head_dim, const int* position_ptr,
                                      int tokens, const float* cos_table, const float* sin_table,
                                      int rotary_dim, cudaStream_t stream) {
  const int rot = rotary_dim > 0 ? rotary_dim : head_dim;
  const dim3 grid(static_cast<unsigned>(num_heads), static_cast<unsigned>(tokens));
  rope_seq_table_device_pos_kernel<<<grid, rot / 2, 0, stream>>>(
      x, num_heads, head_dim, position_ptr, cos_table, sin_table, rotary_dim);
}

}  // namespace kernels
