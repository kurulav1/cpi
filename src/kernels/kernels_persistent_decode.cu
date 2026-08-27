// Persistent-decode interpreter: the whole single-token forward in one cooperative launch.
//
// The measured ceiling that forced this design (see memory: cpi-decode-kernel-count): after
// graph capture, split-K attention and two falsified fusion passes, Gemma 4 decode still spent
// ~3 ms/token crossing ~800 kernel boundaries; more than its ~2.5 ms of mandatory weight
// bandwidth. No per-kernel change touches that floor; executing the plan INSIDE one kernel
// does. The op plan is already data, so this is the same plan the eager executor and the CUDA
// graph walk, compiled to a device array of resolved descriptors and interpreted by a grid
// that stays resident for the whole token: block-strided work per op, grid.sync() between ops.
//
// Deliberately not a re-derivation of the model: every op body mirrors the standalone kernel
// it replaces (same accumulation types, same rounding points). Reduction GROUPING differs
// work is grid-strided instead of per-launch; so streams can shift at near-ties, exactly as
// the split-K attention change did; the gates are output quality and the eager path, not
// byte-identity.
//
// Scope: the op kinds a Gemma 4 decode plan uses (embed, scale, copy, add, gemv, rmsnorm,
// table-rope, gelumul, kv-store, split attention, lm-head). A plan containing anything else
// (MoE, delta-net) is rejected at compile time by the engine and falls back to the graph path.

// MSVC's traditional preprocessor trips cccl's guard inside cooperative_groups; the build
// uses it everywhere else without issue, so take the header's own escape hatch rather than
// change compile flags for the whole project.
#define CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING 1
#include <cooperative_groups.h>
#include <cuda_fp16.h>

#include <cstdint>
#include <cstdio>

#include "runtime/kernels.cuh"

namespace cg = cooperative_groups;

namespace kernels {

namespace {

constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;
// Shared memory: the widest consumer is attention stats (q row + an 8-key value tile at
// head_dim 512) at ~9.5 KB; norms and gemvs use a fraction of it.
constexpr int kSmemBytes = 12 * 1024;

__device__ __forceinline__ float warp_sum_f(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return v;
}

// Matches launch_gelu_mul's device body: tanh-approximation GELU.
__device__ __forceinline__ float gelu_tanh(float x) {
  const float c = 0.7978845608028654f;  // sqrt(2/pi)
  const float t = tanhf(c * (x + 0.044715f * x * x * x));
  return 0.5f * x * (1.0f + t);
}

}  // namespace

__global__ void persistent_decode_kernel(const PersistOp* __restrict__ ops, int n_ops,
                                         const int* __restrict__ tok_ptr,
                                         const int* __restrict__ pos_ptr) {
  cg::grid_group grid = cg::this_grid();
  __shared__ unsigned char smem_raw[kSmemBytes];
  const int tid = threadIdx.x;
  const int gtid = blockIdx.x * blockDim.x + tid;
  const int gthreads = gridDim.x * blockDim.x;
  const int warp = tid / 32;
  const int lane = tid % 32;
  const int gwarp = blockIdx.x * kWarps + warp;
  const int gwarps = gridDim.x * kWarps;

  for (int oi = 0; oi < n_ops; ++oi) {
    const PersistOp op = ops[oi];
    switch (op.kind) {
      case PersistOp::kEmbed: {
        const int t = tok_ptr[0];
        const half* row = op.weight + static_cast<std::size_t>(t) * op.cols;
        for (int i = gtid; i < op.cols; i += gthreads) op.out[i] = row[i];
        break;
      }
      case PersistOp::kScale: {
        for (int i = gtid; i < op.cols; i += gthreads) {
          op.out[i] = __float2half(__half2float(op.in[i]) * op.scale);
        }
        break;
      }
      case PersistOp::kCopy: {
        for (int i = gtid; i < op.cols; i += gthreads) op.out[i] = op.in[i];
        break;
      }
      case PersistOp::kAdd: {
        for (int i = gtid; i < op.cols; i += gthreads) {
          op.out[i] = __float2half(__half2float(op.out[i]) + __half2float(op.in[i]));
        }
        break;
      }
      case PersistOp::kGeluMul: {
        // T=1: in2's window is a flat offset (the executor's stride form collapses).
        const half* b = op.in2 + op.aux_offset;
        for (int i = gtid; i < op.cols; i += gthreads) {
          op.out[i] = __float2half(gelu_tanh(__half2float(op.in[i])) * __half2float(b[i]));
        }
        break;
      }
      case PersistOp::kRmsNorm: {
        // One BLOCK per row, rows block-strided; mirrors rmsnorm_kernel_simple's structure
        // (fp32 sum of squares, shared tree reduce, one inv broadcast).
        float* red = reinterpret_cast<float*>(smem_raw);
        for (int r = blockIdx.x; r < op.rows; r += gridDim.x) {
          const half* x = op.in + static_cast<std::size_t>(r) * op.cols;
          half* y = op.out + static_cast<std::size_t>(r) * op.cols;
          float local = 0.0f;
          for (int c = tid; c < op.cols; c += kThreads) {
            const float v = __half2float(x[c]);
            local += v * v;
          }
          red[tid] = local;
          __syncthreads();
          for (int s = kThreads / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
          }
          const float inv = rsqrtf(red[0] / static_cast<float>(op.cols) + op.eps);
          __syncthreads();
          for (int c = tid; c < op.cols; c += kThreads) {
            const float w = (op.weight != nullptr) ? __half2float(op.weight[c]) : 1.0f;
            y[c] = __float2half(__half2float(x[c]) * inv * w);
          }
          __syncthreads();
        }
        break;
      }
      case PersistOp::kRope: {
        // Table rope, in place, whole head (partial rotary lives in the table as identity
        // rotations; same convention as build_rope_tables).
        const int pos = pos_ptr[0];
        const int half_dim = op.head_dim / 2;
        const int pairs = op.heads * half_dim;
        for (int p = gtid; p < pairs; p += gthreads) {
          const int h = p / half_dim;
          const int j = p % half_dim;
          const float c = op.cosT[pos * half_dim + j];
          const float s = op.sinT[pos * half_dim + j];
          const int i0 = h * op.head_dim + j;
          const int i1 = i0 + half_dim;
          const float v0 = __half2float(op.out[i0]);
          const float v1 = __half2float(op.out[i1]);
          op.out[i0] = __float2half(v0 * c - v1 * s);
          op.out[i1] = __float2half(v1 * c + v0 * s);
        }
        break;
      }
      case PersistOp::kGemv: {
        // One WARP per output row, rows grid-warp-strided. 128-bit loads (8 halfs per
        // instruction); 32-bit loads left the warp latency-bound and measurably behind the
        // standalone matvec. fp32 accumulate, one fp16 round at the write.
        const float4* inv4 = reinterpret_cast<const float4*>(op.in);
        const int oct = op.in_dim / 8;
        for (int r = gwarp; r < op.rows; r += gwarps) {
          const float4* w4 =
              reinterpret_cast<const float4*>(op.weight + static_cast<std::size_t>(r) * op.in_dim);
          float acc = 0.0f;
          for (int p = lane; p < oct; p += 32) {
            const float4 aw = w4[p];
            const float4 bw = inv4[p];
            const half2* ah = reinterpret_cast<const half2*>(&aw);
            const half2* bh = reinterpret_cast<const half2*>(&bw);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const float2 a = __half22float2(ah[q]);
              const float2 b = __half22float2(bh[q]);
              acc += a.x * b.x + a.y * b.y;
            }
          }
          acc = warp_sum_f(acc);
          if (lane == 0) op.out[r] = __float2half(acc);
        }
        break;
      }
      case PersistOp::kLmHead: {
        const float4* inv4 = reinterpret_cast<const float4*>(op.in);
        const int oct = op.in_dim / 8;
        for (int r = gwarp; r < op.rows; r += gwarps) {
          const float4* w4 =
              reinterpret_cast<const float4*>(op.weight + static_cast<std::size_t>(r) * op.in_dim);
          float acc = 0.0f;
          for (int p = lane; p < oct; p += 32) {
            const float4 aw = w4[p];
            const float4 bw = inv4[p];
            const half2* ah = reinterpret_cast<const half2*>(&aw);
            const half2* bh = reinterpret_cast<const half2*>(&bw);
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              const float2 a = __half22float2(ah[q]);
              const float2 b = __half22float2(bh[q]);
              acc += a.x * b.x + a.y * b.y;
            }
          }
          acc = warp_sum_f(acc);
          if (lane == 0) op.fout[r] = acc;
        }
        break;
      }
      case PersistOp::kKvStore: {
        const int pos = pos_ptr[0];
        const std::size_t base = static_cast<std::size_t>(pos) * op.cols;
        for (int i = gtid; i < op.cols; i += gthreads) {
          op.kcache[base + i] = op.in[i];
          op.vcache[base + i] = op.in2[i];
        }
        break;
      }
      case PersistOp::kAttnStats: {
        // One BLOCK per (head, chunk) work item, items grid-strided. Mirrors
        // attention_step_chunk_stats_core's math at 8 keys per tile (kWarps warps).
        const int seq_len = pos_ptr[0] + 1;
        const int k_start = (op.window > 0 && seq_len > op.window) ? seq_len - op.window : 0;
        half* q_sh = reinterpret_cast<half*>(smem_raw);
        float* score_sh = reinterpret_cast<float*>(q_sh + op.head_dim);
        float* beta_sh = score_sh + kWarps;
        float* stats_sh = beta_sh + kWarps;  // [running_m, running_l, tile_m, tile_l]
        half* v_tile = reinterpret_cast<half*>(stats_sh + 4);

        const int items = op.heads * op.chunks;
        const int kv_dim = op.kv_heads * op.head_dim;
        const float scale = rsqrtf(static_cast<float>(op.head_dim));
        const int group = (op.heads / op.kv_heads > 0) ? op.heads / op.kv_heads : 1;
        for (int item = blockIdx.x; item < items; item += gridDim.x) {
          const int head = item / op.chunks;
          const int chunk = item % op.chunks;
          const int chunk_lo = max(chunk * op.chunk_size, k_start);
          const int chunk_hi = min(chunk * op.chunk_size + op.chunk_size, seq_len);
          if (chunk_lo >= chunk_hi) continue;
          const int kv_head = head / group;
          const int cidx = head * op.chunks + chunk;

          for (int d = tid; d < op.head_dim; d += kThreads) {
            q_sh[d] = op.in[head * op.head_dim + d];
          }
          if (tid == 0) {
            stats_sh[0] = -1.0e30f;
            stats_sh[1] = 0.0f;
          }
          __syncthreads();

          // Per-thread output accumulators (head_dim <= 512, 256 threads -> 2 per thread).
          float acc0 = 0.0f, acc1 = 0.0f;
          for (int tile = chunk_lo; tile < chunk_hi; tile += kWarps) {
            const int tile_n = min(kWarps, chunk_hi - tile);
            {
              const int t = tile + warp;
              float score = -1.0e30f;
              if (warp < tile_n) {
                const half2* k2 = reinterpret_cast<const half2*>(
                    op.kcache + static_cast<std::size_t>(t) * kv_dim + kv_head * op.head_dim);
                const half2* q2 = reinterpret_cast<const half2*>(q_sh);
                float part = 0.0f;
                for (int p = lane; p < op.head_dim / 2; p += 32) {
                  const float2 a = __half22float2(q2[p]);
                  const float2 b = __half22float2(k2[p]);
                  part += a.x * b.x + a.y * b.y;
                }
                score = warp_sum_f(part) * scale;
              }
              if (lane == 0 && warp < tile_n) score_sh[warp] = score;
            }
            for (int i = 0; i < tile_n; ++i) {
              const half* vt =
                  op.vcache + static_cast<std::size_t>(tile + i) * kv_dim + kv_head * op.head_dim;
              for (int d = tid; d < op.head_dim; d += kThreads) v_tile[i * op.head_dim + d] = vt[d];
            }
            __syncthreads();

            if (tid == 0) {
              float tm = -1.0e30f;
              for (int i = 0; i < tile_n; ++i) tm = fmaxf(tm, score_sh[i]);
              float tl = 0.0f;
              for (int i = 0; i < tile_n; ++i) {
                const float b = expf(score_sh[i] - tm);
                beta_sh[i] = b;
                tl += b;
              }
              stats_sh[2] = tm;
              stats_sh[3] = tl;
            }
            __syncthreads();

            {
              const float tm = stats_sh[2];
              const float tl = stats_sh[3];
              const float rm = stats_sh[0];
              const float rl = stats_sh[1];
              const float nm = fmaxf(rm, tm);
              const float c_prev = (rl == 0.0f) ? 0.0f : expf(rm - nm);
              const float c_tile = expf(tm - nm);
              int j = 0;
              for (int d = tid; d < op.head_dim; d += kThreads, ++j) {
                float to = 0.0f;
                for (int i = 0; i < tile_n; ++i) {
                  to += beta_sh[i] * __half2float(v_tile[i * op.head_dim + d]);
                }
                const float merged = (j == 0 ? acc0 : acc1) * c_prev + to * c_tile;
                if (j == 0) {
                  acc0 = merged;
                } else {
                  acc1 = merged;
                }
              }
              if (tid == 0) {
                stats_sh[0] = nm;
                stats_sh[1] = rl * c_prev + tl * c_tile;
              }
            }
            __syncthreads();
          }

          if (tid == 0) {
            op.sm[cidx] = stats_sh[0];
            op.sl[cidx] = stats_sh[1];
          }
          int j = 0;
          for (int d = tid; d < op.head_dim; d += kThreads, ++j) {
            op.so[static_cast<std::size_t>(cidx) * op.head_dim + d] = (j == 0 ? acc0 : acc1);
          }
          __syncthreads();
        }
        break;
      }
      case PersistOp::kAttnReduce: {
        const int seq_len = pos_ptr[0] + 1;
        const int k_start = (op.window > 0 && seq_len > op.window) ? seq_len - op.window : 0;
        float* scale_sh = reinterpret_cast<float*>(smem_raw);  // [alpha, beta, running_l]
        const int first_chunk = k_start / op.chunk_size;
        const int chunk_count = (seq_len + op.chunk_size - 1) / op.chunk_size;
        for (int head = blockIdx.x; head < op.heads; head += gridDim.x) {
          float acc0 = 0.0f, acc1 = 0.0f;
          float rm = -1.0e30f, rl = 0.0f;
          for (int chunk = first_chunk; chunk < chunk_count; ++chunk) {
            if (tid == 0) {
              const int idx = head * op.chunks + chunk;
              const float cm = op.sm[idx];
              const float cl = op.sl[idx];
              const float nm = fmaxf(rm, cm);
              const float alpha = (rl == 0.0f) ? 0.0f : expf(rm - nm);
              const float beta = (cl == 0.0f) ? 0.0f : expf(cm - nm);
              rl = rl * alpha + cl * beta;
              rm = nm;
              scale_sh[0] = alpha;
              scale_sh[1] = beta;
              scale_sh[2] = rl;
            }
            __syncthreads();
            const float alpha = scale_sh[0];
            const float beta = scale_sh[1];
            const std::size_t base = (static_cast<std::size_t>(head) * op.chunks + chunk) *
                                     static_cast<std::size_t>(op.head_dim);
            int j = 0;
            for (int d = tid; d < op.head_dim; d += kThreads, ++j) {
              const float merged = (j == 0 ? acc0 : acc1) * alpha + op.so[base + d] * beta;
              if (j == 0) {
                acc0 = merged;
              } else {
                acc1 = merged;
              }
            }
            __syncthreads();
          }
          const float inv = 1.0f / fmaxf(scale_sh[2], 1e-8f);
          int j = 0;
          for (int d = tid; d < op.head_dim; d += kThreads, ++j) {
            op.out[head * op.head_dim + d] = __float2half((j == 0 ? acc0 : acc1) * inv);
          }
          __syncthreads();
        }
        break;
      }
      default:
        break;  // compile-time rejection in the engine makes this unreachable
    }
    if (!op.no_sync) grid.sync();
  }
}

int persistent_decode_max_blocks() {
  int dev = 0;
  cudaGetDevice(&dev);
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, persistent_decode_kernel, kThreads, 0);
  if (per_sm <= 0) per_sm = 1;
  return sms * per_sm;
}

bool launch_persistent_decode(const PersistOp* ops, int n_ops, const int* tok, const int* pos,
                              int blocks, cudaStream_t stream) {
  const PersistOp* ops_arg = ops;
  int n_arg = n_ops;
  const int* tok_arg = tok;
  const int* pos_arg = pos;
  void* args[] = {&ops_arg, &n_arg, &tok_arg, &pos_arg};
  const cudaError_t err =
      cudaLaunchCooperativeKernel(reinterpret_cast<void*>(persistent_decode_kernel), dim3(blocks),
                                  dim3(kThreads), args, 0, stream);
  if (err != cudaSuccess) {
    static bool warned = false;
    if (!warned) {
      std::fprintf(stderr, "[persistent] cooperative launch failed: %s\n", cudaGetErrorString(err));
      warned = true;
    }
    return false;
  }
  return true;
}

}  // namespace kernels
