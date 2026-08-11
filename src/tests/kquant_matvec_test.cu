// Packed-weight k-quant matvec vs a host reference.
//
// The kernel multiplies without ever materializing the weight as fp16, which is
// the whole point of native k-quant support. The reference here dequantizes on
// the host (that implementation is gated against an fp16 oracle elsewhere) and
// accumulates in fp32, so a disagreement means the in-kernel unpack or the
// accumulation is wrong -- not that the weights differ.
//
// Tolerance rather than bit-exactness: the kernel accumulates in a different
// order (block reduction) than a serial host dot, so the two agree to fp32
// rounding, not exactly. A real unpacking error is orders of magnitude larger
// than that, which is what makes a loose bound sufficient here.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "model/gguf_kquant.hpp"
#include "model/gguf_loader.hpp"
#include "runtime/kernels.cuh"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
  std::printf("  [%s] %s\n", ok ? "ok" : "FAIL", what.c_str());
  if (!ok) ++failures;
}

float h2f(std::uint16_t h) {
  __half v;
  std::memcpy(&v, &h, sizeof(v));
  return __half2float(v);
}

struct Case {
  const char* name;
  kernels::KQuantType type;
  std::size_t block_bytes;
  void (*host_dequant)(const std::uint8_t*, std::size_t, std::uint16_t*);
};

void run_case(const Case& c, int rows, int cols) {
  const int blocks_per_row = cols / static_cast<int>(model::kquant::kSuperBlock);
  const std::size_t row_bytes = static_cast<std::size_t>(blocks_per_row) * c.block_bytes;
  std::vector<std::uint8_t> w(static_cast<std::size_t>(rows) * row_bytes);
  std::mt19937 rng(987);
  std::uniform_int_distribution<int> byte_dist(0, 255);
  for (auto& b : w) b = static_cast<std::uint8_t>(byte_dist(rng));

  // Random bytes are fine for the bit-exact dequant gate, but not for a dot
  // product: a random fp16 scale field is frequently huge, inf or NaN, and the
  // accumulation then overflows and compares nothing. Real blocks have small
  // scales, so the scale fields are overwritten with a plausible one and only
  // the quantized payload stays random.
  const __half small = __float2half(0.01f);
  std::uint16_t small_bits;
  std::memcpy(&small_bits, &small, sizeof(small_bits));
  for (int r = 0; r < rows; ++r) {
    for (int b = 0; b < blocks_per_row; ++b) {
      std::uint8_t* p = w.data() + static_cast<std::size_t>(r) * row_bytes +
                        static_cast<std::size_t>(b) * c.block_bytes;
      if (c.type == kernels::KQuantType::Q6_K) {
        std::memcpy(p + 208, &small_bits, 2);  // d
      } else {
        std::memcpy(p, &small_bits, 2);      // d
        std::memcpy(p + 2, &small_bits, 2);  // dmin
      }
    }
  }

  std::vector<__half> x(cols);
  std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
  for (int i = 0; i < cols; ++i) x[i] = __float2half(xd(rng));

  // Host reference: dequantize each row, dot in fp32.
  std::vector<float> ref(rows, 0.0f);
  std::vector<std::uint16_t> row_fp16(static_cast<std::size_t>(cols));
  for (int r = 0; r < rows; ++r) {
    c.host_dequant(w.data() + static_cast<std::size_t>(r) * row_bytes,
                   static_cast<std::size_t>(blocks_per_row), row_fp16.data());
    double acc = 0.0;
    for (int i = 0; i < cols; ++i) {
      acc += static_cast<double>(h2f(row_fp16[i])) * __half2float(x[i]);
    }
    ref[r] = static_cast<float>(acc);
  }

  std::uint8_t* d_w = nullptr;
  __half* d_x = nullptr;
  __half* d_y = nullptr;
  cudaMalloc(&d_w, w.size());
  cudaMalloc(&d_x, x.size() * sizeof(__half));
  cudaMalloc(&d_y, static_cast<std::size_t>(rows) * sizeof(__half));
  cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
  kernels::launch_kquant_matvec(d_w, c.type, d_x, d_y, rows, cols, nullptr);
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    check(false, std::string(c.name) + ": kernel error " + cudaGetErrorString(err));
    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y);
    return;
  }
  std::vector<__half> y(rows);
  cudaMemcpy(y.data(), d_y, y.size() * sizeof(__half), cudaMemcpyDeviceToHost);
  cudaFree(d_w);
  cudaFree(d_x);
  cudaFree(d_y);

  double worst_rel = 0.0;
  double scale = 0.0;
  for (int r = 0; r < rows; ++r) scale = std::max(scale, std::abs(static_cast<double>(ref[r])));
  for (int r = 0; r < rows; ++r) {
    const double got = __half2float(y[r]);
    // Relative to the largest output, so a near-zero row does not dominate.
    worst_rel = std::max(worst_rel, std::abs(got - ref[r]) / (scale > 0.0 ? scale : 1.0));
  }
  std::printf("  %-6s rows=%-4d cols=%-5d worst_rel=%.5f (scale %.3f)\n", c.name, rows, cols,
              worst_rel, scale);
  // fp16 output rounding alone is ~1e-3 relative; anything structural is far larger.
  check(worst_rel < 0.01, std::string(c.name) + ": matches the host reference");
}

}  // namespace

// Real weights: the same projection multiplied two ways -- once from the packed
// blocks by the kernel, once from the fp16 the loader produces. This is the last
// unknown before residency: synthetic blocks prove the arithmetic, only a real
// tensor proves the layout assumptions (row/col order, block-per-row stride)
// against a file a quantizer actually wrote.
void run_real(const std::string& gguf_path, bool bandwidth) {
  // CPI_MMQ_BENCH=N times the batched tensor-core path at batch N per shape.
  const char* mb = std::getenv("CPI_MMQ_BENCH");
  const int mmq_batch = mb ? atoi(mb) : 0;
  model::GgufLoader g;
  try {
    g.open(gguf_path);
  } catch (const std::exception& e) {
    std::printf("  [FAIL] open threw: %s\n", e.what());
    ++failures;
    return;
  }
  int checked = 0;
  for (const auto& name : g.tensor_names()) {
    const auto pk = g.packed_kquant(name);
    if (!pk.valid() || pk.rows < 2 || pk.cols % 256 != 0) continue;

    const int rows = std::min(pk.rows, 32);  // a slice is enough; rows are independent
    const std::size_t row_bytes = pk.bytes / static_cast<std::size_t>(pk.rows);

    // Q and K are stored with the converter's RoPE row interleave, which the
    // engine undoes on the packed bytes while staging (upload_packed_rows).
    // Doing the same here is what makes this a gate for that: if reordering
    // packed rows were not equivalent to the host un-permute of fp16, this
    // comparison against the loader's fp16 would fail.
    std::vector<std::uint8_t> permuted;
    const auto* packed_src = reinterpret_cast<const std::uint8_t*>(pk.data);
    if (pk.permute_heads > 0 && pk.rows % pk.permute_heads == 0) {
      permuted.resize(pk.bytes);
      const int hd = pk.rows / pk.permute_heads;
      const int hh = hd / 2;
      for (int h = 0; h < pk.permute_heads; ++h) {
        for (int r = 0; r < hd; ++r) {
          const int src_row = (r < hh) ? (r * 2) : ((r - hh) * 2 + 1);
          std::memcpy(permuted.data() + static_cast<std::size_t>(h * hd + r) * row_bytes,
                      packed_src + static_cast<std::size_t>(h * hd + src_row) * row_bytes,
                      row_bytes);
        }
      }
      packed_src = permuted.data();
    }

    std::vector<__half> x(pk.cols);
    std::mt19937 rng(4242);
    std::uniform_real_distribution<float> xd(-1.0f, 1.0f);
    for (int i = 0; i < pk.cols; ++i) x[i] = __float2half(xd(rng));

    // Reference from the loader's fp16 expansion of the same tensor.
    const auto* fp16 = reinterpret_cast<const std::uint16_t*>(g.tensor_data(name));
    std::vector<float> ref(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
      double acc = 0.0;
      for (int i = 0; i < pk.cols; ++i) {
        acc += static_cast<double>(h2f(fp16[static_cast<std::size_t>(r) * pk.cols + i])) *
               __half2float(x[i]);
      }
      ref[r] = static_cast<float>(acc);
    }

    std::uint8_t* d_w = nullptr;
    __half* d_x = nullptr;
    __half* d_y = nullptr;
    cudaMalloc(&d_w, row_bytes * rows);
    cudaMalloc(&d_x, x.size() * sizeof(__half));
    cudaMalloc(&d_y, static_cast<std::size_t>(rows) * sizeof(__half));
    cudaMemcpy(d_w, packed_src, row_bytes * rows, cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(__half), cudaMemcpyHostToDevice);
    kernels::launch_kquant_matvec(d_w, static_cast<kernels::KQuantType>(pk.kind), d_x, d_y, rows,
                                  pk.cols, nullptr);
    cudaDeviceSynchronize();
    std::vector<__half> y(rows);
    cudaMemcpy(y.data(), d_y, y.size() * sizeof(__half), cudaMemcpyDeviceToHost);
    // Batched form against the same fp16 reference, at a batch on each side of
    // the kernel's BMAX tile so both the single-tile and multi-tile paths run.
    // Batch element 0 reuses x, so its expected answer is `ref` -- a batched
    // kernel that silently ignored the batch index would still match there,
    // which is why the other elements get their own activations.
    for (const int bsz : {4, 20}) {
      std::vector<__half> xb(static_cast<std::size_t>(bsz) * pk.cols);
      for (int b = 0; b < bsz; ++b) {
        for (int i = 0; i < pk.cols; ++i) {
          xb[static_cast<std::size_t>(b) * pk.cols + i] =
              (b == 0) ? x[i] : __float2half(__half2float(x[i]) * (0.5f + 0.25f * b));
        }
      }
      __half* d_xb = nullptr;
      __half* d_yb = nullptr;
      cudaMalloc(&d_xb, xb.size() * sizeof(__half));
      cudaMalloc(&d_yb, static_cast<std::size_t>(bsz) * rows * sizeof(__half));
      cudaMemcpy(d_xb, xb.data(), xb.size() * sizeof(__half), cudaMemcpyHostToDevice);
      double refmax = 0.0;
      for (int r = 0; r < rows; ++r) {
        refmax = std::max(refmax, std::abs(static_cast<double>(ref[r])));
      }
      const bool took =
          kernels::launch_kquant_matmul(d_w, static_cast<kernels::KQuantType>(pk.kind), d_xb, d_yb,
                                        rows, pk.cols, bsz, /*ldy=*/rows, nullptr);
      cudaDeviceSynchronize();
      if (took) {
        std::vector<__half> yb(static_cast<std::size_t>(bsz) * rows);
        cudaMemcpy(yb.data(), d_yb, yb.size() * sizeof(__half), cudaMemcpyDeviceToHost);
        double wb = 0.0;
        for (int b = 0; b < bsz; ++b) {
          const float mul = (b == 0) ? 1.0f : (0.5f + 0.25f * b);
          for (int r = 0; r < rows; ++r) {
            const double want = static_cast<double>(ref[r]) * mul;
            const double got = __half2float(yb[static_cast<std::size_t>(b) * rows + r]);
            wb = std::max(wb, std::abs(got - want) / (refmax * mul > 0.0 ? refmax * mul : 1.0));
          }
        }
        check(wb < 0.02, name + ": batched matmul (batch " + std::to_string(bsz) + ") matches");
      } else {
        // Declining is a valid answer: the launcher hands batches past its
        // cutoff back to the caller. Only a wrong result is a failure.
        std::printf("  %-28s fp32 matmul declined batch %d (cutoff)\n", name.c_str(), bsz);
      }
      // dp4a form of the same product. Activations go through int8, so this is
      // checked to a percent rather than to fp16 rounding -- the point is that
      // the integer path computes the same thing, not that it is bit-equal.
      if (pk.cols % 32 == 0) {  // Q6_K enters for the MMQ check below; dp4a is skipped
        std::int8_t* d_q = nullptr;
        float* d_s = nullptr;
        float* d_sum = nullptr;
        const int groups = pk.cols / 32;
        cudaMalloc(&d_q, static_cast<std::size_t>(bsz) * pk.cols);
        cudaMalloc(&d_s, static_cast<std::size_t>(bsz) * groups * sizeof(float));
        cudaMalloc(&d_sum, static_cast<std::size_t>(bsz) * groups * sizeof(float));
        cudaMemset(d_yb, 0, static_cast<std::size_t>(bsz) * rows * sizeof(__half));
        const bool ok = pk.kind != 2 && kernels::launch_kquant_matmul_dp4a(
            d_w, static_cast<kernels::KQuantType>(pk.kind), d_xb, d_yb, rows, pk.cols, bsz,
            /*ldy=*/rows, d_q, d_s, d_sum, nullptr);
        cudaDeviceSynchronize();
        if (ok) {
          std::vector<__half> yq(static_cast<std::size_t>(bsz) * rows);
          cudaMemcpy(yq.data(), d_yb, yq.size() * sizeof(__half), cudaMemcpyDeviceToHost);
          double wq = 0.0;
          for (int b = 0; b < bsz; ++b) {
            const float mul = (b == 0) ? 1.0f : (0.5f + 0.25f * b);
            for (int r = 0; r < rows; ++r) {
              const double want = static_cast<double>(ref[r]) * mul;
              const double got = __half2float(yq[static_cast<std::size_t>(b) * rows + r]);
              wq = std::max(wq, std::abs(got - want) / (refmax * mul > 0.0 ? refmax * mul : 1.0));
            }
          }
          std::printf("  %-28s dp4a batch %-3d worst_rel=%.5f\n", name.c_str(), bsz, wq);
          check(wq < 0.02, name + ": dp4a matmul (batch " + std::to_string(bsz) + ") matches");
        }
        // MMQ: same product on int8 tensor cores. Q4_K only, batch must fit one
        // M tile. A wrong mma fragment mapping shows up here as a large error,
        // not a crash, which is the whole reason this is checked against the
        // fp16 reference rather than eyeballed.
        if ((pk.kind == 0 || pk.kind == 2) && bsz <= 64 && pk.cols % 256 == 0) {
          cudaMemset(d_yb, 0, static_cast<std::size_t>(bsz) * rows * sizeof(__half));
          const bool okm = kernels::launch_kquant_mmq(
              d_w, static_cast<kernels::KQuantType>(pk.kind), d_xb, d_yb, rows, pk.cols, bsz,
              /*ldy=*/rows, d_q, d_s, d_sum, nullptr);
          cudaDeviceSynchronize();
          if (okm) {
            std::vector<__half> ym(static_cast<std::size_t>(bsz) * rows);
            cudaMemcpy(ym.data(), d_yb, ym.size() * sizeof(__half), cudaMemcpyDeviceToHost);
            double wm = 0.0;
            for (int b = 0; b < bsz; ++b) {
              const float mul = (b == 0) ? 1.0f : (0.5f + 0.25f * b);
              for (int r = 0; r < rows; ++r) {
                const double want = static_cast<double>(ref[r]) * mul;
                const double got = __half2float(ym[static_cast<std::size_t>(b) * rows + r]);
                wm = std::max(wm, std::abs(got - want) / (refmax * mul > 0.0 ? refmax * mul : 1.0));
              }
            }
            std::printf("  %-28s mmq  batch %-3d worst_rel=%.5f\n", name.c_str(), bsz, wm);
            check(wm < 0.02, name + ": mmq matmul (batch " + std::to_string(bsz) + ") matches");
          }
        }
        cudaFree(d_q);
        cudaFree(d_s);
        cudaFree(d_sum);
      }
      cudaFree(d_xb);
      cudaFree(d_yb);
    }
    // Achieved bandwidth at the real shape. A matvec reads its weight exactly
    // once, so bytes/time is the number to compare against the card's peak --
    // and it says which shape is worth tuning rather than which is biggest.
    // Uses the full tensor, not the 32-row correctness slice.
    double gbps = 0.0;
    if (bandwidth) {
      std::uint8_t* d_full = nullptr;
      __half* d_yfull = nullptr;
      if (cudaMalloc(&d_full, pk.bytes) == cudaSuccess &&
          cudaMalloc(&d_yfull, static_cast<std::size_t>(pk.rows) * sizeof(__half)) == cudaSuccess) {
        cudaMemcpy(d_full, packed_src, pk.bytes, cudaMemcpyHostToDevice);
        for (int it = 0; it < 3; ++it) {
          kernels::launch_kquant_matvec(d_full, static_cast<kernels::KQuantType>(pk.kind), d_x,
                                        d_yfull, pk.rows, pk.cols, nullptr);
        }
        cudaDeviceSynchronize();
        cudaEvent_t t0, t1;
        cudaEventCreate(&t0);
        cudaEventCreate(&t1);
        constexpr int kIters = 50;
        cudaEventRecord(t0);
        for (int it = 0; it < kIters; ++it) {
          kernels::launch_kquant_matvec(d_full, static_cast<kernels::KQuantType>(pk.kind), d_x,
                                        d_yfull, pk.rows, pk.cols, nullptr);
        }
        cudaEventRecord(t1);
        cudaEventSynchronize(t1);
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, t0, t1);
        gbps = static_cast<double>(pk.bytes) * kIters / (ms * 1e-3) / 1e9;
        cudaEventDestroy(t0);
        cudaEventDestroy(t1);
      }

      // WARNING: the GB/s reported here is an L2 number for anything that fits
      // in cache, which on this part is everything in this model. The 5090 has
      // roughly 96-128 MB of L2; w2 is 48 MB and w13 is 33, so thirty
      // back-to-back iterations over one weight are served from L2 and never
      // touch DRAM. That is why times look flat across weight sizes and why w2
      // read "1416 GB/s" here while the same kernel averages 102.7 us in the
      // engine, where each weight is read once per step from memory.
      //
      // Use this to compare KERNEL VARIANTS on one shape -- that comparison is
      // still fair, both variants get the same cache -- and never to conclude
      // anything about achieved bandwidth or to compare across shapes.
      //
      // Same for the batched tensor-core path. The matvec bench above is
      // launch-overhead bound and cannot be used to tune anything -- every shape
      // came out at 18-26 us regardless of size. MMQ is not: at a real shape it
      // runs 80-140 us a call, so launch cost is a few percent and bytes/time is
      // a signal you can iterate a kernel against. That matters because the
      // end-to-end batched delta for one of these kernels is ~1% of a step,
      // which is too coarse to tune with.
      if (mmq_batch > 0 && (pk.kind == 0 || pk.kind == 2) && pk.cols % 256 == 0) {
        const int B = mmq_batch;
        std::int8_t* mq = nullptr;
        float* ms = nullptr;
        float* msum = nullptr;
        __half* mx = nullptr;
        __half* my = nullptr;
        const int mg = pk.cols / 32;
        if (cudaMalloc(&mq, static_cast<std::size_t>(B) * pk.cols) == cudaSuccess &&
            cudaMalloc(&ms, static_cast<std::size_t>(B) * mg * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&msum, static_cast<std::size_t>(B) * mg * sizeof(float)) == cudaSuccess &&
            cudaMalloc(&mx, static_cast<std::size_t>(B) * pk.cols * sizeof(__half)) ==
                cudaSuccess &&
            cudaMalloc(&my, static_cast<std::size_t>(B) * rows * sizeof(__half)) == cudaSuccess) {
          std::vector<__half> hx(static_cast<std::size_t>(B) * pk.cols);
          for (std::size_t i = 0; i < hx.size(); ++i) {
            hx[i] = __float2half(0.02f * static_cast<float>((i % 61) - 30));
          }
          cudaMemcpy(mx, hx.data(), hx.size() * sizeof(__half), cudaMemcpyHostToDevice);
          bool ok_mmq = true;
          for (int it = 0; it < 3; ++it) {
            ok_mmq = kernels::launch_kquant_mmq(d_full, static_cast<kernels::KQuantType>(pk.kind),
                                                mx, my, rows, pk.cols, B, rows, mq, ms, msum,
                                                nullptr);
          }
          cudaDeviceSynchronize();
          if (ok_mmq) {
            cudaEvent_t m0, m1;
            cudaEventCreate(&m0);
            cudaEventCreate(&m1);
            constexpr int kMmqIters = 30;
            cudaEventRecord(m0);
            for (int it = 0; it < kMmqIters; ++it) {
              kernels::launch_kquant_mmq(d_full, static_cast<kernels::KQuantType>(pk.kind), mx, my,
                                         rows, pk.cols, B, rows, mq, ms, msum, nullptr);
            }
            cudaEventRecord(m1);
            cudaEventSynchronize(m1);
            float mms = 0.0f;
            cudaEventElapsedTime(&mms, m0, m1);
            const double us = mms * 1e3 / kMmqIters;
            std::printf("  %-28s MMQ B=%-3d %8.1f us  %7.1f GB/s\n", name.c_str(), B, us,
                        static_cast<double>(pk.bytes) / (us * 1e-6) / 1e9);
            cudaEventDestroy(m0);
            cudaEventDestroy(m1);
          } else {
            std::printf("  %-28s MMQ B=%-3d declined\n", name.c_str(), B);
          }
        }
        cudaFree(mq);
        cudaFree(ms);
        cudaFree(msum);
        cudaFree(mx);
        cudaFree(my);
      }
      cudaFree(d_full);
      cudaFree(d_yfull);
    }

    cudaFree(d_w);
    cudaFree(d_x);
    cudaFree(d_y);

    double scale = 0.0;
    for (int r = 0; r < rows; ++r) scale = std::max(scale, std::abs(static_cast<double>(ref[r])));
    double worst = 0.0;
    for (int r = 0; r < rows; ++r) {
      worst = std::max(worst, std::abs(__half2float(y[r]) - ref[r]) / (scale > 0.0 ? scale : 1.0));
    }

    std::printf("  %-28s kind=%d rows=%-5d cols=%-5d worst_rel=%.5f", name.c_str(), pk.kind,
                pk.rows, pk.cols, worst);
    if (bandwidth) std::printf("  %7.1f GB/s", gbps);
    std::printf("\n");
    check(worst < 0.01, name + ": packed matvec matches the fp16 expansion");
    if (++checked >= 6) break;
  }
  check(checked > 0, "found packed k-quant tensors to check");
}

int main(int argc, char** argv) {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0) {
    std::printf("kquant_matvec_test: no CUDA device; skipping\n");
    return 0;
  }
  const Case cases[] = {
      {"Q4_K", kernels::KQuantType::Q4_K, model::kquant::kQ4KBlockBytes,
       &model::kquant::dequant_q4_k},
      {"Q5_K", kernels::KQuantType::Q5_K, model::kquant::kQ5KBlockBytes,
       &model::kquant::dequant_q5_k},
      {"Q6_K", kernels::KQuantType::Q6_K, model::kquant::kQ6KResidentBytes,
       &model::kquant::dequant_q6_k},
  };
  std::printf("kquant_matvec_test: packed matvec vs host dequant + fp32 dot\n");
  for (const Case& c : cases) {
    run_case(c, 8, 512);      // several rows, two super-blocks each
    run_case(c, 64, 4096);    // a realistic projection shape
  }
  if (argc > 1) {
    std::printf("kquant_matvec_test: real weights from %s\n", argv[1]);
    // A second argument turns on the per-shape bandwidth report.
    run_real(argv[1], argc > 2);
  }
  std::printf("%s\n", failures == 0 ? "KQUANT MATVEC: PASS" : "KQUANT MATVEC: FAIL");
  return failures == 0 ? 0 : 1;
}
