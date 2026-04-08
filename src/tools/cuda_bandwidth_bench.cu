#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "runtime/cuda_utils.cuh"

namespace {

struct Options {
  int device = 0;
  std::size_t bytes = 1024ull * 1024ull * 1024ull;
  int warmup = 3;
  int iters = 100;
  int repeats = 5;
  double bytes_per_token_gb = 0.0;
  bool have_bytes_per_token = false;
};

struct BenchResult {
  double best_ms = 0.0;
  double avg_ms = 0.0;
  double best_gbps = 0.0;
  double avg_gbps = 0.0;
};

double bytes_to_gb(std::size_t bytes) { return static_cast<double>(bytes) / 1.0e9; }

double bytes_to_gib(std::size_t bytes) { return static_cast<double>(bytes) / static_cast<double>(1ull << 30); }

std::string need_value(int& i, int argc, char** argv, const std::string& flag) {
  if (i + 1 >= argc) {
    throw std::runtime_error("missing value for " + flag);
  }
  return argv[++i];
}

Options parse_args(int argc, char** argv) {
  Options opts;
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--device") {
      opts.device = std::stoi(need_value(i, argc, argv, arg));
    } else if (arg == "--mb") {
      opts.bytes = static_cast<std::size_t>(std::stoull(need_value(i, argc, argv, arg))) * 1024ull * 1024ull;
    } else if (arg == "--iters") {
      opts.iters = std::stoi(need_value(i, argc, argv, arg));
    } else if (arg == "--warmup") {
      opts.warmup = std::stoi(need_value(i, argc, argv, arg));
    } else if (arg == "--repeats") {
      opts.repeats = std::stoi(need_value(i, argc, argv, arg));
    } else if (arg == "--bytes-per-token-gb") {
      opts.bytes_per_token_gb = std::stod(need_value(i, argc, argv, arg));
      opts.have_bytes_per_token = true;
    } else if (arg == "--help" || arg == "-h") {
      std::cout
          << "Usage: cuda_bandwidth_bench [--device N] [--mb N] [--iters N] [--warmup N] [--repeats N] "
             "[--bytes-per-token-gb X]\n";
      std::exit(0);
    } else {
      throw std::runtime_error("unknown argument: " + arg);
    }
  }

  if (opts.bytes < 64ull * 1024ull * 1024ull) {
    throw std::runtime_error("--mb must be at least 64");
  }
  if (opts.iters < 1 || opts.warmup < 0 || opts.repeats < 1) {
    throw std::runtime_error("--iters >= 1, --warmup >= 0, --repeats >= 1 required");
  }
  if (opts.have_bytes_per_token && opts.bytes_per_token_gb <= 0.0) {
    throw std::runtime_error("--bytes-per-token-gb must be > 0");
  }

  return opts;
}

__device__ __forceinline__ unsigned int warp_sum_u32(unsigned int v) {
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    v += __shfl_down_sync(0xffffffffu, v, offset);
  }
  return v;
}

__global__ void read_int4_kernel(const int4* src, unsigned int* sink, std::size_t n) {
  __shared__ unsigned int warp_sums[32];
  const int tid = threadIdx.x;
  const int lane = tid & (warpSize - 1);
  const int warp = tid / warpSize;
  const int warp_count = (blockDim.x + warpSize - 1) / warpSize;
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
  std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(tid);

  unsigned int local = 0;
  while (idx < n) {
    const int4 v = src[idx];
    local += static_cast<unsigned int>(v.x) + static_cast<unsigned int>(v.y) +
             static_cast<unsigned int>(v.z) + static_cast<unsigned int>(v.w);
    idx += stride;
  }

  local = warp_sum_u32(local);
  if (lane == 0) {
    warp_sums[warp] = local;
  }
  __syncthreads();

  if (warp == 0) {
    unsigned int block_sum = (lane < warp_count) ? warp_sums[lane] : 0u;
    block_sum = warp_sum_u32(block_sum);
    if (lane == 0) {
      sink[blockIdx.x] = block_sum;
    }
  }
}

__global__ void write_int4_kernel(int4* dst, int4 value, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
  std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
  while (idx < n) {
    dst[idx] = value;
    idx += stride;
  }
}

__global__ void copy_int4_kernel(const int4* src, int4* dst, std::size_t n) {
  const std::size_t stride = static_cast<std::size_t>(gridDim.x) * static_cast<std::size_t>(blockDim.x);
  std::size_t idx = static_cast<std::size_t>(blockIdx.x) * static_cast<std::size_t>(blockDim.x) + static_cast<std::size_t>(threadIdx.x);
  while (idx < n) {
    dst[idx] = src[idx];
    idx += stride;
  }
}

template <typename Launch>
BenchResult run_bench(Launch&& launch,
                      double bytes_per_iter,
                      int warmup,
                      int iters,
                      int repeats,
                      cudaStream_t stream) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) {
    launch(stream);
  }
  CUDA_CHECK(cudaStreamSynchronize(stream));

  std::vector<double> times_ms;
  times_ms.reserve(static_cast<std::size_t>(repeats));
  for (int rep = 0; rep < repeats; ++rep) {
    CUDA_CHECK(cudaEventRecord(start, stream));
    for (int i = 0; i < iters; ++i) {
      launch(stream);
    }
    CUDA_CHECK(cudaEventRecord(stop, stream));
    CUDA_CHECK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    times_ms.push_back(static_cast<double>(elapsed_ms));
  }

  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaEventDestroy(start));

  const double best_ms = *std::min_element(times_ms.begin(), times_ms.end());
  double sum_ms = 0.0;
  for (double ms : times_ms) {
    sum_ms += ms;
  }
  const double avg_ms = sum_ms / static_cast<double>(times_ms.size());
  const double total_bytes = bytes_per_iter * static_cast<double>(iters);

  BenchResult result;
  result.best_ms = best_ms;
  result.avg_ms = avg_ms;
  result.best_gbps = total_bytes / (best_ms / 1000.0) / 1.0e9;
  result.avg_gbps = total_bytes / (avg_ms / 1000.0) / 1.0e9;
  return result;
}

void print_result(const std::string& label,
                  const BenchResult& payload,
                  double traffic_multiplier = 1.0) {
  std::cout << std::fixed << std::setprecision(2);
  std::cout << "[bandwidth] " << label << " payload_best_gbps=" << payload.best_gbps
            << " payload_avg_gbps=" << payload.avg_gbps;
  if (traffic_multiplier != 1.0) {
    std::cout << " dram_best_gbps=" << (payload.best_gbps * traffic_multiplier)
              << " dram_avg_gbps=" << (payload.avg_gbps * traffic_multiplier);
  }
  std::cout << " best_ms=" << payload.best_ms << " avg_ms=" << payload.avg_ms << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options opts = parse_args(argc, argv);

    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    if (opts.device < 0 || opts.device >= device_count) {
      throw std::runtime_error("invalid CUDA device index");
    }

    CUDA_CHECK(cudaSetDevice(opts.device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, opts.device));

    std::size_t free_b = 0;
    std::size_t total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));

    std::size_t bytes = opts.bytes;
    const std::size_t reserve_bytes = 256ull * 1024ull * 1024ull;
    const std::size_t max_safe_bytes = (free_b > reserve_bytes) ? ((free_b - reserve_bytes) / 2ull) : 0ull;
    if (bytes > max_safe_bytes && max_safe_bytes >= 64ull * 1024ull * 1024ull) {
      bytes = max_safe_bytes;
    }
    bytes = (bytes / sizeof(int4)) * sizeof(int4);
    if (bytes == 0) {
      throw std::runtime_error("not enough free VRAM for benchmark allocations");
    }

    const std::size_t n_vec = bytes / sizeof(int4);
    constexpr int threads = 256;
    const int blocks = std::min<std::size_t>((n_vec + threads - 1) / threads,
                                             static_cast<std::size_t>(std::max(1, prop.multiProcessorCount * 32)));
    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    int4* d_src = nullptr;
    int4* d_dst = nullptr;
    unsigned int* d_sink = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, bytes));
    CUDA_CHECK(cudaMalloc(&d_dst, bytes));
    CUDA_CHECK(cudaMalloc(&d_sink, static_cast<std::size_t>(blocks) * sizeof(unsigned int)));
    CUDA_CHECK(cudaMemsetAsync(d_src, 0x5a, bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(d_dst, 0, bytes, stream));
    CUDA_CHECK(cudaMemsetAsync(d_sink, 0, static_cast<std::size_t>(blocks) * sizeof(unsigned int), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    int mem_clock_khz = 0;
    int bus_width_bits = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, opts.device));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, opts.device));
    const double theoretical_gbps =
        2.0 * static_cast<double>(mem_clock_khz) * (static_cast<double>(bus_width_bits) / 8.0) / 1.0e6;

    std::cout << "[device] index=" << opts.device << " name=" << prop.name
              << " sm=" << prop.multiProcessorCount
              << " cc=" << prop.major << "." << prop.minor
              << " free_gb=" << std::fixed << std::setprecision(2) << bytes_to_gb(free_b)
              << " total_gb=" << bytes_to_gb(total_b)
              << " benchmark_buffer_gib=" << bytes_to_gib(bytes)
              << " theoretical_mem_gbps=" << theoretical_gbps << "\n";

    const int4 pattern{0x01020304, 0x05060708, 0x09101112, 0x13141516};
    const auto read_result = run_bench(
        [&](cudaStream_t s) { read_int4_kernel<<<blocks, threads, 0, s>>>(d_src, d_sink, n_vec); },
        static_cast<double>(bytes),
        opts.warmup,
        opts.iters,
        opts.repeats,
        stream);
    CUDA_CHECK(cudaGetLastError());

    const auto write_result = run_bench(
        [&](cudaStream_t s) { write_int4_kernel<<<blocks, threads, 0, s>>>(d_dst, pattern, n_vec); },
        static_cast<double>(bytes),
        opts.warmup,
        opts.iters,
        opts.repeats,
        stream);
    CUDA_CHECK(cudaGetLastError());

    const auto copy_kernel_result = run_bench(
        [&](cudaStream_t s) { copy_int4_kernel<<<blocks, threads, 0, s>>>(d_src, d_dst, n_vec); },
        static_cast<double>(bytes),
        opts.warmup,
        opts.iters,
        opts.repeats,
        stream);
    CUDA_CHECK(cudaGetLastError());

    const auto memcpy_result = run_bench(
        [&](cudaStream_t s) { CUDA_CHECK(cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, s)); },
        static_cast<double>(bytes),
        opts.warmup,
        opts.iters,
        opts.repeats,
        stream);

    print_result("read_int4", read_result);
    print_result("write_int4", write_result);
    print_result("copy_int4", copy_kernel_result, 2.0);
    print_result("cudaMemcpy_d2d", memcpy_result, 2.0);

    if (opts.have_bytes_per_token) {
      const double roofline_tok_s = read_result.best_gbps / opts.bytes_per_token_gb;
      const double roofline_tok_s_avg = read_result.avg_gbps / opts.bytes_per_token_gb;
      std::cout << "[roofline] bytes_per_token_gb=" << opts.bytes_per_token_gb
                << " read_best_tok_per_s=" << roofline_tok_s
                << " read_avg_tok_per_s=" << roofline_tok_s_avg << "\n";
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_sink));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_src));
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Fatal: " << ex.what() << "\n";
    return 1;
  }
}
