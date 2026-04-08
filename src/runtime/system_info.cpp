#include "runtime/system_info.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <iomanip>
#include <mutex>
#include <sstream>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#else
#include <fstream>
#include <string>
#endif

namespace runtime {

std::string format_bytes(std::size_t bytes) {
  constexpr const char* suffixes[] = {"B", "KB", "MB", "GB", "TB"};
  double value = static_cast<double>(bytes);
  int idx = 0;
  while (value >= 1024.0 && idx < 4) {
    value /= 1024.0;
    ++idx;
  }
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(2) << value << " " << suffixes[idx];
  return oss.str();
}

SystemSnapshot collect_system_snapshot() {
  SystemSnapshot out;
#ifdef _WIN32
  out.os_name = "Windows";
#elif __APPLE__
  out.os_name = "macOS";
#else
  out.os_name = "Linux";
#endif

  const cudaError_t count_status = cudaGetDeviceCount(&out.device_count);
  if (count_status != cudaSuccess || out.device_count <= 0) {
    out.device_count = 0;
    return out;
  }

  int original_device = 0;
  const cudaError_t current_status = cudaGetDevice(&original_device);
  const bool can_restore_device = (current_status == cudaSuccess);

  out.gpu_names.reserve(out.device_count);
  out.gpu_memory.reserve(out.device_count);

  for (int i = 0; i < out.device_count; ++i) {
    if (cudaSetDevice(i) != cudaSuccess) {
      out.gpu_names.emplace_back("unavailable");
      out.gpu_memory.push_back({0, 0});
      continue;
    }
    cudaDeviceProp prop{};
    const cudaError_t prop_status = cudaGetDeviceProperties(&prop, i);
    std::size_t free_b = 0;
    std::size_t total_b = 0;
    const cudaError_t mem_status = cudaMemGetInfo(&free_b, &total_b);
    out.gpu_names.emplace_back(
        (prop_status == cudaSuccess) ? prop.name : "unavailable");
    if (mem_status != cudaSuccess) {
      free_b = 0;
      total_b = 0;
    }
    out.gpu_memory.push_back({free_b, total_b});
  }

  if (can_restore_device) {
    cudaSetDevice(original_device);
  }

  return out;
}

HostResourceUsage query_host_resource_usage() {
  HostResourceUsage out{};

#ifdef _WIN32
  MEMORYSTATUSEX mem{};
  mem.dwLength = sizeof(mem);
  if (GlobalMemoryStatusEx(&mem)) {
    out.memory_percent = static_cast<double>(mem.dwMemoryLoad);
  }

  FILETIME idle_ft{};
  FILETIME kernel_ft{};
  FILETIME user_ft{};
  if (GetSystemTimes(&idle_ft, &kernel_ft, &user_ft)) {
    auto to_u64 = [](const FILETIME& ft) -> std::uint64_t {
      ULARGE_INTEGER u{};
      u.LowPart = ft.dwLowDateTime;
      u.HighPart = ft.dwHighDateTime;
      return static_cast<std::uint64_t>(u.QuadPart);
    };

    const std::uint64_t idle = to_u64(idle_ft);
    const std::uint64_t kernel = to_u64(kernel_ft);
    const std::uint64_t user = to_u64(user_ft);
    const std::uint64_t total = kernel + user;

    static std::mutex cpu_mu;
    static std::uint64_t prev_idle = 0;
    static std::uint64_t prev_total = 0;
    std::lock_guard<std::mutex> lock(cpu_mu);

    if (prev_total > 0 && total > prev_total && idle >= prev_idle) {
      const double total_diff = static_cast<double>(total - prev_total);
      const double idle_diff = static_cast<double>(idle - prev_idle);
      out.cpu_percent = std::max(0.0, std::min(100.0, 100.0 * (1.0 - idle_diff / total_diff)));
    }
    prev_idle = idle;
    prev_total = total;
  }
#else
  {
    std::ifstream meminfo("/proc/meminfo");
    std::string key;
    std::uint64_t value = 0;
    std::string unit;
    std::uint64_t total_kb = 0;
    std::uint64_t avail_kb = 0;
    while (meminfo >> key >> value >> unit) {
      if (key == "MemTotal:") {
        total_kb = value;
      } else if (key == "MemAvailable:") {
        avail_kb = value;
      }
      if (total_kb > 0 && avail_kb > 0) {
        break;
      }
    }
    if (total_kb > 0 && avail_kb <= total_kb) {
      out.memory_percent =
          100.0 * (1.0 - static_cast<double>(avail_kb) / static_cast<double>(total_kb));
    }
  }

  {
    std::ifstream stat("/proc/stat");
    std::string cpu_label;
    std::uint64_t user = 0;
    std::uint64_t nice = 0;
    std::uint64_t system = 0;
    std::uint64_t idle = 0;
    std::uint64_t iowait = 0;
    std::uint64_t irq = 0;
    std::uint64_t softirq = 0;
    std::uint64_t steal = 0;
    if (stat >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal) {
      const std::uint64_t idle_total = idle + iowait;
      const std::uint64_t total = user + nice + system + idle + iowait + irq + softirq + steal;

      static std::mutex cpu_mu;
      static std::uint64_t prev_idle = 0;
      static std::uint64_t prev_total = 0;
      std::lock_guard<std::mutex> lock(cpu_mu);

      if (prev_total > 0 && total > prev_total && idle_total >= prev_idle) {
        const double total_diff = static_cast<double>(total - prev_total);
        const double idle_diff = static_cast<double>(idle_total - prev_idle);
        out.cpu_percent = std::max(0.0, std::min(100.0, 100.0 * (1.0 - idle_diff / total_diff)));
      }
      prev_idle = idle_total;
      prev_total = total;
    }
  }
#endif

  return out;
}

}  // namespace runtime
