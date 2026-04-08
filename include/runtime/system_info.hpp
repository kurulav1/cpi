#pragma once

// Runtime system-information helpers.
//
// Provides a lightweight snapshot of the host OS name and the available CUDA
// devices with their free/total VRAM. The snapshot is queried once during
// engine initialisation and used to decide how many model layers can be kept
// resident in GPU memory, and to emit a human-readable hardware summary to
// stdout.

#include <cstddef>
#include <string>
#include <vector>

namespace runtime {

// Free and total VRAM for a single CUDA device, in bytes.
struct GpuMemoryStats {
  std::size_t free_bytes = 0;
  std::size_t total_bytes = 0;
};

// A point-in-time snapshot of the host and all visible CUDA devices.
// Fields are populated by collect_system_snapshot().
struct SystemSnapshot {
  std::string os_name;                       // e.g. "Windows 11" or "Linux 5.15"
  int device_count = 0;                      // number of CUDA-visible GPUs
  std::vector<std::string> gpu_names;        // human-readable device names, indexed by device id
  std::vector<GpuMemoryStats> gpu_memory;   // per-device VRAM statistics at time of query
};

// Point-in-time host-resource usage.
// cpu_percent can be negative when a backend cannot provide a reading.
struct HostResourceUsage {
  double cpu_percent = -1.0;
  double memory_percent = -1.0;
};

// Queries the OS name and iterates over all CUDA devices to collect their
// names and current memory usage. The result is used to decide safe
// batch/context sizes and to log the hardware configuration at startup.
SystemSnapshot collect_system_snapshot();

// Formats a byte count as a human-readable string with an appropriate unit
// suffix (B, KB, MB, or GB). Used for log output only.
std::string format_bytes(std::size_t bytes);

// Queries host CPU and physical-memory usage percentages.
// CPU is calculated from cumulative counters and may require two calls before
// stabilizing; callers should sample periodically.
HostResourceUsage query_host_resource_usage();

}  // namespace runtime
