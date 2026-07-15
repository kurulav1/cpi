#pragma once

// Metal device context: the Apple Silicon analogue of runtime/cuda_utils.cuh.
//
// This header is deliberately free of both CUDA and Objective-C. Metal's types
// (id<MTLDevice>, id<MTLBuffer>, ...) are Objective-C objects and cannot appear
// in a header that plain C++ translation units include, so everything crosses
// this boundary as an opaque handle and the .mm file does the bridging.
//
// The other reason it matters: unified memory. On Apple Silicon the GPU and CPU
// share physical memory, so a "device buffer" is host-addressable -- contents()
// returns a pointer you can read and write directly, with no copy. Most of the
// H2D/D2H machinery the CUDA path needs simply has no counterpart here.

#include <cstddef>
#include <cstdint>
#include <string>

namespace runtime {

// An opaque device buffer. Owns its MTLBuffer; host-addressable via contents().
class MetalBuffer {
public:
  MetalBuffer() = default;
  ~MetalBuffer();
  MetalBuffer(const MetalBuffer&) = delete;
  MetalBuffer& operator=(const MetalBuffer&) = delete;
  MetalBuffer(MetalBuffer&& o) noexcept;
  MetalBuffer& operator=(MetalBuffer&& o) noexcept;

  // Host pointer into the shared allocation. No copy, no synchronisation beyond
  // having waited on any command buffer that writes it.
  void* contents() const;
  std::size_t size() const {
    return size_;
  }
  bool valid() const {
    return handle_ != nullptr;
  }

  // The underlying id<MTLBuffer>, for the .mm layer only.
  void* handle() const {
    return handle_;
  }

private:
  friend class MetalContext;
  void* handle_ = nullptr;  // id<MTLBuffer>
  std::size_t size_ = 0;
};

class MetalContext {
public:
  MetalContext();
  ~MetalContext();
  MetalContext(const MetalContext&) = delete;
  MetalContext& operator=(const MetalContext&) = delete;

  // True when a real GPU was found. This is FALSE on GitHub's macOS runners --
  // they are VMs with no GPU, so MTLCreateSystemDefaultDevice() returns nil.
  // Callers must check; do not assume a Mac has a usable Metal device.
  bool available() const {
    return device_ != nullptr;
  }
  std::string device_name() const;

  // Loads the compiled shader library. Tries, in order: an explicit path, the
  // CPI_METALLIB env var, the default library next to the executable.
  bool load_library(const std::string& path_hint = "");

  // Compiles the shaders from MSL SOURCE at runtime, via newLibraryWithSource.
  //
  // This is what makes a bare Mac usable. The offline `metal` compiler ships with
  // Xcode -- NOT with the Command Line Tools -- so requiring a .metallib would mean
  // a ~15 GB Xcode download just to run a kernel. But the Metal *framework* carries
  // its own compiler service, which is present on every Mac, so the shader source
  // can simply be handed to the driver.
  //
  // Slower to start (compiles at load), irrelevant for our purposes, and the
  // metallib path stays for builds that have the toolchain.
  bool load_library_from_source(const std::string& metal_source_path);

  MetalBuffer alloc(std::size_t bytes);
  MetalBuffer alloc_from(const void* src, std::size_t bytes);

  // Encodes `name` over `total_threads`, with `tg_size` threads per group, and
  // the given buffers bound at successive indices. `params` (if non-null) is
  // copied into the last binding as a small constant block.
  //
  // grid_mode: Threads = dispatch exactly total_threads (Metal splits into
  // groups itself); Groups = treat total_threads as a THREADGROUP count, for
  // kernels that key off threadgroup_position_in_grid.
  enum class Grid { Threads, Groups };
  void dispatch(const std::string& name, Grid grid, std::size_t total, std::size_t tg_size,
                const void* const* buffers, const std::size_t* offsets, int n_buffers,
                const void* params, std::size_t params_bytes);

  // Submits everything encoded so far and blocks until the GPU is done.
  void commit_and_wait();

  const std::string& last_error() const {
    return last_error_;
  }

  // Instrumentation for the overhead-vs-kernel question. gpu_busy_ms is the summed
  // GPUEndTime-GPUStartTime of every committed command buffer -- time the GPU was actually
  // running kernels, which against wall-clock reveals how much is dispatch/CPU overhead.
  // dispatches counts compute encoders created; cmdbufs counts submissions.
  double gpu_busy_ms() const {
    return gpu_busy_ms_;
  }
  std::uint64_t dispatch_count() const {
    return dispatch_count_;
  }
  std::uint64_t cmdbuf_count() const {
    return cmdbuf_count_;
  }
  void reset_counters() {
    gpu_busy_ms_ = 0.0;
    dispatch_count_ = 0;
    cmdbuf_count_ = 0;
  }

private:
  void* device_ = nullptr;     // id<MTLDevice>
  void* queue_ = nullptr;      // id<MTLCommandQueue>
  void* library_ = nullptr;    // id<MTLLibrary>
  void* cmdbuf_ = nullptr;     // id<MTLCommandBuffer>, lazily opened
  void* pipelines_ = nullptr;  // NSMutableDictionary name -> MTLComputePipelineState
  std::string last_error_;

  double gpu_busy_ms_ = 0.0;
  std::uint64_t dispatch_count_ = 0;
  std::uint64_t cmdbuf_count_ = 0;
};

}  // namespace runtime
