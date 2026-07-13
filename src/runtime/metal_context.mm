// Objective-C++ bridge to Metal. The only file in the engine that knows Metal's
// types exist; everything above it sees opaque void* handles (metal_context.hpp).
//
// Deliberately compiled WITHOUT ARC, so a Metal object can be stored as a plain
// void* and released by hand. Under ARC the same handles would need __bridge
// casts everywhere and the header would have to leak Objective-C.
//
// Unified memory is the thing that makes this simpler than the CUDA path: a
// MTLBuffer allocated with MTLResourceStorageModeShared is addressable by both
// CPU and GPU, so there is no upload, no download, and no staging buffer. What
// the CUDA backend spends cudaMemcpyAsync on, this backend does with memcpy --
// or, often, with nothing at all.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "runtime/metal_context.hpp"

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace runtime {

// ---------------------------------------------------------------------------
// MetalBuffer
// ---------------------------------------------------------------------------

MetalBuffer::~MetalBuffer() {
  if (handle_ != nullptr) {
    [(id<MTLBuffer>)handle_ release];
    handle_ = nullptr;
  }
}

MetalBuffer::MetalBuffer(MetalBuffer&& o) noexcept : handle_(o.handle_), size_(o.size_) {
  o.handle_ = nullptr;
  o.size_ = 0;
}

MetalBuffer& MetalBuffer::operator=(MetalBuffer&& o) noexcept {
  if (this != &o) {
    if (handle_ != nullptr) [(id<MTLBuffer>)handle_ release];
    handle_ = o.handle_;
    size_ = o.size_;
    o.handle_ = nullptr;
    o.size_ = 0;
  }
  return *this;
}

void* MetalBuffer::contents() const {
  if (handle_ == nullptr) return nullptr;
  return [(id<MTLBuffer>)handle_ contents];
}

// ---------------------------------------------------------------------------
// MetalContext
// ---------------------------------------------------------------------------

MetalContext::MetalContext() {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (dev == nil) {
    // No GPU. This is the normal outcome inside a VM -- notably GitHub's macOS
    // runners -- so it must be reported, not asserted away.
    last_error_ = "MTLCreateSystemDefaultDevice() returned nil (no Metal GPU available)";
    return;
  }
  device_ = (void*)[dev retain];

  id<MTLCommandQueue> q = [dev newCommandQueue];
  if (q == nil) {
    last_error_ = "newCommandQueue failed";
    return;
  }
  queue_ = (void*)q;  // already +1

  pipelines_ = (void*)[[NSMutableDictionary alloc] init];
}

MetalContext::~MetalContext() {
  if (cmdbuf_ != nullptr) [(id<MTLCommandBuffer>)cmdbuf_ release];
  if (pipelines_ != nullptr) [(NSMutableDictionary*)pipelines_ release];
  if (library_ != nullptr) [(id<MTLLibrary>)library_ release];
  if (queue_ != nullptr) [(id<MTLCommandQueue>)queue_ release];
  if (device_ != nullptr) [(id<MTLDevice>)device_ release];
}

std::string MetalContext::device_name() const {
  if (device_ == nullptr) return "<none>";
  id<MTLDevice> dev = (id<MTLDevice>)device_;
  return std::string([[dev name] UTF8String]);
}

bool MetalContext::load_library(const std::string& path_hint) {
  if (device_ == nullptr) return false;
  id<MTLDevice> dev = (id<MTLDevice>)device_;

  std::string path = path_hint;
  if (path.empty()) {
    const char* env = std::getenv("CPI_METALLIB");
    if (env != nullptr) path = env;
  }

  NSError* err = nil;
  id<MTLLibrary> lib = nil;

  if (!path.empty()) {
    NSString* p = [NSString stringWithUTF8String:path.c_str()];
    lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:p] error:&err];
  } else {
    lib = [dev newDefaultLibrary];
  }

  if (lib == nil) {
    // No prebuilt library. Fall back to compiling the MSL source at runtime, which
    // needs only the Metal framework -- so a Mac with no Xcode still works.
    const char* srcpath = std::getenv("CPI_METAL_SOURCE");
    if (srcpath != nullptr && load_library_from_source(srcpath)) {
      return true;
    }
    last_error_ = "failed to load metallib";
    if (err != nil) {
      last_error_ += ": ";
      last_error_ += [[err localizedDescription] UTF8String];
    }
    last_error_ += " (and CPI_METAL_SOURCE was unset or failed to compile)";
    return false;
  }
  library_ = (void*)lib;  // +1
  return true;
}

bool MetalContext::load_library_from_source(const std::string& metal_source_path) {
  if (device_ == nullptr) return false;
  id<MTLDevice> dev = (id<MTLDevice>)device_;

  NSError* err = nil;
  NSString* path = [NSString stringWithUTF8String:metal_source_path.c_str()];
  NSString* src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
  if (src == nil) {
    last_error_ = "could not read the shader source: " + metal_source_path;
    return false;
  }

  MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
  id<MTLLibrary> lib = [dev newLibraryWithSource:src options:opts error:&err];
  [opts release];

  if (lib == nil) {
    // A compile error here is a real shader bug -- surface the driver's message
    // verbatim rather than a generic failure.
    last_error_ = "runtime shader compilation failed";
    if (err != nil) {
      last_error_ += ": ";
      last_error_ += [[err localizedDescription] UTF8String];
    }
    return false;
  }
  if (library_ != nullptr) [(id<MTLLibrary>)library_ release];
  library_ = (void*)lib;  // +1
  return true;
}

MetalBuffer MetalContext::alloc(std::size_t bytes) {
  MetalBuffer b;
  if (device_ == nullptr || bytes == 0) return b;
  id<MTLDevice> dev = (id<MTLDevice>)device_;
  id<MTLBuffer> buf = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
  if (buf == nil) {
    last_error_ = "newBufferWithLength failed";
    return b;
  }
  b.handle_ = (void*)buf;  // +1
  b.size_ = bytes;
  return b;
}

MetalBuffer MetalContext::alloc_from(const void* src, std::size_t bytes) {
  MetalBuffer b = alloc(bytes);
  if (b.valid() && src != nullptr) {
    // Unified memory: this is the whole "upload". No staging, no async copy.
    std::memcpy(b.contents(), src, bytes);
  }
  return b;
}

void MetalContext::dispatch(const std::string& name, Grid grid, std::size_t total,
                            std::size_t tg_size, const void* const* buffers,
                            const std::size_t* offsets, int n_buffers, const void* params,
                            std::size_t params_bytes) {
  if (device_ == nullptr || library_ == nullptr || total == 0) return;

  id<MTLDevice> dev = (id<MTLDevice>)device_;
  NSMutableDictionary* cache = (NSMutableDictionary*)pipelines_;
  NSString* key = [NSString stringWithUTF8String:name.c_str()];

  id<MTLComputePipelineState> pso = [cache objectForKey:key];
  if (pso == nil) {
    id<MTLLibrary> lib = (id<MTLLibrary>)library_;
    id<MTLFunction> fn = [lib newFunctionWithName:key];
    if (fn == nil) {
      last_error_ = "no such kernel in metallib: " + name;
      return;
    }
    NSError* err = nil;
    pso = [dev newComputePipelineStateWithFunction:fn error:&err];
    [fn release];
    if (pso == nil) {
      last_error_ = "pipeline creation failed for " + name;
      if (err != nil) {
        last_error_ += ": ";
        last_error_ += [[err localizedDescription] UTF8String];
      }
      return;
    }
    [cache setObject:pso forKey:key];
    [pso release];  // the dictionary owns it now
    pso = [cache objectForKey:key];
  }

  // One command buffer accumulates every dispatch until commit_and_wait(). This
  // is the cheap analogue of a CUDA stream: encoding is the low-overhead part on
  // Metal, so a decode step's ops all land in one buffer and one submission.
  if (cmdbuf_ == nullptr) {
    id<MTLCommandQueue> q = (id<MTLCommandQueue>)queue_;
    cmdbuf_ = (void*)[[q commandBuffer] retain];
  }
  id<MTLCommandBuffer> cb = (id<MTLCommandBuffer>)cmdbuf_;

  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso];

  for (int i = 0; i < n_buffers; ++i) {
    id<MTLBuffer> b = (id<MTLBuffer>)buffers[i];
    const NSUInteger off = (offsets != nullptr) ? (NSUInteger)offsets[i] : 0;
    [enc setBuffer:b offset:off atIndex:(NSUInteger)i];
  }
  if (params != nullptr && params_bytes > 0) {
    [enc setBytes:params length:(NSUInteger)params_bytes atIndex:(NSUInteger)n_buffers];
  }

  const NSUInteger tg = (tg_size > 0) ? (NSUInteger)tg_size : 256;
  if (grid == Grid::Groups) {
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  } else {
    // Non-uniform threadgroups: Metal clamps the final group, so kernels still
    // bounds-check but the grid needs no manual rounding.
    [enc dispatchThreads:MTLSizeMake((NSUInteger)total, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
  }
  [enc endEncoding];
}

void MetalContext::commit_and_wait() {
  if (cmdbuf_ == nullptr) return;
  id<MTLCommandBuffer> cb = (id<MTLCommandBuffer>)cmdbuf_;
  [cb commit];
  [cb waitUntilCompleted];
  if ([cb status] == MTLCommandBufferStatusError) {
    NSError* err = [cb error];
    last_error_ = "command buffer failed";
    if (err != nil) {
      last_error_ += ": ";
      last_error_ += [[err localizedDescription] UTF8String];
    }
  }
  [cb release];
  cmdbuf_ = nullptr;
}

}  // namespace runtime
