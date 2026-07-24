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

#include <mach-o/dyld.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>
#include <unordered_map>
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

  // The pipeline cache lives behind the existing void* so the HEADER LAYOUT IS UNCHANGED.
  // MetalContext is embedded BY VALUE in PlanMetalEngine, so growing it breaks any translation
  // unit still compiled against the old layout -- which a first attempt at this did: the engine
  // read its config through a shifted `this` and reported layers=7, vocab=397.
  pipelines_ = (void*)new std::unordered_map<std::string, void*>();
}

MetalContext::~MetalContext() {
  if (encoder_ != nullptr) [(id<MTLComputeCommandEncoder>)encoder_ release];
  if (cmdbuf_ != nullptr) [(id<MTLCommandBuffer>)cmdbuf_ release];
  if (sample_buf_ != nullptr) [(id<MTLCounterSampleBuffer>)sample_buf_ release];
  // The map holds a +1 on each pipeline state.
  if (pipelines_ != nullptr) {
    auto* pmap = (std::unordered_map<std::string, void*>*)pipelines_;
    for (auto& kv : *pmap) {
      if (kv.second != nullptr) [(id<MTLComputePipelineState>)kv.second release];
    }
    delete pmap;
    pipelines_ = nullptr;
  }
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
    // newDefaultLibrary only ever finds a metallib EMBEDDED IN A BUNDLE. These are plain
    // CLI executables, so it returns nil even when cpi_kernels.metallib is sitting right
    // next to the binary -- which made an installed copy unrunnable unless the user also
    // had the MSL sources and set CPI_METAL_SOURCE. Look beside the executable first.
    char exe[4096];
    std::uint32_t exe_len = sizeof(exe);
    if (_NSGetExecutablePath(exe, &exe_len) == 0) {
      std::string dir(exe);
      const std::size_t slash = dir.find_last_of('/');
      dir = (slash == std::string::npos) ? std::string(".") : dir.substr(0, slash);
      const std::string candidates[] = {
          dir + "/cpi_kernels.metallib",
          dir + "/../lib/cpi_kernels.metallib",       // install prefix: bin/ + lib/
          dir + "/../Resources/cpi_kernels.metallib",  // .app bundle layout
      };
      for (const std::string& c : candidates) {
        NSString* p = [NSString stringWithUTF8String:c.c_str()];
        if (![[NSFileManager defaultManager] fileExistsAtPath:p]) continue;
        NSError* cerr = nil;
        lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:p] error:&cerr];
        if (lib != nil) break;
        if (err == nil) err = cerr;  // keep the first real load error for the message
      }
    }
    if (lib == nil) lib = [dev newDefaultLibrary];
  }

  if (lib == nil) {
    // No prebuilt library. Fall back to compiling the MSL source at runtime, which
    // needs only the Metal framework -- so a Mac with no Xcode still works.
    const char* srcpath = std::getenv("CPI_METAL_SOURCE");
    if (srcpath != nullptr) {
      if (load_library_from_source(srcpath)) return true;
      // KEEP the compiler's message. Overwriting it with a generic "failed to load"
      // hides the one thing that says what is actually wrong with the shader.
      return false;
    }
    last_error_ = "failed to load metallib";
    if (err != nil) {
      last_error_ += ": ";
      last_error_ += [[err localizedDescription] UTF8String];
    }
    last_error_ += " (and CPI_METAL_SOURCE was unset)";
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
  NSFileManager* fm = [NSFileManager defaultManager];

  // The shaders live as several family files (00_common, 10_dense, ...). The runtime compiler
  // takes ONE source string and does not resolve #include from a source string, so a directory
  // is concatenated here in filename order -- the numeric prefixes make that order the dependency
  // order (common first). The offline CMake path cats the same files in the same order. A plain
  // file path still works, for a prebuilt single-file blob or a caller that points straight at one.
  NSString* src = nil;
  BOOL is_dir = NO;
  if ([fm fileExistsAtPath:path isDirectory:&is_dir] && is_dir) {
    NSArray<NSString*>* entries = [[fm contentsOfDirectoryAtPath:path error:&err]
        sortedArrayUsingSelector:@selector(compare:)];
    if (entries == nil) {
      last_error_ = "could not list the shader directory: " + metal_source_path;
      return false;
    }
    NSMutableString* joined = [NSMutableString string];
    for (NSString* name in entries) {
      if (![[name pathExtension] isEqualToString:@"metal"]) continue;
      NSString* fp = [path stringByAppendingPathComponent:name];
      NSString* part = [NSString stringWithContentsOfFile:fp encoding:NSUTF8StringEncoding
                                                    error:&err];
      if (part == nil) {
        last_error_ = "could not read shader part: " + std::string([fp UTF8String]);
        return false;
      }
      [joined appendString:part];
      [joined appendString:@"\n"];
    }
    src = joined;
  } else {
    src = [NSString stringWithContentsOfFile:path encoding:NSUTF8StringEncoding error:&err];
  }
  if (src == nil) {
    last_error_ = "could not read the shader source: " + metal_source_path;
    return false;
  }

  MTLCompileOptions* opts = [[MTLCompileOptions alloc] init];
  // Metal enables FAST MATH by default, which lowers the precision of exactly the
  // functions this engine leans on: sin/cos in RoPE, exp in the attention softmax,
  // rsqrt in RMSNorm. Left on, logits drift ~0.5 against the fp32 CPU reference --
  // an order of magnitude worse than the CUDA backend's 0.06 -- which is enough for
  // a greedy stream to flip a token after a few steps. Correctness first.
  // MTLCompileOptions.mathMode / MTLMathModeSafe are macOS 15 SDK symbols. Guard on the SDK
  // version at COMPILE time (a runtime @available alone still needs the symbol to exist in the
  // headers, so it fails to build against Xcode 15). Older SDKs use the deprecated-but-equivalent
  // fastMathEnabled = NO -- both simply disable fast math.
#if defined(__MAC_OS_X_VERSION_MAX_ALLOWED) && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000
  if (@available(macOS 15.0, iOS 18.0, *)) {
    opts.mathMode = MTLMathModeSafe;
  } else {
    opts.fastMathEnabled = NO;
  }
#else
  opts.fastMathEnabled = NO;
#endif
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

void MetalContext::enable_gpu_profile(bool on) {
  gpu_profile_ = on;
  if (!on || device_ == nullptr || sample_buf_ != nullptr) return;

  id<MTLDevice> dev = (id<MTLDevice>)device_;
  id<MTLCounterSet> ts = nil;
  for (id<MTLCounterSet> cs in [dev counterSets]) {
    if ([[cs name] isEqualToString:@"timestamp"]) ts = cs;
  }
  if (ts == nil) {
    last_error_ = "this device exposes no timestamp counter set";
    gpu_profile_ = false;
    return;
  }

  // Two samples per dispatch. The driver caps a sample buffer at 32768 B = 4096 timestamps, so
  // 2048 dispatches is the ceiling -- a 24-layer prefill issues ~750, comfortably inside it.
  // Recording stops past the cap rather than reallocating mid-pass.
  sample_slots_ = 4096;
  MTLCounterSampleBufferDescriptor* d = [[MTLCounterSampleBufferDescriptor alloc] init];
  d.counterSet = ts;
  d.storageMode = MTLStorageModeShared;
  d.sampleCount = (NSUInteger)sample_slots_;
  NSError* err = nil;
  id<MTLCounterSampleBuffer> sb = [dev newCounterSampleBufferWithDescriptor:d error:&err];
  [d release];
  if (sb == nil) {
    last_error_ = "counter sample buffer creation failed";
    if (err != nil) {
      last_error_ += ": ";
      last_error_ += [[err localizedDescription] UTF8String];
    }
    gpu_profile_ = false;
    return;
  }
  sample_buf_ = (void*)sb;  // +1
  sample_next_ = 0;
  sample_names_.clear();
}

void MetalContext::resolve_gpu_profile() {
  gpu_profile_out_.clear();
  if (sample_buf_ == nullptr || sample_names_.empty()) return;
  id<MTLCounterSampleBuffer> sb = (id<MTLCounterSampleBuffer>)sample_buf_;
  NSData* data = [sb resolveCounterRange:NSMakeRange(0, (NSUInteger)sample_next_)];
  if (data == nil) return;
  const MTLCounterResultTimestamp* r = (const MTLCounterResultTimestamp*)[data bytes];
  const std::size_t pairs = std::min(sample_names_.size(), (std::size_t)(sample_next_ / 2));

  std::unordered_map<std::string, std::pair<std::uint64_t, std::uint64_t>> acc;
  for (std::size_t i = 0; i < pairs; ++i) {
    const std::uint64_t t0 = r[i * 2].timestamp, t1 = r[i * 2 + 1].timestamp;
    // A sample pair can come back zeroed when the encoder was empty or the counter was not
    // written; counting those as 0 ns would quietly deflate a kernel's total.
    if (t1 <= t0) continue;
    auto& e = acc[sample_names_[i]];
    e.first += t1 - t0;
    e.second += 1;
  }
  gpu_profile_out_.assign(acc.begin(), acc.end());
  std::sort(gpu_profile_out_.begin(), gpu_profile_out_.end(),
            [](const auto& a, const auto& b) { return a.second.first > b.second.first; });
}

const std::vector<std::pair<std::string, std::pair<std::uint64_t, std::uint64_t>>>&
MetalContext::gpu_profile_results() {
  return gpu_profile_out_;
}

void MetalContext::set_next_specialization(const std::uint32_t* values, int count) {
  if (count < 0) count = 0;
  if (count > kMaxSpecConstants) {
    throw std::runtime_error("MetalContext: too many specialization constants");
  }
  for (int i = 0; i < count; ++i) next_spec_[i] = values[i];
  next_spec_n_ = count;
}

void MetalContext::dispatch(const std::string& name, Grid grid, std::size_t total,
                            std::size_t tg_size, const void* const* buffers,
                            const std::size_t* offsets, int n_buffers, const void* params,
                            std::size_t params_bytes) {
  if (device_ == nullptr || library_ == nullptr || total == 0) return;

  id<MTLDevice> dev = (id<MTLDevice>)device_;

  // The cache key carries the specialization, so each shape gets its own pipeline and an
  // unspecialized dispatch of the same kernel never picks up a specialized one.
  const int spec_n = next_spec_n_;
  std::uint32_t spec[kMaxSpecConstants];
  for (int i = 0; i < spec_n; ++i) spec[i] = next_spec_[i];
  next_spec_n_ = 0;

  // Key built as a std::string. The unspecialized case -- which is every dispatch in a decode
  // step -- does no allocation at all: the lookup hashes `name` in place. Only the rare
  // specialized dispatch builds a string, and only the first time does any of this touch ObjC.
  const std::string* keyp = &name;
  std::string spec_key;
  if (spec_n > 0) {
    spec_key = name;
    for (int i = 0; i < spec_n; ++i) {
      spec_key += '/';
      spec_key += std::to_string(spec[i]);
    }
    keyp = &spec_key;
  }

  id<MTLComputePipelineState> pso = nil;
  auto& pmap = *(std::unordered_map<std::string, void*>*)pipelines_;
  auto it = pmap.find(*keyp);
  if (it != pmap.end()) {
    pso = (id<MTLComputePipelineState>)it->second;
  }
  if (pso == nil) {
    NSString* fname = [NSString stringWithUTF8String:name.c_str()];
    id<MTLLibrary> lib = (id<MTLLibrary>)library_;
    NSError* err = nil;
    id<MTLFunction> fn = nil;
    if (spec_n > 0) {
      MTLFunctionConstantValues* fcv = [[MTLFunctionConstantValues alloc] init];
      for (int i = 0; i < spec_n; ++i) {
        [fcv setConstantValue:&spec[i] type:MTLDataTypeUInt atIndex:(NSUInteger)i];
      }
      fn = [lib newFunctionWithName:fname constantValues:fcv error:&err];
      [fcv release];
    } else {
      fn = [lib newFunctionWithName:fname];
    }
    if (fn == nil) {
      last_error_ = "no such kernel in metallib: " + name;
      if (err != nil) {
        last_error_ += ": ";
        last_error_ += [[err localizedDescription] UTF8String];
      }
      return;
    }
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
    // The map takes the +1 from newComputePipelineState; ~MetalContext releases it.
    pmap.emplace(*keyp, (void*)pso);
  }

  // One command buffer accumulates every dispatch until commit_and_wait(). This
  // is the cheap analogue of a CUDA stream: encoding is the low-overhead part on
  // Metal, so a decode step's ops all land in one buffer and one submission.
  if (cmdbuf_ == nullptr) {
    id<MTLCommandQueue> q = (id<MTLCommandQueue>)queue_;
    cmdbuf_ = (void*)[[q commandBuffer] retain];
  }
  id<MTLCommandBuffer> cb = (id<MTLCommandBuffer>)cmdbuf_;

  // ONE compute encoder is reused across every dispatch in the command buffer, closed only at
  // commit. A decode step issues a few hundred dispatches; a fresh encoder per dispatch (the
  // previous design) paid the driver's encoder setup and an implicit full barrier every time,
  // which is a large fixed per-token cost -- and that overhead is a much bigger fraction of a
  // bandwidth-light int4 decode than of a bandwidth-heavy fp16 one. Ordering between dependent
  // dispatches is kept by an explicit memory barrier (below), which is far cheaper than an
  // encoder boundary.
  if (encoder_ == nullptr) {
    // CONCURRENT dispatch, not the default serial. The slot/weight buffers are Shared and
    // hazard-TRACKED (no Untracked flag on alloc), so Metal inserts a dependency barrier
    // automatically wherever a dispatch reads what an earlier one wrote -- and ONLY there. Under a
    // serial encoder that automatic tracking is moot: serial orders every dispatch regardless, so
    // the independent ops in a layer (the three Q/K/V projections, gate/up) are force-serialised
    // and each pays a pipeline drain. Concurrent lets those overlap -- the small grid-starved k/v
    // projections in particular, which never fill the GPU alone -- while the tracker still
    // serialises the real dependency chain (norm -> qkv -> rope -> attention -> o -> ...). This is
    // the correct-by-construction version of an earlier attempt that tracked hazards by hand and
    // reset the state per layer, missing cross-layer dependencies; Metal's tracking spans the whole
    // command buffer and has no such seam.
    encoder_ =
        (void*)[[cb computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent] retain];
  }
  // Profiling brackets each dispatch with its own encoder, because the timestamp counter samples
  // at ENCODER boundaries -- the device does not support dispatch-boundary sampling. That
  // serialises work the concurrent encoder would overlap, which is the price of per-kernel numbers.
  if (gpu_profile_ && sample_buf_ != nullptr && sample_next_ + 2 <= sample_slots_) {
    if (encoder_ != nullptr) {
      [(id<MTLComputeCommandEncoder>)encoder_ endEncoding];
      [(id<MTLComputeCommandEncoder>)encoder_ release];
      encoder_ = nullptr;
    }
    MTLComputePassDescriptor* pd = [MTLComputePassDescriptor computePassDescriptor];
    pd.sampleBufferAttachments[0].sampleBuffer = (id<MTLCounterSampleBuffer>)sample_buf_;
    pd.sampleBufferAttachments[0].startOfEncoderSampleIndex = (NSUInteger)sample_next_;
    pd.sampleBufferAttachments[0].endOfEncoderSampleIndex = (NSUInteger)(sample_next_ + 1);
    id<MTLCommandBuffer> pcb = (id<MTLCommandBuffer>)cmdbuf_;
    encoder_ = (void*)[[pcb computeCommandEncoderWithDescriptor:pd] retain];
    sample_names_.push_back(name);
    sample_next_ += 2;
  }

  id<MTLComputeCommandEncoder> enc = (id<MTLComputeCommandEncoder>)encoder_;
  // The encoder is concurrent, so a dispatch runs as soon as its inputs are ready UNLESS a barrier
  // orders it. By default every dispatch barriers first -- correctness-equivalent to a serial
  // encoder. The caller drops the barrier (set_next_barrier(false)) only for a dispatch it has
  // proven independent of everything since the last barrier, which lets the small grid-starved
  // projections (q/k/v, gate/up) overlap. The flag is one-shot: it resets to true here so an
  // un-marked dispatch is always safe, which is what keeps every multi-dispatch op's internal
  // ordering correct without the caller having to think about it.
  if (next_barrier_) [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
  next_barrier_ = true;
  [enc setComputePipelineState:pso];
  ++dispatch_count_;

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
}

bool MetalContext::begin_gputrace(const std::string& path) {
  if (device_ == nullptr || queue_ == nullptr) {
    last_error_ = "no Metal device/queue to capture";
    return false;
  }
  MTLCaptureManager* mgr = [MTLCaptureManager sharedCaptureManager];
  // A .gputrace DOCUMENT is the point -- the default destination is the Xcode UI over a live
  // debug session, which is useless from a headless SSH run.
  if (![mgr supportsDestination:MTLCaptureDestinationGPUTraceDocument]) {
    last_error_ =
        "GPU trace documents are not supported here; set MTL_CAPTURE_ENABLED=1 before starting "
        "the process (Metal arms capture at device creation, not at startCapture)";
    return false;
  }
  // Metal refuses to overwrite an existing trace, and the error it gives for that is not
  // obvious, so clear it first.
  NSString* p = [NSString stringWithUTF8String:path.c_str()];
  [[NSFileManager defaultManager] removeItemAtPath:p error:nil];

  MTLCaptureDescriptor* d = [[MTLCaptureDescriptor alloc] init];
  // Capture the QUEUE, not the device: the device scope records every queue in the process,
  // and this one only ever uses one.
  d.captureObject = (id<MTLCommandQueue>)queue_;
  d.destination = MTLCaptureDestinationGPUTraceDocument;
  d.outputURL = [NSURL fileURLWithPath:p];
  NSError* err = nil;
  const bool ok = [mgr startCaptureWithDescriptor:d error:&err];
  [d release];
  if (!ok) {
    last_error_ = std::string("startCapture failed: ") +
                  (err != nil ? [[err localizedDescription] UTF8String] : "unknown");
    return false;
  }
  capturing_ = true;
  return true;
}

void MetalContext::end_gputrace() {
  if (!capturing_) return;
  // Anything still encoded but unsubmitted would be missing from the trace, so flush first.
  commit_and_wait();
  [[MTLCaptureManager sharedCaptureManager] stopCapture];
  capturing_ = false;
}

void MetalContext::commit_async() {
  if (cmdbuf_ == nullptr) return;
  id<MTLCommandBuffer> cb = (id<MTLCommandBuffer>)cmdbuf_;
  if (encoder_ != nullptr) {
    [(id<MTLComputeCommandEncoder>)encoder_ endEncoding];
    [(id<MTLComputeCommandEncoder>)encoder_ release];
    encoder_ = nullptr;
  }
  [cb commit];  // retained reference moves to in_flight_; wait_pending() releases it
  in_flight_.push_back(cmdbuf_);
  ++cmdbuf_count_;
  cmdbuf_ = nullptr;
}

void MetalContext::wait_pending() {
  if (in_flight_.empty()) return;
  // The queue is serial, so completion of the LAST buffer implies all earlier ones -- but each
  // is waited (cheap once complete) so its GPU-busy time and error status are collected.
  for (void* p : in_flight_) {
    id<MTLCommandBuffer> cb = (id<MTLCommandBuffer>)p;
    [cb waitUntilCompleted];
    const double busy = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
    if (busy > 0.0) gpu_busy_ms_ += busy;
    if ([cb status] == MTLCommandBufferStatusError) {
      NSError* err = [cb error];
      last_error_ = "command buffer failed";
      if (err != nil) {
        last_error_ += ": ";
        last_error_ += [[err localizedDescription] UTF8String];
      }
    }
    [cb release];
  }
  in_flight_.clear();
}

void MetalContext::commit_and_wait() {
  // Anything committed asynchronously belongs to the same stream of work; a caller asking for
  // a full sync means ALL of it.
  wait_pending();
  if (cmdbuf_ == nullptr) return;
  id<MTLCommandBuffer> cb = (id<MTLCommandBuffer>)cmdbuf_;
  if (encoder_ != nullptr) {
    [(id<MTLComputeCommandEncoder>)encoder_ endEncoding];
    [(id<MTLComputeCommandEncoder>)encoder_ release];
    encoder_ = nullptr;
  }
  [cb commit];
  [cb waitUntilCompleted];
  // GPUStartTime/GPUEndTime bracket the actual on-GPU execution of this buffer. Summed
  // against wall-clock it separates kernel time from dispatch/CPU overhead.
  const double busy = ([cb GPUEndTime] - [cb GPUStartTime]) * 1000.0;
  if (busy > 0.0) gpu_busy_ms_ += busy;
  ++cmdbuf_count_;
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
  if (gpu_profile_) resolve_gpu_profile();
}

}  // namespace runtime
