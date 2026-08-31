cmake_minimum_required(VERSION 3.24)
project(cpi VERSION 0.4.0 LANGUAGES CXX)

# Stamp the build with the project version (and the short git SHA when available) so
# `cpi --version` reports exactly what was built. CPI_GIT_SHA is best-effort:
# a source archive with no .git resolves to "unknown", which is fine.
set(CPI_GIT_SHA "unknown")
find_package(Git QUIET)
if(Git_FOUND)
  execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE CPI_GIT_SHA_OUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
  if(CPI_GIT_SHA_OUT)
    set(CPI_GIT_SHA "${CPI_GIT_SHA_OUT}")
  endif()
endif()
add_compile_definitions(CPI_VERSION="${PROJECT_VERSION}" CPI_GIT_SHA="${CPI_GIT_SHA}")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)
enable_testing()

option(CPI_ENABLE_CUDA "Enable CUDA GPU inference and tools" ON)
# Turns the graceful "no toolkit -> CPU-only" fallback below into a hard error.
# That fallback is right for a developer without CUDA and wrong for a job whose
# entire purpose is a CUDA binary: v0.3.0-rc1 shipped a 385 MB asset named
# cpi-windows-x64-cuda that reported "backends: cpu", because the toolkit was not
# usable, CMake degraded silently, and the build went green.
option(CPI_REQUIRE_CUDA "Fail configure if CUDA was requested but is unavailable" OFF)
# Metal is the Apple Silicon backend. Defaults ON only on Apple, and is the exact
# analogue of CPI_ENABLE_CUDA: the two are independent, and a machine
# with neither still gets the portable CPU engines.
if(APPLE)
  option(CPI_ENABLE_METAL "Enable the Metal GPU backend (Apple Silicon)" ON)
else()
  option(CPI_ENABLE_METAL "Enable the Metal GPU backend (Apple Silicon)" OFF)
endif()
option(CPI_ENABLE_LTO "Enable LTO" ON)
option(CPI_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(CPI_ENABLE_CLANG_TIDY "Enable clang-tidy during build" OFF)
option(CPI_ENABLE_SENTENCEPIECE "Enable sentencepiece tokenizer support" ON)

if(CPI_ENABLE_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    # CUDA architecture selection.
    #
    # Default "native" compiles for the GPU installed in the build machine, which
    # gives the smallest and fastest binary for local development.
    #
    # For distributable/release builds that must run on multiple GPU generations,
    # use the provided "cuda-distributable-release" preset or set
    # CMAKE_CUDA_ARCHITECTURES explicitly, e.g.:
    #
    #   cmake -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89;90" ..
    #
    # Common values: 75=Turing (RTX 20xx), 80=Ampere A100, 86=Ampere RTX 30xx,
    #                89=Ada RTX 40xx, 90=Hopper H100.
    # Multi-arch builds are larger and slower to compile.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
      set(CMAKE_CUDA_ARCHITECTURES "native" CACHE STRING "CUDA architectures" FORCE)
    endif()

    find_package(CUDAToolkit QUIET)
    if(NOT CUDAToolkit_FOUND)
      if(CPI_REQUIRE_CUDA)
        message(FATAL_ERROR "CPI_REQUIRE_CUDA=ON but CUDAToolkit was not found.")
      endif()
      message(WARNING "CUDAToolkit was not found; falling back to a CPU-only build. Set CPI_ENABLE_CUDA=OFF to silence this warning.")
      set(CPI_ENABLE_CUDA OFF)
    endif()
  else()
    if(CPI_REQUIRE_CUDA)
      message(FATAL_ERROR "CPI_REQUIRE_CUDA=ON but no CUDA compiler was found. CMAKE_CUDA_COMPILER "
                        "detection failed; the log above has the reason, commonly an "
                        "MSVC/nvcc version mismatch.")
    endif()
    message(WARNING "A CUDA compiler was not found; falling back to a CPU-only build. Set CPI_ENABLE_CUDA=OFF to silence this warning.")
    set(CPI_ENABLE_CUDA OFF)
  endif()
endif()

if(CPI_ENABLE_CUDA)
  message(STATUS "CUDA support enabled")
  message(STATUS "Building CUDA kernels for architecture(s): ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "CUDA support disabled; building CPU-only targets")
endif()

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
  if(CPI_WARNINGS_AS_ERRORS)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/WX>)
  endif()
else()
  # Quote the genexp so its internal ';' list stays inside it; unquoted, CMake would
  # split at parse time and leave a stray "-Wpedantic>" token on gcc/clang.
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
  if(CPI_WARNINGS_AS_ERRORS)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  endif()
endif()

find_package(OpenMP)
find_program(CLANG_FORMAT_EXE NAMES clang-format)
find_program(CLANG_TIDY_EXE NAMES clang-tidy)

if(CPI_ENABLE_SENTENCEPIECE)
  find_package(sentencepiece CONFIG QUIET)
  if(NOT sentencepiece_FOUND)
    find_path(SENTENCEPIECE_INCLUDE_DIR
      NAMES sentencepiece_processor.h
      PATH_SUFFIXES include
    )
    find_library(SENTENCEPIECE_LIBRARY
      NAMES sentencepiece
      PATH_SUFFIXES lib
    )
    if(SENTENCEPIECE_INCLUDE_DIR AND SENTENCEPIECE_LIBRARY)
      set(sentencepiece_FOUND TRUE)
      add_library(sentencepiece::sentencepiece UNKNOWN IMPORTED)
      set_target_properties(sentencepiece::sentencepiece PROPERTIES
        IMPORTED_LOCATION "${SENTENCEPIECE_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${SENTENCEPIECE_INCLUDE_DIR}"
      )
    endif()
  endif()
endif()

set(CPI_ENGINE_SOURCES
  src/platform/mmap_file.cpp
  src/model/weight_loader.cpp
  src/model/safetensors_loader.cpp
  src/model/gguf_loader.cpp
  src/model/hf_bpe_tokenizer.cpp
  src/model/tokenizer.cpp
  src/model/png.cpp
  src/model/image_preprocess.cpp
  src/model/wordpiece_tokenizer.cpp
  src/engine/embedding_config.cpp
  # Backend-neutral: a plan is just data, so the builder belongs outside any
  # GPU backend and is shared by the CUDA and Metal executors alike.
  src/engine/op_plan_builder.cpp
  # Same reasoning for the config the builder reads: "what shape is this model" is
  # not a backend question. Gemma 4 was CUDA-only because this parse sat in a header
  # that included <cuda_runtime.h>, not because of any missing kernel.
  src/engine/plan_model_config.cpp
  # The shared sampler (temperature / top-k / top-p / repetition penalty / n-gram).
  # It has no CUDA in it and never did; it was only ever in the CUDA block because it
  # lived next to LlamaEngine. Metal needs it, so it moves to the portable list.
  src/engine/llama_engine_sampling_utils.cpp
  # Paged KV bookkeeping (allocator + block tables) and the continuous-batching
  # scheduler. Both are pure host logic with no GPU in them: the scheduler reaches a
  # backend through exactly two virtual calls. paged_kv.cpp used to sit in the CUDA
  # block, and that placement was the only reason continuous batching was CUDA-only;
  # where the code lived, not what it needed.
  src/engine/paged_kv.cpp
  src/engine/det_perturb.cpp
  src/engine/batch_scheduler.cpp
  src/grammar/grammar.cpp
  src/grammar/json_schema_to_grammar.cpp
  src/grammar/grammar_sampler.cpp
  src/engine/cpu_engine.cpp
  src/engine/cpu_gemm_avx512.cpp
  src/engine/llama4_cpu_engine.cpp
  src/engine/qwen35_cpu_engine.cpp
)

if(CPI_ENABLE_CUDA)
  list(APPEND CPI_ENGINE_SOURCES
    src/runtime/system_info.cpp
    src/runtime/cuda_utils.cu
    src/kernels/kernels.cu
    src/kernels/kernels_attention_decode.cu
    src/kernels/kernels_persistent_decode.cu
    src/kernels/kernels_attention_decode_int4.cu
    src/kernels/kernels_paged_quant.cu
    src/kernels/kernels_ops_matvec.cu
    src/kernels/kernels_moe_experts.cu
    src/kernels/kernels_gemm.cu
    src/kernels/kernels_vision.cu
    src/kernels/kernels_weight_only_matvec.cu
    src/kernels/kernels_turboquant.cu
    src/kernels/kernels_kquant.cu
    src/engine/tensor_parallel.cpp
    src/engine/llama_engine.cpp
    src/engine/llama_engine_lowbit_utils.cpp
    src/engine/llama_engine_cache.cpp
    src/engine/llama_engine_cache_streaming.cpp
    src/engine/llama_engine_lifecycle.cpp
    src/engine/llama_engine_tuning.cpp
    src/engine/llama_engine_graph.cpp
    src/engine/llama_engine_forward.cpp
    src/engine/llama_engine_forward_decode.cpp
    src/engine/llama_engine_generation.cpp
    src/engine/llama_engine_eagle.cpp
    src/engine/llama_engine_eagle_tree.cpp
    src/engine/llama_engine_batched_decode.cpp
    src/engine/llama_engine_sampling.cpp
    src/engine/llama_engine_parity.cpp
    src/engine/llama4_cuda_engine.cu
    src/engine/llama4_cuda_engine_setup.cu
    src/engine/plan_cuda_engine.cu
    src/engine/decode_driver.cpp
    src/engine/bert_embedder.cu
  )
endif()

if(CPI_ENABLE_METAL)
  list(APPEND CPI_ENGINE_SOURCES
    src/runtime/metal_context.mm
    src/engine/plan_metal_engine.cpp
    src/engine/plan_metal_vision.cpp
    src/engine/metal_bert_embedder.cpp
  )
endif()

add_library(cpi_engine ${CPI_ENGINE_SOURCES})
target_include_directories(cpi_engine PUBLIC include)

if(CPI_ENABLE_CUDA)
  target_link_libraries(cpi_engine
    PUBLIC
      CUDA::cudart
      CUDA::cublas
      CUDA::cublasLt
  )
endif()

target_compile_definitions(cpi_engine PUBLIC CPI_HAS_CUDA=$<BOOL:${CPI_ENABLE_CUDA}>)
target_compile_definitions(cpi_engine PUBLIC CPI_HAS_METAL=$<BOOL:${CPI_ENABLE_METAL}>)

# ---------------------------------------------------------------------------
# Metal backend (Apple Silicon)
# ---------------------------------------------------------------------------
if(CPI_ENABLE_METAL)
  # metal_context.mm is Objective-C++ and is deliberately built WITHOUT ARC, so a
  # Metal object can live behind a plain void* in a header that knows nothing
  # about Objective-C.
  set_source_files_properties(src/runtime/metal_context.mm PROPERTIES
    COMPILE_FLAGS "-fno-objc-arc")

  target_link_libraries(cpi_engine PUBLIC
    "-framework Metal"
    "-framework Foundation"
  )

  # Exposed as a compile macro (PUBLIC, so cpi sees it) so the main binary can
  # conditionally wire the Metal engine into its dispatch; mirrors CPI_HAS_CUDA.
  target_compile_definitions(cpi_engine PUBLIC CPI_ENABLE_METAL=1)

  # The shaders are split into family files (00_common.metal, 10_dense.metal, ...) under this
  # directory. CPI_METAL_SOURCE is the DIRECTORY: the runtime loader concatenates its *.metal in
  # filename order (the numeric prefixes are the dependency order), and the offline metallib rule
  # below cats the same files the same way. One place defines the order; the filenames.
  set(CPI_METAL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/metal)
  set(CPI_METAL_SRC ${CPI_METAL_DIR})
  file(GLOB CPI_METAL_PARTS ${CPI_METAL_DIR}/[0-9][0-9]_*.metal)
  list(SORT CPI_METAL_PARTS)

  # The offline `metal` compiler ships with XCODE, not with the Command Line Tools.
  # A bare Mac (a fresh cloud rental, say) has clang but no `metal`, and pulling
  # ~15 GB of Xcode just to build a shader is not a reasonable requirement.
  #
  # So the metallib is OPTIONAL. When the toolchain is present we precompile, which
  # is faster to start and type-checks at build time. When it is not, the engine
  # compiles the MSL from source at runtime via newLibraryWithSource; that needs
  # only the Metal framework, which every Mac has. CPI_METAL_SOURCE points at it.
  execute_process(
    COMMAND xcrun -sdk macosx metal --version
    RESULT_VARIABLE CPI_METAL_CC_MISSING
    OUTPUT_QUIET ERROR_QUIET
  )

  if(CPI_METAL_CC_MISSING EQUAL 0)
    set(CPI_METAL_COMBINED ${CMAKE_CURRENT_BINARY_DIR}/cpi_kernels.metal)
    set(CPI_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/cpi_kernels.air)
    set(CPI_METALLIB  ${CMAKE_CURRENT_BINARY_DIR}/cpi_kernels.metallib)
    add_custom_command(
      OUTPUT ${CPI_METALLIB}
      # Concatenate the family files (sorted) into one translation unit, then compile. The
      # offline `metal` compiler could resolve #include, but the runtime loader cannot, so the
      # files carry no includes and both paths just concatenate; same bytes, one source of order.
      COMMAND ${CMAKE_COMMAND} -DOUT=${CPI_METAL_COMBINED} "-DPARTS=${CPI_METAL_PARTS}"
              -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/concat_metal.cmake
      COMMAND xcrun -sdk macosx metal -Werror -c ${CPI_METAL_COMBINED} -o ${CPI_METAL_AIR}
      COMMAND xcrun -sdk macosx metallib ${CPI_METAL_AIR} -o ${CPI_METALLIB}
      DEPENDS ${CPI_METAL_PARTS}
      COMMENT "Compiling Metal shaders -> cpi_kernels.metallib"
      VERBATIM
    )
    add_custom_target(cpi_metallib ALL DEPENDS ${CPI_METALLIB})
    add_dependencies(cpi_engine cpi_metallib)
    message(STATUS "Metal: offline toolchain found; precompiling cpi_kernels.metallib")
  else()
    set(CPI_METALLIB "")
    message(STATUS "Metal: no offline `metal` compiler (Xcode absent); "
                   "shaders will be compiled at runtime from source")
  endif()
endif()

if(sentencepiece_FOUND)
  target_link_libraries(cpi_engine PUBLIC sentencepiece::sentencepiece)
  target_compile_definitions(cpi_engine PUBLIC CPI_HAS_SENTENCEPIECE=1)
endif()

# OpenMP for the CPU engine (parallel GEMV rows).
if(OpenMP_CXX_FOUND)
  target_link_libraries(cpi_engine PUBLIC OpenMP::OpenMP_CXX)
endif()

# AVX2 + F16C for the CPU engine GEMV (8-wide FP32 SIMD, on-the-fly FP16
# conversion).  Applied only to cpu_engine.cpp to avoid forcing AVX2 on CUDA
# or kernels that do not benefit from it.
#
# These are x86 flags and must be gated on the target ISA. Apple Silicon's clang 21
# rejects -mavx2 outright ("unsupported option for target arm64-apple-darwin"); older
# Apple clangs merely ignored it, which is why CI's macOS runner did not catch this.
# The sources already compile to scalar fallbacks on non-x86.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|X86_64|amd64|AMD64|i386|i686)$")
  if(MSVC)
    set_source_files_properties(src/engine/cpu_engine.cpp PROPERTIES
      COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/engine/llama4_cpu_engine.cpp PROPERTIES
      COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/engine/qwen35_cpu_engine.cpp PROPERTIES
      COMPILE_OPTIONS "/arch:AVX2")
    # The AVX-512 GEMM tier lives in exactly one TU; it is dispatched behind a
    # runtime CPUID check in cpu_engine.cpp, so these flags must never spread
    # to files whose code runs unconditionally.
    set_source_files_properties(src/engine/cpu_gemm_avx512.cpp PROPERTIES
      COMPILE_OPTIONS "/arch:AVX512")
  else()
    set_source_files_properties(src/engine/cpu_engine.cpp PROPERTIES
      COMPILE_OPTIONS "-mavx2;-mf16c;-mfma")
    set_source_files_properties(src/engine/llama4_cpu_engine.cpp PROPERTIES
      COMPILE_OPTIONS "-mavx2;-mfma")
    set_source_files_properties(src/engine/qwen35_cpu_engine.cpp PROPERTIES
      COMPILE_OPTIONS "-mavx2;-mfma")
    set_source_files_properties(src/engine/cpu_gemm_avx512.cpp PROPERTIES
      COMPILE_OPTIONS "-mavx512f;-mf16c;-mfma")
  endif()
else()
  message(STATUS "CPU engines: non-x86 target (${CMAKE_SYSTEM_PROCESSOR}); using scalar fallbacks")
endif()

set_target_properties(cpi_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(CPI_ENABLE_CUDA)
  set_target_properties(cpi_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

if(CPI_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
  if(LTO_SUPPORTED)
    set_property(TARGET cpi_engine PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(WARNING "LTO not supported: ${LTO_ERROR}")
  endif()
endif()

add_executable(cpi
  src/main.cpp
  src/app/main_helpers.cpp
  src/app/main_cli.cpp
  src/app/main_modes.cpp
  src/app/main_interactive_batch.cpp
  # The batching loop and the HTTP transport that shares it with the stdin worker.
  src/app/batch_worker.cpp
  src/app/main_http_serve.cpp
  src/net/http_server.cpp
)
target_link_libraries(cpi PRIVATE cpi_engine)

# shard_planner: pre-deployment "will it fit, on how many GPUs" report for a model + tp/pp/ep split.
# Pure host arithmetic over header-only shard_plan.hpp; no GPU, no cluster, no engine link, so it
# builds on any platform. Part of the multi-GPU/HPC prep track (see include/engine/shard_plan.hpp).
add_executable(shard_planner src/app/shard_planner.cpp)
target_include_directories(shard_planner PRIVATE include)

# Backward-compat alias: the binary was renamed llama_infer -> cpi. Drop a `llama_infer` next to it
# (symlink on Unix, copy on Windows) so existing local scripts keep working for one release. This
# is source-build only; the release archives ship `cpi`.
if(NOT DEFINED CPI_NO_LEGACY_ALIAS)
  if(WIN32)
    add_custom_command(TARGET cpi POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              "$<TARGET_FILE:cpi>" "$<TARGET_FILE_DIR:cpi>/llama_infer.exe"
      VERBATIM)
  else()
    add_custom_command(TARGET cpi POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E create_symlink
              "$<TARGET_FILE_NAME:cpi>" "$<TARGET_FILE_DIR:cpi>/llama_infer"
      VERBATIM)
  endif()
endif()

# cpi_embed: embedding worker (BERT-family encoder). Needs a GPU backend; CUDA or Metal.
#
# Was CUDA-only, which is what left embeddings, RAG and folder-search dead on Apple Silicon: the
# target simply did not exist, so the Node server resolved cpi_embed from a CUDA build dir, found
# nothing, and 503'd every embedding route. src/cpi_embed.cpp now selects the implementation at
# compile time; the encoder itself is engine::MetalBertEmbedder on Apple.
if(CPI_ENABLE_CUDA OR CPI_ENABLE_METAL)
  add_executable(cpi_embed
    src/cpi_embed.cpp
    src/app/main_helpers.cpp
  )
  target_link_libraries(cpi_embed PRIVATE cpi_engine)
endif()

# Grammar unit tests. CUDA-free: built from the grammar sources directly so the
# test runs in CI without a GPU and without linking the full engine.
add_executable(grammar_test
  src/tests/grammar_test.cpp
  src/grammar/grammar.cpp
  src/grammar/json_schema_to_grammar.cpp
  src/grammar/grammar_sampler.cpp
)
target_include_directories(grammar_test PRIVATE include)
add_test(NAME grammar_test COMMAND grammar_test)

# Plan-builder test. CUDA-free and GPU-free, so it runs in CI on every platform;
# it is the one part of the Metal bring-up that can be verified without a Mac.
add_executable(op_plan_builder_test
  src/tests/op_plan_builder_test.cpp
  src/engine/op_plan_builder.cpp
  # The shared sampler (temperature / top-k / top-p / repetition penalty / n-gram).
  # It has no CUDA in it and never did; it was only ever in the CUDA block because it
  # lived next to LlamaEngine. Metal needs it, so it moves to the portable list.
  src/engine/llama_engine_sampling_utils.cpp
)
target_include_directories(op_plan_builder_test PRIVATE include)
add_test(NAME op_plan_builder_test COMMAND op_plan_builder_test)

# IEEE binary16 conversions. Header-only maths with no Metal and no GPU, so it is registered
# OUTSIDE the Metal block deliberately: the converters it gates feed every Metal weight upload,
# but the bug they carried (truncation instead of round-to-nearest, subnormals flushed to zero)
# is arithmetic and can be caught anywhere; including on CI and on Windows, where none of the
# Metal targets build. Optional args: a vals/expect fixture pair generated by numpy.
add_executable(metal_fp16_test src/tests/metal_fp16_test.cpp)
target_include_directories(metal_fp16_test PRIVATE include)
add_test(NAME metal_fp16_test COMMAND metal_fp16_test)

# Qwen2-VL / Qwen3.5 image preprocessing vs the HuggingFace processor. Pure CPU (no Metal, no
# CUDA), so it lives OUTSIDE the backend guards and builds anywhere; unlike image_preprocess_test,
# which is stranded under the CUDA guard and so never built on a Mac. No add_test: it needs an
# oracle dump from a torch/torchvision run that CI does not have.
# Repack a .ll2c into a .cpi (safetensors + JSON __metadata__). CPU-only, no backend needed.
add_executable(ll2c_to_cpi src/tools/ll2c_to_cpi.cpp)
target_include_directories(ll2c_to_cpi PRIVATE include)
target_link_libraries(ll2c_to_cpi PRIVATE cpi_engine)

# LlamaConfig <-> .cpi __metadata__ round trip. Header-only and CPU-only, so it runs everywhere.
# GGUF reader. Takes a .gguf path (and optionally a .ll2c of the same model as an
# oracle), so it is not registered with add_test: there is no checked-in fixture
# big enough to be a real model.
# Device k-quant dequant vs the host reference. Needs CUDA; takes an optional
# .gguf so it can run on real blocks rather than only synthetic ones.
# CUDA-only: both are .cu and exercise device kernels. They sat outside the guard,
# so a CPU-only configure still created the targets and registered the tests while
# producing no executables, and ctest failed every CPU-only CI job with "Not Run".
if(CPI_ENABLE_CUDA)
  add_executable(kquant_dequant_test src/tests/kquant_dequant_test.cu)
  target_include_directories(kquant_dequant_test PRIVATE include)
  target_link_libraries(kquant_dequant_test PRIVATE cpi_engine)
  add_test(NAME kquant_dequant_test COMMAND kquant_dequant_test)

  add_executable(kquant_matvec_test src/tests/kquant_matvec_test.cu)
  target_include_directories(kquant_matvec_test PRIVATE include)
  target_link_libraries(kquant_matvec_test PRIVATE cpi_engine)
  add_test(NAME kquant_matvec_test COMMAND kquant_matvec_test)
endif()

add_executable(gguf_loader_test src/tests/gguf_loader_test.cpp)
target_include_directories(gguf_loader_test PRIVATE include)
target_link_libraries(gguf_loader_test PRIVATE cpi_engine)

add_executable(config_json_test src/tests/config_json_test.cpp)
target_include_directories(config_json_test PRIVATE include)
add_test(NAME config_json_test COMMAND config_json_test)

# Which engine a .cpi resolves to. Compiles main_helpers.cpp directly because that file lives in
# the executables, not in llama_engine. CPU-only, so it runs on CI, Windows and the Mac alike.
add_executable(probe_model_test src/tests/probe_model_test.cpp src/app/main_helpers.cpp)
target_include_directories(probe_model_test PRIVATE include)
target_link_libraries(probe_model_test PRIVATE cpi_engine)
add_test(NAME probe_model_test COMMAND probe_model_test)

# The response_format.json_schema unwrap. Compiles main_helpers.cpp directly for the same reason
# probe_model_test does: the helper lives in the executables, not in llama_engine. CPU-only.
add_executable(json_schema_unwrap_test src/tests/json_schema_unwrap_test.cpp src/app/main_helpers.cpp)
target_include_directories(json_schema_unwrap_test PRIVATE include)
target_link_libraries(json_schema_unwrap_test PRIVATE cpi_engine)
add_test(NAME json_schema_unwrap_test COMMAND json_schema_unwrap_test)

add_executable(qwen2vl_preproc_test src/tests/qwen2vl_preproc_test.cpp)
target_include_directories(qwen2vl_preproc_test PRIVATE include)
target_link_libraries(qwen2vl_preproc_test PRIVATE cpi_engine)

# Metal kernel smoke test: runs each shader against a CPU reference. On a machine
# with no Metal GPU (any VM, including GitHub's macOS runners) it reports SKIP and
# passes, so a green run here proves it COMPILES AND LINKS, never that the
# kernels are correct. Only real Apple Silicon can prove that.
if(CPI_ENABLE_METAL)
  # Text generation on Apple Silicon. Separate from cpi, whose engine dispatch
  # is templated over the CUDA engines; keeping this standalone means the Metal path
  # has no CUDA anywhere in its dependency graph.
  add_executable(metal_infer src/app/metal_infer.cpp)
  target_link_libraries(metal_infer PRIVATE cpi_engine)
  target_include_directories(metal_infer PRIVATE include)

  add_executable(metal_vision_test src/tests/metal_vision_test.cpp)
  target_link_libraries(metal_vision_test PRIVATE cpi_engine)
  target_include_directories(metal_vision_test PRIVATE include)
  # No add_test: it needs an oracle directory produced by a torch/transformers run, which the
  # Mac cannot generate. Run it by hand with the dumps copied over.

  # Qwen3.5's text stack vs a HuggingFace logits oracle. No add_test: it needs both a container
  # and an oracle dump from a torch run, neither of which the Mac can produce. It is also the
  # control for metal_vision_test's FIRST_STEP_LOGITS; see the file header.
  add_executable(qwen35_text_test src/tests/qwen35_text_test.cpp)
  target_link_libraries(qwen35_text_test PRIVATE cpi_engine)
  target_include_directories(qwen35_text_test PRIVATE include)

  # The Metal BERT embedder vs a HuggingFace reference. No add_test: it needs both a model dir
  # and an oracle dump from a torch run, neither of which the Mac can produce.
  add_executable(metal_embed_test src/tests/metal_embed_test.cpp)
  target_link_libraries(metal_embed_test PRIVATE cpi_engine)
  target_include_directories(metal_embed_test PRIVATE include)

  add_executable(metal_smoke src/tests/metal_smoke.cpp)
  target_link_libraries(metal_smoke PRIVATE cpi_engine)
  target_include_directories(metal_smoke PRIVATE include)
  add_test(NAME metal_smoke COMMAND metal_smoke)
  # Both are set: the metallib if we have one, and the source path as the fallback
  # the engine uses when there is no offline toolchain.
  set_tests_properties(metal_smoke PROPERTIES
    ENVIRONMENT "CPI_METALLIB=${CPI_METALLIB};CPI_METAL_SOURCE=${CPI_METAL_SRC}")

  # Metal decode vs the CPU engine on the same weights. Needs a model, so it SKIPs
  # (exit 0) with no argument and ctest stays green without one.
  add_executable(metal_decode_test src/tests/metal_decode_test.cpp)
  target_link_libraries(metal_decode_test PRIVATE cpi_engine)
  target_include_directories(metal_decode_test PRIVATE include)
  add_test(NAME metal_decode_test COMMAND metal_decode_test)
  set_tests_properties(metal_decode_test PROPERTIES
    ENVIRONMENT "CPI_METALLIB=${CPI_METALLIB};CPI_METAL_SOURCE=${CPI_METAL_SRC}")

  # Perplexity. Needs a model and a corpus, so it is a tool rather than a ctest case.
  add_executable(metal_ppl src/tests/metal_ppl.cpp)
  target_link_libraries(metal_ppl PRIVATE cpi_engine)
  target_include_directories(metal_ppl PRIVATE include)

  # Isolated fp16 GEMM benchmark that verifies what it times. No model needed.
  add_executable(metal_gemm_bench src/tests/metal_gemm_bench.cpp)
  target_link_libraries(metal_gemm_bench PRIVATE cpi_engine)
  target_include_directories(metal_gemm_bench PRIVATE include)

  # Batched paged decode vs the single-sequence path. Needs a model, so it SKIPs without one.
  add_executable(metal_batched_test src/tests/metal_batched_test.cpp)
  target_link_libraries(metal_batched_test PRIVATE cpi_engine)
  target_include_directories(metal_batched_test PRIVATE include)
  add_test(NAME metal_batched_test COMMAND metal_batched_test)
  set_tests_properties(metal_batched_test PROPERTIES
    ENVIRONMENT "CPI_METALLIB=${CPI_METALLIB};CPI_METAL_SOURCE=${CPI_METAL_SRC}")
endif()

# Tokenizer byte-table test. CUDA-free: built from the self-contained HfBpe
# tokenizer backend so it runs in CI without a GPU.
add_executable(tokenizer_pieces_test
  src/tests/tokenizer_pieces_test.cpp
  src/model/hf_bpe_tokenizer.cpp
)
target_include_directories(tokenizer_pieces_test PRIVATE include)
add_test(NAME tokenizer_pieces_test COMMAND tokenizer_pieces_test)

# The grammar's byte table must equal what decode() emits, per token. Needs a real GGUF, so it
# SKIPs without one; pass a path in CPI_TEST_GGUF to make it run.
add_executable(gguf_pieces_test src/tests/gguf_pieces_test.cpp)
target_include_directories(gguf_pieces_test PRIVATE include)
target_link_libraries(gguf_pieces_test PRIVATE cpi_engine)
add_test(NAME gguf_pieces_test COMMAND gguf_pieces_test)

# Paged-KV allocator/block-table test (CUDA-free host logic; P3 Phase 1).
add_executable(paged_kv_test
  src/tests/paged_kv_test.cpp
  src/engine/paged_kv.cpp
)
target_include_directories(paged_kv_test PRIVATE include)
add_test(NAME paged_kv_test COMMAND paged_kv_test)

# WordPiece tokenizer test (CUDA-free). Needs the bge model dir at run time.
add_executable(wordpiece_test
  src/tests/wordpiece_test.cpp
  src/model/wordpiece_tokenizer.cpp
)
target_include_directories(wordpiece_test PRIVATE include)
add_test(NAME wordpiece_test
  COMMAND wordpiece_test "${CMAKE_SOURCE_DIR}/artifacts/hub/BAAI__bge-small-en-v1.5")
# The bge model dir is gitignored; the test returns 77 to mark itself SKIPPED
# (not FAILED) when the model is absent, e.g. on CI / clean checkouts.
set_tests_properties(wordpiece_test PROPERTIES SKIP_RETURN_CODE 77)

if(CPI_ENABLE_CUDA)
  add_executable(cuda_bandwidth_bench
    src/tools/cuda_bandwidth_bench.cu
    src/runtime/cuda_utils.cu
  )
  target_include_directories(cuda_bandwidth_bench PRIVATE include)
  target_link_libraries(cuda_bandwidth_bench PRIVATE CUDA::cudart)
  set_target_properties(cuda_bandwidth_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  add_executable(moe_kernel_parity_test
    src/tests/moe_kernel_parity_test.cu
  )
  target_include_directories(moe_kernel_parity_test PRIVATE include)
  target_link_libraries(moe_kernel_parity_test PRIVATE cpi_engine)
  set_target_properties(moe_kernel_parity_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME moe_kernel_parity_test COMMAND moe_kernel_parity_test)

  # Batched greedy argmax tie-break vs the host's lowest-index rule (std::max_element).
  add_executable(argmax_tiebreak_test
    src/tests/argmax_tiebreak_test.cu
  )
  target_include_directories(argmax_tiebreak_test PRIVATE include)
  target_link_libraries(argmax_tiebreak_test PRIVATE cpi_engine)
  set_target_properties(argmax_tiebreak_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME argmax_tiebreak_test COMMAND argmax_tiebreak_test)

  # MXFP4 (K3 weight format) dequant: device kernel == host reference, E2M1 LUT + representable
  # round-trip. Gates the decode math before it is wired into the matvec path.
  add_executable(mxfp4_dequant_test
    src/tests/mxfp4_dequant_test.cu
  )
  target_include_directories(mxfp4_dequant_test PRIVATE include)
  target_link_libraries(mxfp4_dequant_test PRIVATE cpi_engine)
  set_target_properties(mxfp4_dequant_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME mxfp4_dequant_test COMMAND mxfp4_dequant_test)

  # Kimi-K3-style MoE router (sigmoid gates, grouped node-limited top-16) vs a host reference.
  # Gates the routing math before it is wired to a model.
  add_executable(moe_router_sigmoid_test
    src/tests/moe_router_sigmoid_test.cu
  )
  target_include_directories(moe_router_sigmoid_test PRIVATE include)
  target_link_libraries(moe_router_sigmoid_test PRIVATE cpi_engine)
  set_target_properties(moe_router_sigmoid_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME moe_router_sigmoid_test COMMAND moe_router_sigmoid_test)

  # Multi-GPU prep brick 1: TensorParallelLinear sharding math verified on ONE GPU (N ranks all
  # mapped to device 0) vs the unsharded forward. Only the cross-device transport needs real HW.
  add_executable(tensor_parallel_test
    src/tests/tensor_parallel_test.cu
  )
  target_include_directories(tensor_parallel_test PRIVATE include)
  target_link_libraries(tensor_parallel_test PRIVATE cpi_engine)
  set_target_properties(tensor_parallel_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME tensor_parallel_test COMMAND tensor_parallel_test)

  # Multi-GPU prep brick 2: RowParallelLinear (input-split + all-reduce) verified on ONE GPU vs the
  # unsharded forward. The all-reduce is the ncclAllReduce seam for real multi-GPU.
  add_executable(row_parallel_test
    src/tests/row_parallel_test.cu
  )
  target_include_directories(row_parallel_test PRIVATE include)
  target_link_libraries(row_parallel_test PRIVATE cpi_engine)
  set_target_properties(row_parallel_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME row_parallel_test COMMAND row_parallel_test)

  # Multi-GPU prep brick 3: expert-parallel MoE (experts sharded across ranks, dispatch + combine)
  # verified on ONE GPU vs all-local, using the real MoE kernels. The per-rank sum is the all_to_all seam.
  add_executable(expert_parallel_test
    src/tests/expert_parallel_test.cu
  )
  target_include_directories(expert_parallel_test PRIVATE include)
  target_link_libraries(expert_parallel_test PRIVATE cpi_engine)
  set_target_properties(expert_parallel_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME expert_parallel_test COMMAND expert_parallel_test)

  # Multi-GPU prep: the collective abstraction (LocalCollective::all_reduce_sum) verified on one GPU.
  # The seam where NcclCollective drops in for real multi-GPU.
  add_executable(collective_test
    src/tests/collective_test.cu
  )
  target_include_directories(collective_test PRIVATE include)
  target_link_libraries(collective_test PRIVATE cpi_engine)
  set_target_properties(collective_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME collective_test COMMAND collective_test)

  # Multi-GPU prep brick 4: pipeline-parallel layer partitioning verified on ONE GPU; a stack of
  # layers split into P contiguous stages, activation handed across each stage boundary, vs unsharded.
  add_executable(pipeline_parallel_test
    src/tests/pipeline_parallel_test.cu
  )
  target_include_directories(pipeline_parallel_test PRIVATE include)
  target_link_libraries(pipeline_parallel_test PRIVATE cpi_engine)
  set_target_properties(pipeline_parallel_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME pipeline_parallel_test COMMAND pipeline_parallel_test)

  # Multi-GPU prep brick 5: the multi-PROCESS collective prototype; ring all-reduce over real TCP
  # sockets (ranks as threads, each a genuine 127.0.0.1 endpoint), the distributed control flow that a
  # single-process test can't reach. Host-only (no CUDA); the transport NcclCollective replaces.
  add_executable(socket_ring_allreduce_test
    src/tests/socket_ring_allreduce_test.cpp
  )
  target_include_directories(socket_ring_allreduce_test PRIVATE include)
  add_test(NAME socket_ring_allreduce_test COMMAND socket_ring_allreduce_test)

  # Shard-fit planner invariants: TP/EP/PP divisors actually divide the footprint; fit scan returns
  # only feasible, world-sorted configs. Host-only (no CUDA); pairs with the empirical world=1 checks.
  add_executable(shard_plan_test
    src/tests/shard_plan_test.cpp
  )
  target_include_directories(shard_plan_test PRIVATE include)
  add_test(NAME shard_plan_test COMMAND shard_plan_test)

  # DeepSeek-style Multi-head Latent Attention (MLA) verified on-device vs an fp32 oracle; isolated
  # groundwork for native DeepSeek-V3/R1 (full checkpoints are model-blocked on this box).
  add_executable(mla_attention_test
    src/tests/mla_attention_test.cu
  )
  target_include_directories(mla_attention_test PRIVATE include)
  target_link_libraries(mla_attention_test PRIVATE cpi_engine)
  set_target_properties(mla_attention_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME mla_attention_test COMMAND mla_attention_test)

  # Kimi Delta Attention (KDA): per-channel gated delta-rule linear attention (recurrent state, no KV
  # cache) verified on-device vs an fp32 oracle. Isolated groundwork for Kimi K3 (model-blocked here).
  add_executable(kda_attention_test
    src/tests/kda_attention_test.cu
  )
  target_include_directories(kda_attention_test PRIVATE include)
  target_link_libraries(kda_attention_test PRIVATE cpi_engine)
  set_target_properties(kda_attention_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME kda_attention_test COMMAND kda_attention_test)

  # DeepSeek MLA composed into a multi-layer decoder stack (MLA + SwiGLU, pre-norm, residuals) verified
  # device vs oracle; the "wire MLA toward a DeepSeek-arch config" step (synthetic weights; a native-
  # MLA checkpoint is the only remaining end-to-end gate).
  add_executable(deepseek_mla_stack_test
    src/tests/deepseek_mla_stack_test.cu
  )
  target_include_directories(deepseek_mla_stack_test PRIVATE include)
  target_link_libraries(deepseek_mla_stack_test PRIVATE cpi_engine)
  set_target_properties(deepseek_mla_stack_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME deepseek_mla_stack_test COMMAND deepseek_mla_stack_test)

  # DeepSeek MLA op matched against a REAL DeepSeek-V2-Lite layer dumped from transformers (test-driven
  # against real weights). Skips cleanly if the artifact hasn't been generated (tools/deepseek_mla_oracle.py).
  add_executable(deepseek_mla_real_test
    src/tests/deepseek_mla_real_test.cu
  )
  target_include_directories(deepseek_mla_real_test PRIVATE include)
  target_link_libraries(deepseek_mla_real_test PRIVATE cpi_engine)
  set_target_properties(deepseek_mla_real_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME deepseek_mla_real_test
           COMMAND deepseek_mla_real_test ${CMAKE_SOURCE_DIR}/artifacts/deepseek_mla_ref)

  # DeepSeek-V2 YARN rope frequencies reproduce transformers' inv_freq (host-only; skips if the HF
  # reference dump is absent). Prerequisite for the in-engine MLA op.
  add_executable(deepseek_rope_test
    src/tests/deepseek_rope_test.cpp
  )
  target_include_directories(deepseek_rope_test PRIVATE include)
  add_test(NAME deepseek_rope_test
           COMMAND deepseek_rope_test WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # DeepSeek MLA assemble+rope kernel (per-head K/V assembly + interleaved rope) verified fp16 vs fp32.
  add_executable(deepseek_mla_assemble_test
    src/tests/deepseek_mla_assemble_test.cu
  )
  target_include_directories(deepseek_mla_assemble_test PRIVATE include)
  target_link_libraries(deepseek_mla_assemble_test PRIVATE cpi_engine)
  set_target_properties(deepseek_mla_assemble_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME deepseek_mla_assemble_test COMMAND deepseek_mla_assemble_test)

  # Image preprocessing: bicubic resize vs PIL + patch-grid layout.
  add_executable(image_preprocess_test
    src/tests/image_preprocess_test.cpp
  )
  target_include_directories(image_preprocess_test PRIVATE include)
  target_link_libraries(image_preprocess_test PRIVATE cpi_engine)

  # Hand-rolled PNG decoder: byte-exact vs fixtures made with a real DEFLATE encoder.
  add_executable(png_test
    src/tests/png_test.cpp
  )
  target_include_directories(png_test PRIVATE include)
  target_link_libraries(png_test PRIVATE cpi_engine)

  # Sequence prefill == token-by-token prefill (incl. beyond the sliding window).
  add_executable(seq_prefill_test
    src/tests/seq_prefill_test.cu
  )
  target_include_directories(seq_prefill_test PRIVATE include)
  target_link_libraries(seq_prefill_test PRIVATE cpi_engine)
  set_target_properties(seq_prefill_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Vision-tower parity against the HF oracle (tools/gemma4_vision_oracle.py).
  add_executable(vision_forward_test
    src/tests/vision_forward_test.cu
  )
  target_include_directories(vision_forward_test PRIVATE include)
  target_link_libraries(vision_forward_test PRIVATE cpi_engine)
  set_target_properties(vision_forward_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # LM-head GEMV bandwidth: it is ~28% of a small model's entire weight traffic.
  add_executable(lm_head_bench
    src/tests/lm_head_bench.cu
  )
  target_include_directories(lm_head_bench PRIVATE include)
  target_link_libraries(lm_head_bench PRIVATE cpi_engine)
  set_target_properties(lm_head_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Sequence-mode fp16 GEMM parity (vs a CPU reference and vs the GEMV).
  add_executable(gemm_parity_test
    src/tests/gemm_parity_test.cu
  )
  target_include_directories(gemm_parity_test PRIVATE include)
  target_link_libraries(gemm_parity_test PRIVATE cpi_engine)
  set_target_properties(gemm_parity_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Sequence-prefill vs per-token parity, all quant configs, short + chunk-crossing
  # prompts. Skips gracefully when the model artifact is absent. Exists because four
  # stacked prefill defects lived behind a green suite.
  add_executable(prefill_parity_test
    src/tests/prefill_parity_test.cu
  )
  target_include_directories(prefill_parity_test PRIVATE include)
  target_link_libraries(prefill_parity_test PRIVATE cpi_engine)
  set_target_properties(prefill_parity_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME prefill_parity_test COMMAND prefill_parity_test
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  # Gemma 4 forward parity driver (checks the fork engine against the HF oracle).
  add_executable(gemma4_forward_test
    src/tests/gemma4_forward_test.cu
  )
  target_include_directories(gemma4_forward_test PRIVATE include)
  target_link_libraries(gemma4_forward_test PRIVATE cpi_engine)
  set_target_properties(gemma4_forward_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Reference-oracle decode attention parity (Phase 0 safety net): the kernel
  # output is compared against an independent naive float reference across
  # head_dim ∈ {64,128,256} and GQA ratios, catching head_dim > blockDim bugs
  # that self-consistency tests miss.
  add_executable(attention_reference_parity_test
    src/tests/attention_reference_parity_test.cu
  )
  target_include_directories(attention_reference_parity_test PRIVATE include)
  target_link_libraries(attention_reference_parity_test PRIVATE cpi_engine)
  set_target_properties(attention_reference_parity_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME attention_reference_parity_test COMMAND attention_reference_parity_test)

  # Reference-oracle batched paged decode attention parity (Phase 0, serving
  # path): independent naive reference vs launch_attention_step_batched_paged
  # across head_dim ∈ {128,256}, catching head_dim > blockDim bugs in the batched
  # split-K reduce that only manifest under batched serving.
  add_executable(batched_paged_attention_reference_parity_test
    src/tests/batched_paged_attention_reference_parity_test.cu
  )
  target_include_directories(batched_paged_attention_reference_parity_test PRIVATE include)
  target_link_libraries(batched_paged_attention_reference_parity_test PRIVATE cpi_engine)
  set_target_properties(batched_paged_attention_reference_parity_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME batched_paged_attention_reference_parity_test
           COMMAND batched_paged_attention_reference_parity_test)

  # Reference-oracle INT4 KV-cache decode attention parity (Phase 0): naive float
  # reference over the dequantized int4 K/V vs launch_attention_step_int4 across
  # head_dim ∈ {128,256} (256 exercises the serial fallback kernel).
  add_executable(int4_attention_reference_parity_test
    src/tests/int4_attention_reference_parity_test.cu
  )
  target_include_directories(int4_attention_reference_parity_test PRIVATE include)
  target_link_libraries(int4_attention_reference_parity_test PRIVATE cpi_engine)
  set_target_properties(int4_attention_reference_parity_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME int4_attention_reference_parity_test
           COMMAND int4_attention_reference_parity_test)

  # Generalized quantized-KV parity: store kernel (incl. R3 Hadamard) + decode
  # attention across (k_bits, v_bits, rotate) combinations vs host reference.
  add_executable(kv_quant_parity_test
    src/tests/kv_quant_parity_test.cu
  )
  target_include_directories(kv_quant_parity_test PRIVATE include)
  target_link_libraries(kv_quant_parity_test PRIVATE cpi_engine)
  set_target_properties(kv_quant_parity_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME kv_quant_parity_test
           COMMAND kv_quant_parity_test)

  # Paged split-K decode attention parity (P3 phase 2b): paged == contiguous.
  add_executable(paged_attention_parity_test
    src/tests/paged_attention_parity_test.cu
  )
  target_include_directories(paged_attention_parity_test PRIVATE include)
  target_link_libraries(paged_attention_parity_test PRIVATE cpi_engine)
  set_target_properties(paged_attention_parity_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME paged_attention_parity_test COMMAND paged_attention_parity_test)

  # Batched paged decode attention parity (P2 primitive): batched == per-sequence.
  add_executable(paged_batched_attention_parity_test
    src/tests/paged_batched_attention_parity_test.cu
  )
  target_include_directories(paged_batched_attention_parity_test PRIVATE include)
  target_link_libraries(paged_batched_attention_parity_test PRIVATE cpi_engine)
  set_target_properties(paged_batched_attention_parity_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME paged_batched_attention_parity_test COMMAND paged_batched_attention_parity_test)

  # Batched-decode primitives (P2): per-position RoPE + batched KV scatter.
  add_executable(paged_batched_primitives_test
    src/tests/paged_batched_primitives_test.cu
  )
  target_include_directories(paged_batched_primitives_test PRIVATE include)
  target_link_libraries(paged_batched_primitives_test PRIVATE cpi_engine)
  set_target_properties(paged_batched_primitives_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME paged_batched_primitives_test COMMAND paged_batched_primitives_test)

  add_executable(int4_gemv_bench
    src/tests/int4_gemv_bench.cu
  )
  target_include_directories(int4_gemv_bench PRIVATE include)
  target_link_libraries(int4_gemv_bench PRIVATE cpi_engine)
  set_target_properties(int4_gemv_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Decode-attention bandwidth bench: locates the memory roofline of the batched
  # paged decode path before any kernel rewrite (harness-first).
  add_executable(attention_decode_bench
    src/tests/attention_decode_bench.cu
  )
  target_include_directories(attention_decode_bench PRIVATE include)
  target_link_libraries(attention_decode_bench PRIVATE cpi_engine)
  set_target_properties(attention_decode_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

if(CPI_ENABLE_CLANG_TIDY AND CLANG_TIDY_EXE)
  set_target_properties(cpi_engine PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  set_target_properties(cpi PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  if(TARGET cuda_bandwidth_bench)
    set_target_properties(cuda_bandwidth_bench PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  endif()
endif()

file(GLOB_RECURSE CPI_ENGINE_SOURCES
  CONFIGURE_DEPENDS
  include/*.hpp
  include/*.h
  include/*.cuh
  src/*.cpp
  src/*.cu
)

if(CLANG_FORMAT_EXE)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXE} -i ${CPI_ENGINE_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-format on project sources"
  )

  add_custom_target(format-check
    COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror ${CPI_ENGINE_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Checking source formatting with clang-format"
  )
endif()

if(CLANG_TIDY_EXE)
  add_custom_target(tidy
    COMMAND ${CLANG_TIDY_EXE} ${CPI_ENGINE_SOURCES}; -I${CMAKE_SOURCE_DIR}/include
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-tidy (standalone mode)"
  )
endif()
