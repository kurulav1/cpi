cmake_minimum_required(VERSION 3.24)
project(cpi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)
enable_testing()

option(LLAMA_ENGINE_ENABLE_CUDA "Enable CUDA GPU inference and tools" ON)
# Metal is the Apple Silicon backend. Defaults ON only on Apple, and is the exact
# analogue of LLAMA_ENGINE_ENABLE_CUDA: the two are independent, and a machine
# with neither still gets the portable CPU engines.
if(APPLE)
  option(LLAMA_ENGINE_ENABLE_METAL "Enable the Metal GPU backend (Apple Silicon)" ON)
else()
  option(LLAMA_ENGINE_ENABLE_METAL "Enable the Metal GPU backend (Apple Silicon)" OFF)
endif()
option(LLAMA_ENGINE_ENABLE_LTO "Enable LTO" ON)
option(LLAMA_ENGINE_WARNINGS_AS_ERRORS "Treat warnings as errors" OFF)
option(LLAMA_ENGINE_ENABLE_CLANG_TIDY "Enable clang-tidy during build" OFF)
option(LLAMA_ENGINE_ENABLE_SENTENCEPIECE "Enable sentencepiece tokenizer support" ON)

if(LLAMA_ENGINE_ENABLE_CUDA)
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
      message(WARNING "CUDAToolkit was not found; falling back to a CPU-only build. Set LLAMA_ENGINE_ENABLE_CUDA=OFF to silence this warning.")
      set(LLAMA_ENGINE_ENABLE_CUDA OFF)
    endif()
  else()
    message(WARNING "A CUDA compiler was not found; falling back to a CPU-only build. Set LLAMA_ENGINE_ENABLE_CUDA=OFF to silence this warning.")
    set(LLAMA_ENGINE_ENABLE_CUDA OFF)
  endif()
endif()

if(LLAMA_ENGINE_ENABLE_CUDA)
  message(STATUS "CUDA support enabled")
  message(STATUS "Building CUDA kernels for architecture(s): ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "CUDA support disabled; building CPU-only targets")
endif()

if(MSVC)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/W4>)
  if(LLAMA_ENGINE_WARNINGS_AS_ERRORS)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:/WX>)
  endif()
else()
  # Quote the genexp so its internal ';' list stays inside it; unquoted, CMake would
  # split at parse time and leave a stray "-Wpedantic>" token on gcc/clang.
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>")
  if(LLAMA_ENGINE_WARNINGS_AS_ERRORS)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Werror>)
  endif()
endif()

find_package(OpenMP)
find_program(CLANG_FORMAT_EXE NAMES clang-format)
find_program(CLANG_TIDY_EXE NAMES clang-tidy)

if(LLAMA_ENGINE_ENABLE_SENTENCEPIECE)
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

set(LLAMA_ENGINE_SOURCES
  src/platform/mmap_file.cpp
  src/model/weight_loader.cpp
  src/model/safetensors_loader.cpp
  src/model/hf_bpe_tokenizer.cpp
  src/model/tokenizer.cpp
  src/model/png.cpp
  src/model/image_preprocess.cpp
  src/model/wordpiece_tokenizer.cpp
  src/engine/embedding_config.cpp
  src/grammar/grammar.cpp
  src/grammar/json_schema_to_grammar.cpp
  src/grammar/grammar_sampler.cpp
  src/engine/cpu_engine.cpp
  src/engine/llama4_cpu_engine.cpp
  src/engine/qwen35_cpu_engine.cpp
)

if(LLAMA_ENGINE_ENABLE_CUDA)
  list(APPEND LLAMA_ENGINE_SOURCES
    # Splices an encoded image into the prompt -- calls PlanCudaEngine directly,
    # so it belongs to the CUDA path, not the portable one.
    src/app/image_prompt.cpp
    src/runtime/system_info.cpp
    src/runtime/cuda_utils.cu
    src/runtime/kv_cache.cpp
    src/kernels/kernels.cu
    src/kernels/kernels_attention_decode.cu
    src/kernels/kernels_attention_decode_int4.cu
    src/kernels/kernels_ops_matvec.cu
    src/kernels/kernels_moe_experts.cu
    src/kernels/kernels_gemm.cu
    src/kernels/kernels_vision.cu
    src/kernels/kernels_weight_only_matvec.cu
    src/kernels/kernels_turboquant.cu
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
    src/engine/llama_engine_sampling_utils.cpp
    src/engine/llama_engine_generation.cpp
    src/engine/llama_engine_batched_decode.cpp
    src/engine/llama_engine_sampling.cpp
    src/engine/paged_kv.cpp
    src/engine/speculative_decoder.cpp
    src/engine/llama_engine_parity.cpp
    src/engine/llama4_cuda_engine.cu
    src/engine/llama4_cuda_engine_setup.cu
    src/engine/plan_cuda_engine.cu
    src/engine/decode_driver.cpp
    src/engine/bert_embedder.cu
  )
endif()

if(LLAMA_ENGINE_ENABLE_METAL)
  list(APPEND LLAMA_ENGINE_SOURCES src/runtime/metal_context.mm)
endif()

add_library(llama_engine ${LLAMA_ENGINE_SOURCES})
target_include_directories(llama_engine PUBLIC include)

if(LLAMA_ENGINE_ENABLE_CUDA)
  target_link_libraries(llama_engine
    PUBLIC
      CUDA::cudart
      CUDA::cublas
      CUDA::cublasLt
  )
endif()

target_compile_definitions(llama_engine PUBLIC LLAMA_ENGINE_HAS_CUDA=$<BOOL:${LLAMA_ENGINE_ENABLE_CUDA}>)

# ---------------------------------------------------------------------------
# Metal backend (Apple Silicon)
# ---------------------------------------------------------------------------
if(LLAMA_ENGINE_ENABLE_METAL)
  # metal_context.mm is Objective-C++ and is deliberately built WITHOUT ARC, so a
  # Metal object can live behind a plain void* in a header that knows nothing
  # about Objective-C.
  set_source_files_properties(src/runtime/metal_context.mm PROPERTIES
    COMPILE_FLAGS "-fno-objc-arc")

  target_link_libraries(llama_engine PUBLIC
    "-framework Metal"
    "-framework Foundation"
  )

  # Compile the shaders to a .metallib next to the binaries. `metal` is an offline
  # compiler, so this step works even on a machine with no GPU (a CI VM) -- only
  # RUNNING the kernels needs real hardware.
  set(CPI_METAL_SRC ${CMAKE_CURRENT_SOURCE_DIR}/src/kernels/metal/cpi_kernels.metal)
  set(CPI_METAL_AIR ${CMAKE_CURRENT_BINARY_DIR}/cpi_kernels.air)
  set(CPI_METALLIB  ${CMAKE_CURRENT_BINARY_DIR}/cpi_kernels.metallib)

  add_custom_command(
    OUTPUT ${CPI_METALLIB}
    COMMAND xcrun -sdk macosx metal -Werror -c ${CPI_METAL_SRC} -o ${CPI_METAL_AIR}
    COMMAND xcrun -sdk macosx metallib ${CPI_METAL_AIR} -o ${CPI_METALLIB}
    DEPENDS ${CPI_METAL_SRC}
    COMMENT "Compiling Metal shaders -> cpi_kernels.metallib"
    VERBATIM
  )
  add_custom_target(cpi_metallib ALL DEPENDS ${CPI_METALLIB})
  add_dependencies(llama_engine cpi_metallib)
endif()

if(sentencepiece_FOUND)
  target_link_libraries(llama_engine PUBLIC sentencepiece::sentencepiece)
  target_compile_definitions(llama_engine PUBLIC LLAMA_ENGINE_HAS_SENTENCEPIECE=1)
endif()

# OpenMP for the CPU engine (parallel GEMV rows).
if(OpenMP_CXX_FOUND)
  target_link_libraries(llama_engine PUBLIC OpenMP::OpenMP_CXX)
endif()

# AVX2 + F16C for the CPU engine GEMV (8-wide FP32 SIMD, on-the-fly FP16
# conversion).  Applied only to cpu_engine.cpp to avoid forcing AVX2 on CUDA
# or kernels that do not benefit from it.
if(MSVC)
  set_source_files_properties(src/engine/cpu_engine.cpp PROPERTIES
    COMPILE_OPTIONS "/arch:AVX2")
  set_source_files_properties(src/engine/llama4_cpu_engine.cpp PROPERTIES
    COMPILE_OPTIONS "/arch:AVX2")
  set_source_files_properties(src/engine/qwen35_cpu_engine.cpp PROPERTIES
    COMPILE_OPTIONS "/arch:AVX2")
else()
  set_source_files_properties(src/engine/cpu_engine.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mf16c;-mfma")
  set_source_files_properties(src/engine/llama4_cpu_engine.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mfma")
  set_source_files_properties(src/engine/qwen35_cpu_engine.cpp PROPERTIES
    COMPILE_OPTIONS "-mavx2;-mfma")
endif()

set_target_properties(llama_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(LLAMA_ENGINE_ENABLE_CUDA)
  set_target_properties(llama_engine PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

if(LLAMA_ENGINE_ENABLE_LTO)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT LTO_SUPPORTED OUTPUT LTO_ERROR)
  if(LTO_SUPPORTED)
    set_property(TARGET llama_engine PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(WARNING "LTO not supported: ${LTO_ERROR}")
  endif()
endif()

add_executable(llama_infer
  src/main.cpp
  src/app/main_helpers.cpp
  src/app/main_cli.cpp
  src/app/main_modes.cpp
  src/app/main_interactive_batch.cpp
)
target_link_libraries(llama_infer PRIVATE llama_engine)

# cpi_embed: embedding worker (CUDA BERT-family encoder). Needs CUDA.
if(LLAMA_ENGINE_ENABLE_CUDA)
  add_executable(cpi_embed
    src/cpi_embed.cpp
    src/app/main_helpers.cpp
  )
  target_link_libraries(cpi_embed PRIVATE llama_engine)
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

# Metal kernel smoke test: runs each shader against a CPU reference. On a machine
# with no Metal GPU (any VM, including GitHub's macOS runners) it reports SKIP and
# passes -- so a green run here proves it COMPILES AND LINKS, never that the
# kernels are correct. Only real Apple Silicon can prove that.
if(LLAMA_ENGINE_ENABLE_METAL)
  add_executable(metal_smoke src/tests/metal_smoke.cpp)
  target_link_libraries(metal_smoke PRIVATE llama_engine)
  target_include_directories(metal_smoke PRIVATE include)
  add_test(NAME metal_smoke COMMAND metal_smoke)
  set_tests_properties(metal_smoke PROPERTIES
    ENVIRONMENT "CPI_METALLIB=${CPI_METALLIB}")
endif()

# Tokenizer byte-table test. CUDA-free: built from the self-contained HfBpe
# tokenizer backend so it runs in CI without a GPU.
add_executable(tokenizer_pieces_test
  src/tests/tokenizer_pieces_test.cpp
  src/model/hf_bpe_tokenizer.cpp
)
target_include_directories(tokenizer_pieces_test PRIVATE include)
add_test(NAME tokenizer_pieces_test COMMAND tokenizer_pieces_test)

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

if(LLAMA_ENGINE_ENABLE_CUDA)
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
  target_link_libraries(moe_kernel_parity_test PRIVATE llama_engine)
  set_target_properties(moe_kernel_parity_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME moe_kernel_parity_test COMMAND moe_kernel_parity_test)

  # Image preprocessing: bicubic resize vs PIL + patch-grid layout.
  add_executable(image_preprocess_test
    src/tests/image_preprocess_test.cpp
  )
  target_include_directories(image_preprocess_test PRIVATE include)
  target_link_libraries(image_preprocess_test PRIVATE llama_engine)

  # Hand-rolled PNG decoder: byte-exact vs fixtures made with a real DEFLATE encoder.
  add_executable(png_test
    src/tests/png_test.cpp
  )
  target_include_directories(png_test PRIVATE include)
  target_link_libraries(png_test PRIVATE llama_engine)

  # Sequence prefill == token-by-token prefill (incl. beyond the sliding window).
  add_executable(seq_prefill_test
    src/tests/seq_prefill_test.cu
  )
  target_include_directories(seq_prefill_test PRIVATE include)
  target_link_libraries(seq_prefill_test PRIVATE llama_engine)
  set_target_properties(seq_prefill_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Vision-tower parity against the HF oracle (tools/gemma4_vision_oracle.py).
  add_executable(vision_forward_test
    src/tests/vision_forward_test.cu
  )
  target_include_directories(vision_forward_test PRIVATE include)
  target_link_libraries(vision_forward_test PRIVATE llama_engine)
  set_target_properties(vision_forward_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # LM-head GEMV bandwidth: it is ~28% of a small model's entire weight traffic.
  add_executable(lm_head_bench
    src/tests/lm_head_bench.cu
  )
  target_include_directories(lm_head_bench PRIVATE include)
  target_link_libraries(lm_head_bench PRIVATE llama_engine)
  set_target_properties(lm_head_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Sequence-mode fp16 GEMM parity (vs a CPU reference and vs the GEMV).
  add_executable(gemm_parity_test
    src/tests/gemm_parity_test.cu
  )
  target_include_directories(gemm_parity_test PRIVATE include)
  target_link_libraries(gemm_parity_test PRIVATE llama_engine)
  set_target_properties(gemm_parity_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )

  # Gemma 4 forward parity driver (checks the fork engine against the HF oracle).
  add_executable(gemma4_forward_test
    src/tests/gemma4_forward_test.cu
  )
  target_include_directories(gemma4_forward_test PRIVATE include)
  target_link_libraries(gemma4_forward_test PRIVATE llama_engine)
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
  target_link_libraries(attention_reference_parity_test PRIVATE llama_engine)
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
  target_link_libraries(batched_paged_attention_reference_parity_test PRIVATE llama_engine)
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
  target_link_libraries(int4_attention_reference_parity_test PRIVATE llama_engine)
  set_target_properties(int4_attention_reference_parity_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME int4_attention_reference_parity_test
           COMMAND int4_attention_reference_parity_test)

  # Paged split-K decode attention parity (P3 phase 2b): paged == contiguous.
  add_executable(paged_attention_parity_test
    src/tests/paged_attention_parity_test.cu
  )
  target_include_directories(paged_attention_parity_test PRIVATE include)
  target_link_libraries(paged_attention_parity_test PRIVATE llama_engine)
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
  target_link_libraries(paged_batched_attention_parity_test PRIVATE llama_engine)
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
  target_link_libraries(paged_batched_primitives_test PRIVATE llama_engine)
  set_target_properties(paged_batched_primitives_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
  add_test(NAME paged_batched_primitives_test COMMAND paged_batched_primitives_test)

  add_executable(int4_gemv_bench
    src/tests/int4_gemv_bench.cu
  )
  target_include_directories(int4_gemv_bench PRIVATE include)
  target_link_libraries(int4_gemv_bench PRIVATE llama_engine)
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
  target_link_libraries(attention_decode_bench PRIVATE llama_engine)
  set_target_properties(attention_decode_bench PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
  )
endif()

if(LLAMA_ENGINE_ENABLE_CLANG_TIDY AND CLANG_TIDY_EXE)
  set_target_properties(llama_engine PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  set_target_properties(llama_infer PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  if(TARGET cuda_bandwidth_bench)
    set_target_properties(cuda_bandwidth_bench PROPERTIES CXX_CLANG_TIDY "${CLANG_TIDY_EXE}")
  endif()
endif()

file(GLOB_RECURSE LLAMA_ENGINE_SOURCES
  CONFIGURE_DEPENDS
  include/*.hpp
  include/*.h
  include/*.cuh
  src/*.cpp
  src/*.cu
)

if(CLANG_FORMAT_EXE)
  add_custom_target(format
    COMMAND ${CLANG_FORMAT_EXE} -i ${LLAMA_ENGINE_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-format on project sources"
  )

  add_custom_target(format-check
    COMMAND ${CLANG_FORMAT_EXE} --dry-run --Werror ${LLAMA_ENGINE_SOURCES}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Checking source formatting with clang-format"
  )
endif()

if(CLANG_TIDY_EXE)
  add_custom_target(tidy
    COMMAND ${CLANG_TIDY_EXE} ${LLAMA_ENGINE_SOURCES} -- -I${CMAKE_SOURCE_DIR}/include
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running clang-tidy (standalone mode)"
  )
endif()
