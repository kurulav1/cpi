cmake_minimum_required(VERSION 3.24)
project(cpi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
include(CTest)
enable_testing()

option(LLAMA_ENGINE_ENABLE_CUDA "Enable CUDA GPU inference and tools" ON)
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
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic>)
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
  src/engine/cpu_engine.cpp
  src/engine/llama4_cpu_engine.cpp
  src/engine/qwen35_cpu_engine.cpp
)

if(LLAMA_ENGINE_ENABLE_CUDA)
  list(APPEND LLAMA_ENGINE_SOURCES
    src/runtime/system_info.cpp
    src/runtime/cuda_utils.cu
    src/runtime/kv_cache.cpp
    src/kernels/kernels.cu
    src/kernels/kernels_attention_decode.cu
    src/kernels/kernels_attention_decode_int4.cu
    src/kernels/kernels_ops_matvec.cu
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
    src/engine/llama_engine_sampling.cpp
    src/engine/llama_engine_parity.cpp
    src/engine/llama4_cuda_engine.cu
    src/engine/llama4_cuda_engine_setup.cu
    src/engine/qwen35_cuda_engine.cu
  )
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
)
target_link_libraries(llama_infer PRIVATE llama_engine)

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
