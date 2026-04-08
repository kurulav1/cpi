#pragma once

#include <cuda_runtime.h>

#include <string>

namespace runtime {

void cuda_check(cudaError_t status, const char* file, int line);
void cublas_check(int status, const char* file, int line, const char* expr);

#define CUDA_CHECK(expr) ::runtime::cuda_check((expr), __FILE__, __LINE__)
#define CUBLAS_CHECK(expr) ::runtime::cublas_check((expr), __FILE__, __LINE__, #expr)

/*
 * Returns false if requested bytes exceed available VRAM minus safety margin.
 */
bool can_fit_on_device(std::size_t requested_bytes, std::size_t safety_margin_bytes);

}  // namespace runtime