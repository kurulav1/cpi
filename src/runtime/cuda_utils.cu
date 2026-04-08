#include "runtime/cuda_utils.cuh"

#include <cublas_v2.h>

#include <sstream>
#include <stdexcept>

namespace runtime {

void cuda_check(cudaError_t status, const char* file, int line) {
  if (status == cudaSuccess) {
    return;
  }
  std::ostringstream oss;
  oss << file << ":" << line << " CUDA error: " << cudaGetErrorString(status);
  throw std::runtime_error(oss.str());
}

void cublas_check(int status, const char* file, int line, const char* expr) {
  if (status == CUBLAS_STATUS_SUCCESS) {
    return;
  }
  std::ostringstream oss;
  oss << file << ":" << line << " CUBLAS error in " << expr << ": " << status;
  throw std::runtime_error(oss.str());
}

bool can_fit_on_device(std::size_t requested_bytes, std::size_t safety_margin_bytes) {
  std::size_t free_b = 0;
  std::size_t total_b = 0;
  cudaError_t rc = cudaMemGetInfo(&free_b, &total_b);
  if (rc != cudaSuccess) {
    return false;
  }
  if (free_b <= safety_margin_bytes) {
    return false;
  }
  return requested_bytes < (free_b - safety_margin_bytes);
}

}  // namespace runtime