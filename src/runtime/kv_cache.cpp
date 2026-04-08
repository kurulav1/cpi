#include "runtime/kv_cache.hpp"

#include <algorithm>
#include <limits>

#include "runtime/cuda_utils.cuh"

namespace runtime {

KVCachePager::~KVCachePager() { reset(); }

void KVCachePager::initialize(std::size_t num_pages,
                              std::size_t page_bytes,
                              std::size_t max_device_pages) {
  reset();
  page_bytes_ = page_bytes;

  host_pages_.resize(num_pages, nullptr);
  for (std::size_t i = 0; i < num_pages; ++i) {
    CUDA_CHECK(cudaHostAlloc(&host_pages_[i], page_bytes_, cudaHostAllocPortable));
    CUDA_CHECK(cudaMemset(host_pages_[i], 0, page_bytes_));
  }

  device_pages_.resize(max_device_pages);
  for (auto& slot : device_pages_) {
    CUDA_CHECK(cudaMalloc(&slot.ptr, page_bytes_));
    CUDA_CHECK(cudaMemset(slot.ptr, 0, page_bytes_));
  }
}

void* KVCachePager::touch_page(int page_id) {
  auto it = resident_.find(page_id);
  if (it != resident_.end()) {
    auto& slot = device_pages_[it->second];
    slot.last_access = tick_++;
    return slot.ptr;
  }

  const std::size_t slot = find_free_or_lru_slot();
  page_in(page_id, slot);
  auto& resident_slot = device_pages_[slot];
  resident_slot.last_access = tick_++;
  return resident_slot.ptr;
}

void KVCachePager::reset() {
  for (void* ptr : host_pages_) {
    if (ptr) {
      cudaFreeHost(ptr);
    }
  }
  host_pages_.clear();

  for (auto& slot : device_pages_) {
    if (slot.ptr) {
      cudaFree(slot.ptr);
    }
    slot = {};
  }
  device_pages_.clear();
  resident_.clear();
  tick_ = 1;
  page_bytes_ = 0;
}

std::size_t KVCachePager::find_free_or_lru_slot() const {
  for (std::size_t i = 0; i < device_pages_.size(); ++i) {
    if (device_pages_[i].page_id < 0) {
      return i;
    }
  }

  std::size_t best_idx = 0;
  std::uint64_t best_tick = std::numeric_limits<std::uint64_t>::max();
  for (std::size_t i = 0; i < device_pages_.size(); ++i) {
    if (device_pages_[i].last_access < best_tick) {
      best_tick = device_pages_[i].last_access;
      best_idx = i;
    }
  }
  return best_idx;
}

void KVCachePager::page_in(int page_id, std::size_t slot) {
  if (device_pages_[slot].page_id >= 0) {
    page_out(slot);
  }

  CUDA_CHECK(cudaMemcpy(device_pages_[slot].ptr, host_pages_[page_id], page_bytes_,
                        cudaMemcpyHostToDevice));
  device_pages_[slot].page_id = page_id;
  resident_[page_id] = slot;
}

void KVCachePager::page_out(std::size_t slot) {
  auto& dev = device_pages_[slot];
  if (dev.page_id < 0) {
    return;
  }

  CUDA_CHECK(cudaMemcpy(host_pages_[dev.page_id], dev.ptr, page_bytes_,
                        cudaMemcpyDeviceToHost));
  resident_.erase(dev.page_id);
  dev.page_id = -1;
  dev.last_access = 0;
}

}  // namespace runtime