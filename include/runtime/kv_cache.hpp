#pragma once

// Paged key-value cache manager for transformer attention layers.
//
// KVCachePager implements a two-level (device + host) paging scheme for the
// KV cache used during autoregressive decoding.  A fixed number of pages are
// kept resident in GPU device memory; when all device slots are occupied the
// least-recently-used page is evicted to a host-pinned buffer to make room
// for the newly requested page (LRU eviction policy).
//
// Usage:
//   1. Call initialize() once with the desired capacity parameters.
//   2. During each decode step call touch_page() for every page referenced by
//      the current sequence; the returned pointer is always device-resident.
//   3. Call reset() to invalidate all pages between sequences.

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace runtime {

// Metadata describing a single KV-cache page from the perspective of a
// sequence manager that tracks which pages belong to which sequence.
struct CachePage {
  int page_id = -1;           // Logical page identifier assigned by the sequence manager.
  std::size_t device_slot = 0; // Index into the device page array when the page is resident.
  std::uint64_t last_access = 0; // Monotonically increasing tick value of the last touch_page() call.
};

// Paged KV cache manager.
// Keeps a fixed set of device-resident pages and evicts to host-pinned memory.
class KVCachePager {
 public:
  KVCachePager() = default;
  ~KVCachePager();

  // Allocates num_pages host-pinned pages and max_device_pages GPU device pages,
  // each of page_bytes bytes.  Must be called exactly once before touch_page().
  // Throws on CUDA allocation failure.
  void initialize(std::size_t num_pages,
                  std::size_t page_bytes,
                  std::size_t max_device_pages);

  // Returns a device pointer to the data for page_id, paging it in from host
  // memory if it is not currently resident.  If no free device slot is
  // available the LRU resident page is evicted to host memory first.
  // The returned pointer is valid until the next touch_page() call that
  // triggers an eviction of this same page.
  void* touch_page(int page_id);

  // Marks all pages as invalid and resets the access-tick counter.
  // Does not free any allocations; the pager can be reused after reset().
  void reset();

  // Returns the size in bytes of a single cache page.
  [[nodiscard]] std::size_t page_bytes() const { return page_bytes_; }

  // Returns the number of pages currently resident in device memory.
  [[nodiscard]] std::size_t resident_pages() const { return resident_.size(); }

 private:
  // Represents one physical device-memory slot in the device page pool.
  struct DevicePage {
    void* ptr = nullptr;          // CUDA device pointer for this slot.
    int page_id = -1;             // Logical page currently occupying this slot, or -1 if free.
    std::uint64_t last_access = 0; // Tick of the most recent access, used for LRU eviction.
  };

  std::size_t page_bytes_ = 0;           // Byte size of each individual page.
  std::size_t tick_ = 1;                 // Monotonic counter incremented on every touch_page() call.

  std::vector<void*> host_pages_;        // Host-pinned backing buffers, one per logical page.
  std::vector<DevicePage> device_pages_; // Fixed-size pool of device-resident slots.
  std::unordered_map<int, std::size_t> resident_; // Maps logical page_id -> device_pages_ index.

  // Returns the index of the first free device slot, or the index of the LRU
  // occupied slot if all slots are in use.
  std::size_t find_free_or_lru_slot() const;

  // Copies host page page_id into device slot slot, updating the resident_ map.
  void page_in(int page_id, std::size_t slot);

  // Copies device slot slot back to its corresponding host page and marks the
  // slot as free in the resident_ map.
  void page_out(std::size_t slot);
};

}  // namespace runtime
