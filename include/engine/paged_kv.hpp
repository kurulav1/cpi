#pragma once

#include <cstdint>
#include <vector>

// Paged KV cache foundation (P3, Phase 1) — pure host-side bookkeeping, no CUDA.
//
// vLLM-style block paging: the KV cache is one device pool carved into fixed-size
// blocks of `block_size` tokens. Each sequence owns a *block table* mapping its
// logical token positions to physical block ids; blocks are handed out from a
// free-list on demand and reclaimed when a sequence ends. A block may be shared
// by several sequences (an identical system-prompt prefix) via reference counts,
// which is what makes concurrent shared-prefix (P4) cheap.
//
// This module is the allocator + block-table logic ONLY. Wiring it into the KV
// write/read hot path (paged attention) and batched multi-sequence decode are
// later phases; keeping this piece pure makes it unit-testable without a GPU.
namespace engine {

// Physical KV VRAM required for `num_blocks` blocks, fp16 K and V:
//   num_blocks * num_layers * 2 * block_size * kv_hidden * sizeof(half)
std::size_t paged_kv_pool_bytes(int num_blocks, int num_layers, int block_size, int kv_hidden);

// Largest block count whose pool fits in `budget_bytes`.
int paged_kv_blocks_for_budget(std::size_t budget_bytes, int num_layers, int block_size,
                               int kv_hidden);

// Fixed-size-block allocator with reference counting for shared blocks.
// Not thread-safe (the engine serializes access); single owner.
class BlockAllocator {
public:
  static constexpr int kInvalidBlock = -1;

  explicit BlockAllocator(int num_blocks);

  // Allocate one free block (refcount 1). Returns kInvalidBlock if the pool is
  // exhausted (caller triggers preemption/eviction).
  int allocate();

  // Increment a block's refcount (share it with another sequence).
  void add_ref(int block);

  // Decrement a block's refcount; returns it to the free-list at zero.
  void release(int block);

  int ref_count(int block) const;
  int free_count() const {
    return static_cast<int>(free_list_.size());
  }
  int num_blocks() const {
    return num_blocks_;
  }
  int used_count() const {
    return num_blocks_ - free_count();
  }

private:
  int num_blocks_;
  std::vector<int> free_list_;  // stack of free block ids
  std::vector<int> refcounts_;  // per-block refcount (0 = free)
};

// A single sequence's logical->physical block mapping. Grows as tokens are
// appended; `block_size` tokens live in each block.
class SequenceBlockTable {
public:
  SequenceBlockTable(BlockAllocator* alloc, int block_size);
  ~SequenceBlockTable();

  SequenceBlockTable(const SequenceBlockTable&) = delete;
  SequenceBlockTable& operator=(const SequenceBlockTable&) = delete;

  // Ensure the table can address token position `pos` (0-based), allocating
  // blocks as needed. Returns false if the pool is exhausted.
  bool ensure_position(int pos);

  // Physical block id and in-block offset for a token position. `pos` must have
  // been covered by ensure_position (or a shared prefix).
  int block_for(int pos) const;
  int offset_for(int pos) const {
    return pos % block_size_;
  }

  // Adopt another table's blocks as a shared read-only prefix of `prefix_len`
  // tokens (whole blocks only), bumping their refcounts. Used for shared-prompt
  // reuse across concurrent sequences.
  bool share_prefix_from(const SequenceBlockTable& other, int prefix_len);

  // Release all owned blocks (called on sequence end / destructor).
  void clear();

  int length_tokens() const {
    return length_tokens_;
  }
  const std::vector<int>& blocks() const {
    return blocks_;
  }

private:
  BlockAllocator* alloc_;
  int block_size_;
  int length_tokens_ = 0;  // highest covered position + 1
  int shared_prefix_blocks_ =
      0;                     // leading blocks that are shared (refcounted), not owned-exclusive
  std::vector<int> blocks_;  // logical block index -> physical block id
};

}  // namespace engine
