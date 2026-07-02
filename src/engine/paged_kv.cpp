#include "engine/paged_kv.hpp"

#include <cstddef>

namespace engine {

std::size_t paged_kv_pool_bytes(int num_blocks, int num_layers, int block_size,
                                int kv_hidden) {
  // K and V, fp16, per block per layer.
  return static_cast<std::size_t>(num_blocks) * static_cast<std::size_t>(num_layers) *
         2ull * static_cast<std::size_t>(block_size) *
         static_cast<std::size_t>(kv_hidden) * sizeof(unsigned short);
}

int paged_kv_blocks_for_budget(std::size_t budget_bytes, int num_layers,
                               int block_size, int kv_hidden) {
  const std::size_t per_block =
      paged_kv_pool_bytes(1, num_layers, block_size, kv_hidden);
  if (per_block == 0) return 0;
  const std::size_t n = budget_bytes / per_block;
  return static_cast<int>(n);
}

BlockAllocator::BlockAllocator(int num_blocks)
    : num_blocks_(num_blocks < 0 ? 0 : num_blocks),
      refcounts_(static_cast<std::size_t>(num_blocks_ < 0 ? 0 : num_blocks_), 0) {
  free_list_.reserve(static_cast<std::size_t>(num_blocks_));
  // Hand out low ids first: push high->low so pop() returns 0,1,2,...
  for (int i = num_blocks_ - 1; i >= 0; --i) {
    free_list_.push_back(i);
  }
}

int BlockAllocator::allocate() {
  if (free_list_.empty()) return kInvalidBlock;
  const int block = free_list_.back();
  free_list_.pop_back();
  refcounts_[static_cast<std::size_t>(block)] = 1;
  return block;
}

void BlockAllocator::add_ref(int block) {
  if (block < 0 || block >= num_blocks_) return;
  ++refcounts_[static_cast<std::size_t>(block)];
}

void BlockAllocator::release(int block) {
  if (block < 0 || block >= num_blocks_) return;
  int& rc = refcounts_[static_cast<std::size_t>(block)];
  if (rc <= 0) return;  // already free; ignore double-free
  if (--rc == 0) {
    free_list_.push_back(block);
  }
}

int BlockAllocator::ref_count(int block) const {
  if (block < 0 || block >= num_blocks_) return 0;
  return refcounts_[static_cast<std::size_t>(block)];
}

SequenceBlockTable::SequenceBlockTable(BlockAllocator* alloc, int block_size)
    : alloc_(alloc), block_size_(block_size < 1 ? 1 : block_size) {}

SequenceBlockTable::~SequenceBlockTable() { clear(); }

bool SequenceBlockTable::ensure_position(int pos) {
  if (pos < 0) return true;
  const int needed_blocks = pos / block_size_ + 1;
  while (static_cast<int>(blocks_.size()) < needed_blocks) {
    const int b = alloc_->allocate();
    if (b == BlockAllocator::kInvalidBlock) {
      return false;  // pool exhausted; caller preempts/evicts
    }
    blocks_.push_back(b);
  }
  if (pos + 1 > length_tokens_) {
    length_tokens_ = pos + 1;
  }
  return true;
}

int SequenceBlockTable::block_for(int pos) const {
  const int idx = pos / block_size_;
  if (idx < 0 || idx >= static_cast<int>(blocks_.size())) {
    return BlockAllocator::kInvalidBlock;
  }
  return blocks_[static_cast<std::size_t>(idx)];
}

bool SequenceBlockTable::share_prefix_from(const SequenceBlockTable& other,
                                           int prefix_len) {
  // Only whole shared blocks are adopted; a partial trailing block of the prefix
  // stays per-sequence (copy-on-write territory, handled by the caller writing
  // its own tail). Must be called before this table owns any blocks.
  if (!blocks_.empty() || prefix_len <= 0) return false;
  const int whole_blocks = prefix_len / block_size_;
  if (whole_blocks > static_cast<int>(other.blocks_.size())) return false;
  for (int i = 0; i < whole_blocks; ++i) {
    const int b = other.blocks_[static_cast<std::size_t>(i)];
    alloc_->add_ref(b);
    blocks_.push_back(b);
  }
  shared_prefix_blocks_ = whole_blocks;
  length_tokens_ = whole_blocks * block_size_;
  return true;
}

void SequenceBlockTable::clear() {
  // Release every block we hold a reference to (shared prefix blocks were
  // add_ref'd, exclusive blocks were allocate'd — both are released here).
  for (int b : blocks_) {
    alloc_->release(b);
  }
  blocks_.clear();
  length_tokens_ = 0;
  shared_prefix_blocks_ = 0;
}

}  // namespace engine
