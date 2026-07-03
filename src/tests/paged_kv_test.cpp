// Unit test for the paged-KV allocator + block table (P3 Phase 1 foundation).
// Pure host logic, no CUDA. Verifies allocation, refcounted sharing, block-table
// growth, shared-prefix adoption, and leak-freedom (all blocks return to free).
#include "engine/paged_kv.hpp"

#include <cstdio>

using namespace engine;

static int failures = 0;
static void check(bool cond, const char* msg) {
  std::printf("%s  %s\n", cond ? "ok  " : "FAIL", msg);
  if (!cond) ++failures;
}

int main() {
  // Pool sizing math.
  {
    const std::size_t one = paged_kv_pool_bytes(1, 28, 16, 512);
    check(one == static_cast<std::size_t>(28) * 2 * 16 * 512 * 2, "pool bytes per block");
    check(paged_kv_blocks_for_budget(one * 10 + 5, 28, 16, 512) == 10, "blocks_for_budget floors");
  }

  // Allocator: low ids first, exhaustion, free-list balance.
  {
    BlockAllocator a(4);
    check(a.free_count() == 4, "starts with all free");
    int b0 = a.allocate(), b1 = a.allocate(), b2 = a.allocate(), b3 = a.allocate();
    check(b0 == 0 && b1 == 1 && b2 == 2 && b3 == 3, "hands out low ids first");
    check(a.allocate() == BlockAllocator::kInvalidBlock, "exhaustion returns invalid");
    check(a.used_count() == 4, "used=4");
    a.release(b1);
    check(a.free_count() == 1, "release frees one");
    check(a.allocate() == 1, "reallocates the freed id");
  }

  // Refcounted sharing: a block freed only when the last ref drops.
  {
    BlockAllocator a(2);
    int b = a.allocate();  // rc 1
    a.add_ref(b);          // rc 2
    a.release(b);          // rc 1 -> still held
    check(a.free_count() == 1, "shared block not freed while referenced");
    check(a.ref_count(b) == 1, "refcount tracks shares");
    a.release(b);  // rc 0 -> free
    check(a.free_count() == 2, "freed when last ref drops");
    a.release(b);  // double-free ignored
    check(a.free_count() == 2, "double-free is a no-op");
  }

  // Block table growth: block_size=4.
  {
    BlockAllocator a(8);
    {
      SequenceBlockTable t(&a, 4);
      check(t.ensure_position(0), "pos 0 ok");
      check(t.block_for(0) == 0 && t.offset_for(0) == 0, "pos0 -> block0 off0");
      check(t.block_for(3) == 0 && t.offset_for(3) == 3, "pos3 -> block0 off3");
      check(t.ensure_position(4), "pos 4 ok (new block)");
      check(t.block_for(4) == 1 && t.offset_for(4) == 0, "pos4 -> block1 off0");
      check(a.used_count() == 2, "two blocks used for 5 tokens @bs4");
      check(t.length_tokens() == 5, "length tracks highest pos");
    }  // table destructor releases
    check(a.free_count() == 8, "block table frees all blocks on destruct (no leak)");
  }

  // Shared prefix: second sequence adopts whole prefix blocks (refcounted).
  {
    BlockAllocator a(8);
    {
      SequenceBlockTable sys(&a, 4);
      for (int p = 0; p < 8; ++p) sys.ensure_position(p);  // 2 whole blocks
      check(a.used_count() == 2, "prefix uses 2 blocks");

      SequenceBlockTable seq(&a, 4);
      check(seq.share_prefix_from(sys, 8), "adopt 8-token prefix (2 blocks)");
      check(a.used_count() == 2, "sharing adds NO new blocks");
      check(a.ref_count(0) == 2 && a.ref_count(1) == 2, "shared blocks refcount=2");
      // seq appends its own suffix beyond the shared prefix.
      check(seq.ensure_position(8), "seq appends suffix block");
      check(a.used_count() == 3, "suffix adds one block");
    }  // both tables destruct
    check(a.free_count() == 8, "shared prefix + suffix fully released (no leak)");
  }

  std::printf("\npaged_kv_test: %s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
  return failures == 0 ? 0 : 1;
}
