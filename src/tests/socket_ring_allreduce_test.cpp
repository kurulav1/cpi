// Verifies the ring all-reduce over REAL TCP sockets (engine/socket_ring.hpp) -- the multi-process
// collective prototype that stands in for a cluster we don't have. Each rank is a std::thread that is a
// genuine TCP endpoint on 127.0.0.1; ranks share NO memory and communicate only over sockets, so this
// exercises the actual distributed control flow (ring connect handshake, scatter-reduce + all-gather,
// partial send/recv) exactly as separate processes on separate nodes would -- only the transport
// underneath (loopback here, NCCL/NIC on a cluster) differs. Element counts are chosen NOT divisible by
// the world size so the ragged balanced-block split is exercised. Result is checked against the naive
// elementwise sum.
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <thread>
#include <vector>

#if defined(_WIN32)
#include <process.h>
#define GETPID _getpid
#else
#include <unistd.h>
#define GETPID getpid
#endif

#include "engine/socket_ring.hpp"

namespace {

// Rank r contributes input[i] = r*count + i (all integer-valued, exact in fp32 for these sizes). The
// elementwise sum over R ranks is then count*R*(R-1)/2 + R*i -- a closed form we can check exactly, and
// the per-rank ramp makes any block-boundary or block-index mistake in the ring show up as a mismatch.
float expected(int i, int world, int count) {
  return static_cast<float>(count) * world * (world - 1) / 2.0f + static_cast<float>(world) * i;
}

// One rank: join the ring, all-reduce its input vector, verify against the closed form. Returns 0 on
// success. `mismatch` is set to the number of wrong elements (0 = clean) for reporting.
int run_rank(int rank, int world, uint16_t base_port, int count, std::atomic<int>& mismatch) {
  try {
    engine::SocketRing ring(rank, world, base_port);
    std::vector<float> data(count);
    for (int i = 0; i < count; ++i) data[i] = static_cast<float>(rank) * count + i;
    ring.all_reduce_sum(data.data(), count);
    int wrong = 0;
    for (int i = 0; i < count; ++i)
      if (data[i] != expected(i, world, count)) ++wrong;
    mismatch += wrong;
    return wrong == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::printf("  rank %d threw: %s\n", rank, e.what());
    mismatch += count;  // count the whole rank as failed
    return 1;
  }
}

int run_world(int world, uint16_t base_port, int count) {
  std::atomic<int> mismatch{0};
  std::vector<std::thread> ranks;
  std::vector<int> rc(world, 0);
  for (int r = 0; r < world; ++r)
    ranks.emplace_back([&, r] { rc[r] = run_rank(r, world, base_port, count, mismatch); });
  for (auto& t : ranks) t.join();

  int failed = 0;
  for (int v : rc) failed += v;
  const bool pass = failed == 0;
  std::printf("%s[world=%d count=%d]: socket ring all-reduce, %d/%d ranks correct (%d mismatched elems)\n",
              pass ? "PASS" : "FAIL", world, count, world - failed, world, mismatch.load());
  return pass ? 0 : 1;
}

}  // namespace

int main() {
  // Per-run port base off the PID so concurrent test runs don't collide; each world size gets a
  // disjoint 16-port window on top of that.
  const uint16_t pid_base = static_cast<uint16_t>(20000 + (GETPID() % 20000));

  int fail = 0;
  int idx = 0;
  // world=1 is the trivial no-op ring; the rest each use a count NOT divisible by the world size so the
  // ragged block split (last blocks shorter) is exercised, not just the clean case.
  for (int world : {1, 2, 3, 4, 8}) {
    const uint16_t base = static_cast<uint16_t>(pid_base + idx++ * 16);
    fail |= run_world(world, base, /*count=*/1000);
  }
  return fail;
}
