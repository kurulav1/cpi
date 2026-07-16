// Batched paged decode vs the single-sequence path, on the same weights.
//
// The paged pool and the contiguous cache coincide when a sequence's block table is the
// IDENTITY map: phys = block_table[pos/bs]*bs + pos%bs collapses to pos when block i maps to
// block i. That is what lets this compare the two paths directly -- same engine, same pool,
// same KV bytes -- so any disagreement is the batched kernels' own doing rather than a
// different model, a different cache, or a different prefill.
//
// Two cases, mirroring the CUDA gate (run_batched_decode_check):
//   N=1  -- gates the paged gather, the per-row rope position and the LM-head row offset.
//   N=2, duplicate rows -- gates the batch dimension itself. Two identical rows must produce
//        identical logits, and both must equal the single-sequence answer. A row-indexing
//        bug (a kernel reading row 0's query for row 1, or writing both rows to the same
//        logit slice) survives N=1 and dies here.
//
// This does NOT yet cover a ragged batch of genuinely different sequences: that needs paged
// PREFILL to lay down each sequence's history in its own blocks, which is the next piece.
// The gate is honest about that rather than implying coverage it does not have.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "engine/plan_metal_engine.hpp"

namespace {

// Largest |a - b| over the vocab, plus whether the argmax agrees -- the two numbers that
// actually matter: tiny logit noise is survivable, a different argmax is a different token.
struct Diff {
  float max_abs = 0.0f;
  int argmax_a = -1;
  int argmax_b = -1;
};

Diff compare(const std::vector<float>& a, const std::vector<float>& b) {
  Diff d;
  float best_a = -1e30f, best_b = -1e30f;
  const std::size_t n = a.size() < b.size() ? a.size() : b.size();
  for (std::size_t i = 0; i < n; ++i) {
    d.max_abs = std::max(d.max_abs, std::fabs(a[i] - b[i]));
    if (a[i] > best_a) {
      best_a = a[i];
      d.argmax_a = static_cast<int>(i);
    }
    if (b[i] > best_b) {
      best_b = b[i];
      d.argmax_b = static_cast<int>(i);
    }
  }
  return d;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::printf("[metal_batched] SKIP: no model given (usage: metal_batched_test <model.ll2c>)\n");
    return 0;
  }
  const std::string model = argv[1];
  const int block_size = (argc > 2) ? std::atoi(argv[2]) : 32;
  const int max_ctx = 256;

  engine::PlanMetalEngine eng;
  if (!eng.available()) {
    std::printf("[metal_batched] SKIP: no Metal GPU (%s)\n", eng.last_error().c_str());
    return 0;
  }

  // Enough blocks to cover the context. set_paged_kv must precede open(), which is what
  // sizes the pool.
  const int blocks = (max_ctx + block_size - 1) / block_size;
  eng.set_paged_kv(blocks, block_size);
  eng.open(model, max_ctx);
  std::printf("[metal_batched] %s | pool %d blocks x %d tokens\n", eng.device_name().c_str(),
              blocks, block_size);

  // A short deterministic prompt. The content is irrelevant -- this compares two code paths
  // on identical state, it does not judge the model's output.
  std::vector<int> prompt;
  for (int i = 0; i < 24; ++i) prompt.push_back(100 + i);

  // Lay down the history one token at a time. With an identity block table these contiguous
  // writes ARE the paged layout, which is the whole trick.
  for (std::size_t i = 0; i + 1 < prompt.size(); ++i) {
    eng.forward_token(prompt[i], static_cast<int>(i));
  }

  const int last_tok = prompt.back();
  const int last_pos = static_cast<int>(prompt.size()) - 1;

  // Reference: the single-sequence path.
  const std::vector<float> ref = eng.forward_token(last_tok, last_pos);

  // The identity block table: logical block i -> physical block i.
  const int max_blocks = blocks;
  std::vector<int> identity(static_cast<std::size_t>(max_blocks));
  for (int i = 0; i < max_blocks; ++i) identity[static_cast<std::size_t>(i)] = i;

  int failures = 0;

  // ---- N = 1 -------------------------------------------------------------
  // Re-running the same token at the same position rewrites the same KV slot with the same
  // bytes, so the reference stays valid.
  {
    std::vector<std::vector<float>> out;
    eng.decode_step_batched_logits({last_tok}, {last_pos}, identity, max_blocks, out);
    if (out.size() != 1) {
      std::printf("  FAIL  N=1: expected 1 row, got %zu\n", out.size());
      ++failures;
    } else {
      const Diff d = compare(ref, out[0]);
      const bool ok = d.argmax_a == d.argmax_b && d.max_abs < 1e-2f;
      std::printf("  %s  N=1: max|d|=%.3g argmax %d vs %d\n", ok ? "PASS" : "FAIL", d.max_abs,
                  d.argmax_a, d.argmax_b);
      if (!ok) ++failures;
    }
  }

  // ---- N = 2, duplicate rows ---------------------------------------------
  // Both rows share the identity table (legitimately: shared blocks are what refcounting is
  // for) and sit at the same position, so both must equal the reference AND each other.
  {
    std::vector<int> bt2;
    bt2.insert(bt2.end(), identity.begin(), identity.end());
    bt2.insert(bt2.end(), identity.begin(), identity.end());

    std::vector<std::vector<float>> out;
    eng.decode_step_batched_logits({last_tok, last_tok}, {last_pos, last_pos}, bt2, max_blocks,
                                   out);
    if (out.size() != 2) {
      std::printf("  FAIL  N=2: expected 2 rows, got %zu\n", out.size());
      ++failures;
    } else {
      const Diff d0 = compare(ref, out[0]);
      const Diff d1 = compare(ref, out[1]);
      const Diff dr = compare(out[0], out[1]);
      const bool ok0 = d0.argmax_a == d0.argmax_b && d0.max_abs < 1e-2f;
      const bool ok1 = d1.argmax_a == d1.argmax_b && d1.max_abs < 1e-2f;
      const bool okr = dr.max_abs == 0.0f;  // identical inputs, identical kernel: exact
      std::printf("  %s  N=2 row0 vs single: max|d|=%.3g argmax %d vs %d\n",
                  ok0 ? "PASS" : "FAIL", d0.max_abs, d0.argmax_a, d0.argmax_b);
      std::printf("  %s  N=2 row1 vs single: max|d|=%.3g argmax %d vs %d\n",
                  ok1 ? "PASS" : "FAIL", d1.max_abs, d1.argmax_a, d1.argmax_b);
      std::printf("  %s  N=2 row0 vs row1 (must be exact): max|d|=%.3g\n", okr ? "PASS" : "FAIL",
                  dr.max_abs);
      if (!ok0 || !ok1 || !okr) ++failures;
    }
  }

  // ---- Negative control: the block table must actually be CONSULTED ------
  //
  // Everything above uses the identity table, where phys == pos. That makes the two paths
  // comparable, but it also means a kernel that ignored the block table completely and
  // indexed by raw position would pass every check so far. So: point the table at blocks the
  // sequence never wrote. The answer MUST change. If it does not, the gather is a fiction and
  // the identity results were a coincidence.
  {
    std::vector<int> wrong(static_cast<std::size_t>(max_blocks));
    for (int i = 0; i < max_blocks; ++i) {
      wrong[static_cast<std::size_t>(i)] = (i + max_blocks / 2) % max_blocks;
    }
    std::vector<std::vector<float>> out;
    eng.decode_step_batched_logits({last_tok}, {last_pos}, wrong, max_blocks, out);
    bool differs = true;
    if (out.size() == 1) {
      const Diff d = compare(ref, out[0]);
      // NaNs are fine here: reading never-written pool bytes is exactly what this asks for.
      differs = !(d.max_abs < 1e-2f) || d.argmax_a != d.argmax_b;
      std::printf("  %s  wrong table changes the answer: max|d|=%.3g argmax %d vs %d\n",
                  differs ? "PASS" : "FAIL", d.max_abs, d.argmax_a, d.argmax_b);
    } else {
      std::printf("  FAIL  negative control: expected 1 row, got %zu\n", out.size());
    }
    if (!differs) {
      std::printf("        (the block table is being IGNORED -- the gather indexes by raw\n");
      std::printf("         position, so the identity-table passes above proved nothing)\n");
      ++failures;
    }
  }

  if (failures != 0) {
    std::printf("\n[metal_batched] FAIL\n");
    return 1;
  }
  std::printf("\n[metal_batched] PASS\n");
  return 0;
}
