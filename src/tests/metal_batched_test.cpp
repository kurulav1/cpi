// Batched paged decode vs the single-sequence path, on the same weights.
//
// The paged pool and the contiguous cache coincide when a sequence's block table is the
// IDENTITY map: phys = block_table[pos/bs]*bs + pos%bs collapses to pos when block i maps to
// block i. That is what lets this compare the two paths directly -- same engine, same pool,
// same KV bytes -- so any disagreement is the batched kernels' own doing rather than a
// different model, a different cache, or a different prefill.
//
// The cases, in the order they narrow things down:
//   N=1  -- the paged gather, the per-row rope position, the LM-head row offset.
//   N=2, duplicate rows -- the batch dimension itself. A row-indexing bug (reading row 0's
//        query for row 1, or writing both rows to one logit slice) survives N=1 and dies here.
//   wrong block table -- a NEGATIVE control. Everything above uses the identity table, where
//        phys == pos, so a kernel ignoring the table entirely would pass. The answer must change.
//   chunk sweep -- paged prefill at several row counts. This is what caught the fp16 GEMM
//        being dispatched with half the threads it needs: it only misbehaves at
//        T >= kGemmMinTokens, and no golden prompt is that long.
//   ragged batch -- two DIFFERENT sequences, different lengths, different positions, disjoint
//        blocks, stepping together. The case continuous batching exists for.
//
// A note on tolerances: comparisons that cross kernels (a GEMM path vs the token-at-a-time
// GEMV) are held to kCrossKernelTol, because the matrix units accumulate in a different
// order. Comparisons between two runs of the SAME kernel are held to EXACT equality -- that
// is where batching contamination would show, and there is no excuse for a difference.

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

// Comparing a GEMM-based path against a GEMV-based one cannot be bit-exact: the matrix
// units accumulate in a different order. 0.05 is the tolerance metal_smoke holds both
// kernels to against a CPU reference. Comparisons between two runs of the SAME kernel stay
// exact (== 0) -- those are checked separately below.
constexpr float kCrossKernelTol = 5e-2f;

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
  // Deliberately small: the prompts below are ~24 tokens, and at the serving default of 32
  // they would each fit in ONE block, so a "shuffled" table would never cross a block
  // boundary and the gather would go untested. At 8 a 24-token history spans three blocks.
  const int block_size = (argc > 2) ? std::atoi(argv[2]) : 8;
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

  // Cross-check the CONTIGUOUS chunked prefill against the token-at-a-time reference.
  // inspect_next_logits() prefills the whole prompt in one chunk, so at 24 tokens it takes
  // the same cpi_gemm_f16 path that the batched prefill breaks on. If THIS disagrees too,
  // the fault is pre-existing and has nothing to do with batching.
  {
    int ref_argmax = 0;
    for (std::size_t i = 1; i < ref.size(); ++i) {
      if (ref[i] > ref[ref_argmax]) ref_argmax = static_cast<int>(i);
    }
    const auto top = eng.inspect_next_logits(prompt, 1);
    const int got = top.empty() ? -1 : top[0].first;
    std::printf("  [xcheck] contiguous CHUNK prefill (T=%zu, uses the GEMM) top-1 = %d ; "
                "token-at-a-time reference argmax = %d  -> %s\n",
                prompt.size(), got, ref_argmax, got == ref_argmax ? "AGREE" : "DISAGREE");
  }

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
      const bool ok = d.argmax_a == d.argmax_b && d.max_abs < kCrossKernelTol;
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
      const bool ok0 = d0.argmax_a == d0.argmax_b && d0.max_abs < kCrossKernelTol;
      const bool ok1 = d1.argmax_a == d1.argmax_b && d1.max_abs < kCrossKernelTol;
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
      differs = !(d.max_abs < kCrossKernelTol) || d.argmax_a != d.argmax_b;
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

  // ---- Paged prefill into SHUFFLED blocks --------------------------------
  //
  // Everything above leaned on the identity table. This lays the same sequence down through
  // a shuffled table -- its history is scattered across the pool in an order that has nothing
  // to do with its positions -- and then decodes it. It must still reproduce the contiguous
  // reference exactly. This is the first check that paged PREFILL works at all, and the first
  // where phys != pos throughout.
  const std::vector<int> blocks_a = {5, 2, 9, 0};
  // Sweep the chunk length. T=1 per call is exact but T=23 in one pass is not, so something
  // branches on the row count. Each L is self-contained: lay the history down contiguously,
  // take the reference, then redo it as ONE paged chunk of L rows and compare.
  {
    std::vector<int> ident2(static_cast<std::size_t>(max_blocks));
    for (int i = 0; i < max_blocks; ++i) ident2[static_cast<std::size_t>(i)] = i;
    for (int L : {2, 4, 7, 8, 9, 16, 23}) {
      for (int i = 0; i < L; ++i) eng.forward_token(prompt[static_cast<std::size_t>(i)], i);
      const std::vector<float> rL = eng.forward_token(prompt[static_cast<std::size_t>(L)], L);
      std::vector<int> hist(prompt.begin(), prompt.begin() + L);
      eng.prefill_paged(hist, 0, ident2);
      std::vector<std::vector<float>> out;
      eng.decode_step_batched_logits({prompt[static_cast<std::size_t>(L)]}, {L}, ident2,
                                     max_blocks, out);
      const Diff d = compare(rL, out[0]);
      const bool ok = d.argmax_a == d.argmax_b && d.max_abs < kCrossKernelTol;
      std::printf("  %s  paged prefill chunk T=%-2d : max|d|=%.3g argmax %d vs %d\n",
                  ok ? "PASS" : "FAIL", L, d.max_abs, d.argmax_a, d.argmax_b);
      if (!ok) ++failures;
    }
  }

  // ---- A genuinely ragged batch ------------------------------------------
  //
  // The case continuous batching exists for, and the one nothing above covers: two DIFFERENT
  // sequences, of different lengths, at different positions, in disjoint blocks, stepping
  // together. Each must get exactly the answer it gets alone -- if a kernel leaks one row's
  // query, length or block table into another, this is where it shows.
  {
    std::vector<int> prompt_b;
    for (int i = 0; i < 12; ++i) prompt_b.push_back(4000 + i * 7);
    const std::vector<int> blocks_b = {7, 3, 11, 1};  // disjoint from blocks_a
    const int last_b = prompt_b.back();
    const int last_pos_b = static_cast<int>(prompt_b.size()) - 1;

    auto prefill_both = [&]() {
      eng.prefill_paged(std::vector<int>(prompt.begin(), prompt.end() - 1), 0, blocks_a);
      eng.prefill_paged(std::vector<int>(prompt_b.begin(), prompt_b.end() - 1), 0, blocks_b);
    };

    // Each alone, through the same paged path (so this isolates BATCHING, not paging).
    prefill_both();
    std::vector<std::vector<float>> solo_a, solo_b;
    eng.decode_step_batched_logits({last_tok}, {last_pos}, blocks_a, 4, solo_a);
    eng.decode_step_batched_logits({last_b}, {last_pos_b}, blocks_b, 4, solo_b);

    // Both together: one ragged batch, lengths 24 and 12.
    prefill_both();
    std::vector<int> bt_ragged;
    bt_ragged.insert(bt_ragged.end(), blocks_a.begin(), blocks_a.end());
    bt_ragged.insert(bt_ragged.end(), blocks_b.begin(), blocks_b.end());
    std::vector<std::vector<float>> both;
    eng.decode_step_batched_logits({last_tok, last_b}, {last_pos, last_pos_b}, bt_ragged, 4, both);

    if (both.size() != 2 || solo_a.size() != 1 || solo_b.size() != 1) {
      std::printf("  FAIL  ragged batch: unexpected row counts\n");
      ++failures;
    } else {
      const Diff da = compare(solo_a[0], both[0]);
      const Diff db = compare(solo_b[0], both[1]);
      // Same kernel, same inputs, same order: batching must not perturb a row at all.
      const bool oka = da.max_abs == 0.0f;
      const bool okb = db.max_abs == 0.0f;
      std::printf("  %s  ragged: seq A (len %d) batched == alone: max|d|=%.3g\n",
                  oka ? "PASS" : "FAIL", static_cast<int>(prompt.size()), da.max_abs);
      std::printf("  %s  ragged: seq B (len %d) batched == alone: max|d|=%.3g\n",
                  okb ? "PASS" : "FAIL", static_cast<int>(prompt_b.size()), db.max_abs);
      // Two different sequences must not agree -- if they do, one row is reading the other's
      // state and every "match" above is vacuous.
      const Diff dab = compare(both[0], both[1]);
      const bool distinct = dab.max_abs > 0.0f;
      std::printf("  %s  ragged: the two rows are actually different (max|d|=%.3g)\n",
                  distinct ? "PASS" : "FAIL", dab.max_abs);
      if (!oka || !okb || !distinct) ++failures;
    }
  }

  if (failures != 0) {
    std::printf("\n[metal_batched] FAIL\n");
    return 1;
  }
  std::printf("\n[metal_batched] PASS\n");
  return 0;
}
