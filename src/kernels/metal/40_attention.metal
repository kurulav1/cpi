#define Q_BLOCK 8
// The matrix-unit prefill kernel's query-block size. Separate from Q_BLOCK because the scalar
// kernel sizes its shared arrays for head_dim 256 (Gemma) and cannot afford a wider block --
// raising a shared constant would silently break a path whose golden is skipped on any host
// without a Gemma checkpoint.
//
// The block is QMM_BLOCK/8 query row fragments, and 16 (two fragments) is what fills the scoring
// loop: it runs qfs*n_kg work items, and with n_kg = KEY_BLOCK/8 = 4 key groups, one query
// fragment leaves 4 of the 8 simdgroups idle where two fill them all. Measured on a 2041-token
// prefill -- where attention is 39% of the pass, not the 15% it is at 541 -- attention goes
// 330 -> 301 ms (-9%) from 8 to 16, and back to 330 at 32 (four fragments loop twice for no more
// parallelism and cost more threadgroup memory). At 541 tokens the change is real but invisible:
// 9% of a 15% share is under the run-to-run noise, which is why it first read as a dead end when
// measured only there. Measure a small term where it is small and you conclude it is zero.
//
// It does NOT help by cutting K/V re-reads, the theory it was first tried under: those are
// cache-served (a layer's K+V is ~277 KB at T=541, held in the LLC without trying), and 32 would
// cut them further while running slower. It is only about keeping all 8 simdgroups busy in the
// score matmul.
//
// Raising it USED to compute garbage -- the fragment plan scored exactly one simdgroup_float8x8
// from row 0, so a block of 16 wrote its first 8 queries and left the rest as whatever shared
// memory held (the fp16 GEMM's bug exactly, a tile widened past the threads that serve it). The
// matrix ops below now loop over QMM_BLOCK/8 fragments, so it is a real knob.
//
// Threadgroup memory is the ceiling: q_sh + acc + sc_sh + pw_sh is QMM_BLOCK * (2*128 + 4*128 +
// 4*32 + 2*32) bytes, so 16 costs ~15 KB of the 32 KB budget and 32 ~30 KB.
#define QMM_BLOCK 16

// RCA instrumentation for the prefill attention kernel. 0 = normal. Each bit replaces one phase
// with a dependency-preserving stub -- the answer becomes wrong, but every other phase still runs
// and nothing downstream gets dead-code-eliminated, so the wall-clock delta is that phase's cost.
// This is the only way to see inside a kernel whose total is all we can otherwise measure.
//   1 = scoring matmul stubbed (sc_sh still written, so the softmax is still fed)
//   2 = softmax stubbed (pw_sh still written, so P.V is still fed)
//   4 = P.V matmul stubbed (acc untouched; the output loop still reads it)
// Bits combine: 7 stubs all three at once, which prices the residue no phase accounts for -- the
// K/V loads, the barriers, and the output loop. Read the singles against that, not against 0: the
// phases overlap across concurrent threadgroups, so their measured costs are superadditive.
#define ATTN_RCA 0

// ---------------------------------------------------------------------------
// Prefill attention on the MATRIX UNITS (head_dim <= 128). The scalar kernel below scores
// QK^T and does P.V with per-thread dot products -- the matrix units sit idle through the
// whole of attention, which the RCA put at ~21% of prefill. This computes both products with
// simdgroup_matrix (fp32 accumulate, the fast path), keeping the same per-query online softmax.
//
// One threadgroup per (query block, head); 8 simdgroups. The block is QMM_BLOCK queries, which is
// QMM_BLOCK/8 row fragments -- every matrix op below loops over them, so the constant can move
// without the kernel silently computing only its first 8 rows. For a KEY_BLOCK of 32 keys:
//   scoring  S[8q x 8k]  = sum over head_dim/8 of  Q[8q x 8d] @ (K[8k x 8d])^T   (K loaded transposed)
//   P.V      O[8q x 8d] += sum over the key groups of  P[8q x 8k] @ V[8k x 8d]
// The online rescale between key blocks is applied to the fp32 accumulator in threadgroup
// memory (a per-row scalar, which a matrix fragment cannot scale), then the P.V matmul
// accumulates into it. Gemma's head_dim 256 does not fit the fragment plan and keeps the
// scalar kernel.
// ---------------------------------------------------------------------------
kernel void cpi_attention_prefill_mm(device const half* q [[buffer(0)]],
                                     device const half* k_cache [[buffer(1)]],
                                     device const half* v_cache [[buffer(2)]],
                                     device half* out [[buffer(3)]],
                                     device const int* positions [[buffer(4)]],
                                     // ONE sequence's logical->physical block map, used only when
                                     // p.paged != 0. Bound to a dummy otherwise (the params block
                                     // must stay last, so the slot cannot simply be absent).
                                     device const int* block_tables [[buffer(5)]],
                                     constant AttnParams& p [[buffer(6)]],
                                     uint gid [[threadgroup_position_in_grid]],
                                     uint lid [[thread_position_in_threadgroup]],
                                     uint nthr [[threads_per_threadgroup]]) {
  const uint head = gid % p.heads;
  const uint t0 = (gid / p.heads) * QMM_BLOCK;
  if (t0 >= p.tokens) return;
  const uint nq = min((uint)QMM_BLOCK, p.tokens - t0);

  uint base = p.position;
  if (p.use_position_buffer != 0u) base = uint(positions[0]);

  const uint group = p.heads / p.kv_heads;
  const uint kv_head = head / group;
  const uint kv_dim = p.kv_heads * p.head_dim;
  const uint q_dim = p.heads * p.head_dim;
  const uint hd = p.head_dim;  // <= 128

  const uint simd_id = lid / 32u;
  const uint lane = lid % 32u;
  const uint n_simd = nthr / 32u;

  threadgroup half q_sh[QMM_BLOCK * 128];         // [query][d]
  threadgroup float acc[QMM_BLOCK * 128];         // [query][d] fp32 output accumulator
  threadgroup float sc_sh[QMM_BLOCK * MM_KEY_BLOCK]; // scores [query][key]
  threadgroup half pw_sh[QMM_BLOCK * MM_KEY_BLOCK];  // softmax weights [query][key], half for the matmul
  threadgroup float m_sh[QMM_BLOCK];              // running max per query
  threadgroup float l_sh[QMM_BLOCK];              // running sum per query

  // Zero the FULL QMM_BLOCK rows, not just nq: the matrix ops load 8-row fragments, so the
  // padding rows must be defined (0) rather than garbage that could turn into NaN.
  for (uint c = lid; c < QMM_BLOCK * hd; c += nthr) {
    const uint qi = c / hd, i = c % hd;
    q_sh[qi * hd + i] = (qi < nq) ? q[(ulong)(t0 + qi) * (ulong)q_dim + head * hd + i] : half(0);
    acc[qi * hd + i] = 0.0f;
  }
  for (uint qi = lid; qi < QMM_BLOCK; qi += nthr) {
    m_sh[qi] = -INFINITY;
    l_sh[qi] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint last_pos = base + t0 + nq - 1u;
  const uint first_pos = base + t0;
  uint start = 0u;
  if (p.window != 0u && first_pos + 1u > p.window) start = first_pos + 1u - p.window;

  const uint dfs = hd / 8u;              // head_dim fragments
  const uint qfs = (nq + 7u) / 8u;       // query fragments actually occupied (<= QMM_BLOCK/8)

  for (uint kb = start; kb <= last_pos; kb += MM_KEY_BLOCK) {
    const uint nk = min((uint)MM_KEY_BLOCK, last_pos - kb + 1u);
    const uint n_kg = (nk + 7u) / 8u;  // 8-key groups this block

    // Where this key block physically lives. ONE lookup per block, not per key: the caller
    // guarantees block_size % MM_KEY_BLOCK == 0 and no sliding window (so start == 0 and every kb
    // is MM_KEY_BLOCK-aligned), which together mean the whole run [kb, kb+MM_KEY_BLOCK) sits inside a
    // single physical block and stays contiguous -- so the simdgroup_loads below keep working
    // unchanged, just from a different base. Break either guarantee and a key block could
    // straddle two blocks, which this addressing cannot express; the dispatch enforces both.
    const uint kphys = (p.paged != 0u)
                           ? uint(block_tables[kb / p.block_size]) * p.block_size + (kb % p.block_size)
                           : kb;

    // --- Scoring: each simdgroup computes a QT x KT TILE of S[8q x 8k] fragments. ---
    // REGISTER TILING, the same trick that makes the GEMM fast. One output fragment per work item
    // means loading a Qf and a Kf for every single matrix op -- two loads per MAC, an arithmetic
    // intensity of 0.5, against the GEMM's 2.0. That is why attention runs at a few percent of the
    // GEMM's FLOP rate: it is starved on fragment loads, not short of matrix throughput. Holding a
    // 2x2 tile reuses each loaded Qf across KT key groups and each Kf across QT query fragments:
    // 4 loads feed 4 MACs (intensity 1.0), and with qfs=2 and n_kg up to 16 the tile count still
    // fills all 8 simdgroups.
    //
    // Query fragments past qfs are safe to compute: q_sh is zeroed over the FULL QMM_BLOCK rows, so
    // they contribute zeros that the softmax's nq bound ignores. Key groups past n_kg are NOT safe
    // to read (they can run off the end of the KV cache), so their load is clamped and their store
    // suppressed.
    constexpr uint QT = 2u, KT = 2u;
    const uint qt_n = (qfs + QT - 1u) / QT;
    const uint kt_n = (n_kg + KT - 1u) / KT;
#if !(ATTN_RCA & 1)
    for (uint w = simd_id; w < qt_n * kt_n; w += n_simd) {
      const uint qt = w / kt_n, kt = w % kt_n;
      simdgroup_float8x8 S[QT][KT];
      for (uint a = 0u; a < QT; ++a)
        for (uint b = 0u; b < KT; ++b) S[a][b] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

      for (uint df = 0u; df < dfs; ++df) {
        simdgroup_half8x8 Qf[QT], Kf[KT];
        for (uint a = 0u; a < QT; ++a)
          simdgroup_load(Qf[a], q_sh + (qt * QT + a) * 8u * hd + df * 8u, hd);  // [8q x 8d]
        for (uint b = 0u; b < KT; ++b) {
          const uint kg = min(kt * KT + b, n_kg - 1u);  // clamped: out-of-range reads are discarded
          // K is [key][d]; load the [8k x 8d] tile TRANSPOSED to get [8d x 8k].
          simdgroup_load(Kf[b],
                         k_cache + (ulong)(kphys + kg * 8u) * (ulong)kv_dim + kv_head * hd + df * 8u,
                         kv_dim, ulong2(0, 0), true);
        }
        for (uint a = 0u; a < QT; ++a)
          for (uint b = 0u; b < KT; ++b)
            simdgroup_multiply_accumulate(S[a][b], Qf[a], Kf[b], S[a][b]);
      }

      for (uint a = 0u; a < QT; ++a) {
        for (uint b = 0u; b < KT; ++b) {
          const uint kg = kt * KT + b;
          if (kg >= n_kg) continue;  // suppressed: this fragment's keys are past the block
          simdgroup_store(S[a][b], sc_sh + (qt * QT + a) * 8u * MM_KEY_BLOCK + kg * 8u,
                          MM_KEY_BLOCK);
        }
      }
    }
#else
    // RCA bit 0: scoring matmul stubbed. sc_sh still written so the softmax is still fed.
    (void)qt_n; (void)kt_n; (void)QT; (void)KT;
    for (uint c = lid; c < QMM_BLOCK * MM_KEY_BLOCK; c += nthr) sc_sh[c] = 0.0f;
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax, a simdgroup per query. The simd is 32 wide and MM_KEY_BLOCK is wider, so each
    // lane owns MM_KEY_BLOCK/32 keys and
    // folds their max and sum before the simd reduction. A wide key block halves the number of
    // iterations (and their barriers and reductions) versus a narrow one, which is the point:
    // attention's per-block overhead, not its matmuls, is what makes this kernel slower than the
    // flash-attention path it is measured against.
    //
    // Scale and the causal/window mask are folded in here rather than a separate sc_sh sweep: the
    // softmax already holds the query and its keys, so the masked, scaled score feeds the reduction
    // directly -- one fewer barrier and one fewer pass per block.
    //
    // Three things this loop does NOT cost, each measured and each reverted: a fast exp, skipping
    // the accumulator rescale when the running max did not move, and interleaving two queries so
    // their reduction chains overlap. All neutral. Taken with the phase decomposition (ATTN_RCA
    // above), which prices this loop above either matmul despite its doing a sixty-fourth of their
    // arithmetic, the cost is not in the body at all -- it is the score round-trip through
    // threadgroup memory that phase-partitioning forces, which no edit inside the phase can remove.
#if !(ATTN_RCA & 2)
    for (uint qi = simd_id; qi < nq; qi += n_simd) {
      const uint pos_i = base + t0 + qi;
      uint start_i = 0u;
      if (p.window != 0u && pos_i + 1u > p.window) start_i = pos_i + 1u - p.window;
      // MM_KEY_BLOCK/32 keys per lane. Fold their max and sum across the group before the simd
      // reduction; each is written back to its own pw_sh column.
      float lmax = -INFINITY;
      float sc[MM_KEY_BLOCK / 32u];
      for (uint g = 0u; g < MM_KEY_BLOCK / 32u; ++g) {
        const uint key = kb + lane + g * 32u;
        const bool keep = (lane + g * 32u < nk) && (key <= pos_i) && (key >= start_i);
        sc[g] = keep ? sc_sh[qi * MM_KEY_BLOCK + lane + g * 32u] * p.scale : -INFINITY;
        lmax = max(lmax, sc[g]);
      }
      const float old_max = m_sh[qi];
      const float new_max = max(old_max, simd_max(lmax));
      const float rescale = (old_max == -INFINITY) ? 0.0f : exp(old_max - new_max);
      float lsum = 0.0f;
      for (uint g = 0u; g < MM_KEY_BLOCK / 32u; ++g) {
        const float w = (sc[g] != -INFINITY && new_max != -INFINITY) ? exp(sc[g] - new_max) : 0.0f;
        pw_sh[qi * MM_KEY_BLOCK + lane + g * 32u] = half(w);  // 0 past nk, so P.V masks itself
        lsum += w;
      }
      const float bsum = simd_sum(lsum);
      // Rescale this query's running accumulator right here, by the simdgroup that owns the query
      // (rescale is uniform across the lanes, from the simd_max), lanes splitting head_dim.
      for (uint i = lane; i < hd; i += 32u) acc[qi * hd + i] = acc[qi * hd + i] * rescale;
      if (lane == 0u) {
        l_sh[qi] = l_sh[qi] * rescale + bsum;
        m_sh[qi] = new_max;
      }
    }
#else
    // RCA bit 1: softmax stubbed. pw_sh still written so P.V is still fed and not eliminated.
    for (uint qi = simd_id; qi < nq; qi += n_simd) {
      for (uint g = 0u; g < MM_KEY_BLOCK / 32u; ++g)
        pw_sh[qi * MM_KEY_BLOCK + lane + g * 32u] = half(1.0f);
      if (lane == 0u) l_sh[qi] = 1.0f;
    }
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- P.V: O[8q x 8d] += P[8q x 8k] @ V[8k x 8d], accumulated into the fp32 acc. ---
    // Tiled over QT query fragments for the same reason as the scoring above: one output fragment
    // per work item meant loading a Pf AND a Vf for every MAC (intensity 0.5). Holding QT of them
    // reuses each Vf across the query fragments -- 3 loads feed 2 MACs (0.67) -- while dt_n = dfs
    // keeps all 8 simdgroups busy. The accumulator load/store is hoisted out of the key loop, so it
    // amortises over all n_kg groups.
#if !(ATTN_RCA & 4)
    for (uint w = simd_id; w < qt_n * dfs; w += n_simd) {
      const uint qt = w / dfs, dg = w % dfs;
      simdgroup_float8x8 O[QT];
      for (uint a = 0u; a < QT; ++a)
        simdgroup_load(O[a], acc + (qt * QT + a) * 8u * hd + dg * 8u, hd);  // rescaled accumulator
      for (uint kg = 0u; kg < n_kg; ++kg) {
        simdgroup_half8x8 Pf[QT], Vf;
        for (uint a = 0u; a < QT; ++a)
          simdgroup_load(Pf[a], pw_sh + (qt * QT + a) * 8u * MM_KEY_BLOCK + kg * 8u,
                         MM_KEY_BLOCK);  // [8q x 8k]
        simdgroup_load(Vf, v_cache + (ulong)(kphys + kg * 8u) * (ulong)kv_dim + kv_head * hd + dg * 8u,
                       kv_dim);  // [8k x 8d], shared by every query fragment in the tile
        for (uint a = 0u; a < QT; ++a) simdgroup_multiply_accumulate(O[a], Pf[a], Vf, O[a]);
      }
      for (uint a = 0u; a < QT; ++a)
        simdgroup_store(O[a], acc + (qt * QT + a) * 8u * hd + dg * 8u, hd);
    }
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint c = lid; c < nq * hd; c += nthr) {
    const uint qi = c / hd, i = c % hd;
    const float inv = (l_sh[qi] > 0.0f) ? 1.0f / l_sh[qi] : 0.0f;
    out[(ulong)(t0 + qi) * (ulong)q_dim + head * hd + i] = half(acc[qi * hd + i] * inv);
  }
}

// ---------------------------------------------------------------------------
// QUERY-PARTITIONED prefill attention -- the research rewrite.
//
// The kernel above is PHASE-partitioned: all 8 simdgroups cooperate on all QMM_BLOCK queries and
// re-partition between scoring / softmax / P.V, so every phase boundary pushes the intermediate
// state through threadgroup memory behind a threadgroup_barrier. Measured, that structure costs
// exactly what llama.cpp's flash-attention-OFF path costs (116 vs 117 ms at T=2048) -- it is the
// unfused algorithm in a fused kernel's clothing, and no amount of tiling, block-size or occupancy
// tuning moved it (all measured neutral).
//
// This one is QUERY-partitioned, which is what actually makes flash attention fast: each simdgroup
// owns QP_QPS queries OUTRIGHT and carries them through scoring, softmax and accumulation itself.
// Nothing it writes is read by any other simdgroup, so every scratch array is a per-simdgroup slice
// and every barrier drops from threadgroup_barrier to simdgroup_barrier -- ordering within one
// simdgroup, which is nearly free. There is no cross-simdgroup synchronisation in the key loop at
// all, and a simdgroup with no queries can return early without deadlocking the others.
//
// 4 simdgroups x 8 queries = 32 queries per threadgroup. Scratch is 7.5 KB per simdgroup (q, sc,
// pw, acc sized for head_dim <= 128), 30 KB total, just inside the 32 KB budget -- which is why
// this uses 4 simdgroups and a 32-key block rather than the 8 and 128 the phase-partitioned kernel
// settled on. Selected by CPI_METAL_ATTN_QP=1; the proven kernel above stays the default.
//
// STATUS: CORRECT (goldens pass) but currently 3.9x SLOWER -- 449 ms of attention at T=2042 against
// the phase-partitioned kernel's 116 ms. The decomposition is not what is wrong; the BLOCK SIZE is.
// Giving every simdgroup private scratch multiplies the scratch by NSG, which forces QP_KB down to
// 32 where the phase kernel runs 128. The online softmax (its two simd reductions plus the
// accumulator rescale) runs once per (query, key block), so a 32-key block does 4x the softmax work
// of a 128-key one -- and that same effect is what made 32 -> 128 the single largest win on the
// phase kernel. Hoisting the query fragments out of the key-group loop (intensity 0.5 -> 0.8) was
// measured neutral, confirming the cost is the softmax count, not fragment loads.
//
// So making this competitive is a MEMORY problem, not a restructuring one: it needs the per-
// simdgroup scratch to fit a 128-key block, which means sizing the arrays to the actual head_dim
// instead of the 128 maximum (dynamic threadgroup memory via setThreadgroupMemoryLength) and
// re-tuning NSG against QP_KB. At head_dim 64 that alone would roughly halve the scratch. Left here
// behind the flag because it is correct and the remaining work is parameter tuning rather than a
// rewrite -- which is a much better starting point than the blank page this began as.
#define QP_NSG  4                     // simdgroups per threadgroup
#define QP_QPS  8                     // queries per simdgroup == one 8x8 row fragment
#define QP_QBLK (QP_QPS * QP_NSG)     // 32 queries per threadgroup
// Keys per block, == the simd width, so lane == key and there is nothing to fold before the
// reduction. That looked like this kernel's whole problem: the cap is the threadgroup budget
// (per-simdgroup scratch times QP_NSG, and with q_sh and acc sized for head_dim 128 the four
// simdgroups already spend ~30 KB of 32 KB here), and a narrow block leaves the softmax as almost
// pure reduction latency.
//
// It was tried. Sizing q_sh and acc at dispatch from the real head_dim lifts the cap, and wider is
// monotonically WORSE: 32 keys 743 ms, 64 764, 96 852, against 610 for the phase-partitioned
// kernel. Threadgroup memory buys occupancy, not block width -- it decides how many threadgroups
// stay resident per core, and at 96 keys only one fits. Latency hiding is worth more here than any
// fold. The dynamic-allocation plumbing was reverted with the hypothesis (it also let a caller that
// forgot to size the arrays get zero-length ones silently, which is how it broke the window golden).
// Query-partitioning is not being held back by this constant.
#define QP_KB   32
kernel void cpi_attention_prefill_qp(device const half* q [[buffer(0)]],
                                     device const half* k_cache [[buffer(1)]],
                                     device const half* v_cache [[buffer(2)]],
                                     device half* out [[buffer(3)]],
                                     device const int* positions [[buffer(4)]],
                                     device const int* block_tables [[buffer(5)]],
                                     constant AttnParams& p [[buffer(6)]],
                                     uint gid [[threadgroup_position_in_grid]],
                                     uint lid [[thread_position_in_threadgroup]],
                                     uint nthr [[threads_per_threadgroup]]) {
  const uint sg = lid / 32u;
  const uint lane = lid % 32u;

  const uint head = gid % p.heads;
  const uint t0 = (gid / p.heads) * QP_QBLK;
  if (t0 >= p.tokens) return;

  uint base = p.position;
  if (p.use_position_buffer != 0u) base = uint(positions[0]);

  const uint group = p.heads / p.kv_heads;
  const uint kv_head = head / group;
  const uint kv_dim = p.kv_heads * p.head_dim;
  const uint q_dim = p.heads * p.head_dim;
  const uint hd = p.head_dim;
  const uint dfs = hd / 8u;

  // This simdgroup's OWN queries. Nothing below is shared with another simdgroup.
  const uint q0 = t0 + sg * QP_QPS;
  const uint nq = (q0 < p.tokens) ? min((uint)QP_QPS, p.tokens - q0) : 0u;

  threadgroup half q_sh[QP_NSG][QP_QPS * 128];
  threadgroup float acc[QP_NSG][QP_QPS * 128];
  threadgroup float sc[QP_NSG][QP_QPS * QP_KB];
  threadgroup half pw[QP_NSG][QP_QPS * QP_KB];
  threadgroup float m_sh[QP_NSG][QP_QPS];
  threadgroup float l_sh[QP_NSG][QP_QPS];

  // Zero the FULL QP_QPS rows: the matrix ops load 8-row fragments, so padding rows must be
  // defined rather than garbage that could become NaN.
  for (uint c = lane; c < QP_QPS * hd; c += 32u) {
    const uint qi = c / hd, i = c % hd;
    q_sh[sg][qi * hd + i] =
        (qi < nq) ? q[(ulong)(q0 + qi) * (ulong)q_dim + head * hd + i] : half(0);
    acc[sg][qi * hd + i] = 0.0f;
  }
  for (uint qi = lane; qi < QP_QPS; qi += 32u) {
    m_sh[sg][qi] = -INFINITY;
    l_sh[sg][qi] = 0.0f;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);

  if (nq == 0u) return;  // safe: no threadgroup_barrier below, so this cannot deadlock the others

  const uint last_pos = base + q0 + nq - 1u;
  const uint first_pos = base + q0;
  uint start = 0u;
  if (p.window != 0u && first_pos + 1u > p.window) start = first_pos + 1u - p.window;

  for (uint kb = start; kb <= last_pos; kb += QP_KB) {
    const uint nk = min((uint)QP_KB, last_pos - kb + 1u);
    const uint n_kg = (nk + 7u) / 8u;
    const uint kphys =
        (p.paged != 0u)
            ? uint(block_tables[kb / p.block_size]) * p.block_size + (kb % p.block_size)
            : kb;

    // Scoring: this simdgroup's 8 queries against each key group. The query fragments are the SAME
    // for every key group, so they are loaded ONCE into registers and reused across all of them --
    // dfs + n_kg*dfs loads for n_kg*dfs MACs, rather than reloading Qf per key group (which would
    // be two loads per MAC, the 0.5 intensity that starved the phase-partitioned kernel).
    simdgroup_half8x8 Qf[16];  // dfs <= 128/8
    for (uint df = 0u; df < dfs; ++df) simdgroup_load(Qf[df], q_sh[sg] + df * 8u, hd);
    for (uint kg = 0u; kg < n_kg; ++kg) {
      simdgroup_float8x8 S = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
      for (uint df = 0u; df < dfs; ++df) {
        simdgroup_half8x8 Kf;
        simdgroup_load(Kf, k_cache + (ulong)(kphys + kg * 8u) * (ulong)kv_dim + kv_head * hd + df * 8u,
                       kv_dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(S, Qf[df], Kf, S);
      }
      simdgroup_store(S, sc[sg] + kg * 8u, QP_KB);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax over this simdgroup's own queries; lane owns key `lane` (QP_KB == 32).
    // Scale, causal/window mask and the accumulator rescale are all folded in.
    for (uint qi = 0u; qi < nq; ++qi) {
      const uint key = kb + lane;
      const uint pos_i = base + q0 + qi;
      uint start_i = 0u;
      if (p.window != 0u && pos_i + 1u > p.window) start_i = pos_i + 1u - p.window;
      const bool keep = (lane < nk) && (key <= pos_i) && (key >= start_i);
      const float s = keep ? sc[sg][qi * QP_KB + lane] * p.scale : -INFINITY;
      const float old_max = m_sh[sg][qi];
      const float new_max = max(old_max, simd_max(s));
      const float rescale = (old_max == -INFINITY) ? 0.0f : exp(old_max - new_max);
      const float w = (s != -INFINITY && new_max != -INFINITY) ? exp(s - new_max) : 0.0f;
      pw[sg][qi * QP_KB + lane] = half(w);
      const float bsum = simd_sum(w);
      for (uint i = lane; i < hd; i += 32u) acc[sg][qi * hd + i] = acc[sg][qi * hd + i] * rescale;
      if (lane == 0u) {
        l_sh[sg][qi] = l_sh[sg][qi] * rescale + bsum;
        m_sh[sg][qi] = new_max;
      }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);

    // P.V into this simdgroup's own accumulator.
    for (uint dg = 0u; dg < dfs; ++dg) {
      simdgroup_float8x8 O;
      simdgroup_load(O, acc[sg] + dg * 8u, hd);
      for (uint kg = 0u; kg < n_kg; ++kg) {
        simdgroup_half8x8 Pf, Vf;
        simdgroup_load(Pf, pw[sg] + kg * 8u, QP_KB);
        simdgroup_load(Vf, v_cache + (ulong)(kphys + kg * 8u) * (ulong)kv_dim + kv_head * hd + dg * 8u,
                       kv_dim);
        simdgroup_multiply_accumulate(O, Pf, Vf, O);
      }
      simdgroup_store(O, acc[sg] + dg * 8u, hd);
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint c = lane; c < nq * hd; c += 32u) {
    const uint qi = c / hd, i = c % hd;
    const float inv = (l_sh[sg][qi] > 0.0f) ? 1.0f / l_sh[sg][qi] : 0.0f;
    out[(ulong)(q0 + qi) * (ulong)q_dim + head * hd + i] = half(acc[sg][qi * hd + i] * inv);
  }
}

kernel void cpi_attention_prefill(device const half* q [[buffer(0)]],
                                  device const half* k_cache [[buffer(1)]],
                                  device const half* v_cache [[buffer(2)]],
                                  device half* out [[buffer(3)]],
                                  device const int* positions [[buffer(4)]],
                                  constant AttnParams& p [[buffer(5)]],
                                  uint gid [[threadgroup_position_in_grid]],
                                  uint lid [[thread_position_in_threadgroup]],
                                  uint nthr [[threads_per_threadgroup]]) {
  const uint head = gid % p.heads;
  const uint t0 = (gid / p.heads) * Q_BLOCK;
  if (t0 >= p.tokens) return;
  const uint nq = min((uint)Q_BLOCK, p.tokens - t0);

  uint base = p.position;
  if (p.use_position_buffer != 0u) base = uint(positions[0]);

  const uint group = p.heads / p.kv_heads;
  const uint kv_head = head / group;
  const uint kv_dim = p.kv_heads * p.head_dim;
  const uint q_dim = p.heads * p.head_dim;
  const uint hd = p.head_dim;

  const uint simd_id = lid / 32u;
  const uint lane = lid % 32u;

  threadgroup float q_sh[Q_BLOCK * 256];
  threadgroup float acc[Q_BLOCK * 256];
  threadgroup float sc_sh[Q_BLOCK * KEY_BLOCK];
  threadgroup float w_sh[Q_BLOCK * KEY_BLOCK];
  threadgroup float m_sh[Q_BLOCK];  // running max, per query
  threadgroup float l_sh[Q_BLOCK];  // running sum, per query
  threadgroup float r_sh[Q_BLOCK];  // this block's rescale, per query

  for (uint c = lid; c < nq * hd; c += nthr) {
    const uint qi = c / hd, i = c % hd;
    q_sh[qi * hd + i] = float(q[(ulong)(t0 + qi) * (ulong)q_dim + head * hd + i]);
    acc[qi * hd + i] = 0.0f;
  }
  for (uint qi = lid; qi < nq; qi += nthr) {
    m_sh[qi] = -INFINITY;
    l_sh[qi] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // The block's queries span positions [base+t0, base+t0+nq-1]; the last one reaches
  // furthest, and the first one's window (if any) starts earliest.
  const uint last_pos = base + t0 + nq - 1u;
  const uint first_pos = base + t0;
  uint start = 0u;
  if (p.window != 0u && first_pos + 1u > p.window) start = first_pos + 1u - p.window;

  for (uint kb = start; kb <= last_pos; kb += KEY_BLOCK) {
    const uint nk = min((uint)KEY_BLOCK, last_pos - kb + 1u);

    // Score every (query, key) pair: ONE THREAD per pair, a full dot product with no
    // cross-lane reduction. The previous version gave each pair a whole simdgroup and summed
    // across 32 lanes -- five shuffle steps of reduction overhead for ~two useful multiplies
    // per lane. nq*nk <= Q_BLOCK*KEY_BLOCK == nthr, so this is a single pass at full occupancy.
    for (uint pidx = lid; pidx < nq * nk; pidx += nthr) {
      const uint qi = pidx / nk, j = pidx % nk;
      const uint key = kb + j;
      const uint pos_i = base + t0 + qi;

      uint start_i = 0u;
      if (p.window != 0u && pos_i + 1u > p.window) start_i = pos_i + 1u - p.window;

      float d = -INFINITY;
      if (key <= pos_i && key >= start_i) {  // causal mask, plus the sliding window
        device const half* kt = k_cache + (ulong)key * (ulong)kv_dim + kv_head * hd;
        float acc_d = 0.0f;
        for (uint i = 0u; i < hd; ++i) acc_d += q_sh[qi * hd + i] * float(kt[i]);
        d = acc_d * p.scale;
      }
      sc_sh[qi * KEY_BLOCK + j] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fold the block into each query's own online softmax. ONE SIMDGROUP per query, so all
    // 256 threads work -- the old thread-per-query version left 8 threads busy and 248 idle at
    // the barrier below. KEY_BLOCK == 32 == simd width, so lane j owns key j and the max/sum are
    // lane-parallel reductions.
    const uint qi = simd_id;
    if (qi < nq) {
      // A fully masked block leaves new_max at -INFINITY; guard so exp(-inf - -inf) is not NaN.
      const float sc = (lane < nk) ? sc_sh[qi * KEY_BLOCK + lane] : -INFINITY;
      const float bmax = simd_max(sc);

      const float old_max = m_sh[qi];
      const float new_max = max(old_max, bmax);
      const float rescale = (old_max == -INFINITY) ? 0.0f : exp(old_max - new_max);

      const float w =
          (lane < nk && sc != -INFINITY && new_max != -INFINITY) ? exp(sc - new_max) : 0.0f;
      if (lane < nk) w_sh[qi * KEY_BLOCK + lane] = w;
      const float bsum = simd_sum(w);

      if (lane == 0u) {
        l_sh[qi] = l_sh[qi] * rescale + bsum;
        m_sh[qi] = new_max;
        r_sh[qi] = rescale;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = lid; c < nq * hd; c += nthr) {
      const uint qi = c / hd, i = c % hd;
      float a = acc[qi * hd + i] * r_sh[qi];
      for (uint j = 0u; j < nk; ++j) {
        device const half* vt = v_cache + (ulong)(kb + j) * (ulong)kv_dim + kv_head * hd;
        a += w_sh[qi * KEY_BLOCK + j] * float(vt[i]);
      }
      acc[qi * hd + i] = a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  for (uint c = lid; c < nq * hd; c += nthr) {
    const uint qi = c / hd, i = c % hd;
    const float inv = (l_sh[qi] > 0.0f) ? 1.0f / l_sh[qi] : 0.0f;
    out[(ulong)(t0 + qi) * (ulong)q_dim + head * hd + i] = half(acc[qi * hd + i] * inv);
  }
}

// ---------------------------------------------------------------------------

kernel void cpi_attention_decode(
    device const half*  q         [[buffer(0)]],
    device const half*  k_cache   [[buffer(1)]],
    device const half*  v_cache   [[buffer(2)]],
    device half*        out       [[buffer(3)]],
    device const int*   positions [[buffer(4)]],
    constant AttnParams& p        [[buffer(5)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  // One threadgroup per (query token, head). Decode is just T = 1, so prefill and decode
  // share this kernel instead of duplicating the softmax and the GQA maths.
  //
  // KEYS ARE PROCESSED IN BLOCKS. The first version walked the cache ONE KEY AT A TIME,
  // with two threadgroup barriers per key -- about 1100 barriers per (token, head) on a
  // 551-token prompt, which made prefill attention O(T^2) in barriers as well as in work.
  // Scoring a block of 32 keys and folding it into the online softmax ONCE cuts the
  // barrier count ~20x and lets the score dot-products run across all simdgroups at once.
  const uint total = p.heads * p.tokens;
  if (gid >= total) return;
  const uint t    = gid / p.heads;
  const uint head = gid % p.heads;

  uint base = p.position;
  if (p.use_position_buffer != 0u) base = uint(positions[0]);
  const uint pos = base + t;  // causality: token t attends to keys [start, base+t]

  const uint group   = p.heads / p.kv_heads;
  const uint kv_head = head / group;
  const uint kv_dim  = p.kv_heads * p.head_dim;
  const uint q_dim   = p.heads * p.head_dim;

  uint start = 0u;
  if (p.window != 0u && pos + 1u > p.window) start = pos + 1u - p.window;

  device const half* qh = q + (ulong)t * (ulong)q_dim + head * p.head_dim;

  const uint simd_id = lid / 32u;
  const uint lane    = lid % 32u;
  const uint n_simd  = nthr / 32u;

  threadgroup float q_sh[256];    // the query, read once instead of per key
  threadgroup float sc_sh[KEY_BLOCK];
  threadgroup float w_sh[KEY_BLOCK];
  threadgroup float tg_acc[256];
  threadgroup float tg_max;
  threadgroup float tg_sum;

  for (uint i = lid; i < p.head_dim; i += nthr) {
    q_sh[i] = float(qh[i]);
    tg_acc[i] = 0.0f;
  }
  if (lid == 0u) {
    tg_max = -INFINITY;
    tg_sum = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint kb = start; kb <= pos; kb += KEY_BLOCK) {
    const uint nk = min((uint)KEY_BLOCK, pos - kb + 1u);

    // Score the whole block: each simdgroup takes a key and reduces its dot product.
    for (uint j = simd_id; j < nk; j += n_simd) {
      device const half* kt = k_cache + (ulong)(kb + j) * (ulong)kv_dim + kv_head * p.head_dim;
      float d = 0.0f;
      for (uint i = lane; i < p.head_dim; i += 32u) d += q_sh[i] * float(kt[i]);
      d = simd_sum(d);
      if (lane == 0u) sc_sh[j] = d * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax, folded in ONCE for the block rather than once per key.
    float bmax = -INFINITY;
    for (uint j = 0u; j < nk; ++j) bmax = max(bmax, sc_sh[j]);

    const float old_max = tg_max;
    const float new_max = max(old_max, bmax);
    const float rescale = (old_max == -INFINITY) ? 0.0f : exp(old_max - new_max);

    for (uint j = lid; j < nk; j += nthr) w_sh[j] = exp(sc_sh[j] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bsum = 0.0f;
    for (uint j = 0u; j < nk; ++j) bsum += w_sh[j];

    for (uint i = lid; i < p.head_dim; i += nthr) {
      float a = tg_acc[i] * rescale;
      for (uint j = 0u; j < nk; ++j) {
        device const half* vt = v_cache + (ulong)(kb + j) * (ulong)kv_dim + kv_head * p.head_dim;
        a += w_sh[j] * float(vt[i]);
      }
      tg_acc[i] = a;
    }

    // Every thread read tg_max above; thread 0 is about to overwrite it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
      tg_sum = tg_sum * rescale + bsum;
      tg_max = new_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv = 1.0f / tg_sum;
  for (uint i = lid; i < p.head_dim; i += nthr) {
    out[(ulong)t * (ulong)q_dim + head * p.head_dim + i] = half(tg_acc[i] * inv);
  }
}

// ---------------------------------------------------------------------------
// Split-KV decode attention.
//
// cpi_attention_decode above gives one threadgroup to each (token, head) and walks the whole
// cache inside it. For a PREFILL that is fine: T tokens x heads is thousands of threadgroups.
// At DECODE T is 1, so the grid is just `heads` -- 14 threadgroups for Qwen2.5-0.5B, on a GPU
// that wants hundreds. The cache walk becomes a serial loop in a nearly empty machine, so
// decode slows down with context for a reason that has nothing to do with bandwidth: at 2048
// keys the KV cache is ~25 MB against 1.17 GB of weights, 2% of the traffic for a 2.6x
// slowdown (88 -> 34 tok/s, where llama.cpp holds 96 -> 89).
//
// So split the keys. Pass 1 gives each (head, chunk) its own threadgroup and an INDEPENDENT
// online softmax over its slice, writing that slice's running max, running sum and an
// UNNORMALIZED accumulator. Pass 2 merges the slices per head.
//
// The merge is exact, not an approximation: softmax combines under the standard log-sum-exp
// rule, so rescaling each slice by exp(m_c - m) and summing reproduces the single-threadgroup
// result. It is the same decomposition the CUDA split-K decode path uses
// (kernels_attention_decode.cu), on purpose -- same statistics, same merge, so the backends
// cannot drift and one golden covers both.
// ---------------------------------------------------------------------------

// Keys per pass of the decode scoring loop, and the threadgroup width that serves it. They are
// the same number ON PURPOSE: the loop gives each THREAD one whole key, so the 64-element dot
// product stays in one thread and there is no cross-lane reduction at all.
//
// The kernel this replaced split each key's dot product across a simdgroup -- 32 lanes doing 2
// MACs each, then a 5-step shuffle reduction, PER KEY. That is two MACs of work per reduction and
// it ran the per-key term at ~23 GFLOP/s, 0.5% of peak. One key per lane needs as many keys in
// flight as threads, which is why this threadgroup is 64 wide and not the 256 the rest of the file
// uses: a decode chunk is only ~56 keys once the chunks are sized to fill the GPU, so 256 threads
// would leave 200 of them idle and trade one bottleneck for another.
#define DEC_KEY_BLOCK 64
#define DEC_TG 64

struct AttnSplitParams {
  uint heads;
  uint kv_heads;
  uint head_dim;
  uint position;
  uint use_position_buffer;
  uint window;
  float scale;
  uint chunk_size;  // keys per chunk
  uint chunks;      // chunks per head, and the row stride of the partial buffers
};

kernel void cpi_attention_decode_split(
    device const half*  q         [[buffer(0)]],
    device const half*  k_cache   [[buffer(1)]],
    device const half*  v_cache   [[buffer(2)]],
    device float*       part_m    [[buffer(3)]],
    device float*       part_l    [[buffer(4)]],
    device float*       part_o    [[buffer(5)]],
    device const int*   positions [[buffer(6)]],
    constant AttnSplitParams& p   [[buffer(7)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  const uint head  = gid / p.chunks;
  const uint chunk = gid % p.chunks;
  if (head >= p.heads) return;  // uniform across the threadgroup: gid is a threadgroup id

  uint pos = p.position;
  if (p.use_position_buffer != 0u) pos = uint(positions[0]);

  uint start = 0u;
  if (p.window != 0u && pos + 1u > p.window) start = pos + 1u - p.window;

  const uint c0 = start + chunk * p.chunk_size;
  const uint c1 = min(c0 + p.chunk_size, pos + 1u);
  const uint slot = head * p.chunks + chunk;

  // An empty chunk still has to say so: the merge reads every slot. -INF max with a zero sum
  // is the identity of the log-sum-exp merge, and it lets pass 2 skip the accumulator without
  // reading whatever was left in it.
  if (c0 >= c1) {
    if (lid == 0u) {
      part_m[slot] = -INFINITY;
      part_l[slot] = 0.0f;
    }
    return;
  }

  const uint group   = p.heads / p.kv_heads;
  const uint kv_head = head / group;
  const uint kv_dim  = p.kv_heads * p.head_dim;

  device const half* qh = q + head * p.head_dim;  // decode: the query is the only token

  const uint simd_id = lid / 32u;
  const uint lane    = lid % 32u;
  const uint n_simd  = nthr / 32u;

  threadgroup float q_sh[256];
  threadgroup float sc_sh[DEC_KEY_BLOCK];
  threadgroup float w_sh[DEC_KEY_BLOCK];
  threadgroup float tg_acc[256];
  threadgroup float tg_max;
  threadgroup float tg_sum;
  threadgroup float red[DEC_TG / 32];  // one slot per simdgroup, for the block reductions

  for (uint i = lid; i < p.head_dim; i += nthr) {
    q_sh[i] = float(qh[i]);
    tg_acc[i] = 0.0f;
  }
  if (lid == 0u) {
    tg_max = -INFINITY;
    tg_sum = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint kb = c0; kb < c1; kb += DEC_KEY_BLOCK) {
    const uint nk = min((uint)DEC_KEY_BLOCK, c1 - kb);

    // Score: ONE KEY PER THREAD. The whole dot product lives in one thread, so no shuffles.
    for (uint j = lid; j < nk; j += nthr) {
      device const half* kt = k_cache + (ulong)(kb + j) * (ulong)kv_dim + kv_head * p.head_dim;
      float d = 0.0f;
      for (uint i = 0u; i < p.head_dim; ++i) d += q_sh[i] * float(kt[i]);
      sc_sh[j] = d * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Block max, as a tree. Every thread used to scan all nk keys to find it, which is nk*nthr
    // reads of threadgroup memory to compute one number.
    float lmax = -INFINITY;
    for (uint j = lid; j < nk; j += nthr) lmax = max(lmax, sc_sh[j]);
    lmax = simd_max(lmax);
    if (lane == 0u) red[simd_id] = lmax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float bmax = -INFINITY;
    for (uint s = 0u; s < n_simd; ++s) bmax = max(bmax, red[s]);

    const float old_max = tg_max;
    const float new_max = max(old_max, bmax);
    const float rescale = (old_max == -INFINITY) ? 0.0f : exp(old_max - new_max);

    for (uint j = lid; j < nk; j += nthr) w_sh[j] = exp(sc_sh[j] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Block sum, same shape. The barrier above also fences `red` between its two uses.
    float lsum = 0.0f;
    for (uint j = lid; j < nk; j += nthr) lsum += w_sh[j];
    lsum = simd_sum(lsum);
    if (lane == 0u) red[simd_id] = lsum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float bsum = 0.0f;
    for (uint s = 0u; s < n_simd; ++s) bsum += red[s];

    for (uint i = lid; i < p.head_dim; i += nthr) {
      float a = tg_acc[i] * rescale;
      for (uint j = 0u; j < nk; ++j) {
        device const half* vt = v_cache + (ulong)(kb + j) * (ulong)kv_dim + kv_head * p.head_dim;
        a += w_sh[j] * float(vt[i]);
      }
      tg_acc[i] = a;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
      tg_sum = tg_sum * rescale + bsum;
      tg_max = new_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Unnormalized on purpose: 1/sum belongs to the merge, which is the only place the whole
  // sum is known.
  if (lid == 0u) {
    part_m[slot] = tg_max;
    part_l[slot] = tg_sum;
  }
  for (uint i = lid; i < p.head_dim; i += nthr) {
    part_o[(ulong)slot * (ulong)p.head_dim + i] = tg_acc[i];
  }
}

kernel void cpi_attention_decode_merge(
    device const float* part_m  [[buffer(0)]],
    device const float* part_l  [[buffer(1)]],
    device const float* part_o  [[buffer(2)]],
    device half*        out     [[buffer(3)]],
    constant AttnSplitParams& p [[buffer(4)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  const uint head = gid;
  if (head >= p.heads) return;
  const uint row = head * p.chunks;

  // Every thread redoes this reduction rather than staging it through threadgroup memory and
  // two more barriers. chunks is small (tens), so the arithmetic is cheaper than the barriers.
  float m = -INFINITY;
  for (uint c = 0u; c < p.chunks; ++c) m = max(m, part_m[row + c]);
  if (m == -INFINITY) {  // no keys at all: nothing to normalize by
    for (uint i = lid; i < p.head_dim; i += nthr) out[head * p.head_dim + i] = half(0.0f);
    return;
  }

  float l = 0.0f;
  for (uint c = 0u; c < p.chunks; ++c) {
    if (part_m[row + c] == -INFINITY) continue;
    l += part_l[row + c] * exp(part_m[row + c] - m);
  }
  const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;

  for (uint i = lid; i < p.head_dim; i += nthr) {
    float a = 0.0f;
    for (uint c = 0u; c < p.chunks; ++c) {
      if (part_m[row + c] == -INFINITY) continue;
      a += part_o[(ulong)(row + c) * (ulong)p.head_dim + i] * exp(part_m[row + c] - m);
    }
    out[head * p.head_dim + i] = half(a * inv);
  }
}

// ---------------------------------------------------------------------------
// Batched paged decode -- the kernels continuous batching needs.
//
// The kernels above serve ONE sequence, whose KV is contiguous: the key at logical
// position p lives at p * kv_dim. Continuous batching cannot use that layout, because the
// sequences in a batch start, grow and finish independently -- a contiguous per-sequence
// cache would have to reserve max_context for every slot. Instead the cache is one pool per
// layer, carved into fixed-size blocks, and each sequence owns a block table mapping its
// logical positions onto physical blocks it does not have to own contiguously:
//
//   phys = block_table[b][pos / block_size] * block_size + (pos % block_size)
//
// That is deliberately the same arithmetic, pool layout and block-table encoding the CUDA
// paged kernels use, so a block table built by the shared host-side allocator
// (engine::SequenceBlockTable, which has no backend in it at all) means the same thing on
// both backends. Blocks are refcounted, so two sequences sharing a system prompt share its
// blocks outright -- which is why the table is a gather rather than a base offset.
// ---------------------------------------------------------------------------

struct KvPagedParams {
  uint kv_hidden;   // kv_heads * head_dim
  uint max_blocks;  // row stride of block_tables
  uint block_size;  // tokens per block
  uint batch;
};

// Scatter each sequence's one new K/V row into its own block. One threadgroup per sequence.
kernel void cpi_kv_store_batched_paged(
    device const half*  k_src        [[buffer(0)]],  // [batch][kv_hidden]
    device const half*  v_src        [[buffer(1)]],
    device half*        k_pool       [[buffer(2)]],
    device half*        v_pool       [[buffer(3)]],
    device const int*   block_tables [[buffer(4)]],  // [batch][max_blocks]
    device const int*   positions    [[buffer(5)]],  // [batch] -- each row's own position
    constant KvPagedParams& p        [[buffer(6)]],
    uint b    [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  if (b >= p.batch) return;

  const uint pos = uint(positions[b]);
  device const int* bt = block_tables + (ulong)b * (ulong)p.max_blocks;
  const uint phys = uint(bt[pos / p.block_size]) * p.block_size + (pos % p.block_size);

  device half* kd = k_pool + (ulong)phys * (ulong)p.kv_hidden;
  device half* vd = v_pool + (ulong)phys * (ulong)p.kv_hidden;
  device const half* ks = k_src + (ulong)b * (ulong)p.kv_hidden;
  device const half* vs = v_src + (ulong)b * (ulong)p.kv_hidden;
  for (uint d = lid; d < p.kv_hidden; d += nthr) {
    kd[d] = ks[d];
    vd[d] = vs[d];
  }
}

struct AttnPagedParams {
  uint heads;
  uint kv_heads;
  uint head_dim;
  uint max_blocks;
  uint block_size;
  uint window;  // 0 = full causal; else sliding window length
  float scale;
  uint batch;
};

// One threadgroup per (sequence, head). Each sequence attends over its OWN length, gathered
// through its OWN block table -- the ragged batch the scheduler hands us. Structurally this
// is cpi_attention_decode with the contiguous key address replaced by a paged gather; the
// online softmax is unchanged, so the two agree key-for-key when a block table happens to be
// identity (which is what the parity gate checks).
kernel void cpi_attention_decode_batched_paged(
    device const half*  q            [[buffer(0)]],  // [batch][heads*head_dim]
    device const half*  k_pool       [[buffer(1)]],
    device const half*  v_pool       [[buffer(2)]],
    device half*        out          [[buffer(3)]],  // [batch][heads*head_dim]
    device const int*   block_tables [[buffer(4)]],  // [batch][max_blocks]
    device const int*   seq_lens     [[buffer(5)]],  // [batch] -- length INCLUDING the new token
    constant AttnPagedParams& p      [[buffer(6)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  const uint total = p.batch * p.heads;
  if (gid >= total) return;
  const uint b    = gid / p.heads;
  const uint head = gid % p.heads;

  const uint seq_len = uint(seq_lens[b]);
  if (seq_len == 0u) return;
  const uint pos = seq_len - 1u;  // the query is this sequence's newest token

  const uint group   = p.heads / p.kv_heads;
  const uint kv_head = head / group;
  const uint kv_dim  = p.kv_heads * p.head_dim;
  const uint q_dim   = p.heads * p.head_dim;

  uint start = 0u;
  if (p.window != 0u && seq_len > p.window) start = seq_len - p.window;

  device const int* bt = block_tables + (ulong)b * (ulong)p.max_blocks;
  device const half* qh = q + (ulong)b * (ulong)q_dim + head * p.head_dim;

  const uint simd_id = lid / 32u;
  const uint lane    = lid % 32u;
  const uint n_simd  = nthr / 32u;

  threadgroup float q_sh[256];
  threadgroup float sc_sh[KEY_BLOCK];
  threadgroup float w_sh[KEY_BLOCK];
  threadgroup float tg_acc[256];
  threadgroup float tg_max;
  threadgroup float tg_sum;

  for (uint i = lid; i < p.head_dim; i += nthr) {
    q_sh[i] = float(qh[i]);
    tg_acc[i] = 0.0f;
  }
  if (lid == 0u) {
    tg_max = -INFINITY;
    tg_sum = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint kb = start; kb <= pos; kb += KEY_BLOCK) {
    const uint nk = min((uint)KEY_BLOCK, pos - kb + 1u);

    for (uint j = simd_id; j < nk; j += n_simd) {
      const uint kpos = kb + j;
      const uint phys = uint(bt[kpos / p.block_size]) * p.block_size + (kpos % p.block_size);
      device const half* kt = k_pool + (ulong)phys * (ulong)kv_dim + kv_head * p.head_dim;
      float d = 0.0f;
      for (uint i = lane; i < p.head_dim; i += 32u) d += q_sh[i] * float(kt[i]);
      d = simd_sum(d);
      if (lane == 0u) sc_sh[j] = d * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bmax = -INFINITY;
    for (uint j = 0u; j < nk; ++j) bmax = max(bmax, sc_sh[j]);

    const float old_max = tg_max;
    const float new_max = max(old_max, bmax);
    const float rescale = (old_max == -INFINITY) ? 0.0f : exp(old_max - new_max);

    for (uint j = lid; j < nk; j += nthr) w_sh[j] = exp(sc_sh[j] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bsum = 0.0f;
    for (uint j = 0u; j < nk; ++j) bsum += w_sh[j];

    for (uint i = lid; i < p.head_dim; i += nthr) {
      float a = tg_acc[i] * rescale;
      for (uint j = 0u; j < nk; ++j) {
        const uint kpos = kb + j;
        const uint phys = uint(bt[kpos / p.block_size]) * p.block_size + (kpos % p.block_size);
        device const half* vt = v_pool + (ulong)phys * (ulong)kv_dim + kv_head * p.head_dim;
        a += w_sh[j] * float(vt[i]);
      }
      tg_acc[i] = a;
    }

    // Every thread read tg_max above; thread 0 is about to overwrite it.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
      tg_sum = tg_sum * rescale + bsum;
      tg_max = new_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv = 1.0f / tg_sum;
  for (uint i = lid; i < p.head_dim; i += nthr) {
    out[(ulong)b * (ulong)q_dim + head * p.head_dim + i] = half(tg_acc[i] * inv);
  }
}

// ---------------------------------------------------------------------------
// Mixture of Experts (Mixtral).
//
// The shape of an MoE layer, for orientation:
//   router logits [experts] --RouterTopk--> idx[k], weight[k]   (softmax, top-k, renorm)
//   x [hidden]              --GateUpGeglu-> inter[k][einter]    (per selected expert)
//   inter                   --DownAccum---> y [hidden]          (routing-weighted sum)
//
// The selection crosses ops in DEVICE buffers, so a token never round-trips to the host
// mid-layer. These mirror PlanCudaEngine's kernels; the gate is reproducing the CUDA token
// stream on tiny-mixtral, so a deviation here is a bug even where it looks like an
// improvement.
// ---------------------------------------------------------------------------

