#define Q_BLOCK 8
// The largest head_dim the scalar attention kernels can hold in threadgroup memory. Was a bare
// 256 repeated in six places, which is exactly the head_dim of Qwen3.5 and Gemma 4's sliding
// layers, so it fit every model anyone had run. Gemma 4 E2B's full layers are head_dim 512 and
// wrote straight past these arrays, which is where its NaN logits came from.
//
// Cost is 2 arrays * ATTN_MAX_HEAD_DIM floats per threadgroup: 4 KB at 512, up from 2 KB. Well
// inside the 32 KB budget, but it is occupancy that pays, so this is measured, not assumed.
#define ATTN_MAX_HEAD_DIM 512
// Query-block size of cpi_attention_prefill_wide, the head_dim > 256 variant of the scalar
// prefill kernel. The main kernel's q_sh/acc are Q_BLOCK * 256 floats and cannot be raised to
// ATTN_MAX_HEAD_DIM at Q_BLOCK 8: that is 2 * 16 KB against a 32 KB threadgroup budget.
// Halving the query block keeps the pair at 16 KB. Only head_dim > 256 layers (Gemma 4's full
// layers) take the wide kernel; the tuned 256 path is untouched.
#define Q_BLOCK_WIDE 4
// The matrix-unit prefill kernel's query-block size, in queries (QMM_BLOCK/8 row fragments).
// Separate from Q_BLOCK: the scalar kernel sizes its shared arrays for head_dim 256 and cannot
// afford a wider block. 16 fills the scoring loop, which runs qfs*n_kg work items: with
// n_kg = KEY_BLOCK/8 = 4, one query fragment leaves 4 of 8 simdgroups idle where two fill them.
// Measured at T=2041 (attention 39% of the pass): 330 -> 301 ms (-9%) from 8 to 16, back to 330 at
// 32 (four fragments loop twice for the same parallelism at more threadgroup memory). This is
// purely about simdgroup occupancy in the score matmul, not K/V re-reads; those are LLC-served.
//
// Raising it above 8 requires the matrix ops to loop over QMM_BLOCK/8 fragments, which they now do;
// before that a block of 16 scored only its first 8 queries and left the rest as garbage.
//
// Threadgroup ceiling: q_sh + acc + sc_sh + pw_sh = QMM_BLOCK * (2*128 + 4*128 + 4*32 + 2*32)
// bytes, so 16 is ~15 KB of the 32 KB budget and 32 is ~30 KB.
#define QMM_BLOCK 16

// The scoring and P.V loops hold a QMM_QT x QMM_KT register tile, so they walk query fragments in
// groups of QMM_QT and round the count up. A block that is not a whole number of those groups
// therefore reads query fragments past its own rows: q_sh and acc are sized QMM_BLOCK rows, so
// that is off the end of the array, not merely zero padding, and the kernel silently computes
// wrong attention. QMM_BLOCK 8 did exactly that (the golden failed while every benchmark looked
// fine), and 24 would too: being a multiple of 8 is not enough, it must be a multiple of 8*QMM_QT.
#define QMM_QT 2
#define QMM_KT 2
#if (QMM_BLOCK % (8 * QMM_QT)) != 0
#error "QMM_BLOCK must be a multiple of 8*QMM_QT -- otherwise the register tile reads past q_sh"
#endif

// RCA instrumentation for the prefill attention kernel. 0 = normal. Each bit replaces one phase
// with a dependency-preserving stub: the answer becomes wrong, but every other phase still runs
// and nothing downstream gets dead-code-eliminated, so the wall-clock delta is that phase's cost.
// This is the only way to see inside a kernel whose total is all we can otherwise measure.
//   1 = scoring matmul stubbed (sc_sh still written, so the softmax is still fed)
//   2 = softmax stubbed (pw_sh still written, so P.V is still fed)
//   4 = P.V matmul stubbed (acc untouched; the output loop still reads it)
// Bits combine: 7 stubs all three at once, which prices the residue no phase accounts for (the
// K/V loads, the barriers, and the output loop). Read the singles against that, not against 0: the
// phases overlap across concurrent threadgroups, so their measured costs are superadditive.
#define ATTN_RCA 0

// Shape constants for the matrix-unit prefill attention kernel, supplied at pipeline creation
// rather than read out of the params block. head_dim bounds the kernel's inner loops and sizes its
// threadgroup arrays, and the two strides land in every address it computes; as runtime values the
// compiler can neither unroll against them nor fold them. Specialising measured 612 -> 593 ms on a
// 2041-token prefill for byte-identical output. See MetalContext::set_next_specialization: a
// caller that fails to supply these gets a loud pipeline-creation failure, not a wrong answer.
constant uint FC_head_dim [[function_constant(0)]];
constant uint FC_kv_dim [[function_constant(1)]];
constant uint FC_q_dim [[function_constant(2)]];

// ---------------------------------------------------------------------------
// Prefill attention on the matrix units (head_dim <= 128). The scalar kernel below scores
// QK^T and does P.V with per-thread dot products; the matrix units sit idle through the
// whole of attention, which the RCA put at ~21% of prefill. This computes both products with
// simdgroup_matrix (fp32 accumulate, the fast path), keeping the same per-query online softmax.
//
// One threadgroup per (query block, head); 8 simdgroups. The block is QMM_BLOCK queries, which is
// QMM_BLOCK/8 row fragments: every matrix op below loops over them, so the constant can move
// without the kernel silently computing only its first 8 rows. For a KEY_BLOCK of 32 keys:
//   scoring  S[8q x 8k]  = sum over head_dim/8 of  Q[8q x 8d] @ (K[8k x 8d])^T   (K loaded transposed)
//   P.V      O[8q x 8d] += sum over the key groups of  P[8q x 8k] @ V[8k x 8d]
// The online rescale between key blocks is applied to the fp32 accumulator in threadgroup
// memory (a per-row scalar, which a matrix fragment cannot scale), then the P.V matmul
// accumulates into it. Gemma's head_dim 256 does not fit the fragment plan and keeps the
// scalar kernel.
// ---------------------------------------------------------------------------
// Body shared by the narrow and wide variants, parameterized by query block and register tile.
// The threadgroup arrays are the caller's: MSL cannot size an array from a function constant,
// so each kernel declares its own at its max head_dim.
template <uint QB, uint QT_, uint KT_>
static void attn_prefill_mm_body(device const half* q, device const half* k_cache,
                                 device const half* v_cache, device half* out,
                                 device const int* positions, device const int* block_tables,
                                 constant AttnParams& p, threadgroup half* q_sh,
                                 threadgroup float* acc, threadgroup float* sc_sh,
                                 threadgroup half* pw_sh, threadgroup float* m_sh,
                                 threadgroup float* l_sh, uint gid, uint lid, uint nthr) {
  static_assert(QB % (8u * QT_) == 0u,
                "query block must be a whole number of QT-sized fragment groups");
  const uint head = gid % p.heads;
  const uint t0 = (gid / p.heads) * QB;
  if (t0 >= p.tokens) return;
  const uint nq = min((uint)QB, p.tokens - t0);

  uint base = p.position;
  if (p.use_position_buffer != 0u) base = uint(positions[0]);

  const uint group = p.heads / p.kv_heads;
  const uint kv_head = head / group;
  const uint kv_dim = FC_kv_dim;
  const uint q_dim = FC_q_dim;
  const uint hd = FC_head_dim;  // bounded by the caller's arrays, compile-time

  const uint simd_id = lid / 32u;
  const uint lane = lid % 32u;
  const uint n_simd = nthr / 32u;

  // Zero the full QB rows, not just nq: the matrix ops load 8-row fragments, so the
  // padding rows must be defined (0) rather than garbage that could turn into NaN.
  for (uint c = lid; c < QB * hd; c += nthr) {
    const uint qi = c / hd, i = c % hd;
    q_sh[qi * hd + i] = (qi < nq) ? q[(ulong)(t0 + qi) * (ulong)q_dim + head * hd + i] : half(0);
    acc[qi * hd + i] = 0.0f;
  }
  for (uint qi = lid; qi < QB; qi += nthr) {
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

    // Where this key block physically lives. One lookup per block, not per key: the caller
    // guarantees block_size % MM_KEY_BLOCK == 0 and no sliding window (so start == 0 and every kb
    // is MM_KEY_BLOCK-aligned), which together mean the whole run [kb, kb+MM_KEY_BLOCK) sits inside a
    // single physical block and stays contiguous, so the simdgroup_loads below keep working
    // unchanged, just from a different base. Break either guarantee and a key block could
    // straddle two blocks, which this addressing cannot express; the dispatch enforces both.
    const uint kphys = (p.paged != 0u)
                           ? uint(block_tables[kb / p.block_size]) * p.block_size + (kb % p.block_size)
                           : kb;

    // --- Scoring: each simdgroup computes a QT x KT tile of S[8q x 8k] fragments. ---
    // Register tiling, the same trick that makes the GEMM fast. One output fragment per work item
    // means loading a Qf and a Kf for every single matrix op: two loads per MAC, an arithmetic
    // intensity of 0.5, against the GEMM's 2.0. That is why attention runs at a few percent of the
    // GEMM's FLOP rate: it is starved on fragment loads, not short of matrix throughput. Holding a
    // 2x2 tile reuses each loaded Qf across KT key groups and each Kf across QT query fragments:
    // 4 loads feed 4 MACs (intensity 1.0), and with qfs=2 and n_kg up to 16 the tile count still
    // fills all 8 simdgroups.
    //
    // Query fragments past qfs are safe to compute because QB is a whole number of QT-sized
    // groups (the static_assert above): q_sh is zeroed over the full QB rows, so they
    // contribute zeros that the softmax's nq bound ignores. Key groups past n_kg are NOT safe
    // to read (they can run off the end of the KV cache), so their load is clamped and their store
    // suppressed.
    constexpr uint QT = QT_, KT = KT_;
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
          // K is [key][d]; load the [8k x 8d] tile transposed to get [8d x 8k].
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
    for (uint c = lid; c < QB * MM_KEY_BLOCK; c += nthr) sc_sh[c] = 0.0f;
#endif
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax, a simdgroup per query. The simd is 32 wide and MM_KEY_BLOCK is wider, so each
    // lane owns MM_KEY_BLOCK/32 keys and
    // folds their max and sum before the simd reduction. A wide key block halves the number of
    // iterations (and their barriers and reductions) versus a narrow one, which is the point:
    // attention's per-block overhead, not its matmuls, is what makes this kernel slower than the
    // flash-attention path it is measured against.
    //
    // Scale and the causal/window mask fold in here rather than a separate sc_sh sweep: the softmax
    // already holds the query and its keys, so the masked, scaled score feeds the reduction
    // directly: one fewer barrier and pass per block.
    //
    // Falsified, do not retry (all neutral): a fast exp, skipping the accumulator rescale when the
    // max did not move, interleaving two queries' reductions. Per ATTN_RCA the cost is not in this
    // body but in the score round-trip through threadgroup memory that phase-partitioning forces.
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
    // per work item meant loading a Pf and a Vf for every MAC (intensity 0.5). Holding QT of them
    // reuses each Vf across the query fragments, 3 loads feed 2 MACs (0.67), while dt_n = dfs
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

// head_dim <= 128: 16 queries x a 2x2 register tile, ~15 KB of threadgroup memory.
kernel void cpi_attention_prefill_mm(device const half* q [[buffer(0)]],
                                     device const half* k_cache [[buffer(1)]],
                                     device const half* v_cache [[buffer(2)]],
                                     device half* out [[buffer(3)]],
                                     device const int* positions [[buffer(4)]],
                                     // One sequence's logical->physical block map, used only when
                                     // p.paged != 0. Bound to a dummy otherwise (the params block
                                     // must stay last, so the slot cannot simply be absent).
                                     device const int* block_tables [[buffer(5)]],
                                     constant AttnParams& p [[buffer(6)]],
                                     uint gid [[threadgroup_position_in_grid]],
                                     uint lid [[thread_position_in_threadgroup]],
                                     uint nthr [[threads_per_threadgroup]]) {
  threadgroup half q_sh[QMM_BLOCK * 128];
  threadgroup float acc[QMM_BLOCK * 128];
  threadgroup float sc_sh[QMM_BLOCK * MM_KEY_BLOCK];
  threadgroup half pw_sh[QMM_BLOCK * MM_KEY_BLOCK];
  threadgroup float m_sh[QMM_BLOCK];
  threadgroup float l_sh[QMM_BLOCK];
  attn_prefill_mm_body<QMM_BLOCK, QMM_QT, QMM_KT>(q, k_cache, v_cache, out, positions, block_tables,
                                                  p, q_sh, acc, sc_sh, pw_sh, m_sh, l_sh, gid, lid,
                                                  nthr);
}

// 128 < head_dim <= 256 (Gemma 4's sliding layers, Qwen3.5). These used to fall to the scalar
// kernel, leaving the matrix units idle for 41% of a Gemma prefill.
//
// The query block halves to pay for the wider rows: q_sh + acc alone are QB*256*6 bytes, so 16
// queries puts the four arrays at 36 KB against a 32 KB budget. 8 queries is one row fragment,
// so QT must drop to 1 with it; QB 8 at QT 2 is the read-past-q_sh case described above
// QMM_QT. KT stays 2: key groups fill the simdgroups once the query tile cannot.
#define QMM_BLOCK_W 8
kernel void cpi_attention_prefill_mm_wide(device const half* q [[buffer(0)]],
                                          device const half* k_cache [[buffer(1)]],
                                          device const half* v_cache [[buffer(2)]],
                                          device half* out [[buffer(3)]],
                                          device const int* positions [[buffer(4)]],
                                          device const int* block_tables [[buffer(5)]],
                                          constant AttnParams& p [[buffer(6)]],
                                          uint gid [[threadgroup_position_in_grid]],
                                          uint lid [[thread_position_in_threadgroup]],
                                          uint nthr [[threads_per_threadgroup]]) {
  threadgroup half q_sh[QMM_BLOCK_W * 256];
  threadgroup float acc[QMM_BLOCK_W * 256];
  threadgroup float sc_sh[QMM_BLOCK_W * MM_KEY_BLOCK];
  threadgroup half pw_sh[QMM_BLOCK_W * MM_KEY_BLOCK];
  threadgroup float m_sh[QMM_BLOCK_W];
  threadgroup float l_sh[QMM_BLOCK_W];
  attn_prefill_mm_body<QMM_BLOCK_W, 1, QMM_KT>(q, k_cache, v_cache, out, positions, block_tables, p,
                                               q_sh, acc, sc_sh, pw_sh, m_sh, l_sh, gid, lid, nthr);
}

// ---------------------------------------------------------------------------
// Register-resident prefill attention (head_dim <= 128).
//
// S is computed into simdgroup_float8x8 registers, the softmax rewrites them in place and P.V
// consumes them: no threadgroup memory and no barrier in the key loop, where the phase-partitioned
// kernel above spills S between each phase. Q is gathered and the output scattered per lane, so the
// threadgroup footprint is zero and occupancy is register-bound.
//
// Depends on which lane of a simdgroup holds which element of an 8x8 fragment, which MSL does not
// document. cpi_simdgroup_layout_probe measures it at start-up and the engine selects this kernel
// only if the device matches; anything else keeps the phase-partitioned kernel. Measured mapping
// (32-wide simd, 2 elements per lane, identical for half and float fragments, which is what lets P
// be written back for the matmul):
//     row         = ((lane & 16) >> 2) | ((lane & 6) >> 1)
//     col of e[0] = ((lane & 8) >> 1) | ((lane & 1) << 1),   e[1] at col + 1
// A row therefore lives in four lanes differing only in bits 0 and 3 (row reduction = two shuffles,
// not a 32-lane simd_max), and a lane holds the same row of every fragment, so each lane tracks one
// query's running max and sum in registers.
//
// STATUS: correct (goldens pass) but research-only, and slower by enough to retire the idea.
// Attention ~283 ms vs the phase kernel's ~117 at T=2041; CPI_METAL_ATTN_FA=1 makes prefill ~774 ms
// against 612. Per phase (FA_RCA above):
//
//                  scoring   softmax    P.V     total
//   phase           48 ms     60 ms    50 ms    117 ms
//   register-res.  130 ms     63 ms   107 ms    283 ms
//
// The softmax is this kernel's whole point and costs the same (63 vs 60), so the phase kernel's
// 60 ms is intrinsic per-element work over ~700M score elements, not the threadgroup round trip it
// was attributed to. That also explains why every memory-shaped attack on it measured neutral.
//
// Falsified, do not retry: FA_QT 1/FA_KB 32 753 ms (best), 1/64 789, 2/32 820 (fragment spill),
// 2/64 1006; K/V staged in threadgroup memory 775 vs 771; K pre-transposed in that copy 769;
// head_dim as a compile-time constant neutral. The matmuls are 2.2-2.7x slower and none of
// registers, occupancy, the transposing load, cross-simdgroup redundancy, intensity or block width
// accounts for it.
//
// Remaining gap in the phase kernel: its phases sum to 158 ms serial but cost 117, so only ~35%
// overlaps and that comes from concurrent threadgroups; three barriers per key block serialise
// the phases within one. Fully overlapped is max(48, 60, 50), which is llama.cpp's ~66 ms. The
// lever is pipelining the key loop, not making a phase cheaper.
// ---------------------------------------------------------------------------
#define FA_NSG 4                       // simdgroups per threadgroup; one query row fragment each
#define FA_QBLK (8u * FA_NSG)          // queries per threadgroup
#define FA_KB  32                      // keys per block -> FA_KB/8 score fragments held at once
#define FA_KG  (FA_KB / 8u)

// Phase decomposition, same contract as ATTN_RCA on the phase-partitioned kernel: each bit removes
// one phase while leaving what the next phase reads intact, so nothing downstream is dead-code
// eliminated and the wall-clock delta is that phase.
//   1 = scoring matmul skipped (S keeps its zero fill, so the softmax still consumes it)
//   4 = P.V matmul skipped (O keeps its zero fill, so the output loop still consumes it)
// A stub whose result goes unread lets the compiler delete the phase outright and then prices
// nothing. That is not hypothetical: a first attempt here filled P directly and left S written but
// unused, the scoring matmul vanished, and the softmax looked like it cost 28 ms when it costs 63.
#define FA_RCA 0

#define FA_ROW(l) (((((uint)(l)) & 16u) >> 2) | ((((uint)(l)) & 6u) >> 1))
#define FA_COL(l) (((((uint)(l)) & 8u) >> 1) | ((((uint)(l)) & 1u) << 1))

// Reduce across the four lanes holding one fragment row. They differ only in bits 0 and 3, so two
// butterfly steps suffice and every participating lane ends with the full result.
inline float fa_row_max(float v) {
  v = max(v, simd_shuffle_xor(v, 1u));
  v = max(v, simd_shuffle_xor(v, 8u));
  return v;
}
inline float fa_row_sum(float v) {
  v += simd_shuffle_xor(v, 1u);
  v += simd_shuffle_xor(v, 8u);
  return v;
}

// Writes the fragment layout to a 64-element row-major buffer: element (r,c) receives the index of
// the lane that owns it times two, plus which of that lane's two elements it is. The host checks
// this against FA_ROW/FA_COL before cpi_attention_prefill_fa is allowed to run.
kernel void cpi_simdgroup_layout_probe(device float* out [[buffer(0)]],
                                       uint lane [[thread_index_in_simdgroup]]) {
  simdgroup_float8x8 S = make_filled_simdgroup_matrix<float, 8, 8>(-1.0f);
  thread auto& e = S.thread_elements();
  e[0] = float(lane * 2u);
  e[1] = float(lane * 2u + 1u);
  simdgroup_store(S, out, 8);
}

kernel void cpi_attention_prefill_fa(device const half* q [[buffer(0)]],
                                     device const half* k_cache [[buffer(1)]],
                                     device const half* v_cache [[buffer(2)]],
                                     device half* out [[buffer(3)]],
                                     device const int* positions [[buffer(4)]],
                                     device const int* block_tables [[buffer(5)]],
                                     constant AttnParams& p [[buffer(6)]],
                                     uint gid [[threadgroup_position_in_grid]],
                                     uint lid [[thread_position_in_threadgroup]]) {
  const uint sg = lid / 32u;
  const uint lane = lid % 32u;

  const uint head = gid % p.heads;
  const uint t0 = (gid / p.heads) * FA_QBLK;
  if (t0 >= p.tokens) return;

  uint base = p.position;
  if (p.use_position_buffer != 0u) base = uint(positions[0]);

  const uint group = p.heads / p.kv_heads;
  const uint kv_head = head / group;
  const uint kv_dim = p.kv_heads * p.head_dim;
  const uint q_dim = p.heads * p.head_dim;
  const uint hd = p.head_dim;
  const uint dfs = hd / 8u;

  // This simdgroup's own 8 queries. Nothing below is shared with any other simdgroup, so there is
  // no barrier and an early return here cannot strand anyone.
  const uint q0 = t0 + sg * 8u;
  if (q0 >= p.tokens) return;
  const uint nq = min(8u, p.tokens - q0);

  // This lane's fixed position in every fragment: one query row, two adjacent columns.
  const uint row = FA_ROW(lane);
  const uint col = FA_COL(lane);
  const bool row_live = (row < nq);
  const uint qtok = q0 + row;

  // Q straight into registers: each lane gathers exactly the two elements it owns of each
  // fragment. Padding rows read nothing and hold 0, so they cannot turn into NaN downstream.
  simdgroup_half8x8 Qf[16];
  for (uint df = 0u; df < dfs; ++df) {
    thread auto& e = Qf[df].thread_elements();
    const ulong qb = (ulong)qtok * (ulong)q_dim + (ulong)(head * hd + df * 8u + col);
    e[0] = row_live ? q[qb] : half(0);
    e[1] = row_live ? q[qb + 1u] : half(0);
  }

  simdgroup_float8x8 O[16];
  for (uint df = 0u; df < dfs; ++df) O[df] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  // The running max and sum of this lane's query, in registers, replicated across the four lanes
  // of the row by the butterfly reductions.
  float m_run = -INFINITY;
  float l_run = 0.0f;

  const uint pos_i = base + qtok;
  uint start_i = 0u;
  if (p.window != 0u && pos_i + 1u > p.window) start_i = pos_i + 1u - p.window;

  const uint last_pos = base + q0 + nq - 1u;
  const uint first_pos = base + q0;
  uint start = 0u;
  if (p.window != 0u && first_pos + 1u > p.window) start = first_pos + 1u - p.window;

  for (uint kb = start; kb <= last_pos; kb += FA_KB) {
    const uint nk = min((uint)FA_KB, last_pos - kb + 1u);
    const uint n_kg = (nk + 7u) / 8u;
    const uint kphys =
        (p.paged != 0u)
            ? uint(block_tables[kb / p.block_size]) * p.block_size + (kb % p.block_size)
            : kb;

    // --- scoring: S[g] = Q @ K^T, [8 queries x 8 keys], held in registers ---
    simdgroup_float8x8 S[FA_KG];
    for (uint g = 0u; g < FA_KG; ++g) S[g] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
#if !(FA_RCA & 1)
    for (uint df = 0u; df < dfs; ++df) {
      for (uint g = 0u; g < n_kg; ++g) {
        simdgroup_half8x8 Kf;
        simdgroup_load(Kf,
                       k_cache + (ulong)(kphys + g * 8u) * (ulong)kv_dim + kv_head * hd + df * 8u,
                       kv_dim, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(S[g], Qf[df], Kf, S[g]);
      }
    }
#endif

    // --- online softmax, entirely in registers ---
    // This lane owns keys (g*8 + col) and (g*8 + col + 1) of every group, all in its own row, so
    // scale and the causal/window mask fold in as it reads its own elements.
    float sv[FA_KG][2];
    float lmax = -INFINITY;
    for (uint g = 0u; g < n_kg; ++g) {
      thread auto& se = S[g].thread_elements();
      for (uint j = 0u; j < 2u; ++j) {
        const uint koff = g * 8u + col + j;
        const uint key = kb + koff;
        const bool keep = row_live && (koff < nk) && (key <= pos_i) && (key >= start_i);
        sv[g][j] = keep ? se[j] * p.scale : -INFINITY;
        lmax = max(lmax, sv[g][j]);
      }
    }
    const float new_max = max(m_run, fa_row_max(lmax));
    const float rescale = (m_run == -INFINITY) ? 0.0f : exp(m_run - new_max);

    // P written back into half fragments, same layout as the float ones, which is what lets the
    // matmul consume the softmax output without it ever reaching memory.
    simdgroup_half8x8 P[FA_KG];
    float lsum = 0.0f;
    for (uint g = 0u; g < n_kg; ++g) {
      thread auto& pe = P[g].thread_elements();
      for (uint j = 0u; j < 2u; ++j) {
        const float w =
            (sv[g][j] != -INFINITY && new_max != -INFINITY) ? exp(sv[g][j] - new_max) : 0.0f;
        pe[j] = half(w);  // 0 where masked, so P.V masks itself
        lsum += w;
      }
    }
    l_run = l_run * rescale + fa_row_sum(lsum);
    m_run = new_max;

    // The accumulator rescale is two multiplies on this lane's own elements.
    for (uint df = 0u; df < dfs; ++df) {
      thread auto& oe = O[df].thread_elements();
      oe[0] *= rescale;
      oe[1] *= rescale;
    }

    // --- P.V straight into the register accumulator ---
#if !(FA_RCA & 4)
    for (uint df = 0u; df < dfs; ++df) {
      for (uint g = 0u; g < n_kg; ++g) {
        simdgroup_half8x8 Vf;
        simdgroup_load(Vf,
                       v_cache + (ulong)(kphys + g * 8u) * (ulong)kv_dim + kv_head * hd + df * 8u,
                       kv_dim);
        simdgroup_multiply_accumulate(O[df], P[g], Vf, O[df]);
      }
    }
#endif
  }

  // Scatter out, per lane, bounds-checked; no staging buffer and no barrier.
  if (!row_live) return;
  const float inv = (l_run > 0.0f) ? 1.0f / l_run : 0.0f;
  for (uint df = 0u; df < dfs; ++df) {
    thread auto& oe = O[df].thread_elements();
    const ulong ob = (ulong)qtok * (ulong)q_dim + (ulong)(head * hd + df * 8u + col);
    out[ob] = half(oe[0] * inv);
    out[ob + 1u] = half(oe[1] * inv);
  }
}

// ---------------------------------------------------------------------------
// Query-partitioned prefill attention: the research rewrite.
//
// The kernel above is phase-partitioned: all 8 simdgroups cooperate on all QMM_BLOCK queries and
// re-partition between scoring / softmax / P.V, so every phase boundary pushes the intermediate
// state through threadgroup memory behind a threadgroup_barrier. Measured, that structure costs
// exactly what llama.cpp's flash-attention-OFF path costs (116 vs 117 ms at T=2048): it is the
// unfused algorithm in a fused kernel's clothing, and no amount of tiling, block-size or occupancy
// tuning moved it (all measured neutral).
//
// This one is query-partitioned, which is what actually makes flash attention fast: each simdgroup
// owns QP_QPS queries outright and carries them through scoring, softmax and accumulation itself.
// Nothing it writes is read by any other simdgroup, so every scratch array is a per-simdgroup slice
// and every barrier drops from threadgroup_barrier to simdgroup_barrier: ordering within one
// simdgroup, which is nearly free. There is no cross-simdgroup synchronisation in the key loop at
// all, and a simdgroup with no queries can return early without deadlocking the others.
//
// 4 simdgroups x 8 queries = 32 queries per threadgroup. Scratch is 7.5 KB per simdgroup (q, sc,
// pw, acc sized for head_dim <= 128), 30 KB total, just inside the 32 KB budget, which is why
// this uses 4 simdgroups and a 32-key block rather than the 8 and 128 the phase-partitioned kernel
// settled on. Selected by CPI_METAL_ATTN_QP=1; the proven kernel above stays the default.
//
// STATUS: correct (goldens pass) but 3.9x slower: 449 ms of attention at T=2042 vs the phase
// kernel's 116. The block size is the cause, not the decomposition: private per-simdgroup scratch
// multiplies the budget by NSG and forces QP_KB to 32 where the phase kernel runs 128. The online
// softmax runs once per (query, key block), so a 32-key block does 4x the softmax work, the same
// effect that made 32 -> 128 the largest win on the phase kernel. Hoisting query fragments out of
// the key-group loop (intensity 0.5 -> 0.8) was neutral, confirming the cost is softmax count.
//
// Making it competitive is a memory problem: the per-simdgroup scratch must fit a 128-key block,
// which means sizing the arrays to the real head_dim (dynamic threadgroup memory) and re-tuning NSG
// against QP_KB. Left behind the flag as parameter-tuning work rather than a rewrite.
#define QP_NSG  4                     // simdgroups per threadgroup
#define QP_QPS  8                     // queries per simdgroup == one 8x8 row fragment
#define QP_QBLK (QP_QPS * QP_NSG)     // 32 queries per threadgroup
// Keys per block, == the simd width, so lane == key with nothing to fold before the reduction. The
// cap is the threadgroup budget (per-simdgroup scratch times QP_NSG; q_sh and acc sized for
// head_dim 128 already spend ~30 KB of 32 KB). Widening it via dynamic head_dim sizing was tried
// and is monotonically worse (32 keys 743 ms, 64 764, 96 852) because threadgroup memory buys
// occupancy, not block width: at 96 keys only one threadgroup stays resident per core.
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

  // This simdgroup's own queries. Nothing below is shared with another simdgroup.
  const uint q0 = t0 + sg * QP_QPS;
  const uint nq = (q0 < p.tokens) ? min((uint)QP_QPS, p.tokens - q0) : 0u;

  threadgroup half q_sh[QP_NSG][QP_QPS * 128];
  threadgroup float acc[QP_NSG][QP_QPS * 128];
  threadgroup float sc[QP_NSG][QP_QPS * QP_KB];
  threadgroup half pw[QP_NSG][QP_QPS * QP_KB];
  threadgroup float m_sh[QP_NSG][QP_QPS];
  threadgroup float l_sh[QP_NSG][QP_QPS];

  // Zero the full QP_QPS rows: the matrix ops load 8-row fragments, so padding rows must be
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

    // Scoring: this simdgroup's 8 queries against each key group. The query fragments are the same
    // for every key group, so they are loaded once into registers and reused across all of them:
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
                                  device const int* limits [[buffer(5)]],
                                  constant AttnParams& p [[buffer(6)]],
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

  // 256 is a hard head_dim ceiling, NOT ATTN_MAX_HEAD_DIM: at Q_BLOCK 8 the 512-wide pair alone
  // is the whole 32 KB threadgroup budget. The executor routes head_dim > 256 to
  // cpi_attention_prefill_wide below; dispatching this kernel wider silently smashes acc and the
  // softmax scratch.
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
  // furthest, and the first one's window (if any) starts earliest. Per-token limits widen
  // that: a bidirectional span token's bound can sit past its causal position (its span's
  // keys are all written; KvStore appends the whole chunk before attention runs).
  const uint last_pos = base + t0 + nq - 1u;
  const uint first_pos = base + t0;
  uint start = 0u;
  if (p.window != 0u && first_pos + 1u > p.window) start = first_pos + 1u - p.window;
  uint end_total = last_pos + 1u;
  if (p.use_limits != 0u) {
    start = 0u;  // per-pair windows handle the start; the block loop must not clip them
    end_total = 0u;
    for (uint qi = 0u; qi < nq; ++qi) end_total = max(end_total, uint(limits[t0 + qi]));
  }

  for (uint kb = start; kb < end_total; kb += KEY_BLOCK) {
    const uint nk = min((uint)KEY_BLOCK, end_total - kb);

    // Score every (query, key) pair: one thread per pair, a full dot product with no
    // cross-lane reduction. The previous version gave each pair a whole simdgroup and summed
    // across 32 lanes: five shuffle steps of reduction overhead for ~two useful multiplies
    // per lane. nq*nk <= Q_BLOCK*KEY_BLOCK == nthr, so this is a single pass at full occupancy.
    for (uint pidx = lid; pidx < nq * nk; pidx += nthr) {
      const uint qi = pidx / nk, j = pidx % nk;
      const uint key = kb + j;
      const uint pos_i = base + t0 + qi;

      const uint end_i = (p.use_limits != 0u) ? uint(limits[t0 + qi]) : pos_i + 1u;
      uint start_i = 0u;
      if (p.window != 0u && end_i > p.window) start_i = end_i - p.window;

      float d = -INFINITY;
      if (key < end_i && key >= start_i) {  // causal / span mask, plus the sliding window
        device const half* kt = k_cache + (ulong)key * (ulong)kv_dim + kv_head * hd;
        float acc_d = 0.0f;
        for (uint i = 0u; i < hd; ++i) acc_d += q_sh[qi * hd + i] * float(kt[i]);
        d = acc_d * p.scale;
      }
      sc_sh[qi * KEY_BLOCK + j] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Fold the block into each query's own online softmax. One simdgroup per query, so all
    // 256 threads work; the old thread-per-query version left 8 threads busy and 248 idle at
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

// The head_dim > 256 copy of cpi_attention_prefill: same online softmax over KEY_BLOCK key
// blocks, arrays sized ATTN_MAX_HEAD_DIM with the query block halved to fit them (see
// Q_BLOCK_WIDE above). A separate kernel rather than a widening of the one above: threadgroup
// array sizes are compile-time, so one kernel would pay the 512 footprint (and its occupancy)
// on every model, and the 256 path is measured and tuned.
kernel void cpi_attention_prefill_wide(device const half* q [[buffer(0)]],
                                       device const half* k_cache [[buffer(1)]],
                                       device const half* v_cache [[buffer(2)]],
                                       device half* out [[buffer(3)]],
                                       device const int* positions [[buffer(4)]],
                                       device const int* limits [[buffer(5)]],
                                       constant AttnParams& p [[buffer(6)]],
                                       uint gid [[threadgroup_position_in_grid]],
                                       uint lid [[thread_position_in_threadgroup]],
                                       uint nthr [[threads_per_threadgroup]]) {
  const uint head = gid % p.heads;
  const uint t0 = (gid / p.heads) * Q_BLOCK_WIDE;
  if (t0 >= p.tokens) return;
  const uint nq = min((uint)Q_BLOCK_WIDE, p.tokens - t0);

  uint base = p.position;
  if (p.use_position_buffer != 0u) base = uint(positions[0]);

  const uint group = p.heads / p.kv_heads;
  const uint kv_head = head / group;
  const uint kv_dim = p.kv_heads * p.head_dim;
  const uint q_dim = p.heads * p.head_dim;
  const uint hd = p.head_dim;

  const uint simd_id = lid / 32u;
  const uint lane = lid % 32u;

  threadgroup float q_sh[Q_BLOCK_WIDE * ATTN_MAX_HEAD_DIM];
  threadgroup float acc[Q_BLOCK_WIDE * ATTN_MAX_HEAD_DIM];
  threadgroup float sc_sh[Q_BLOCK_WIDE * KEY_BLOCK];
  threadgroup float w_sh[Q_BLOCK_WIDE * KEY_BLOCK];
  threadgroup float m_sh[Q_BLOCK_WIDE];  // running max, per query
  threadgroup float l_sh[Q_BLOCK_WIDE];  // running sum, per query
  threadgroup float r_sh[Q_BLOCK_WIDE];  // this block's rescale, per query

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

  const uint last_pos = base + t0 + nq - 1u;
  const uint first_pos = base + t0;
  uint start = 0u;
  if (p.window != 0u && first_pos + 1u > p.window) start = first_pos + 1u - p.window;
  uint end_total = last_pos + 1u;
  if (p.use_limits != 0u) {
    start = 0u;
    end_total = 0u;
    for (uint qi = 0u; qi < nq; ++qi) end_total = max(end_total, uint(limits[t0 + qi]));
  }

  for (uint kb = start; kb < end_total; kb += KEY_BLOCK) {
    const uint nk = min((uint)KEY_BLOCK, end_total - kb);

    for (uint pidx = lid; pidx < nq * nk; pidx += nthr) {
      const uint qi = pidx / nk, j = pidx % nk;
      const uint key = kb + j;
      const uint pos_i = base + t0 + qi;

      const uint end_i = (p.use_limits != 0u) ? uint(limits[t0 + qi]) : pos_i + 1u;
      uint start_i = 0u;
      if (p.window != 0u && end_i > p.window) start_i = end_i - p.window;

      float d = -INFINITY;
      if (key < end_i && key >= start_i) {  // causal / span mask, plus the sliding window
        device const half* kt = k_cache + (ulong)key * (ulong)kv_dim + kv_head * hd;
        float acc_d = 0.0f;
        for (uint i = 0u; i < hd; ++i) acc_d += q_sh[qi * hd + i] * float(kt[i]);
        d = acc_d * p.scale;
      }
      sc_sh[qi * KEY_BLOCK + j] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

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
    device const int*   limits    [[buffer(5)]],
    constant AttnParams& p        [[buffer(6)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  // One threadgroup per (query token, head). Decode is just T = 1, so prefill and decode
  // share this kernel instead of duplicating the softmax and the GQA maths.
  //
  // Keys are processed in blocks. The first version walked the cache one key at a time,
  // with two threadgroup barriers per key: about 1100 barriers per (token, head) on a
  // 551-token prompt, which made prefill attention O(T^2) in barriers as well as in work.
  // Scoring a block of 32 keys and folding it into the online softmax once cuts the
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

  // Exclusive key bound: causal pos+1, or this token's limit (bidirectional image span).
  uint kend = pos + 1u;
  if (p.use_limits != 0u) kend = uint(limits[t]);
  uint start = 0u;
  if (p.window != 0u && kend > p.window) start = kend - p.window;

  device const half* qh = q + (ulong)t * (ulong)q_dim + head * p.head_dim;

  const uint simd_id = lid / 32u;
  const uint lane    = lid % 32u;
  const uint n_simd  = nthr / 32u;

  threadgroup float q_sh[ATTN_MAX_HEAD_DIM];    // the query, read once instead of per key
  threadgroup float sc_sh[KEY_BLOCK];
  threadgroup float w_sh[KEY_BLOCK];
  threadgroup float tg_acc[ATTN_MAX_HEAD_DIM];
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

  for (uint kb = start; kb < kend; kb += KEY_BLOCK) {
    const uint nk = min((uint)KEY_BLOCK, kend - kb);

    // Score the whole block: each simdgroup takes a key and reduces its dot product.
    for (uint j = simd_id; j < nk; j += n_simd) {
      device const half* kt = k_cache + (ulong)(kb + j) * (ulong)kv_dim + kv_head * p.head_dim;
      float d = 0.0f;
      {
        // half4 K loads + float4 dots; head_dim % 4 == 0 on every model here.
        threadgroup const float4* q4 = (threadgroup const float4*)q_sh;
        device const half4* kt4 = (device const half4*)kt;
        const uint hd4 = p.head_dim >> 2u;
        for (uint i = lane; i < hd4; i += 32u) d += dot(q4[i], float4(kt4[i]));
      }
      d = simd_sum(d);
      if (lane == 0u) sc_sh[j] = d * p.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Online softmax, folded in once for the block rather than once per key.
    float bmax = -INFINITY;
    for (uint j = 0u; j < nk; ++j) bmax = max(bmax, sc_sh[j]);

    const float old_max = tg_max;
    const float new_max = max(old_max, bmax);
    const float rescale = (old_max == -INFINITY) ? 0.0f : exp(old_max - new_max);

    for (uint j = lid; j < nk; j += nthr) w_sh[j] = exp(sc_sh[j] - new_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float bsum = 0.0f;
    for (uint j = 0u; j < nk; ++j) bsum += w_sh[j];

    {
      // V accumulate, float4 over dims: 4x fewer strided loads per thread.
      threadgroup float4* acc4 = (threadgroup float4*)tg_acc;
      device const half4* vbase = (device const half4*)(v_cache + kv_head * p.head_dim);
      const uint hd4v = p.head_dim >> 2u;
      const uint kvd4 = kv_dim >> 2u;
      for (uint i4 = lid; i4 < hd4v; i4 += nthr) {
        float4 a = acc4[i4] * rescale;
        for (uint j = 0u; j < nk; ++j) {
          a += w_sh[j] * float4(vbase[(ulong)(kb + j) * (ulong)kvd4 + i4]);
        }
        acc4[i4] = a;
      }
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
// cache inside it. For a prefill that is fine: T tokens x heads is thousands of threadgroups.
// At decode T is 1, so the grid is just `heads`: 14 threadgroups for Qwen2.5-0.5B, on a GPU
// that wants hundreds. The cache walk becomes a serial loop in a nearly empty machine, so
// decode slows down with context for a reason that has nothing to do with bandwidth: at 2048
// keys the KV cache is ~25 MB against 1.17 GB of weights, 2% of the traffic for a 2.6x
// slowdown (88 -> 34 tok/s, where llama.cpp holds 96 -> 89).
//
// So split the keys. Pass 1 gives each (head, chunk) its own threadgroup and an independent
// online softmax over its slice, writing that slice's running max, running sum and an
// unnormalized accumulator. Pass 2 merges the slices per head.
//
// The merge is exact, not an approximation: softmax combines under the standard log-sum-exp
// rule, so rescaling each slice by exp(m_c - m) and summing reproduces the single-threadgroup
// result. It is the same decomposition the CUDA split-K decode path uses
// (kernels_attention_decode.cu), on purpose: same statistics, same merge, so the backends
// cannot drift and one golden covers both.
// ---------------------------------------------------------------------------

// Keys per pass of the decode scoring loop, and the threadgroup width that serves it. They are
// the same number on purpose: the loop gives each thread one whole key, so the 64-element dot
// product stays in one thread and there is no cross-lane reduction at all.
//
// The kernel this replaced split each key's dot product across a simdgroup: 32 lanes doing 2
// MACs each, then a 5-step shuffle reduction, per key. That is two MACs of work per reduction and
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

  threadgroup float q_sh[ATTN_MAX_HEAD_DIM];
  threadgroup float sc_sh[DEC_KEY_BLOCK];
  threadgroup float w_sh[DEC_KEY_BLOCK];
  threadgroup float tg_acc[ATTN_MAX_HEAD_DIM];
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

  // V-accumulate work split. Sweeping dims alone leaves most threads idle whenever
  // head_dim/4 < nthr (16 of 64 for head_dim 64), so 75% of the threadgroup sat out the V pass
  // while it walked every key serially. When the dims fit the threadgroup, give each thread a
  // (dim, key-group) pair and keep its partial in a register: every thread works, reads stay
  // contiguous within each key, and the key-group partials are combined once at the end instead
  // of round-tripping tg_acc every block. Wider heads keep the original sweep.
  const uint hd4v = p.head_dim >> 2u;
  const uint kvd4 = kv_dim >> 2u;
  const bool vsplit = (hd4v > 0u) && (hd4v <= nthr) && ((nthr % hd4v) == 0u);
  const uint vDS = vsplit ? hd4v : nthr;         // threads sweeping dims
  const uint vKG = vsplit ? (nthr / hd4v) : 1u;  // key groups running in parallel
  const uint vdim = lid % vDS;
  const uint vkg = lid / vDS;
  float4 vacc = float4(0.0f);

  for (uint kb = c0; kb < c1; kb += DEC_KEY_BLOCK) {
    const uint nk = min((uint)DEC_KEY_BLOCK, c1 - kb);

    // Score: one key per thread, half4 K loads + float4 dots. head_dim % 4 == 0 on
    // every model here.
    {
      threadgroup const float4* q4 = (threadgroup const float4*)q_sh;
      const uint hd4 = p.head_dim >> 2u;
      for (uint j = lid; j < nk; j += nthr) {
        device const half4* kt4 = (device const half4*)(k_cache +
                                                        (ulong)(kb + j) * (ulong)kv_dim +
                                                        kv_head * p.head_dim);
        float d = 0.0f;
        for (uint i = 0u; i < hd4; ++i) d += dot(q4[i], float4(kt4[i]));
        sc_sh[j] = d * p.scale;
      }
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

    {
      // V accumulate, float4 over dims: 4x fewer strided loads per thread.
      device const half4* vbase = (device const half4*)(v_cache + kv_head * p.head_dim);
      if (vsplit) {
        // Every thread owns (vdim, vkg) and walks only its share of the keys; the partial
        // stays in a register until the combine after the block loop.
        vacc *= rescale;
        for (uint j = vkg; j < nk; j += vKG) {
          vacc += w_sh[j] * float4(vbase[(ulong)(kb + j) * (ulong)kvd4 + vdim]);
        }
      } else {
        threadgroup float4* acc4 = (threadgroup float4*)tg_acc;
        for (uint i4 = lid; i4 < hd4v; i4 += nthr) {
          float4 a = acc4[i4] * rescale;
          for (uint j = 0u; j < nk; ++j) {
            a += w_sh[j] * float4(vbase[(ulong)(kb + j) * (ulong)kvd4 + i4]);
          }
          acc4[i4] = a;
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
      tg_sum = tg_sum * rescale + bsum;
      tg_max = new_max;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Fold the per-thread key-group partials into tg_acc, one group at a time. tg_acc was zeroed
  // up front and left untouched by the split path, so this is the only writer. vKG barriers
  // total (not per block), against a V pass that now runs on the whole threadgroup.
  if (vsplit) {
    threadgroup float4* acc4 = (threadgroup float4*)tg_acc;
    for (uint r = 0u; r < vKG; ++r) {
      if (vkg == r) acc4[vdim] += vacc;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
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

// GQA-shared split decode: one threadgroup per key chunk, one simdgroup per q-head, the
// K and V blocks staged in threadgroup memory once for all heads (the per-(head,chunk)
// kernel re-reads the shared kv_head per q-head). Opt-in: the staging barriers measured
// costlier than the traffic they save. Per-head softmax state lives in simd registers:
// lane j owns key (kb + j), online max/sum via simd reductions, the V pass broadcasts
// each weight from its lane. The merge kernel is shared unchanged.
kernel void cpi_attention_decode_split_gqa(
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
  const uint chunk = gid;
  if (chunk >= p.chunks) return;

  uint pos = p.position;
  if (p.use_position_buffer != 0u) pos = uint(positions[0]);
  uint start = 0u;
  if (p.window != 0u && pos + 1u > p.window) start = pos + 1u - p.window;

  const uint c0 = start + chunk * p.chunk_size;
  const uint c1 = min(c0 + p.chunk_size, pos + 1u);

  const uint simd_id = lid / 32u;
  const uint lane    = lid % 32u;
  const uint head    = simd_id;  // one simdgroup per q-head (heads == simds, checked host-side)
  const uint kv_dim  = p.kv_heads * p.head_dim;

  // All heads share kv_head 0 (executor gates on kv_heads == 1).
  if (c0 >= c1) {
    if (head < p.heads && lane == 0u) {
      part_m[head * p.chunks + chunk] = -INFINITY;
      part_l[head * p.chunks + chunk] = 0.0f;
    }
    return;
  }

  // 16 KB staging block, reused for K then V each key-block. 32 keys at head_dim 256;
  // halve the block for 512-wide heads.
  const uint kblk = (p.head_dim <= 256u) ? 32u : 16u;
  threadgroup half kv_sh[32u * 256u];

  // Stage this head's query into registers, float4-wide: hd <= 512 -> <= 4 float4 per lane.
  float4 qreg[4];
  const uint hd4 = p.head_dim >> 2u;
  const uint q4count = (hd4 + 31u) / 32u;
  if (head < p.heads) {
    device const half4* q4p = (device const half4*)(q + head * p.head_dim);
    for (uint c = 0u; c < q4count; ++c) {
      const uint i4 = lane + c * 32u;
      qreg[c] = (i4 < hd4) ? float4(q4p[i4]) : float4(0.0f);
    }
  }

  float run_m = -INFINITY;
  float run_l = 0.0f;
  float acc[16];  // head_dim/32 accumulators per lane, <= 16 at hd 512
  const uint npl = p.head_dim / 32u;  // dims per lane (head_dim % 32 == 0 on every model)
  for (uint i = 0u; i < npl; ++i) acc[i] = 0.0f;

  for (uint kb = c0; kb < c1; kb += kblk) {
    const uint nk = min(kblk, c1 - kb);

    // Stage K block once for every head.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint idx = lid; idx < nk * p.head_dim; idx += nthr) {
      const uint j = idx / p.head_dim;
      const uint i = idx % p.head_dim;
      kv_sh[idx] = k_cache[(ulong)(kb + j) * (ulong)kv_dim + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Lane j scores key (kb + j) for this simd's head, from the staged block.
    float score = -INFINITY;
    if (head < p.heads && lane < nk) {
      threadgroup const half* kt = kv_sh + lane * p.head_dim;
      float d = 0.0f;
      for (uint c = 0u; c < q4count; ++c) {
        for (uint l4 = 0u; l4 < 32u && (l4 + c * 32u) < hd4; ++l4) {
          // dot against qreg held lane-strided: broadcast each lane's q fragment.
          const float4 qf = simd_shuffle(qreg[c], l4);
          const uint i0 = (l4 + c * 32u) * 4u;
          d += qf.x * float(kt[i0]) + qf.y * float(kt[i0 + 1u]) + qf.z * float(kt[i0 + 2u]) +
               qf.w * float(kt[i0 + 3u]);
        }
      }
      score = d * p.scale;
    }

    // Per-head online softmax in simd registers.
    const float bmax = simd_max(score);
    if (head < p.heads && bmax != -INFINITY) {
      const float new_m = max(run_m, bmax);
      const float rescale = (run_m == -INFINITY) ? 0.0f : exp(run_m - new_m);
      const float w = (lane < nk) ? exp(score - new_m) : 0.0f;
      const float bsum = simd_sum(w);

      // Stage V over the same buffer: everyone is past the K reads by the barrier below.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint idx = lid; idx < nk * p.head_dim; idx += nthr) {
        const uint j = idx / p.head_dim;
        const uint i = idx % p.head_dim;
        kv_sh[idx] = v_cache[(ulong)(kb + j) * (ulong)kv_dim + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (uint i = 0u; i < npl; ++i) acc[i] *= rescale;
      for (uint j = 0u; j < nk; ++j) {
        const float wj = simd_shuffle(w, j);
        threadgroup const half* vt = kv_sh + j * p.head_dim;
        for (uint i = 0u; i < npl; ++i) {
          acc[i] += wj * float(vt[lane + i * 32u]);
        }
      }
      run_l = run_l * rescale + bsum;
      run_m = new_m;
    } else {
      // Heads beyond p.heads (or empty scores) still participate in the V restage barriers.
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint idx = lid; idx < nk * p.head_dim; idx += nthr) {
        const uint j = idx / p.head_dim;
        const uint i = idx % p.head_dim;
        kv_sh[idx] = v_cache[(ulong)(kb + j) * (ulong)kv_dim + i];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  if (head < p.heads) {
    const uint slot = head * p.chunks + chunk;
    if (lane == 0u) {
      part_m[slot] = run_m;
      part_l[slot] = run_l;
    }
    for (uint i = 0u; i < npl; ++i) {
      part_o[(ulong)slot * (ulong)p.head_dim + lane + i * 32u] = acc[i];
    }
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
// Batched paged decode: the kernels continuous batching needs.
//
// The kernels above serve one sequence, whose KV is contiguous: the key at logical
// position p lives at p * kv_dim. Continuous batching cannot use that layout, because the
// sequences in a batch start, grow and finish independently; a contiguous per-sequence
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
// blocks outright, which is why the table is a gather rather than a base offset.
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
    device const int*   positions    [[buffer(5)]],  // [batch]: each row's own position
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

// One threadgroup per (sequence, head). Each sequence attends over its own length, gathered
// through its own block table: the ragged batch the scheduler hands us. Structurally this
// is cpi_attention_decode with the contiguous key address replaced by a paged gather; the
// online softmax is unchanged, so the two agree key-for-key when a block table happens to be
// identity (which is what the parity gate checks).
kernel void cpi_attention_decode_batched_paged(
    device const half*  q            [[buffer(0)]],  // [batch][heads*head_dim]
    device const half*  k_pool       [[buffer(1)]],
    device const half*  v_pool       [[buffer(2)]],
    device half*        out          [[buffer(3)]],  // [batch][heads*head_dim]
    device const int*   block_tables [[buffer(4)]],  // [batch][max_blocks]
    device const int*   seq_lens     [[buffer(5)]],  // [batch]: length including the new token
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

  threadgroup float q_sh[ATTN_MAX_HEAD_DIM];
  threadgroup float sc_sh[KEY_BLOCK];
  threadgroup float w_sh[KEY_BLOCK];
  threadgroup float tg_acc[ATTN_MAX_HEAD_DIM];
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
      {
        // half4 K loads + float4 dots; head_dim % 4 == 0 on every model here.
        threadgroup const float4* q4 = (threadgroup const float4*)q_sh;
        device const half4* kt4 = (device const half4*)kt;
        const uint hd4 = p.head_dim >> 2u;
        for (uint i = lane; i < hd4; i += 32u) d += dot(q4[i], float4(kt4[i]));
      }
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

    {
      // V accumulate, float4 over dims, physical row resolved once per key.
      threadgroup float4* acc4 = (threadgroup float4*)tg_acc;
      const uint hd4v = p.head_dim >> 2u;
      for (uint i4 = lid; i4 < hd4v; i4 += nthr) {
        float4 a = acc4[i4] * rescale;
        for (uint j = 0u; j < nk; ++j) {
          const uint kpos = kb + j;
          const uint phys = uint(bt[kpos / p.block_size]) * p.block_size + (kpos % p.block_size);
          device const half4* vt4 = (device const half4*)(v_pool + (ulong)phys * (ulong)kv_dim +
                                                          kv_head * p.head_dim);
          a += w_sh[j] * float4(vt4[i4]);
        }
        acc4[i4] = a;
      }
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
// The selection crosses ops in device buffers, so a token never round-trips to the host
// mid-layer. These mirror PlanCudaEngine's kernels; the gate is reproducing the CUDA token
// stream on tiny-mixtral, so a deviation here is a bug even where it looks like an
// improvement.
// ---------------------------------------------------------------------------


// ---------------------------------------------------------------------------
// Bidirectional (non-causal) attention, for a vision tower.
//
// A separate kernel rather than a mask mode on the causal ones above. Those four are tuned
// against a measured roofline and are on the path of every text model; threading a
// "bidirectional" branch through them would put a vision feature's regressions into the text
// engine. The shapes differ too: no KV cache, no paging, no sliding window, no GQA, and a
// sequence that is a patch grid rather than a growing context.
//
// One threadgroup per (query token, head). Keys are consumed in chunks of nthr with an online
// softmax, so nothing scales with sequence length in threadgroup memory and there is no cap on
// the patch count.
//
// Deliberately NOT tuned. It is correct and readable first; the tower has to match the oracle
// before any of its cost is worth measuring, and the causal kernels' history says the wins are
// not where they look.
// ---------------------------------------------------------------------------
kernel void cpi_attention_bidirectional(
    device const half* q   [[buffer(0)]],
    device const half* k   [[buffer(1)]],
    device const half* v   [[buffer(2)]],
    device half*       out [[buffer(3)]],
    constant BiAttnParams& p [[buffer(4)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  const uint total = p.tokens * p.heads;
  if (gid >= total) return;
  const uint t    = gid / p.heads;
  const uint head = gid % p.heads;
  // The row stride is NOT heads*head_dim in general: a fused qkv projection leaves q, k and v
  // interleaved in one buffer with a stride of 3*hidden, and each is addressed by its own base
  // offset. Assuming the packed stride reads q's row t from wherever k's row t/3 happens to be:
  // plausible numbers, wrong attention.
  const uint dim  = (p.row_stride != 0u) ? p.row_stride : (p.heads * p.head_dim);
  // The output is its own packed [tokens][heads*head_dim] buffer, so it never carries the
  // fused stride. Using `dim` for both (which is the natural way to write this) scatters
  // each token's result three rows apart and leaves two thirds of the buffer untouched.
  const uint out_dim = p.heads * p.head_dim;
  const uint hoff = head * p.head_dim;

  device const half* qh = q + (ulong)t * dim + hoff;

  threadgroup float sc[BIATTN_CHUNK];
  threadgroup float red[BIATTN_CHUNK / 32];
  threadgroup float acc[BIATTN_MAXDIM];   // running unnormalised output
  // [2] holds the max from before this chunk: the rescale is exp(prev_max - run_max), and reading
  // an already-updated stats[0] makes that exp(0) == 1, dropping the correction when the max grows.
  threadgroup float stats[3];             // [0] running max, [1] running sum, [2] previous max

  for (uint d = lid; d < p.head_dim; d += nthr) acc[d] = 0.0f;
  if (lid == 0u) { stats[0] = -INFINITY; stats[1] = 0.0f; stats[2] = -INFINITY; }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (uint chunk = 0u; chunk < p.tokens; chunk += nthr) {
    const uint n = min(nthr, p.tokens - chunk);

    // ---- scores for this chunk: one key per thread, full dot over head_dim ----
    if (lid < n) {
      device const half* kh = k + (ulong)(chunk + lid) * dim + hoff;
      float dot = 0.0f;
      for (uint d = 0u; d < p.head_dim; ++d) dot += float(qh[d]) * float(kh[d]);
      sc[lid] = dot * p.scale;   // no mask: every query sees every key, both directions
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- chunk max ----
    float m = (lid < n) ? sc[lid] : -INFINITY;
    m = simd_max(m);
    if ((lid % 32u) == 0u) red[lid / 32u] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
      float cm = -INFINITY;
      for (uint s = 0u; s < (n + 31u) / 32u; ++s) cm = max(cm, red[s]);
      stats[2] = stats[0];            // keep the old max: the rescale below needs it
      stats[0] = max(stats[0], cm);   // running max across chunks, not just this one
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float run_max = stats[0];
    // On the first chunk prev_max is -INFINITY, so this is exp(-inf) == 0, which is the
    // right factor, since acc and the running sum are both still zero.
    const float rescale = exp(stats[2] - run_max);

    // ---- exponentiate against the running max ----
    if (lid < n) sc[lid] = exp(sc[lid] - run_max);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float s = (lid < n) ? sc[lid] : 0.0f;
    s = simd_sum(s);
    if ((lid % 32u) == 0u) red[lid / 32u] = s;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // The running max can only grow, so everything accumulated so far was exponentiated
    // against a smaller max and is scaled too high by exp(prev_max - run_max). Rescale the
    // accumulator and the running sum by that factor before folding this chunk in.
    if (lid == 0u) {
      float cs = 0.0f;
      for (uint sg = 0u; sg < (n + 31u) / 32u; ++sg) cs += red[sg];
      red[0] = cs;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float chunk_sum = red[0];

    if (lid == 0u) stats[1] = stats[1] * rescale + chunk_sum;
    for (uint d = lid; d < p.head_dim; d += nthr) acc[d] *= rescale;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ---- fold this chunk's values in ----
    for (uint d = lid; d < p.head_dim; d += nthr) {
      float a = 0.0f;
      for (uint j = 0u; j < n; ++j) {
        a += sc[j] * float(v[(ulong)(chunk + j) * dim + hoff + d]);
      }
      acc[d] += a;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  const float inv = 1.0f / (stats[1] + 1e-20f);
  for (uint d = lid; d < p.head_dim; d += nthr) {
    out[(ulong)t * out_dim + hoff + d] = half(acc[d] * inv);
  }
}
