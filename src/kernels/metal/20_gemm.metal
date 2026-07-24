#define GEMM_BM 64  // rows per threadgroup
// Tokens per threadgroup. Like the K-depth, the two GEMMs want DIFFERENT widths, measured on
// the 0.5B fp16 prefill and the int4:
//
//   fp16  (GEMM_BN 32): narrow is better, and further than first measured. This kernel STAGES
//     the activation tile in threadgroup memory, so a wider tile costs more threadgroup memory
//     and drops occupancy; 128 measured 2017 -> 1663 tok/s. 32 is better again, because the tile
//     width also sets how many threadgroups the grid has: out_dim/GEMM_FBM * ceil(T/GEMM_BN), and
//     at short and middling prompts that product is small enough to leave the GPU idle. Halving
//     the tile doubles it. Interleaved A/B, three rounds each:
//       T=137  77 -> 66 ms     T=513  168 -> 157 ms     T=1027  300 -> 293 ms
//       T=273 106 -> 100 ms    T=2041 593 -> 593 ms (neutral -- by then the grid is full anyway)
//     Measured with the smoke test's mirror of this constant corrected: at 64 it disagreed with
//     the shader and every gemm_f16 check failed, which made a correct kernel look broken.
//   quant (GEMM_QBN 128): wide is better. The quant kernel reads activation fragments straight
//     from device (no As tile), so widening only adds simdgroups and weight reuse -- the staged
//     weight block now serves 128 tokens -- at the SAME threadgroup memory. 64 -> 128 measured
//     1527 -> 1649 tok/s.
#define GEMM_BN 32
#define GEMM_QBN 128
// Simdgroups tile the 64-row output as a grid, each owning a 32x32 sub-tile (4x4 fragments).
// Fragments of 8x8 each simdgroup owns: GEMM_RF rows x GEMM_CF cols. This sets arithmetic
// intensity: per 8-deep step a simdgroup issues (RF + CF) simdgroup_loads to do (RF * CF) matrix
// ops, so 4x4 is 8 loads per 16 ops (2.0) and 8x4 would be 12 per 32 (2.67). Everything else --
// thread count, simdgroup grid, row offsets -- DERIVES from these two, so the tile can be swept
// without a hardcoded number drifting out of step, which is how this kernel once came to write
// half its rows.
//
// 4x4 is the ceiling, bounded by the register file. Measured with metal_gemm_bench:
//
//   RF=4, FBM=64  (32x32/simdgroup, 16 accumulator fragments)   2.90 TFLOP/s
//   RF=8, FBM=128 (64x32/simdgroup, 32 accumulator fragments)   0.38 TFLOP/s   <- 7.6x SLOWER
//
// 32 accumulator fragments is 64 float registers per lane before wf/af, and it spills. Holding
// more of the output per thread to raise intensity trades fewer loads for spill traffic at a
// ruinous rate; more simdgroups also measured slower once the geometry was consistent.
#define GEMM_RF 4
#define GEMM_CF 4

// The fp16 GEMM's ACCUMULATOR type. 0 = fp32 (simdgroup_float8x8), 1 = fp16.
//
// This WAS the last open perf question, now closed by re-measurement. Xcode's counters show the
// kernel F32-limited while the F16 pipe sits at 0.00% -- every MAC issues on F32 because the
// accumulator is float -- and Apple's F16 ALU pipe is ~2x the F32 rate, so the hope was that half
// accumulators route there and roughly double the kernel.
//
// They do not. Flipped to 1 and benched honestly (spot-checked, warm, current harness): gate/up
// 3.20 -> 0.81 TFLOP/s, aggregate 2.98 -> 0.74 -- a 4x REGRESSION, which confirms the old note
// that had been distrusted for dating from the bad-measurement era. The 0% F16 utilization is not
// headroom: the matrix unit (simdgroup_multiply_accumulate) accumulates in fp32 NATIVELY, and
// asking for a half accumulator does not move the MAC onto the F16 pipe -- it emulates fp16
// accumulation on the F32 unit, which is strictly more work. The F16 pipe is reachable by scalar
// half math, not by the matrix unit, and a GEMM has to live on the matrix unit. (Precision was
// fine -- the golden still passed -- so this is a pure throughput loss, not an accuracy one.)
#define GEMM_ACC_HALF 0

#if GEMM_ACC_HALF
typedef simdgroup_half8x8 gemm_acc_t;
typedef half gemm_acc_e;
#else
typedef simdgroup_float8x8 gemm_acc_t;
typedef float gemm_acc_e;
#endif
#define GEMM_SG_ROWS (8u * GEMM_RF)  // rows per simdgroup
#define GEMM_SG_TOKS (8u * GEMM_CF)  // tokens per simdgroup
#define GEMM_SG_COLS (GEMM_BN / GEMM_SG_TOKS)
#define GEMM_QSG_COLS (GEMM_QBN / 32)
// K per stage: how much arithmetic each barrier buys, since the fill and the matrix ops are
// separated by barriers.
//
//   fp16  (GEMM_FBK 32): depth does not matter -- 16/32/64 measure 2.85/2.89/2.86 TFLOP/s.
//   quant (GEMM_QBK 32): deeper is worse. The quant tile also holds a scale row, so doubling K
//     doubles its threadgroup memory, and the 8B's GEMM is compute-bound: it needs the occupancy
//     to hide latency more than it needs fewer barriers. Measured 161 -> 148 tok/s at 32 -> 64.
//
// GEMM_QBK must not exceed the quantization group (64 by default), or a K-block straddles two
// scales; the dispatch guards that.
// GEMM_FBM is the fp16 row tile and MUST stay in step with the host's kGemmFBM, which derives both
// the thread count (one simdgroup per 32x32 sub-tile of FBM x BN) and the grid (out_dim / FBM row
// blocks) from it. 128 rows over 8 simdgroups measured ~3% slower than 64 over 4. Both are checked
// against a CPU reference in metal_smoke, which derives its dispatch from these same constants.
#define GEMM_QBK 32
#define GEMM_FBM 64
#define GEMM_FBK 32
// Reduction splits for the split-K prototype below. Not used by the engine.
#define GEMM_SPLITK 4

// ---------------------------------------------------------------------------
// Blocked fp16 GEMM (prefill). A K-block of both operands is staged in threadgroup memory,
// so a weight loaded from device is reused by every token in the tile. Two earlier attempts
// streamed the operands from device on every K-step and LOST to the plain GEMV: the matrix
// units were never the bottleneck, the traffic feeding them was.
//
// REGISTER TILING is the second half of that same idea. With one 8x64 strip per simdgroup
// the inner loop issued nine fragment loads to do eight matrix ops -- starved on
// threadgroup traffic instead of device traffic. Each simdgroup now owns a 32x32 output
// tile (4x4 fragments), so the same eight loads feed sixteen matrix ops. The threadgroup is
// 128 threads rather than 256: fewer threads each holding more accumulators is the point.
// ---------------------------------------------------------------------------
kernel void cpi_gemm_f16(device const half* w [[buffer(0)]], device const half* in [[buffer(1)]],
                         device half* out [[buffer(2)]], device const half* bias [[buffer(3)]],
                         constant GemvParams& p [[buffer(4)]],
                         uint tgid [[threadgroup_position_in_grid]],
                         uint lid [[thread_position_in_threadgroup]],
                         uint nthr [[threads_per_threadgroup]]) {
  const uint sgid = lid / 32u;
  const uint lane = lid % 32u;
  const uint sg_row = sgid / GEMM_SG_COLS;
  const uint sg_col = sgid % GEMM_SG_COLS;

  const uint row_blocks = p.out_dim / GEMM_FBM;
  const uint row0 = (tgid % row_blocks) * GEMM_FBM;
  const uint tok0 = (tgid / row_blocks) * GEMM_BN;
  if (tok0 >= p.tokens) return;

  threadgroup half Ws[GEMM_FBM * GEMM_FBK];
  threadgroup half As[GEMM_BN * GEMM_FBK];

  gemm_acc_t acc[GEMM_RF][GEMM_CF];
  for (uint i = 0u; i < GEMM_RF; ++i)
    for (uint j = 0u; j < GEMM_CF; ++j)
      acc[i][j] = make_filled_simdgroup_matrix<gemm_acc_e, 8, 8>(gemm_acc_e(0));

  const uint chunks = GEMM_FBK / 8u;  // 8 halves == one 128-bit load

  for (uint k0 = 0u; k0 < p.in_dim; k0 += GEMM_FBK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = lid; c < GEMM_FBM * chunks; c += nthr) {
      const uint r = c / chunks, sub = c % chunks;
      *(threadgroup uint4*)(Ws + r * GEMM_FBK + sub * 8u) =
          *(device const uint4*)(w + (ulong)(row0 + r) * (ulong)p.in_dim + k0 + sub * 8u);
    }
    for (uint c = lid; c < GEMM_BN * chunks; c += nthr) {
      const uint t = c / chunks, sub = c % chunks;
      threadgroup uint4* dst = (threadgroup uint4*)(As + t * GEMM_FBK + sub * 8u);
      if (tok0 + t < p.tokens) {
        *dst = *(device const uint4*)(in + (ulong)(tok0 + t) * (ulong)p.in_dim + k0 + sub * 8u);
      } else {
        *dst = uint4(0u);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0u; kk < GEMM_FBK; kk += 8u) {
      simdgroup_half8x8 wf[GEMM_RF], af[GEMM_CF];
      // W is [rows, cols], so a weight fragment loads TRANSPOSED: [8k x 8rows].
      for (uint i = 0u; i < GEMM_RF; ++i)
        simdgroup_load(wf[i], Ws + (sg_row * GEMM_SG_ROWS + i * 8u) * GEMM_FBK + kk, GEMM_FBK,
                       ulong2(0, 0), true);
      for (uint j = 0u; j < GEMM_CF; ++j)
        simdgroup_load(af[j], As + (sg_col * GEMM_SG_TOKS + j * 8u) * GEMM_FBK + kk, GEMM_FBK);
      for (uint i = 0u; i < GEMM_RF; ++i)
        for (uint j = 0u; j < GEMM_CF; ++j)
          simdgroup_multiply_accumulate(acc[i][j], af[j], wf[i], acc[i][j]);
    }
  }

  // The K-loop is done with the weight tile; reuse it to stage the accumulators out.
  threadgroup gemm_acc_e* mine = (threadgroup gemm_acc_e*)Ws + sgid * 64u;
  for (uint i = 0u; i < GEMM_RF; ++i) {
    for (uint j = 0u; j < GEMM_CF; ++j) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_store(acc[i][j], mine, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 64u; e += 32u) {
        const uint t = e / 8u;  // token within the fragment
        const uint r = e % 8u;  // row within the fragment
        const uint tok = tok0 + sg_col * GEMM_SG_TOKS + j * 8u + t;
        if (tok >= p.tokens) continue;
        const uint row = row0 + sg_row * GEMM_SG_ROWS + i * 8u + r;
        float v = float(mine[t * 8u + r]);
        if (p.has_bias != 0u) v += float(bias[row]);
        out[(ulong)tok * (ulong)p.out_dim + row] = half(v);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// SPLIT-K prototype of the fp16 GEMM.
//
// At a short prompt the grid is out_dim/GEMM_FBM * ceil(T/GEMM_BN) threadgroups, which for this
// model's 896-row projections is 28 at T=64 -- on a 10-core GPU. down_proj is the worst of them:
// same FLOPs and same bytes as gate_proj, but a 14x76 grid instead of 76x14, and it measures 2.2x
// slower. Shrinking the row tile to make more threadgroups was tried and LOST (GEMM_FBM 32: 0.49
// -> 0.55 ms), because it also halves how many output rows each staged weight tile serves and so
// doubles the weight traffic.
//
// Splitting the REDUCTION does not have that problem: every threadgroup reads a different slice of
// K, so the total weight bytes are unchanged and only the activation tile is re-read. Each
// threadgroup writes a float partial, and a second pass sums them.
//
// Cost is the partial buffer: GEMM_SPLITK * tokens * out_dim floats, plus one extra read+write of
// it. At T=64 and out_dim 896 with 4 splits that is ~917 KB, a fraction of a millisecond.
// ---------------------------------------------------------------------------
kernel void cpi_gemm_f16_splitk(device const half* w [[buffer(0)]],
                                device const half* in [[buffer(1)]],
                                device float* partial [[buffer(2)]],
                                constant GemvParams& p [[buffer(3)]],
                                uint tgid [[threadgroup_position_in_grid]],
                                uint lid [[thread_position_in_threadgroup]],
                                uint nthr [[threads_per_threadgroup]]) {
  const uint sgid = lid / 32u;
  const uint lane = lid % 32u;
  const uint sg_row = sgid / GEMM_SG_COLS;
  const uint sg_col = sgid % GEMM_SG_COLS;

  const uint row_blocks = p.out_dim / GEMM_FBM;
  const uint tok_tiles = (p.tokens + GEMM_BN - 1u) / GEMM_BN;
  const uint per_split = row_blocks * tok_tiles;

  const uint split = tgid / per_split;          // which slice of K this threadgroup reduces
  const uint within = tgid % per_split;
  const uint row0 = (within % row_blocks) * GEMM_FBM;
  const uint tok0 = (within / row_blocks) * GEMM_BN;
  if (tok0 >= p.tokens) return;

  // K range for this split, snapped to whole GEMM_FBK blocks so the staging loop is unchanged.
  const uint kblocks = (p.in_dim + GEMM_FBK - 1u) / GEMM_FBK;
  const uint kb_per = (kblocks + GEMM_SPLITK - 1u) / GEMM_SPLITK;
  const uint k_begin = split * kb_per * GEMM_FBK;
  const uint k_end = min(p.in_dim, (split + 1u) * kb_per * GEMM_FBK);

  threadgroup half Ws[GEMM_FBM * GEMM_FBK];
  threadgroup half As[GEMM_BN * GEMM_FBK];

  gemm_acc_t acc[GEMM_RF][GEMM_CF];
  for (uint i = 0u; i < GEMM_RF; ++i)
    for (uint j = 0u; j < GEMM_CF; ++j)
      acc[i][j] = make_filled_simdgroup_matrix<gemm_acc_e, 8, 8>(gemm_acc_e(0));

  const uint chunks = GEMM_FBK / 8u;

  for (uint k0 = k_begin; k0 < k_end; k0 += GEMM_FBK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = lid; c < GEMM_FBM * chunks; c += nthr) {
      const uint r = c / chunks, sub = c % chunks;
      *(threadgroup uint4*)(Ws + r * GEMM_FBK + sub * 8u) =
          *(device const uint4*)(w + (ulong)(row0 + r) * (ulong)p.in_dim + k0 + sub * 8u);
    }
    for (uint c = lid; c < GEMM_BN * chunks; c += nthr) {
      const uint t = c / chunks, sub = c % chunks;
      threadgroup uint4* dst = (threadgroup uint4*)(As + t * GEMM_FBK + sub * 8u);
      if (tok0 + t < p.tokens) {
        *dst = *(device const uint4*)(in + (ulong)(tok0 + t) * (ulong)p.in_dim + k0 + sub * 8u);
      } else {
        *dst = uint4(0u);
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0u; kk < GEMM_FBK; kk += 8u) {
      simdgroup_half8x8 wf[GEMM_RF], af[GEMM_CF];
      for (uint i = 0u; i < GEMM_RF; ++i)
        simdgroup_load(wf[i], Ws + (sg_row * GEMM_SG_ROWS + i * 8u) * GEMM_FBK + kk, GEMM_FBK,
                       ulong2(0, 0), true);
      for (uint j = 0u; j < GEMM_CF; ++j)
        simdgroup_load(af[j], As + (sg_col * GEMM_SG_TOKS + j * 8u) * GEMM_FBK + kk, GEMM_FBK);
      for (uint i = 0u; i < GEMM_RF; ++i)
        for (uint j = 0u; j < GEMM_CF; ++j)
          simdgroup_multiply_accumulate(acc[i][j], af[j], wf[i], acc[i][j]);
    }
  }

  // Partials are [split][token][row], so the reduce reads consecutive splits with a fixed stride
  // and writes the output layout unchanged.
  const ulong plane = (ulong)p.tokens * (ulong)p.out_dim;
  threadgroup gemm_acc_e* mine = (threadgroup gemm_acc_e*)Ws + sgid * 64u;
  for (uint i = 0u; i < GEMM_RF; ++i) {
    for (uint j = 0u; j < GEMM_CF; ++j) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_store(acc[i][j], mine, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 64u; e += 32u) {
        const uint t = e / 8u;
        const uint r = e % 8u;
        const uint tok = tok0 + sg_col * GEMM_SG_TOKS + j * 8u + t;
        if (tok >= p.tokens) continue;
        const uint row = row0 + sg_row * GEMM_SG_ROWS + i * 8u + r;
        partial[(ulong)split * plane + (ulong)tok * (ulong)p.out_dim + row] = float(mine[t * 8u + r]);
      }
    }
  }
}

// Sums the GEMM_SPLITK partials into the fp16 output, applying the bias once.
kernel void cpi_gemm_splitk_reduce(device const float* partial [[buffer(0)]],
                                   device half* out [[buffer(1)]],
                                   device const half* bias [[buffer(2)]],
                                   constant GemvParams& p [[buffer(3)]],
                                   uint gid [[thread_position_in_grid]]) {
  const uint n = p.tokens * p.out_dim;
  if (gid >= n) return;
  const ulong plane = (ulong)p.tokens * (ulong)p.out_dim;
  float v = 0.0f;
  for (uint s = 0u; s < GEMM_SPLITK; ++s) v += partial[(ulong)s * plane + gid];
  if (p.has_bias != 0u) v += float(bias[gid % p.out_dim]);
  out[gid] = half(v);
}

// Fills one K-block of the weight tile, dequantizing on the way in.
//
// The tile is stored K-MAJOR -- Ws[k][row], not Ws[row][k]. The weights are [out_dim, in_dim]
// row-major, so a matrix fragment of them is [8k x 8rows]: the TRANSPOSE of how they sit in
// memory. Taking that transpose in simdgroup_load costs real time on every K-step. The
// dequant loop writes scalars anyway, so it can just as easily scatter them into K-major
// order once, and the fragment load becomes an ordinary one.
static inline void load_qblock(threadgroup half* Ws, threadgroup float* sc_row,
                               device const uchar* qw, device const half* scales, uint row0,
                               uint k0, uint groups, uint gsz, uint bits, uint packed_row, uint lid,
                               uint nthr) {
  // Hoisted out of the dequant inner loop: one scale load per row, not one per weight.
  for (uint r = lid; r < GEMM_BM; r += nthr) {
    sc_row[r] = scales[(ulong)(row0 + r) * (ulong)groups + k0 / gsz];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // One 8-weight chunk per thread per pass: a single wide load, no byte read twice. Reading
  // the packed weights a byte at a time is the defect that has bitten every kernel here.
  const uint chunks = GEMM_QBK / 8u;
  for (uint c = lid; c < GEMM_BM * chunks; c += nthr) {
    const uint r = c / chunks, sub = c % chunks;
    const uint col0 = k0 + sub * 8u;
    const float sc = sc_row[r];
    if (bits == 4u) {
      const uint packed =  // 4 bytes == 8 packed int4 weights
          *(device const uint*)(qw + (ulong)(row0 + r) * (ulong)packed_row + (col0 >> 1));
      for (uint e = 0u; e < 8u; ++e) {
        const int nib = int((packed >> (4u * e)) & 0xFu);
        Ws[(sub * 8u + e) * GEMM_BM + r] = half(float((nib ^ 0x8) - 0x8) * sc);
      }
    } else {
      const uint2 packed = *(device const uint2*)(qw + (ulong)(row0 + r) * (ulong)packed_row + col0);
      for (uint e = 0u; e < 8u; ++e) {
        const uint word = (e < 4u) ? packed.x : packed.y;
        const int q = int(as_type<char>(uchar((word >> (8u * (e & 3u))) & 0xFFu)));
        Ws[(sub * 8u + e) * GEMM_BM + r] = half(float(q) * sc);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Quantized blocked GEMM. The 8B's prefill is quantized, so an fp16-only GEMM did nothing
// for it -- this is where the prefill gap against llama.cpp actually lived.
//
// Quantization only changes how the weight tile is FILLED. Dequantize a K-block into
// threadgroup memory once -- every weight unpacked exactly once -- and from there it is an
// ordinary fp16 matmul, identical to cpi_gemm_f16: the same 4x4 register tile, the same
// reuse across every token. The quantized GEMV re-unpacked every weight for every token.
// ---------------------------------------------------------------------------
kernel void cpi_gemm_quant(device const uchar* qw [[buffer(0)]],
                           device const half* in [[buffer(1)]], device half* out [[buffer(2)]],
                           device const half* scales [[buffer(3)]],
                           device const half* bias [[buffer(4)]],
                           constant QuantParams& p [[buffer(5)]],
                           uint tgid [[threadgroup_position_in_grid]],
                           uint lid [[thread_position_in_threadgroup]],
                           uint nthr [[threads_per_threadgroup]]) {
  const uint sgid = lid / 32u;
  const uint lane = lid % 32u;
  const uint sg_row = sgid / GEMM_QSG_COLS;
  const uint sg_col = sgid % GEMM_QSG_COLS;

  const uint row_blocks = p.out_dim / GEMM_BM;
  const uint row0 = (tgid % row_blocks) * GEMM_BM;
  const uint tok0 = (tgid / row_blocks) * GEMM_QBN;
  if (tok0 >= p.tokens) return;

  // The ACTIVATIONS are not staged. They could be -- and were -- but this GEMM turned out to
  // be occupancy-limited rather than barrier-limited (deepening the K-block halves the
  // barriers and made it 8% SLOWER), so threadgroup memory is the scarce resource. Fragments
  // load straight from device instead: the token block's activations are a few hundred KB and
  // every row-block threadgroup reads the same ones, so they sit in L2. Only the weights,
  // which must be dequantized exactly once, are worth the staging.
  threadgroup half Ws[GEMM_BM * GEMM_QBK];
  threadgroup float sc_row[GEMM_BM];

  simdgroup_float8x8 acc[4][4];
  for (uint i = 0u; i < 4u; ++i)
    for (uint j = 0u; j < 4u; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  const uint gsz = (p.group == 0u) ? p.in_dim : p.group;
  const uint packed_row = (p.bits == 4u) ? ((p.in_dim + 1u) / 2u) : p.in_dim;

  // A tail tile reads past p.tokens. That stays inside the slot buffer, which is sized for a
  // whole prefill chunk, and a garbage token only ever corrupts its OWN output row -- which
  // the store below drops. So there is nothing to mask here.
  device const half* atile = in + (ulong)(tok0 + sg_col * 32u) * (ulong)p.in_dim;

  for (uint k0 = 0u; k0 < p.in_dim; k0 += GEMM_QBK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    load_qblock(Ws, sc_row, qw, scales, row0, k0, p.groups, gsz, p.bits, packed_row, lid, nthr);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint kk = 0u; kk < GEMM_QBK; kk += 8u) {
      simdgroup_half8x8 wf[4], af[4];
      // Ws is K-major, so [8k x 8rows] loads straight -- no transpose.
      for (uint i = 0u; i < 4u; ++i)
        simdgroup_load(wf[i], Ws + kk * GEMM_BM + sg_row * 32u + i * 8u, GEMM_BM);
      for (uint j = 0u; j < 4u; ++j)
        simdgroup_load(af[j], atile + (ulong)(j * 8u) * (ulong)p.in_dim + k0 + kk, p.in_dim);
      for (uint i = 0u; i < 4u; ++i)
        for (uint j = 0u; j < 4u; ++j)
          simdgroup_multiply_accumulate(acc[i][j], af[j], wf[i], acc[i][j]);
    }
  }

  threadgroup float* mine = (threadgroup float*)Ws + sgid * 64u;
  for (uint i = 0u; i < 4u; ++i) {
    for (uint j = 0u; j < 4u; ++j) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      simdgroup_store(acc[i][j], mine, 8);
      simdgroup_barrier(mem_flags::mem_threadgroup);
      for (uint e = lane; e < 64u; e += 32u) {
        const uint t = e / 8u;
        const uint r = e % 8u;
        const uint tok = tok0 + sg_col * 32u + j * 8u + t;
        if (tok >= p.tokens) continue;
        const uint row = row0 + sg_row * 32u + i * 8u + r;
        float v = mine[t * 8u + r];
        if (p.has_bias != 0u) v += float(bias[row]);
        out[(ulong)tok * (ulong)p.out_dim + row] = half(v);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Weight-only int4 / int8 GEMV.
//
// Metal has NO __dp4a, so this does not try to mirror the CUDA kernel's packed
// integer dot product. It does not need to: an int4 decode is BANDWIDTH-bound, and
// reading 0.5 bytes per weight instead of 2 is itself the whole win. Dequantise on
// the fly and multiply in fp32.
//
// Format (identical to the CUDA path -- see kernels_weight_only_matvec.cu):
//   * nibble -> signed:  (n ^ 0x8) - 0x8   [so 0..15 maps to -8..7]
//   * a byte packs two weights: low nibble = even column, high nibble = odd column
//   * y[row] = sum_g scale[row][g] * sum_{j in g} (q[j] * x[j])
//     -- weight-only: activations stay fp32, and the scale is applied per group.
// ---------------------------------------------------------------------------


kernel void cpi_gemv_quant(
    device const uchar*  qw     [[buffer(0)]],
    device const half*   in     [[buffer(1)]],
    device half*         out    [[buffer(2)]],
    device const half*  scales [[buffer(3)]],
    device const half*   bias   [[buffer(4)]],
    constant QuantParams& p     [[buffer(5)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint nthr  [[threads_per_threadgroup]]) {
  const uint simds_per_tg = nthr / 32u;
  const uint simd_id      = lid / 32u;
  const uint lane         = lid % 32u;

  const uint gsz0 = (p.group == 0u) ? p.in_dim : p.group;
  // Four-rows-per-simdgroup decode path: one simdgroup owns four consecutive rows and
  // the activation chunk is loaded once for all of them. The executor shrinks the grid
  // under the SAME predicate -- keep them in lockstep or rows go missing.
  if (p.tokens == 1u && p.bits == 4u && (p.in_dim & 31u) == 0u && (gsz0 & 31u) == 0u &&
      p.has_bias == 0u) {
    const uint r0 = (gid * simds_per_tg + simd_id) * 4u;
    if (r0 >= p.out_dim) return;
    const uint nrows = min(4u, p.out_dim - r0);
    float acc4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint n32 = p.in_dim >> 5;
    for (uint k = lane; k < n32; k += 32u) {
      const uint j0 = k << 5;
      device const half4* x4 = (device const half4*)(in + j0);
      float4 xev[4], xod[4];
      for (uint w = 0u; w < 4u; ++w) {
        const half4 xa = x4[2u * w];
        const half4 xb = x4[2u * w + 1u];
        xev[w] = float4(float(xa.x), float(xa.z), float(xb.x), float(xb.z));
        xod[w] = float4(float(xa.y), float(xa.w), float(xb.y), float(xb.w));
      }
      for (uint r = 0u; r < nrows; ++r) {
        const uint rr = r0 + r;
        device const uint4* w128 = (device const uint4*)(qw + (ulong)rr * (ulong)(p.in_dim >> 1));
        const uint4 packed = w128[k];
        const float sc = (scales + (ulong)rr * (ulong)p.groups)[j0 / gsz0];
        float sub = 0.0f;
        for (uint w = 0u; w < 4u; ++w) {
          const uint word =
              (w == 0u) ? packed.x : (w == 1u) ? packed.y : (w == 2u) ? packed.z : packed.w;
          const char4 lo = as_type<char4>((word & 0x0F0F0F0Fu) ^ 0x08080808u) - char4(8);
          const char4 hi = as_type<char4>(((word >> 4u) & 0x0F0F0F0Fu) ^ 0x08080808u) - char4(8);
          sub += dot(float4(lo), xev[w]) + dot(float4(hi), xod[w]);
        }
        acc4[r] += sub * sc;
      }
    }
    for (uint r = 0u; r < nrows; ++r) {
      const float s = simd_sum(acc4[r]);
      if (lane == 0u) out[r0 + r] = half(s);
    }
    return;
  }

  // int8 twin of the four-rows-per-simdgroup decode path. int8 was previously excluded from
  // this shape (the predicate read bits == 4), so every int8 decode fell to the generic
  // one-row-per-simdgroup path below and left the activation re-read once per row. Bytes map
  // straight onto the activations here -- no even/odd nibble split.
  if (p.tokens == 1u && p.bits == 8u && (p.in_dim & 31u) == 0u && (gsz0 & 31u) == 0u &&
      p.has_bias == 0u) {
    const uint r0 = (gid * simds_per_tg + simd_id) * 4u;
    if (r0 >= p.out_dim) return;
    const uint nrows = min(4u, p.out_dim - r0);
    float acc4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint n32 = p.in_dim >> 5;  // 32 weights per lane-step, as in the int4 path
    for (uint k = lane; k < n32; k += 32u) {
      const uint j0 = k << 5;
      device const half4* x4 = (device const half4*)(in + j0);
      float4 xv[8];
      for (uint w = 0u; w < 8u; ++w) {
        const half4 h = x4[w];
        xv[w] = float4(float(h.x), float(h.y), float(h.z), float(h.w));
      }
      for (uint r = 0u; r < nrows; ++r) {
        const uint rr = r0 + r;
        device const uint4* w128 = (device const uint4*)(qw + (ulong)rr * (ulong)p.in_dim);
        const uint4 pa = w128[2u * k];       // 32 int8 weights = two 128-bit loads
        const uint4 pb = w128[2u * k + 1u];
        const float sc = (scales + (ulong)rr * (ulong)p.groups)[j0 / gsz0];
        float sub = 0.0f;
        for (uint w = 0u; w < 8u; ++w) {
          uint word;
          if (w < 4u) {
            word = (w == 0u) ? pa.x : (w == 1u) ? pa.y : (w == 2u) ? pa.z : pa.w;
          } else {
            word = (w == 4u) ? pb.x : (w == 5u) ? pb.y : (w == 6u) ? pb.z : pb.w;
          }
          sub += dot(float4(as_type<char4>(word)), xv[w]);
        }
        acc4[r] += sub * sc;
      }
    }
    for (uint r = 0u; r < nrows; ++r) {
      const float s = simd_sum(acc4[r]);
      if (lane == 0u) out[r0 + r] = half(s);
    }
    return;
  }

  const uint row_blocks = (p.out_dim + simds_per_tg - 1u) / simds_per_tg;
  const uint tile = gid / row_blocks;
  const uint blk  = gid % row_blocks;
  const uint row  = blk * simds_per_tg + simd_id;
  const uint t0   = tile * GEMV_TILE;
  if (t0 >= p.tokens || row >= p.out_dim) return;
  const uint nt = min((uint)GEMV_TILE, p.tokens - t0);

  device const half* srow = scales + (ulong)row * (ulong)p.groups;
  const uint gsz = (p.group == 0u) ? p.in_dim : p.group;

  float acc[GEMV_TILE];
  for (uint t = 0u; t < GEMV_TILE; ++t) acc[t] = 0.0f;

  if (p.bits == 4u) {
    // 128-BIT loads: a uint4 is 32 packed int4 weights. This used to load a bare `uint`
    // (8 weights, 32 bits) and it cost ~40% of decode throughput -- the same narrow-load
    // mistake as the fp16 GEMV, made again in a new kernel. On a bandwidth-bound machine
    // the LOAD WIDTH IS THE KERNEL, whatever the element size.
    //
    // A group size that is a multiple of 32 keeps all 32 weights inside one scale group.
    device const uint4* w128 =
        (device const uint4*)(qw + (ulong)row * (ulong)((p.in_dim + 1u) / 2u));
    const uint n32 = p.in_dim >> 5;  // 32 int4 weights per uint4
    for (uint k = lane; k < n32; k += 32u) {
      const uint4 packed = w128[k];
      const uint j0 = k << 5;
      const float sc = srow[j0 / gsz];

      char4 lo[4], hi[4];
      for (uint w = 0u; w < 4u; ++w) {
        const uint word = (w == 0u) ? packed.x : (w == 1u) ? packed.y
                        : (w == 2u) ? packed.z : packed.w;
        lo[w] = as_type<char4>((word & 0x0F0F0F0Fu) ^ 0x08080808u) - char4(8);
        hi[w] = as_type<char4>(((word >> 4u) & 0x0F0F0F0Fu) ^ 0x08080808u) - char4(8);
      }
      for (uint t = 0u; t < nt; ++t) {
        device const half4* x4 = (device const half4*)(in + (ulong)(t0 + t) * (ulong)p.in_dim + j0);
        float sub = 0.0f;
        for (uint w = 0u; w < 4u; ++w) {
          const half4 xa = x4[2u * w];
          const half4 xb = x4[2u * w + 1u];
          const float4 ev = float4(float(xa.x), float(xa.z), float(xb.x), float(xb.z));
          const float4 od = float4(float(xa.y), float(xa.w), float(xb.y), float(xb.w));
          sub += dot(float4(lo[w]), ev) + dot(float4(hi[w]), od);
        }
        acc[t] += sub * sc;
      }
    }
    // Tail: whatever does not fill a uint4.
    device const uint* w32 = (device const uint*)(qw + (ulong)row * (ulong)((p.in_dim + 1u) / 2u));
    for (uint k = (n32 << 2) + lane; k < (p.in_dim >> 3); k += 32u) {
      const uint packed = w32[k];
      const uint j0 = k << 3;
      const float sc = srow[j0 / gsz];
      for (uint t = 0u; t < nt; ++t) {
        device const half* x = in + (ulong)(t0 + t) * (ulong)p.in_dim + j0;
        float sub = 0.0f;
        for (uint e = 0u; e < 8u; ++e) {
          const int nib = int((packed >> (4u * e)) & 0xFu);
          sub += float((nib ^ 0x8) - 0x8) * float(x[e]);
        }
        acc[t] += sub * sc;
      }
    }
  } else {
    // int8, 128 bits at a time: a uint4 is SIXTEEN weights. Loading them one byte at a
    // time made int8 slower than fp16 -- which is absurd, since it reads half the bytes
    // -- for exactly the reason the fp16 GEMV needed 128-bit loads. Narrow loads waste
    // the transaction and cannot cover DRAM latency, whatever the element size.
    device const uint4* w16 = (device const uint4*)(qw + (ulong)row * (ulong)p.in_dim);
    const uint n16 = p.in_dim >> 4;  // 16 int8s per uint4
    for (uint k = lane; k < n16; k += 32u) {
      const uint4 packed = w16[k];
      const uint j0 = k << 4;
      const float sc = srow[j0 / gsz];  // gsz is a multiple of 16 here

      char4 qb[4];
      for (uint w = 0u; w < 4u; ++w) {
        const uint word = (w == 0u) ? packed.x : (w == 1u) ? packed.y
                        : (w == 2u) ? packed.z : packed.w;
        qb[w] = as_type<char4>(word);  // int8 needs no decode; bytes ARE the weights
      }
      for (uint t = 0u; t < nt; ++t) {
        device const half4* x4 = (device const half4*)(in + (ulong)(t0 + t) * (ulong)p.in_dim + j0);
        float sub = 0.0f;
        for (uint w = 0u; w < 4u; ++w) {
          sub += dot(float4(qb[w]), float4(x4[w]));
        }
        acc[t] += sub * sc;
      }
    }
    // Tail: whatever does not fill a uint4.
    for (uint k = (n16 << 4) + lane; k < p.in_dim; k += 32u) {
      const int q = int(((device const char*)(qw + (ulong)row * (ulong)p.in_dim))[k]);
      const float sc = srow[k / gsz];
      for (uint t = 0u; t < nt; ++t) {
        acc[t] += float(q) * sc * float(in[(ulong)(t0 + t) * (ulong)p.in_dim + k]);
      }
    }
  }

  for (uint t = 0u; t < nt; ++t) {
    float s = simd_sum(acc[t]);
    if (lane == 0u) {
      if (p.has_bias != 0u) s += float(bias[row]);
      out[(ulong)(t0 + t) * (ulong)p.out_dim + row] = half(s);
    }
  }
}

// ---------------------------------------------------------------------------
// FUSED quantized GEMV over 2 or 3 concatenated weight matrices that share the same input.
// Decode fires ~7 separate GEMV dispatches per layer; three of them (Q, K, V) read the same
// normalized input, and two more (gate, up) do too. Each tiny GEMV is launch/latency-bound,
// not bandwidth-bound, so collapsing the group into ONE dispatch removes that fixed per-token
// cost -- which is the dominant share of a bandwidth-light int4 decode. No weight
// pre-concatenation: each output row across the [n0 + n1 + n2] space is ROUTED to its matrix.
// ---------------------------------------------------------------------------
struct QCatParams {
  uint n0, n1, n2;  // rows (out_dim) of each sub-matrix; n2 == 0 for the 2-way (gate/up) case
  uint in_dim;
  uint tokens;
  uint bits;
  uint group;
  uint groups;
  uint has_bias;  // bias is ONE concatenated tensor over the [n0+n1+n2] rows (Qwen's bqkv)
};

// One output row's quantized dot product against the token tile, written to `out`.
static inline void qcat_row(device const uchar* qw, device const half* scales,
                            device const half* in, device half* out, device const half* bias,
                            uint row, uint global_row, uint out_dim_m, uint has_bias, uint in_dim,
                            uint bits, uint gsz, uint groups, uint t0, uint tokens, uint lane) {
  const uint packed_row = (bits == 4u) ? ((in_dim + 1u) / 2u) : in_dim;
  device const half* srow = scales + (ulong)row * (ulong)groups;
  const uint nt = min((uint)GEMV_TILE, tokens - t0);
  float acc[GEMV_TILE];
  for (uint t = 0u; t < GEMV_TILE; ++t) acc[t] = 0.0f;

  if (bits == 4u) {
    device const uint4* w128 = (device const uint4*)(qw + (ulong)row * (ulong)packed_row);
    const uint n32 = in_dim >> 5;  // 32 int4 weights per uint4 (128-bit load)
    for (uint k = lane; k < n32; k += 32u) {
      const uint4 packed = w128[k];
      const uint j0 = k << 5;
      const float sc = srow[j0 / gsz];
      // A uint's low nibbles are elements {0,2,4,6}, high nibbles {1,3,5,7} (LSB-first
      // packing); two's-complement decode is (n ^ 8) - 8, one char4 op per word. The
      // half4 x loads use the matching even/odd swizzle.
      char4 lo[4], hi[4];
      for (uint w = 0u; w < 4u; ++w) {
        const uint word =
            (w == 0u) ? packed.x : (w == 1u) ? packed.y : (w == 2u) ? packed.z : packed.w;
        lo[w] = as_type<char4>((word & 0x0F0F0F0Fu) ^ 0x08080808u) - char4(8);
        hi[w] = as_type<char4>(((word >> 4u) & 0x0F0F0F0Fu) ^ 0x08080808u) - char4(8);
      }
      for (uint t = 0u; t < nt; ++t) {
        device const half4* x4 = (device const half4*)(in + (ulong)(t0 + t) * (ulong)in_dim + j0);
        float sub = 0.0f;
        for (uint w = 0u; w < 4u; ++w) {
          const half4 xa = x4[2u * w];
          const half4 xb = x4[2u * w + 1u];
          const float4 ev = float4(float(xa.x), float(xa.z), float(xb.x), float(xb.z));
          const float4 od = float4(float(xa.y), float(xa.w), float(xb.y), float(xb.w));
          sub += dot(float4(lo[w]), ev) + dot(float4(hi[w]), od);
        }
        acc[t] += sub * sc;
      }
    }
    device const uint* w32 = (device const uint*)(qw + (ulong)row * (ulong)packed_row);
    for (uint k = (n32 << 2) + lane; k < (in_dim >> 3); k += 32u) {
      const uint packed = w32[k];
      const uint j0 = k << 3;
      const float sc = srow[j0 / gsz];
      for (uint t = 0u; t < nt; ++t) {
        device const half* x = in + (ulong)(t0 + t) * (ulong)in_dim + j0;
        float sub = 0.0f;
        for (uint e = 0u; e < 8u; ++e) {
          const int nib = int((packed >> (4u * e)) & 0xFu);
          sub += float((nib ^ 0x8) - 0x8) * float(x[e]);
        }
        acc[t] += sub * sc;
      }
    }
  } else {
    device const uint4* w16 = (device const uint4*)(qw + (ulong)row * (ulong)in_dim);
    const uint n16 = in_dim >> 4;  // 16 int8s per uint4
    for (uint k = lane; k < n16; k += 32u) {
      const uint4 packed = w16[k];
      const uint j0 = k << 4;
      const float sc = srow[j0 / gsz];
      char4 qb[4];
      for (uint w = 0u; w < 4u; ++w) {
        const uint word =
            (w == 0u) ? packed.x : (w == 1u) ? packed.y : (w == 2u) ? packed.z : packed.w;
        qb[w] = as_type<char4>(word);  // int8 needs no decode; bytes ARE the weights
      }
      for (uint t = 0u; t < nt; ++t) {
        device const half4* x4 = (device const half4*)(in + (ulong)(t0 + t) * (ulong)in_dim + j0);
        float sub = 0.0f;
        for (uint w = 0u; w < 4u; ++w) {
          sub += dot(float4(qb[w]), float4(x4[w]));
        }
        acc[t] += sub * sc;
      }
    }
    for (uint k = (n16 << 4) + lane; k < in_dim; k += 32u) {
      const int q = int(((device const char*)(qw + (ulong)row * (ulong)in_dim))[k]);
      const float sc = srow[k / gsz];
      for (uint t = 0u; t < nt; ++t)
        acc[t] += float(q) * sc * float(in[(ulong)(t0 + t) * (ulong)in_dim + k]);
    }
  }

  for (uint t = 0u; t < nt; ++t) {
    float s = simd_sum(acc[t]);
    if (lane == 0u) {
      if (has_bias != 0u) s += float(bias[global_row]);
      out[(ulong)(t0 + t) * (ulong)out_dim_m + row] = half(s);
    }
  }
}

kernel void cpi_gemv_quant_cat(
    device const half*   in   [[buffer(0)]],
    device const uchar*  qw0  [[buffer(1)]], device const half* sc0 [[buffer(2)]],
    device half*         out0 [[buffer(3)]],
    device const uchar*  qw1  [[buffer(4)]], device const half* sc1 [[buffer(5)]],
    device half*         out1 [[buffer(6)]],
    device const uchar*  qw2  [[buffer(7)]], device const half* sc2 [[buffer(8)]],
    device half*         out2 [[buffer(9)]],
    device const half*   bias [[buffer(10)]],
    constant QCatParams& p    [[buffer(11)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  const uint simds_per_tg = nthr / 32u;
  const uint simd_id = lid / 32u;
  const uint lane = lid % 32u;

  const uint total = p.n0 + p.n1 + p.n2;
  const uint gsz = (p.group == 0u) ? p.in_dim : p.group;

  // Four-rows-per-simdgroup decode path (see cpi_gemv_quant); executor shrinks the grid
  // under the SAME predicate.
  if (p.tokens == 1u && p.bits == 4u && (p.in_dim & 31u) == 0u && (gsz & 31u) == 0u &&
      p.has_bias == 0u) {
    const uint r0 = (gid * simds_per_tg + simd_id) * 4u;
    if (r0 >= total) return;
    const uint nrows = min(4u, total - r0);
    float acc4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint n32 = p.in_dim >> 5;
    for (uint k = lane; k < n32; k += 32u) {
      const uint j0 = k << 5;
      device const half4* x4 = (device const half4*)(in + j0);
      float4 xev[4], xod[4];
      for (uint w = 0u; w < 4u; ++w) {
        const half4 xa = x4[2u * w];
        const half4 xb = x4[2u * w + 1u];
        xev[w] = float4(float(xa.x), float(xa.z), float(xb.x), float(xb.z));
        xod[w] = float4(float(xa.y), float(xa.w), float(xb.y), float(xb.w));
      }
      for (uint r = 0u; r < nrows; ++r) {
        const uint g = r0 + r;
        device const uchar* qw = (g < p.n0) ? qw0 : (g < p.n0 + p.n1) ? qw1 : qw2;
        device const half* sc = (g < p.n0) ? sc0 : (g < p.n0 + p.n1) ? sc1 : sc2;
        const uint lr = (g < p.n0) ? g : (g < p.n0 + p.n1) ? (g - p.n0) : (g - p.n0 - p.n1);
        device const uint4* w128 = (device const uint4*)(qw + (ulong)lr * (ulong)(p.in_dim >> 1));
        const uint4 packed = w128[k];
        const float scl = (sc + (ulong)lr * (ulong)p.groups)[j0 / gsz];
        float sub = 0.0f;
        for (uint w = 0u; w < 4u; ++w) {
          const uint word =
              (w == 0u) ? packed.x : (w == 1u) ? packed.y : (w == 2u) ? packed.z : packed.w;
          const char4 lo = as_type<char4>((word & 0x0F0F0F0Fu) ^ 0x08080808u) - char4(8);
          const char4 hi = as_type<char4>(((word >> 4u) & 0x0F0F0F0Fu) ^ 0x08080808u) - char4(8);
          sub += dot(float4(lo), xev[w]) + dot(float4(hi), xod[w]);
        }
        acc4[r] += sub * scl;
      }
    }
    for (uint r = 0u; r < nrows; ++r) {
      const float s = simd_sum(acc4[r]);
      if (lane == 0u) {
        const uint g = r0 + r;
        device half* o = (g < p.n0) ? out0 : (g < p.n0 + p.n1) ? out1 : out2;
        const uint lr = (g < p.n0) ? g : (g < p.n0 + p.n1) ? (g - p.n0) : (g - p.n0 - p.n1);
        o[lr] = half(s);
      }
    }
    return;
  }

  // int8 twin of the four-rows-per-simdgroup cat path (q|k|v and gate|up at decode). Same
  // segment routing; int8 rows are in_dim bytes and map straight onto the activations.
  if (p.tokens == 1u && p.bits == 8u && (p.in_dim & 31u) == 0u && (gsz & 31u) == 0u &&
      p.has_bias == 0u) {
    const uint r0 = (gid * simds_per_tg + simd_id) * 4u;
    if (r0 >= total) return;
    const uint nrows = min(4u, total - r0);
    float acc4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint n32 = p.in_dim >> 5;
    for (uint k = lane; k < n32; k += 32u) {
      const uint j0 = k << 5;
      device const half4* x4 = (device const half4*)(in + j0);
      float4 xv[8];
      for (uint w = 0u; w < 8u; ++w) {
        const half4 h = x4[w];
        xv[w] = float4(float(h.x), float(h.y), float(h.z), float(h.w));
      }
      for (uint r = 0u; r < nrows; ++r) {
        const uint g = r0 + r;
        device const uchar* qw = (g < p.n0) ? qw0 : (g < p.n0 + p.n1) ? qw1 : qw2;
        device const half* sc = (g < p.n0) ? sc0 : (g < p.n0 + p.n1) ? sc1 : sc2;
        const uint lr = (g < p.n0) ? g : (g < p.n0 + p.n1) ? (g - p.n0) : (g - p.n0 - p.n1);
        device const uint4* w128 = (device const uint4*)(qw + (ulong)lr * (ulong)p.in_dim);
        const uint4 pa = w128[2u * k];
        const uint4 pb = w128[2u * k + 1u];
        const float scl = (sc + (ulong)lr * (ulong)p.groups)[j0 / gsz];
        float sub = 0.0f;
        for (uint w = 0u; w < 8u; ++w) {
          uint word;
          if (w < 4u) {
            word = (w == 0u) ? pa.x : (w == 1u) ? pa.y : (w == 2u) ? pa.z : pa.w;
          } else {
            word = (w == 4u) ? pb.x : (w == 5u) ? pb.y : (w == 6u) ? pb.z : pb.w;
          }
          sub += dot(float4(as_type<char4>(word)), xv[w]);
        }
        acc4[r] += sub * scl;
      }
    }
    for (uint r = 0u; r < nrows; ++r) {
      const float s = simd_sum(acc4[r]);
      if (lane == 0u) {
        const uint g = r0 + r;
        device half* o = (g < p.n0) ? out0 : (g < p.n0 + p.n1) ? out1 : out2;
        const uint lr = (g < p.n0) ? g : (g < p.n0 + p.n1) ? (g - p.n0) : (g - p.n0 - p.n1);
        o[lr] = half(s);
      }
    }
    return;
  }

  const uint row_blocks = (total + simds_per_tg - 1u) / simds_per_tg;
  const uint tile = gid / row_blocks;
  const uint t0 = tile * GEMV_TILE;
  const uint grow = (gid % row_blocks) * simds_per_tg + simd_id;  // global row
  if (t0 >= p.tokens || grow >= total) return;

  if (grow < p.n0) {
    qcat_row(qw0, sc0, in, out0, bias, grow, grow, p.n0, p.has_bias, p.in_dim, p.bits, gsz,
             p.groups, t0, p.tokens, lane);
  } else if (grow < p.n0 + p.n1) {
    const uint r = grow - p.n0;
    qcat_row(qw1, sc1, in, out1, bias, r, grow, p.n1, p.has_bias, p.in_dim, p.bits, gsz, p.groups,
             t0, p.tokens, lane);
  } else {
    const uint r = grow - p.n0 - p.n1;
    qcat_row(qw2, sc2, in, out2, bias, r, grow, p.n2, p.has_bias, p.in_dim, p.bits, gsz, p.groups,
             t0, p.tokens, lane);
  }
}

// Fused GeGLU GEMV: out[t, r] = gelu_tanh(gate_r . x_t) * (up_r . x_t) in one dispatch --
// no Gate/Up slot round-trip, no separate cpi_gelu_mul. One row per simdgroup: the first
// half of the threadgroup's simds compute gate rows, the second half their up partners,
// paired through threadgroup memory. Both dots round to half before the gelu, exactly as
// the unfused sequence rounds them.
struct QGluParams {
  uint n0, n1, n2, in_dim, tokens, bits, group, groups, has_bias;
  float glu_scale;  // cpi_gelu_mul's p.scale: 0 means 1; multiplied into the stored product
};

kernel void cpi_gemv_quant_glu(
    device const half*   in   [[buffer(0)]],
    device const uchar*  qwg  [[buffer(1)]], device const half* scg [[buffer(2)]],
    device const uchar*  qwu  [[buffer(3)]], device const half* scu [[buffer(4)]],
    device half*         out  [[buffer(5)]],
    constant QGluParams& p    [[buffer(6)]],
    uint gid  [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint nthr [[threads_per_threadgroup]]) {
  const uint simds_per_tg = nthr / 32u;
  const uint pairs_per_tg = simds_per_tg / 2u;
  const uint simd_id = lid / 32u;
  const uint lane = lid % 32u;
  const bool is_up = simd_id >= pairs_per_tg;
  const uint local_pair = is_up ? (simd_id - pairs_per_tg) : simd_id;

  const uint row_blocks = (p.n0 + pairs_per_tg - 1u) / pairs_per_tg;
  const uint tile = gid / row_blocks;
  const uint t0 = tile * GEMV_TILE;
  const uint row = (gid % row_blocks) * pairs_per_tg + local_pair;

  threadgroup float pair_acc[GEMV_TILE * 8u];  // [token][simd], simds_per_tg <= 8

  float acc[GEMV_TILE];
  for (uint t = 0u; t < GEMV_TILE; ++t) acc[t] = 0.0f;
  const uint nt = min((uint)GEMV_TILE, p.tokens - min(t0, p.tokens));

  if (t0 < p.tokens && row < p.n0) {
    device const uchar* qw = is_up ? qwu : qwg;
    device const half* sc = is_up ? scu : scg;
    const uint gsz = (p.group == 0u) ? p.in_dim : p.group;
    device const half* srow = sc + (ulong)row * (ulong)p.groups;
    if (p.bits == 4u) {
      device const uint* w32 = (device const uint*)(qw + (ulong)row * (ulong)(p.in_dim >> 1));
      const uint n8 = p.in_dim >> 3;
      for (uint k = lane; k < n8; k += 32u) {
        const uint packed = w32[k];
        const uint j0 = k << 3;
        const float scl = srow[j0 / gsz];
        for (uint t = 0u; t < nt; ++t) {
          device const half* x = in + (ulong)(t0 + t) * (ulong)p.in_dim + j0;
          float sub = 0.0f;
          for (uint e = 0u; e < 8u; ++e) {
            const int nib = int((packed >> (4u * e)) & 0xFu);
            sub += float((nib ^ 0x8) - 0x8) * float(x[e]);
          }
          acc[t] += sub * scl;
        }
      }
    } else {
      device const uint4* w16 = (device const uint4*)(qw + (ulong)row * (ulong)p.in_dim);
      const uint n16 = p.in_dim >> 4;
      for (uint k = lane; k < n16; k += 32u) {
        const uint4 packed = w16[k];
        const uint j0 = k << 4;
        const float scl = srow[j0 / gsz];
        for (uint t = 0u; t < nt; ++t) {
          device const half* x = in + (ulong)(t0 + t) * (ulong)p.in_dim + j0;
          float sub = 0.0f;
          for (uint w = 0u; w < 4u; ++w) {
            const uint word =
                (w == 0u) ? packed.x : (w == 1u) ? packed.y : (w == 2u) ? packed.z : packed.w;
            for (uint e = 0u; e < 4u; ++e) {
              const int q = int(char((word >> (8u * e)) & 0xFFu));
              sub += float(q) * float(x[w * 4u + e]);
            }
          }
          acc[t] += sub * scl;
        }
      }
    }
  }

  for (uint t = 0u; t < GEMV_TILE; ++t) {
    const float s = simd_sum(acc[t]);
    if (lane == 0u) pair_acc[t * 8u + simd_id] = s;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  if (simd_id == 0u && lane < pairs_per_tg) {
    const uint orow = (gid % row_blocks) * pairs_per_tg + lane;
    if (t0 < p.tokens && orow < p.n0) {
      // EXACTLY cpi_gelu_mul's semantics or the fusion is a numerics change in disguise:
      // sigmoid-form gelu (tanh overflows via exp(+large)), the fp16-headroom scale that
      // the plan multiplies back after the down-projection (first version omitted it --
      // activations landed 2^k too large and generation died), and the clamp backstop.
      for (uint t = 0u; t < nt; ++t) {
        const float g = float(half(pair_acc[t * 8u + lane]));
        const float u = float(half(pair_acc[t * 8u + pairs_per_tg + lane]));
        const float kg = 0.7978845608028654f;  // sqrt(2/pi)
        const float inner = kg * (g + 0.044715f * g * g * g);
        const float gelu = g / (1.0f + exp(-2.0f * inner));
        const float prod = gelu * u;
        const float scaled = (p.glu_scale != 0.0f) ? (prod * p.glu_scale) : prod;
        out[(ulong)(t0 + t) * (ulong)p.n0 + orow] = half(clamp(scaled, -65504.0f, 65504.0f));
      }
    }
  }
}

// Same, but fp32 output: the LM head feeds the sampler.
kernel void cpi_lm_head_quant(
    device const uchar*  qw     [[buffer(0)]],
    device const half*   in     [[buffer(1)]],
    device float*        out    [[buffer(2)]],
    device const half*  scales [[buffer(3)]],
    constant QuantParams& p     [[buffer(4)]],
    uint gid   [[threadgroup_position_in_grid]],
    uint lid   [[thread_position_in_threadgroup]],
    uint nthr  [[threads_per_threadgroup]]) {
  const uint simds_per_tg = nthr / 32u;
  const uint simd_id      = lid / 32u;
  const uint lane         = lid % 32u;
  const uint gsz = (p.group == 0u) ? p.in_dim : p.group;

  // The head is the biggest single read of a quantized decode step (vocab x hidden), and this
  // kernel had BOTH known GEMV mistakes at once: 32-bit weight loads with a scalar per-nibble
  // decode (the "narrow-load mistake" that cost ~40% in cpi_gemv_quant), and one row per
  // simdgroup, re-reading the activation once per vocab row. This is the wide-load + four-rows
  // shape from cpi_gemv_quant with fp32 outputs; the executor passes 64-thread threadgroups
  // under the same predicate (in_dim % 32 == 0, group % 32 == 0).
  if ((p.in_dim & 31u) == 0u && (gsz & 31u) == 0u) {
    const uint r0 = (gid * simds_per_tg + simd_id) * 4u;
    if (r0 >= p.out_dim) return;
    const uint nrows = min(4u, p.out_dim - r0);
    float acc4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const uint n32 = p.in_dim >> 5;
    if (p.bits == 4u) {
      for (uint k = lane; k < n32; k += 32u) {
        const uint j0 = k << 5;
        device const half4* x4 = (device const half4*)(in + j0);
        float4 xev[4], xod[4];
        for (uint w = 0u; w < 4u; ++w) {
          const half4 xa = x4[2u * w];
          const half4 xb = x4[2u * w + 1u];
          xev[w] = float4(float(xa.x), float(xa.z), float(xb.x), float(xb.z));
          xod[w] = float4(float(xa.y), float(xa.w), float(xb.y), float(xb.w));
        }
        for (uint r = 0u; r < nrows; ++r) {
          const uint rr = r0 + r;
          device const uint4* w128 =
              (device const uint4*)(qw + (ulong)rr * (ulong)(p.in_dim >> 1));
          const uint4 packed = w128[k];
          const float sc = (scales + (ulong)rr * (ulong)p.groups)[j0 / gsz];
          float sub = 0.0f;
          for (uint w = 0u; w < 4u; ++w) {
            const uint word =
                (w == 0u) ? packed.x : (w == 1u) ? packed.y : (w == 2u) ? packed.z : packed.w;
            const char4 lo = as_type<char4>((word & 0x0F0F0F0Fu) ^ 0x08080808u) - char4(8);
            const char4 hi = as_type<char4>(((word >> 4u) & 0x0F0F0F0Fu) ^ 0x08080808u) - char4(8);
            sub += dot(float4(lo), xev[w]) + dot(float4(hi), xod[w]);
          }
          acc4[r] += sub * sc;
        }
      }
    } else {
      for (uint k = lane; k < n32; k += 32u) {
        const uint j0 = k << 5;
        device const half4* x4 = (device const half4*)(in + j0);
        float4 xv[8];
        for (uint w = 0u; w < 8u; ++w) {
          const half4 h = x4[w];
          xv[w] = float4(float(h.x), float(h.y), float(h.z), float(h.w));
        }
        for (uint r = 0u; r < nrows; ++r) {
          const uint rr = r0 + r;
          device const uint4* w128 = (device const uint4*)(qw + (ulong)rr * (ulong)p.in_dim);
          const uint4 pa = w128[2u * k];
          const uint4 pb = w128[2u * k + 1u];
          const float sc = (scales + (ulong)rr * (ulong)p.groups)[j0 / gsz];
          float sub = 0.0f;
          for (uint w = 0u; w < 8u; ++w) {
            uint word;
            if (w < 4u) {
              word = (w == 0u) ? pa.x : (w == 1u) ? pa.y : (w == 2u) ? pa.z : pa.w;
            } else {
              word = (w == 4u) ? pb.x : (w == 5u) ? pb.y : (w == 6u) ? pb.z : pb.w;
            }
            sub += dot(float4(as_type<char4>(word)), xv[w]);
          }
          acc4[r] += sub * sc;
        }
      }
    }
    for (uint r = 0u; r < nrows; ++r) {
      const float s = simd_sum(acc4[r]);
      if (lane == 0u) out[r0 + r] = s;
    }
    return;
  }

  const uint row = gid * simds_per_tg + simd_id;
  if (row >= p.out_dim) return;

  device const half* srow = scales + (ulong)row * (ulong)p.groups;
  float acc = 0.0f;

  if (p.bits == 4u) {
    device const uint* w32 = (device const uint*)(qw + (ulong)row * (ulong)((p.in_dim + 1u) / 2u));
    const uint n8 = p.in_dim >> 3;
    for (uint k = lane; k < n8; k += 32u) {
      const uint packed = w32[k];
      const uint j0 = k << 3;
      float sub = 0.0f;
      for (uint e = 0u; e < 8u; ++e) {
        const int nib = int((packed >> (4u * e)) & 0xFu);
        const int q = (nib ^ 0x8) - 0x8;
        sub += float(q) * float(in[j0 + e]);
      }
      acc += sub * srow[j0 / gsz];
    }
  } else {
    device const char* w8 = (device const char*)(qw + (ulong)row * (ulong)p.in_dim);
    for (uint k = lane; k < p.in_dim; k += 32u) {
      acc += float(w8[k]) * srow[k / gsz] * float(in[k]);
    }
  }

  acc = simd_sum(acc);
  if (lane == 0u) out[row] = acc;
}

// ---------------------------------------------------------------------------
// RoPE, half-split (NeoX) convention: element i is paired with i + head_dim/2.
// In-place over `heads` heads of `head_dim`.
// ---------------------------------------------------------------------------

// NOTE the buffer order: the params block is ALWAYS the last binding, in every
// kernel here. MetalContext::dispatch() binds it at index n_buffers, so a kernel
// that puts it anywhere else would have its params written over another buffer --
// which compiles cleanly and only misbehaves on real hardware.
