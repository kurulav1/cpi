// Vision-tower kernels -- the Gemma 4 image path.
//
// The Metal side of the four ops the CUDA plan engine already runs: PatchEmbed, Rope2D,
// AvgPoolPatches and Standardize. The arithmetic mirrors kernels_vision.cu, including which
// operands are fp32 (the pixels and the RoPE tables) and which are half, so the two backends can
// be compared directly rather than approximately.
//
// Padding patches are the thread running through all of these: a patch whose position is negative
// is padding, and it must contribute nothing. PatchEmbed zeroes it, AvgPoolPatches skips it in the
// average. Getting that wrong does not crash -- it quietly mixes padding into real cells.

#include <metal_stdlib>
using namespace metal;

// MetalContext dispatches a 1-D grid, so every kernel here flattens its own index. Keeping the
// token count in the params is what makes that possible.
struct VisPatchParams {
  uint hidden;
  uint patch_dim;
  uint pos_table_size;
  uint tokens;
};

struct VisRope2DParams {
  uint num_heads;
  uint head_dim;
  uint pairs_per_half;  // head_dim / 4
  uint tokens;
};

struct VisPoolParams {
  uint tokens;
  uint hidden;
  uint k;         // pool window, k x k patches per cell
  uint cells_x;
  uint out_tokens;
  float gain;
};

struct VisStdParams {
  uint n;       // tokens * hidden
  uint hidden;
};

// out[t] = proj . (2*(pixels[t] - 0.5)) + pos_table[0][x] + pos_table[1][y]
//
// The position table is [2, pos_table_size, hidden]: the x coordinate indexes plane 0, y plane 1,
// and the two are summed. (HF one-hots the coordinates into the table and sums; a gather is the
// same thing without materialising the one-hot.) Gemma applies no mean/std normalisation here --
// it rescales [0,1] to [-1,1] inline, which is the 2*(p-0.5).
kernel void cpi_patch_embed(device const half* proj [[buffer(0)]],
                            device const float* pixels [[buffer(1)]],
                            device const half* pos_table [[buffer(2)]],
                            device const int* pos_x [[buffer(3)]],
                            device const int* pos_y [[buffer(4)]],
                            device half* out [[buffer(5)]],
                            constant VisPatchParams& p [[buffer(6)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid >= p.tokens * p.hidden) return;
  const uint token = gid / p.hidden;
  const uint h = gid % p.hidden;

  const int px = pos_x[token];
  const int py = pos_y[token];
  const uint o = token * p.hidden + h;
  if (px < 0 || py < 0) {  // padding patch contributes nothing downstream
    out[o] = half(0.0f);
    return;
  }

  device const float* pix = pixels + (ulong)token * (ulong)p.patch_dim;
  device const half* w = proj + (ulong)h * (ulong)p.patch_dim;
  float acc = 0.0f;
  for (uint k = 0u; k < p.patch_dim; ++k) acc += float(w[k]) * (2.0f * (pix[k] - 0.5f));

  const ulong plane = (ulong)p.pos_table_size * (ulong)p.hidden;
  acc += float(pos_table[(ulong)px * (ulong)p.hidden + h]);
  acc += float(pos_table[plane + (ulong)py * (ulong)p.hidden + h]);
  out[o] = half(acc);
}

// 2-D RoPE: the head splits in half, the first half rotated by the x coordinate and the second by
// y. Within each half the pairs are (j, j + pairs_per_half), NOT adjacent elements -- the same
// split-halves convention the 1-D RoPE uses, so a head_dim not divisible by 4 has no valid pairing
// and the caller must not dispatch this.
kernel void cpi_rope_2d_inplace(device half* x [[buffer(0)]], device const int* pos_x [[buffer(1)]],
                                device const int* pos_y [[buffer(2)]],
                                device const float* cos_table [[buffer(3)]],
                                device const float* sin_table [[buffer(4)]],
                                constant VisRope2DParams& p [[buffer(5)]],
                                uint gid [[thread_position_in_grid]]) {
  const uint half_pairs = p.pairs_per_half;
  const uint per_head = 2u * half_pairs;
  if (gid >= p.tokens * p.num_heads * per_head) return;
  const uint t = gid % per_head;
  const uint head = (gid / per_head) % p.num_heads;
  const uint token = gid / (per_head * p.num_heads);

  const uint axis = t / half_pairs;  // 0 = x, 1 = y
  const uint j = t - axis * half_pairs;
  const uint base = axis * (p.head_dim / 2u);

  const int pos = (axis == 0u) ? pos_x[token] : pos_y[token];
  const uint tab = uint(pos) * half_pairs + j;
  const float c = cos_table[tab];
  const float s = sin_table[tab];

  const ulong row = ((ulong)token * (ulong)p.num_heads + head) * (ulong)p.head_dim;
  const uint i0 = base + j;
  const uint i1 = base + j + half_pairs;
  const float v0 = float(x[row + i0]);
  const float v1 = float(x[row + i1]);
  x[row + i0] = half(v0 * c - v1 * s);
  x[row + i1] = half(v1 * c + v0 * s);
}

// Average the patches falling in each k x k cell, then apply a gain. The divisor is k*k, NOT the
// number of patches actually found -- padding patches are skipped but do not shrink the divisor,
// which is what makes a partially-filled edge cell scale correctly.
//
// The cell index is computed in signed arithmetic, matching the CUDA kernel. That is deliberate
// and it is what makes the padding guard load-bearing: in signed division -1/k is 0, so an
// unguarded padding patch lands in cell 0 and quietly corrupts it. Casting to unsigned first would
// send it to a cell index no real cell can equal, which hides the bug AND makes the guard
// untestable -- removing it would then change nothing, so no check could ever catch its loss.
kernel void cpi_avg_pool_patches(device const half* in [[buffer(0)]],
                                 device const int* pos_x [[buffer(1)]],
                                 device const int* pos_y [[buffer(2)]],
                                 device half* out [[buffer(3)]],
                                 constant VisPoolParams& p [[buffer(4)]],
                                 uint gid [[thread_position_in_grid]]) {
  if (gid >= p.out_tokens * p.hidden) return;
  const uint cell = gid / p.hidden;
  const uint h = gid % p.hidden;

  float acc = 0.0f;
  const int ki = int(p.k), cxi = int(p.cells_x);
  for (uint t = 0u; t < p.tokens; ++t) {
    const int px = pos_x[t];
    const int py = pos_y[t];
    if (px < 0 || py < 0) continue;
    const int c = (px / ki) + cxi * (py / ki);
    if (c == int(cell)) acc += float(in[(ulong)t * (ulong)p.hidden + h]);
  }
  out[(ulong)cell * (ulong)p.hidden + h] = half(acc / float(p.k * p.k) * p.gain);
}

// x = (x - bias) * scale, elementwise over the hidden dim and broadcast across tokens.
kernel void cpi_standardize(device half* x [[buffer(0)]], device const half* bias [[buffer(1)]],
                            device const half* scale [[buffer(2)]],
                            constant VisStdParams& p [[buffer(3)]],
                            uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  const uint h = gid % p.hidden;
  x[gid] = half((float(x[gid]) - float(bias[h])) * float(scale[h]));
}

struct ClampParams {
  uint n;
  float lo, hi;
};

// out = clamp(in, lo, hi). Gemma 4 E2B's clipped linears (Gemma4ClippableLinear): HF clamps both
// the input and the output of every vision projection with per-projection bounds. in == out is
// allowed (the output clamp runs in place); the input clamp must go through a staging buffer
// instead, because one normalised activation feeds several projections with different bounds.
kernel void cpi_clamp(device const half* in [[buffer(0)]], device half* out [[buffer(1)]],
                      constant ClampParams& p [[buffer(2)]],
                      uint gid [[thread_position_in_grid]]) {
  if (gid >= p.n) return;
  out[gid] = half(clamp(float(in[gid]), p.lo, p.hi));
}
