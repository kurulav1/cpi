// Metal compute shaders for the op-plan decode path.
//
// These mirror the CUDA kernels op-for-op (see include/runtime/kernels.cuh); the
// op-plan IR in include/engine/op_plan.hpp is what both backends execute, so the
// semantics here must match the CUDA ones exactly -- same normalisation, same RoPE
// convention, same causal masking. Numerics are NOT bit-identical across backends
// (different accumulation order), so correctness is judged against the CPU
// reference engine, not against CUDA bit-for-bit.
//
// Conventions shared with the CUDA side:
//   - weights are row-major fp16, out_dim x in_dim
//   - the residual stream and all slots are fp16; reductions accumulate in fp32
//   - RoPE is the half-split (NeoX) convention: element i pairs with i + head_dim/2
//   - the KV cache is [max_context][kv_heads * head_dim], fp16

#include <metal_stdlib>

using namespace metal;

// Tokens per threadgroup in the GEMV. The weight row is read once and reused across
// the tile, so this is the prefill's bandwidth-amplification factor. 8 fits in
// registers comfortably; decode uses a single token and is unaffected.
#define GEMV_TILE 8

// Keys scored per pass in the attention kernel. The online softmax folds in one block at
// a time, so this is the barrier-amortisation factor.
#define KEY_BLOCK 32

// The matrix-unit prefill attention kernel uses a WIDER key block than the scalar/decode paths:
// its per-block cost (three barriers + the softmax reductions) is amortised over the keys in the
// block, and attention is the prefill bottleneck (measured ~3x llama.cpp), so a 64-key block halves
// the iteration count. The softmax then covers 64 keys with 32 lanes (two keys per lane). Separate
// from KEY_BLOCK because the scalar kernel keeps one-key-per-lane and the paged block alignment is
// still stated in KEY_BLOCK units.
#define MM_KEY_BLOCK 128

// ---------------------------------------------------------------------------
// Parameter blocks. One per op family, bound at buffer(N) as a `constant` ref.
// Kept flat and POD so the C++ side can memcpy them without a shared header.
// ---------------------------------------------------------------------------

struct NormParams {
  uint rows;      // number of independent groups (tokens x groups-per-token)
  uint cols;      // elements per group
  float eps;
  uint weight_offset;  // 1 => scale by (1 + w) [Gemma], 0 => scale by w
  uint has_weight;     // 0 => weightless (ones)
};

// True LayerNorm: subtracts the mean and scales by weight, then adds bias. RMSNorm does
// neither, so this cannot be folded into NormParams -- a bias pointer has nowhere to go there.
struct LayerNormParams {
  uint rows;
  uint cols;
  float eps;
  uint has_bias;  // 0 => weight-only (the bias buffer is still bound but never read)
};

struct GemvParams {
  uint out_dim;
  uint in_dim;
  uint tokens;    // T: batched over tokens (sequence prefill); 1 for decode
  uint has_bias;  // 0 => the bias buffer is bound but never read (Llama); 1 => Qwen2's Q/K/V
};

struct RopeParams {
  uint heads;
  uint head_dim;
  uint position;   // used when positions_buf is not bound
  uint tokens;
  float theta;
  uint use_position_buffer;  // read the position from a device buffer instead
  uint row_stride;           // elements between consecutive tokens in the slot
  // Batched decode: every row is a DIFFERENT sequence, so row t takes positions[t] rather
  // than base+t. The default (0) keeps the prefill meaning, where the rows are consecutive
  // tokens of one sequence.
  uint per_row_positions;
};

struct ElemParams {
  uint n;       // total elements
  float scale;
};

struct KvParams {
  uint kv_heads;
  uint head_dim;
  uint position;  // position of the FIRST token in the batch
  uint max_context;
  uint use_position_buffer;
  uint tokens;  // T: 1 for decode, N for a prefill chunk
};

struct AttnParams {
  uint heads;
  uint kv_heads;
  uint head_dim;
  uint position;  // position of the FIRST query token (0-based)
  uint max_context;
  uint window;  // 0 = full causal; else sliding window length
  float scale;  // usually 1/sqrt(head_dim)
  uint use_position_buffer;
  uint tokens;  // T: 1 for decode, N for a prefill chunk
  // Paged prefill (cpi_attention_prefill_mm only; the others ignore these). 0 = the KV is one
  // contiguous run and a key's row IS its position.
  uint paged;
  uint block_size;  // tokens per KV block; only meaningful when paged != 0
};

struct QuantParams {
  uint out_dim;
  uint in_dim;
  uint tokens;
  uint bits;      // 4 or 8
  uint group;     // 0 = one scale per row
  uint groups;    // scales per row
  uint has_bias;  // Qwen2's Q/K/V carry one; quantizing the weights does not remove it
};

struct EmbedParams {
  uint hidden;
  uint tokens;
};

// ---------------------------------------------------------------------------
// RMSNorm -- one threadgroup per row. x is cached in registers only when it
// fits; otherwise re-read. Reduction is fp32 in threadgroup memory.
// ---------------------------------------------------------------------------

