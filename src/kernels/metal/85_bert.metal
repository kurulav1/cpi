// BERT-family embedding encoder: the two kernels the rest of the tower does NOT already have.
// The rest maps onto kernels the Qwen3.5 vision tower already needed:
//
//   CUDA embed_kernel          -> cpi_bert_embed          (here)
//   CUDA layernorm_kernel      -> cpi_layernorm + cpi_add_inplace for the residual
//   CUDA add_bias_kernel       -> cpi_gemm_f16 applies its own bias
//   CUDA gelu_kernel           -> cpi_gelu_erf  (see the note below; NOT cpi_gelu)
//   CUDA attention_kernel      -> cpi_attention_bidirectional, written for the vision tower
//   CUDA pool_normalize_kernel -> cpi_bert_pool_normalize (here)

#include <metal_stdlib>
using namespace metal;

struct BertEmbedParams {
  uint tokens;   // L
  uint hidden;   // H
  uint type_id;  // which token_type row to add; BERT sentence-pair input would vary this
};

struct BertPoolParams {
  uint tokens;     // L
  uint hidden;     // H
  uint mean_pool;  // 1 = mean over tokens, 0 = CLS (row 0)
  uint normalize;  // 1 = L2-normalise the pooled vector
};

// out[t][d] = word[token[t]][d] + pos[t][d] + type[type_id][d]
//
// Position embeddings are absolute and looked up by index, not rotary: BERT's
// position_embedding_type is "absolute" and the table is a learned [max_pos][H]. Feeding a rotary
// path here would be silently wrong rather than an error.
//
// Accumulated in fp32 and rounded once, matching the CUDA kernel: three fp16 addends rounded
// pairwise would drift from the reference for no reason.
kernel void cpi_bert_embed(device const half* word_emb [[buffer(0)]],
                           device const half* pos_emb  [[buffer(1)]],
                           device const half* type_emb [[buffer(2)]],
                           device const int*  tokens   [[buffer(3)]],
                           device half*       out      [[buffer(4)]],
                           constant BertEmbedParams& p [[buffer(5)]],
                           uint gid [[thread_position_in_grid]]) {
  const uint n = p.tokens * p.hidden;
  if (gid >= n) return;
  const uint t = gid / p.hidden;
  const uint d = gid - t * p.hidden;
  const int tok = tokens[t];
  const float v = float(word_emb[(ulong)tok * (ulong)p.hidden + d]) +
                  float(pos_emb[(ulong)t * (ulong)p.hidden + d]) +
                  float(type_emb[(ulong)p.type_id * (ulong)p.hidden + d]);
  out[gid] = half(v);
}

// Pool [L][H] -> [H], then optionally L2-normalise. One threadgroup: the output is a single
// vector, and the normalisation needs every element before it can scale any of them.
//
// Output is f32, not half. This is the value that leaves the GPU and gets compared for cosine
// similarity, where fp16's ~3 decimal digits are visible in a ranking; the CUDA path makes the
// same choice.
kernel void cpi_bert_pool_normalize(device const half* x   [[buffer(0)]],
                                    device float*      out [[buffer(1)]],
                                    constant BertPoolParams& p [[buffer(2)]],
                                    uint lid  [[thread_position_in_threadgroup]],
                                    uint nthr [[threads_per_threadgroup]]) {
  threadgroup float red[256];

  for (uint d = lid; d < p.hidden; d += nthr) {
    float v;
    if (p.mean_pool != 0u) {
      float s = 0.0f;
      for (uint t = 0; t < p.tokens; ++t) s += float(x[(ulong)t * (ulong)p.hidden + d]);
      v = s / float(p.tokens);
    } else {
      v = float(x[d]);  // CLS is row 0
    }
    out[d] = v;
  }
  threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
  if (p.normalize == 0u) return;

  float local = 0.0f;
  for (uint d = lid; d < p.hidden; d += nthr) local += out[d] * out[d];
  red[lid] = local;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Tree reduction over the threadgroup. nthr is a power of two by construction (the host
  // dispatches 256); halving from nthr/2 is only correct under that, so the host must not
  // quietly pass something else.
  for (uint s = nthr / 2u; s > 0u; s >>= 1u) {
    if (lid < s) red[lid] += red[lid + s];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  // The +1e-12 is the CUDA path's, and it matters for an all-zero row: rsqrt(0) is inf and would
  // turn the whole vector into NaN rather than leaving it zero.
  const float inv = rsqrt(red[0] + 1e-12f);
  for (uint d = lid; d < p.hidden; d += nthr) out[d] *= inv;
}
