#include <cuda_fp16.h>

#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "engine/bert_embedder.hpp"

namespace engine {
namespace {

#define CUDA_CHECK(x)                                                                        \
  do {                                                                                       \
    const cudaError_t e_ = (x);                                                              \
    if (e_ != cudaSuccess) {                                                                 \
      throw std::runtime_error(std::string("bert cuda: ") + cudaGetErrorString(e_) + " @ " + \
                               __FILE__ + ":" + std::to_string(__LINE__));                   \
    }                                                                                        \
  } while (0)

#define CUBLAS_CHECK(x)                                                                 \
  do {                                                                                  \
    const cublasStatus_t s_ = (x);                                                      \
    if (s_ != CUBLAS_STATUS_SUCCESS) {                                                  \
      throw std::runtime_error("bert cublas error " + std::to_string((int)s_) + " @ " + \
                               std::to_string(__LINE__));                               \
    }                                                                                   \
  } while (0)

// Portable host float -> IEEE half (round-to-nearest-even-ish; fine for weights).
std::uint16_t f32_to_f16(float f) {
  std::uint32_t x;
  std::memcpy(&x, &f, sizeof(x));
  const std::uint32_t sign = (x >> 16) & 0x8000u;
  std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xFF) - 127 + 15;
  std::uint32_t mant = x & 0x7FFFFFu;
  if (((x >> 23) & 0xFF) == 0xFF) {  // inf/nan
    return static_cast<std::uint16_t>(sign | 0x7C00u | (mant ? 0x200u : 0));
  }
  if (exp >= 0x1F) {  // overflow -> inf
    return static_cast<std::uint16_t>(sign | 0x7C00u);
  }
  if (exp <= 0) {  // subnormal/underflow
    if (exp < -10) return static_cast<std::uint16_t>(sign);
    mant |= 0x800000u;
    const int shift = 14 - exp;
    std::uint32_t h = mant >> shift;
    if ((mant >> (shift - 1)) & 1u) ++h;  // round
    return static_cast<std::uint16_t>(sign | h);
  }
  std::uint16_t h =
      static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13));
  if (mant & 0x1000u) ++h;  // round to nearest
  return h;
}

// h[i*H + d] = word_emb[tok[i]*H+d] + pos_emb[i*H+d] + type_emb[d]  (type 0)
__global__ void embed_kernel(const half* word_emb, const half* pos_emb, const half* type_emb,
                             const int* tokens, half* out, int L, int H) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= L * H) return;
  const int i = idx / H;
  const int d = idx % H;
  const int tok = tokens[i];
  const float v = __half2float(word_emb[tok * H + d]) + __half2float(pos_emb[i * H + d]) +
                  __half2float(type_emb[d]);
  out[idx] = __float2half(v);
}

// LayerNorm(input + residual) per row. One block per row, blockDim threads loop H.
__global__ void layernorm_kernel(const half* input, const half* residual, const half* w,
                                 const half* b, half* out, int H, float eps) {
  const int row = blockIdx.x;
  const half* xin = input + static_cast<std::size_t>(row) * H;
  const half* res = residual ? residual + static_cast<std::size_t>(row) * H : nullptr;
  half* yout = out + static_cast<std::size_t>(row) * H;
  extern __shared__ float sh[];  // blockDim floats for reduction
  // mean
  float local = 0.0f;
  for (int d = threadIdx.x; d < H; d += blockDim.x) {
    float v = __half2float(xin[d]);
    if (res) v += __half2float(res[d]);
    local += v;
  }
  sh[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  const float mean = sh[0] / H;
  __syncthreads();
  // variance
  local = 0.0f;
  for (int d = threadIdx.x; d < H; d += blockDim.x) {
    float v = __half2float(xin[d]);
    if (res) v += __half2float(res[d]);
    const float c = v - mean;
    local += c * c;
  }
  sh[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = rsqrtf(sh[0] / H + eps);
  for (int d = threadIdx.x; d < H; d += blockDim.x) {
    float v = __half2float(xin[d]);
    if (res) v += __half2float(res[d]);
    const float n = (v - mean) * inv;
    yout[d] = __float2half(n * __half2float(w[d]) + __half2float(b[d]));
  }
}

__global__ void add_bias_kernel(half* y, const half* bias, int rows, int cols) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= rows * cols) return;
  y[idx] = __float2half(__half2float(y[idx]) + __half2float(bias[idx % cols]));
}

__global__ void gelu_kernel(half* y, int n) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const float x = __half2float(y[idx]);
  // exact GELU: 0.5 x (1 + erf(x / sqrt(2)))
  y[idx] = __float2half(0.5f * x * (1.0f + erff(x * 0.7071067811865476f)));
}

// Bidirectional multi-head attention. grid=(num_heads, L), blockDim=32 (a warp);
// the warp loops over head_dim and the sequence. q/k/v/att are [L, num_heads*head_dim].
__global__ void attention_kernel(const half* q, const half* k, const half* v, half* att, int L,
                                 int num_heads, int head_dim) {
  const int head = blockIdx.x;
  const int i = blockIdx.y;
  const int lane = threadIdx.x;  // 0..31
  const int H = num_heads * head_dim;
  const int hbase = head * head_dim;
  extern __shared__ float smem[];
  float* q_sh = smem;               // head_dim
  float* scores = smem + head_dim;  // L
  for (int d = lane; d < head_dim; d += 32) {
    q_sh[d] = __half2float(q[i * H + hbase + d]);
  }
  __syncwarp();
  const float scale = rsqrtf(static_cast<float>(head_dim));
  // scores[j] = (q_i . k_j) * scale
  for (int j = 0; j < L; ++j) {
    float p = 0.0f;
    for (int d = lane; d < head_dim; d += 32) {
      p += q_sh[d] * __half2float(k[j * H + hbase + d]);
    }
    for (int o = 16; o > 0; o >>= 1) p += __shfl_down_sync(0xFFFFFFFFu, p, o);
    if (lane == 0) scores[j] = p * scale;
  }
  __syncwarp();
  // softmax over scores[0..L)
  float m = -1e30f;
  for (int j = lane; j < L; j += 32) m = fmaxf(m, scores[j]);
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xFFFFFFFFu, m, o));
  float sum = 0.0f;
  for (int j = lane; j < L; j += 32) {
    const float e = __expf(scores[j] - m);
    scores[j] = e;
    sum += e;
  }
  for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xFFFFFFFFu, sum, o);
  const float inv = 1.0f / sum;
  __syncwarp();
  // out[d] = sum_j p[j] * v[j,d]
  for (int d = lane; d < head_dim; d += 32) {
    float acc = 0.0f;
    for (int j = 0; j < L; ++j) {
      acc += scores[j] * __half2float(v[j * H + hbase + d]);
    }
    att[i * H + hbase + d] = __float2half(acc * inv);
  }
}

// Pools (CLS=row0 or mean over rows) then optionally L2-normalizes -> fp32 out[H].
__global__ void pool_normalize_kernel(const half* x, float* out, int L, int H, int mean_pool,
                                      int normalize) {
  // single block, blockDim threads over H
  extern __shared__ float red[];
  for (int d = threadIdx.x; d < H; d += blockDim.x) {
    float v;
    if (mean_pool) {
      float s = 0.0f;
      for (int i = 0; i < L; ++i) s += __half2float(x[i * H + d]);
      v = s / L;
    } else {
      v = __half2float(x[d]);  // CLS = row 0
    }
    out[d] = v;
  }
  __syncthreads();
  if (!normalize) return;
  float local = 0.0f;
  for (int d = threadIdx.x; d < H; d += blockDim.x) local += out[d] * out[d];
  red[threadIdx.x] = local;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float inv = rsqrtf(red[0] + 1e-12f);
  for (int d = threadIdx.x; d < H; d += blockDim.x) out[d] *= inv;
}

}  // namespace

BertEmbedder::~BertEmbedder() {
  destroy();
}

void* BertEmbedder::upload_fp16(const std::string& name, std::size_t expected_elems) {
  if (!weights_.has_tensor(name)) {
    throw std::runtime_error("bert: missing tensor " + name);
  }
  const std::size_t bytes = weights_.tensor_bytes(name);
  if (bytes != expected_elems * sizeof(float)) {
    throw std::runtime_error("bert: tensor " + name + " expected " +
                             std::to_string(expected_elems) + " fp32 elems, got " +
                             std::to_string(bytes / sizeof(float)));
  }
  const float* src = reinterpret_cast<const float*>(weights_.tensor_ptr(name));
  std::vector<std::uint16_t> half_host(expected_elems);
  for (std::size_t i = 0; i < expected_elems; ++i) half_host[i] = f32_to_f16(src[i]);
  void* dptr = nullptr;
  CUDA_CHECK(cudaMalloc(&dptr, expected_elems * sizeof(half)));
  CUDA_CHECK(
      cudaMemcpy(dptr, half_host.data(), expected_elems * sizeof(half), cudaMemcpyHostToDevice));
  return dptr;
}

void BertEmbedder::linear(const void* w, const void* bias, const void* x, void* y, int out, int in,
                          int rows) {
  const float alpha = 1.0f, beta = 0.0f;
  // y[rows,out] = x[rows,in] @ W[out,in]^T  (row-major). See derivation in commit.
  CUBLAS_CHECK(cublasGemmEx(cublas_, CUBLAS_OP_T, CUBLAS_OP_N, out, rows, in, &alpha, w, CUDA_R_16F,
                            in, x, CUDA_R_16F, in, &beta, y, CUDA_R_16F, out, CUBLAS_COMPUTE_32F,
                            CUBLAS_GEMM_DEFAULT));
  if (bias) {
    const int n = rows * out;
    add_bias_kernel<<<(n + 255) / 256, 256, 0, stream_>>>(
        static_cast<half*>(y), static_cast<const half*>(bias), rows, out);
  }
}

void BertEmbedder::initialize(const std::string& model_dir) {
  cfg_ = EmbeddingConfig::load(model_dir);
  if (cfg_.model_type != "bert" || cfg_.position_embedding_type != "absolute") {
    throw std::runtime_error(
        "bert embedder: only absolute-position BERT-family models are "
        "supported (got model_type=" +
        cfg_.model_type + ")");
  }
  weights_.open(model_dir);
  CUBLAS_CHECK(cublasCreate(&cublas_));
  CUDA_CHECK(cudaStreamCreate(&stream_));
  CUBLAS_CHECK(cublasSetStream(cublas_, stream_));

  const int H = cfg_.hidden_size, I = cfg_.intermediate_size;
  word_emb_ = upload_fp16("embeddings.word_embeddings.weight",
                          static_cast<std::size_t>(cfg_.vocab_size) * H);
  pos_emb_ = upload_fp16("embeddings.position_embeddings.weight",
                         static_cast<std::size_t>(cfg_.max_position_embeddings) * H);
  type_emb_ = upload_fp16("embeddings.token_type_embeddings.weight",
                          static_cast<std::size_t>(cfg_.type_vocab_size) * H);
  emb_ln_w_ = upload_fp16("embeddings.LayerNorm.weight", H);
  emb_ln_b_ = upload_fp16("embeddings.LayerNorm.bias", H);

  layers_.resize(cfg_.num_layers);
  for (int l = 0; l < cfg_.num_layers; ++l) {
    const std::string p = "encoder.layer." + std::to_string(l) + ".";
    LayerWeights& w = layers_[l];
    w.q_w = upload_fp16(p + "attention.self.query.weight", static_cast<std::size_t>(H) * H);
    w.q_b = upload_fp16(p + "attention.self.query.bias", H);
    w.k_w = upload_fp16(p + "attention.self.key.weight", static_cast<std::size_t>(H) * H);
    w.k_b = upload_fp16(p + "attention.self.key.bias", H);
    w.v_w = upload_fp16(p + "attention.self.value.weight", static_cast<std::size_t>(H) * H);
    w.v_b = upload_fp16(p + "attention.self.value.bias", H);
    w.o_w = upload_fp16(p + "attention.output.dense.weight", static_cast<std::size_t>(H) * H);
    w.o_b = upload_fp16(p + "attention.output.dense.bias", H);
    w.attn_ln_w = upload_fp16(p + "attention.output.LayerNorm.weight", H);
    w.attn_ln_b = upload_fp16(p + "attention.output.LayerNorm.bias", H);
    w.inter_w = upload_fp16(p + "intermediate.dense.weight", static_cast<std::size_t>(I) * H);
    w.inter_b = upload_fp16(p + "intermediate.dense.bias", I);
    w.out_w = upload_fp16(p + "output.dense.weight", static_cast<std::size_t>(H) * I);
    w.out_b = upload_fp16(p + "output.dense.bias", H);
    w.out_ln_w = upload_fp16(p + "output.LayerNorm.weight", H);
    w.out_ln_b = upload_fp16(p + "output.LayerNorm.bias", H);
  }

  const int M = cfg_.max_tokens;
  CUDA_CHECK(cudaMalloc(&d_tokens_, static_cast<std::size_t>(M) * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_x_, static_cast<std::size_t>(M) * H * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_tmp_, static_cast<std::size_t>(M) * H * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_q_, static_cast<std::size_t>(M) * H * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_k_, static_cast<std::size_t>(M) * H * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_v_, static_cast<std::size_t>(M) * H * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_att_, static_cast<std::size_t>(M) * H * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_inter_, static_cast<std::size_t>(M) * I * sizeof(half)));
  CUDA_CHECK(cudaMalloc(&d_pooled_, static_cast<std::size_t>(H) * sizeof(float)));
  h_pooled_.resize(H);
}

std::vector<float> BertEmbedder::embed(const std::vector<int>& token_ids) {
  const int H = cfg_.hidden_size, I = cfg_.intermediate_size;
  const int head_dim = H / cfg_.num_heads;
  int L = static_cast<int>(token_ids.size());
  if (L < 1) L = 1;
  if (L > cfg_.max_tokens) L = cfg_.max_tokens;
  CUDA_CHECK(cudaMemcpyAsync(d_tokens_, token_ids.data(), static_cast<std::size_t>(L) * sizeof(int),
                             cudaMemcpyHostToDevice, stream_));

  const int lnThreads = 256;
  const std::size_t lnSh = lnThreads * sizeof(float);
  const auto ln = [&](const void* in, const void* res, const void* w, const void* b, void* out) {
    layernorm_kernel<<<L, lnThreads, lnSh, stream_>>>(
        static_cast<const half*>(in), static_cast<const half*>(res), static_cast<const half*>(w),
        static_cast<const half*>(b), static_cast<half*>(out), H, cfg_.layer_norm_eps);
  };

  // Embeddings + LayerNorm.
  {
    const int n = L * H;
    embed_kernel<<<(n + 255) / 256, 256, 0, stream_>>>(
        static_cast<const half*>(word_emb_), static_cast<const half*>(pos_emb_),
        static_cast<const half*>(type_emb_), d_tokens_, static_cast<half*>(d_tmp_), L, H);
    ln(d_tmp_, nullptr, emb_ln_w_, emb_ln_b_, d_x_);  // d_x_ = hidden state
  }

  const dim3 attnGrid(cfg_.num_heads, L);
  const std::size_t attnSh = (static_cast<std::size_t>(head_dim) + L) * sizeof(float);
  for (int l = 0; l < cfg_.num_layers; ++l) {
    const LayerWeights& w = layers_[l];
    // Self-attention.
    linear(w.q_w, w.q_b, d_x_, d_q_, H, H, L);
    linear(w.k_w, w.k_b, d_x_, d_k_, H, H, L);
    linear(w.v_w, w.v_b, d_x_, d_v_, H, H, L);
    attention_kernel<<<attnGrid, 32, attnSh, stream_>>>(
        static_cast<const half*>(d_q_), static_cast<const half*>(d_k_),
        static_cast<const half*>(d_v_), static_cast<half*>(d_att_), L, cfg_.num_heads, head_dim);
    linear(w.o_w, w.o_b, d_att_, d_tmp_, H, H, L);     // attn output dense -> d_tmp_
    ln(d_tmp_, d_x_, w.attn_ln_w, w.attn_ln_b, d_x_);  // d_x_ = LN(d_tmp_ + d_x_)
    // FFN.
    linear(w.inter_w, w.inter_b, d_x_, d_inter_, I, H, L);
    gelu_kernel<<<(L * I + 255) / 256, 256, 0, stream_>>>(static_cast<half*>(d_inter_), L * I);
    linear(w.out_w, w.out_b, d_inter_, d_tmp_, H, I, L);
    ln(d_tmp_, d_x_, w.out_ln_w, w.out_ln_b, d_x_);  // d_x_ = LN(d_tmp_ + d_x_)
  }

  // Pool + L2-normalize.
  const int poolThreads = 256;
  pool_normalize_kernel<<<1, poolThreads, poolThreads * sizeof(float), stream_>>>(
      static_cast<const half*>(d_x_), d_pooled_, L, H, cfg_.pooling == PoolingMode::Mean ? 1 : 0,
      cfg_.normalize ? 1 : 0);

  CUDA_CHECK(cudaMemcpyAsync(h_pooled_.data(), d_pooled_,
                             static_cast<std::size_t>(H) * sizeof(float), cudaMemcpyDeviceToHost,
                             stream_));
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  if (cfg_.dimension == H) {
    return h_pooled_;
  }
  return std::vector<float>(h_pooled_.begin(), h_pooled_.begin() + cfg_.dimension);
}

void BertEmbedder::destroy() {
  const auto free_d = [](void*& p) {
    if (p) {
      cudaFree(p);
      p = nullptr;
    }
  };
  free_d(word_emb_);
  free_d(pos_emb_);
  free_d(type_emb_);
  free_d(emb_ln_w_);
  free_d(emb_ln_b_);
  for (LayerWeights& w : layers_) {
    free_d(w.q_w);
    free_d(w.q_b);
    free_d(w.k_w);
    free_d(w.k_b);
    free_d(w.v_w);
    free_d(w.v_b);
    free_d(w.o_w);
    free_d(w.o_b);
    free_d(w.attn_ln_w);
    free_d(w.attn_ln_b);
    free_d(w.inter_w);
    free_d(w.inter_b);
    free_d(w.out_w);
    free_d(w.out_b);
    free_d(w.out_ln_w);
    free_d(w.out_ln_b);
  }
  layers_.clear();
  free_d(d_x_);
  free_d(d_tmp_);
  free_d(d_q_);
  free_d(d_k_);
  free_d(d_v_);
  free_d(d_att_);
  free_d(d_inter_);
  if (d_tokens_) {
    cudaFree(d_tokens_);
    d_tokens_ = nullptr;
  }
  if (d_pooled_) {
    cudaFree(d_pooled_);
    d_pooled_ = nullptr;
  }
  if (cublas_) {
    cublasDestroy(cublas_);
    cublas_ = nullptr;
  }
  if (stream_) {
    cudaStreamDestroy(stream_);
    stream_ = nullptr;
  }
}

}  // namespace engine
