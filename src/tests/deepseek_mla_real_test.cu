// TEST-DRIVEN validation of the DeepSeek MLA op (engine/mla_deepseek.hpp) against a REAL DeepSeek-V2-
// Lite layer: tools/deepseek_mla_oracle.py runs transformers' DeepseekV2Attention on layer 0's actual
// weights and dumps {hidden, weights, inv_freq, attn_out}; this loads that dump, runs mla_prefill_
// deepseek on the same real weights + input, and checks the output matches HF. This is the end-to-end
// proof that CPI's MLA == real DeepSeek MLA on real weights (not just self-consistent on toy configs).
//
// Reads <artifact>.dims (whitespace scalars) + <artifact>.bin (fp32 tensors in a fixed order). Pass the
// artifact stem as argv[1]; defaults to artifacts/deepseek_mla_ref (repo-root relative).
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include "engine/mla_deepseek.hpp"

namespace {
float* dev(const float* p, size_t n) {
  float* d = nullptr;
  cudaMalloc(&d, n * sizeof(float));
  cudaMemcpy(d, p, n * sizeof(float), cudaMemcpyHostToDevice);
  return d;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string stem = argc > 1 ? argv[1] : "artifacts/deepseek_mla_ref";

  std::ifstream df(stem + ".dims");
  if (!df) {
    std::printf("SKIP: %s.dims not found (run tools/deepseek_mla_oracle.py first)\n", stem.c_str());
    return 0;  // not a failure -- the artifact is generated out-of-band
  }
  engine::DSMLADims m;
  int seq = 0;
  df >> m.H >> m.nh >> m.qk_nope >> m.qk_rope >> m.v_head >> m.kv_lora >> seq >> m.softmax_scale >>
      m.attn_scaling >> m.rms_eps;
  const int QKHD = m.qkhd(), KVH = m.kvh(), KVA = m.kva();

  // Fixed tensor order in the .bin (see the oracle).
  const size_t n_hidden = (size_t)seq * m.H, n_attn = (size_t)seq * m.H,
               n_qproj = (size_t)m.nh * QKHD * m.H, n_kva = (size_t)KVA * m.H,
               n_kvaln = (size_t)m.kv_lora, n_kvb = (size_t)m.nh * KVH * m.kv_lora,
               n_oproj = (size_t)m.H * m.nh * m.v_head, n_invfreq = (size_t)m.qk_rope / 2;
  const size_t total =
      n_hidden + n_attn + n_qproj + n_kva + n_kvaln + n_kvb + n_oproj + n_invfreq;

  std::ifstream bf(stem + ".bin", std::ios::binary);
  std::vector<float> buf(total);
  bf.read(reinterpret_cast<char*>(buf.data()), total * sizeof(float));
  if (!bf) {
    std::printf("FAIL: could not read %s.bin (%zu floats)\n", stem.c_str(), total);
    return 1;
  }
  size_t o = 0;
  const float* hidden = &buf[o];
  o += n_hidden;
  const float* attn_ref = &buf[o];
  o += n_attn;
  const float* q_proj = &buf[o];
  o += n_qproj;
  const float* kv_a = &buf[o];
  o += n_kva;
  const float* kv_a_ln = &buf[o];
  o += n_kvaln;
  const float* kv_b = &buf[o];
  o += n_kvb;
  const float* o_proj = &buf[o];
  o += n_oproj;
  const float* inv_freq = &buf[o];

  engine::DSMLAWeights w;
  w.q_proj = dev(q_proj, n_qproj);
  w.kv_a = dev(kv_a, n_kva);
  w.kv_a_ln = dev(kv_a_ln, n_kvaln);
  w.kv_b = dev(kv_b, n_kvb);
  w.o_proj = dev(o_proj, n_oproj);
  w.inv_freq = dev(inv_freq, n_invfreq);
  float* d_hidden = dev(hidden, n_hidden);
  float* d_out = nullptr;
  cudaMalloc(&d_out, n_attn * sizeof(float));

  engine::mla_prefill_deepseek(m, w, d_hidden, seq, d_out);
  cudaDeviceSynchronize();
  const cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    std::printf("FAIL: CUDA error %s\n", cudaGetErrorString(err));
    return 1;
  }

  std::vector<float> out(n_attn);
  cudaMemcpy(out.data(), d_out, n_attn * sizeof(float), cudaMemcpyDeviceToHost);

  float maxabs = 0, denom = 1e-6f;
  for (size_t i = 0; i < n_attn; ++i) {
    maxabs = std::max(maxabs, std::fabs(attn_ref[i] - out[i]));
    denom = std::max(denom, std::fabs(attn_ref[i]));
  }
  const float rel = maxabs / denom;
  // fp32 throughout, but the reconstruction path accumulates over 2048-wide GEMVs, so allow a modest
  // fp32 tolerance. A structural mismatch (wrong rope convention, missing norm, etc.) would be >>1e-2.
  const bool pass = rel < 5e-4f;
  std::printf("%s[DeepSeek-V2-Lite MLA, layer 0]: CPI op vs transformers HF, seq=%d heads=%d "
              "qk=%d+%d v=%d, max rel %.2e\n",
              pass ? "PASS" : "FAIL", seq, m.nh, m.qk_nope, m.qk_rope, m.v_head, rel);
  return pass ? 0 : 1;
}
