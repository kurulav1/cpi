// Parity for the two remaining batched-decode primitives (P2):
//   - per-position RoPE: with contiguous positions [base..base+N) it must equal
//     the existing contiguous batched RoPE (start_position=base).
//   - batched KV scatter: each sequence's token must land at its block table's
//     physical slot (block_tables[b][pos/bs]*bs + pos%bs).
#include "runtime/kernels.cuh"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#define CK(x) do { cudaError_t _e=(x); if(_e!=cudaSuccess){ std::printf("CUDA %s@%d: %s\n",#x,__LINE__,cudaGetErrorString(_e)); return 2;} } while(0)

int main() {
  int fails = 0;
  const int num_tokens = 5, nhq = 28, nhk = 4, head_dim = 128, half_dim = head_dim / 2;
  const int base = 37, max_pos = 4096;
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> uni(-1.f, 1.f);

  // RoPE tables.
  std::vector<float> cosT(static_cast<std::size_t>(max_pos) * half_dim), sinT(cosT.size());
  for (int p = 0; p < max_pos; ++p) for (int i = 0; i < half_dim; ++i) {
    const float ang = p / powf(10000.f, (2.f * i) / head_dim);
    cosT[p * half_dim + i] = cosf(ang); sinT[p * half_dim + i] = sinf(ang);
  }
  auto rnd = [&](std::size_t n){ std::vector<half> h(n); for (auto& x : h) x = __float2half(uni(rng)); return h; };
  std::vector<half> q = rnd(static_cast<std::size_t>(num_tokens) * nhq * head_dim);
  std::vector<half> k = rnd(static_cast<std::size_t>(num_tokens) * nhk * head_dim);
  std::vector<int> positions(num_tokens); for (int i = 0; i < num_tokens; ++i) positions[i] = base + i;

  half *dq1,*dk1,*dq2,*dk2; float *dcos,*dsin; int *dpos;
  CK(cudaMalloc(&dq1,q.size()*2)); CK(cudaMalloc(&dk1,k.size()*2));
  CK(cudaMalloc(&dq2,q.size()*2)); CK(cudaMalloc(&dk2,k.size()*2));
  CK(cudaMalloc(&dcos,cosT.size()*4)); CK(cudaMalloc(&dsin,sinT.size()*4)); CK(cudaMalloc(&dpos,num_tokens*4));
  CK(cudaMemcpy(dq1,q.data(),q.size()*2,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dk1,k.data(),k.size()*2,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dq2,q.data(),q.size()*2,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dk2,k.data(),k.size()*2,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dcos,cosT.data(),cosT.size()*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dsin,sinT.data(),sinT.size()*4,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dpos,positions.data(),num_tokens*4,cudaMemcpyHostToDevice));

  kernels::launch_rope_inplace_batched(dq1,dk1,num_tokens,nhq,nhk,head_dim,base,dcos,dsin,0);
  kernels::launch_rope_inplace_perpos(dq2,dk2,num_tokens,nhq,nhk,head_dim,dpos,dcos,dsin,0);
  CK(cudaDeviceSynchronize());
  std::vector<half> q1(q.size()),q2(q.size());
  CK(cudaMemcpy(q1.data(),dq1,q.size()*2,cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(q2.data(),dq2,q.size()*2,cudaMemcpyDeviceToHost));
  double rope_max=0; for (std::size_t i=0;i<q1.size();++i) rope_max=fmax(rope_max,fabs((double)__half2float(q1[i])-__half2float(q2[i])));
  std::printf("%s  per-pos RoPE == contiguous batched (max_diff=%.3e)\n", rope_max<1e-3?"ok  ":"FAIL", rope_max);
  if (rope_max>=1e-3) ++fails;

  // Batched KV scatter.
  const int batch = 3, kv_hidden = nhk * head_dim, block_size = 32, max_blocks = 8, pool_blocks = 40;
  std::vector<int> bpos = {30, 65, 200};                       // straddle/interior positions
  std::vector<int> btab(static_cast<std::size_t>(batch) * max_blocks, 0);
  for (int b=0;b<batch;++b) for (int c=0;c<max_blocks;++c) btab[b*max_blocks+c] = (b*7 + c*3 + 1) % pool_blocks; // arbitrary distinct-ish
  std::vector<half> ksrc = rnd(static_cast<std::size_t>(batch)*kv_hidden), vsrc = rnd(static_cast<std::size_t>(batch)*kv_hidden);
  std::vector<half> pool(static_cast<std::size_t>(pool_blocks)*block_size*kv_hidden, __float2half(0.f));
  half *dkp,*dvp,*dks,*dvs; int *dbt,*dbp;
  CK(cudaMalloc(&dkp,pool.size()*2)); CK(cudaMalloc(&dvp,pool.size()*2));
  CK(cudaMalloc(&dks,ksrc.size()*2)); CK(cudaMalloc(&dvs,vsrc.size()*2));
  CK(cudaMalloc(&dbt,btab.size()*4)); CK(cudaMalloc(&dbp,batch*4));
  CK(cudaMemcpy(dkp,pool.data(),pool.size()*2,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dvp,pool.data(),pool.size()*2,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dks,ksrc.data(),ksrc.size()*2,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dvs,vsrc.data(),vsrc.size()*2,cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dbt,btab.data(),btab.size()*4,cudaMemcpyHostToDevice)); CK(cudaMemcpy(dbp,bpos.data(),batch*4,cudaMemcpyHostToDevice));
  kernels::launch_store_kv_batched_paged(dkp,dvp,dks,dvs,dbt,dbp,max_blocks,batch,kv_hidden,block_size,0);
  CK(cudaDeviceSynchronize());
  std::vector<half> poolk(pool.size()); CK(cudaMemcpy(poolk.data(),dkp,pool.size()*2,cudaMemcpyDeviceToHost));
  int scatter_bad=0;
  for (int b=0;b<batch;++b){ const int pos=bpos[b]; const int phys=btab[b*max_blocks+pos/block_size]*block_size+(pos%block_size);
    for (int d=0; d<kv_hidden; ++d){ const float got=__half2float(poolk[(std::size_t)phys*kv_hidden+d]); const float want=__half2float(ksrc[(std::size_t)b*kv_hidden+d]); if (fabs(got-want)>1e-6) ++scatter_bad; } }
  std::printf("%s  batched KV scatter lands at block-table slots (%d mismatched elems)\n", scatter_bad==0?"ok  ":"FAIL", scatter_bad);
  if (scatter_bad) ++fails;

  std::printf("\npaged_batched_primitives: %s (%d failures)\n", fails==0?"PASS":"FAIL", fails);
  return fails==0?0:1;
}
