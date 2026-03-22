#include "phase2.h"
#include "kernels.cuh"
#include "cuda_check.h"
#include <vector>
#include <cstdio>
#include <cmath>
#include <cuda_runtime.h>

static int p2_pass=0,p2_fail=0;
static void chk(const char* n,bool ok){if(ok){printf("  [PASS] %s\n",n);p2_pass++;}else{printf("  [FAIL] %s\n",n);p2_fail++;}}

void TransformerBlock::init(){
    w_rms1.fill(1.0f); w_rms2.fill(1.0f);
    std::vector<float> hq(Wq.numel),hk(Wk.numel),hv(Wv.numel),ho(Wo.numel),h1(W1.numel),h2(W2.numel);
    for(size_t i=0;i<hq.size();i++)hq[i]=0.02f*((int)(i%7)-3);
    for(size_t i=0;i<hk.size();i++)hk[i]=0.02f*((int)(i%5)-2);
    for(size_t i=0;i<hv.size();i++)hv[i]=0.02f*((int)(i%11)-5);
    for(size_t i=0;i<ho.size();i++)ho[i]=0.02f*((int)(i%13)-6);
    for(size_t i=0;i<h1.size();i++)h1[i]=0.02f*((int)(i%9)-4);
    for(size_t i=0;i<h2.size();i++)h2[i]=0.02f*((int)(i%15)-7);
    Wq.from_host(hq); Wk.from_host(hk); Wv.from_host(hv); Wo.from_host(ho); W1.from_host(h1); W2.from_host(h2);
}

void TransformerBlock::forward(MemPool& pool, Tensor& x, Tensor& out){
    Tensor xn(pool,{seq,dim}),Q(pool,{seq,dim}),K(pool,{seq,dim}),V(pool,{seq,dim}),S(pool,{seq,seq}),P(pool,{seq,seq}),A(pool,{seq,dim}),AO(pool,{seq,dim});
    Tensor fn(pool,{seq,dim}),H(pool,{seq,hidden_dim}),Hs(pool,{seq,hidden_dim}),F(pool,{seq,dim}),tmp(pool,{seq,dim});
    k_rmsnorm(x.data,w_rms1.data,xn.data,seq,dim,1e-5f);
    k_gemm_tiled(xn.data,Wq.data,Q.data,seq,dim,dim);
    k_gemm_tiled(xn.data,Wk.data,K.data,seq,dim,dim);
    k_gemm_tiled(xn.data,Wv.data,V.data,seq,dim,dim);
    k_rope(Q.data,seq,dim,0);
    k_rope(K.data,seq,dim,0);
    k_attention_scores(Q.data,K.data,S.data,seq,dim);
    k_apply_causal_mask(S.data,seq);
    k_row_softmax(S.data,P.data,seq,seq);
    k_attention_weighted_sum(P.data,V.data,A.data,seq,dim);
    k_gemm_tiled(A.data,Wo.data,AO.data,seq,dim,dim);
    k_add(x.data,AO.data,tmp.data,seq*dim);
    k_rmsnorm(tmp.data,w_rms2.data,fn.data,seq,dim,1e-5f);
    k_gemm_tiled(fn.data,W1.data,H.data,seq,hidden_dim,dim);
    k_silu(H.data,Hs.data,seq*hidden_dim);
    k_gemm_tiled(Hs.data,W2.data,F.data,seq,dim,hidden_dim);
    k_add(tmp.data,F.data,out.data,seq*dim);
}

void test_rmsnorm(MemPool& pool){
    Tensor x(pool,{2,4}),w(pool,{4}),y(pool,{2,4});
    std::vector<float> hx={1,2,3,4,2,2,2,2},hw={1,1,1,1};
    x.from_host(hx); w.from_host(hw);
    k_rmsnorm(x.data,w.data,y.data,2,4,1e-5f);
    cudaDeviceSynchronize();
    std::vector<float> hy; y.to_host(hy);
    float s0=1.0f/std::sqrt((1+4+9+16)/4.0f+1e-5f),s1=1.0f/std::sqrt((4+4+4+4)/4.0f+1e-5f);
    bool ok=fabs(hy[0]-1*s0)<1e-3&&fabs(hy[3]-4*s0)<1e-3&&fabs(hy[4]-2*s1)<1e-3;
    chk("rmsnorm",ok);
}

void test_silu(MemPool& pool){
    Tensor x(pool,{4}),y(pool,{4});
    std::vector<float> hx={-1,0,1,2};
    x.from_host(hx);
    k_silu(x.data,y.data,4);
    cudaDeviceSynchronize();
    std::vector<float> hy; y.to_host(hy);
    auto f=[](float v){return v/(1.0f+std::exp(-v));};
    bool ok=fabs(hy[0]-f(-1))<1e-4&&fabs(hy[1]-f(0))<1e-4&&fabs(hy[3]-f(2))<1e-4;
    chk("silu",ok);
}

void test_softmax(MemPool& pool){
    Tensor x(pool,{2,3}),y(pool,{2,3});
    std::vector<float> hx={1,2,3,1,1,1};
    x.from_host(hx);
    k_row_softmax(x.data,y.data,2,3);
    cudaDeviceSynchronize();
    std::vector<float> hy; y.to_host(hy);
    float s=std::exp(1)+std::exp(2)+std::exp(3);
    bool ok=fabs(hy[0]-std::exp(1)/s)<1e-4&&fabs(hy[2]-std::exp(3)/s)<1e-4&&fabs(hy[3]-1.0f/3.0f)<1e-4;
    chk("row_softmax",ok);
}

void test_rope(MemPool& pool){
    Tensor x(pool,{2,4});
    std::vector<float> hx={1,0,1,0,1,0,1,0};
    x.from_host(hx);
    k_rope(x.data,2,4,0);
    cudaDeviceSynchronize();
    std::vector<float> hy; x.to_host(hy);
    bool ok=fabs(hy[0]-1.0f)<1e-4&&fabs(hy[1]-0.0f)<1e-4;
    chk("rope",ok);
}

void test_attention(MemPool& pool){
    int seq=4,dim=4;
    Tensor Q(pool,{seq,dim}),K(pool,{seq,dim}),V(pool,{seq,dim}),S(pool,{seq,seq}),P(pool,{seq,seq}),O(pool,{seq,dim});
    std::vector<float> h(seq*dim,0.0f);
    for(int i=0;i<seq;i++)for(int d=0;d<dim;d++)h[i*dim+d]=(i==d)?1.0f:0.1f;
    Q.from_host(h); K.from_host(h); V.from_host(h);
    k_attention_scores(Q.data,K.data,S.data,seq,dim);
    k_apply_causal_mask(S.data,seq);
    k_row_softmax(S.data,P.data,seq,seq);
    k_attention_weighted_sum(P.data,V.data,O.data,seq,dim);
    cudaDeviceSynchronize();
    std::vector<float> ho; O.to_host(ho);
    bool ok=true;
    for(int i=0;i<seq*dim;i++) if(!std::isfinite(ho[i])) ok=false;
    chk("naive_attention",ok);
}

void test_block(MemPool& pool){
    int seq=8,dim=64,h=128;
    TransformerBlock b(pool,seq,dim,h);
    b.init();
    Tensor x(pool,{seq,dim}),o(pool,{seq,dim});
    std::vector<float> hx(seq*dim);
    for(int i=0;i<seq*dim;i++)hx[i]=0.01f*((i%17)-8);
    x.from_host(hx);
    b.forward(pool,x,o);
    cudaDeviceSynchronize();
    std::vector<float> ho; o.to_host(ho);
    bool ok=true;
    for(float v:ho) if(!std::isfinite(v)) { ok=false; break; }
    chk("transformer_block_forward",ok);
}

struct Tm{
    cudaEvent_t s,e;
    Tm(){cudaEventCreate(&s);cudaEventCreate(&e);}
    ~Tm(){cudaEventDestroy(s);cudaEventDestroy(e);}
    void st(){cudaEventRecord(s);}
    float ed(){cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);return ms;}
};

void bench_phase2(MemPool& pool){
    int seq=128,dim=256,h=512,it=20;
    Tensor x(pool,{seq,dim}),w(pool,{dim}),y(pool,{seq,dim}),S(pool,{seq,seq}),P(pool,{seq,seq}),Q(pool,{seq,dim}),K(pool,{seq,dim}),V(pool,{seq,dim}),O(pool,{seq,dim});
    x.fill(1.0f); w.fill(1.0f); Q.fill(0.1f); K.fill(0.2f); V.fill(0.3f);
    Tm t;
    cudaDeviceSynchronize(); t.st(); for(int i=0;i<it;i++)k_rmsnorm(x.data,w.data,y.data,seq,dim,1e-5f); float ms1=t.ed()/it;
    cudaDeviceSynchronize(); t.st(); for(int i=0;i<it;i++)k_silu(x.data,y.data,seq*dim); float ms2=t.ed()/it;
    cudaDeviceSynchronize(); t.st(); for(int i=0;i<it;i++)k_row_softmax(S.data,P.data,seq,seq); float ms3=t.ed()/it;
    cudaDeviceSynchronize(); t.st(); for(int i=0;i<it;i++){k_attention_scores(Q.data,K.data,S.data,seq,dim);k_apply_causal_mask(S.data,seq);k_row_softmax(S.data,P.data,seq,seq);k_attention_weighted_sum(P.data,V.data,O.data,seq,dim);} float ms4=t.ed()/it;
    printf("=== PHASE 2 TESTS ===\n");
    printf("Results: %d passed, %d failed\n\n",p2_pass,p2_fail);
    printf("=== PHASE 2 BENCHMARKS ===\n");
    printf("RMSNorm   [%d x %d] : %.4f ms\n",seq,dim,ms1);
    printf("SiLU      [%d] : %.4f ms\n",seq*dim,ms2);
    printf("Softmax   [%d x %d] : %.4f ms\n",seq,seq,ms3);
    printf("Attention [%d x %d] : %.4f ms\n\n",seq,dim,ms4);
}