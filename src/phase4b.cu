#include "phase4b.h"
#include "kernels.cuh"
#include "cuda_check.h"
#include <cstdio>
#include <vector>
#include <cmath>

static int p4b_pass=0,p4b_fail=0;
static void chk4(const char* n,bool ok){if(ok){printf("  [PASS] %s\n",n);p4b_pass++;}else{printf("  [FAIL] %s\n",n);p4b_fail++;}}

TransformerBlockH::TransformerBlockH(MemPool& pool,int d,int h):
    dim(d),hidden_dim(h),
    w_rms1(pool,{d}),w_rms2(pool,{d}),
    Wq(pool,{d,d}),Wk(pool,{d,d}),Wv(pool,{d,d}),Wo(pool,{d,d}),
    W1(pool,{d,h}),W2(pool,{h,d}) {}

void TransformerBlockH::init(){
    w_rms1.fill(1.0f); w_rms2.fill(1.0f);
    std::vector<float> hq(Wq.numel),hk(Wk.numel),hv(Wv.numel),ho(Wo.numel),h1(W1.numel),h2(W2.numel);
    for(size_t i=0;i<hq.size();i++)hq[i]=0.02f*((int)(i%7)-3);
    for(size_t i=0;i<hk.size();i++)hk[i]=0.02f*((int)(i%5)-2);
    for(size_t i=0;i<hv.size();i++)hv[i]=0.02f*((int)(i%11)-5);
    for(size_t i=0;i<ho.size();i++)ho[i]=0.02f*((int)(i%13)-6);
    for(size_t i=0;i<h1.size();i++)h1[i]=0.02f*((int)(i%9)-4);
    for(size_t i=0;i<h2.size();i++)h2[i]=0.02f*((int)(i%15)-7);

    MemPool tmp(64ULL*1024*1024);
    Tensor fq(tmp,{dim,dim}),fk(tmp,{dim,dim}),fv(tmp,{dim,dim}),fo(tmp,{dim,dim}),f1(tmp,{dim,hidden_dim}),f2(tmp,{hidden_dim,dim});
    fq.from_host(hq); fk.from_host(hk); fv.from_host(hv); fo.from_host(ho); f1.from_host(h1); f2.from_host(h2);
    k_fp32_to_fp16(fq.data,Wq.data,(int)Wq.numel);
    k_fp32_to_fp16(fk.data,Wk.data,(int)Wk.numel);
    k_fp32_to_fp16(fv.data,Wv.data,(int)Wv.numel);
    k_fp32_to_fp16(fo.data,Wo.data,(int)Wo.numel);
    k_fp32_to_fp16(f1.data,W1.data,(int)W1.numel);
    k_fp32_to_fp16(f2.data,W2.data,(int)W2.numel);
    cudaDeviceSynchronize();
}

void TransformerBlockH::forward_one(MemPool& scratch, Tensor& x, Tensor& out, Tensor& k_cache, Tensor& v_cache, int pos){
    Tensor xn(scratch,{1,dim}),Q(scratch,{1,dim}),K(scratch,{1,dim}),V(scratch,{1,dim}),S(scratch,{1,pos+1}),P(scratch,{1,pos+1}),A(scratch,{1,dim}),AO(scratch,{1,dim});
    Tensor fn(scratch,{1,dim}),H(scratch,{1,hidden_dim}),Hs(scratch,{1,hidden_dim}),F(scratch,{1,dim}),tmp(scratch,{1,dim});
    k_rmsnorm(x.data,w_rms1.data,xn.data,1,dim,1e-5f);
    k_gemm_tiled_hf(xn.data,Wq.data,Q.data,1,dim,dim);
    k_gemm_tiled_hf(xn.data,Wk.data,K.data,1,dim,dim);
    k_gemm_tiled_hf(xn.data,Wv.data,V.data,1,dim,dim);
    k_rope(Q.data,1,dim,pos);
    k_rope(K.data,1,dim,pos);
    k_copy_row_to_cache(K.data,k_cache.data,pos,dim);
    k_copy_row_to_cache(V.data,v_cache.data,pos,dim);
    k_attention_scores_one(Q.data,k_cache.data,S.data,pos+1,dim);
    k_row_softmax(S.data,P.data,1,pos+1);
    k_attention_weighted_sum_one(P.data,v_cache.data,A.data,pos+1,dim);
    k_gemm_tiled_hf(A.data,Wo.data,AO.data,1,dim,dim);
    k_add(x.data,AO.data,tmp.data,dim);
    k_rmsnorm(tmp.data,w_rms2.data,fn.data,1,dim,1e-5f);
    k_gemm_tiled_hf(fn.data,W1.data,H.data,1,hidden_dim,dim);
    k_silu(H.data,Hs.data,hidden_dim);
    k_gemm_tiled_hf(Hs.data,W2.data,F.data,1,dim,hidden_dim);
    k_add(tmp.data,F.data,out.data,dim);
}

TinyModelH::TinyModelH(MemPool& pool,int v,int s,int d,int h,int l):
    vocab(v),max_seq(s),dim(d),hidden(h),layers(l),
    tok_embed(pool,{v,d}),lm_head(pool,{d,v}),lm_bias(pool,{v}) {
    for(int i=0;i<layers;i++) blocks.push_back(new TransformerBlockH(pool,dim,hidden));
    for(int i=0;i<layers;i++) caches.push_back(new LayerCache(pool,max_seq,dim));
}

void TinyModelH::init(){
    std::vector<float> te(vocab*dim),lh(dim*vocab),lb(vocab);
    for(size_t i=0;i<te.size();i++)te[i]=0.02f*((int)(i%23)-11);
    for(size_t i=0;i<lh.size();i++)lh[i]=0.02f*((int)(i%19)-9);
    for(size_t i=0;i<lb.size();i++)lb[i]=0.001f*((int)(i%7)-3);

    MemPool tmp(128ULL*1024*1024);
    Tensor fte(tmp,{vocab,dim}),flh(tmp,{dim,vocab}),flb(tmp,{vocab});
    fte.from_host(te); flh.from_host(lh); flb.from_host(lb);
    k_fp32_to_fp16(fte.data,tok_embed.data,(int)tok_embed.numel);
    k_fp32_to_fp16(flh.data,lm_head.data,(int)lm_head.numel);
    k_fp32_to_fp16(flb.data,lm_bias.data,(int)lm_bias.numel);
    cudaDeviceSynchronize();

    for(int i=0;i<layers;i++) blocks[i]->init();
}

void TinyModelH::prefill(MemPool& scratch, const std::vector<int>& ids, Tensor& last_hidden){
    int n=(int)ids.size();
    for(int pos=0;pos<n;pos++){
        scratch.reset();
        IntTensor tid(scratch,1);
        std::vector<int> h={ids[pos]};
        tid.from_host(h);
        Tensor x(scratch,{1,dim}),tmp(scratch,{1,dim});
        k_embedding_lookup_h(tid.data,tok_embed.data,x.data,1,dim);
        for(int l=0;l<layers;l++){
            blocks[l]->forward_one(scratch,x,tmp,caches[l]->K,caches[l]->V,pos);
            k_copy(tmp.data,x.data,dim);
        }
        if(pos==n-1) CUDA_CHECK(cudaMemcpy(last_hidden.data,x.data,dim*sizeof(float),cudaMemcpyDeviceToDevice));
        cudaDeviceSynchronize();
    }
}

int TinyModelH::logits_to_token(MemPool& scratch, Tensor& hidden){
    Tensor logits(scratch,{1,vocab});
    IntTensor out(scratch,1);
    k_gemm_tiled_hf(hidden.data,lm_head.data,logits.data,1,vocab,dim);
    k_row_add_bias_h(logits.data,lm_bias.data,1,vocab);
    k_argmax_row(logits.data,out.data,1,vocab);
    cudaDeviceSynchronize();
    std::vector<int> h; out.to_host(h);
    return h[0];
}

int TinyModelH::decode_next(MemPool& scratch, int token_id, int pos){
    scratch.reset();
    IntTensor tid(scratch,1);
    std::vector<int> h={token_id};
    tid.from_host(h);
    Tensor x(scratch,{1,dim}),tmp(scratch,{1,dim});
    k_embedding_lookup_h(tid.data,tok_embed.data,x.data,1,dim);
    for(int l=0;l<layers;l++){
        blocks[l]->forward_one(scratch,x,tmp,caches[l]->K,caches[l]->V,pos);
        k_copy(tmp.data,x.data,dim);
    }
    return logits_to_token(scratch,x);
}

std::vector<int> TinyModelH::generate_cached(MemPool& scratch, const std::vector<int>& prompt, int max_new_tokens){
    std::vector<int> ids=prompt;
    scratch.reset();
    Tensor last_hidden(scratch,{1,dim});
    prefill(scratch,prompt,last_hidden);
    Tensor keep(scratch,{1,dim});
    CUDA_CHECK(cudaMemcpy(keep.data,last_hidden.data,dim*sizeof(float),cudaMemcpyDeviceToDevice));
    int nxt=logits_to_token(scratch,keep);
    ids.push_back(nxt);
    for(int t=1;t<max_new_tokens;t++){
        int pos=(int)ids.size()-1;
        nxt=decode_next(scratch,ids[pos],pos);
        ids.push_back(nxt);
        if((int)ids.size()>=max_seq)break;
    }
    return ids;
}

void test_fp16_kernels(MemPool& model_pool, MemPool& scratch){
    Tensor f(scratch,{8});
    HalfTensor h(model_pool,{8});
    std::vector<float> x={1,2,3,4,5,6,7,8};
    f.from_host(x);
    k_fp32_to_fp16(f.data,h.data,8);
    Tensor back(scratch,{8});
    k_fp16_to_fp32(h.data,back.data,8);
    cudaDeviceSynchronize();
    std::vector<float> y; back.to_host(y);
    bool ok=true; for(int i=0;i<8;i++) if(fabs(y[i]-x[i])>1e-2f) ok=false;
    chk4("fp32_fp16_roundtrip",ok);
}

void test_fp16_model(MemPool& model_pool, MemPool& scratch){
    TinyModelH m(model_pool,32,12,32,64,2);
    m.init();
    std::vector<int> prompt={3,4,7};
    auto out=m.generate_cached(scratch,prompt,3);
    bool ok=out.size()==6;
    chk4("fp16_cached_generation",ok);
}

struct T4{
    cudaEvent_t s,e;
    T4(){cudaEventCreate(&s);cudaEventCreate(&e);}
    ~T4(){cudaEventDestroy(s);cudaEventDestroy(e);}
    void st(){cudaEventRecord(s);}
    float ed(){cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);return ms;}
};

void bench_phase4b(MemPool& scratch){
    std::vector<int> prompt={3,4,5,6,7,8,9,10};
    int it=20;
    T4 t;

    cudaDeviceSynchronize();
    t.st();
    for(int i=0;i<it;i++){
        MemPool mp(512ULL*1024*1024);
        TinyModel a(mp,1000,32,64,128,2);
        a.init();
        auto out=a.generate_cached(scratch,prompt,8);
        (void)out;
    }
    float ms_fp32=t.ed()/it;

    cudaDeviceSynchronize();
    t.st();
    for(int i=0;i<it;i++){
        MemPool mp(512ULL*1024*1024);
        TinyModelH b(mp,1000,32,64,128,2);
        b.init();
        auto out=b.generate_cached(scratch,prompt,8);
        (void)out;
    }
    float ms_fp16=t.ed()/it;

    printf("=== PHASE 4B TESTS ===\n");
    printf("Results: %d passed, %d failed\n\n",p4b_pass,p4b_fail);
    printf("=== PHASE 4B BENCHMARKS ===\n");
    printf("FP32 cached generate: %.4f ms\n",ms_fp32);
    printf("FP16 cached generate: %.4f ms\n",ms_fp16);
    printf("Speedup            : %.2fx\n\n",ms_fp32/ms_fp16);
}