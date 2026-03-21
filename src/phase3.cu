#include "phase3.h"
#include "kernels.cuh"
#include "cuda_check.h"
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

static int p3_pass=0,p3_fail=0;
static void chk3(const char* n,bool ok){if(ok){printf("  [PASS] %s\n",n);p3_pass++;}else{printf("  [FAIL] %s\n",n);p3_fail++;}}

TinyTokenizer::TinyTokenizer(){
    itos={"<pad>","<bos>","<eos>","hello","world","i","am","cuda","llm","you","the","a",".",",","!","gpu","test","from","scratch","build"};
    for(int i=0;i<(int)itos.size();i++)stoi[itos[i]]=i;
}
std::vector<int> TinyTokenizer::encode(const std::string& s){
    std::vector<int> out;
    std::string cur;
    for(char c:s){
        if(c==' '){
            if(!cur.empty()){out.push_back(stoi.count(cur)?stoi[cur]:0);cur.clear();}
        }else cur.push_back(c);
    }
    if(!cur.empty())out.push_back(stoi.count(cur)?stoi[cur]:0);
    return out;
}
std::string TinyTokenizer::decode(const std::vector<int>& ids){
    std::string s;
    for(int i=0;i<(int)ids.size();i++){
        int id=ids[i];
        if(id>=0&&id<(int)itos.size()){
            if(!s.empty())s+=" ";
            s+=itos[id];
        }
    }
    return s;
}

TinyModel::TinyModel(MemPool& model_pool,int v,int s,int d,int h,int l):
    vocab(v),max_seq(s),dim(d),hidden(h),layers(l),
    tok_embed(model_pool,{v,d}),lm_head(model_pool,{d,v}),lm_bias(model_pool,{v}) {
    for(int i=0;i<layers;i++) blocks.push_back(new TransformerBlock(model_pool,max_seq,dim,hidden));
    for(int i=0;i<layers;i++) caches.push_back(new LayerCache(model_pool,max_seq,dim));
}

void TinyModel::init(){
    std::vector<float> te(tok_embed.numel),lh(lm_head.numel),lb(lm_bias.numel);
    for(size_t i=0;i<te.size();i++)te[i]=0.02f*((int)(i%23)-11);
    for(size_t i=0;i<lh.size();i++)lh[i]=0.02f*((int)(i%19)-9);
    for(size_t i=0;i<lb.size();i++)lb[i]=0.001f*((int)(i%7)-3);
    tok_embed.from_host(te);
    lm_head.from_host(lh);
    lm_bias.from_host(lb);
    for(int i=0;i<layers;i++)blocks[i]->init();
}

void TinyModel::embed(MemPool& scratch, IntTensor& ids, Tensor& x, int seq_len){
    k_embedding_lookup(ids.data,tok_embed.data,x.data,seq_len,dim);
}

void TinyModel::forward_full(MemPool& scratch, IntTensor& ids, Tensor& logits, int seq_len){
    Tensor x(scratch,{seq_len,dim}),tmp(scratch,{seq_len,dim});
    embed(scratch,ids,x,seq_len);
    for(int i=0;i<layers;i++){
        TransformerBlock b(scratch,seq_len,dim,hidden);
        CUDA_CHECK(cudaMemcpy(b.w_rms1.data,blocks[i]->w_rms1.data,dim*sizeof(float),cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(b.w_rms2.data,blocks[i]->w_rms2.data,dim*sizeof(float),cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(b.Wq.data,blocks[i]->Wq.data,dim*dim*sizeof(float),cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(b.Wk.data,blocks[i]->Wk.data,dim*dim*sizeof(float),cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(b.Wv.data,blocks[i]->Wv.data,dim*dim*sizeof(float),cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(b.Wo.data,blocks[i]->Wo.data,dim*dim*sizeof(float),cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(b.W1.data,blocks[i]->W1.data,dim*hidden*sizeof(float),cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(b.W2.data,blocks[i]->W2.data,hidden*dim*sizeof(float),cudaMemcpyDeviceToDevice));
        b.forward(scratch,x,tmp);
        k_copy(tmp.data,x.data,seq_len*dim);
    }
    k_gemm_tiled(x.data,lm_head.data,logits.data,seq_len,vocab,dim);
    k_row_add_bias(logits.data,lm_bias.data,seq_len,vocab);
}

void TinyModel::prefill(MemPool& scratch, const std::vector<int>& ids, Tensor& last_hidden){
    int n=(int)ids.size();
    for(int pos=0;pos<n;pos++){
        scratch.reset();
        IntTensor tid(scratch,1);
        std::vector<int> h={ids[pos]};
        tid.from_host(h);
        Tensor x(scratch,{1,dim}),tmp(scratch,{1,dim});
        k_embedding_lookup(tid.data,tok_embed.data,x.data,1,dim);
        for(int l=0;l<layers;l++){
            blocks[l]->forward_one(scratch,x,tmp,caches[l]->K,caches[l]->V,pos);
            k_copy(tmp.data,x.data,dim);
        }
        if(pos==n-1) CUDA_CHECK(cudaMemcpy(last_hidden.data,x.data,dim*sizeof(float),cudaMemcpyDeviceToDevice));
        cudaDeviceSynchronize();
    }
}

int TinyModel::logits_to_token(MemPool& scratch, Tensor& hidden){
    Tensor logits(scratch,{1,vocab});
    IntTensor out(scratch,1);
    k_gemm_tiled(hidden.data,lm_head.data,logits.data,1,vocab,dim);
    k_row_add_bias(logits.data,lm_bias.data,1,vocab);
    k_argmax_row(logits.data,out.data,1,vocab);
    cudaDeviceSynchronize();
    std::vector<int> h; out.to_host(h);
    return h[0];
}

int TinyModel::decode_next(MemPool& scratch, int token_id, int pos){
    scratch.reset();
    IntTensor tid(scratch,1);
    std::vector<int> h={token_id};
    tid.from_host(h);
    Tensor x(scratch,{1,dim}),tmp(scratch,{1,dim});
    k_embedding_lookup(tid.data,tok_embed.data,x.data,1,dim);
    for(int l=0;l<layers;l++){
        blocks[l]->forward_one(scratch,x,tmp,caches[l]->K,caches[l]->V,pos);
        k_copy(tmp.data,x.data,dim);
    }
    return logits_to_token(scratch,x);
}

int TinyModel::generate_next_full(MemPool& scratch, const std::vector<int>& ids){
    scratch.reset();
    int seq_len=(int)ids.size();
    IntTensor dids(scratch,seq_len);
    dids.from_host(ids);
    Tensor logits(scratch,{seq_len,vocab}),last(scratch,{1,vocab});
    IntTensor out(scratch,1);
    forward_full(scratch,dids,logits,seq_len);
    k_gather_last_token(logits.data,last.data,seq_len,vocab);
    k_argmax_row(last.data,out.data,1,vocab);
    cudaDeviceSynchronize();
    std::vector<int> h; out.to_host(h);
    return h[0];
}

std::vector<int> TinyModel::generate_full(MemPool& scratch, const std::vector<int>& prompt, int max_new_tokens){
    std::vector<int> ids=prompt;
    for(int t=0;t<max_new_tokens;t++){
        int nxt=generate_next_full(scratch,ids);
        ids.push_back(nxt);
        if((int)ids.size()>=max_seq)break;
    }
    return ids;
}

std::vector<int> TinyModel::generate_cached(MemPool& scratch, const std::vector<int>& prompt, int max_new_tokens){
    std::vector<int> ids=prompt;
    scratch.reset();
    Tensor last_hidden(scratch,{1,dim});
    prefill(scratch,prompt,last_hidden);
    scratch.reset();
    Tensor hidden0(scratch,{1,dim});
    CUDA_CHECK(cudaMemcpy(hidden0.data,last_hidden.data,dim*sizeof(float),cudaMemcpyDeviceToDevice));
    int nxt=logits_to_token(scratch,hidden0);
    ids.push_back(nxt);
    for(int t=1;t<max_new_tokens;t++){
        int pos=(int)ids.size()-1;
        nxt=decode_next(scratch,ids[pos],pos);
        ids.push_back(nxt);
        if((int)ids.size()>=max_seq)break;
    }
    return ids;
}

void test_embedding(MemPool& pool){
    Tensor table(pool,{4,3}),out(pool,{2,3});
    IntTensor ids(pool,2);
    std::vector<float> h={1,2,3,4,5,6,7,8,9,10,11,12};
    std::vector<int> hi={2,0};
    table.from_host(h); ids.from_host(hi);
    k_embedding_lookup(ids.data,table.data,out.data,2,3);
    cudaDeviceSynchronize();
    std::vector<float> ho; out.to_host(ho);
    bool ok=ho[0]==7&&ho[1]==8&&ho[2]==9&&ho[3]==1&&ho[4]==2&&ho[5]==3;
    chk3("embedding_lookup",ok);
}

void test_argmax(MemPool& pool){
    Tensor x(pool,{1,5});
    IntTensor out(pool,1);
    std::vector<float> h={1,9,3,2,8};
    x.from_host(h);
    k_argmax_row(x.data,out.data,1,5);
    cudaDeviceSynchronize();
    std::vector<int> ho; out.to_host(ho);
    chk3("argmax",ho[0]==1);
}

void test_tiny_tokenizer(){
    TinyTokenizer tok;
    auto ids=tok.encode("hello world cuda");
    std::string s=tok.decode(ids);
    bool ok=ids.size()==3&&s=="hello world cuda";
    chk3("tiny_tokenizer",ok);
}

void test_tiny_model(MemPool& model_pool, MemPool& scratch){
    TinyModel m(model_pool,32,8,32,64,2);
    m.init();
    std::vector<int> prompt={3,4,7};
    int nxt=m.generate_next_full(scratch,prompt);
    bool ok=nxt>=0&&nxt<32;
    chk3("tiny_model_generate_next",ok);
}

void test_kv_cache_equivalence(MemPool& model_pool, MemPool& scratch){
    TinyModel m(model_pool,32,12,32,64,2);
    m.init();
    std::vector<int> prompt={3,4,7};
    auto a=m.generate_full(scratch,prompt,3);
    scratch.reset();
    MemPool model_pool2(512ULL*1024*1024);
    TinyModel m2(model_pool2,32,12,32,64,2);
    m2.init();
    auto b=m2.generate_cached(scratch,prompt,3);
    bool ok=a.size()==b.size();
    if(ok) for(int i=0;i<(int)a.size();i++) if(a[i]!=b[i]) { ok=false; break; }
    chk3("kv_cache_equivalence",ok);
}

struct Tm3{
    cudaEvent_t s,e;
    Tm3(){cudaEventCreate(&s);cudaEventCreate(&e);}
    ~Tm3(){cudaEventDestroy(s);cudaEventDestroy(e);}
    void st(){cudaEventRecord(s);}
    float ed(){cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);return ms;}
};

void bench_phase3(MemPool& model_pool, MemPool& scratch){
    TinyModel m(model_pool,1000,32,64,128,2);
    m.init();
    std::vector<int> prompt={3,4,5,6,7};
    Tm3 t;
    int it=20;
    cudaDeviceSynchronize();
    t.st();
    for(int i=0;i<it;i++){
        int nxt=m.generate_next_full(scratch,prompt);
        (void)nxt;
    }
    float ms=t.ed()/it;
    float tok_s=1000.0f/ms;
    printf("=== PHASE 3 TESTS ===\n");
    printf("Results: %d passed, %d failed\n\n",p3_pass,p3_fail);
    printf("=== PHASE 3 BENCHMARKS ===\n");
    printf("TinyModel full-recompute next: %.4f ms/token | %.2f tok/s\n\n",ms,tok_s);
}

void bench_phase4a(MemPool& model_pool, MemPool& scratch){
    int it=20;
    std::vector<int> prompt={3,4,5,6,7,8,9,10};
    Tm3 t;

    cudaDeviceSynchronize();
    t.st();
    for(int i=0;i<it;i++){
        MemPool mp(512ULL*1024*1024);
        TinyModel a(mp,1000,32,64,128,2);
        a.init();
        auto out=a.generate_full(scratch,prompt,8);
        (void)out;
    }
    float ms_full=t.ed()/it;

    cudaDeviceSynchronize();
    t.st();
    for(int i=0;i<it;i++){
        MemPool mp(512ULL*1024*1024);
        TinyModel b(mp,1000,32,64,128,2);
        b.init();
        auto out=b.generate_cached(scratch,prompt,8);
        (void)out;
    }
    float ms_cached=t.ed()/it;

    printf("=== PHASE 4A BENCHMARKS ===\n");
    printf("Full recompute generate: %.4f ms\n",ms_full);
    printf("KV cache generate      : %.4f ms\n",ms_cached);
    printf("Speedup                : %.2fx\n\n",ms_full/ms_cached);
}