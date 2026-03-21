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

TinyModel::TinyModel(MemPool& pool,int v,int s,int d,int h,int l):
    vocab(v),seq(s),dim(d),hidden(h),layers(l),
    tok_embed(pool,{v,d}),lm_head(pool,{d,v}),lm_bias(pool,{v}) {
    for(int i=0;i<layers;i++) blocks.push_back(new TransformerBlock(pool,seq,dim,hidden));
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

void TinyModel::embed(MemPool& pool, IntTensor& ids, Tensor& x){
    k_embedding_lookup(ids.data,tok_embed.data,x.data,seq,dim);
}

void TinyModel::forward(MemPool& pool, IntTensor& ids, Tensor& logits){
    Tensor x(pool,{seq,dim}),tmp(pool,{seq,dim});
    embed(pool,ids,x);
    for(int i=0;i<layers;i++){
        blocks[i]->forward(pool,x,tmp);
        k_copy(tmp.data,x.data,seq*dim);
    }
    k_gemm_tiled(x.data,lm_head.data,logits.data,seq,vocab,dim);
    k_row_add_bias(logits.data,lm_bias.data,seq,vocab);
}

int TinyModel::generate_next(MemPool& pool, const std::vector<int>& ids){
    IntTensor dids(pool,seq);
    std::vector<int> padded(seq,0);
    int n=(int)ids.size();
    if(n>seq)n=seq;
    for(int i=0;i<n;i++) padded[i]=ids[i];
    dids.from_host(padded);
    Tensor logits(pool,{seq,vocab}),last(pool,{vocab});
    IntTensor out(pool,1);
    forward(pool,dids,logits);
    k_gather_last_token(logits.data,last.data,seq,vocab);
    k_argmax_row(last.data,out.data,1,vocab);
    cudaDeviceSynchronize();
    std::vector<int> h; out.to_host(h);
    return h[0];
}

std::vector<int> TinyModel::generate(MemPool& pool, const std::vector<int>& prompt, int max_new_tokens){
    std::vector<int> ids=prompt;
    for(int t=0;t<max_new_tokens;t++){
        int nxt=generate_next(pool,ids);
        ids.push_back(nxt);
        if((int)ids.size()>=seq)break;
        pool.reset();
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

void test_tiny_model(MemPool& pool){
    TinyModel m(pool,32,8,32,64,2);
    m.init();
    std::vector<int> prompt={3,4,7};
    int nxt=m.generate_next(pool,prompt);
    bool ok=nxt>=0&&nxt<32;
    chk3("tiny_model_generate_next",ok);
}

struct Tm3{
    cudaEvent_t s,e;
    Tm3(){cudaEventCreate(&s);cudaEventCreate(&e);}
    ~Tm3(){cudaEventDestroy(s);cudaEventDestroy(e);}
    void st(){cudaEventRecord(s);}
    float ed(){cudaEventRecord(e);cudaEventSynchronize(e);float ms;cudaEventElapsedTime(&ms,s,e);return ms;}
};

void bench_phase3(MemPool& pool){
    TinyModel m(pool,1000,32,64,128,2);
    m.init();
    std::vector<int> prompt={3,4,5,6,7};
    Tm3 t;
    int it=20;
    cudaDeviceSynchronize();
    t.st();
    for(int i=0;i<it;i++){
        int nxt=m.generate_next(pool,prompt);
        (void)nxt;
        pool.reset();
    }
    float ms=t.ed()/it;
    float tok_s=1000.0f/ms;
    printf("=== PHASE 3 TESTS ===\n");
    printf("Results: %d passed, %d failed\n\n",p3_pass,p3_fail);
    printf("=== PHASE 3 BENCHMARKS ===\n");
    printf("TinyModel generate_next: %.4f ms/token | %.2f tok/s\n\n",ms,tok_s);
}