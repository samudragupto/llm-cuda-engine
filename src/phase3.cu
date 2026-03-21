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

void TinyModel::embed(MemPool& scratch, IntTensor& ids, Tensor& x, int seq_len){k_embedding_lookup(ids.data,tok_embed.data,x.data,seq_len,dim);}

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

TransformerBlockFP16::TransformerBlockFP16(MemPool& pool,int s,int d,int h):
    dim(d),hidden_dim(h),seq(s),
    w_rms1(pool,{d}),w_rms2(pool,{d}),
    Wq(pool,{d,d}),Wk(pool,{d,d}),Wv(pool,{d,d}),Wo(pool,{d,d}),
    W1(pool,{d,h}),W2(pool,{h,d}) {}

void TransformerBlockFP16::init(){
    w_rms1.fill(1.0f); w_rms2.fill(1.0f);
    Tensor fq(*(MemPool*)nullptr,{});
}

TinyModelFP16::TinyModelFP16(MemPool& model_pool,int v,int s,int d,int h,int l):
    vocab(v),max_seq(s),dim(d),hidden(h),layers(l),
    tok_embed(model_pool,{v,d}),lm_head(model_pool,{d,v}),lm_bias(model_pool,{v}) {
    for(int i=0;i<layers;i++) blocks.push_back(new TransformerBlockFP16(model_pool,max_seq,dim,hidden));
    for(int i=0;i<layers;i++) caches.push_back(new LayerCacheFP16(model_pool,max_seq,dim));
}

void TinyModelFP16::init(){
    std::vector<float> te(vocab*dim),lh(dim*vocab),lb(vocab);
    for(size_t i=0;i<te.size();i++)te[i]=0.02f*((int)(i%23)-11);
    for(size_t i=0;i<lh.size();i++)lh[i]=0.02f*((int)(i%19)-9);
    for(size_t i=0;i<lb.size();i++)lb[i]=0.001f*((int)(i%7)-3);

    Tensor fte(*(MemPool*)nullptr,{}),flh(*(MemPool*)nullptr,{}),flb(*(MemPool*)nullptr,{});
}