#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "tensor.h"
#include "mem_pool.h"
#include "phase2.h"
#include "fp16_tensor.h"

struct IntTensor {
    int* data;
    size_t numel;
    IntTensor(): data(nullptr), numel(0) {}
    IntTensor(MemPool& pool,size_t n): numel(n) { data = pool.alloc<int>(n); }
    void from_host(const std::vector<int>& in) { cudaMemcpy(data,in.data(),in.size()*sizeof(int),cudaMemcpyHostToDevice); }
    void to_host(std::vector<int>& out) { out.resize(numel); cudaMemcpy(out.data(),data,numel*sizeof(int),cudaMemcpyDeviceToHost); }
};

struct TinyTokenizer {
    std::unordered_map<std::string,int> stoi;
    std::vector<std::string> itos;
    TinyTokenizer();
    std::vector<int> encode(const std::string& s);
    std::string decode(const std::vector<int>& ids);
};

struct LayerCache {
    Tensor K;
    Tensor V;
    LayerCache(MemPool& pool,int max_seq,int dim): K(pool,{max_seq,dim}),V(pool,{max_seq,dim}) {}
};

struct TransformerBlockFP16 {
    int dim, hidden_dim, seq;
    Tensor w_rms1, w_rms2;
    HalfTensor Wq, Wk, Wv, Wo;
    HalfTensor W1, W2;
    TransformerBlockFP16(MemPool& pool,int s,int d,int h):
        dim(d),hidden_dim(h),seq(s),
        w_rms1(pool,{d}),w_rms2(pool,{d}),
        Wq(pool,{d,d}),Wk(pool,{d,d}),Wv(pool,{d,d}),Wo(pool,{d,d}),
        W1(pool,{d,h}),W2(pool,{h,d}) {}
    void init();
    void forward_one(MemPool& scratch, Tensor& x, Tensor& out, Tensor& k_cache, Tensor& v_cache, int pos);
};

struct LayerCacheFP16 {
    Tensor K;
    Tensor V;
    LayerCacheFP16(MemPool& pool,int max_seq,int dim): K(pool,{max_seq,dim}),V(pool,{max_seq,dim}) {}
};

struct TinyModel {
    int vocab, max_seq, dim, hidden, layers;
    Tensor tok_embed;
    Tensor lm_head;
    Tensor lm_bias;
    std::vector<TransformerBlock*> blocks;
    std::vector<LayerCache*> caches;
    TinyModel(MemPool& model_pool,int v,int s,int d,int h,int l);
    void init();
    void embed(MemPool& scratch, IntTensor& ids, Tensor& x, int seq_len);
    void forward_full(MemPool& scratch, IntTensor& ids, Tensor& logits, int seq_len);
    void prefill(MemPool& scratch, const std::vector<int>& ids, Tensor& last_hidden);
    int logits_to_token(MemPool& scratch, Tensor& hidden);
    int decode_next(MemPool& scratch, int token_id, int pos);
    int generate_next_full(MemPool& scratch, const std::vector<int>& ids);
    std::vector<int> generate_full(MemPool& scratch, const std::vector<int>& prompt, int max_new_tokens);
    std::vector<int> generate_cached(MemPool& scratch, const std::vector<int>& prompt, int max_new_tokens);
};

struct TinyModelFP16 {
    int vocab, max_seq, dim, hidden, layers;
    HalfTensor tok_embed;
    HalfTensor lm_head;
    HalfTensor lm_bias;
    std::vector<TransformerBlockFP16*> blocks;
    std::vector<LayerCacheFP16*> caches;
    TinyModelFP16(MemPool& model_pool,int v,int s,int d,int h,int l);
    void init();
    void embed(MemPool& scratch, IntTensor& ids, Tensor& x, int seq_len);
    void prefill(MemPool& scratch, const std::vector<int>& ids, Tensor& last_hidden);
    int logits_to_token(MemPool& scratch, Tensor& hidden);
    int decode_next(MemPool& scratch, int token_id, int pos);
    std::vector<int> generate_cached(MemPool& scratch, const std::vector<int>& prompt, int max_new_tokens);
};

void test_embedding(MemPool& pool);
void test_argmax(MemPool& pool);
void test_tiny_tokenizer();
void test_tiny_model(MemPool& model_pool, MemPool& scratch);
void test_kv_cache_equivalence(MemPool& model_pool, MemPool& scratch);
void test_fp16_model(MemPool& model_pool, MemPool& scratch);
void bench_phase3(MemPool& model_pool, MemPool& scratch);
void bench_phase4a(MemPool& model_pool, MemPool& scratch);
void bench_phase4b(MemPool& scratch);