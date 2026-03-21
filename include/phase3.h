#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "tensor.h"
#include "mem_pool.h"
#include "phase2.h"

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

struct GenerationConfig {
    float temperature = 1.0f;
    float top_p = 1.0f;
    int max_new_tokens = 50;
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
    
    // Updated sampling signatures
    int logits_to_token(MemPool& scratch, Tensor& hidden, GenerationConfig config);
    int decode_next(MemPool& scratch, int token_id, int pos, GenerationConfig config);
    int generate_next_full(MemPool& scratch, const std::vector<int>& ids, GenerationConfig config);
    
    std::vector<int> generate_full(MemPool& scratch, const std::vector<int>& prompt, GenerationConfig config);
    std::vector<int> generate_cached(MemPool& scratch, const std::vector<int>& prompt, GenerationConfig config);
};

void test_embedding(MemPool& pool);
void test_argmax(MemPool& pool);
void test_tiny_tokenizer();
void test_tiny_model(MemPool& model_pool, MemPool& scratch);
void test_kv_cache_equivalence(MemPool& model_pool, MemPool& scratch);
void bench_phase3(MemPool& model_pool, MemPool& scratch);
void bench_phase4a(MemPool& model_pool, MemPool& scratch);