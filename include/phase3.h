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

struct KVCache {
    Tensor K;
    Tensor V;
    int max_seq, dim;
    KVCache(MemPool& pool,int ms,int d): K(pool,{ms,d}), V(pool,{ms,d}), max_seq(ms), dim(d) {}
};

struct TinyModel {
    int vocab, seq, dim, hidden, layers;
    Tensor tok_embed;
    Tensor lm_head;
    Tensor lm_bias;
    std::vector<TransformerBlock*> blocks;
    TinyModel(MemPool& pool,int v,int s,int d,int h,int l);
    void init();
    void embed(MemPool& pool, IntTensor& ids, Tensor& x);
    void forward(MemPool& pool, IntTensor& ids, Tensor& logits);
    int generate_next(MemPool& pool, const std::vector<int>& ids);
    std::vector<int> generate(MemPool& pool, const std::vector<int>& prompt, int max_new_tokens);
};

void test_embedding(MemPool& pool);
void test_argmax(MemPool& pool);
void test_tiny_tokenizer();
void test_tiny_model(MemPool& pool);
void bench_phase3(MemPool& pool);