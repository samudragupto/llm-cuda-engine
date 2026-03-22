#pragma once
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "tensor.h"

struct GenerationConfig { int max_new_tokens = 50; float temperature = 0.7f; float top_p = 0.9f; float repetition_penalty = 1.1f; };
struct LlamaTokenizer { std::vector<std::string> vocab; void load(const char* path); std::string decode(int id); std::string decode(const std::vector<int>& ids); };

struct LlamaLayerMixed {
    int dim, hidden_dim, n_heads, n_kv_heads, head_dim, max_seq;
    HalfTensor w_rms1, w_rms2, Wq, Wk, Wv, Wo, k_cache, v_cache;
    QuantizedTensor W1, W2, W3;

    LlamaLayerMixed(MemPool& pool, int seq, int d, int hd, int nh, int nkv);
    void load(FILE* f);
    void forward_prefill(MemPool& scratch, cublasHandle_t handle, HalfTensor& x, HalfTensor& out, int seq_len);
    void forward_decode(MemPool& scratch, HalfTensor& x, HalfTensor& out, int pos);
};

struct Llama2Mixed {
    int vocab=32000, dim=2048, hidden=5632, layers=22, heads=32, kv_heads=4, max_seq=256;
    HalfTensor tok_embed, norm_w, lm_head;
    std::vector<LlamaLayerMixed*> transformer;
    LlamaTokenizer tokenizer;
    cublasHandle_t handle; 

    Llama2Mixed(MemPool& pool);
    void load_weights(const char* path);
    void prefill(MemPool& scratch, const std::vector<int>& prompt_ids);
    int decode_next(MemPool& scratch, int pos, const GenerationConfig& cfg, const std::vector<int>& past_tokens);
    void chat(MemPool& scratch, const std::vector<int>& prompt_ids, GenerationConfig cfg);
};