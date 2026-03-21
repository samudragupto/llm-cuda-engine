#pragma once
#include <string>
#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "tensor.h"
#include "mem_pool.h"

// Phase 3 Upgrade: Full Generation Config API
struct GenerationConfig {
    int max_new_tokens = 50;
    float temperature = 0.7f;
    float top_p = 0.9f;
    float repetition_penalty = 1.1f;
};

struct HalfTensor {
    half* data;
    int numel;
    std::vector<int> shape;
    HalfTensor(MemPool& pool, std::vector<int> s);
};

struct LlamaTokenizer {
    std::vector<std::string> vocab;
    void load(const char* path);
    std::string decode(int id);
    std::string decode(const std::vector<int>& ids);
};

struct LlamaLayerFP16 {
    int dim, hidden_dim, n_heads, n_kv_heads, head_dim, max_seq;
    HalfTensor w_rms1, w_rms2, Wq, Wk, Wv, Wo, W1, W2, W3, k_cache, v_cache;

    LlamaLayerFP16(MemPool& pool, int seq, int d, int hd, int nh, int nkv);
    void load(FILE* f);
    void forward_one(MemPool& scratch, cublasHandle_t handle, HalfTensor& x, HalfTensor& out, int pos);
};

struct Llama2FP16 {
    int vocab=32000, dim=2048, hidden=5632, layers=22, heads=32, kv_heads=4, max_seq=256;
    HalfTensor tok_embed, norm_w, lm_head;
    std::vector<LlamaLayerFP16*> transformer;
    LlamaTokenizer tokenizer;
    cublasHandle_t handle;

    Llama2FP16(MemPool& pool);
    void load_weights(const char* path);
    // Updated to accept config and past tokens for sampling/repetition penalty
    int generate_next(MemPool& scratch, int token_id, int pos, const GenerationConfig& cfg, const std::vector<int>& past_tokens);
    void chat(MemPool& scratch, const std::vector<int>& prompt_ids, GenerationConfig cfg);
};