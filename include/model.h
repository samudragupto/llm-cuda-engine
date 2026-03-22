#pragma once
#include <vector>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include "tensor.h"
#include "tokenizer.h"

struct GenerationConfig { int max_new_tokens = 50; float repetition_penalty = 1.1f; };

struct LlamaLayerMixed {
    int dim, hidden_dim, n_heads, n_kv_heads, head_dim, max_seq;
    HalfTensor w_rms1, w_rms2, Wq, Wk, Wv, Wo, k_cache, v_cache;
    QuantizedTensor W1, W2, W3;

    LlamaLayerMixed(MemPool& pool, int seq, int d, int hd, int nh, int nkv);
    void load(FILE* f);
    void forward_prefill(MemPool& scratch, cublasHandle_t handle, HalfTensor& x, int seq_len);
    void forward_decode_graph(MemPool& scratch, HalfTensor& x, int* d_pos);
};

struct Llama2MixedGraph {
    int vocab=32000, dim=2048, hidden=5632, layers=22, heads=32, kv_heads=4, max_seq=256;
    HalfTensor tok_embed, norm_w, lm_head;
    std::vector<LlamaLayerMixed*> transformer;
    LlamaTokenizer tokenizer;
    cublasHandle_t handle; 
    cudaStream_t stream;
    cudaGraph_t graph;
    cudaGraphExec_t graph_exec = nullptr;
    int* d_pos; int* d_token; int* d_out;

    Llama2MixedGraph(MemPool& pool);
    void load_weights(const char* path);
    void prefill(MemPool& scratch, const std::vector<int>& prompt_ids);
    void capture_graph(MemPool& scratch);
    void chat(MemPool& scratch, const std::vector<int>& prompt_ids, GenerationConfig cfg);
};