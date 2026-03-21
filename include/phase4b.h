#pragma once
#include <vector>
#include "tensor.h"
#include "phase3.h"
#include "fp16_tensor.h"

struct TransformerBlockH {
    int dim, hidden_dim;
    Tensor w_rms1, w_rms2;
    HalfTensor Wq, Wk, Wv, Wo;
    HalfTensor W1, W2;
    TransformerBlockH(MemPool& pool,int d,int h);
    void init();
    void forward_one(MemPool& scratch, Tensor& x, Tensor& out, Tensor& k_cache, Tensor& v_cache, int pos);
};

struct TinyModelH {
    int vocab, max_seq, dim, hidden, layers;
    HalfTensor tok_embed;
    HalfTensor lm_head;
    HalfTensor lm_bias;
    std::vector<TransformerBlockH*> blocks;
    std::vector<LayerCache*> caches;
    TinyModelH(MemPool& pool,int v,int s,int d,int h,int l);
    void init();
    void prefill(MemPool& scratch, const std::vector<int>& ids, Tensor& last_hidden);
    int logits_to_token(MemPool& scratch, Tensor& hidden);
    int decode_next(MemPool& scratch, int token_id, int pos);
    std::vector<int> generate_cached(MemPool& scratch, const std::vector<int>& prompt, int max_new_tokens);
};

void test_fp16_kernels(MemPool& model_pool, MemPool& scratch);
void test_fp16_model(MemPool& model_pool, MemPool& scratch);
void bench_phase4b(MemPool& scratch);
void bench_phase4c(MemPool& scratch);