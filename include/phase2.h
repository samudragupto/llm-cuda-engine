#pragma once
#include "tensor.h"
#include "mem_pool.h"

struct TransformerBlock {
    int dim, hidden_dim, seq;
    Tensor w_rms1, w_rms2;
    Tensor Wq, Wk, Wv, Wo;
    Tensor W1, W2;
    TransformerBlock(MemPool& pool,int s,int d,int h):
        dim(d),hidden_dim(h),seq(s),
        w_rms1(pool,{d}),w_rms2(pool,{d}),
        Wq(pool,{d,d}),Wk(pool,{d,d}),Wv(pool,{d,d}),Wo(pool,{d,d}),
        W1(pool,{d,h}),W2(pool,{h,d}) {}
    void init();
    void forward(MemPool& pool, Tensor& x, Tensor& out);
    void forward_one(MemPool& pool, Tensor& x, Tensor& out, Tensor& k_cache, Tensor& v_cache, int pos);
};

void test_rmsnorm(MemPool& pool);
void test_silu(MemPool& pool);
void test_softmax(MemPool& pool);
void test_rope(MemPool& pool);
void test_attention(MemPool& pool);
void test_block(MemPool& pool);
void bench_phase2(MemPool& pool);