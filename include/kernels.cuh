#pragma once
#include <cuda_fp16.h>
void k_add(float* a, float* b, float* c, int n);
void k_mul(float* a, float* b, float* c, int n);
void k_scale(float* a, float s, float* c, int n);
void k_fill(float* a, float val, int n);
void k_copy(float* src, float* dst, int n);
void k_gemm_naive(float* A, float* B, float* C, int M, int N, int K);
void k_gemm_tiled(float* A, float* B, float* C, int M, int N, int K);
void k_transpose(float* in, float* out, int rows, int cols);
void k_rmsnorm(float* x, float* w, float* y, int rows, int cols, float eps);
void k_silu(float* x, float* y, int n);
void k_row_softmax(float* x, float* y, int rows, int cols);
void k_rope(float* x, int seq, int dim, int pos_base);
void k_attention_scores(float* Q, float* K, float* S, int seq, int dim);
void k_apply_causal_mask(float* S, int seq);
void k_attention_weighted_sum(float* P, float* V, float* O, int seq, int dim);
void k_embedding_lookup(int* ids, float* table, float* out, int seq, int dim);
void k_gather_last_token(float* x, float* out, int seq, int dim);
void k_argmax_row(float* x, int* out, int rows, int cols);
void k_row_add_bias(float* x, float* b, int rows, int cols);
void k_copy_row_to_cache(float* src_row, float* cache, int pos, int dim);
void k_attention_scores_one(float* q, float* K, float* s, int len, int dim);
void k_attention_weighted_sum_one(float* p, float* V, float* o, int len, int dim);

void k_fp32_to_fp16(float* src, half* dst, int n);
void k_fp16_to_fp32(half* src, float* dst, int n);
void k_copy_h(half* src, half* dst, int n);
void k_embedding_lookup_h(int* ids, half* table, float* out, int seq, int dim);
void k_row_add_bias_h(float* x, half* b, int rows, int cols);
void k_gemm_tiled_hf(float* A, half* B, float* C, int M, int N, int K);

void k_gemv(float* x, float* W, float* y, int K, int N);
void k_gemv_hf(float* x, half* W, float* y, int K, int N);