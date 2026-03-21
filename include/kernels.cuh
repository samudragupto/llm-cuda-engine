#pragma once
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