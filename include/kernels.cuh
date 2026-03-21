#pragma once
#include <cuda_fp16.h>
#include <cublas_v2.h>

void k_add(float* a, float* b, float* c, int n);
void k_mul(float* a, float* b, float* c, int n);
void k_scale(float* a, float s, float* c, int n);
void k_fill(float* a, float val, int n);
void k_copy(float* src, float* dst, int n);

void k_gemm_naive(float* A, float* B, float* C, int M, int N, int K);
void k_gemm_tiled(float* A, float* B, float* C, int M, int N, int K);
void k_transpose(float* in, float* out, int rows, int cols);
void k_linear(float* x, float* W, float* y, int in_features, int out_features);

void k_rmsnorm(float* x, float* w, float* y, int rows, int cols, float eps);
void k_silu(float* x, float* y, int n);
void k_swiglu(float* gate, float* up, float* out, int n);

void k_mha_scores_fused_mask(float* Q, float* K, float* S, int seq, int n_heads, int n_kv_heads, int head_dim);
void k_mha_weighted_sum(float* P, float* V, float* O, int seq, int n_heads, int n_kv_heads, int head_dim);
void k_mha_scores_one(float* q, float* K_cache, float* s, int pos, int n_heads, int n_kv_heads, int head_dim);
void k_mha_weighted_sum_one(float* p, float* V_cache, float* o, int pos, int n_heads, int n_kv_heads, int head_dim);

void k_row_softmax(float* x, float* y, int rows, int cols);
void k_llama_rope(float* x, int seq, int n_heads, int head_dim, int pos_base); 

void k_embedding_lookup(int* ids, float* table, float* out, int seq, int dim);
void k_gather_last_token(float* x, float* out, int seq, int dim);
void k_argmax_row(float* x, int* out, int rows, int cols);
void k_row_add_bias(float* x, float* b, int rows, int cols);
void k_copy_row_to_cache(float* src_row, float* cache, int pos, int dim);
void k_gemv(float* x, float* W, float* y, int K, int N);

void k_fused_add_rmsnorm(float* x, float* residual_in, float* w, float* norm_out, int rows, int cols, float eps);

void k_apply_temperature(float* logits, float temp, int vocab_size);
void k_sample_top_p(float* probs, int* out_idx, float p, float random_val, int vocab_size);
void k_apply_repetition_penalty(float* logits, int* past_tokens, int num_past, float penalty);
void k_half_to_float(half* src, float* dst, int n);

// ==========================================
// PHASE 4: FP16 Kernels
// ==========================================
void k_half_linear(cublasHandle_t handle, half* x, half* W, half* y, int in_features, int out_features);
void k_half_rmsnorm(half* x, half* w, half* y, int rows, int cols, float eps);
void k_half_fused_add_rmsnorm(half* x, half* res, half* w, half* out, int rows, int cols, float eps);
void k_half_llama_rope(half* x, int seq, int n_heads, int head_dim, int pos_base);
void k_half_mha_scores_one(half* q, half* K, float* s, int pos, int n_heads, int n_kv_heads, int head_dim);
void k_half_mha_weighted_sum_one(float* p, half* V, half* o, int pos, int n_heads, int n_kv_heads, int head_dim);
void k_half_swiglu(half* gate, half* up, half* out, int n);
void k_half_add(half* a, half* b, half* c, int n);
void k_half_embedding_lookup(int* ids, half* table, half* out, int seq, int dim);
void k_half_copy_row_to_cache(half* src, half* cache, int pos, int dim);
void k_half_copy(half* src, half* dst, int n);
void k_half_argmax_row(half* x, int* out, int rows, int cols);
void k_half_to_float(half* src, float* dst, int n);