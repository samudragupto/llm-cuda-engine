#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <stdint.h>

void k_half_to_float(half* src, float* dst, int n);
void k_half_copy(half* src, half* dst, int n);
void k_half_gemv(half* x, half* W, half* y, int in_features, int out_features);
void k_int8_gemv(half* x, int8_t* W, half* scales, half* y, int in_features, int out_features);
void k_half_linear(cublasHandle_t handle, half* x, half* W, half* y, int b, int in_features, int out_features);
void k_flash_attention_prefill(half* Q, half* K, half* V, half* O, int seq, int n_heads, int n_kv_heads, int head_dim);

void k_half_mha_scores_one(half* q, half* K, float* s, int* d_pos, int n_heads, int n_kv_heads, int head_dim);
void k_half_mha_weighted_sum_one(float* p, half* V, half* o, int* d_pos, int n_heads, int n_kv_heads, int head_dim);
void k_half_llama_rope_graph(half* x, int n_heads, int head_dim, int* d_pos);
void k_half_copy_to_cache_graph(half* src, half* cache, int* d_pos, int dim);
void k_row_softmax_graph(float* x, float* y, int n_heads, int* d_pos);

void k_half_rmsnorm(half* x, half* w, half* y, int rows, int cols, float eps);
void k_half_fused_add_rmsnorm(half* x, half* res, half* w, half* out, int rows, int cols, float eps);
void k_half_llama_rope(half* x, int seq, int n_heads, int head_dim, int pos_base);
void k_half_add(half* a, half* b, half* c, int n);
void k_half_swiglu(half* gate, half* up, half* out, int n);
void k_half_embedding_lookup(int* ids, half* table, half* out, int seq, int dim);
void k_half_copy_block_to_cache(half* src, half* cache, int pos_base, int seq_len, int dim);
void k_argmax_row(float* x, int* out, int rows, int cols);