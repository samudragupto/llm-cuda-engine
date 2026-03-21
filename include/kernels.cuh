#pragma once
void k_add(float* a, float* b, float* c, int n);
void k_mul(float* a, float* b, float* c, int n);
void k_scale(float* a, float s, float* c, int n);
void k_fill(float* a, float val, int n);
void k_copy(float* src, float* dst, int n);
void k_gemm_naive(float* A, float* B, float* C, int M, int N, int K);
void k_gemm_tiled(float* A, float* B, float* C, int M, int N, int K);
void k_transpose(float* in, float* out, int rows, int cols);