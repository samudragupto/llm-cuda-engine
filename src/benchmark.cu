#include <cstdio>
#include <cuda_runtime.h>
#include "kernels.cuh"
#include "mem_pool.h"
#include "tensor.h"
#include <cublas_v2.h>

struct Timer {
    cudaEvent_t s, e;
    Timer() { cudaEventCreate(&s); cudaEventCreate(&e); }
    ~Timer() { cudaEventDestroy(s); cudaEventDestroy(e); }
    void start() { cudaEventRecord(s); }
    float stop() {
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, s, e);
        return ms;
    }
};

void bench_gemm(MemPool& pool, int M, int N, int K, int iters) {
    Tensor A(pool, {M, K}), B(pool, {K, N}), C(pool, {M, N});
    A.fill(1.0f); B.fill(0.5f);
    
    cublasHandle_t handle;
    cublasCreate(&handle);

    
    
    for (int i = 0; i < 5; i++) {
        k_gemm_naive(A.data, B.data, C.data, M, N, K);
        k_gemm_tiled(A.data, B.data, C.data, M, N, K);
        k_cublas_gemm(handle, A.data, B.data, C.data, M, N, K);
    }
    cudaDeviceSynchronize();
    

    Timer t;
    
    
    t.start();
    for (int i = 0; i < iters; i++) k_gemm_naive(A.data, B.data, C.data, M, N, K);
    float ms_naive = t.stop() / iters;
    
    
    t.start();
    for (int i = 0; i < iters; i++) k_gemm_tiled(A.data, B.data, C.data, M, N, K);
    float ms_tiled = t.stop() / iters;

    
    t.start();
    for (int i = 0; i < iters; i++) k_cublas_gemm(handle, A.data, B.data, C.data, M, N, K);
    float ms_cublas = t.stop() / iters;

    cublasDestroy(handle);

    double flops = 2.0 * M * N * K;
    printf("GEMM [%d x %d x %d]\n", M, N, K);
    printf("  Naive  : %8.3f ms | %7.1f GFLOPS\n", ms_naive, flops / (ms_naive * 1e6));
    printf("  Tiled  : %8.3f ms | %7.1f GFLOPS\n", ms_tiled, flops / (ms_tiled * 1e6));
    printf("  cuBLAS : %8.3f ms | %7.1f GFLOPS\n", ms_cublas, flops / (ms_cublas * 1e6));
    printf("  Speedup vs Naive: %.2fx\n", ms_naive / ms_tiled);
    
    
    
    printf("  Custom vs cuBLAS: %.2fx (1.0 = equal)\n\n", ms_cublas / ms_tiled); 
}

void bench_wmma(MemPool& pool, int M, int N, int K, int iters) {
    
    Tensor A_fp32(pool, {M, K}), B_fp32(pool, {K, N}), C_fp32(pool, {M, N});
    A_fp32.fill(1.0f); B_fp32.fill(0.5f);
    
    
    half* d_A_fp16 = pool.alloc<half>(M * K);
    half* d_B_fp16 = pool.alloc<half>(K * N); 
    
    
    int nA = M * K;
    int nB = K * N;
    k_fp32_to_fp16(A_fp32.data, d_A_fp16, nA);
    k_fp32_to_fp16(B_fp32.data, d_B_fp16, nB);
    cudaDeviceSynchronize();

    cublasHandle_t handle;
    cublasCreate(&handle);

    
    for(int i=0; i<5; i++){
        k_gemm_tiled(A_fp32.data, B_fp32.data, C_fp32.data, M, N, K);
        k_gemm_wmma(d_A_fp16, d_B_fp16, C_fp32.data, M, N, K);
    }
    cudaDeviceSynchronize();

    Timer t;
    
    
    t.start();
    for (int i = 0; i < iters; i++) k_gemm_tiled(A_fp32.data, B_fp32.data, C_fp32.data, M, N, K);
    float ms_tiled = t.stop() / iters;

    
    t.start();
    for (int i = 0; i < iters; i++) k_gemm_wmma(d_A_fp16, d_B_fp16, C_fp32.data, M, N, K);
    float ms_wmma = t.stop() / iters;

    
    float alpha = 1.0f, beta = 0.0f;
    t.start();
    for (int i = 0; i < iters; i++) {
        cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
                     N, M, K,
                     &alpha,
                     d_B_fp16, CUDA_R_16F, K, 
                     d_A_fp16, CUDA_R_16F, K,
                     &beta,
                     C_fp32.data, CUDA_R_32F, N,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    }
    float ms_cublas = t.stop() / iters;

    cublasDestroy(handle);

    double flops = 2.0 * M * N * K;
    printf("TENSOR CORE WMMA BENCHMARK [%d x %d x %d]\n", M, N, K);
    printf("  Custom Tiled (FP32) : %8.3f ms | %7.1f GFLOPS\n", ms_tiled, flops / (ms_tiled * 1e6));
    printf("  Custom WMMA  (FP16) : %8.3f ms | %7.1f GFLOPS\n", ms_wmma, flops / (ms_wmma * 1e6));
    printf("  cuBLAS WMMA  (FP16) : %8.3f ms | %7.1f GFLOPS\n", ms_cublas, flops / (ms_cublas * 1e6));
    printf("  Speedup vs Tiled    : %.2fx\n", ms_tiled / ms_wmma);
    printf("  Custom vs cuBLAS    : %.2fx (1.0 = equal)\n\n", ms_cublas / ms_wmma); 
}

void bench_elementwise(MemPool& pool, int n, int iters) {
    Tensor A(pool, {n}), B(pool, {n}), C(pool, {n});
    A.fill(1.0f); B.fill(2.0f);
    Timer t;
    cudaDeviceSynchronize();
    t.start();
    for (int i = 0; i < iters; i++) k_add(A.data, B.data, C.data, n);
    float ms = t.stop() / iters;
    printf("Add   [%d]: %.4f ms | %.1f GB/s\n", n, ms, 3.0 * n * 4 / (ms * 1e6));
    t.start();
    for (int i = 0; i < iters; i++) k_mul(A.data, B.data, C.data, n);
    ms = t.stop() / iters;
    printf("Mul   [%d]: %.4f ms | %.1f GB/s\n", n, ms, 3.0 * n * 4 / (ms * 1e6));
    t.start();
    for (int i = 0; i < iters; i++) k_scale(A.data, 2.5f, C.data, n);
    ms = t.stop() / iters;
    printf("Scale [%d]: %.4f ms | %.1f GB/s\n\n", n, ms, 2.0 * n * 4 / (ms * 1e6));
}

void bench_transpose(MemPool& pool, int rows, int cols, int iters) {
    Tensor A(pool, {rows, cols}), B(pool, {cols, rows});
    A.fill(1.0f);
    Timer t;
    cudaDeviceSynchronize();
    t.start();
    for (int i = 0; i < iters; i++) k_transpose(A.data, B.data, rows, cols);
    float ms = t.stop() / iters;
    printf("Transpose [%d x %d]: %.4f ms | %.1f GB/s\n\n", rows, cols, ms, 2.0 * rows * cols * 4 / (ms * 1e6));
}