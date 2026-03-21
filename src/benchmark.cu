#include <cstdio>
#include <cuda_runtime.h>
#include "kernels.cuh"
#include "mem_pool.h"
#include "tensor.h"

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
    Timer t;
    cudaDeviceSynchronize();
    t.start();
    for (int i = 0; i < iters; i++) k_gemm_naive(A.data, B.data, C.data, M, N, K);
    float ms_naive = t.stop() / iters;
    t.start();
    for (int i = 0; i < iters; i++) k_gemm_tiled(A.data, B.data, C.data, M, N, K);
    float ms_tiled = t.stop() / iters;
    double flops = 2.0 * M * N * K;
    printf("GEMM [%d x %d x %d]\n", M, N, K);
    printf("  Naive : %8.3f ms | %7.1f GFLOPS\n", ms_naive, flops / (ms_naive * 1e6));
    printf("  Tiled : %8.3f ms | %7.1f GFLOPS\n", ms_tiled, flops / (ms_tiled * 1e6));
    printf("  Speedup: %.2fx\n\n", ms_naive / ms_tiled);
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