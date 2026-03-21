#include <cstdio>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "mem_pool.h"
#include "tensor.h"
#include "kernels.cuh"
#include "weight_loader.h"
#include "phase2.h"

extern void bench_gemm(MemPool&, int, int, int, int);
extern void bench_elementwise(MemPool&, int, int);
extern void bench_transpose(MemPool&, int, int, int);
extern void test_add(MemPool&);
extern void test_mul(MemPool&);
extern void test_scale(MemPool&);
extern void test_gemm_small(MemPool&);
extern void test_gemm_large(MemPool&);
extern void test_gemm_naive_vs_tiled(MemPool&);
extern void test_transpose(MemPool&);
extern void test_safetensors(const char*, MemPool&);
extern int tests_passed, tests_failed;

void print_gpu_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("========================================\n");
    printf(" GPU: %s\n", prop.name);
    printf(" SMs: %d | Compute: %d.%d\n", prop.multiProcessorCount, prop.major, prop.minor);
    printf(" Global Mem: %.1f GB\n", prop.totalGlobalMem / 1073741824.0);
    printf(" Shared/Block: %zu bytes\n", prop.sharedMemPerBlock);
    printf(" Max Threads/Block: %d\n", prop.maxThreadsPerBlock);
    printf(" Warp Size: %d\n", prop.warpSize);
    printf("========================================\n\n");
}

int main(int argc, char** argv) {
    print_gpu_info();
    MemPool pool(1536ULL * 1024 * 1024);

    printf("=== PHASE 1 TESTS ===\n");
    test_add(pool); pool.reset();
    test_mul(pool); pool.reset();
    test_scale(pool); pool.reset();
    test_gemm_small(pool); pool.reset();
    test_gemm_large(pool); pool.reset();
    test_gemm_naive_vs_tiled(pool); pool.reset();
    test_transpose(pool); pool.reset();
    printf("\nResults: %d passed, %d failed\n\n", tests_passed, tests_failed);

    printf("=== PHASE 1 BENCHMARKS ===\n");
    bench_elementwise(pool, 1 << 20, 100); pool.reset();
    bench_transpose(pool, 4096, 4096, 50); pool.reset();
    bench_gemm(pool, 512, 512, 512, 20); pool.reset();
    bench_gemm(pool, 1024, 1024, 1024, 20); pool.reset();
    bench_gemm(pool, 4096, 4096, 4096, 10); pool.reset();

    test_rmsnorm(pool); pool.reset();
    test_silu(pool); pool.reset();
    test_softmax(pool); pool.reset();
    test_rope(pool); pool.reset();
    test_attention(pool); pool.reset();
    test_block(pool); pool.reset();
    bench_phase2(pool); pool.reset();

    if (argc > 1) test_safetensors(argv[1], pool);
    else {
        printf("Tip: engine_p1.exe test_model.safetensors\n");
    }

    printf("=== PHASE 2 COMPLETE ===\n");
    return 0;
}