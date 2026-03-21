#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "mem_pool.h"
#include "tensor.h"
#include "kernels.cuh"
#include "weight_loader.h"
#include "phase2.h"
#include "phase3.h"
#include "profiler.h" 

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
extern void test_swiglu(MemPool&);
extern void test_mha(MemPool&);
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
    test_swiglu(pool); pool.reset();
    test_mha(pool); pool.reset();
    printf("\nResults: %d passed, %d failed\n\n", tests_passed, tests_failed);

    test_rmsnorm(pool); pool.reset();
    test_silu(pool); pool.reset();
    test_softmax(pool); pool.reset();
    test_rope(pool); pool.reset();
    test_attention(pool); pool.reset();
    test_block(pool); pool.reset();

    test_embedding(pool); pool.reset();
    test_argmax(pool); pool.reset();
    test_tiny_tokenizer();

    MemPool model_pool(512ULL * 1024 * 1024);
    MemPool scratch(512ULL * 1024 * 1024);

    test_tiny_model(model_pool, scratch);
    scratch.reset();
    test_kv_cache_equivalence(model_pool, scratch);
    scratch.reset();

    if (argc > 1) test_safetensors(argv[1], pool);
    else printf("Tip: engine_p1_upgrades.exe test_model.safetensors\n");

    TinyTokenizer tok;
    MemPool demo_model_pool(512ULL * 1024 * 1024);
    MemPool demo_scratch(512ULL * 1024 * 1024);
    TinyModel model(demo_model_pool, 32, 12, 32, 64, 2);
    model.init();
    std::vector<int> prompt = tok.encode("hello world cuda");
    
    // Create Configs for Greedy vs Repetition Penalized
    GenerationConfig greedy_cfg;
    greedy_cfg.max_new_tokens = 10;
    greedy_cfg.temperature = 0.0f; // Exact reproduction
    greedy_cfg.repetition_penalty = 1.0f;

    GenerationConfig penalty_cfg;
    penalty_cfg.max_new_tokens = 10;
    penalty_cfg.temperature = 0.0f; // Keep greedy so we ONLY see the penalty effect
    penalty_cfg.repetition_penalty = 2.0f; // Strongly penalize repeating tokens

    auto out1 = model.generate_cached(demo_scratch, prompt, greedy_cfg);
    demo_scratch.reset();
    auto out_penalty = model.generate_cached(demo_scratch, prompt, penalty_cfg);

    printf("\n=== DEMO ===\n");
    printf("Greedy (No Penalty):   %s\n", tok.decode(out1).c_str());
    printf("Greedy (Rep Pen 2.0):  %s\n", tok.decode(out_penalty).c_str());

    // --- PHASE 1 UPGRADES DEMO ---
    printf("\n=== PHASE 1 UPGRADES DEMO ===\n");
    demo_model_pool.print_stats("Persistent Model Pool");
    demo_scratch.print_stats("Step Scratch Pool");

    Profiler prof;
    prof.start("Dummy Kernel Simulation");
    cudaDeviceSynchronize(); // simulate work
    prof.stop("Dummy Kernel Simulation");
    
    prof.start("Matrix Math Sim");
    bench_gemm(pool, 512, 512, 512, 5); // Use existing func to simulate work
    prof.stop("Matrix Math Sim");
    
    prof.print_summary();
    printf("=== PHASE 1 UPGRADES COMPLETE ===\n\n");

    return 0;
}