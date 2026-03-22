#include <cstdio>
#include <cmath>
#include <vector>
#include "kernels.cuh"
#include "mem_pool.h"
#include "tensor.h"

int tests_passed = 0, tests_failed = 0;

void check(const char* name, bool cond) {
    if (cond) { printf("  [PASS] %s\n", name); tests_passed++; }
    else { printf("  [FAIL] %s\n", name); tests_failed++; }
}

void test_add(MemPool& pool) {
    Tensor a(pool, {1024}), b(pool, {1024}), c(pool, {1024});
    a.fill(3.0f); b.fill(7.0f);
    k_add(a.data, b.data, c.data, 1024);
    cudaDeviceSynchronize();
    std::vector<float> h; c.to_host(h);
    bool ok = true;
    for (int i = 0; i < 1024; i++) if (fabsf(h[i] - 10.0f) > 1e-5f) { ok = false; break; }
    check("add 3+7=10", ok);
}

void test_mul(MemPool& pool) {
    Tensor a(pool, {1024}), b(pool, {1024}), c(pool, {1024});
    a.fill(3.0f); b.fill(4.0f);
    k_mul(a.data, b.data, c.data, 1024);
    cudaDeviceSynchronize();
    std::vector<float> h; c.to_host(h);
    bool ok = true;
    for (int i = 0; i < 1024; i++) if (fabsf(h[i] - 12.0f) > 1e-5f) { ok = false; break; }
    check("mul 3*4=12", ok);
}

void test_scale(MemPool& pool) {
    Tensor a(pool, {1024}), c(pool, {1024});
    a.fill(5.0f);
    k_scale(a.data, 3.0f, c.data, 1024);
    cudaDeviceSynchronize();
    std::vector<float> h; c.to_host(h);
    bool ok = true;
    for (int i = 0; i < 1024; i++) if (fabsf(h[i] - 15.0f) > 1e-5f) { ok = false; break; }
    check("scale 5*3=15", ok);
}

void test_gemm_small(MemPool& pool) {
    Tensor A(pool, {2, 3}), B(pool, {3, 2}), C(pool, {2, 2});
    std::vector<float> hA = {1, 2, 3, 4, 5, 6};
    std::vector<float> hB = {7, 8, 9, 10, 11, 12};
    A.from_host(hA); B.from_host(hB);
    k_gemm_tiled(A.data, B.data, C.data, 2, 2, 3);
    cudaDeviceSynchronize();
    std::vector<float> hC; C.to_host(hC);
    float expected[] = {58.0f, 64.0f, 139.0f, 154.0f};
    bool ok = true;
    for (int i = 0; i < 4; i++) if (fabsf(hC[i] - expected[i]) > 1e-3f) { ok = false; break; }
    check("gemm 2x3 * 3x2", ok);
}

void test_gemm_large(MemPool& pool) {
    int M = 256, N = 256, K = 256;
    Tensor A(pool, {M, K}), B(pool, {K, N}), C(pool, {M, N});
    A.fill(1.0f); B.fill(1.0f);
    k_gemm_tiled(A.data, B.data, C.data, M, N, K);
    cudaDeviceSynchronize();
    std::vector<float> hC; C.to_host(hC);
    bool ok = true;
    for (int i = 0; i < M * N; i++) if (fabsf(hC[i] - (float)K) > 0.5f) { ok = false; break; }
    check("gemm 256x256 all-ones", ok);
}

void test_gemm_naive_vs_tiled(MemPool& pool) {
    int M = 128, N = 128, K = 128;
    Tensor A(pool, {M, K}), B(pool, {K, N}), C1(pool, {M, N}), C2(pool, {M, N});
    std::vector<float> hA(M * K), hB(K * N);
    for (int i = 0; i < M * K; i++) hA[i] = (float)(i % 7) - 3.0f;
    for (int i = 0; i < K * N; i++) hB[i] = (float)(i % 5) - 2.0f;
    A.from_host(hA); B.from_host(hB);
    k_gemm_naive(A.data, B.data, C1.data, M, N, K);
    k_gemm_tiled(A.data, B.data, C2.data, M, N, K);
    cudaDeviceSynchronize();
    std::vector<float> h1, h2; C1.to_host(h1); C2.to_host(h2);
    float maxdiff = 0;
    for (int i = 0; i < M * N; i++) {
        float d = fabsf(h1[i] - h2[i]);
        if (d > maxdiff) maxdiff = d;
    }
    check("naive vs tiled match", maxdiff < 0.1f);
}

void test_transpose(MemPool& pool) {
    Tensor A(pool, {3, 4}), B(pool, {4, 3});
    std::vector<float> hA = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    A.from_host(hA);
    k_transpose(A.data, B.data, 3, 4);
    cudaDeviceSynchronize();
    std::vector<float> hB; B.to_host(hB);
    float expected[] = {1, 5, 9, 2, 6, 10, 3, 7, 11, 4, 8, 12};
    bool ok = true;
    for (int i = 0; i < 12; i++) if (fabsf(hB[i] - expected[i]) > 1e-5f) { ok = false; break; }
    check("transpose 3x4", ok);
}

void test_swiglu(MemPool& pool) {
    Tensor gate(pool, {2}), up(pool, {2}), out(pool, {2});
    std::vector<float> hg = {0.0f, 2.0f}, hu = {1.0f, 3.0f};
    gate.from_host(hg); up.from_host(hu);
    k_swiglu(gate.data, up.data, out.data, 2);
    cudaDeviceSynchronize();
    std::vector<float> ho; out.to_host(ho);
    float exp0 = 0.0f * (1.0f/(1.0f+expf(0.0f))) * 1.0f;
    float exp1 = 2.0f * (1.0f/(1.0f+expf(-2.0f))) * 3.0f;
    check("swiglu FFN", fabsf(ho[0] - exp0) < 1e-4 && fabsf(ho[1] - exp1) < 1e-4);
}

void test_mha(MemPool& pool) {
    int seq=2, n_heads=2, head_dim=2;
    Tensor Q(pool, {seq, n_heads, head_dim}), K(pool, {seq, n_heads, head_dim}), V(pool, {seq, n_heads, head_dim});
    Tensor S(pool, {n_heads, seq, seq}), P(pool, {n_heads, seq, seq}), O(pool, {seq, n_heads, head_dim});
    Q.fill(1.0f); K.fill(1.0f); V.fill(1.0f);
    
    k_mha_scores_fused_mask(Q.data, K.data, S.data, seq, n_heads, n_heads, head_dim);

    k_row_softmax(S.data, P.data, n_heads*seq, seq);
    k_mha_weighted_sum(P.data, V.data, O.data, seq, n_heads, n_heads, head_dim);

    cudaDeviceSynchronize();
    
    std::vector<float> hs; S.to_host(hs);
    bool masked = (hs[1] < -1e10f); // head 0, row 0, col 1 should be masked
    check("mha fused mask", masked);
}