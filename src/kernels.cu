#include "kernels.cuh"
#include <cuda_runtime.h>
#define TILE 32

__global__ void _add(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
__global__ void _mul(float* a, float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}
__global__ void _scale(float* a, float s, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * s;
}
__global__ void _fill(float* a, float val, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = val;
}
__global__ void _copy(float* s, float* d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = s[i];
}
__global__ void _gemm_naive(float* A, float* B, float* C, int M, int N, int K) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M && c < N) {
        float s = 0;
        for (int i = 0; i < K; i++) s += A[r * K + i] * B[i * N + c];
        C[r * N + c] = s;
    }
}
__global__ void _gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE + 1];
    int r = blockIdx.y * TILE + threadIdx.y;
    int c = blockIdx.x * TILE + threadIdx.x;
    float s = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int ak = t * TILE + threadIdx.x;
        int bk = t * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (r < M && ak < K) ? A[r * K + ak] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bk < K && c < N) ? B[bk * N + c] : 0.0f;
        __syncthreads();
        #pragma unroll
        for (int i = 0; i < TILE; i++) s += As[threadIdx.y][i] * Bs[i][threadIdx.x];
        __syncthreads();
    }
    if (r < M && c < N) C[r * N + c] = s;
}
__global__ void _transpose(float* in, float* out, int rows, int cols) {
    __shared__ float tile[TILE][TILE + 1];
    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x;
    y = blockIdx.x * TILE + threadIdx.y;
    if (x < rows && y < cols) out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}
void k_add(float* a, float* b, float* c, int n) { _add<<<(n + 255) / 256, 256>>>(a, b, c, n); }
void k_mul(float* a, float* b, float* c, int n) { _mul<<<(n + 255) / 256, 256>>>(a, b, c, n); }
void k_scale(float* a, float s, float* c, int n) { _scale<<<(n + 255) / 256, 256>>>(a, s, c, n); }
void k_fill(float* a, float val, int n) { _fill<<<(n + 255) / 256, 256>>>(a, val, n); }
void k_copy(float* src, float* dst, int n) { _copy<<<(n + 255) / 256, 256>>>(src, dst, n); }
void k_gemm_naive(float* A, float* B, float* C, int M, int N, int K) {
    dim3 g((N + 15) / 16, (M + 15) / 16), b(16, 16);
    _gemm_naive<<<g, b>>>(A, B, C, M, N, K);
}
void k_gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {
    dim3 g((N + TILE - 1) / TILE, (M + TILE - 1) / TILE), b(TILE, TILE);
    _gemm_tiled<<<g, b>>>(A, B, C, M, N, K);
}
void k_transpose(float* in, float* out, int rows, int cols) {
    dim3 g((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE), b(TILE, TILE);
    _transpose<<<g, b>>>(in, out, rows, cols);
}