#include "kernels.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <mma.h>
#include <cublas_v2.h>

#define TILE 32
using namespace nvcuda;

__global__ void _add(float* a, float* b, float* c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = a[i] + b[i]; }
__global__ void _mul(float* a, float* b, float* c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = a[i] * b[i]; }
__global__ void _scale(float* a, float s, float* c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = a[i] * s; }
__global__ void _fill(float* a, float v, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] = v; }
__global__ void _copy(float* s, float* d, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) d[i] = s[i]; }
__global__ void _fp32_to_fp16(float* src, half* dst, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = __float2half(src[i]); }
__global__ void _half_to_float(half* src, float* dst, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = __half2float(src[i]); }

__global__ void _gemm_naive(float* A, float* B, float* C, int M, int N, int K) {
    int r = blockIdx.y * blockDim.y + threadIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < M && c < N) { float s = 0; for (int i = 0; i < K; i++) s += A[r * K + i] * B[i * N + c]; C[r * N + c] = s; }
}

__global__ void _gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE], Bs[TILE][TILE + 1];
    int r = blockIdx.y * TILE + threadIdx.y, c = blockIdx.x * TILE + threadIdx.x;
    float s = 0;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int ak = t * TILE + threadIdx.x, bk = t * TILE + threadIdx.y;
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
    int x = blockIdx.x * TILE + threadIdx.x, y = blockIdx.y * TILE + threadIdx.y;
    if (x < cols && y < rows) tile[threadIdx.y][threadIdx.x] = in[y * cols + x];
    __syncthreads();
    x = blockIdx.y * TILE + threadIdx.x; y = blockIdx.x * TILE + threadIdx.y;
    if (x < rows && y < cols) out[y * rows + x] = tile[threadIdx.x][threadIdx.y];
}

__global__ void _gemv(float* x, float* W, float* y, int K, int N) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) sum += x[k] * W[k * N + col];
        y[col] = sum;
    }
}

__global__ void _linear(float* x, float* W, float* y, int in_features, int out_features) {
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx < out_features) {
        float sum = 0.0f;
        int w_base = out_idx * in_features; 
        for (int i = 0; i < in_features; i++) sum += x[i] * W[w_base + i];
        y[out_idx] = sum;
    }
}

__global__ void _rmsnorm(float* x, float* w, float* y, int rows, int cols, float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float s;
    if (threadIdx.x == 0) {
        float ss = 0.0f;
        for (int i = 0; i < cols; i++) { float v = x[r * cols + i]; ss += v * v; }
        s = rsqrtf(ss / cols + eps);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) y[r * cols + i] = x[r * cols + i] * s * w[i];
}

__global__ void _fused_add_rmsnorm(float* x, float* residual_in, float* w, float* norm_out, int rows, int cols, float eps) {
    int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float s;
    if (threadIdx.x == 0) s = 0.0f;
    __syncthreads();
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = x[r * cols + i] + residual_in[r * cols + i];
        x[r * cols + i] = val;
        ss += val * val;
    }
    for (int offset = 16; offset > 0; offset /= 2) ss += __shfl_down_sync(0xffffffff, ss, offset);
    if (threadIdx.x % 32 == 0) atomicAdd(&s, ss);
    __syncthreads();
    if (threadIdx.x == 0) s = rsqrtf(s / cols + eps);
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) norm_out[r * cols + i] = x[r * cols + i] * s * w[i];
}

__global__ void _silu(float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = v / (1.0f + expf(-v)); }
}

__global__ void _swiglu(float* gate, float* up, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float g = gate[i]; out[i] = (g / (1.0f + expf(-g))) * up[i]; }
}

__global__ void _row_softmax(float* x, float* y, int rows, int cols) {
    int r = blockIdx.x;
    if (r >= rows) return;
    __shared__ float m, s;
    if (threadIdx.x == 0) {
        float mx = x[r * cols];
        for (int i = 1; i < cols; i++) if (x[r * cols + i] > mx) mx = x[r * cols + i];
        float sm = 0.0f;
        for (int i = 0; i < cols; i++) sm += expf(x[r * cols + i] - mx);
        m = mx; s = sm;
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) y[r * cols + i] = expf(x[r * cols + i] - m) / s;
}

__global__ void _llama_rope(float* x, int seq, int n_heads, int head_dim, int pos_base) {
    int pos = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    if (pos < seq && h < n_heads && i < head_dim / 2) {
        int base = pos * (n_heads * head_dim) + h * head_dim;
        float x0 = x[base + i], x1 = x[base + i + head_dim / 2];
        float freq = powf(10000.0f, -(2.0f * i) / head_dim);
        float th = (pos_base + pos) * freq;
        float c = cosf(th), s = sinf(th);
        x[base + i] = x0 * c - x1 * s;
        x[base + i + head_dim / 2] = x0 * s + x1 * c;
    }
}

__global__ void _mha_scores_fused_mask(float* Q, float* K, float* S, int seq, int n_heads, int n_kv_heads, int head_dim){
    int h = blockIdx.z, r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < n_heads && r < seq && c < seq) {
        if (c > r) { S[h * seq * seq + r * seq + c] = -1e20f; return; }
        float sum = 0.0f;
        int kv_h = h / (n_heads / n_kv_heads);
        int q_base = r * (n_heads * head_dim) + h * head_dim;
        int k_base = c * (n_kv_heads * head_dim) + kv_h * head_dim;
        for (int d = 0; d < head_dim; d++) sum += Q[q_base + d] * K[k_base + d];
        S[h * seq * seq + r * seq + c] = sum / sqrtf((float)head_dim);
    }
}

__global__ void _mha_weighted_sum(float* P, float* V, float* O, int seq, int n_heads, int n_kv_heads, int head_dim){
    int h = blockIdx.z, r = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < n_heads && r < seq && d < head_dim) {
        float sum = 0.0f;
        int kv_h = h / (n_heads / n_kv_heads);
        for (int c = 0; c < seq; c++) {
            sum += P[h * seq * seq + r * seq + c] * V[c * (n_kv_heads * head_dim) + kv_h * head_dim + d];
        }
        O[r * (n_heads * head_dim) + h * head_dim + d] = sum;
    }
}

__global__ void _mha_scores_one(float* q, float* K_cache, float* s, int pos, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < n_heads && c <= pos) {
        float sum = 0.0f;
        int kv_h = h / (n_heads / n_kv_heads);
        int q_base = h * head_dim;
        int k_base = c * (n_kv_heads * head_dim) + kv_h * head_dim;
        for(int d=0; d < head_dim; d++) sum += q[q_base + d] * K_cache[k_base + d];
        s[h * (pos + 1) + c] = sum / sqrtf((float)head_dim);
    }
}

__global__ void _mha_weighted_sum_one(float* p, float* V_cache, float* o, int pos, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y;
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < n_heads && d < head_dim) {
        float sum = 0.0f;
        int kv_h = h / (n_heads / n_kv_heads);
        for(int c=0; c <= pos; c++) {
            sum += p[h * (pos + 1) + c] * V_cache[c * (n_kv_heads * head_dim) + kv_h * head_dim + d];
        }
        o[h * head_dim + d] = sum;
    }
}

__global__ void _embedding_lookup(int* ids, float* table, float* out, int seq, int dim) {
    int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < seq && d < dim) { int id = ids[t]; out[t * dim + d] = table[id * dim + d]; }
}

__global__ void _gather_last_token(float* x, float* out, int seq, int dim) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < dim) out[d] = x[(seq - 1) * dim + d];
}

__global__ void _argmax_row(float* x, int* out, int rows, int cols) {
    int r = blockIdx.x;
    if (r < rows && threadIdx.x == 0) {
        int idx = 0; float best = x[r * cols];
        for (int i = 1; i < cols; i++) if (x[r * cols + i] > best) { best = x[r * cols + i]; idx = i; }
        out[r] = idx;
    }
}

__global__ void _row_add_bias(float* x, float* b, int rows, int cols) {
    int r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (r < rows && c < cols) x[r * cols + c] += b[c];
}

__global__ void _copy_row_to_cache(float* src_row, float* cache, int pos, int dim) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d < dim) cache[pos * dim + d] = src_row[d];
}

__global__ void _apply_temperature(float* logits, float temp, int vocab_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vocab_size) logits[i] /= temp;
}

__global__ void _sample_top_p(float* probs, int* out_idx, float p, float random_val, int vocab_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float cdf = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cdf += probs[i];
            if (cdf >= random_val) { out_idx[0] = i; return; }
        }
        out_idx[0] = vocab_size - 1;
    }
}

__global__ void _apply_repetition_penalty(float* logits, int* past_tokens, int num_past, float penalty) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_past) {
        int token_id = past_tokens[i];
        float logit = logits[token_id];
        logits[token_id] = logit > 0.0f ? logit / penalty : logit * penalty;
    }
}

__global__ void _half_gemv(half* x, half* W, half* y, int in_features, int out_features) {
    int row = blockIdx.x; 
    if (row >= out_features) return;
    
    float sum = 0.0f;
    int w_base = row * in_features;
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) {
        sum += __half2float(x[i]) * __half2float(W[w_base + i]);
    }
    
    __shared__ float sdata[32];
    int warpID = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) sdata[warpID] = sum;
    __syncthreads();
    
    if (warpID == 0) {
        sum = (lane < (blockDim.x / 32)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (threadIdx.x == 0) y[row] = __float2half(sum);
    }
}

__global__ void _half_mha_scores_fused_mask(half* Q, half* K, float* S, int seq, int n_heads, int n_kv_heads, int head_dim){
    int h = blockIdx.z, r = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < n_heads && r < seq && c < seq) {
        if (c > r) { S[h * seq * seq + r * seq + c] = -1e20f; return; }
        float sum = 0.0f;
        int kv_h = h / (n_heads / n_kv_heads);
        int q_base = r * (n_heads * head_dim) + h * head_dim;
        int k_base = c * (n_kv_heads * head_dim) + kv_h * head_dim;
        for (int d = 0; d < head_dim; d++) sum += __half2float(Q[q_base + d]) * __half2float(K[k_base + d]);
        S[h * seq * seq + r * seq + c] = sum / sqrtf((float)head_dim);
    }
}

__global__ void _half_mha_weighted_sum(float* P, half* V, half* O, int seq, int n_heads, int n_kv_heads, int head_dim){
    int h = blockIdx.z, r = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < n_heads && r < seq && d < head_dim) {
        float sum = 0.0f;
        int kv_h = h / (n_heads / n_kv_heads);
        for (int c = 0; c < seq; c++) {
            sum += P[h * seq * seq + r * seq + c] * __half2float(V[c * (n_kv_heads * head_dim) + kv_h * head_dim + d]);
        }
        O[r * (n_heads * head_dim) + h * head_dim + d] = __float2half(sum);
    }
}

__global__ void _half_copy_block_to_cache(half* src, half* cache, int pos_base, int seq_len, int dim) {
    int seq_idx = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (seq_idx < seq_len && d < dim) cache[(pos_base + seq_idx) * dim + d] = src[seq_idx * dim + d];
}

__global__ void _half_fused_add_rmsnorm(half* x, half* residual_in, half* w, half* norm_out, int rows, int cols, float eps) {
    int r = blockIdx.x; if (r >= rows) return;
    __shared__ float s; if (threadIdx.x == 0) s = 0.0f; __syncthreads();
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = __half2float(x[r * cols + i]) + __half2float(residual_in[r * cols + i]);
        x[r * cols + i] = __float2half(val); ss += val * val;
    }
    for (int offset = 16; offset > 0; offset /= 2) ss += __shfl_down_sync(0xffffffff, ss, offset);
    if (threadIdx.x % 32 == 0) atomicAdd(&s, ss); __syncthreads();
    if (threadIdx.x == 0) s = rsqrtf(s / cols + eps); __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) norm_out[r * cols + i] = __float2half(__half2float(x[r * cols + i]) * s * __half2float(w[i]));
}

__global__ void _half_rmsnorm(half* x, half* w, half* y, int rows, int cols, float eps) {
    int r = blockIdx.x; if (r >= rows) return;
    __shared__ float s;
    if (threadIdx.x == 0) { float ss = 0.0f; for (int i = 0; i < cols; i++) ss += __half2float(x[r*cols+i])*__half2float(x[r*cols+i]); s = rsqrtf(ss/cols+eps); }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) y[r*cols+i] = __float2half(__half2float(x[r*cols+i])*s*__half2float(w[i]));
}

__global__ void _half_add(half* a, half* b, half* c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = __float2half(__half2float(a[i]) + __half2float(b[i])); }

__global__ void _half_llama_rope(half* x, int seq, int n_heads, int head_dim, int pos_base) {
    int pos = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    if (pos < seq && h < n_heads && i < head_dim / 2) {
        int base = pos * (n_heads * head_dim) + h * head_dim;
        float x0 = __half2float(x[base+i]), x1 = __half2float(x[base+i+head_dim/2]);
        float th = (pos_base + pos) * powf(10000.0f, -(2.0f * i) / head_dim);
        x[base+i] = __float2half(x0*cosf(th) - x1*sinf(th)); x[base+i+head_dim/2] = __float2half(x0*sinf(th) + x1*cosf(th));
    }
}

__global__ void _half_mha_scores_one(half* q, half* K_cache, float* s, int pos, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < n_heads && c <= pos) {
        float sum = 0.0f; int kv_h = h / (n_heads / n_kv_heads);
        int q_base = h * head_dim, k_base = c * (n_kv_heads * head_dim) + kv_h * head_dim;
        for(int d=0; d < head_dim; d++) sum += __half2float(q[q_base+d]) * __half2float(K_cache[k_base+d]);
        s[h*(pos+1)+c] = sum / sqrtf((float)head_dim);
    }
}

__global__ void _half_mha_weighted_sum_one(float* p, half* V_cache, half* o, int pos, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < n_heads && d < head_dim) {
        float sum = 0.0f; int kv_h = h / (n_heads / n_kv_heads);
        for(int c=0; c <= pos; c++) sum += p[h*(pos+1)+c] * __half2float(V_cache[c*(n_kv_heads*head_dim)+kv_h*head_dim+d]);
        o[h*head_dim+d] = __float2half(sum);
    }
}

__global__ void _half_swiglu(half* gate, half* up, half* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float g = __half2float(gate[i]), u = __half2float(up[i]); out[i] = __float2half((g / (1.0f + expf(-g))) * u); } }

__global__ void _half_embedding_lookup(int* ids, half* table, half* out, int seq, int dim) { int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x; if (t < seq && d < dim) out[t*dim+d] = table[ids[t]*dim+d]; }

__global__ void _half_copy(half* src, half* dst, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = src[i]; }

__global__ void _half_argmax_row(half* x, int* out, int rows, int cols) {
    int r = blockIdx.x; if (r < rows && threadIdx.x == 0) {
        int idx = 0; float best = __half2float(x[r*cols]);
        for (int i=1; i<cols; i++) { float val = __half2float(x[r*cols+i]); if (val > best) { best = val; idx = i; } }
        out[r] = idx;
    }
}

void k_add(float* a, float* b, float* c, int n) { _add<<<(n + 255) / 256, 256>>>(a, b, c, n); }
void k_mul(float* a, float* b, float* c, int n) { _mul<<<(n + 255) / 256, 256>>>(a, b, c, n); }
void k_scale(float* a, float s, float* c, int n) { _scale<<<(n + 255) / 256, 256>>>(a, s, c, n); }
void k_fill(float* a, float val, int n) { _fill<<<(n + 255) / 256, 256>>>(a, val, n); }
void k_copy(float* src, float* dst, int n) { _copy<<<(n + 255) / 256, 256>>>(src, dst, n); }
void k_fp32_to_fp16(float* src, half* dst, int n) { _fp32_to_fp16<<<(n + 255) / 256, 256>>>(src, dst, n); }
void k_half_to_float(half* src, float* dst, int n) { _half_to_float<<<(n + 255) / 256, 256>>>(src, dst, n); }
void k_gemm_naive(float* A, float* B, float* C, int M, int N, int K) { dim3 g((N + 15) / 16, (M + 15) / 16), b(16, 16); _gemm_naive<<<g, b>>>(A, B, C, M, N, K); }
void k_gemm_tiled(float* A, float* B, float* C, int M, int N, int K) { dim3 g((N + TILE - 1) / TILE, (M + TILE - 1) / TILE), b(TILE, TILE); _gemm_tiled<<<g, b>>>(A, B, C, M, N, K); }
void k_transpose(float* in, float* out, int rows, int cols) { dim3 g((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE), b(TILE, TILE); _transpose<<<g, b>>>(in, out, rows, cols); }
void k_gemv(float* x, float* W, float* y, int K, int N) { _gemv<<<(N + 255) / 256, 256>>>(x, W, y, K, N); }
void k_linear(float* x, float* W, float* y, int in_features, int out_features) { dim3 g((out_features + 255) / 256); dim3 b(256); _linear<<<g, b>>>(x, W, y, in_features, out_features); }
void k_rmsnorm(float* x, float* w, float* y, int rows, int cols, float eps) { _rmsnorm<<<rows, 256>>>(x, w, y, rows, cols, eps); }
void k_fused_add_rmsnorm(float* x, float* residual_in, float* w, float* norm_out, int rows, int cols, float eps) { _fused_add_rmsnorm<<<rows, 32>>>(x, residual_in, w, norm_out, rows, cols, eps); }
void k_silu(float* x, float* y, int n) { _silu<<<(n + 255) / 256, 256>>>(x, y, n); }
void k_swiglu(float* gate, float* up, float* out, int n) { _swiglu<<<(n + 255) / 256, 256>>>(gate, up, out, n); }
void k_row_softmax(float* x, float* y, int rows, int cols) { _row_softmax<<<rows, 256>>>(x, y, rows, cols); }
void k_llama_rope(float* x, int seq, int n_heads, int head_dim, int pos_base) { dim3 g(seq, n_heads); _llama_rope<<<g, head_dim / 2>>>(x, seq, n_heads, head_dim, pos_base); }
void k_mha_scores_fused_mask(float* Q, float* K, float* S, int seq, int n_heads, int n_kv_heads, int head_dim) { dim3 g((seq + 31) / 32, seq, n_heads), b(32); _mha_scores_fused_mask<<<g, b>>>(Q, K, S, seq, n_heads, n_kv_heads, head_dim); }
void k_mha_weighted_sum(float* P, float* V, float* O, int seq, int n_heads, int n_kv_heads, int head_dim) { dim3 g((head_dim + 31) / 32, seq, n_heads), b(32); _mha_weighted_sum<<<g, b>>>(P, V, O, seq, n_heads, n_kv_heads, head_dim); }
void k_mha_scores_one(float* q, float* K, float* s, int pos, int n_heads, int n_kv_heads, int head_dim) { dim3 g((pos + 1 + 255) / 256, n_heads), b(256); _mha_scores_one<<<g, b>>>(q, K, s, pos, n_heads, n_kv_heads, head_dim); }
void k_mha_weighted_sum_one(float* p, float* V, float* o, int pos, int n_heads, int n_kv_heads, int head_dim) { dim3 g((head_dim + 255) / 256, n_heads), b(256); _mha_weighted_sum_one<<<g, b>>>(p, V, o, pos, n_heads, n_kv_heads, head_dim); }
void k_embedding_lookup(int* ids, float* table, float* out, int seq, int dim) { dim3 g((dim + 255) / 256, seq), b(256); _embedding_lookup<<<g, b>>>(ids, table, out, seq, dim); }
void k_gather_last_token(float* x, float* out, int seq, int dim) { _gather_last_token<<<(dim + 255) / 256, 256>>>(x, out, seq, dim); }
void k_argmax_row(float* x, int* out, int rows, int cols) { _argmax_row<<<rows, 1>>>(x, out, rows, cols); }
void k_row_add_bias(float* x, float* b, int rows, int cols) { dim3 g((cols + 255) / 256, rows), bb(256); _row_add_bias<<<g, bb>>>(x, b, rows, cols); }
void k_copy_row_to_cache(float* src_row, float* cache, int pos, int dim) { _copy_row_to_cache<<<(dim + 255) / 256, 256>>>(src_row, cache, pos, dim); }
void k_apply_temperature(float* logits, float temp, int vocab_size) { _apply_temperature<<<(vocab_size + 255) / 256, 256>>>(logits, temp, vocab_size); }
void k_sample_top_p(float* probs, int* out_idx, float p, float random_val, int vocab_size) { _sample_top_p<<<1, 1>>>(probs, out_idx, p, random_val, vocab_size); }
void k_apply_repetition_penalty(float* logits, int* past_tokens, int num_past, float penalty) { _apply_repetition_penalty<<<(num_past + 255) / 256, 256>>>(logits, past_tokens, num_past, penalty); }

void k_half_linear(cublasHandle_t handle, half* x, half* W, half* y, int batch_size, int in_features, int out_features) { half alpha = __float2half(1.0f), beta = __float2half(0.0f); cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, batch_size, in_features, &alpha, W, in_features, x, in_features, &beta, y, out_features); }
void k_half_gemv(half* x, half* W, half* y, int in_features, int out_features) { _half_gemv<<<out_features, 256>>>(x, W, y, in_features, out_features); }
void k_half_mha_scores_fused_mask(half* Q, half* K, float* S, int seq, int n_heads, int n_kv_heads, int head_dim) { dim3 g((seq + 31) / 32, seq, n_heads), b(32); _half_mha_scores_fused_mask<<<g, b>>>(Q, K, S, seq, n_heads, n_kv_heads, head_dim); }
void k_half_mha_weighted_sum(float* P, half* V, half* O, int seq, int n_heads, int n_kv_heads, int head_dim) { dim3 g((head_dim + 31) / 32, seq, n_heads), b(32); _half_mha_weighted_sum<<<g, b>>>(P, V, O, seq, n_heads, n_kv_heads, head_dim); }
void k_half_copy_block_to_cache(half* src, half* cache, int pos_base, int seq_len, int dim) { dim3 g((dim + 255) / 256, seq_len), b(256); _half_copy_block_to_cache<<<g, b>>>(src, cache, pos_base, seq_len, dim); }
void k_half_fused_add_rmsnorm(half* x, half* res, half* w, half* out, int rows, int cols, float eps) { _half_fused_add_rmsnorm<<<rows, 32>>>(x, res, w, out, rows, cols, eps); }
void k_half_rmsnorm(half* x, half* w, half* y, int rows, int cols, float eps) { _half_rmsnorm<<<rows, 256>>>(x, w, y, rows, cols, eps); }
void k_half_add(half* a, half* b, half* c, int n) { _half_add<<<(n + 255) / 256, 256>>>(a, b, c, n); }
void k_half_llama_rope(half* x, int seq, int n_heads, int head_dim, int pos_base) { dim3 g(seq, n_heads); _half_llama_rope<<<g, head_dim / 2>>>(x, seq, n_heads, head_dim, pos_base); }
void k_half_mha_scores_one(half* q, half* K, float* s, int pos, int n_heads, int n_kv_heads, int head_dim) { dim3 g((pos + 1 + 255) / 256, n_heads), b(256); _half_mha_scores_one<<<g, b>>>(q, K, s, pos, n_heads, n_kv_heads, head_dim); }
void k_half_mha_weighted_sum_one(float* p, half* V, half* o, int pos, int n_heads, int n_kv_heads, int head_dim) { dim3 g((head_dim + 255) / 256, n_heads), b(256); _half_mha_weighted_sum_one<<<g, b>>>(p, V, o, pos, n_heads, n_kv_heads, head_dim); }
void k_half_swiglu(half* gate, half* up, half* out, int n) { _half_swiglu<<<(n + 255) / 256, 256>>>(gate, up, out, n); }
void k_half_embedding_lookup(int* ids, half* table, half* out, int seq, int dim) { dim3 g((dim + 255) / 256, seq), b(256); _half_embedding_lookup<<<g, b>>>(ids, table, out, seq, dim); }
void k_half_copy(half* src, half* dst, int n) { _half_copy<<<(n + 255) / 256, 256>>>(src, dst, n); }
void k_half_argmax_row(half* x, int* out, int rows, int cols) { _half_argmax_row<<<rows, 1>>>(x, out, rows, cols); }