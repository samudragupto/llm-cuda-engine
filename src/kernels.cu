#include "kernels.cuh"
#include <cuda_runtime.h>
#include <math.h>

// --- CORE UTILS ---
__global__ void _half_to_float(half* src, float* dst, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = __half2float(src[i]); }
void k_half_to_float(half* src, float* dst, int n) { _half_to_float<<<(n+255)/256, 256>>>(src, dst, n); }

__global__ void _half_copy(half* src, half* dst, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = src[i]; }
void k_half_copy(half* src, half* dst, int n) { _half_copy<<<(n+255)/256, 256>>>(src, dst, n); }

// --- INT8 & FP16 MATRIX MATH ---
__global__ void _half_gemv(half* x, half* W, half* y, int in_features, int out_features) {
    int row = blockIdx.x; if (row >= out_features) return;
    float sum = 0.0f; int w_base = row * in_features;
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) sum += __half2float(x[i]) * __half2float(W[w_base + i]);
    __shared__ float sdata[32]; int warpID = threadIdx.x / 32; int lane = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) sdata[warpID] = sum; __syncthreads();
    if (warpID == 0) {
        sum = (lane < (blockDim.x / 32)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (threadIdx.x == 0) y[row] = __float2half(sum);
    }
}
void k_half_gemv(half* x, half* W, half* y, int in_features, int out_features) { _half_gemv<<<out_features, 256>>>(x, W, y, in_features, out_features); }

__global__ void _int8_gemv(half* x, int8_t* W, half* scales, half* y, int in_features, int out_features) {
    int row = blockIdx.x; if (row >= out_features) return;
    float sum = 0.0f; int w_base = row * in_features; float scale = __half2float(scales[row]);
    for (int i = threadIdx.x; i < in_features; i += blockDim.x) sum += __half2float(x[i]) * (float)(W[w_base + i]) * scale;
    __shared__ float sdata[32]; int warpID = threadIdx.x / 32; int lane = threadIdx.x % 32;
    for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) sdata[warpID] = sum; __syncthreads();
    if (warpID == 0) {
        sum = (lane < (blockDim.x / 32)) ? sdata[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
        if (threadIdx.x == 0) y[row] = __float2half(sum);
    }
}
void k_int8_gemv(half* x, int8_t* W, half* scales, half* y, int in_features, int out_features) { _int8_gemv<<<out_features, 256>>>(x, W, scales, y, in_features, out_features); }

void k_half_linear(cublasHandle_t handle, half* x, half* W, half* y, int b, int in_features, int out_features) { 
    half alpha = __float2half(1.0f), beta = __float2half(0.0f); 
    cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, b, in_features, &alpha, W, in_features, x, in_features, &beta, y, out_features); 
}

// --- STABLE PREFILL ATTENTION ---
__global__ void _stable_attention_prefill(half* Q, half* K, half* V, half* O, int seq, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y, r = blockIdx.x; if (r >= seq) return;
    int kv_h = h / (n_heads / n_kv_heads), q_base = r * (n_heads * head_dim) + h * head_dim;
    float scale = 1.0f / sqrtf((float)head_dim), max_score = -1e20f;
    float scores[2048]; 
    
    for (int c = 0; c <= r; c++) {
        int k_base = c * (n_kv_heads * head_dim) + kv_h * head_dim;
        float s = 0.0f;
        for (int d = 0; d < head_dim; d++) s += __half2float(Q[q_base + d]) * __half2float(K[k_base + d]);
        scores[c] = s * scale;
        if (scores[c] > max_score) max_score = scores[c];
    }
    float sum_exp = 0.0f;
    for (int c = 0; c <= r; c++) { scores[c] = expf(scores[c] - max_score); sum_exp += scores[c]; }
    for (int d = 0; d < head_dim; d++) {
        float val = 0.0f;
        for (int c = 0; c <= r; c++) {
            int v_base = c * (n_kv_heads * head_dim) + kv_h * head_dim;
            val += scores[c] * __half2float(V[v_base + d]);
        }
        O[q_base + d] = __float2half(val / sum_exp);
    }
}
void k_flash_attention_prefill(half* Q, half* K, half* V, half* O, int seq, int n_heads, int n_kv_heads, int head_dim) { 
    dim3 g(seq, n_heads); _stable_attention_prefill<<<g, 1>>>(Q, K, V, O, seq, n_heads, n_kv_heads, head_dim); 
}

// --- CUDA GRAPH DECODE KERNELS (Fixed 4096 Stride) ---
__global__ void _half_mha_scores_one_graph(half* q, half* K_cache, float* s, int* d_pos, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x, pos = *d_pos;
    if (h < n_heads && c <= pos) {
        float sum = 0.0f; int kv_h = h / (n_heads / n_kv_heads);
        int q_base = h * head_dim, k_base = c * (n_kv_heads * head_dim) + kv_h * head_dim;
        for(int d=0; d < head_dim; d++) sum += __half2float(q[q_base+d]) * __half2float(K_cache[k_base+d]);
        // FIX: Matrix S is allocated as [n_heads x 4096]. Stride must be 4096!
        s[h * 4096 + c] = sum / sqrtf((float)head_dim);
    }
}
void k_half_mha_scores_one(half* q, half* K, float* s, int* d_pos, int n_heads, int n_kv_heads, int head_dim) { 
    dim3 g((4096)/256, n_heads), b(256); _half_mha_scores_one_graph<<<g, b>>>(q, K, s, d_pos, n_heads, n_kv_heads, head_dim); 
}

__global__ void _half_mha_weighted_sum_one_graph(float* p, half* V_cache, half* o, int* d_pos, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x, pos = *d_pos;
    if (h < n_heads && d < head_dim) {
        float sum = 0.0f; int kv_h = h / (n_heads / n_kv_heads);
        for(int c=0; c <= pos; c++) {
            // FIX: Matrix P stride is 4096
            sum += p[h * 4096 + c] * __half2float(V_cache[c*(n_kv_heads*head_dim)+kv_h*head_dim+d]);
        }
        o[h*head_dim+d] = __float2half(sum);
    }
}
void k_half_mha_weighted_sum_one(float* p, half* V, half* o, int* d_pos, int n_heads, int n_kv_heads, int head_dim) { 
    dim3 g((head_dim+255)/256, n_heads), b(256); _half_mha_weighted_sum_one_graph<<<g, b>>>(p, V, o, d_pos, n_heads, n_kv_heads, head_dim); 
}

__global__ void _row_softmax_graph(float* x, float* y, int n_heads, int* d_pos) {
    int r = blockIdx.x, pos = *d_pos, cols = pos + 1;
    if (r >= n_heads) return;
    __shared__ float m, s;
    if (threadIdx.x == 0) { 
        // FIX: Matrix stride is 4096
        float mx = x[r * 4096]; 
        for (int i = 1; i < cols; i++) if (x[r * 4096 + i] > mx) mx = x[r * 4096 + i]; 
        float sm = 0.0f; 
        for (int i = 0; i < cols; i++) sm += expf(x[r * 4096 + i] - mx); 
        m = mx; s = sm; 
    }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        y[r * 4096 + i] = expf(x[r * 4096 + i] - m) / s;
    }
}
void k_row_softmax_graph(float* x, float* y, int n_heads, int* d_pos) { _row_softmax_graph<<<n_heads, 256>>>(x, y, n_heads, d_pos); }

__global__ void _half_llama_rope_graph(half* x, int n_heads, int head_dim, int* d_pos) {
    int h = blockIdx.y, i = threadIdx.x, pos = *d_pos;
    if (h < n_heads && i < head_dim / 2) {
        int base = h * head_dim;
        float x0 = __half2float(x[base+i]), x1 = __half2float(x[base+i+head_dim/2]);
        float th = pos * powf(10000.0f, -(2.0f * i) / head_dim);
        x[base+i] = __float2half(x0*cosf(th) - x1*sinf(th)); x[base+i+head_dim/2] = __float2half(x0*sinf(th) + x1*cosf(th));
    }
}
void k_half_llama_rope_graph(half* x, int n_heads, int head_dim, int* d_pos) { dim3 g(1, n_heads); _half_llama_rope_graph<<<g, head_dim / 2>>>(x, n_heads, head_dim, d_pos); }

__global__ void _half_copy_to_cache_graph(half* src, half* cache, int* d_pos, int dim) { 
    int d = blockIdx.x * blockDim.x + threadIdx.x, pos = *d_pos; 
    if (d < dim) cache[pos * dim + d] = src[d]; 
}
void k_half_copy_to_cache_graph(half* src, half* cache, int* d_pos, int dim) { _half_copy_to_cache_graph<<<(dim + 255) / 256, 256>>>(src, cache, d_pos, dim); }

// --- UTILS (NON-GRAPH) ---
__global__ void _half_rmsnorm(half* x, half* w, half* y, int rows, int cols, float eps) {
    int r = blockIdx.x; if (r >= rows) return;
    __shared__ float s;
    if (threadIdx.x == 0) { float ss = 0.0f; for (int i = 0; i < cols; i++) ss += __half2float(x[r*cols+i])*__half2float(x[r*cols+i]); s = rsqrtf(ss/cols+eps); }
    __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) y[r*cols+i] = __float2half(__half2float(x[r*cols+i])*s*__half2float(w[i]));
}
void k_half_rmsnorm(half* x, half* w, half* y, int rows, int cols, float eps) { _half_rmsnorm<<<rows, 256>>>(x, w, y, rows, cols, eps); }

__global__ void _half_fused_add_rmsnorm(half* x, half* res, half* w, half* out, int rows, int cols, float eps) {
    int r = blockIdx.x; if (r >= rows) return;
    __shared__ float s; if (threadIdx.x == 0) s = 0.0f; __syncthreads();
    float ss = 0.0f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = __half2float(x[r * cols + i]) + __half2float(res[r * cols + i]);
        x[r * cols + i] = __float2half(val); ss += val * val;
    }
    for (int offset = 16; offset > 0; offset /= 2) ss += __shfl_down_sync(0xffffffff, ss, offset);
    if (threadIdx.x % 32 == 0) atomicAdd(&s, ss); __syncthreads();
    if (threadIdx.x == 0) s = rsqrtf(s / cols + eps); __syncthreads();
    for (int i = threadIdx.x; i < cols; i += blockDim.x) out[r * cols + i] = __float2half(__half2float(x[r * cols + i]) * s * __half2float(w[i]));
}
void k_half_fused_add_rmsnorm(half* x, half* res, half* w, half* out, int rows, int cols, float eps) { _half_fused_add_rmsnorm<<<rows, 32>>>(x, res, w, out, rows, cols, eps); }

__global__ void _half_llama_rope(half* x, int seq, int n_heads, int head_dim, int pos_base) {
    int pos = blockIdx.x, h = blockIdx.y, i = threadIdx.x;
    if (pos < seq && h < n_heads && i < head_dim / 2) {
        int base = pos * (n_heads * head_dim) + h * head_dim;
        float x0 = __half2float(x[base+i]), x1 = __half2float(x[base+i+head_dim/2]);
        float th = (pos_base + pos) * powf(10000.0f, -(2.0f * i) / head_dim);
        x[base+i] = __float2half(x0*cosf(th) - x1*sinf(th)); x[base+i+head_dim/2] = __float2half(x0*sinf(th) + x1*cosf(th));
    }
}
void k_half_llama_rope(half* x, int seq, int n_heads, int head_dim, int pos_base) { dim3 g(seq, n_heads); _half_llama_rope<<<g, head_dim / 2>>>(x, seq, n_heads, head_dim, pos_base); }

__global__ void _half_add(half* a, half* b, half* c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = __float2half(__half2float(a[i]) + __half2float(b[i])); }
void k_half_add(half* a, half* b, half* c, int n) { _half_add<<<(n+255)/256, 256>>>(a, b, c, n); }

__global__ void _half_swiglu(half* gate, half* up, half* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float g = __half2float(gate[i]), u = __half2float(up[i]); out[i] = __float2half((g / (1.0f + expf(-g))) * u); } }
void k_half_swiglu(half* gate, half* up, half* out, int n) { _half_swiglu<<<(n+255)/256, 256>>>(gate, up, out, n); }

__global__ void _half_embedding_lookup(int* ids, half* table, half* out, int seq, int dim) { int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x; if (t < seq && d < dim) out[t*dim+d] = table[ids[t]*dim+d]; }
void k_half_embedding_lookup(int* ids, half* table, half* out, int seq, int dim) { dim3 g((dim+255)/256, seq), b(256); _half_embedding_lookup<<<g, b>>>(ids, table, out, seq, dim); }

__global__ void _half_copy_block_to_cache(half* src, half* cache, int pos_base, int seq_len, int dim) { int seq_idx = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x; if (seq_idx < seq_len && d < dim) cache[(pos_base + seq_idx) * dim + d] = src[seq_idx * dim + d]; }
void k_half_copy_block_to_cache(half* src, half* cache, int pos_base, int seq_len, int dim) { dim3 g((dim+255)/256, seq_len), b(256); _half_copy_block_to_cache<<<g, b>>>(src, cache, pos_base, seq_len, dim); }

__global__ void _half_argmax_row(half* x, int* out, int rows, int cols) { int r = blockIdx.x; if (r < rows && threadIdx.x == 0) { int idx = 0; float best = __half2float(x[r*cols]); for (int i=1; i<cols; i++) { float val = __half2float(x[r*cols+i]); if (val > best) { best = val; idx = i; } } out[r] = idx; } }
void k_half_argmax_row(half* x, int* out, int rows, int cols) { _half_argmax_row<<<rows, 1>>>(x, out, rows, cols); }

__global__ void _argmax_row(float* x, int* out, int rows, int cols) { int r = blockIdx.x; if (r < rows && threadIdx.x == 0) { int idx = 0; float best = x[r*cols]; for (int i=1; i<cols; i++) { if (x[r*cols+i] > best) { best = x[r*cols+i]; idx = i; } } out[r] = idx; } }
void k_argmax_row(float* x, int* out, int rows, int cols) { _argmax_row<<<rows, 1>>>(x, out, rows, cols); }

__global__ void _row_softmax(float* x, float* y, int rows, int cols) { int r = blockIdx.x; if (r >= rows) return; __shared__ float m, s; if (threadIdx.x == 0) { float mx = x[r * cols]; for (int i = 1; i < cols; i++) if (x[r * cols + i] > mx) mx = x[r * cols + i]; float sm = 0.0f; for (int i = 0; i < cols; i++) sm += expf(x[r * cols + i] - mx); m = mx; s = sm; } __syncthreads(); for (int i = threadIdx.x; i < cols; i += blockDim.x) y[r * cols + i] = expf(x[r * cols + i] - m) / s; }
void k_row_softmax(float* x, float* y, int rows, int cols) { _row_softmax<<<rows, 256>>>(x, y, rows, cols); }

// STUBS
void k_cublas_gemm(cublasHandle_t handle, float* A, float* B, float* C, int M, int N, int K) {}
void k_fp32_to_fp16(float* src, half* dst, int n) {}
void k_gemm_wmma(half* A, half* B, float* C, int M, int N, int K) {}
void k_apply_temperature(float* logits, float temp, int vocab_size) {}
void k_sample_top_p(float* probs, int* out_idx, float p, float random_val, int vocab_size) {}
void k_apply_repetition_penalty(float* logits, int* past_tokens, int num_past, float penalty) {}
void k_gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {}
void k_add(float* a, float* b, float* c, int n) {}
void k_rmsnorm(float* x, float* w, float* y, int rows, int cols, float eps) {}
void k_silu(float* x, float* y, int n) {}
void k_swiglu(float* gate, float* up, float* out, int n) {}
void k_rope(float* x, int seq, int dim, int pos_base) {}
void k_attention_scores(float* Q, float* K, float* S, int seq, int dim) {}
void k_apply_causal_mask(float* S, int seq) {}
void k_attention_weighted_sum(float* P, float* V, float* O, int seq, int dim) {}
void k_mha_scores_fused_mask(float* Q, float* K, float* S, int seq, int n_heads, int n_kv_heads, int head_dim) {}
void k_mha_weighted_sum(float* P, float* V, float* O, int seq, int n_heads, int n_kv_heads, int head_dim) {}
void k_mha_scores_one(float* q, float* K_cache, float* s, int pos, int n_heads, int n_kv_heads, int head_dim) {}
void k_mha_weighted_sum_one(float* p, float* V_cache, float* o, int pos, int n_heads, int n_kv_heads, int head_dim) {}
void k_embedding_lookup(int* ids, float* table, float* out, int seq, int dim) {}
void k_gather_last_token(float* x, float* out, int seq, int dim) {}
void k_row_add_bias(float* x, float* b, int rows, int cols) {}
void k_copy_row_to_cache(float* src_row, float* cache, int pos, int dim) {}
void k_gemv(float* x, float* W, float* y, int K, int N) {}
void k_fused_add_rmsnorm(float* x, float* residual_in, float* w, float* norm_out, int rows, int cols, float eps) {}
void k_mul(float* a, float* b, float* c, int n) {}
void k_scale(float* a, float s, float* c, int n) {}
void k_fill(float* a, float val, int n) {}
void k_copy(float* src, float* dst, int n) {}
void k_gemm_naive(float* A, float* B, float* C, int M, int N, int K) {}
void k_transpose(float* in, float* out, int rows, int cols) {}
void k_linear(float* x, float* W, float* y, int in_features, int out_features) {}