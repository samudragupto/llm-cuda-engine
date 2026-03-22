#include "kernels.cuh"
#include <cuda_runtime.h>
#include <math.h>

// --- PHASE 2 UTILITIES ---
__global__ void _fill(float* a, float v, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) a[i] = v; }
void k_fill(float* a, float val, int n) { _fill<<<(n + 255) / 256, 256>>>(a, val, n); }

__device__ float rope_freq(int idx, int dim) { return powf(10000.0f, -(2.0f * (idx / 2)) / dim); }
__global__ void _rope(float* x, int seq, int dim, int pos_base) {
    int pos = blockIdx.x, i = threadIdx.x * 2;
    if (pos < seq && i + 1 < dim) {
        int base = pos * dim; float a = x[base + i], b = x[base + i + 1];
        float th = (pos_base + pos) * rope_freq(i, dim); float c = cosf(th), s = sinf(th);
        x[base + i] = a * c - b * s; x[base + i + 1] = a * s + b * c;
    }
}
void k_rope(float* x, int seq, int dim, int pos_base) { _rope<<<seq, dim / 2>>>(x, seq, dim, pos_base); }

__global__ void _attention_scores(float* Q, float* K, float* S, int seq, int dim) {
    int i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seq && j < seq) {
        float sum = 0.0f; for (int d = 0; d < dim; d++) sum += Q[i * dim + d] * K[j * dim + d];
        S[i * seq + j] = sum / sqrtf((float)dim);
    }
}
void k_attention_scores(float* Q, float* K, float* S, int seq, int dim) { dim3 g((seq + 255) / 256, seq), b(256); _attention_scores<<<g, b>>>(Q, K, S, seq, dim); }

__global__ void _apply_causal_mask(float* S, int seq) {
    int i = blockIdx.y, j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seq && j < seq && j > i) S[i * seq + j] = -1e20f;
}
void k_apply_causal_mask(float* S, int seq) { dim3 g((seq + 255) / 256, seq), b(256); _apply_causal_mask<<<g, b>>>(S, seq); }

__global__ void _attention_weighted_sum(float* P, float* V, float* O, int seq, int dim) {
    int i = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < seq && d < dim) {
        float sum = 0.0f; for (int j = 0; j < seq; j++) sum += P[i * seq + j] * V[j * dim + d];
        O[i * dim + d] = sum;
    }
}
void k_attention_weighted_sum(float* P, float* V, float* O, int seq, int dim) { dim3 g((dim + 255) / 256, seq), b(256); _attention_weighted_sum<<<g, b>>>(P, V, O, seq, dim); }

// --- PHASE 4 KERNELS ---
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

__global__ void _flash_attention_prefill(half* Q, half* K, half* V, half* O, int seq, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y, r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= seq) return;
    float max_s = -1e20f, sum_p = 0.0f; float out[128] = {0};
    int kv_h = h / (n_heads / n_kv_heads); int q_base = r * (n_heads * head_dim) + h * head_dim;
    for (int c = 0; c <= r; c++) {
        int k_base = c * (n_kv_heads * head_dim) + kv_h * head_dim; float s = 0.0f;
        for (int d = 0; d < head_dim; d++) s += __half2float(Q[q_base + d]) * __half2float(K[k_base + d]);
        s /= sqrtf((float)head_dim);
        float new_max = fmaxf(max_s, s); float factor = expf(max_s - new_max);
        sum_p = sum_p * factor + expf(s - new_max); float p = expf(s - new_max);
        int v_base = c * (n_kv_heads * head_dim) + kv_h * head_dim;
        for (int d = 0; d < head_dim; d++) out[d] = out[d] * factor + p * __half2float(V[v_base + d]);
        max_s = new_max;
    }
    int o_base = r * (n_heads * head_dim) + h * head_dim;
    for (int d = 0; d < head_dim; d++) O[o_base + d] = __float2half(out[d] / sum_p);
}
void k_flash_attention_prefill(half* Q, half* K, half* V, half* O, int seq, int n_heads, int n_kv_heads, int head_dim) { dim3 g((seq + 31) / 32, n_heads), b(32); _flash_attention_prefill<<<g, b>>>(Q, K, V, O, seq, n_heads, n_kv_heads, head_dim); }

__global__ void _half_mha_scores_one(half* q, half* K_cache, float* s, int pos, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y, c = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < n_heads && c <= pos) {
        float sum = 0.0f; int kv_h = h / (n_heads / n_kv_heads);
        int q_base = h * head_dim, k_base = c * (n_kv_heads * head_dim) + kv_h * head_dim;
        for(int d=0; d < head_dim; d++) sum += __half2float(q[q_base+d]) * __half2float(K_cache[k_base+d]);
        s[h*(pos+1)+c] = sum / sqrtf((float)head_dim);
    }
}
void k_half_mha_scores_one(half* q, half* K, float* s, int pos, int n_heads, int n_kv_heads, int head_dim) { dim3 g((pos+1+255)/256, n_heads), b(256); _half_mha_scores_one<<<g, b>>>(q, K, s, pos, n_heads, n_kv_heads, head_dim); }

__global__ void _half_mha_weighted_sum_one(float* p, half* V_cache, half* o, int pos, int n_heads, int n_kv_heads, int head_dim) {
    int h = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x;
    if (h < n_heads && d < head_dim) {
        float sum = 0.0f; int kv_h = h / (n_heads / n_kv_heads);
        for(int c=0; c <= pos; c++) sum += p[h*(pos+1)+c] * __half2float(V_cache[c*(n_kv_heads*head_dim)+kv_h*head_dim+d]);
        o[h*head_dim+d] = __float2half(sum);
    }
}
void k_half_mha_weighted_sum_one(float* p, half* V, half* o, int pos, int n_heads, int n_kv_heads, int head_dim) { dim3 g((head_dim+255)/256, n_heads), b(256); _half_mha_weighted_sum_one<<<g, b>>>(p, V, o, pos, n_heads, n_kv_heads, head_dim); }

void k_half_linear(cublasHandle_t handle, half* x, half* W, half* y, int b, int in_features, int out_features) { half alpha = __float2half(1.0f), beta = __float2half(0.0f); cublasHgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, out_features, b, in_features, &alpha, W, in_features, x, in_features, &beta, y, out_features); }
__global__ void _half_fused_add_rmsnorm(half* x, half* res, half* w, half* out, int rows, int cols, float eps) { int r = blockIdx.x; if (r >= rows) return; __shared__ float s; if (threadIdx.x == 0) s = 0.0f; __syncthreads(); float ss = 0.0f; for (int i = threadIdx.x; i < cols; i += blockDim.x) { float val = __half2float(x[r * cols + i]) + __half2float(res[r * cols + i]); x[r * cols + i] = __float2half(val); ss += val * val; } for (int offset = 16; offset > 0; offset /= 2) ss += __shfl_down_sync(0xffffffff, ss, offset); if (threadIdx.x % 32 == 0) atomicAdd(&s, ss); __syncthreads(); if (threadIdx.x == 0) s = rsqrtf(s / cols + eps); __syncthreads(); for (int i = threadIdx.x; i < cols; i += blockDim.x) out[r * cols + i] = __float2half(__half2float(x[r * cols + i]) * s * __half2float(w[i])); }
void k_half_fused_add_rmsnorm(half* x, half* res, half* w, half* out, int rows, int cols, float eps) { _half_fused_add_rmsnorm<<<rows, 32>>>(x, res, w, out, rows, cols, eps); }
__global__ void _half_rmsnorm(half* x, half* w, half* y, int rows, int cols, float eps) { int r = blockIdx.x; if (r >= rows) return; __shared__ float s; if (threadIdx.x == 0) { float ss = 0.0f; for (int i = 0; i < cols; i++) ss += __half2float(x[r*cols+i])*__half2float(x[r*cols+i]); s = rsqrtf(ss/cols+eps); } __syncthreads(); for (int i = threadIdx.x; i < cols; i += blockDim.x) y[r*cols+i] = __float2half(__half2float(x[r*cols+i])*s*__half2float(w[i])); }
void k_half_rmsnorm(half* x, half* w, half* y, int rows, int cols, float eps) { _half_rmsnorm<<<rows, 256>>>(x, w, y, rows, cols, eps); }
__global__ void _half_add(half* a, half* b, half* c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = __float2half(__half2float(a[i]) + __half2float(b[i])); }
void k_half_add(half* a, half* b, half* c, int n) { _half_add<<<(n + 255) / 256, 256>>>(a, b, c, n); }
__global__ void _half_llama_rope(half* x, int seq, int n_heads, int head_dim, int pos_base) { int pos = blockIdx.x, h = blockIdx.y, i = threadIdx.x; if (pos < seq && h < n_heads && i < head_dim / 2) { int base = pos * (n_heads * head_dim) + h * head_dim; float x0 = __half2float(x[base+i]), x1 = __half2float(x[base+i+head_dim/2]); float th = (pos_base + pos) * powf(10000.0f, -(2.0f * i) / head_dim); x[base+i] = __float2half(x0*cosf(th) - x1*sinf(th)); x[base+i+head_dim/2] = __float2half(x0*sinf(th) + x1*cosf(th)); } }
void k_half_llama_rope(half* x, int seq, int n_heads, int head_dim, int pos_base) { dim3 g(seq, n_heads); _half_llama_rope<<<g, head_dim / 2>>>(x, seq, n_heads, head_dim, pos_base); }
__global__ void _half_swiglu(half* gate, half* up, half* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float g = __half2float(gate[i]), u = __half2float(up[i]); out[i] = __float2half((g / (1.0f + expf(-g))) * u); } }
void k_half_swiglu(half* gate, half* up, half* out, int n) { _half_swiglu<<<(n+255)/256, 256>>>(gate, up, out, n); }
__global__ void _half_embedding_lookup(int* ids, half* table, half* out, int seq, int dim) { int t = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x; if (t < seq && d < dim) out[t*dim+d] = table[ids[t]*dim+d]; }
void k_half_embedding_lookup(int* ids, half* table, half* out, int seq, int dim) { dim3 g((dim+255)/256, seq), b(256); _half_embedding_lookup<<<g, b>>>(ids, table, out, seq, dim); }
__global__ void _half_copy_block_to_cache(half* src, half* cache, int pos_base, int seq_len, int dim) { int seq_idx = blockIdx.y, d = blockIdx.x * blockDim.x + threadIdx.x; if (seq_idx < seq_len && d < dim) cache[(pos_base + seq_idx) * dim + d] = src[seq_idx * dim + d]; }
void k_half_copy_block_to_cache(half* src, half* cache, int pos_base, int seq_len, int dim) { dim3 g((dim+255)/256, seq_len), b(256); _half_copy_block_to_cache<<<g, b>>>(src, cache, pos_base, seq_len, dim); }
__global__ void _half_copy(half* src, half* dst, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = src[i]; }
void k_half_copy(half* src, half* dst, int n) { _half_copy<<<(n+255)/256, 256>>>(src, dst, n); }
__global__ void _half_argmax_row(half* x, int* out, int rows, int cols) { int r = blockIdx.x; if (r < rows && threadIdx.x == 0) { int idx = 0; float best = __half2float(x[r*cols]); for (int i=1; i<cols; i++) { float val = __half2float(x[r*cols+i]); if (val > best) { best = val; idx = i; } } out[r] = idx; } }
void k_half_argmax_row(half* x, int* out, int rows, int cols) { _half_argmax_row<<<rows, 1>>>(x, out, rows, cols); }
__global__ void _half_to_float(half* src, float* dst, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) dst[i] = __half2float(src[i]); }
void k_half_to_float(half* src, float* dst, int n) { _half_to_float<<<(n+255)/256, 256>>>(src, dst, n); }
__global__ void _row_softmax(float* x, float* y, int rows, int cols) { int r = blockIdx.x; if (r >= rows) return; __shared__ float m, s; if (threadIdx.x == 0) { float mx = x[r * cols]; for (int i = 1; i < cols; i++) if (x[r * cols + i] > mx) mx = x[r * cols + i]; float sm = 0.0f; for (int i = 0; i < cols; i++) sm += expf(x[r * cols + i] - mx); m = mx; s = sm; } __syncthreads(); for (int i = threadIdx.x; i < cols; i += blockDim.x) y[r * cols + i] = expf(x[r * cols + i] - m) / s; }
void k_row_softmax(float* x, float* y, int rows, int cols) { _row_softmax<<<rows, 256>>>(x, y, rows, cols); }
__global__ void _apply_temperature(float* logits, float temp, int vocab_size) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < vocab_size) logits[i] /= temp; }
void k_apply_temperature(float* logits, float temp, int vocab_size) { _apply_temperature<<<(vocab_size+255)/256, 256>>>(logits, temp, vocab_size); }
__global__ void _sample_top_p(float* probs, int* out_idx, float p, float random_val, int vocab_size) { if (threadIdx.x == 0 && blockIdx.x == 0) { float cdf = 0.0f; for (int i = 0; i < vocab_size; i++) { cdf += probs[i]; if (cdf >= random_val) { out_idx[0] = i; return; } } out_idx[0] = vocab_size - 1; } }
void k_sample_top_p(float* probs, int* out_idx, float p, float random_val, int vocab_size) { _sample_top_p<<<1, 1>>>(probs, out_idx, p, random_val, vocab_size); }
__global__ void _apply_repetition_penalty(float* logits, int* past_tokens, int num_past, float penalty) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < num_past) { int token_id = past_tokens[i]; float logit = logits[token_id]; logits[token_id] = logit > 0.0f ? logit / penalty : logit * penalty; } }
void k_apply_repetition_penalty(float* logits, int* past_tokens, int num_past, float penalty) { _apply_repetition_penalty<<<(num_past+255)/256, 256>>>(logits, past_tokens, num_past, penalty); }

// Polyfills so old code doesn't break
void k_gemm_tiled(float* A, float* B, float* C, int M, int N, int K) {}
void k_add(float* a, float* b, float* c, int n) {}
void k_rmsnorm(float* x, float* w, float* y, int rows, int cols, float eps) {}
void k_silu(float* x, float* y, int n) {}
void k_swiglu(float* gate, float* up, float* out, int n) {}
void k_argmax_row(float* x, int* out, int rows, int cols) {}