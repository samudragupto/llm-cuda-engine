#include "kernels.cuh"
#include <cuda_runtime.h>
#include <math.h>
#define TILE 32

__global__ void _add(float* a,float* b,float* c,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]+b[i];}
__global__ void _mul(float* a,float* b,float* c,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]*b[i];}
__global__ void _scale(float* a,float s,float* c,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)c[i]=a[i]*s;}
__global__ void _fill(float* a,float v,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)a[i]=v;}
__global__ void _copy(float* s,float* d,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)d[i]=s[i];}

__global__ void _gemm_naive(float* A,float* B,float* C,int M,int N,int K){
    int r=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
    if(r<M&&c<N){float s=0;for(int i=0;i<K;i++)s+=A[r*K+i]*B[i*N+c];C[r*N+c]=s;}
}

__global__ void _gemm_tiled(float* A,float* B,float* C,int M,int N,int K){
    __shared__ float As[TILE][TILE],Bs[TILE][TILE+1];
    int r=blockIdx.y*TILE+threadIdx.y,c=blockIdx.x*TILE+threadIdx.x;
    float s=0;
    for(int t=0;t<(K+TILE-1)/TILE;t++){
        int ak=t*TILE+threadIdx.x,bk=t*TILE+threadIdx.y;
        As[threadIdx.y][threadIdx.x]=(r<M&&ak<K)?A[r*K+ak]:0.0f;
        Bs[threadIdx.y][threadIdx.x]=(bk<K&&c<N)?B[bk*N+c]:0.0f;
        __syncthreads();
        #pragma unroll
        for(int i=0;i<TILE;i++)s+=As[threadIdx.y][i]*Bs[i][threadIdx.x];
        __syncthreads();
    }
    if(r<M&&c<N)C[r*N+c]=s;
}

__global__ void _transpose(float* in,float* out,int rows,int cols){
    __shared__ float tile[TILE][TILE+1];
    int x=blockIdx.x*TILE+threadIdx.x,y=blockIdx.y*TILE+threadIdx.y;
    if(x<cols&&y<rows)tile[threadIdx.y][threadIdx.x]=in[y*cols+x];
    __syncthreads();
    x=blockIdx.y*TILE+threadIdx.x;y=blockIdx.x*TILE+threadIdx.y;
    if(x<rows&&y<cols)out[y*rows+x]=tile[threadIdx.x][threadIdx.y];
}

__global__ void _rmsnorm(float* x,float* w,float* y,int rows,int cols,float eps){
    int r=blockIdx.x;
    if(r>=rows)return;
    __shared__ float s;
    if(threadIdx.x==0){
        float ss=0.0f;
        for(int i=0;i<cols;i++){float v=x[r*cols+i];ss+=v*v;}
        s=rsqrtf(ss/cols+eps);
    }
    __syncthreads();
    for(int i=threadIdx.x;i<cols;i+=blockDim.x)y[r*cols+i]=x[r*cols+i]*s*w[i];
}

__global__ void _silu(float* x,float* y,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){float v=x[i];y[i]=v/(1.0f+expf(-v));}
}

__global__ void _row_softmax(float* x,float* y,int rows,int cols){
    int r=blockIdx.x;
    if(r>=rows)return;
    __shared__ float m,s;
    if(threadIdx.x==0){
        float mx=x[r*cols];
        for(int i=1;i<cols;i++)if(x[r*cols+i]>mx)mx=x[r*cols+i];
        float sm=0.0f;
        for(int i=0;i<cols;i++)sm+=expf(x[r*cols+i]-mx);
        m=mx;s=sm;
    }
    __syncthreads();
    for(int i=threadIdx.x;i<cols;i+=blockDim.x)y[r*cols+i]=expf(x[r*cols+i]-m)/s;
}

__device__ float rope_freq(int idx,int dim){
    float inv=powf(10000.0f,-(2.0f*(idx/2))/dim);
    return inv;
}

__global__ void _rope(float* x,int seq,int dim,int pos_base){
    int pos=blockIdx.x;
    int i=threadIdx.x*2;
    if(pos<seq&&i+1<dim){
        int base=pos*dim;
        float a=x[base+i],b=x[base+i+1];
        float th=(pos_base+pos)*rope_freq(i,dim);
        float c=cosf(th),s=sinf(th);
        x[base+i]=a*c-b*s;
        x[base+i+1]=a*s+b*c;
    }
}

__global__ void _attention_scores(float* Q,float* K,float* S,int seq,int dim){
    int i=blockIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<seq&&j<seq){
        float sum=0.0f;
        for(int d=0;d<dim;d++)sum+=Q[i*dim+d]*K[j*dim+d];
        S[i*seq+j]=sum/sqrtf((float)dim);
    }
}

__global__ void _apply_causal_mask(float* S,int seq){
    int i=blockIdx.y,j=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<seq&&j<seq&&j>i)S[i*seq+j]=-1e20f;
}

__global__ void _attention_weighted_sum(float* P,float* V,float* O,int seq,int dim){
    int i=blockIdx.y,d=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<seq&&d<dim){
        float sum=0.0f;
        for(int j=0;j<seq;j++)sum+=P[i*seq+j]*V[j*dim+d];
        O[i*dim+d]=sum;
    }
}

__global__ void _embedding_lookup(int* ids,float* table,float* out,int seq,int dim){
    int t=blockIdx.y,d=blockIdx.x*blockDim.x+threadIdx.x;
    if(t<seq&&d<dim){int id=ids[t];out[t*dim+d]=table[id*dim+d];}
}

__global__ void _gather_last_token(float* x,float* out,int seq,int dim){
    int d=blockIdx.x*blockDim.x+threadIdx.x;
    if(d<dim)out[d]=x[(seq-1)*dim+d];
}

__global__ void _argmax_row(float* x,int* out,int rows,int cols){
    int r=blockIdx.x;
    if(r<rows&&threadIdx.x==0){
        int idx=0; float best=x[r*cols];
        for(int i=1;i<cols;i++) if(x[r*cols+i]>best){best=x[r*cols+i];idx=i;}
        out[r]=idx;
    }
}

__global__ void _row_add_bias(float* x,float* b,int rows,int cols){
    int r=blockIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;
    if(r<rows&&c<cols)x[r*cols+c]+=b[c];
}

__global__ void _copy_row_to_cache(float* src_row,float* cache,int pos,int dim){
    int d=blockIdx.x*blockDim.x+threadIdx.x;
    if(d<dim)cache[pos*dim+d]=src_row[d];
}

__global__ void _attention_scores_one(float* q,float* K,float* s,int len,int dim){
    int j=blockIdx.x*blockDim.x+threadIdx.x;
    if(j<len){
        float sum=0.0f;
        for(int d=0;d<dim;d++)sum+=q[d]*K[j*dim+d];
        s[j]=sum/sqrtf((float)dim);
    }
}

__global__ void _attention_weighted_sum_one(float* p,float* V,float* o,int len,int dim){
    int d=blockIdx.x*blockDim.x+threadIdx.x;
    if(d<dim){
        float sum=0.0f;
        for(int j=0;j<len;j++)sum+=p[j]*V[j*dim+d];
        o[d]=sum;
    }
}

// --- PHASE 2 UPGRADES ---

// SwiGLU: out = SiLU(gate) * up
__global__ void _swiglu(float* gate, float* up, float* out, int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){
        float g = gate[i];
        float silu = g / (1.0f + expf(-g));
        out[i] = silu * up[i];
    }
}

// Multi-Head Attention Scores + Fused Causal Mask
// Q, K: [seq, n_heads, head_dim] -> S: [n_heads, seq, seq]
__global__ void _mha_scores_fused_mask(float* Q, float* K, float* S, int seq, int n_heads, int head_dim){
    int h = blockIdx.z;
    int r = blockIdx.y; // seq query pos
    int c = blockIdx.x * blockDim.x + threadIdx.x; // seq key pos
    
    if(h < n_heads && r < seq && c < seq){
        if (c > r) { // Fused Causal Mask
            S[h*seq*seq + r*seq + c] = -1e20f;
            return;
        }
        float sum = 0.0f;
        int q_base = r * (n_heads * head_dim) + h * head_dim;
        int k_base = c * (n_heads * head_dim) + h * head_dim;
        for(int d=0; d<head_dim; d++){
            sum += Q[q_base + d] * K[k_base + d];
        }
        S[h*seq*seq + r*seq + c] = sum / sqrtf((float)head_dim);
    }
}

// Multi-Head Attention Output
// P: [n_heads, seq, seq], V: [seq, n_heads, head_dim] -> O: [seq, n_heads, head_dim]
__global__ void _mha_weighted_sum(float* P, float* V, float* O, int seq, int n_heads, int head_dim){
    int h = blockIdx.z;
    int r = blockIdx.y; // seq pos
    int d = blockIdx.x * blockDim.x + threadIdx.x; // head_dim
    
    if(h < n_heads && r < seq && d < head_dim){
        float sum = 0.0f;
        for(int c=0; c<seq; c++){
            int p_idx = h*seq*seq + r*seq + c;
            int v_idx = c*(n_heads*head_dim) + h*head_dim + d;
            sum += P[p_idx] * V[v_idx];
        }
        O[r*(n_heads*head_dim) + h*head_dim + d] = sum;
    }
}
// --- PHASE 3 UPGRADES: SAMPLING ---

__global__ void _apply_temperature(float* logits, float temp, int vocab_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < vocab_size) {
        logits[i] /= temp;
    }
}

// A simple cumulative sum sampler. 
// Note: In a true prod engine (vLLM/llama.cpp), this involves a fused bitonic sort for Top-K. 
// For this stage, we do a basic linear scan on a single thread block.
__global__ void _sample_top_p(float* probs, int* out_idx, float p, float random_val, int vocab_size) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float cdf = 0.0f;
        int last_idx = vocab_size - 1;
        for (int i = 0; i < vocab_size; i++) {
            cdf += probs[i];
            if (cdf >= random_val) {
                out_idx[0] = i;
                return;
            }
        }
        out_idx[0] = last_idx; // Fallback
    }
}
// --- PHASE 3 UPGRADES: REPETITION PENALTY ---
__global__ void _apply_repetition_penalty(float* logits, int* past_tokens, int num_past, float penalty) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < num_past) {
        int token_id = past_tokens[i];
        float logit = logits[token_id];
        // Standard HuggingFace repetition penalty math
        if (logit > 0.0f) {
            logits[token_id] = logit / penalty;
        } else {
            logits[token_id] = logit * penalty;
        }
    }
}
void k_add(float* a,float* b,float* c,int n){_add<<<(n+255)/256,256>>>(a,b,c,n);}
void k_mul(float* a,float* b,float* c,int n){_mul<<<(n+255)/256,256>>>(a,b,c,n);}
void k_scale(float* a,float s,float* c,int n){_scale<<<(n+255)/256,256>>>(a,s,c,n);}
void k_fill(float* a,float val,int n){_fill<<<(n+255)/256,256>>>(a,val,n);}
void k_copy(float* src,float* dst,int n){_copy<<<(n+255)/256,256>>>(src,dst,n);}
void k_gemm_naive(float* A,float* B,float* C,int M,int N,int K){dim3 g((N+15)/16,(M+15)/16),b(16,16);_gemm_naive<<<g,b>>>(A,B,C,M,N,K);}
void k_gemm_tiled(float* A,float* B,float* C,int M,int N,int K){dim3 g((N+TILE-1)/TILE,(M+TILE-1)/TILE),b(TILE,TILE);_gemm_tiled<<<g,b>>>(A,B,C,M,N,K);}
void k_transpose(float* in,float* out,int rows,int cols){dim3 g((cols+TILE-1)/TILE,(rows+TILE-1)/TILE),b(TILE,TILE);_transpose<<<g,b>>>(in,out,rows,cols);}
void k_rmsnorm(float* x,float* w,float* y,int rows,int cols,float eps){_rmsnorm<<<rows,256>>>(x,w,y,rows,cols,eps);}
void k_silu(float* x,float* y,int n){_silu<<<(n+255)/256,256>>>(x,y,n);}
void k_row_softmax(float* x,float* y,int rows,int cols){_row_softmax<<<rows,256>>>(x,y,rows,cols);}
void k_rope(float* x,int seq,int dim,int pos_base){_rope<<<seq,dim/2>>>(x,seq,dim,pos_base);}
void k_attention_scores(float* Q,float* K,float* S,int seq,int dim){dim3 g((seq+255)/256,seq),b(256);_attention_scores<<<g,b>>>(Q,K,S,seq,dim);}
void k_apply_causal_mask(float* S,int seq){dim3 g((seq+255)/256,seq),b(256);_apply_causal_mask<<<g,b>>>(S,seq);}
void k_attention_weighted_sum(float* P,float* V,float* O,int seq,int dim){dim3 g((dim+255)/256,seq),b(256);_attention_weighted_sum<<<g,b>>>(P,V,O,seq,dim);}
void k_embedding_lookup(int* ids,float* table,float* out,int seq,int dim){dim3 g((dim+255)/256,seq),b(256);_embedding_lookup<<<g,b>>>(ids,table,out,seq,dim);}
void k_gather_last_token(float* x,float* out,int seq,int dim){_gather_last_token<<<(dim+255)/256,256>>>(x,out,seq,dim);}
void k_argmax_row(float* x,int* out,int rows,int cols){_argmax_row<<<rows,1>>>(x,out,rows,cols);}
void k_row_add_bias(float* x,float* b,int rows,int cols){dim3 g((cols+255)/256,rows),bb(256);_row_add_bias<<<g,bb>>>(x,b,rows,cols);}
void k_copy_row_to_cache(float* src_row,float* cache,int pos,int dim){_copy_row_to_cache<<<(dim+255)/256,256>>>(src_row,cache,pos,dim);}
void k_attention_scores_one(float* q,float* K,float* s,int len,int dim){_attention_scores_one<<<(len+255)/256,256>>>(q,K,s,len,dim);}
void k_attention_weighted_sum_one(float* p,float* V,float* o,int len,int dim){_attention_weighted_sum_one<<<(dim+255)/256,256>>>(p,V,o,len,dim);}
void k_swiglu(float* gate, float* up, float* out, int n){ _swiglu<<<(n+255)/256, 256>>>(gate,up,out,n); }

void k_mha_scores_fused_mask(float* Q, float* K, float* S, int seq, int n_heads, int head_dim){
    dim3 g((seq+31)/32, seq, n_heads), b(32);
    _mha_scores_fused_mask<<<g,b>>>(Q,K,S,seq,n_heads,head_dim);
}

void k_mha_weighted_sum(float* P, float* V, float* O, int seq, int n_heads, int head_dim){
    dim3 g((head_dim+31)/32, seq, n_heads), b(32);
    _mha_weighted_sum<<<g,b>>>(P,V,O,seq,n_heads,head_dim);
}
void k_apply_temperature(float* logits, float temp, int vocab_size) {
    _apply_temperature<<<(vocab_size+255)/256, 256>>>(logits, temp, vocab_size);
}
void k_sample_top_p(float* probs, int* out_idx, float p, float random_val, int vocab_size) {
    _sample_top_p<<<1, 1>>>(probs, out_idx, p, random_val, vocab_size);
}
void k_apply_repetition_penalty(float* logits, int* past_tokens, int num_past, float penalty) {
    _apply_repetition_penalty<<<(num_past+255)/256, 256>>>(logits, past_tokens, num_past, penalty);
}
// --- PHASE 4 UPGRADES: cuBLAS Baseline ---
void k_cublas_gemm(cublasHandle_t handle, float* A, float* B, float* C, int M, int N, int K) {
    float alpha = 1.0f;
    float beta = 0.0f;
    
    // Note: cuBLAS uses column-major order by default.
    // Our custom kernels assume row-major. 
    // To do C = A * B in row-major, we compute C^T = B^T * A^T in column-major.
    // That means we swap A and B in the cublasSgemm call and tell it not to transpose them.
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N);
}