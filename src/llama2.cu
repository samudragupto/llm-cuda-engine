#include "llama2.h"
#include "kernels.cuh"
#include "cuda_check.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>

HalfTensor::HalfTensor(MemPool& pool, std::vector<int> s) : shape(s) {
    numel = 1; for (int d : s) numel *= d; data = pool.alloc<half>(numel);
}

void read_into_half_tensor(FILE* f, HalfTensor& t) {
    size_t bytes = t.numel * sizeof(half); half* host_buf = (half*)malloc(bytes);
    fread(host_buf, sizeof(half), t.numel, f);
    cudaMemcpy(t.data, host_buf, bytes, cudaMemcpyHostToDevice);
    free(host_buf);
}

void LlamaTokenizer::load(const char* path) {
    FILE* f = fopen(path, "rb"); int vocab_size; fread(&vocab_size, sizeof(int), 1, f); vocab.resize(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        int len; fread(&len, sizeof(int), 1, f);
        std::string s(len, '\0'); fread(&s[0], 1, len, f); vocab[i] = s;
    } fclose(f);
}
std::string LlamaTokenizer::decode(int id) {
    if (id < 0 || id >= vocab.size() || id == 0 || id == 1 || id == 2) return ""; 
    std::string text = vocab[id];
    size_t pos = 0; while ((pos = text.find("\xe2\x96\x81", pos)) != std::string::npos) { text.replace(pos, 3, " "); pos += 1; }
    if (text == "<0x0A>") return "\n"; return text;
}
std::string LlamaTokenizer::decode(const std::vector<int>& ids) { std::string s; for (int id : ids) s += decode(id); return s; }

LlamaLayerFP16::LlamaLayerFP16(MemPool& pool, int seq, int d, int hd, int nh, int nkv) : 
    dim(d), hidden_dim(hd), n_heads(nh), n_kv_heads(nkv), head_dim(d/nh), max_seq(seq),
    w_rms1(pool, {d}), w_rms2(pool, {d}),
    Wq(pool, {d, d}), Wk(pool, {nkv * head_dim, d}), Wv(pool, {nkv * head_dim, d}), Wo(pool, {d, d}),
    W1(pool, {hd, d}), W2(pool, {hd, d}), W3(pool, {d, hd}),
    k_cache(pool, {seq, nkv * head_dim}), v_cache(pool, {seq, nkv * head_dim}) {}

void LlamaLayerFP16::load(FILE* f) {
    read_into_half_tensor(f, w_rms1); read_into_half_tensor(f, Wq); read_into_half_tensor(f, Wk);
    read_into_half_tensor(f, Wv); read_into_half_tensor(f, Wo); read_into_half_tensor(f, w_rms2);
    read_into_half_tensor(f, W1); read_into_half_tensor(f, W2); read_into_half_tensor(f, W3);
}

// ==========================================
// PREFILL PHASE (Parallel cuBLAS GEMM)
// ==========================================
void LlamaLayerFP16::forward_prefill(MemPool& scratch, cublasHandle_t handle, HalfTensor& x, HalfTensor& out, int seq) {
    int hd = head_dim;
    HalfTensor xn(scratch, {seq, dim}), Q(scratch, {seq, dim}), K(scratch, {seq, n_kv_heads * hd}), V(scratch, {seq, n_kv_heads * hd});
    Tensor S(scratch, {n_heads, seq, seq}), P(scratch, {n_heads, seq, seq});
    HalfTensor A(scratch, {seq, dim}), AO(scratch, {seq, dim}), fn(scratch, {seq, dim});
    HalfTensor gate(scratch, {seq, hidden_dim}), up(scratch, {seq, hidden_dim}), swi(scratch, {seq, hidden_dim}), F(scratch, {seq, dim});

    k_half_rmsnorm(x.data, w_rms1.data, xn.data, seq, dim, 1e-5f);
    
    k_half_linear(handle, xn.data, Wq.data, Q.data, seq, dim, dim);
    k_half_linear(handle, xn.data, Wk.data, K.data, seq, dim, n_kv_heads * hd);
    k_half_linear(handle, xn.data, Wv.data, V.data, seq, dim, n_kv_heads * hd);

    k_half_llama_rope(Q.data, seq, n_heads, hd, 0); // Rotate all tokens from pos 0
    k_half_llama_rope(K.data, seq, n_kv_heads, hd, 0);

    k_half_copy_block_to_cache(K.data, k_cache.data, 0, seq, n_kv_heads * hd);
    k_half_copy_block_to_cache(V.data, v_cache.data, 0, seq, n_kv_heads * hd);

    k_half_mha_scores_fused_mask(Q.data, K.data, S.data, seq, n_heads, n_kv_heads, hd);
    k_row_softmax(S.data, P.data, n_heads * seq, seq);
    k_half_mha_weighted_sum(P.data, V.data, A.data, seq, n_heads, n_kv_heads, hd);

    k_half_linear(handle, A.data, Wo.data, AO.data, seq, dim, dim);
    k_half_fused_add_rmsnorm(x.data, AO.data, w_rms2.data, fn.data, seq, dim, 1e-5f);

    k_half_linear(handle, fn.data, W1.data, gate.data, seq, dim, hidden_dim);
    k_half_linear(handle, fn.data, W2.data, up.data, seq, dim, hidden_dim);
    k_half_swiglu(gate.data, up.data, swi.data, seq * hidden_dim);
    k_half_linear(handle, swi.data, W3.data, F.data, seq, hidden_dim, dim);

    k_half_add(x.data, F.data, out.data, seq * dim);
}

// ==========================================
// DECODE PHASE (Specialized GEMV)
// ==========================================
void LlamaLayerFP16::forward_decode(MemPool& scratch, HalfTensor& x, HalfTensor& out, int pos) {
    int hd = head_dim;
    HalfTensor xn(scratch, {1, dim}), Q(scratch, {1, dim}), K(scratch, {1, n_kv_heads * hd}), V(scratch, {1, n_kv_heads * hd});
    Tensor S(scratch, {n_heads, pos + 1}), P(scratch, {n_heads, pos + 1});
    HalfTensor A(scratch, {1, dim}), AO(scratch, {1, dim}), fn(scratch, {1, dim});
    HalfTensor gate(scratch, {1, hidden_dim}), up(scratch, {1, hidden_dim}), swi(scratch, {1, hidden_dim}), F(scratch, {1, dim});

    k_half_rmsnorm(x.data, w_rms1.data, xn.data, 1, dim, 1e-5f);
    
    // CUSTOM MEMORY-BOUND GEMV INSTEAD OF CUBLAS!
    k_half_gemv(xn.data, Wq.data, Q.data, dim, dim);
    k_half_gemv(xn.data, Wk.data, K.data, dim, n_kv_heads * hd);
    k_half_gemv(xn.data, Wv.data, V.data, dim, n_kv_heads * hd);

    k_half_llama_rope(Q.data, 1, n_heads, hd, pos);
    k_half_llama_rope(K.data, 1, n_kv_heads, hd, pos);

    k_half_copy_block_to_cache(K.data, k_cache.data, pos, 1, n_kv_heads * hd);
    k_half_copy_block_to_cache(V.data, v_cache.data, pos, 1, n_kv_heads * hd);

    k_half_mha_scores_one(Q.data, k_cache.data, S.data, pos, n_heads, n_kv_heads, hd);
    k_row_softmax(S.data, P.data, n_heads, pos + 1);
    k_half_mha_weighted_sum_one(P.data, v_cache.data, A.data, pos, n_heads, n_kv_heads, hd);

    k_half_gemv(A.data, Wo.data, AO.data, dim, dim);
    k_half_fused_add_rmsnorm(x.data, AO.data, w_rms2.data, fn.data, 1, dim, 1e-5f);

    k_half_gemv(fn.data, W1.data, gate.data, dim, hidden_dim);
    k_half_gemv(fn.data, W2.data, up.data, dim, hidden_dim);
    k_half_swiglu(gate.data, up.data, swi.data, hidden_dim);
    k_half_gemv(swi.data, W3.data, F.data, hidden_dim, dim);

    k_half_add(x.data, F.data, out.data, dim);
}

Llama2FP16::Llama2FP16(MemPool& pool) : 
    tok_embed(pool, {vocab, dim}), norm_w(pool, {dim}), lm_head(pool, {vocab, dim}) {
    cublasCreate(&handle);
    for (int i = 0; i < layers; i++) transformer.push_back(new LlamaLayerFP16(pool, max_seq, dim, hidden, heads, kv_heads));
    tokenizer.load("tokenizer.bin");
}
void Llama2FP16::load_weights(const char* path) {
    printf("Loading 2.2GB FP16 weights into Tensor Cores...\n");
    FILE* f = fopen(path, "rb");
    read_into_half_tensor(f, tok_embed);
    for (int i = 0; i < layers; i++) transformer[i]->load(f);
    read_into_half_tensor(f, norm_w); read_into_half_tensor(f, lm_head); fclose(f);
}

void Llama2FP16::prefill(MemPool& scratch, const std::vector<int>& prompt_ids) {
    scratch.reset();
    int seq = prompt_ids.size();
    int* d_ids = scratch.alloc<int>(seq + (seq % 4 == 0 ? 0 : 4 - (seq % 4))); // Align memory
    cudaMemcpy(d_ids, prompt_ids.data(), seq * sizeof(int), cudaMemcpyHostToDevice);
    
    HalfTensor x(scratch, {seq, dim}), tmp(scratch, {seq, dim});
    k_half_embedding_lookup(d_ids, tok_embed.data, x.data, seq, dim);
    
    for (int i = 0; i < layers; i++) {
        transformer[i]->forward_prefill(scratch, handle, x, tmp, seq);
        k_half_copy(tmp.data, x.data, seq * dim);
    }
}

int Llama2FP16::decode_next(MemPool& scratch, int pos, const GenerationConfig& cfg, const std::vector<int>& past_tokens) {
    scratch.reset();
    int last_token = past_tokens.back();
    int* tid_data = scratch.alloc<int>(4); 
    cudaMemcpy(tid_data, &last_token, sizeof(int), cudaMemcpyHostToDevice);
    
    HalfTensor x(scratch, {1, dim}), tmp(scratch, {1, dim});
    k_half_embedding_lookup(tid_data, tok_embed.data, x.data, 1, dim);
    
    for (int i = 0; i < layers; i++) {
        transformer[i]->forward_decode(scratch, x, tmp, pos);
        k_half_copy(tmp.data, x.data, dim);
    }
    
    HalfTensor final_norm(scratch, {1, dim}), logits_fp16(scratch, {1, vocab});
    k_half_rmsnorm(x.data, norm_w.data, final_norm.data, 1, dim, 1e-5f);
    k_half_gemv(final_norm.data, lm_head.data, logits_fp16.data, dim, vocab);
    
    Tensor logits_fp32(scratch, {1, vocab});
    k_half_to_float(logits_fp16.data, logits_fp32.data, vocab);

    if (cfg.repetition_penalty > 1.0f && past_tokens.size() > 0) {
        int* d_past = scratch.alloc<int>(past_tokens.size());
        cudaMemcpy(d_past, past_tokens.data(), past_tokens.size() * sizeof(int), cudaMemcpyHostToDevice);
        k_apply_repetition_penalty(logits_fp32.data, d_past, past_tokens.size(), cfg.repetition_penalty);
    }
    if (cfg.temperature != 1.0f && cfg.temperature > 0.0f) k_apply_temperature(logits_fp32.data, cfg.temperature, vocab);

    int* out_data = scratch.alloc<int>(4);
    if (cfg.temperature <= 0.0f) {
        k_argmax_row(logits_fp32.data, out_data, 1, vocab);
    } else {
        Tensor probs(scratch, {1, vocab});
        k_row_softmax(logits_fp32.data, probs.data, 1, vocab);
        k_sample_top_p(probs.data, out_data, cfg.top_p, (float)rand() / (float)RAND_MAX, vocab);
    }
    
    int h_out = 0; cudaMemcpy(&h_out, out_data, sizeof(int), cudaMemcpyDeviceToHost); return h_out;
}

void Llama2FP16::chat(MemPool& scratch, const std::vector<int>& prompt_ids, GenerationConfig cfg) {
    printf("\n[TinyLlama FP16]: ");
    std::vector<int> past_tokens;
    for(int id : prompt_ids) { printf("%s", tokenizer.decode(id).c_str()); fflush(stdout); past_tokens.push_back(id); }
    
    auto t0 = std::chrono::high_resolution_clock::now();
    
    // 1. MASSIVE PREFILL
    prefill(scratch, prompt_ids);
    
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    
    // 2. FAST DECODE
    int tokens_generated = 0;
    int pos = prompt_ids.size() - 1;

    for (int i = 0; i < cfg.max_new_tokens; i++) {
        int next_token = decode_next(scratch, pos++, cfg, past_tokens);
        if (next_token == 2 || next_token == 0) break;
        past_tokens.push_back(next_token);
        printf("%s", tokenizer.decode(next_token).c_str()); fflush(stdout);
        tokens_generated++;
    }
    printf("\n");

    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prefill_time = t1 - t0;
    std::chrono::duration<double> decode_time = t2 - t1;
    
    printf("\n========================================\n");
    printf(" PREFILL VS DECODE BENCHMARK\n");
    printf("========================================\n");
    printf(" Prompt Tokens    : %zu\n", prompt_ids.size());
    printf(" Prefill Time     : %.3f sec (%.2f tok/s)\n", prefill_time.count(), prompt_ids.size() / prefill_time.count());
    printf(" Generated Tokens : %d\n", tokens_generated);
    printf(" Decode Speed     : %.2f tok/s\n", tokens_generated / decode_time.count());
    printf("========================================\n");
}