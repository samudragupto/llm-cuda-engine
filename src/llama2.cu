#include "llama2.h"
#include "kernels.cuh"
#include "cuda_check.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>

HalfTensor::HalfTensor(MemPool& pool, std::vector<int> s) : shape(s) {
    numel = 1; for (int d : s) numel *= d;
    data = pool.alloc<half>(numel);
}

void read_into_half_tensor(FILE* f, HalfTensor& t) {
    size_t bytes = t.numel * sizeof(half);
    half* host_buf = (half*)malloc(bytes);
    fread(host_buf, sizeof(half), t.numel, f);
    cudaMemcpy(t.data, host_buf, bytes, cudaMemcpyHostToDevice);
    free(host_buf);
}

void LlamaTokenizer::load(const char* path) {
    FILE* f = fopen(path, "rb");
    int vocab_size; fread(&vocab_size, sizeof(int), 1, f);
    vocab.resize(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        int len; fread(&len, sizeof(int), 1, f);
        std::string s(len, '\0'); fread(&s[0], 1, len, f);
        vocab[i] = s;
    }
    fclose(f);
}

std::string LlamaTokenizer::decode(int id) {
    if (id < 0 || id >= vocab.size() || id == 0 || id == 1 || id == 2) return ""; 
    std::string text = vocab[id];
    size_t pos = 0;
    while ((pos = text.find("\xe2\x96\x81", pos)) != std::string::npos) { text.replace(pos, 3, " "); pos += 1; }
    if (text == "<0x0A>") return "\n";
    return text;
}
std::string LlamaTokenizer::decode(const std::vector<int>& ids) {
    std::string s; for (int id : ids) s += decode(id); return s;
}

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

void LlamaLayerFP16::forward_one(MemPool& scratch, cublasHandle_t handle, HalfTensor& x, HalfTensor& out, int pos) {
    int hd = head_dim;
    HalfTensor xn(scratch, {1, dim}), Q(scratch, {1, dim}), K(scratch, {1, n_kv_heads * hd}), V(scratch, {1, n_kv_heads * hd});
    Tensor S(scratch, {n_heads, pos + 1}), P(scratch, {n_heads, pos + 1});
    HalfTensor A(scratch, {1, dim}), AO(scratch, {1, dim}), fn(scratch, {1, dim});
    HalfTensor gate(scratch, {1, hidden_dim}), up(scratch, {1, hidden_dim}), swi(scratch, {1, hidden_dim}), F(scratch, {1, dim});

    k_half_rmsnorm(x.data, w_rms1.data, xn.data, 1, dim, 1e-5f);
    
    k_half_linear(handle, xn.data, Wq.data, Q.data, dim, dim);
    k_half_linear(handle, xn.data, Wk.data, K.data, dim, n_kv_heads * hd);
    k_half_linear(handle, xn.data, Wv.data, V.data, dim, n_kv_heads * hd);

    k_half_llama_rope(Q.data, 1, n_heads, hd, pos);
    k_half_llama_rope(K.data, 1, n_kv_heads, hd, pos);

    k_half_copy_row_to_cache(K.data, k_cache.data, pos, n_kv_heads * hd);
    k_half_copy_row_to_cache(V.data, v_cache.data, pos, n_kv_heads * hd);

    k_half_mha_scores_one(Q.data, k_cache.data, S.data, pos, n_heads, n_kv_heads, hd);
    k_row_softmax(S.data, P.data, n_heads, pos + 1);
    k_half_mha_weighted_sum_one(P.data, v_cache.data, A.data, pos, n_heads, n_kv_heads, hd);

    k_half_linear(handle, A.data, Wo.data, AO.data, dim, dim);
    
    k_half_fused_add_rmsnorm(x.data, AO.data, w_rms2.data, fn.data, 1, dim, 1e-5f);

    k_half_linear(handle, fn.data, W1.data, gate.data, dim, hidden_dim);
    k_half_linear(handle, fn.data, W2.data, up.data, dim, hidden_dim);
    k_half_swiglu(gate.data, up.data, swi.data, hidden_dim);
    k_half_linear(handle, swi.data, W3.data, F.data, hidden_dim, dim);

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
    if (!f) { printf("Error: Could not open %s\n", path); exit(1); }
    read_into_half_tensor(f, tok_embed);
    for (int i = 0; i < layers; i++) transformer[i]->load(f);
    read_into_half_tensor(f, norm_w);
    read_into_half_tensor(f, lm_head);
    fclose(f);
}

int Llama2FP16::generate_next(MemPool& scratch, int token_id, int pos) {
    scratch.reset();
    int* tid_data = scratch.alloc<int>(4); 
    cudaMemcpy(tid_data, &token_id, sizeof(int), cudaMemcpyHostToDevice);
    
    HalfTensor x(scratch, {1, dim}), tmp(scratch, {1, dim});
    k_half_embedding_lookup(tid_data, tok_embed.data, x.data, 1, dim);
    
    for (int i = 0; i < layers; i++) {
        transformer[i]->forward_one(scratch, handle, x, tmp, pos);
        k_half_copy(tmp.data, x.data, dim);
    }
    
    HalfTensor final_norm(scratch, {1, dim}), logits(scratch, {1, vocab});
    k_half_rmsnorm(x.data, norm_w.data, final_norm.data, 1, dim, 1e-5f);
    
    int* out_data = scratch.alloc<int>(4);
    k_half_linear(handle, final_norm.data, lm_head.data, logits.data, dim, vocab);
    k_half_argmax_row(logits.data, out_data, 1, vocab);
    
    int h_out = 0;
    cudaMemcpy(&h_out, out_data, sizeof(int), cudaMemcpyDeviceToHost);
    return h_out;
}

void Llama2FP16::chat(MemPool& scratch, const std::vector<int>& prompt_ids, int max_tokens) {
    printf("\n[TinyLlama FP16]: ");
    int pos = 0;
    
    // Print Prompt
    for(int id : prompt_ids) { 
        printf("%s", tokenizer.decode(id).c_str()); 
        fflush(stdout); 
    }
    
    // Prefill Phase
    int next_token = prompt_ids[0];
    for (int i = 0; i < prompt_ids.size() - 1; i++) {
        next_token = generate_next(scratch, prompt_ids[i], pos++);
    }
    next_token = prompt_ids.back();

    // Start Timer!
    cudaDeviceSynchronize();
    auto start_time = std::chrono::high_resolution_clock::now();
    int tokens_generated = 0;

    // Decode Phase
    for (int i = 0; i < max_tokens; i++) {
        next_token = generate_next(scratch, next_token, pos++);
        if (next_token == 2 || next_token == 0) break;
        printf("%s", tokenizer.decode(next_token).c_str()); 
        fflush(stdout);
        tokens_generated++;
    }
    printf("\n");

    // Stop Timer!
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    
    printf("\n========================================\n");
    printf(" PERFORMANCE BENCHMARK\n");
    printf("========================================\n");
    printf(" Tokens Generated : %d\n", tokens_generated);
    printf(" Elapsed Time     : %.3f seconds\n", elapsed.count());
    printf(" Speed            : %.2f Tokens / Second\n", tokens_generated / elapsed.count());
    printf("========================================\n");
}