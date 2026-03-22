#include "llama2.h"
#include "kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <chrono>

void read_ht(FILE* f, HalfTensor& t) { size_t b = t.numel * 2; half* buf = (half*)malloc(b); fread(buf, 2, t.numel, f); cudaMemcpy(t.data, buf, b, cudaMemcpyHostToDevice); free(buf); }
void read_qt(FILE* f, QuantizedTensor& t) { size_t b1 = t.rows * t.cols; int8_t* buf1 = (int8_t*)malloc(b1); fread(buf1, 1, b1, f); cudaMemcpy(t.data, buf1, b1, cudaMemcpyHostToDevice); free(buf1); size_t b2 = t.rows * 2; half* buf2 = (half*)malloc(b2); fread(buf2, 2, t.rows, f); cudaMemcpy(t.scales, buf2, b2, cudaMemcpyHostToDevice); free(buf2); }

void LlamaTokenizer::load(const char* path) { FILE* f = fopen(path, "rb"); int vs; fread(&vs, 4, 1, f); vocab.resize(vs); for (int i=0; i<vs; i++) { int l; fread(&l, 4, 1, f); std::string s(l, '\0'); fread(&s[0], 1, l, f); vocab[i] = s; } fclose(f); }
std::string LlamaTokenizer::decode(int id) { if (id < 0 || id >= vocab.size() || id == 0 || id == 1 || id == 2) return ""; std::string t = vocab[id]; size_t p = 0; while ((p = t.find("\xe2\x96\x81", p)) != std::string::npos) { t.replace(p, 3, " "); p += 1; } if (t == "<0x0A>") return "\n"; return t; }
std::string LlamaTokenizer::decode(const std::vector<int>& ids) { std::string s; for (int id : ids) s += decode(id); return s; }

LlamaLayerMixed::LlamaLayerMixed(MemPool& pool, int seq, int d, int hd, int nh, int nkv) : dim(d), hidden_dim(hd), n_heads(nh), n_kv_heads(nkv), head_dim(d/nh), max_seq(seq), w_rms1(pool, {d}), w_rms2(pool, {d}), Wq(pool, {d, d}), Wk(pool, {nkv*head_dim, d}), Wv(pool, {nkv*head_dim, d}), Wo(pool, {d, d}), W1(pool, hd, d), W2(pool, hd, d), W3(pool, d, hd), k_cache(pool, {seq, nkv*head_dim}), v_cache(pool, {seq, nkv*head_dim}) {}

void LlamaLayerMixed::load(FILE* f) { read_ht(f, w_rms1); read_ht(f, Wq); read_ht(f, Wk); read_ht(f, Wv); read_ht(f, Wo); read_ht(f, w_rms2); read_qt(f, W1); read_qt(f, W2); read_qt(f, W3); }

void LlamaLayerMixed::forward_prefill(MemPool& scratch, cublasHandle_t handle, HalfTensor& x, HalfTensor& out, int seq) {
    int hd = head_dim; HalfTensor xn(scratch, {seq, dim}), Q(scratch, {seq, dim}), K(scratch, {seq, n_kv_heads * hd}), V(scratch, {seq, n_kv_heads * hd}), A(scratch, {seq, dim}), AO(scratch, {seq, dim}), fn(scratch, {seq, dim}), gate(scratch, {seq, hidden_dim}), up(scratch, {seq, hidden_dim}), swi(scratch, {seq, hidden_dim}), F(scratch, {seq, dim});
    k_half_rmsnorm(x.data, w_rms1.data, xn.data, seq, dim, 1e-5f);
    k_half_linear(handle, xn.data, Wq.data, Q.data, seq, dim, dim); k_half_linear(handle, xn.data, Wk.data, K.data, seq, dim, n_kv_heads*hd); k_half_linear(handle, xn.data, Wv.data, V.data, seq, dim, n_kv_heads*hd);
    k_half_llama_rope(Q.data, seq, n_heads, hd, 0); k_half_llama_rope(K.data, seq, n_kv_heads, hd, 0);
    k_half_copy_block_to_cache(K.data, k_cache.data, 0, seq, n_kv_heads * hd); k_half_copy_block_to_cache(V.data, v_cache.data, 0, seq, n_kv_heads * hd);
    k_flash_attention_prefill(Q.data, K.data, V.data, A.data, seq, n_heads, n_kv_heads, hd);
    k_half_linear(handle, A.data, Wo.data, AO.data, seq, dim, dim);
    k_half_fused_add_rmsnorm(x.data, AO.data, w_rms2.data, fn.data, seq, dim, 1e-5f);
    for(int i=0; i<seq; i++) { k_int8_gemv(fn.data + i*dim, W1.data, W1.scales, gate.data + i*hidden_dim, dim, hidden_dim); k_int8_gemv(fn.data + i*dim, W2.data, W2.scales, up.data + i*hidden_dim, dim, hidden_dim); }
    k_half_swiglu(gate.data, up.data, swi.data, seq * hidden_dim);
    for(int i=0; i<seq; i++) k_int8_gemv(swi.data + i*hidden_dim, W3.data, W3.scales, F.data + i*dim, hidden_dim, dim);
    k_half_add(x.data, F.data, out.data, seq * dim);
}

void LlamaLayerMixed::forward_decode(MemPool& scratch, HalfTensor& x, HalfTensor& out, int pos) {
    int hd = head_dim; HalfTensor xn(scratch, {1, dim}), Q(scratch, {1, dim}), K(scratch, {1, n_kv_heads * hd}), V(scratch, {1, n_kv_heads * hd}); Tensor S(scratch, {n_heads, 4096}), P(scratch, {n_heads, 4096}); HalfTensor A(scratch, {1, dim}), AO(scratch, {1, dim}), fn(scratch, {1, dim}), gate(scratch, {1, hidden_dim}), up(scratch, {1, hidden_dim}), swi(scratch, {1, hidden_dim}), F(scratch, {1, dim});
    k_half_rmsnorm(x.data, w_rms1.data, xn.data, 1, dim, 1e-5f);
    k_half_gemv(xn.data, Wq.data, Q.data, dim, dim); k_half_gemv(xn.data, Wk.data, K.data, dim, n_kv_heads * hd); k_half_gemv(xn.data, Wv.data, V.data, dim, n_kv_heads * hd);
    k_half_llama_rope(Q.data, 1, n_heads, hd, pos); k_half_llama_rope(K.data, 1, n_kv_heads, hd, pos);
    k_half_copy_block_to_cache(K.data, k_cache.data, pos, 1, n_kv_heads * hd); k_half_copy_block_to_cache(V.data, v_cache.data, pos, 1, n_kv_heads * hd);
    k_half_mha_scores_one(Q.data, k_cache.data, S.data, pos, n_heads, n_kv_heads, hd);
    k_row_softmax(S.data, P.data, n_heads, pos + 1);
    k_half_mha_weighted_sum_one(P.data, v_cache.data, A.data, pos, n_heads, n_kv_heads, hd);
    k_half_gemv(A.data, Wo.data, AO.data, dim, dim);
    k_half_fused_add_rmsnorm(x.data, AO.data, w_rms2.data, fn.data, 1, dim, 1e-5f);
    k_int8_gemv(fn.data, W1.data, W1.scales, gate.data, dim, hidden_dim); k_int8_gemv(fn.data, W2.data, W2.scales, up.data, dim, hidden_dim);
    k_half_swiglu(gate.data, up.data, swi.data, hidden_dim);
    k_int8_gemv(swi.data, W3.data, W3.scales, F.data, hidden_dim, dim);
    k_half_add(x.data, F.data, out.data, dim);
}

Llama2Mixed::Llama2Mixed(MemPool& pool) : tok_embed(pool, {vocab, dim}), norm_w(pool, {dim}), lm_head(pool, {vocab, dim}) {
    cublasCreate(&handle); for (int i=0; i<layers; i++) transformer.push_back(new LlamaLayerMixed(pool, max_seq, dim, hidden, heads, kv_heads)); tokenizer.load("tokenizer.bin");
}
void Llama2Mixed::load_weights(const char* path) { FILE* f = fopen(path, "rb"); read_ht(f, tok_embed); for (int i=0; i<layers; i++) transformer[i]->load(f); read_ht(f, norm_w); read_ht(f, lm_head); fclose(f); }

void Llama2Mixed::prefill(MemPool& scratch, const std::vector<int>& prompt_ids) {
    scratch.reset(); int seq = prompt_ids.size(); int* d_ids = scratch.alloc<int>(seq + (seq % 4 == 0 ? 0 : 4 - (seq % 4))); 
    cudaMemcpy(d_ids, prompt_ids.data(), seq * sizeof(int), cudaMemcpyHostToDevice);
    HalfTensor x(scratch, {seq, dim}), tmp(scratch, {seq, dim});
    k_half_embedding_lookup(d_ids, tok_embed.data, x.data, seq, dim);
    for (int i=0; i<layers; i++) { transformer[i]->forward_prefill(scratch, handle, x, tmp, seq); k_half_copy(tmp.data, x.data, seq * dim); }
}

int Llama2Mixed::decode_next(MemPool& scratch, int pos, const GenerationConfig& cfg, const std::vector<int>& past_tokens) {
    scratch.reset(); int token = past_tokens.back(); int* d_token = scratch.alloc<int>(4); cudaMemcpy(d_token, &token, sizeof(int), cudaMemcpyHostToDevice);
    HalfTensor x(scratch, {1, dim}), tmp(scratch, {1, dim});
    k_half_embedding_lookup(d_token, tok_embed.data, x.data, 1, dim);
    for (int i=0; i<layers; i++) { transformer[i]->forward_decode(scratch, x, tmp, pos); k_half_copy(tmp.data, x.data, dim); }
    HalfTensor fn(scratch, {1, dim}), logits16(scratch, {1, vocab});
    k_half_rmsnorm(x.data, norm_w.data, fn.data, 1, dim, 1e-5f);
    k_half_gemv(fn.data, lm_head.data, logits16.data, dim, vocab);
    Tensor logits32(scratch, {1, vocab}); k_half_to_float(logits16.data, logits32.data, vocab);
    if (cfg.repetition_penalty > 1.0f && past_tokens.size() > 0) {
        int* d_past = scratch.alloc<int>(past_tokens.size()); cudaMemcpy(d_past, past_tokens.data(), past_tokens.size() * sizeof(int), cudaMemcpyHostToDevice);
        k_apply_repetition_penalty(logits32.data, d_past, past_tokens.size(), cfg.repetition_penalty);
    }
    if (cfg.temperature != 1.0f && cfg.temperature > 0.0f) k_apply_temperature(logits32.data, cfg.temperature, vocab);
    int* d_out = scratch.alloc<int>(4);
    if (cfg.temperature <= 0.0f) k_argmax_row(logits32.data, d_out, 1, vocab);
    else { Tensor probs(scratch, {1, vocab}); k_row_softmax(logits32.data, probs.data, 1, vocab); k_sample_top_p(probs.data, d_out, cfg.top_p, (float)rand()/RAND_MAX, vocab); }
    int out_tok; cudaMemcpy(&out_tok, d_out, sizeof(int), cudaMemcpyDeviceToHost); return out_tok;
}

void Llama2Mixed::chat(MemPool& scratch, const std::vector<int>& prompt_ids, GenerationConfig cfg) {
    printf("\n[TinyLlama INT8-Mixed]: "); std::vector<int> past;
    for(int id : prompt_ids) { printf("%s", tokenizer.decode(id).c_str()); fflush(stdout); past.push_back(id); }
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<int> prefill_ids(prompt_ids.begin(), prompt_ids.end() - 1);
    if (prefill_ids.size() > 0) prefill(scratch, prefill_ids); 
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    int pos = prefill_ids.size(), tokens_generated = 0;
    for (int i=0; i<cfg.max_new_tokens; i++) {
        int next_token = decode_next(scratch, pos, cfg, past);
        if (next_token == 2 || next_token == 0) break; 
        past.push_back(next_token);
        printf("%s", tokenizer.decode(next_token).c_str()); fflush(stdout); pos++; tokens_generated++;
    }
    printf("\n");
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t2 - t1;
    printf("\n[Decode Speed: %.2f tok/s]\n", tokens_generated / dt.count());
}