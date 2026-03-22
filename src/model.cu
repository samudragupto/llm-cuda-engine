#include "model.h"
#include "kernels.cuh"
#include <cstdio>
#include <cstdlib>
#include <chrono>

void read_ht(FILE* f, HalfTensor& t) { size_t b = t.numel * 2; half* buf = (half*)malloc(b); fread(buf, 2, t.numel, f); cudaMemcpy(t.data, buf, b, cudaMemcpyHostToDevice); free(buf); }
void read_qt(FILE* f, QuantizedTensor& t) { size_t b1 = t.rows * t.cols; int8_t* buf1 = (int8_t*)malloc(b1); fread(buf1, 1, b1, f); cudaMemcpy(t.data, buf1, b1, cudaMemcpyHostToDevice); free(buf1); size_t b2 = t.rows * 2; half* buf2 = (half*)malloc(b2); fread(buf2, 2, t.rows, f); cudaMemcpy(t.scales, buf2, b2, cudaMemcpyHostToDevice); free(buf2); }

LlamaLayerMixed::LlamaLayerMixed(MemPool& pool, int seq, int d, int hd, int nh, int nkv) : dim(d), hidden_dim(hd), n_heads(nh), n_kv_heads(nkv), head_dim(d/nh), max_seq(seq), w_rms1(pool, {d}), w_rms2(pool, {d}), Wq(pool, {d, d}), Wk(pool, {nkv*head_dim, d}), Wv(pool, {nkv*head_dim, d}), Wo(pool, {d, d}), W1(pool, hd, d), W2(pool, hd, d), W3(pool, d, hd), k_cache(pool, {seq, nkv*head_dim}), v_cache(pool, {seq, nkv*head_dim}) {}

void LlamaLayerMixed::load(FILE* f) { read_ht(f, w_rms1); read_ht(f, Wq); read_ht(f, Wk); read_ht(f, Wv); read_ht(f, Wo); read_ht(f, w_rms2); read_qt(f, W1); read_qt(f, W2); read_qt(f, W3); }

void LlamaLayerMixed::forward_prefill(MemPool& scratch, cublasHandle_t handle, HalfTensor& x, int seq) {
    int hd = head_dim; HalfTensor xn(scratch, {seq, dim}), Q(scratch, {seq, dim}), K(scratch, {seq, n_kv_heads * hd}), V(scratch, {seq, n_kv_heads * hd}), A(scratch, {seq, dim}), AO(scratch, {seq, dim}), fn(scratch, {seq, dim}), gate(scratch, {seq, hidden_dim}), up(scratch, {seq, hidden_dim}), swi(scratch, {seq, hidden_dim}), F(scratch, {seq, dim});
    k_half_rmsnorm(x.data, w_rms1.data, xn.data, seq, dim, 1e-5f);
    k_half_linear(handle, xn.data, Wq.data, Q.data, seq, dim, dim);
    k_half_linear(handle, xn.data, Wk.data, K.data, seq, dim, n_kv_heads*hd);
    k_half_linear(handle, xn.data, Wv.data, V.data, seq, dim, n_kv_heads*hd);
    k_half_llama_rope(Q.data, seq, n_heads, hd, 0); k_half_llama_rope(K.data, seq, n_kv_heads, hd, 0);
    k_half_copy_block_to_cache(K.data, k_cache.data, 0, seq, n_kv_heads * hd); k_half_copy_block_to_cache(V.data, v_cache.data, 0, seq, n_kv_heads * hd);
    k_flash_attention_prefill(Q.data, K.data, V.data, A.data, seq, n_heads, n_kv_heads, hd);
    k_half_linear(handle, A.data, Wo.data, AO.data, seq, dim, dim);
    k_half_fused_add_rmsnorm(x.data, AO.data, w_rms2.data, fn.data, seq, dim, 1e-5f);
    for(int i=0; i<seq; i++) { k_int8_gemv(fn.data + i*dim, W1.data, W1.scales, gate.data + i*hidden_dim, dim, hidden_dim); k_int8_gemv(fn.data + i*dim, W2.data, W2.scales, up.data + i*hidden_dim, dim, hidden_dim); }
    k_half_swiglu(gate.data, up.data, swi.data, seq * hidden_dim);
    for(int i=0; i<seq; i++) k_int8_gemv(swi.data + i*hidden_dim, W3.data, W3.scales, F.data + i*dim, hidden_dim, dim);
    k_half_add(x.data, F.data, x.data, seq * dim);
}

void LlamaLayerMixed::forward_decode_graph(MemPool& scratch, HalfTensor& x, int* d_pos) {
    int hd = head_dim; HalfTensor xn(scratch, {1, dim}), Q(scratch, {1, dim}), K(scratch, {1, n_kv_heads * hd}), V(scratch, {1, n_kv_heads * hd}); Tensor S(scratch, {n_heads, 4096}), P(scratch, {n_heads, 4096}); HalfTensor A(scratch, {1, dim}), AO(scratch, {1, dim}), fn(scratch, {1, dim}), gate(scratch, {1, hidden_dim}), up(scratch, {1, hidden_dim}), swi(scratch, {1, hidden_dim}), F(scratch, {1, dim});
    k_half_rmsnorm(x.data, w_rms1.data, xn.data, 1, dim, 1e-5f);
    k_half_gemv(xn.data, Wq.data, Q.data, dim, dim); k_half_gemv(xn.data, Wk.data, K.data, dim, n_kv_heads * hd); k_half_gemv(xn.data, Wv.data, V.data, dim, n_kv_heads * hd);
    k_half_llama_rope_graph(Q.data, n_heads, hd, d_pos); k_half_llama_rope_graph(K.data, n_kv_heads, hd, d_pos);
    k_half_copy_to_cache_graph(K.data, k_cache.data, d_pos, n_kv_heads * hd); k_half_copy_to_cache_graph(V.data, v_cache.data, d_pos, n_kv_heads * hd);
    k_half_mha_scores_one(Q.data, k_cache.data, S.data, d_pos, n_heads, n_kv_heads, hd);
    k_row_softmax_graph(S.data, P.data, n_heads, d_pos);
    k_half_mha_weighted_sum_one(P.data, v_cache.data, A.data, d_pos, n_heads, n_kv_heads, hd);
    k_half_gemv(A.data, Wo.data, AO.data, dim, dim);
    k_half_fused_add_rmsnorm(x.data, AO.data, w_rms2.data, fn.data, 1, dim, 1e-5f);
    k_int8_gemv(fn.data, W1.data, W1.scales, gate.data, dim, hidden_dim); k_int8_gemv(fn.data, W2.data, W2.scales, up.data, dim, hidden_dim);
    k_half_swiglu(gate.data, up.data, swi.data, hidden_dim);
    k_int8_gemv(swi.data, W3.data, W3.scales, F.data, hidden_dim, dim);
    k_half_add(x.data, F.data, x.data, dim);
}

Llama2MixedGraph::Llama2MixedGraph(MemPool& pool) : tok_embed(pool, {vocab, dim}), norm_w(pool, {dim}), lm_head(pool, {vocab, dim}) {
    cublasCreate(&handle); d_pos = pool.alloc<int>(1); d_token = pool.alloc<int>(1); d_out = pool.alloc<int>(1);
    for (int i=0; i<layers; i++) transformer.push_back(new LlamaLayerMixed(pool, max_seq, dim, hidden, heads, kv_heads)); 
    tokenizer.load("tokenizer.bin");
}

void Llama2MixedGraph::load_weights(const char* path) { 
    FILE* f = fopen(path, "rb"); read_ht(f, tok_embed); for (int i=0; i<layers; i++) transformer[i]->load(f); read_ht(f, norm_w); read_ht(f, lm_head); fclose(f); 
}

void Llama2MixedGraph::prefill(MemPool& scratch, const std::vector<int>& prompt_ids) {
    scratch.reset(); int seq = prompt_ids.size(); int* d_ids = scratch.alloc<int>(seq + (seq % 4 == 0 ? 0 : 4 - (seq % 4))); 
    cudaMemcpyAsync(d_ids, prompt_ids.data(), seq * sizeof(int), cudaMemcpyHostToDevice, 0); 
    HalfTensor x(scratch, {seq, dim}); k_half_embedding_lookup(d_ids, tok_embed.data, x.data, seq, dim);
    for (int i=0; i<layers; i++) transformer[i]->forward_prefill(scratch, handle, x, seq);
    cudaStreamSynchronize(0);
}

void Llama2MixedGraph::capture_graph(MemPool& scratch) {
    scratch.reset(); HalfTensor x(scratch, {1, dim});
    cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal);
    k_half_embedding_lookup(d_token, tok_embed.data, x.data, 1, dim);
    for (int i=0; i<layers; i++) transformer[i]->forward_decode_graph(scratch, x, d_pos); 
    HalfTensor fn(scratch, {1, dim}), logits16(scratch, {1, vocab});
    k_half_rmsnorm(x.data, norm_w.data, fn.data, 1, dim, 1e-5f);
    k_half_gemv(fn.data, lm_head.data, logits16.data, dim, vocab);
    Tensor logits32(scratch, {1, vocab});
    k_half_to_float(logits16.data, logits32.data, vocab);
    cudaStreamEndCapture(0, &graph); 
    if(cudaGraphInstantiate(&graph_exec, graph, NULL, NULL, 0) != cudaSuccess) graph_exec = nullptr; 
}

void Llama2MixedGraph::chat(MemPool& scratch, const std::vector<int>& prompt_ids, GenerationConfig cfg) {
    std::vector<int> prefill_ids(prompt_ids.begin(), prompt_ids.end() - 1);
    if (prefill_ids.size() > 0) prefill(scratch, prefill_ids); 
    capture_graph(scratch);
    
    printf("\n[TinyLlama INT8 + CUDA Graphs]: "); fflush(stdout);
    int pos = prefill_ids.size(); std::vector<int> past;
    for(int id : prompt_ids) { printf("%s", tokenizer.decode(id).c_str()); fflush(stdout); past.push_back(id); }
    
    auto t1 = std::chrono::high_resolution_clock::now();
    int token = prompt_ids.back(); int tokens_generated = 0;
    std::vector<float> h_logits(vocab);

    for (int i=0; i<cfg.max_new_tokens; i++) {
        cudaMemcpy(d_pos, &pos, sizeof(int), cudaMemcpyHostToDevice); 
        cudaMemcpy(d_token, &token, sizeof(int), cudaMemcpyHostToDevice);
        
        if (graph_exec != nullptr) cudaGraphLaunch(graph_exec, 0);
        else {
            scratch.reset(); HalfTensor dx(scratch, {1, dim}); k_half_embedding_lookup(d_token, tok_embed.data, dx.data, 1, dim);
            for (int l=0; l<layers; l++) transformer[l]->forward_decode_graph(scratch, dx, d_pos); 
            HalfTensor dfn(scratch, {1, dim}), dlogits16(scratch, {1, vocab});
            k_half_rmsnorm(dx.data, norm_w.data, dfn.data, 1, dim, 1e-5f);
            k_half_gemv(dfn.data, lm_head.data, dlogits16.data, dim, vocab);
            Tensor logits32(scratch, {1, vocab}); k_half_to_float(dlogits16.data, logits32.data, vocab);
        }
        
        float* d_logits32 = (float*)(scratch.base + scratch.offset - (vocab * sizeof(float)));
        cudaMemcpy(h_logits.data(), d_logits32, vocab * sizeof(float), cudaMemcpyDeviceToHost);
        for (int p_id : past) h_logits[p_id] = h_logits[p_id] > 0 ? h_logits[p_id] / cfg.repetition_penalty : h_logits[p_id] * cfg.repetition_penalty;
        
        int best_id = 0; float best_val = -1e9f;
        for (int v = 0; v < vocab; v++) if (h_logits[v] > best_val) { best_val = h_logits[v]; best_id = v; }
        token = best_id;
        
        if (token <= 2) break; 
        past.push_back(token);
        printf("%s", tokenizer.decode(token).c_str()); fflush(stdout); 
        pos++; tokens_generated++;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> dt = t2 - t1;
    if (tokens_generated > 0) printf("\n\n[Decode Speed: %.2f tok/s]\n", tokens_generated / dt.count());
}