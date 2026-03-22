#include "mem_pool.h"
#include "llama2.h"
#include <cstdio>

int main() {
    printf("[DEBUG] Program started.\n"); fflush(stdout);

    printf("[DEBUG] Allocating 2GB GPU Memory Pool...\n"); fflush(stdout);
    MemPool model_pool(2ULL * 1024 * 1024 * 1024);   
    MemPool scratch_pool(256ULL * 1024 * 1024);      

    printf("[DEBUG] Memory allocated. Initializing model structure...\n"); fflush(stdout);
    Llama2MixedGraph model(model_pool);

    printf("[DEBUG] Model structure initialized. Loading weights...\n"); fflush(stdout);
    model.load_weights("model_mixed.bin");

    printf("[DEBUG] Weights loaded. Preparing generation...\n"); fflush(stdout);
    std::vector<int> prompt = {1, 450, 7483, 310, 3444, 338}; 
    
    GenerationConfig cfg;
    cfg.max_new_tokens = 50;

    printf("[DEBUG] Launching Chat Engine...\n"); fflush(stdout);
    model.chat(scratch_pool, prompt, cfg);

    printf("[DEBUG] Finished.\n"); fflush(stdout);
    return 0;
}