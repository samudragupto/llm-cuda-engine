#include "memory_pool.h"
#include "model.h"

int main() {
    MemPool model_pool(2ULL * 1024 * 1024 * 1024);   
    MemPool scratch_pool(256ULL * 1024 * 1024);      

    Llama2MixedGraph model(model_pool);
    model.load_weights("model_mixed.bin");

    std::vector<int> prompt = {1, 450, 7483, 310, 3444, 338}; // "The capital of France is"
    
    GenerationConfig cfg;
    cfg.max_new_tokens = 50;
    cfg.repetition_penalty = 1.1f;

    model.chat(scratch_pool, prompt, cfg);
    return 0;
}