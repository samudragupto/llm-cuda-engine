#include <cstdio>
#include <vector>
#include "mem_pool.h"
#include "llama2.h"

int main() {
    MemPool model_pool(1600ULL * 1024 * 1024); 
    MemPool scratch_pool(128ULL * 1024 * 1024);
    Llama2Mixed llama(model_pool);
    llama.load_weights("model_mixed.bin");
    model_pool.print_stats("Mixed Precision Footprint");
    
    // Test a long prompt to verify stability
    std::vector<int> prompt = {1, 450, 7483, 310, 3444, 338}; // "The capital of France is"
    
    // TEMP = 0.0 (GREEDY) TO PROVE CORRECTNESS
    GenerationConfig config; config.max_new_tokens = 50; config.temperature = 0.0f;
    llama.chat(scratch_pool, prompt, config);
    return 0;
}