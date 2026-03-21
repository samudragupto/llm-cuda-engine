#include <cstdio>
#include <vector>
#include "mem_pool.h"
#include "llama2.h"

int main() {
    printf("========================================\n");
    printf(" TINYLLAMA FP16 TENSOR CORE ENGINE      \n");
    printf("========================================\n");
    
    // We only need ~2.5GB now instead of 5.5GB!
    MemPool model_pool(2800ULL * 1024 * 1024); 
    MemPool scratch_pool(256ULL * 1024 * 1024);

    Llama2FP16 llama(model_pool);
    llama.load_weights("model_fp16.bin");
    
    model_pool.print_stats("Llama FP16 Memory footprint");

    std::vector<int> prompt = {1, 450, 7483, 310, 3444, 338};
    llama.chat(scratch_pool, prompt, 30);

    return 0;
}