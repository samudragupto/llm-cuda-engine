#include <cstdio>
#include <vector>
#include "weight_loader.h"
#include "mem_pool.h"

void test_safetensors(const char* path, MemPool& pool) {
    SafetensorsLoader loader;
    if (!loader.load(path)) {
        printf("Could not open '%s' - skipping weight loader test\n", path);
        return;
    }
    printf("\n=== Safetensors: %s ===\n", path);
    printf("Header size: %llu bytes | Tensors: %zu\n\n",
           (unsigned long long)loader.header_size, loader.entries.size());
    loader.list_tensors();
    if (!loader.entries.empty()) {
        std::string name = loader.entries.begin()->first;
        printf("\nLoading '%s' to GPU...\n", name.c_str());
        Tensor t = loader.load_tensor(name, pool);
        std::vector<float> h; t.to_host(h);
        printf("First 10 values: ");
        int show = (int)h.size() < 10 ? (int)h.size() : 10;
        for (int i = 0; i < show; i++) printf("%.4f ", h[i]);
        printf("\n");
    }
}