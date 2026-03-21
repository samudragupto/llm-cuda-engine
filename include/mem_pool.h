#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "cuda_check.h"
struct MemPool {
    char* base;
    size_t offset, capacity;
    MemPool(size_t cap) : offset(0), capacity(cap) {
        CUDA_CHECK(cudaMalloc(&base, cap));
    }
    ~MemPool() { cudaFree(base); }
    template<typename T>
    T* alloc(size_t n) {
        size_t bytes = n * sizeof(T);
        size_t aligned = (bytes + 127) & ~(size_t)127;
        if (offset + aligned > capacity) {
            fprintf(stderr, "Pool OOM: need %zu, have %zu\n", offset + aligned, capacity);
            exit(1);
        }
        T* ptr = (T*)(base + offset);
        offset += aligned;
        return ptr;
    }
    void reset() { offset = 0; }
    size_t used() { return offset; }
    size_t remaining() { return capacity - offset; }
};