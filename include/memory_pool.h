#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

struct MemPool {
    char* base; size_t offset; size_t capacity;
    MemPool(size_t cap) : offset(0), capacity(cap) {
        if (cudaMalloc(&base, cap) != cudaSuccess) { printf("[FATAL] OOM\n"); exit(1); }
    }
    ~MemPool() { cudaFree(base); }
    template<typename T> T* alloc(size_t n) {
        size_t bytes = n * sizeof(T);
        size_t aligned = (bytes + 255) & ~(size_t)255; 
        if (offset + aligned > capacity) { printf("[FATAL] Pool OOM\n"); exit(1); }
        T* ptr = (T*)(base + offset);
        offset += aligned;
        return ptr;
    }
    void reset() { offset = 0; }
};