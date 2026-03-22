#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "cuda_check.h"

struct MemPool {
    char* base;
    size_t offset;
    size_t capacity;
    size_t peak_offset;
    int alloc_count;

    MemPool(size_t cap) : offset(0), capacity(cap), peak_offset(0), alloc_count(0) {
        CUDA_CHECK(cudaMalloc(&base, cap));
    }
    
    ~MemPool() { 
        cudaFree(base); 
    }
    
    template<typename T>
    T* alloc(size_t n) {
        size_t bytes = n * sizeof(T);
        
        size_t aligned = (bytes + 255) & ~(size_t)255; 
        
        if (offset + aligned > capacity) {
            fprintf(stderr, "\n[FATAL] Pool OOM: Need %zu bytes, have %zu free out of %zu capacity.\n", 
                    aligned, capacity - offset, capacity);
            exit(1);
        }
        
        T* ptr = (T*)(base + offset);
        offset += aligned;
        alloc_count++;
        
        if (offset > peak_offset) peak_offset = offset;
        return ptr;
    }
    
    void reset() { 
        offset = 0; 
        alloc_count = 0;
    }
    
    void print_stats(const char* name) {
        printf("MemPool [%s]: Peak %.2f MB / %.2f MB (%.1f%%) | Active Allocs: %d\n", 
               name, 
               peak_offset / (1024.0 * 1024.0), 
               capacity / (1024.0 * 1024.0), 
               (double)peak_offset / capacity * 100.0,
               alloc_count);
    }
};