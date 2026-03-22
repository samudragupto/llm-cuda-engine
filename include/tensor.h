#pragma once
#include <vector>
#include <cuda_fp16.h>
#include <cstdint>
#include <cuda_runtime.h>
#include "mem_pool.h"


void k_fill(float* a, float val, int n);

struct Tensor {
    float* data; int numel; std::vector<int> shape;
    Tensor(MemPool& pool, std::vector<int> s) : shape(s) { 
        numel = 1; for (int d : s) numel *= d; data = pool.alloc<float>(numel); 
    }
    void from_host(const std::vector<float>& h_data) {
        cudaMemcpy(data, h_data.data(), numel * sizeof(float), cudaMemcpyHostToDevice);
    }
    void to_host(std::vector<float>& h_data) {
        h_data.resize(numel);
        cudaMemcpy(h_data.data(), data, numel * sizeof(float), cudaMemcpyDeviceToHost);
    }
    void fill(float val) {
        k_fill(data, val, numel);
    }
};

struct IntTensor {
    int* data; int numel; std::vector<int> shape;
    IntTensor(MemPool& pool, std::vector<int> s) : shape(s) { 
        numel = 1; for (int d : s) numel *= d; data = pool.alloc<int>(numel); 
    }
    void from_host(const std::vector<int>& h_data) {
        cudaMemcpy(data, h_data.data(), numel * sizeof(int), cudaMemcpyHostToDevice);
    }
    void to_host(std::vector<int>& h_data) {
        h_data.resize(numel);
        cudaMemcpy(h_data.data(), data, numel * sizeof(int), cudaMemcpyDeviceToHost);
    }
};

struct HalfTensor {
    half* data; int numel; std::vector<int> shape;
    HalfTensor(MemPool& pool, std::vector<int> s) : shape(s) { 
        numel = 1; for (int d : s) numel *= d; data = pool.alloc<half>(numel); 
    }
};

struct QuantizedTensor {
    int8_t* data; half* scales; int rows, cols;
    QuantizedTensor(MemPool& pool, int r, int c) : rows(r), cols(c) {
        data = pool.alloc<int8_t>(r * c);
        scales = pool.alloc<half>(r);
    }
};