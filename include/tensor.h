#pragma once
#include <vector>
#include <cuda_fp16.h>
#include <cstdint>
#include "memory_pool.h"

struct Tensor {
    float* data; int numel; std::vector<int> shape;
    Tensor(MemPool& pool, std::vector<int> s) : shape(s) { numel = 1; for (int d : s) numel *= d; data = pool.alloc<float>(numel); }
};
struct HalfTensor {
    half* data; int numel; std::vector<int> shape;
    HalfTensor(MemPool& pool, std::vector<int> s) : shape(s) { numel = 1; for (int d : s) numel *= d; data = pool.alloc<half>(numel); }
};
struct QuantizedTensor {
    int8_t* data; half* scales; int rows, cols;
    QuantizedTensor(MemPool& pool, int r, int c) : rows(r), cols(c) { data = pool.alloc<int8_t>(r * c); scales = pool.alloc<half>(r); }
};