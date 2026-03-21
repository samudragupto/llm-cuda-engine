#pragma once
#include <vector>
#include <cuda_fp16.h>
#include "mem_pool.h"
#include "cuda_check.h"

struct HalfTensor {
    half* data;
    std::vector<int> shape;
    size_t numel;
    HalfTensor(): data(nullptr), numel(0) {}
    HalfTensor(MemPool& pool, std::vector<int> s): shape(s) {
        numel = 1;
        for (int d : s) numel *= (size_t)d;
        data = pool.alloc<half>(numel);
    }
};