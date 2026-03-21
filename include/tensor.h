#pragma once
#include <vector>
#include <cuda_runtime.h>
#include "cuda_check.h"
#include "mem_pool.h"

struct Tensor {
    float* data;
    std::vector<int> shape;
    size_t numel;
    Tensor(): data(nullptr), numel(0) {}
    Tensor(MemPool& pool, std::vector<int> s): shape(s) {
        numel = 1;
        for (int d : s) numel *= (size_t)d;
        data = pool.alloc<float>(numel);
    }
    void fill(float val) {
        std::vector<float> h(numel, val);
        CUDA_CHECK(cudaMemcpy(data, h.data(), numel * sizeof(float), cudaMemcpyHostToDevice));
    }
    void to_host(std::vector<float>& out) {
        out.resize(numel);
        CUDA_CHECK(cudaMemcpy(out.data(), data, numel * sizeof(float), cudaMemcpyDeviceToHost));
    }
    void from_host(const std::vector<float>& in) {
        CUDA_CHECK(cudaMemcpy(data, in.data(), in.size() * sizeof(float), cudaMemcpyHostToDevice));
    }
    int dim(int i) { return shape[i]; }
};