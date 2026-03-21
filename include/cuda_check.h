#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#define CUDA_CHECK(x) do { cudaError_t cuda_err__ = (x); if(cuda_err__ != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(cuda_err__)); exit(1); } } while(0)