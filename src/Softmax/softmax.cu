#include <cuda_runtime.h>
#include <cfloat>
#include <cmath>

#define TILE 256

__device__ inline float atomicMaxFloat(float* address, float val) {
    unsigned int* addr_as_ui = reinterpret_cast<unsigned int*>(address);

    unsigned int old = *addr_as_ui;

    while (true) {
        float old_f = __uint_as_float(old);

        float max_f = fmaxf(old_f, val);
        unsigned int desired = __float_as_uint(max_f);

        if (desired == old) {
            return old_f;
        }
        unsigned int prior = atomicCAS(addr_as_ui, old, desired);
        if (prior == old) {
            return __uint_as_float(old);
        }
        old = prior;
    }
}

__global__ void max_kernel(const float* d_input, float* d_output, int N){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ float Mds[TILE];
    float v = (tid < N) ? d_input[tid] : -FLT_MAX;
    Mds[threadIdx.x] = v;
    __syncthreads();
    for(int j = TILE / 2; j > 0; j >>= 1){
        if(threadIdx.x + j < TILE){
            Mds[threadIdx.x] = max(Mds[threadIdx.x], Mds[threadIdx.x + j]);
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        atomicMaxFloat(d_output, Mds[0]);
    }
}

__global__ void exp_sum_kernel(const float* d_input, float* d_output, const float* d_m, int N){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ float Mds[TILE];
    float v = (tid < N) ? exp(d_input[tid] - *d_m) : 0;
    Mds[threadIdx.x] = v;
    __syncthreads();
    for(int j = TILE / 2; j > 0; j >>= 1){
        if(threadIdx.x + j < TILE){
            Mds[threadIdx.x] += Mds[threadIdx.x + j];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        atomicAdd(d_output, Mds[0]);
    }
}

__global__ void softmax_kernel(const float* d_input, float* d_output, const float* d_m, const float* d_s, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < N){
        d_output[tid] = (exp(d_input[tid] - *d_m)) / *d_s;
    }
}
