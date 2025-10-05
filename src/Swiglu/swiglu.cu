#include <cuda_runtime.h>
#include <cmath>

__global__ void swiglu_kernel(const float* d_input, float* d_output, int halfN) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < halfN){
        float x1 = d_input[tid];
        float x2 = d_input[halfN + tid];
        d_output[tid] = x2 * x1 / (1 + exp(-x1));
    }
}