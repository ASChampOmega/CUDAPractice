#include <cuda_runtime.h>

__global__ void relu_kernel(const float* d_input, float* d_output, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < N && d_input[tid] >= 0){
        d_output[tid] = d_input[tid];
    }
}