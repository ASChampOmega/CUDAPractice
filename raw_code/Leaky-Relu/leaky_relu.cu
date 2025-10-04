#include <cuda_runtime.h>

__global__ void leaky_relu_kernel(const float* d_input, float* d_output, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < N){
        float v = d_input[tid];
        if(v > 0){
            d_output[tid] = v;
        }
        else{
            d_output[tid] = 0.01 * v;
        }
    }
}