#include <cuda_runtime.h>
#include <cmath>

__global__ void silu_kernel(const float* d_input, float* d_output, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < N){
        float v = d_input[tid];
        d_output[tid] = v / (1 + exp(-v));
    }
}
