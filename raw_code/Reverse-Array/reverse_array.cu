#include <cuda_runtime.h>

__global__ void reverse_array(float* d_input, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < N / 2){
        int t = d_input[tid];
        d_input[tid] = d_input[N - tid - 1];
        d_input[N - tid - 1] = t;
    }
}