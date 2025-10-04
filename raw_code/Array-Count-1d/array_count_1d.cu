#include <cuda_runtime.h>

__global__ void count_equal_kernel(const int* d_input, int* d_output, int N, int K) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < N && d_input[tid] == K){
        atomicAdd(d_output, 1);
    }
}