#include <cuda_runtime.h>

__global__ void vector_add(const float* d_A, const float* d_B, float* d_C, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid < N){
        d_C[tid] = d_A[tid] + d_B[tid];
    }
}