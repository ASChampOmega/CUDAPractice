#include <cuda_runtime.h>

__global__ void copy_matrix_kernel(const float* d_A, float* d_B, int N) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if(tid < N * N){
        d_B[tid] = d_A[tid];
    }
}