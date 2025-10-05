#include <cuda_runtime.h>

#define BLOCKSIZE 256

__global__ void dot_prod(const float* d_A, const float* d_B, float* d_result, int N){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ float C[BLOCKSIZE];

    float a = (tid < N) ? d_A[tid] : 0;
    float b = (tid < N) ? d_B[tid] : 0;
    
    C[threadIdx.x] = a * b;
    __syncthreads();

    for(int j = blockDim.x / 2; j > 0; j >>= 1){
        if(threadIdx.x < j){
            C[threadIdx.x] += C[threadIdx.x + j];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0){
        atomicAdd(d_result, C[0]);
    }

}