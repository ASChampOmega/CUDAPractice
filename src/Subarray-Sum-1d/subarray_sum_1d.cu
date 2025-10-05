#include <cuda_runtime.h>

#define TILE 256

__global__ void subarray_sum(const int* d_input, int* d_output, int N, int S, int E){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ int M[TILE];
    
    int v = (S <= tid && tid <= E) ? d_input[tid] : 0;
    M[threadIdx.x] = v;
    __syncthreads();

    for(int j = TILE / 2; j > 0; j >>= 1){
        if(threadIdx.x < j){
            M[threadIdx.x] += M[threadIdx.x + j];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0){
        atomicAdd(d_output, M[0]);
    }
}
