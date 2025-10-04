#include <cuda_runtime.h>
#define TILE 256
#define DEPTH 8

__global__ void thread_reduce(const float* d_input, float* d_output, int N){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    
    __shared__ float Mds[TILE];

    float val = (tid < N) ? d_input[tid] : 0.0f;
    Mds[threadIdx.x] = val;

    __syncthreads();

    for(int j = blockDim.x/2; j > 0; j /= 2){
        if(threadIdx.x < j){
            Mds[threadIdx.x] += Mds[threadIdx.x + j];
        }
        __syncthreads();
    }

    if(threadIdx.x == 0)
        atomicAdd(d_output, Mds[0]);
}