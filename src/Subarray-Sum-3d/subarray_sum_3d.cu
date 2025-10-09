#include <cuda_runtime.h>

#define TILE 8
__global__ void subarray_sum_3d(const int* d_input, int* d_output, int N, int M, int K, int S_DEP, int E_DEP, int S_ROW, int E_ROW, int S_COL, int E_COL){
    int z = threadIdx.z + blockDim.z * blockIdx.z;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    
    __shared__ int Ms[TILE*TILE*TILE];
    int v = 0;
    if(
        (S_DEP <= z && z <= E_DEP) &&
        (S_ROW <= y && y <= E_ROW) &&
        (S_COL <= x && x <= E_COL)
    ) {
        v = d_input[z * M * K + y * K + x];
    }
    
    int pos = threadIdx.z * TILE * TILE + threadIdx.y * TILE + threadIdx.x;
    Ms[pos] = v;
    
    __syncthreads();

    for(int j = TILE*TILE*TILE / 2; j > 0; j >>= 1){
        if(pos < j){
            Ms[pos] += Ms[pos + j];
        }
        __syncthreads();
    }
    if(pos == 0){
        atomicAdd(d_output, Ms[0]);
    }
}