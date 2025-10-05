#include <cuda_runtime.h>

#define TILE_X 16
#define TILE_Y 16
__global__ void subarray_sum_2d(const int* d_input, int* d_output, int N, int M, int S_ROW, 
                            int E_ROW, int S_COL, int E_COL){
    int x_ = threadIdx.x + blockDim.x * blockIdx.x;
    int y_ = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ int Ms[TILE_X * TILE_Y];
    
    int v = (S_ROW <= y_ && y_ <= E_ROW && S_COL <= x_ && x_ <= E_COL) ? d_input[y_ * M + x_] : 0;
    Ms[threadIdx.y * TILE_X + threadIdx.x] = v;
    __syncthreads();

    int p = threadIdx.y * TILE_X + threadIdx.x;
    for(int j = TILE_X * TILE_Y / 2; j > 0; j >>= 1){
        if(p < j){
            Ms[p] += Ms[p + j];
        }
        __syncthreads();
    }

    if(p == 0){
        atomicAdd(d_output, Ms[0]);
    }
}