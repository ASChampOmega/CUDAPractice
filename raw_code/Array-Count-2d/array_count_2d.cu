#include <cuda_runtime.h>

__global__ void count_2d_equal_kernel(const int* d_input, int* d_output, int N, int M, int K) {
    int y_ = threadIdx.y + blockDim.y * blockIdx.y;
    int x_ = threadIdx.x + blockDim.x * blockIdx.x;
    if(y_ < N && x_ < M && d_input[y_ * M + x_] == K){
        atomicAdd(d_output, 1);
    }
}