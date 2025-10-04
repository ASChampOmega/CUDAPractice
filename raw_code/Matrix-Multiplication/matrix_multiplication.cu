#include <cuda_runtime.h>

// TODO: Write this tiled

__global__ void matrix_multiplication_kernel(const float* d_A, const float* d_B, float* d_C, int M, int N, int K) {
    int x_ = threadIdx.x + blockDim.x * blockIdx.x;
    int y_ = threadIdx.y + blockDim.y * blockIdx.y;
    if(y_ < M && x_ < K){
        float sum = 0;
        for(int k = 0; k < N; k++){
            sum += d_A[y_ * N + k] * d_B[k * K + x_];
        }
        d_C[y_ * K + x_] = sum;
    }
}