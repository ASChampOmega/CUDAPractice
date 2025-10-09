#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define TILE 16
__global__ void gemm(const half* d_A, const half* d_B, half* d_C, int M, int N, int K, float alpha, float beta){
    int y_ = threadIdx.y + blockDim.y * blockIdx.y;
    int x_ = threadIdx.x + blockDim.x * blockIdx.x;
    __shared__ half As[TILE][TILE];
    __shared__ half Bs[TILE][TILE];

    if(0 <= y_ && y_ < M && 0 <= x_ && x_ < N){
        d_C[y_ * N + x_] = __float2half(beta * __half2float(d_C[y_ * N + x_]));
    }

    float sum = 0;

    for(int i = 0; i < (K + TILE - 1) / TILE; i++){
        int aCol = i * TILE + threadIdx.x;
        int bRow = i * TILE + threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (y_ < M && aCol < K) ? d_A[y_ * K + aCol] : __float2half(0.0f);
        Bs[threadIdx.y][threadIdx.x] = (bRow < K && x_ < N) ? d_B[bRow * N + x_] : __float2half(0.0f);

        __syncthreads();

        for(int j = 0; j < TILE; j++){
            sum += __half2float(As[threadIdx.y][j]) * __half2float(Bs[j][threadIdx.x]);
        }

        __syncthreads();

    }

    if(0 <= y_ && y_ < M && 0 <= x_ && x_ < N){
        d_C[y_ * N + x_] = __float2half(__half2float(d_C[y_ * N + x_]) + alpha * sum);
    }

}
