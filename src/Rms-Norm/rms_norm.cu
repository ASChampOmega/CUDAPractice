#include <cuda_runtime.h>
#include <cmath>

#define TILE 256

__global__ void rms_comp(const float* __restrict__ d_input,
                         float* __restrict__ d_output,
                         int N)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    __shared__ float M[TILE];
    float val = 0.0f;
    if (tid < N) val = d_input[tid] * d_input[tid];
    M[threadIdx.x] = val;
    __syncthreads();

    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            M[threadIdx.x] += M[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(d_output, M[0] / static_cast<float>(N));
    }
}

__global__ void finalize_rms(float* d_rms, float eps) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *d_rms = sqrtf(*d_rms + eps);
    }
}

__global__ void apply_norm(const float* __restrict__ d_input,
                           float gamma, float beta,
                           float* __restrict__ d_output,
                           int N, const float* __restrict__ d_rms)
{
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if (tid < N) {
        float r = *d_rms;
        d_output[tid] = (d_input[tid] / r) * gamma + beta;
    }
}