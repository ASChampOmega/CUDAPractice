#include <cuda_runtime.h>

__global__ void convolution_1d_kernel(const float* d_input, const float* d_kernel, float* d_output,
                                      int input_size, int kernel_size) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    if(tid <= input_size - kernel_size){
        float s = 0;
        for(int i = 0; i < kernel_size; i++){
            s += d_input[tid + i] * d_kernel[i];
        }
        d_output[tid] = s;
    }
}