#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* d_input, float* d_output, int rows, int cols) {
    int r = threadIdx.y + blockIdx.y * blockDim.y;
    int c = threadIdx.x + blockIdx.x * blockDim.x;

    if(c < cols && r < rows)
        d_output[rows * c + r] = d_input[cols * r + c];

}
