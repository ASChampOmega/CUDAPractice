#include <cuda_runtime.h>

// Tiling Idea for improvement: Launch threads in d_kernel dims. Each thread loads at most 4 values into shared
// (0, 0) loads: Image (0, 0), (kernel_rows, 0), (0, kernel_cols), (kernel_rows, kernel_cols)
// (0, 1) loads: Image (0, 1), (kernel_rows, 1), (0, kernel_cols + 1), (kernel_rows, kernel_cols + 1)
// Might have low occupancy problems though

__global__ void conv_2d(const float *input, const float* d_kernel, float *output, 
        int input_rows, int input_cols, int kernel_rows, int kernel_cols){
    int y_ = threadIdx.y + blockDim.y * blockIdx.y;
    int x_ = threadIdx.x + blockDim.x * blockIdx.x;

    int out_rows = input_rows - kernel_rows + 1;
    int out_cols = input_cols - kernel_cols + 1;

    if(y_ < out_rows && x_ < out_cols){
        float v = 0;
        for(int dy = 0; dy < kernel_rows; dy++){
            const int in_r = y_ + dy;
            const int in_base = in_r * input_cols + x_;
            const int k_base  = dy * kernel_cols;

            for(int dx = 0; dx < kernel_cols; dx++){
                v += input[in_base + dx] * d_kernel[k_base + dx];
            }
        }
        output[y_ * out_cols + x_] = v;
    }
}