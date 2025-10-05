#include <cuda_runtime.h>

__global__ void invert_kernel(unsigned char* d_image, int width, int height) {
    int c = threadIdx.x + blockIdx.x * blockDim.x;

    if(c < width * height){
        int pos = 4 * c;
        d_image[pos] = 255 - d_image[pos];
        d_image[pos+1] = 255 - d_image[pos+1];
        d_image[pos+2] = 255 - d_image[pos+2];
    }
}