#include <cuda.h>
#include <cuda_runtime.h>

#define TILE_SIZE 256

__global__ void convolution_1d_tiled_caching_kernel(float *n, float *p, int mask_width, int width) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float n_ds[TILE_SIZE];

    n_ds[threadIdx.x] = N[i];

    __syncthreads();

    int this_tile_start_point = blockIdx.x * blockDim.x;
    int next_tile_start_point = (blockIdx.x + 1) * blockDim.x;
    int n_start_point = i - (mask_width/2);
    float pvalue = 0;

    for (int j=0; j<mask_width; j++) {
        int n_index = n_start_point + j;
        if (n_index >= 0 && n_index < width) {
            if ((n_index >= this_tile_start_point) && (n_index < next_tile_start_point)) {
                pvalue += n_ds[threadIdx.x + j - (mask_width/2)] * M[j];
            } else {
                pvalue += n[n_index] * m[j];
            }
        }
    }
    p[i] = pvalue;
}