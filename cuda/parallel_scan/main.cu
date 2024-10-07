#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <float16.h>
#include <bfloat16.h>
#include <time.h>

#define BLOCK_SIZE 256


__global__
void base_scan_kernel(float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    shared float sdata[BLOCK_SIZE];
    if (idx < N) {
        sdata[tid] = a[idx];
    } else {
        sdata[tid] = 0.0;
    }
    __syncthreads();

    for (unsigned int s = 1; s < blockDim.x; s *= 2) {
        float val = 0.0f;
        if (idx + s < N) {
            val = sdata[idx];
        }
        __syncthreads();
        sdata[idx] += val;
        __syncthreads();
    }
    if (idx < N) {
        b[idx] = sdata[tid];
    }
}

int main() {
    int length = 16000;
    double *base, *output;
    cudaMallocManaged(&base, length * sizeof(double));
    cudaMallocManaged(&output, length * sizeof(double));

    for (int i = 0; i < length; i++) {
        base[i] = 1.0;
        output[i] = 0.0;
    }

    int blockSize = 256;
    int numBlocks = (length + blockSize - 1) / blockSize;
    increment_section<<<numBlocks, blockSize>>>(base, output, length);
    cudaDeviceSynchronize();

    for (int i = 0; i < length; i++) {
        printf("%f ", output[i]);
    }
    printf("\n");

    cudaFree(base);
    cudaFree(output);
    return 0;
}