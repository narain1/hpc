#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// divergence as this kernel is blocking all threads until kernel completion
__global__ void simple_sum_reduction(float *inp, float *out) {
    unsigned int i = 2 * threadIdx.x;
    for (unsigned int stride=1; stride <= blockDim.x; stride *=2) {
        if (threadIdx.x % stride == 0) {
            inp[i] += inp[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *out = inp[0];
}

__global__ void covergent_sum_kernel(float *inp, float *out) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride=blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            inp[i] += inp[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
        *out = inp[0];
} 


void rand_vector(float *arr, int size) {
    for (int i=0; i<size; i++)
        arr[i] = (float)rand() / RAND_MAX;
}

int main() {
    int size = 1024 * 16;
    float *arr = (float *)malloc(sizeof(float) * size);
    rand_vector(arr, size);
    clock_t start, end;
    double time_spent;

    float *arr_d, out_d, *out_host;
    cudaMalloc(&arr_d, sizeof(float) * size);
    cudaMalloc(&out_d, sizeof(float));
    cudaMemcpy(arr_d, arr, size * sizeof(float), cudaMemcpyHostToDevice);

    start = clock();
    simple_sum_reduction<<<1, size / 2>>>(arr_d, out_d);
    cudaDeviceSynchronize();
    end = clock();
    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    cudaMemcpy(&out_host, out_d, sizeof(float), cudaMemcpyDeviceToHost);
    printf("avx sum = %f, time spend: %f\n", out_host, time_spent);

    free(arr);
    cudaFree(arr_d);
    cudaFree(out_d);
}