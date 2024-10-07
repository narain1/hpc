#include <cuda.h>
#include <cuda_runtime.h>
#include <float.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <stdlib.h>

#define WARPSIZE 32

__global__
void sigmoid_f32_kernel(float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)  {
        b[idx] = 1.0f / (1.0f + expf(-a[idx]));
    }
}

__global__
void sigmoid_f16_kernel(half *a, half *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)  {
        b[idx] = 1.0f / (1.0f + expf(-__half2float(a[idx])));
    }
}

void rand_init(float *a, int N) {
    for (int i = 0; i < N; i++) {
        a[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    int n = 16000;
    float *a = (float *)malloc(n * sizeof(float));
    float *out = (float *)malloc(n * sizeof(float));
    rand_init(a, n);
    float *d_a, *d_out;
    cudaMalloc(&d_a, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMemcpy(d_a, a, n * sizeof(float), cudaMemcpyHostToDevice);
    sigmoid_f32_kernel<<<(n + WARPSIZE - 1) / WARPSIZE, WARPSIZE>>>(d_a, d_out, n);
    cudaMemcpy(a, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < 5; i++) {
        printf("%f ", a[i]);
    }

    printf("\n");
    
    half *a_h = (half *)malloc(n * sizeof(half));
    half *out_h = (half *)malloc(n * sizeof(half));
    for (int i = 0; i < 5; i++) {
        a_h[i] = __float2half(a[i]);
    }
    half *d_a_h, *d_out_h;
    cudaMalloc(&d_a_h, n * sizeof(half));
    cudaMalloc(&d_out_h, n * sizeof(half));
    cudaMemcpy(d_a_h, a_h, n * sizeof(half), cudaMemcpyHostToDevice);
    sigmoid_f16_kernel<<<(n + WARPSIZE - 1) / WARPSIZE, WARPSIZE>>>(d_a_h, d_out_h, n);
    cudaMemcpy(out_h, d_out_h, n * sizeof(half), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++) {
        printf("%f ", __half2float(out_h[i]));
    }

    cudaFree(d_a);
    cudaFree(d_out);
    cudaFree(d_a_h);
    cudaFree(d_out_h);
    free(a);
    free(out);
    free(a_h);
    free(out_h);
    return 0;

}