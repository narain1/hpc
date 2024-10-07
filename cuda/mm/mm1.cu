#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))

__global__
void mm_kernel(float *a, float *b, float *c, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += a[row * n + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

__global__
void mm_kernel_coaleced(float *a, float *b, float *c, int n) {
    const int x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const int y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if (x < n && y < n) {
        float sum = 0.0f;
        for (int i = 0; i < n; i++) {
            sum += a[x * n + i] * b[i * n + y];
        }
        c[x * n + y] = sum;
    }
}

int main() {
    int n = 4096;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    int size = n * n * sizeof(float);

    a = (float *)malloc(size);
    b = (float *)malloc(size);
    c = (float *)malloc(size);

    for (int i = 0; i < n * n; i++) {
        a[i] = 1.0f;
        b[i] = 2.0f;
    }

    cudaMalloc((void **)&d_a, size);
    cudaMalloc((void **)&d_b, size);
    cudaMalloc((void **)&d_c, size);

    cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32 * 32);
    dim3 numBlocks(CEIL_DIV(n, 32), CEIL_DIV(n, 32), 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    mm_kernel<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Elapsed time: %f ms\n", elapsedTime);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaEventRecord(start, 0);

    mm_kernel_coaleced<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Elapsed time coalleced: %f ms\n", elapsedTime);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
 
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%f ", c[i * n + j]);
        }
        printf("\n");
    }

    free(a);
    free(b);
    free(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}