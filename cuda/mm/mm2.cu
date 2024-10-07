#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 16

__global__
void mm_kernel(float *a, float *b, float *c, int n) {
    __shared__ float ds_a[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_b[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float sum = 0.0f;
    for (int m = 0; m < n / TILE_WIDTH; m++) {
        /// loading data into shared memory
        ds_a[ty][tx] = a[row * n + m * TILE_WIDTH + tx];
        ds_b[ty][tx] = b[(m * TILE_WIDTH + ty) * n + col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++) {
            sum += ds_a[ty][k] * ds_b[k][tx];
        }
        __syncthreads();
    }
    c[row * n + col] = sum;

}
int main() {
    int n = 4096;
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;

    a = (float*)malloc(n * n * sizeof(float));
    b = (float*)malloc(n * n * sizeof(float));
    c = (float*)malloc(n * n * sizeof(float));
    
    for (int i = 0; i < n * n; i++) {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }

    printf("Matrix A\n");
    for (int i=0; i<5; i++) {
        for (int j=0; j<5; j++) {
            printf("%f ", a[i * n + j]);
        }
        printf("\n");
    }

    printf("Matrix B\n");
    for (int i=0; i<5; i++) {
        for (int j=0; j<5; j++) {
            printf("%f ", b[i * n + j]);
        }
        printf("\n");
    }

    cudaMalloc(&d_a, n * n * sizeof(float));
    cudaMalloc(&d_b, n * n * sizeof(float));
    cudaMalloc(&d_c, n * n * sizeof(float));

    cudaMemcpy(d_a, a, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * n * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 dimGrid(n / TILE_WIDTH, n / TILE_WIDTH, 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    mm_kernel<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, n);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);

    printf("Elapsed time: %f ms\n", elapsedTime);
    cudaMemcpy(c, d_c, n * n * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Matrix C\n");    
    for (int i=0; i<5; i++) {
        for (int j=0; j<5; j++) {
            printf("%f ", c[i * n + j]);
        }
        printf("\n");
    }
}