#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 256

__global__
void vec_add(float *a, float *b, float *c) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        c[idx] = a[idx] + b[idx];
    }
}

__global__
void vec_add_coarsened(float *a, float *b, float *c) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx < N) 
        c[idx] = a[idx] + b[idx];
    if (idx + 1 < N)
        c[idx + 1] = a[idx + 1] + b[idx + 1]; 
}

void random_init(float *a, int n) {
    for (int i = 0; i < n; i++) {
        a[i] = rand() % 100;
    }
}

int main() {
    float *a, *b, *c;
    float *d_a, *d_b, *d_c;
    int size = N * sizeof(float);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    a = (float*)malloc(size);
    random_init(a, N);
    b = (float*)malloc(size);
    random_init(b, N);
    c = (float*)malloc(size);
    random_init(c, N);

    cudaEvent_t start, stop, start_coarsened, stop_coarsened;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventCreate(&start_coarsened);
    cudaEventCreate(&stop_coarsened);

    cudaEventRecord(start);
    vec_add<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Vec add time: %f ms\n", milliseconds);

    cudaEventRecord(start_coarsened);
    vec_add_coarsened<<<(N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK / 2>>>(d_a, d_b, d_c);
    cudaEventRecord(stop_coarsened);

    float milliseconds_coarsened = 0;
    cudaEventElapsedTime(&milliseconds_coarsened, start_coarsened, stop_coarsened);
    printf("Vec add coarsened time: %f ms\n", milliseconds_coarsened);

    cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(a);
    free(b);
    free(c);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(start_coarsened);
    cudaEventDestroy(stop_coarsened);

    return 0;

}