#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 16
#define CEIL_DIV(x, y) (((x) + (y) - 1) / (y))
#define BM 64
#define BK 8

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

__global__
void mm_kernel_shared(float *a, float *b, float *c, int n) {
    __shared__ float a_shared[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float b_shared[BLOCK_SIZE * BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float sum = 0.0f;
    int tid = threadIdx.y * threadIdx.x + threadIdx.x;

    for (int i=0; i < CEIL_DIV(n, BLOCK_SIZE); i++) {
        if (row < n && i * BLOCK_SIZE + threadIdx.x < n) {
            a_shared[tid] = a[row * n + i * BLOCK_SIZE + threadIdx.x];
        } else {
            a_shared[tid] = 0.0f;
        }

        if (col < n && i * BLOCK_SIZE + threadIdx.y < n) {
            b_shared[tid] = b[(i * BLOCK_SIZE + threadIdx.y) * n + col];
        } else {
            b_shared[tid] = 0.0f;
        }

        __syncthreads();

        for (int j = 0; j < BLOCK_SIZE; j++) {
            sum += a_shared[threadIdx.y][j] * b_shared[j][threadIdx.x];
        }

        __syncthreads();
    }
    c[row * n + col] = sum;
}

__global__
void mm_1d_tiling_shared(float *a, float *b, float *c, int n) {
    // offsetting the block but unsure where its used
    const uint crow = blockIdx.y; // used in calculating the offset
    const uint ccol = blockIdx.x;

    const int thread_col = threadIdx.x % 64; // (0, 64) trying to group 8 threads
    const int thread_row = threadIdx.x / 64; // (0, 8) trying to group 8 threads

    shared float a_shared[BM * BK];
    shared float b_shared[BM * BK];

    // offsetting the block
    a += crow * BM * n;
    b += ccol * BM;
    c += crow * BM * n + ccol * BM;

    const uint inner_col_a = threadIdx.x % BK;
    const uint inner_row_a = threadIdx.x / BK;
    const uint inner_col_b = threadIdx.x % BK;
    const uint inner_row_b = threadIdx.x / BK;

    float threadResult[BK] = {0.0f};

    for (uint bkIdx=0; bkIdx < n; bkIdx += BK) {
        a_shared[inner_row_a * BK + inner_col_a] = a[inner_row_a * n + inner_col_a];
        b_shared[inner_row_b * BK + inner_col_b] = b[inner_row_b * n + inner_col_b];
        __syncthreads();

        // offset blocktile
        a += BK;
        b += BK * n;

        // compute per result thread
        for (uint dotIdx=0; dotIdx < BK; dotIdx++) {
            float tmp = BS[dotIdx * BN + thread_col];
            for (uint resIdx=0; resIdx < BK; resIdx++) {
                threadResult[resIdx] += a_shared[thread_row * BK + resIdx] * tmp;
            }
        }
        __syncthreads();
    }
    for (uint resIdx=0; resIdx < BK; resIdx++) {
        c[thread_row * n + inner_col_b] = threadResult[resIdx];
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