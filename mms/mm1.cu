#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>

__global__ void matrix_multiply(float *a, float *b, float *c, int m, int n, int k) {
    int row = blockIdx.y * blockIdx.y + threadIdx.y;
    int col = blockIdx.x * blockIdx.x + threadIdx.x;

    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i=0; i<k; ++i) {
            sum += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    const int m = 1024;
    const int n = 1024;
    const int k = 1024;

    size_t size_a = m * k * sizeof(float);
    size_t size_b = k * n * sizeof(float);
    size_t size_c = m * n * sizeof(float);

    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, size_a);
    cudaMalloc(&d_b, size_b);
    cudaMalloc(&d_c, size_c);

    dim3 blockDim(16, 16);
    dim3 gridDim((n + blockDim.x - 1) / blockDim.x, (m + blockDim.y - 1) / blockDim.y);
    matrix_multiply<<<gridDim, blockDim>>>(d_a, d_b, d_c, m, n, k);

    cudaDeviceSynchronize();

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "cuda error: " << cudaGetErrorString(error) << std::endl;
        return -1;
    }

    return 0;
}