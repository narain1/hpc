#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>

__global__ void vector_add(float *a, float *b, float *c, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int i=idx; i < n * m; i += stride) {
        int row = i / m;
        c[i] = a[i] + b[row]; // global memory access
    }
}

int main() {
    int n = 1024;
    int m = 512;

    float *a = new float[n * m];
    float *b = new float[n];
    float *c = new float[n * m];

    for (int i=0; i<n*m; i++) {
        a[i] = static_cast<float>(i);
    }
    for (int i=0; i<n; i++) {
        b[i] = static_cast<float>(i);
    } 

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, n*m * sizeof(float));
    cudaMalloc(&d_b, n * sizeof(float));
    cudaMalloc(&d_c, n*m*sizeof(float));

    cudaMemcpy(d_a, a, n * m * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, n * sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 256;
    int num_blocks = (n * m + blockSize - 1) / blockSize;

    vector_add<<<num_blocks, blockSize>>>(d_a, d_b, d_c, n, m);
    cudaMemcpy(c, d_c, n * m * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    delete[] a;
    delete[] b;
    delete[] c;

}