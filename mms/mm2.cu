#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>


int main() {
    int m = 1024; n = 1024, k = 1024;
    
    size_t size_a = m * n * sizeof(float);
    size_t size_b = n * k * sizeof(float);
    size_t size_c = m * k * sizeof(float);
    
    float *a, float *b, float *c;

    cudaMalloc(&a, size_a);
    cudaMalloc(&b, size_b);
    cudaMalloc(&c, size_c);

    dim3 blockDim(16, 16);
    dim3 gridDim()
}