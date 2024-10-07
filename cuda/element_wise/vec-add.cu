#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>


void create_fp32_vector(float *a, int size) {
    for (int i =0; i<size; i++) {
        a[i] = (float)rand()/RAND_MAX;
    }
}

__global__ void vec_add_kernel(float *a, float *b, float *c, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) {
            exit(code);
        }
    }
}

inline unsigned int cdiv(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

int main() {
    int n = 1024;

    const int n_threads = 256;
    unsigned int num_blocks = cdiv(n, n_threads);

    size_t size = n * sizeof(float);
    float *a = (float *)malloc(n * sizeof(float));
    float *b = (float *)malloc(n * sizeof(float));
    float *c = (float *)malloc(n * sizeof(float));

    create_fp32_vector(a, n);
    create_fp32_vector(b, n);
    printf("array created\n");

    printf("array randomized\n");

    for (int i = 0; i < 5; i++) {
        printf("%f ", a[i]);
    }
    printf("\n");

    for (int i = 0; i < 5; i++) {
        printf("%f ", b[i]);
    }
    printf("\n");

    // device memories
    float *ad, *bd, *cd;
    cudaMalloc((void **)&ad, size);
    cudaMalloc((void **)&bd, size);
    cudaMalloc((void **)&cd, size);

    cudaMemcpy(ad, a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, size, cudaMemcpyHostToDevice);

    vec_add_kernel<<<num_blocks, n_threads>>>(ad, bd, cd, n);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    cudaMemcpy(c, cd, size, cudaMemcpyDeviceToHost);

    printf("array added\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", c[i]);
    }
    printf("\n");

    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);

    free(a);
    free(b);
    free(c);
}
