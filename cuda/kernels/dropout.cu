#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>

__global__ void dropout(float *input, float *output, int size, float dropout_rate, unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        curandState state;
        curand_init(seed, idx, 0, &state);
        float random = curand_uniform(&state);
        output[idx] = (random > dropout_rate) ? input[idx] : 0;
    }
}

int main () {
    int size = 1024;
    float *input, *output;
    float dropout_rate = 0.1;
    unsigned long long seed = 1234;
    cudaMallocManaged(&input, size * sizeof(float));
    cudaMallocManaged(&output, size * sizeof(float));
    for (int i = 0; i < size; i++) {
        input[i] = rand() / (float)RAND_MAX;
    }
    dropout<<<(size + 255) / 256, 256>>>(input, output, size, dropout_rate, seed);
    cudaDeviceSynchronize();
    for (int i = 0; i < 25; i++) {
        printf("%f\n", output[i]);
    }
    cudaFree(input);
    cudaFree(output);
    return 0;
}