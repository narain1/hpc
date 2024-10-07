#include <stdio.h>
#include <cuda_runtime.h>

__global__
void process_array_with_divergence(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        if (idx % 2 == 0) {
            data[idx] = data[idx] * 2;
        } else {
            data[idx] = data[idx] * 3;
        }
    }
}