#include <cuda.h>
#include <cuda_runtime.h>

__global__
void convolution_1d(float *n, float *m, float *r, int n_size, int m_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float p_value = 0.0f;
    if (i < n_size) {
        r[i] = 0;
        for (int j = 0; j < m_size; j++) {
            if (i - j >= 0) {
                p_value += n[i - j] * m[j];
            }
        }
        r[i] = p_value;
    }
}

void convolution(float *n, float *m, float *r, int n_size, int m_size) {
    float *d_n, *d_m, *d_r;
    cudaMalloc(&d_n, n_size * sizeof(float));
    cudaMalloc(&d_m, m_size * sizeof(float));
    cudaMalloc(&d_r, n_size * sizeof(float));

    cudaMemcpy(d_n, n, n_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m, m, m_size * sizeof(float), cudaMemcpyHostToDevice);

    int block_size = 256;
    int num_blocks = (n_size + block_size - 1) / block_size;

    convolution_1d<<<num_blocks, block_size>>>(d_n, d_m, d_r, n_size, m_size);

    cudaMemcpy(r, d_r, n_size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_n);
    cudaFree(d_m);
    cudaFree(d_r);
}

// cudaMemcpyToSymbol