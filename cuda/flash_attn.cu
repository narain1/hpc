#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>


#define d 64
#define NEG_INF -FLT_MAX
#define B_r 32
#define B_c 32
#define BK 32

#define TM 4
#define TN 4

#define CACHE_Q 1

__global__
void flash_tiled(float *out, float *K, float *Q, float *V, float scaling, int batch_stride, int T_r, int T_c) {
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int batch_offset = batch_stride * blockIdx.x;
    
    __shared__ float Q_i[B_r][d];
    __shared__ float K_j[B_r][B_c];
    __shared__ float V_j[B_r][d];

    __shared__ float S_i[B_r][B_c];

    const int num_tiles = d/B_c;
    float l_i, m_i;

    assert (B_r == B_c && B_r == blockDim.x && B_r == blockDim.y);

    float O_i[num_tiles];
    for (int i = 0; i < num_tiles; i++) {
        O_i[i] = 0;
    }

    // row wise statistics
    for (int t=0; t<num_tiles; t++) {
        l_i = 0.f;
        m_i = NEG_INF;
    }

    // load Q_i
    for (int t=0; t<num_tiles; t++) {
        Q_i[tid_y][t*B_c + tid_x] = Q[batch_offset + tid_y * d + t*B_c + tid_x];
    }
    __syncthreads();


}