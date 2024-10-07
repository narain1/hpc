#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__
void softmax(float *input, float *output, int n) {
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    float max_val = -INFINITY;
    for (int i=tid; i<n; i+=blockDim.x) {
        max_val = fmaxf(max_val, input[i]);
    }

    sdata[tid] = max_val;
    __syncthreads();

    // parallel scan to find max value
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        }
        __syncthreads();
    }

    max_val = sdata[0];

    float sum = 0.0f;
    for (int i=tid; i<n; i+=blockDim.x) {
        sum += expf(input[i] - max_val);
    }
    shared_data[tid] = sum;
    __syncthreads();

    for (int stride=blockDim.x/2; stride>0; stride/=2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    sum_exp = sdata[0];

    for (int i=tid; i<n; i+=blockDim.x) {
        output[i] = expf(input[i] - max_val) / sum_exp;
    }
    
}

// warp-level reduction for finding the maximum value
__device__ float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}


__global__
void softmax_forward_online_kernel8(float *out, const float *inp, int n, int c) {
    const int warps_per_block = blockDim.x / warpSize;
    int tid = threadIdx.x;

    if (tid >= C) {
        return;
    }

    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;

    if (row >= N) {
        return;
    }


}