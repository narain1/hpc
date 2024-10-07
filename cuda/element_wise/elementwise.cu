#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <float.h>
#include <vector>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>


#define WARPSIZE 32
#define INT4(value) reinterpret_cast<int4*>(&(value))
#define FLOAT4(value) reinterpret_cast<float4*>(&(value))
#define HALF2(value) reinterpret_cast<half2*>(&(value))
#define BFLOAT2(value) reinterpret_cast<__nv_bfloat162*>(&(value))
#define LDST128BITS(value) reinterpret_cast<float4*>(&(value))

__global__
void elementwise_add_f32_kernel(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)  {
        c[idx] = a[idx] + b[idx];
    }
}

__global__
void elementwise_add_f16_kernel(half *a, half *b, half *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)  {
        c[idx] = a[idx] + b[idx];
    }
}

// vector simd kernel
__global__
void elementwise_add_f32x4_kernel(float *a, float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)  {
        float4 a4 = *FLOAT4(a[idx]); // load 4 floats
        float4 b4 = *FLOAT4(b[idx]);
        float4 c4;
        c4.x = a4.x + b4.x; // add 4 floats
        c4.y = a4.y + b4.y;
        c4.z = a4.z + b4.z;
        c4.w = a4.w + b4.w;
        *FLOAT4(c[idx]) = c4;
    }
}

__global__
void elementwise_add_f16x2_kernel(half *a, half *b, half *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)  {
        half2 a2 = HALF2(a[idx]); // load 2 halfs
        half2 b2 = HALF2(b[idx]);
        half2 c2;
        c2.x = __hadd(a2.x + b2.x); // add 2 halfs
        c2.y = __hadd(a2.y + b2.y);
        *HALF2(c[idx]) = c2;
    }
}

__global__
void relu_f32_kernel(float *a, float *b, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)  {
        b[idx] = fmaxf(0.0f, a[idx]); // hinge
    }
}

__global__
void elementwise_add_f16x8_kernel(half *a, half *b, half *c, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    half2 reg_a_0 = HALF2(a[idx + 0]);
    half2 reg_a_1 = HALF2(a[idx + 1]);
    half2 reg_a_2 = HALF2(a[idx + 2]);
    half2 reg_a_3 = HALF2(a[idx + 3]);
    half2 reg_b_0 = HALF2(b[idx + 0]);
    half2 reg_b_1 = HALF2(b[idx + 1]);
    half2 reg_b_2 = HALF2(b[idx + 2]);
    half2 reg_b_3 = HALF2(b[idx + 3]);
    half2 reg_c_0, reg_c_1, reg_c_2, reg_c_3;
    reg_c_0.x = __hadd(reg_a_0.x + reg_b_0.x);
    reg_c_0.y = __hadd(reg_a_0.y + reg_b_0.y);
    reg_c_1.x = __hadd(reg_a_1.x + reg_b_1.x);
    reg_c_1.y = __hadd(reg_a_1.y + reg_b_1.y); 
    reg_c_2.x = __hadd(reg_a_2.x + reg_b_2.x);
    reg_c_2.y = __hadd(reg_a_2.y + reg_b_2.y);
    reg_c_3.x = __hadd(reg_a_3.x + reg_b_3.x);
    reg_c_3.y = __hadd(reg_a_3.y + reg_b_3.y);
    if ((idx + 0) < N) { HALF2(c[idx + 0]) = reg_c_0; }
    if ((idx + 2) < N) { HALF2(c[idx + 2]) = reg_c_1; }
    if ((idx + 4) < N) { HALF2(c[idx + 4]) = reg_c_2; }
    if ((idx + 6) < N) { HALF2(c[idx + 6]) = reg_c_3; }
}