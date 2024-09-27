#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <iostream>
#include <cuda_fp16.h>

using namespace std;

#define FINAL_MASK 0xffffffff

template <typename T>
__inline__ __device__
T warpReduceSum(T Val) {
    for (int mask=16; mask > 0; mask >>= 1)
        val += __shfl_xor_sync(FINAL_MASK, val, mask, 32);
    return val;
}

