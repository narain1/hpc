#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>


__global__ void print_thread_idx() {
    int ix = threadIdx.x + blockIdx.x * blockDim.x;
    int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * gridDim.x * blockDim.x + ix;
    printf("thread_id(%d, %d) block_id(%d, %d) coordinate(%d, %d) global index(%d)\n",
           threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, ix, iy, idx);
}

void initDevice(int dev) {
    int device = dev;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device);
    printf("Using Device %d: %s\n", device, deviceProp.name);
    cudaSetDevice(device);
}

void initialData(float *ip, const int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
    }
}

void printData(const char *msg, float *ip, const int size) {
    printf("%s: ", msg);
    for (int i = 0; i < size; i++) {
        printf("%1.2f ", ip[i]);
    }
    printf("\n");
}

void CHECK(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);
        fprintf(stderr, "code:%d, reason: %s\n", error, cudaGetErrorString(error));
        exit(1);
    }
}

int main(int argc, char **argv) {
    initDevice(0);
    int nx = 8, ny = 8;
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    float *a_h = (float *)malloc(nBytes);
    initialData(a_h, nxy);
    printData("a", a_h, nxy);

    float *a_d;
    cudaMalloc((float **)&a_d, nBytes);
    cudaMemcpy(a_d, a_h, nBytes, cudaMemcpyHostToDevice);

    dim3 block(4, 2);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    print_thread_idx<<<grid, block>>>();

    CHECK(cudaDeviceSynchronize());
    cudaFree(a_d);
    free(a_h);
    cudaDeviceReset();
    return 0;
    
}