#include <iostream>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int device = 0; device < deviceCount; device++) {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, device);

        std::cout << "Device " << device << ": " << deviceProp.name << std::endl;

        // Check Tensor Core support based on compute capability
        if (deviceProp.major >= 7) {
            std::cout << "  Tensor Cores: Supported" << std::endl;
        } else {
            std::cout << "  Tensor Cores: Not Supported" << std::endl;
        }

        // Additional relevant properties
        std::cout << "  Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "  Multi-Processor Count: " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "  Warp Size: " << deviceProp.warpSize << std::endl;
        std::cout << "  Maximum Threads per Multi-Processor: " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
    }

    return 0;
}
