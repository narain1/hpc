#include <immintrin.h>
#include <stdio.h>
#include <float.h>  // For FLT_MIN
#include <stdlib.h>
#include <time.h>

// Function to find the maximum value using AVX
float avx_max(const float* arr, size_t size) {
    // Initialize max value to the smallest possible float
    __m256 max_val = _mm256_set1_ps(FLT_MIN);
    
    size_t i;
    
    for (i = 0; i <= size - 8; i += 8) {
        __m256 data = _mm256_loadu_ps(&arr[i]);
        max_val = _mm256_max_ps(max_val, data);
    }
    
    float result[8];
    _mm256_storeu_ps(result, max_val);
    float final_max = result[0];
    for (int j = 1; j < 8; ++j) {
        if (result[j] > final_max) {
            final_max = result[j];
        }
    }

    // Handle the remaining elements (if any)
    for (; i < size; ++i) {
        if (arr[i] > final_max) {
            final_max = arr[i];
        }
    }
    
    return final_max;
}

float max(const float *arr, size_t size) {
    float max = FLT_MIN;
    for (int i=0; i<size; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }
    return max;
}

int main() {
    // Example usage
    size_t size = 1024 * 10;
    float *data = (float *)malloc(sizeof(float) * size);
    srand(time(0));
    for (int i=0; i<size; i++)
        data[i] = (float)rand() / RAND_MAX;
     
    clock_t start = clock();
    float max_value = max(data, size);
    clock_t end = clock();

    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Max value: %f, time : %f\n", max_value, end - time_spent);
    
    start = clock();
    max_value = avx_max(data, size);
    end = clock();

    time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Max value: %f, time : %f\n", max_value, end - time_spent);
    
    return 0;
}