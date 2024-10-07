#include <float.h>
#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <stdlib.h>

float add(float *arr, int size) {
    float sum = 0.0f;
    for (int i=0; i<size; i++) {
        sum += arr[i];
    }
    return sum;
}

float avx_add(float *arr, int size) {
    __m256 sum = _mm256_set1_ps(0);
    __m256 data;
    int i;
    for (i=0; i<=size; i+=8) {
        data = _mm256_load_ps(&arr[i]);
        sum = _mm256_add_ps(sum, data); 
    }
    float result[8];
    float total = 0.0f;
    _mm256_store_ps(result, sum);
    for (int j=0; j<8; j++) {
        total += result[j];
    }
    for (; i<size; i++)
        total += arr[i];
    return total;
}

int main() {
    int size = 1024 * 16;
    float *arr = (float *)malloc(sizeof(float) * size);
    struct timespec start, end;
    double time_spent;

    for (int i=0; i<size; i++)
        arr[i] = (float)rand() / RAND_MAX;
    
    clock_gettime(CLOCK_MONOTONIC, &start); 
    float sum = add(arr, size);
    clock_gettime(CLOCK_MONOTONIC, &end);

    time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
    printf("sum = %f, time spend: %f ns\n", sum, time_spent);

    clock_gettime(CLOCK_MONOTONIC, &start);
    sum = avx_add(arr, size);
    clock_gettime(CLOCK_MONOTONIC, &end);

    time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
    printf("avx sum = %f, time spend: %f ns\n", sum, time_spent);

}