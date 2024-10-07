#include <stdio.h>
#include <time.h>
#include <immintrin.h>

void avx_cum_sum(float *arr, int size) {
    __m256 sum = _mm256_setzero_ps();
    for (int i=0; i<size; i+=8) {
        __m256 data = _mm256_loadu_ps(arr + i);
        sum = _mm256_add_ps(sum, data);
        _mm256_storeu_ps(arr + i, sum);
    }
}
void cum_sum(float *arr, int size) {
    for (int i=1; i<size; i++)
        arr[i] += arr[i-1];
}

int main() {
    int size = 1024 * 16;
    float *arr = (float *)malloc(sizeof(float) * size);
    struct timespec start, end;
    double time_spent;

    for (int i=0; i<size; i++)
        arr[i] = (float)rand() / RAND_MAX;
    
    clock_gettime(CLOCK_MONOTONIC, &start); 
    cum_sum(arr, size);
    clock_gettime(CLOCK_MONOTONIC, &end);

    time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
    printf("time spend: %f ns\n", time_spent);

    clock_gettime(CLOCK_MONOTONIC, &start);
    avx_cum_sum(arr, size);
    clock_gettime(CLOCK_MONOTONIC, &end);

    time_spent = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec);
    printf("avx time spend: %f ns\n", time_spent);

}