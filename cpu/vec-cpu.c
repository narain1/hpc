#include <stdio.h>
#include <stdlib.h>

int main() {
    int n = 1000;
    float *a = (float *)malloc(n * sizeof(float));
    for (int i=0; i<n; i++) {
        a[i] = (float)rand() / RAND_MAX;
    } 
    float sum = 0;
    for (int i=0; i<n; i++) {
        sum += a[i];
    }
    printf("sum:  %f\n", sum);
    printf("mean: %f\n", sum/n);
}