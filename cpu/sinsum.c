#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>

inline double sinsum(double x, int terms) {
    double term = x;
    double sum = term;
    double x2 = x * x;
    for (int i = 1; i < terms; i++) {
        term *= -x2 / ((2 * i) * (2 * i + 1));
        sum += term;
    }
    return sum;
}

int main(int argc, char *argv[]) {
    int steps = (argc > 1) ? atoi(argv[1]) : 1000000;
    int terms = (argc > 2) ? atoi(argv[2]) : 1000;
    int threads = (argc > 3) ? atoi(argv[3]) : 4;

    double pi = 3.14159265358979323846;
    double step_size = pi / (steps - 1);

    clock_t start = clock();
    double cpu_sum = 0;
    omp_set_num_threads(threads);
    #pragma omp parallel for reduction(+:cpu_sum)
    for (int i = 0; i < steps; i++) {
        double x = i * step_size;
        cpu_sum += sinsum(x, terms);
    }

    double cpu_time = (double)(clock() - start) / CLOCKS_PER_SEC;
    cpu_sum -= 0.5 * (sinsum(0, terms) + sinsum(pi, terms));
    cpu_sum *= step_size;
    printf("cpu sum = %.10f, steps %d terms %d time %.3f ms\n", cpu_sum, steps, terms, cpu_time * 1000);
    return 0;
}
