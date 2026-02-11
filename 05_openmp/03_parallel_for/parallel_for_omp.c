#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>

int main(int argc, char *argv[]) {
    const int N = 1000000;
    int i;
    double *a = (double *)malloc((size_t)N * sizeof(double));
    double *b = (double *)malloc((size_t)N * sizeof(double));
    double *result = (double *)malloc((size_t)N * sizeof(double));
    if (!a || !b || !result) {
        fprintf(stderr, "Allocation failed for N=%d\n", N);
        free(a);
        free(b);
        free(result);
        return 1;
    }

    // Initialize
    for (i = 0; i < N; i++) {
        a[i] = 1.0 * i;
        b[i] = 2.0 * i;
    }

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    #pragma omp parallel for
    for (i = 0; i < N; i++) {
        result[i] = a[i] + b[i];
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    printf("TEST result[19] = %g\n", result[19]);
    double elapsed = (double)(t1.tv_sec - t0.tv_sec) +
                     1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    printf("parallel_time_s=%.6f\n", elapsed);
    free(a);
    free(b);
    free(result);
    return 0;
}
