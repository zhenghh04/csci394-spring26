#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double run_kernel(const double *a, const double *b, double *result, int n, int repeats) {
    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);

    for (int r = 0; r < repeats; r++) {
        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < n; i++) {
            result[i] = a[i] + b[i];
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &t1);
    return (double)(t1.tv_sec - t0.tv_sec) + 1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
}

int main(int argc, char **argv) {
    int n = 100000000;
    int repeats = 5;
    int print_map = 0;

    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) repeats = atoi(argv[2]);
    if (argc > 3 && strcmp(argv[3], "--map") == 0) print_map = 1;
    if (n <= 0 || repeats <= 0) {
        fprintf(stderr, "Usage: %s [N] [repeats] [--map]\n", argv[0]);
        return 1;
    }

    double *a = (double *)malloc((size_t)n * sizeof(double));
    double *b = (double *)malloc((size_t)n * sizeof(double));
    double *result = (double *)malloc((size_t)n * sizeof(double));
    if (!a || !b || !result) {
        fprintf(stderr, "Allocation failed for N=%d\n", n);
        free(a);
        free(b);
        free(result);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        a[i] = (double)i;
        b[i] = 2.0 * (double)i;
    }

    if (print_map) {
        int m = n < 64 ? n : 64;
        #pragma omp parallel for schedule(runtime)
        for (int i = 0; i < m; i++) {
            int tid = omp_get_thread_num();
            printf("thread %d -> i=%d\n", tid, i);
        }
    }

    double elapsed = run_kernel(a, b, result, n, repeats);
    printf("N=%d repeats=%d threads=%d\n", n, repeats, omp_get_max_threads());
    printf("result[19]=%.1f\n", result[19]);
    printf("elapsed_s=%.6f avg_per_repeat_s=%.6f\n", elapsed, elapsed / (double)repeats);

    free(a);
    free(b);
    free(result);
    return 0;
}
