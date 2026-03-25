#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static inline int idx(int i, int j, int n) {
    return i * n + j;
}

int main(int argc, char **argv) {
    int n = 2048;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n < 8) {
        fprintf(stderr, "Usage: %s [N>=8]\n", argv[0]);
        return 1;
    }

    const size_t count = (size_t)n * (size_t)n;
    const size_t bytes = count * sizeof(double);

    double *u = (double *)malloc(bytes);
    double *lap = (double *)calloc(count, sizeof(double));
    if (!u || !lap) {
        fprintf(stderr, "host allocation failed\n");
        free(u);
        free(lap);
        return 1;
    }

    for (size_t i = 0; i < count; ++i) {
        u[i] = (double)(i % 100) * 0.01;
    }

    double start = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < n - 1; ++i) {
        for (int j = 1; j < n - 1; ++j) {
            int center = idx(i, j, n);
            lap[center] = u[idx(i - 1, j, n)] + u[idx(i + 1, j, n)] +
                          u[idx(i, j - 1, n)] + u[idx(i, j + 1, n)] -
                          4.0 * u[center];
        }
    }
    double stop = omp_get_wtime();
    double cpu_ms = 1000.0 * (stop - start);

    printf("OpenMP 2D Laplacian\n");
    printf("n=%d\n", n);
    printf("cpu_time_ms=%.3f\n", cpu_ms);
    printf("lap center=%.6f\n", lap[idx(n / 2, n / 2, n)]);

    free(u);
    free(lap);
    return 0;
}
