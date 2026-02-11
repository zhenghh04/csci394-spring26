#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static inline int idx(int i, int j, int n) {
    return i * n + j;
}

int main(void) {
    const int n = 2048;
    double *u = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    double *lap = (double *)calloc((size_t)n * (size_t)n, sizeof(double));
    if (!u || !lap) {
        fprintf(stderr, "Allocation failed\n");
        free(u); free(lap);
        return 1;
    }

    for (int i = 0; i < n * n; i++) {
        u[i] = (double)(i % 100) * 0.01;
    }

    double t0 = omp_get_wtime();
    #pragma omp parallel for collapse(2)
    for (int i = 1; i < n - 1; i++) {
        for (int j = 1; j < n - 1; j++) {
            lap[idx(i, j, n)] =
                u[idx(i - 1, j, n)] + u[idx(i + 1, j, n)] +
                u[idx(i, j - 1, n)] + u[idx(i, j + 1, n)] -
                4.0 * u[idx(i, j, n)];
        }
    }
    double t1 = omp_get_wtime();

    printf("2D Laplacian: n=%d time=%.3f s\n", n, t1 - t0);
    printf("lap center: %.6f\n", lap[idx(n/2, n/2, n)]);

    free(u);
    free(lap);
    return 0;
}
