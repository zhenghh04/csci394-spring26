#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static inline int idx(int i, int j, int n) {
    return i * n + j;
}

int main(void) {
    const int n = 1024;
    const int steps = 200;
    const double alpha = 0.1;

    double *u = (double *)calloc((size_t)n * (size_t)n, sizeof(double));
    double *v = (double *)calloc((size_t)n * (size_t)n, sizeof(double));
    if (!u || !v) {
        fprintf(stderr, "Allocation failed\n");
        free(u);
        free(v);
        return 1;
    }

    // Initial hot spot in the center
    u[idx(n/2, n/2, n)] = 100.0;

    double t0 = omp_get_wtime();
    for (int s = 0; s < steps; s++) {
        #pragma omp parallel for collapse(2)
        for (int i = 1; i < n - 1; i++) {
            for (int j = 1; j < n - 1; j++) {
                double center = u[idx(i, j, n)];
                double up = u[idx(i - 1, j, n)];
                double down = u[idx(i + 1, j, n)];
                double left = u[idx(i, j - 1, n)];
                double right = u[idx(i, j + 1, n)];
                v[idx(i, j, n)] = center + alpha * (up + down + left + right - 4.0 * center);
            }
        }
        double *tmp = u; u = v; v = tmp;
    }
    double t1 = omp_get_wtime();

    printf("2D heat diffusion: n=%d steps=%d time=%.3f s\n", n, steps, t1 - t0);
    printf("center temp: %.3f\n", u[idx(n/2, n/2, n)]);

    free(u);
    free(v);
    return 0;
}
