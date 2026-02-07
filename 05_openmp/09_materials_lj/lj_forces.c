#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(void) {
    const int n = 2000;
    const double sigma = 1.0;
    const double epsilon = 1.0;
    const double cutoff = 2.5 * sigma;
    const double cutoff2 = cutoff * cutoff;

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    double *fx = (double *)calloc((size_t)n, sizeof(double));
    double *fy = (double *)calloc((size_t)n, sizeof(double));
    if (!x || !y || !fx || !fy) {
        fprintf(stderr, "Allocation failed\n");
        free(x); free(y); free(fx); free(fy);
        return 1;
    }

    // Simple positions on a grid
    for (int i = 0; i < n; i++) {
        x[i] = (double)(i % 50) * 1.1;
        y[i] = (double)(i / 50) * 1.1;
    }

    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; i++) {
        double fxi = 0.0;
        double fyi = 0.0;
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            double dx = x[i] - x[j];
            double dy = y[i] - y[j];
            double r2 = dx * dx + dy * dy;
            if (r2 > cutoff2) continue;
            double inv_r2 = 1.0 / r2;
            double sr2 = (sigma * sigma) * inv_r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            double f = 24.0 * epsilon * inv_r2 * (2.0 * sr12 - sr6);
            fxi += f * dx;
            fyi += f * dy;
        }
        fx[i] = fxi;
        fy[i] = fyi;
    }
    double t1 = omp_get_wtime();

    printf("LJ forces: n=%d time=%.3f s\n", n, t1 - t0);
    printf("fx[0]=%.3e fy[0]=%.3e\n", fx[0], fy[0]);

    free(x); free(y); free(fx); free(fy);
    return 0;
}
