#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int n = 2000;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n < 2) {
        fprintf(stderr, "Usage: %s [n>=2]\n", argv[0]);
        return 1;
    }

    const double sigma = 1.0;
    const double epsilon = 1.0;
    const double cutoff = 2.5 * sigma;
    const double cutoff2 = cutoff * cutoff;
    const size_t bytes = (size_t)n * sizeof(double);

    double *x = (double *)malloc(bytes);
    double *y = (double *)malloc(bytes);
    double *fx = (double *)malloc(bytes);
    double *fy = (double *)malloc(bytes);
    if (!x || !y || !fx || !fy) {
        fprintf(stderr, "host allocation failed\n");
        free(x);
        free(y);
        free(fx);
        free(fy);
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        x[i] = (double)(i % 50) * 1.1;
        y[i] = (double)(i / 50) * 1.1;
    }

    double start = omp_get_wtime();
    const double sigma2 = sigma * sigma;
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        double xi = x[i];
        double yi = y[i];
        double fxi = 0.0;
        double fyi = 0.0;
        for (int j = 0; j < n; ++j) {
            if (i == j) {
                continue;
            }
            double dx = xi - x[j];
            double dy = yi - y[j];
            double r2 = dx * dx + dy * dy;
            if (r2 > cutoff2) {
                continue;
            }
            double inv_r2 = 1.0 / r2;
            double sr2 = sigma2 * inv_r2;
            double sr6 = sr2 * sr2 * sr2;
            double sr12 = sr6 * sr6;
            double f = 24.0 * epsilon * inv_r2 * (2.0 * sr12 - sr6);
            fxi += f * dx;
            fyi += f * dy;
        }
        fx[i] = fxi;
        fy[i] = fyi;
    }
    double stop = omp_get_wtime();
    double cpu_ms = 1000.0 * (stop - start);

    printf("OpenMP Lennard-Jones forces\n");
    printf("n=%d\n", n);
    printf("cpu_time_ms=%.3f\n", cpu_ms);
    printf("fx[0]=%.3e fy[0]=%.3e\n", fx[0], fy[0]);

    free(x);
    free(y);
    free(fx);
    free(fy);
    return 0;
}
