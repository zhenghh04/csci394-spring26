#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static inline size_t idx(int i, int j, int n) {
    return (size_t)i * (size_t)n + (size_t)j;
}

int main(int argc, char **argv) {
    int n = 1024;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n <= 0) {
        fprintf(stderr, "Usage: %s [N>0]\n", argv[0]);
        return 1;
    }

    size_t nn = (size_t)n * (size_t)n;
    double *a = (double *)malloc(nn * sizeof(double));
    double *b = (double *)malloc(nn * sizeof(double));
    double *c = (double *)malloc(nn * sizeof(double));
    if (!a || !b || !c) {
        fprintf(stderr, "allocation failed\n");
        free(a);
        free(b);
        free(c);
        return 1;
    }

    for (size_t k = 0; k < nn; ++k) {
        a[k] = 1.0 + 0.001 * (double)(k % 100);
        b[k] = 2.0 - 0.0005 * (double)(k % 100);
        c[k] = 0.0;
    }

    double t0 = omp_get_wtime();
#pragma omp target teams distribute parallel for collapse(2) map(to : a[0:nn], b[0:nn]) map(from : c[0:nn])
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            c[idx(i, j, n)] = a[idx(i, j, n)] + b[idx(i, j, n)];
        }
    }
    double t1 = omp_get_wtime();

    double max_abs_err = 0.0;
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            double ref = a[idx(i, j, n)] + b[idx(i, j, n)];
            double err = fabs(c[idx(i, j, n)] - ref);
            if (err > max_abs_err) {
                max_abs_err = err;
            }
        }
    }

    printf("OpenMP target collapse(2) matrix add\n");
    printf("N=%d time_s=%.6f max_abs_err=%.3e\n", n, t1 - t0, max_abs_err);
    printf("c[0]=%.6f c[center]=%.6f\n", c[0], c[idx(n / 2, n / 2, n)]);

    free(a);
    free(b);
    free(c);
    return 0;
}
