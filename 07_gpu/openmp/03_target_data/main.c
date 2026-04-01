#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int n = 1 << 20;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n <= 0) {
        fprintf(stderr, "Usage: %s [N>0]\n", argv[0]);
        return 1;
    }

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    double *out = (double *)malloc((size_t)n * sizeof(double));
    if (!x || !y || !out) {
        fprintf(stderr, "allocation failed\n");
        free(x);
        free(y);
        free(out);
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        x[i] = sin(0.001 * (double)i);
        y[i] = cos(0.001 * (double)i);
        out[i] = 0.0;
    }

    double t_total0 = omp_get_wtime();
    double t_h2d0 = omp_get_wtime();
#pragma omp target enter data map(alloc : x[0:n], y[0:n], out[0:n])
#pragma omp target update to(x[0:n], y[0:n])
    double t_h2d1 = omp_get_wtime();

    double t_kernel0 = omp_get_wtime();
#pragma omp target teams distribute parallel for map(present, to : x[0:n], y[0:n]) map(present, from : out[0:n])
    for (int i = 0; i < n; ++i) {
        out[i] = x[i] * x[i] + y[i] * y[i];
    }
    double t_kernel1 = omp_get_wtime();

    double t_d2h0 = omp_get_wtime();
#pragma omp target update from(out[0:n])
    double t_d2h1 = omp_get_wtime();
#pragma omp target exit data map(delete : x[0:n], y[0:n], out[0:n])
    double t_total1 = omp_get_wtime();

    double max_abs_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double err = fabs(out[i] - 1.0);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
    }

    printf("OpenMP target explicit data region\n");
    printf("N=%d\n", n);
    printf("h2d_s=%.6f kernel_s=%.6f d2h_s=%.6f total_s=%.6f\n",
           t_h2d1 - t_h2d0, t_kernel1 - t_kernel0, t_d2h1 - t_d2h0, t_total1 - t_total0);
    printf("max_abs_err=%.3e out[0]=%.6f out[n/2]=%.6f\n",
           max_abs_err, out[0], out[n / 2]);

    free(x);
    free(y);
    free(out);
    return 0;
}
