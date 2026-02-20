#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int n = 1 << 20;
    if (argc >= 2) {
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

    for (int i = 0; i < n; i++) {
        x[i] = 1.0 + 0.001 * (double)(i % 1000);
        y[i] = 2.0 - 0.0005 * (double)(i % 1000);
        out[i] = 0.0;
    }

    const double a = 2.5;

    double t0 = omp_get_wtime();
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        out[i] = a * x[i] + y[i];
    }
    double t1 = omp_get_wtime();

    double max_abs_err = 0.0;
    double checksum = 0.0;
    for (int i = 0; i < n; i++) {
        double ref = a * x[i] + y[i];
        double err = fabs(out[i] - ref);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
        checksum += out[i];
    }

    printf("OpenMP host parallel for example (AXPY)\n");
    printf("N=%d\n", n);
    printf("threads=%d\n", omp_get_max_threads());
    printf("elapsed_s=%.6f\n", t1 - t0);
    printf("max_abs_err=%.3e checksum=%.6e\n", max_abs_err, checksum);

    free(x);
    free(y);
    free(out);
    return 0;
}
