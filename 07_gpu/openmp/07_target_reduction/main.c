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
    if (!x) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        x[i] = 1.0 / (double)(i + 1);
    }

    double sum_gpu = 0.0;
    double t0 = omp_get_wtime();
#pragma omp target teams distribute parallel for map(to : x[0:n]) reduction(+ : sum_gpu)
    for (int i = 0; i < n; ++i) {
        sum_gpu += x[i] * x[i];
    }
    double t1 = omp_get_wtime();

    double sum_cpu = 0.0;
    for (int i = 0; i < n; ++i) {
        sum_cpu += x[i] * x[i];
    }

    double abs_err = fabs(sum_gpu - sum_cpu);
    double rel_err = abs_err / (fabs(sum_cpu) + 1.0e-30);

    printf("OpenMP target reduction\n");
    printf("N=%d time_s=%.6f\n", n, t1 - t0);
    printf("sum_gpu=%.12f sum_cpu=%.12f abs_err=%.3e rel_err=%.3e\n",
           sum_gpu, sum_cpu, abs_err, rel_err);

    free(x);
    return rel_err > 1.0e-12;
}
