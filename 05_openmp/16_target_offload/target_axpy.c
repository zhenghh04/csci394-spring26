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
    int num_devices = omp_get_num_devices();
    int default_device = omp_get_default_device();

    double t_total0 = omp_get_wtime();

    double t_h2d0 = omp_get_wtime();
#pragma omp target enter data map(alloc : x[0:n], y[0:n], out[0:n])
#pragma omp target update to(x[0:n], y[0:n])
    double t_h2d1 = omp_get_wtime();

    double t_kernel0 = omp_get_wtime();
#pragma omp target teams distribute parallel for map(present, to : x[0:n], y[0:n]) map(present, from : out[0:n])
    for (int i = 0; i < n; i++) {
        out[i] = a * x[i] + y[i];
    }
    double t_kernel1 = omp_get_wtime();

    double t_d2h0 = omp_get_wtime();
#pragma omp target update from(out[0:n])
    double t_d2h1 = omp_get_wtime();

#pragma omp target exit data map(delete : x[0:n], y[0:n], out[0:n])

    double t_total1 = omp_get_wtime();

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

    printf("OpenMP target offload example (AXPY)\n");
    printf("N=%d\n", n);
    printf("num_devices=%d default_device=%d\n", num_devices, default_device);
    printf("h2d_s=%.6f kernel_s=%.6f d2h_s=%.6f total_s=%.6f\n",
           t_h2d1 - t_h2d0, t_kernel1 - t_kernel0, t_d2h1 - t_d2h0, t_total1 - t_total0);
    printf("max_abs_err=%.3e checksum=%.6e\n", max_abs_err, checksum);

    free(x);
    free(y);
    free(out);
    return 0;
}
