#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int n = 512;
    if (argc >= 2) {
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
        fprintf(stderr, "allocation failed for N=%d\n", n);
        free(a);
        free(b);
        free(c);
        return 1;
    }

    for (size_t idx = 0; idx < nn; idx++) {
        a[idx] = 1.0 + 0.0001 * (double)(idx % 1000);
        b[idx] = 0.5 + 0.0002 * (double)((idx * 7) % 1000);
        c[idx] = 0.0;
    }

    int num_devices = omp_get_num_devices();
    int default_device = omp_get_default_device();

    double t_total0 = omp_get_wtime();

    double t_h2d0 = omp_get_wtime();
#pragma omp target enter data map(alloc : a[0:nn], b[0:nn], c[0:nn])
#pragma omp target update to(a[0:nn], b[0:nn])
    double t_h2d1 = omp_get_wtime();

    double t_kernel0 = omp_get_wtime();
#pragma omp target teams distribute parallel for collapse(2) map(present, to : a[0:nn], b[0:nn]) map(present, from : c[0:nn])
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double sum = 0.0;
            for (int k = 0; k < n; k++) {
                sum += a[(size_t)i * n + k] * b[(size_t)k * n + j];
            }
            c[(size_t)i * n + j] = sum;
        }
    }
    double t_kernel1 = omp_get_wtime();

    double t_d2h0 = omp_get_wtime();
#pragma omp target update from(c[0:nn])
    double t_d2h1 = omp_get_wtime();

#pragma omp target exit data map(delete : a[0:nn], b[0:nn], c[0:nn])

    double t_total1 = omp_get_wtime();

    /* Host reference for validation (outside offload timing). */
    double max_abs_err = 0.0;
    double checksum = 0.0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            double ref = 0.0;
            for (int k = 0; k < n; k++) {
                ref += a[(size_t)i * n + k] * b[(size_t)k * n + j];
            }
            double got = c[(size_t)i * n + j];
            double err = fabs(got - ref);
            if (err > max_abs_err) {
                max_abs_err = err;
            }
            checksum += got;
        }
    }

    printf("OpenMP target offload example (MatMul C=A*B)\n");
    printf("N=%d\n", n);
    printf("num_devices=%d default_device=%d\n", num_devices, default_device);
    printf("h2d_s=%.6f kernel_s=%.6f d2h_s=%.6f total_s=%.6f\n",
           t_h2d1 - t_h2d0, t_kernel1 - t_kernel0, t_d2h1 - t_d2h0, t_total1 - t_total0);
    printf("max_abs_err=%.3e checksum=%.6e\n", max_abs_err, checksum);

    free(a);
    free(b);
    free(c);
    return 0;
}
