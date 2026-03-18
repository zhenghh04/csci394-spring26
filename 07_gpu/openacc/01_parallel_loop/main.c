#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1.0e-9 * (double)ts.tv_nsec;
}

int main(int argc, char **argv) {
    const double total_t0 = now_seconds();

    int n = 1 << 20;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n <= 0) {
        fprintf(stderr, "Usage: %s [N>0]\n", argv[0]);
        return 1;
    }

    double *x = malloc((size_t)n * sizeof(*x));
    double *y = malloc((size_t)n * sizeof(*y));
    double *out = malloc((size_t)n * sizeof(*out));
    if (!x || !y || !out) {
        fprintf(stderr, "allocation failed\n");
        free(x);
        free(y);
        free(out);
        return 1;
    }

    const double a = 2.5;
    for (int i = 0; i < n; i++) {
        x[i] = 1.0 + 0.001 * (double)(i % 100);
        y[i] = 2.0 - 0.002 * (double)(i % 100);
    }

    const double kernel_t0 = now_seconds();
#pragma acc parallel loop copyin(x[0:n], y[0:n]) copyout(out[0:n])
    for (int i = 0; i < n; i++) {
        out[i] = a * x[i] + y[i];
    }
    const double kernel_t1 = now_seconds();

    double checksum = 0.0;
    double max_abs_err = 0.0;
    for (int i = 0; i < n; i++) {
        double ref = a * x[i] + y[i];
        double err = fabs(out[i] - ref);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
        checksum += out[i];
    }
    const double total_t1 = now_seconds();

    printf("OpenACC parallel loop SAXPY\n");
    printf("N=%d checksum=%.6f max_abs_err=%.3e\n", n, checksum, max_abs_err);
    printf("kernel_plus_transfer_time=%.6f s total_time=%.6f s\n",
           kernel_t1 - kernel_t0, total_t1 - total_t0);

    free(x);
    free(y);
    free(out);
    return max_abs_err > 1.0e-12;
}
