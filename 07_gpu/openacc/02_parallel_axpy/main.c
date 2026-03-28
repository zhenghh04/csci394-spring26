#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double seconds_now(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

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

    for (int i = 0; i < n; ++i) {
        x[i] = 1.0 + 0.001 * (double)(i % 1000);
        y[i] = 2.0 - 0.0005 * (double)(i % 1000);
        out[i] = 0.0;
    }

    const double a = 2.5;
    double t0 = seconds_now();
#pragma acc parallel loop copyin(x[0:n], y[0:n]) copyout(out[0:n])
    for (int i = 0; i < n; ++i) {
        out[i] = a * x[i] + y[i];
    }
    double t1 = seconds_now();

    double max_abs_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = a * x[i] + y[i];
        double err = fabs(out[i] - ref);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
    }

    printf("OpenACC parallel AXPY\n");
    printf("N=%d time_s=%.6f max_abs_err=%.3e\n", n, t1 - t0, max_abs_err);

    free(x);
    free(y);
    free(out);
    return 0;
}
