#include <math.h>
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

#pragma acc parallel loop copyin(x[0:n], y[0:n]) copyout(out[0:n])
    for (int i = 0; i < n; i++) {
        out[i] = a * x[i] + y[i];
    }

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

    printf("OpenACC parallel loop SAXPY\n");
    printf("N=%d checksum=%.6f max_abs_err=%.3e\n", n, checksum, max_abs_err);

    free(x);
    free(y);
    free(out);
    return max_abs_err > 1.0e-12;
}
