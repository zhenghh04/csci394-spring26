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
    double *tmp = malloc((size_t)n * sizeof(*tmp));
    double *out = malloc((size_t)n * sizeof(*out));
    if (!x || !y || !tmp || !out) {
        fprintf(stderr, "allocation failed\n");
        free(x);
        free(y);
        free(tmp);
        free(out);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        x[i] = 0.5 + 0.0001 * (double)(i % 1000);
        y[i] = 1.5 - 0.0002 * (double)(i % 1000);
        tmp[i] = -999.0;
        out[i] = 0.0;
    }

#pragma acc data copyin(x[0:n], y[0:n]) create(tmp[0:n]) copyout(out[0:n])
    {
#pragma acc parallel loop
        for (int i = 0; i < n; i++) {
            tmp[i] = x[i] + y[i];
        }

#pragma acc parallel loop
        for (int i = 0; i < n; i++) {
            out[i] = 3.0 * tmp[i];
        }
    }

    double checksum = 0.0;
    double max_abs_err = 0.0;
    double tmp_checksum = 0.0;
    for (int i = 0; i < n; i++) {
        double ref_tmp = x[i] + y[i];
        double ref_out = 3.0 * ref_tmp;
        double err = fabs(out[i] - ref_out);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
        checksum += out[i];
        tmp_checksum += tmp[i];
    }

    printf("OpenACC data movement example\n");
    printf("N=%d checksum=%.6f max_abs_err=%.3e host_tmp_checksum=%.1f\n",
           n, checksum, max_abs_err, tmp_checksum);

    free(x);
    free(y);
    free(tmp);
    free(out);
    return max_abs_err > 1.0e-12;
}
