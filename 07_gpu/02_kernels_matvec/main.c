#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static double matrix_value(int row, int col) {
    return (col >= row) ? 1.0 : 0.0;
}

int main(int argc, char **argv) {
    int n = 512;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n <= 0) {
        fprintf(stderr, "Usage: %s [N>0]\n", argv[0]);
        return 1;
    }

    double *matrix = malloc((size_t)n * (size_t)n * sizeof(*matrix));
    double *x = malloc((size_t)n * sizeof(*x));
    double *y = malloc((size_t)n * sizeof(*y));
    if (!matrix || !x || !y) {
        fprintf(stderr, "allocation failed\n");
        free(matrix);
        free(x);
        free(y);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        x[i] = 1.0;
        y[i] = 0.0;
        for (int j = 0; j < n; j++) {
            matrix[(size_t)i * (size_t)n + (size_t)j] = matrix_value(i, j);
        }
    }

#pragma acc kernels copyin(matrix[0:n * n], x[0:n]) copyout(y[0:n])
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            y[i] += matrix[(size_t)i * (size_t)n + (size_t)j] * x[j];
        }
    }

    double max_abs_err = 0.0;
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        double ref = (double)(n - i);
        double err = fabs(y[i] - ref);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
        total += y[i];
    }

    printf("OpenACC kernels matrix-vector multiply\n");
    printf("N=%d total=%.1f expected_total=%.1f max_abs_err=%.3e\n",
           n, total, (double)n * (double)(n + 1) / 2.0, max_abs_err);

    free(matrix);
    free(x);
    free(y);
    return max_abs_err > 1.0e-12;
}
