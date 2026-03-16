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
    if (!x || !y) {
        fprintf(stderr, "allocation failed\n");
        free(x);
        free(y);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        x[i] = 1.0;
        y[i] = -1.0;
    }

    const double a = 2.0;
    double sum = 0.0;

#pragma acc parallel copy(x[0:n], y[0:n])
    {
#pragma acc loop gang worker
        for (int i = 0; i < n; i++) {
            y[i] = a * x[i] + y[i];
        }

#pragma acc loop independent reduction(+:sum)
        for (int i = 0; i < n; i++) {
            sum += y[i];
        }
    }

    printf("OpenACC loop reduction example\n");
    printf("N=%d sum=%.6f expected=%.6f err=%.3e\n",
           n, sum, (double)n, fabs(sum - (double)n));

    free(x);
    free(y);
    return fabs(sum - (double)n) > 1.0e-10;
}
