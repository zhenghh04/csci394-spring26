#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    int n = 1 << 20;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n < 3) {
        fprintf(stderr, "Usage: %s [N>=3]\n", argv[0]);
        return 1;
    }

    size_t bytes = (size_t)n * sizeof(double);
    double *in = (double *)malloc(bytes);
    double *out = (double *)calloc((size_t)n, sizeof(double));

    for (int i = 0; i < n; ++i) {
        in[i] = sin(0.001 * (double)i);
    }

    clock_t start = clock();
    for (int i = 1; i < n - 1; ++i) {
        out[i] = in[i - 1] + 2.0 * in[i] + in[i + 1];
    }
    clock_t stop = clock();

    double cpu_ms = 1000.0 * (double)(stop - start) / (double)CLOCKS_PER_SEC;
    printf("CPU three-point stencil\n");
    printf("N=%d\n", n);
    printf("cpu_time_ms=%.3f\n", cpu_ms);
    printf("out[1]=%.6f out[n/2]=%.6f out[n-2]=%.6f\n",
           out[1], out[n / 2], out[n - 2]);

    free(in);
    free(out);
    return 0;
}
