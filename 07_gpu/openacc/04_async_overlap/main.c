#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char **argv) {
    int n = 1 << 22;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n < 10) {
        fprintf(stderr, "Usage: %s [N>=10]\n", argv[0]);
        return 1;
    }

    double *x = malloc((size_t)n * sizeof(*x));
    if (!x) {
        fprintf(stderr, "allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n; i++) {
        x[i] = 1.0 / (double)(i + 1);
    }

    int split = n / 8;
    double cpu_sum = 0.0;
    double gpu_sum = 0.0;

    clock_t t0 = clock();
#pragma acc parallel loop copyin(x[split:n - split]) reduction(+:gpu_sum) async(1)
    for (int i = split; i < n; i++) {
        gpu_sum += x[i];
    }

    for (int i = 0; i < split; i++) {
        cpu_sum += x[i];
    }
#pragma acc wait(1)
    clock_t t1 = clock();

    double total = cpu_sum + gpu_sum;
    double ref = 0.0;
    for (int i = 0; i < n; i++) {
        ref += x[i];
    }

    printf("OpenACC async overlap example\n");
    printf("N=%d split=%d total=%.12f ref=%.12f err=%.3e elapsed_ms=%.3f\n",
           n, split, total, ref, fabs(total - ref),
           1000.0 * (double)(t1 - t0) / (double)CLOCKS_PER_SEC);

    free(x);
    return fabs(total - ref) > 1.0e-10;
}
