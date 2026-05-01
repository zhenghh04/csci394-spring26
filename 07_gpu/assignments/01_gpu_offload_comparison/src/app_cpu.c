#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "common.h"

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 512;
    int iters = (argc > 2) ? atoi(argv[2]) : 5;
    int warmup = (argc > 3) ? atoi(argv[3]) : 1;
    if (n <= 0 || iters <= 0) { fprintf(stderr, "bad args\n"); return 1; }

    size_t nn = (size_t)n * n;
    float *a = (float *)malloc(nn * sizeof(float));
    float *b = (float *)malloc(nn * sizeof(float));
    float *c = (float *)malloc(nn * sizeof(float));
    if (!a || !b || !c) { fprintf(stderr, "alloc fail\n"); return 1; }
    init_matrices(a, b, c, n);

    /* warmup */
    for (int w = 0; w < warmup; w++) {
        memset(c, 0, nn * sizeof(float));
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float s = 0.0f;
                for (int k = 0; k < n; k++) s += a[(size_t)i*n+k] * b[(size_t)k*n+j];
                c[(size_t)i*n+j] = s;
            }
        }
    }

    double sum_e2e = 0.0, sum_compute = 0.0;
    for (int it = 0; it < iters; it++) {
        memset(c, 0, nn * sizeof(float));
        double t0 = omp_get_wtime();
        double tk0 = omp_get_wtime();
        #pragma omp parallel for collapse(2) schedule(static)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float s = 0.0f;
                for (int k = 0; k < n; k++) s += a[(size_t)i*n+k] * b[(size_t)k*n+j];
                c[(size_t)i*n+j] = s;
            }
        }
        double tk1 = omp_get_wtime();
        double t1 = omp_get_wtime();
        sum_e2e += (t1 - t0);
        sum_compute += (tk1 - tk0);
    }

    float err = cpu_reference_max_err(a, b, c, n);
    double e2e = sum_e2e / iters;
    double compute = sum_compute / iters;
    /* CSV: version,n,iters,warmup,end_to_end_s,compute_s,max_abs_err */
    printf("RESULT,cpu,%d,%d,%d,%.6e,%.6e,%.3e\n",
           n, iters, warmup, e2e, compute, err);

    free(a); free(b); free(c);
    return 0;
}
