#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <openacc.h>

#include "common.h"

static double wtime(void) {
    struct timeval tv; gettimeofday(&tv, NULL);
    return (double)tv.tv_sec + 1.0e-6 * (double)tv.tv_usec;
}

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
        #pragma acc parallel loop collapse(2) copyin(a[0:nn], b[0:nn]) copyout(c[0:nn])
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float s = 0.0f;
                for (int k = 0; k < n; k++) s += a[(size_t)i*n+k] * b[(size_t)k*n+j];
                c[(size_t)i*n+j] = s;
            }
        }
    }

    double sum_e2e = 0.0, sum_compute = 0.0, sum_h2d = 0.0, sum_d2h = 0.0;
    for (int it = 0; it < iters; it++) {
        memset(c, 0, nn * sizeof(float));
        double t0 = wtime();
        double th0 = wtime();
        #pragma acc enter data copyin(a[0:nn], b[0:nn]) create(c[0:nn])
        double th1 = wtime();
        double tk0 = wtime();
        #pragma acc parallel loop collapse(2) present(a[0:nn], b[0:nn], c[0:nn])
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                float s = 0.0f;
                for (int k = 0; k < n; k++) s += a[(size_t)i*n+k] * b[(size_t)k*n+j];
                c[(size_t)i*n+j] = s;
            }
        }
        double tk1 = wtime();
        double td0 = wtime();
        #pragma acc exit data copyout(c[0:nn]) delete(a[0:nn], b[0:nn])
        double td1 = wtime();
        double t1 = wtime();
        sum_e2e += (t1 - t0);
        sum_compute += (tk1 - tk0);
        sum_h2d += (th1 - th0);
        sum_d2h += (td1 - td0);
    }

    float err = cpu_reference_max_err(a, b, c, n);
    printf("RESULT,openacc,%d,%d,%d,%.6e,%.6e,%.3e\n",
           n, iters, warmup, sum_e2e/iters, sum_compute/iters, err);
    printf("DETAIL,openacc,%d,h2d_s=%.6e,d2h_s=%.6e\n",
           n, sum_h2d/iters, sum_d2h/iters);

    free(a); free(b); free(c);
    return 0;
}
