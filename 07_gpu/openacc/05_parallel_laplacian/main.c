#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static inline size_t idx(int i, int j, int n) {
    return (size_t)i * (size_t)n + (size_t)j;
}

static double seconds_now(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

int main(int argc, char **argv) {
    int n = 2048;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n < 8) {
        fprintf(stderr, "Usage: %s [N>=8]\n", argv[0]);
        return 1;
    }

    size_t nn = (size_t)n * (size_t)n;
    double *u = (double *)malloc(nn * sizeof(double));
    double *lap = (double *)calloc(nn, sizeof(double));
    if (!u || !lap) {
        fprintf(stderr, "allocation failed\n");
        free(u);
        free(lap);
        return 1;
    }

    for (size_t k = 0; k < nn; ++k) {
        u[k] = 0.01 * (double)(k % 100);
    }

    double t0 = seconds_now();
#pragma acc parallel loop collapse(2) copyin(u[0:nn]) copyout(lap[0:nn])
    for (int i = 1; i < n - 1; ++i) {
        for (int j = 1; j < n - 1; ++j) {
            lap[idx(i, j, n)] =
                u[idx(i - 1, j, n)] + u[idx(i + 1, j, n)] +
                u[idx(i, j - 1, n)] + u[idx(i, j + 1, n)] -
                4.0 * u[idx(i, j, n)];
        }
    }
    double t1 = seconds_now();

    printf("OpenACC 2D Laplacian\n");
    printf("N=%d time_s=%.6f\n", n, t1 - t0);
    printf("lap[1,1]=%.6f lap[center]=%.6f\n", lap[idx(1, 1, n)], lap[idx(n / 2, n / 2, n)]);

    free(u);
    free(lap);
    return 0;
}
