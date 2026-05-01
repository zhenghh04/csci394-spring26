#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static void fill_inputs(double *a, double *x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0 / (double)(i + 1);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[(size_t)i * (size_t)n + (size_t)j] = 1.0 / (double)(i + j + 1);
        }
    }
}

static void matvec_serial(const double *a, const double *x, double *y, int n) {
    for (int i = 0; i < n; ++i) {
        const double *row = a + (size_t)i * (size_t)n;
        double acc = 0.0;
        for (int j = 0; j < n; ++j) {
            acc += row[j] * x[j];
        }
        y[i] = acc;
    }
}

static double checksum(const double *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += v[i];
    }
    return s;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <n> [iters] [warmup]\n", argv[0]);
        return 1;
    }

    const int n = atoi(argv[1]);
    const int iters = (argc >= 3) ? atoi(argv[2]) : 5;
    const int warmup = (argc >= 4) ? atoi(argv[3]) : 1;

    if (n <= 0 || iters <= 0 || warmup < 0) {
        fprintf(stderr, "Invalid args: n>0, iters>0, warmup>=0 required.\n");
        return 1;
    }

    double *a = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    if (a == NULL || x == NULL || y == NULL) {
        fprintf(stderr, "Failed to allocate buffers for n=%d\n", n);
        free(a);
        free(x);
        free(y);
        return 2;
    }

    fill_inputs(a, x, n);

    for (int w = 0; w < warmup; ++w) {
        matvec_serial(a, x, y, n);
    }

    const double t0 = now_seconds();
    for (int k = 0; k < iters; ++k) {
        matvec_serial(a, x, y, n);
    }
    const double elapsed = now_seconds() - t0;

    printf(
        "MATVEC_SERIES n=%d p=1 rows_local=%d iters=%d warmup=%d time_s=%.9f time_per_iter_s=%.9f checksum=%.12e\n",
        n,
        n,
        iters,
        warmup,
        elapsed,
        elapsed / (double)iters,
        checksum(y, n));

    free(a);
    free(x);
    free(y);
    return 0;
}
