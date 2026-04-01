#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static inline int chunk_begin(int dev, int num_chunks, int n) {
    return (dev * n) / num_chunks;
}

static inline int chunk_end(int dev, int num_chunks, int n) {
    return ((dev + 1) * n) / num_chunks;
}

int main(int argc, char **argv) {
    int n = 1 << 20;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (n <= 0) {
        fprintf(stderr, "Usage: %s [N>0]\n", argv[0]);
        return 1;
    }

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    double *out = (double *)malloc((size_t)n * sizeof(double));
    if (!x || !y || !out) {
        fprintf(stderr, "allocation failed\n");
        free(x);
        free(y);
        free(out);
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        x[i] = 1.0 + 0.001 * (double)(i % 1000);
        y[i] = 2.0 - 0.0005 * (double)(i % 1000);
        out[i] = 0.0;
    }

    const double a = 2.5;
    int num_devices = omp_get_num_devices();
    int num_chunks = (num_devices > 0) ? num_devices : 1;

    double t0 = omp_get_wtime();
    for (int dev = 0; dev < num_chunks; ++dev) {
        int begin = chunk_begin(dev, num_chunks, n);
        int end = chunk_end(dev, num_chunks, n);
        int len = end - begin;
        if (len <= 0) {
            continue;
        }

#pragma omp target enter data device(dev) map(alloc : x[begin:len], y[begin:len], out[begin:len])
#pragma omp target update to(x[begin:len], y[begin:len]) device(dev)
#pragma omp target teams distribute parallel for device(dev) map(present, to : x[begin:len], y[begin:len]) map(present, from : out[begin:len])
        for (int i = begin; i < end; ++i) {
            out[i] = a * x[i] + y[i];
        }
#pragma omp target update from(out[begin:len]) device(dev)
#pragma omp target exit data device(dev) map(delete : x[begin:len], y[begin:len], out[begin:len])
    }
    double t1 = omp_get_wtime();

    double max_abs_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = a * x[i] + y[i];
        double err = fabs(out[i] - ref);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
    }

    printf("OpenMP target multi-GPU AXPY\n");
    printf("N=%d num_devices=%d chunks_used=%d\n", n, num_devices, num_chunks);
    printf("time_s=%.6f max_abs_err=%.3e\n", t1 - t0, max_abs_err);
    if (num_chunks > 1) {
        for (int dev = 0; dev < num_chunks; ++dev) {
            printf("device_%d_range=[%d,%d)\n",
                   dev, chunk_begin(dev, num_chunks, n), chunk_end(dev, num_chunks, n));
        }
    } else {
        printf("single target device or host fallback path used\n");
    }

    free(x);
    free(y);
    free(out);
    return 0;
}
