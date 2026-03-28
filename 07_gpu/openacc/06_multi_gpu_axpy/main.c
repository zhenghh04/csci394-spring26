#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifdef _OPENACC
#include <openacc.h>
#endif

static inline int chunk_begin(int dev, int num_chunks, int n) {
    return (dev * n) / num_chunks;
}

static inline int chunk_end(int dev, int num_chunks, int n) {
    return ((dev + 1) * n) / num_chunks;
}

static double seconds_now(void) {
    return (double)clock() / (double)CLOCKS_PER_SEC;
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
    int num_devices = 1;
#ifdef _OPENACC
    num_devices = acc_get_num_devices(acc_device_default);
    if (num_devices <= 0) {
        num_devices = 1;
    }
#endif

    double t0 = seconds_now();
    for (int dev = 0; dev < num_devices; ++dev) {
        int begin = chunk_begin(dev, num_devices, n);
        int end = chunk_end(dev, num_devices, n);
        int len = end - begin;
        if (len <= 0) {
            continue;
        }
#ifdef _OPENACC
        acc_set_device_num(dev, acc_device_default);
#endif
#pragma acc data copyin(x[begin:len], y[begin:len]) copyout(out[begin:len])
        {
#pragma acc parallel loop present(x[begin:len], y[begin:len], out[begin:len])
            for (int i = begin; i < end; ++i) {
                out[i] = a * x[i] + y[i];
            }
        }
    }
    double t1 = seconds_now();

    double max_abs_err = 0.0;
    for (int i = 0; i < n; ++i) {
        double ref = a * x[i] + y[i];
        double err = fabs(out[i] - ref);
        if (err > max_abs_err) {
            max_abs_err = err;
        }
    }

    printf("OpenACC multi-GPU AXPY\n");
    printf("N=%d num_devices=%d time_s=%.6f max_abs_err=%.3e\n",
           n, num_devices, t1 - t0, max_abs_err);
    for (int dev = 0; dev < num_devices; ++dev) {
        printf("device_%d_range=[%d,%d)\n",
               dev, chunk_begin(dev, num_devices, n), chunk_end(dev, num_devices, n));
    }

    free(x);
    free(y);
    free(out);
    return 0;
}
