#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static double checksum(const double *a, int n) {
    double s = 0.0;
#pragma omp simd reduction(+ : s)
    for (int i = 0; i < n; i++) {
        s += a[i];
    }
    return s;
}

int main(int argc, char **argv) {
    int n = 10000000;
    int repeats = 10;

    if (argc >= 2) {
        n = atoi(argv[1]);
    }
    if (argc >= 3) {
        repeats = atoi(argv[2]);
    }

    if (n <= 0 || repeats <= 0) {
        fprintf(stderr, "Usage: %s <N> [repeats]\n", argv[0]);
        fprintf(stderr, "Example: OMP_NUM_THREADS=8 %s 20000000 10\n", argv[0]);
        return 1;
    }

    double *x = (double *)malloc((size_t)n * sizeof(double));
    double *y = (double *)malloc((size_t)n * sizeof(double));
    double *out = (double *)malloc((size_t)n * sizeof(double));
    if (!x || !y || !out) {
        fprintf(stderr, "allocation failed for N=%d\n", n);
        free(x);
        free(y);
        free(out);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        x[i] = 1.0 + 0.001 * (double)(i % 1000);
        y[i] = 2.0 - 0.0005 * (double)(i % 1000);
        out[i] = 0.0;
    }

    const double a = 2.5;

    printf("\nAXPY SIMD + OpenMP Benchmark\n");
    printf("============================\n");
    printf("N: %d\n", n);
    printf("Repeats: %d\n", repeats);
    printf("OMP_NUM_THREADS: %d\n\n", omp_get_max_threads());

    double best_serial = 1e100, total_serial = 0.0, cs_serial = 0.0;
    for (int r = 0; r < repeats; r++) {
        double t0 = omp_get_wtime();
        for (int i = 0; i < n; i++) {
            out[i] = a * x[i] + y[i];
        }
        double t = omp_get_wtime() - t0;
        total_serial += t;
        if (t < best_serial) best_serial = t;
        cs_serial = checksum(out, n);
    }

    double best_simd = 1e100, total_simd = 0.0, cs_simd = 0.0;
    for (int r = 0; r < repeats; r++) {
        double t0 = omp_get_wtime();
#pragma omp simd
        for (int i = 0; i < n; i++) {
            out[i] = a * x[i] + y[i];
        }
        double t = omp_get_wtime() - t0;
        total_simd += t;
        if (t < best_simd) best_simd = t;
        cs_simd = checksum(out, n);
    }

    double best_threads = 1e100, total_threads = 0.0, cs_threads = 0.0;
    for (int r = 0; r < repeats; r++) {
        double t0 = omp_get_wtime();
#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            out[i] = a * x[i] + y[i];
        }
        double t = omp_get_wtime() - t0;
        total_threads += t;
        if (t < best_threads) best_threads = t;
        cs_threads = checksum(out, n);
    }

    double best_both = 1e100, total_both = 0.0, cs_both = 0.0;
    for (int r = 0; r < repeats; r++) {
        double t0 = omp_get_wtime();
#pragma omp parallel for simd
        for (int i = 0; i < n; i++) {
            out[i] = a * x[i] + y[i];
        }
        double t = omp_get_wtime() - t0;
        total_both += t;
        if (t < best_both) best_both = t;
        cs_both = checksum(out, n);
    }

    printf("%-22s %12s %12s %16s\n", "Case", "Best (s)", "Mean (s)", "Checksum");
    printf("%-22s %12s %12s %16s\n", "----------------------", "----------", "----------", "----------------");
    printf("%-22s %12.6f %12.6f %16.6e\n", "Serial", best_serial, total_serial / repeats, cs_serial);
    printf("%-22s %12.6f %12.6f %16.6e\n", "SIMD", best_simd, total_simd / repeats, cs_simd);
    printf("%-22s %12.6f %12.6f %16.6e\n", "OMP parallel for", best_threads, total_threads / repeats, cs_threads);
    printf("%-22s %12.6f %12.6f %16.6e\n", "OMP parallel for simd", best_both, total_both / repeats, cs_both);
    printf("\n");

    free(x);
    free(y);
    free(out);
    return 0;
}
