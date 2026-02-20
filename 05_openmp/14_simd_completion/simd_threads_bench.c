#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double checksum(const double *a, int n) {
    double s = 0.0;
#pragma omp simd reduction(+ : s)
    for (int i = 0; i < n; i++) {
        s += a[i];
    }
    return s;
}

int main(int argc, char **argv) {
    int n = 512;
    int repeats = 3;

    if (argc >= 2) {
        n = atoi(argv[1]);
    }
    if (argc >= 3) {
        repeats = atoi(argv[2]);
    }

    if (n <= 0 || repeats <= 0) {
        fprintf(stderr, "Usage: %s <N> [repeats]\n", argv[0]);
        fprintf(stderr, "Example: OMP_NUM_THREADS=8 %s 1024 3\n", argv[0]);
        return 1;
    }

    size_t nn = (size_t)n * (size_t)n;
    double *a = (double *)malloc(nn * sizeof(double));
    double *b = (double *)malloc(nn * sizeof(double));
    double *c = (double *)malloc(nn * sizeof(double));
    if (!a || !b || !c) {
        fprintf(stderr, "allocation failed for N=%d (N*N=%zu)\n", n, nn);
        free(a);
        free(b);
        free(c);
        return 1;
    }

    for (size_t idx = 0; idx < nn; idx++) {
        a[idx] = 1.0 + 0.0001 * (double)(idx % 1000);
        b[idx] = 0.5 + 0.0002 * (double)((idx * 7) % 1000);
        c[idx] = 0.0;
    }

    printf("\nSIMD + OpenMP Interplay Benchmark\n");
    printf("=================================\n");
    printf("Kernel: C = A x B (dense GEMM, N x N)\n");
    printf("N: %d (matrix size)\n", n);
    printf("Repeats: %d\n", repeats);
    printf("OMP_NUM_THREADS: %d\n\n", omp_get_max_threads());

    double best_serial = 1e100, total_serial = 0.0, cs_serial = 0.0;
    for (int r = 0; r < repeats; r++) {
        memset(c, 0, nn * sizeof(double));
        double t0 = omp_get_wtime();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += a[(size_t)i * n + k] * b[(size_t)k * n + j];
                }
                c[(size_t)i * n + j] = sum;
            }
        }
        double t = omp_get_wtime() - t0;
        total_serial += t;
        if (t < best_serial) best_serial = t;
        cs_serial = checksum(c, (int)nn);
    }

    double best_simd = 1e100, total_simd = 0.0, cs_simd = 0.0;
    for (int r = 0; r < repeats; r++) {
        memset(c, 0, nn * sizeof(double));
        double t0 = omp_get_wtime();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                for (int k = 0; k < n; k++) {
                    sum += a[(size_t)i * n + k] * b[(size_t)k * n + j];
                }
                c[(size_t)i * n + j] = sum;
            }
        }
        double t = omp_get_wtime() - t0;
        total_simd += t;
        if (t < best_simd) best_simd = t;
        cs_simd = checksum(c, (int)nn);
    }

    double best_threads = 1e100, total_threads = 0.0, cs_threads = 0.0;
    for (int r = 0; r < repeats; r++) {
        memset(c, 0, nn * sizeof(double));
        double t0 = omp_get_wtime();
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int k = 0; k < n; k++) {
                    sum += a[(size_t)i * n + k] * b[(size_t)k * n + j];
                }
                c[(size_t)i * n + j] = sum;
            }
        }
        double t = omp_get_wtime() - t0;
        total_threads += t;
        if (t < best_threads) best_threads = t;
        cs_threads = checksum(c, (int)nn);
    }

    double best_both = 1e100, total_both = 0.0, cs_both = 0.0;
    for (int r = 0; r < repeats; r++) {
        memset(c, 0, nn * sizeof(double));
        double t0 = omp_get_wtime();
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
#pragma omp simd reduction(+ : sum)
                for (int k = 0; k < n; k++) {
                    sum += a[(size_t)i * n + k] * b[(size_t)k * n + j];
                }
                c[(size_t)i * n + j] = sum;
            }
        }
        double t = omp_get_wtime() - t0;
        total_both += t;
        if (t < best_both) best_both = t;
        cs_both = checksum(c, (int)nn);
    }

    printf("%-22s %12s %12s %16s\n", "Case", "Best (s)", "Mean (s)", "Checksum");
    printf("%-22s %12s %12s %16s\n", "----------------------", "----------", "----------", "----------------");
    printf("%-22s %12.6f %12.6f %16.6e\n", "Serial", best_serial, total_serial / repeats, cs_serial);
    printf("%-22s %12.6f %12.6f %16.6e\n", "SIMD", best_simd, total_simd / repeats, cs_simd);
    printf("%-22s %12.6f %12.6f %16.6e\n", "OMP parallel for", best_threads, total_threads / repeats, cs_threads);
    printf("%-22s %12.6f %12.6f %16.6e\n", "OMP parallel for simd", best_both, total_both / repeats, cs_both);
    printf("\n");

    free(a);
    free(b);
    free(c);
    return 0;
}
