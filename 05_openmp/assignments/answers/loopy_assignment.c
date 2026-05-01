/* loopy_assignment.c  (OpenMP parallel version)
 *
 * Tasks (each timed with omp_get_wtime()):
 *   1. Matrix initialization
 *   2. Row sum calculation
 *   3. 5-point stencil (timed twice: without collapse(2), with collapse(2))
 *   4. Total sum reduction
 *
 * Numerical output format is unchanged from the serial version so 1-thread
 * results can be diff'd against any thread count for correctness.
 *
 * Usage: ./loopy_assignment --dim <N> [--output <file>] [--csv <file>]
 *   --csv  appends a per-task timing row (threads,N,t1,t2,t3a,t3b,t4)
 */

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char* prog) {
  fprintf(stderr, "Usage: %s --dim <N> [--output <file>] [--csv <file>]\n", prog);
  fprintf(stderr, "Example: %s --dim 2048 --output loopy.dat --csv timings.csv\n", prog);
}

static void writeResults(const char* filename,
                         double sampleA,
                         double sampleRowSum,
                         double totalSum) {
  FILE* f = fopen(filename, "w");
  if (!f) {
    fprintf(stderr, "Error: cannot open output file %s\n", filename);
    exit(1);
  }
  fprintf(f, "Sample Matrix[500][500]: %.6f\n", sampleA);
  fprintf(f, "Sample RowSums[500]:     %.6f\n", sampleRowSum);
  fprintf(f, "Total Sum:              %.6f\n", totalSum);
  fclose(f);
}

int main(int argc, char** argv) {
  int N = 2048;
  const char* outFile = "loopy.dat";
  const char* csvFile = NULL;

  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--dim") == 0 && i + 1 < argc) {
      N = atoi(argv[++i]);
    } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
      outFile = argv[++i];
    } else if (strcmp(argv[i], "--csv") == 0 && i + 1 < argc) {
      csvFile = argv[++i];
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "Unknown or incomplete argument: %s\n", argv[i]);
      usage(argv[0]);
      return 1;
    }
  }

  if (N < 8) {
    fprintf(stderr, "Error: --dim must be >= 8\n");
    return 1;
  }

  int NN = N * N;

  double* A = (double*)malloc((long long)NN * sizeof(double));
  double* B = (double*)malloc((long long)NN * sizeof(double));
  double* C = (double*)malloc((long long)NN * sizeof(double)); /* second stencil output */
  double* rowSums = (double*)malloc((long long)N * sizeof(double));

  if (!A || !B || !C || !rowSums) {
    fprintf(stderr, "Error: malloc failed (try smaller --dim)\n");
    free(A); free(B); free(C); free(rowSums);
    return 1;
  }

  int nthreads = 1;
  #pragma omp parallel
  {
    #pragma omp single
    nthreads = omp_get_num_threads();
  }

  /* -----------------------------
   * Task 1: Matrix Initialization
   * ----------------------------- */
  double t0 = omp_get_wtime();
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i * N + j] = 0.001 * i + 0.002 * j;
    }
  }
  double t1 = omp_get_wtime();

  /* -----------------------------
   * Task 2: Row Sum Calculation
   * (each row independent; sum is a per-row private accumulator)
   * ----------------------------- */
  double t2_start = omp_get_wtime();
  #pragma omp parallel for schedule(static)
  for (int i = 0; i < N; ++i) {
    double sum = 0.0;
    for (int j = 0; j < N; ++j) {
      sum += A[i * N + j];
    }
    rowSums[i] = sum;
  }
  double t2_end = omp_get_wtime();

  /* -----------------------------
   * Task 3: 5-point stencil
   * (run twice: once without collapse, once with collapse(2))
   * ----------------------------- */

  /* Boundaries (cheap; not timed separately) — fill B and C identically. */
  for (int i = 0; i < N; ++i) {
    B[i * N + 0]       = A[i * N + 0];
    B[i * N + (N - 1)] = A[i * N + (N - 1)];
    C[i * N + 0]       = A[i * N + 0];
    C[i * N + (N - 1)] = A[i * N + (N - 1)];
  }
  for (int j = 0; j < N; ++j) {
    B[0 * N + j]       = A[0 * N + j];
    B[(N - 1) * N + j] = A[(N - 1) * N + j];
    C[0 * N + j]       = A[0 * N + j];
    C[(N - 1) * N + j] = A[(N - 1) * N + j];
  }

  /* 3a: parallel for on outer loop only (no collapse) */
  double t3a_start = omp_get_wtime();
  #pragma omp parallel for schedule(static)
  for (int i = 1; i < N - 1; ++i) {
    for (int j = 1; j < N - 1; ++j) {
      B[i * N + j] =
          0.25 * (A[(i - 1) * N + j] +
                  A[(i + 1) * N + j] +
                  A[i * N + (j - 1)] +
                  A[i * N + (j + 1)]) -
          A[i * N + j];
    }
  }
  double t3a_end = omp_get_wtime();

  /* 3b: collapse(2) over both spatial loops */
  double t3b_start = omp_get_wtime();
  #pragma omp parallel for collapse(2) schedule(static)
  for (int i = 1; i < N - 1; ++i) {
    for (int j = 1; j < N - 1; ++j) {
      C[i * N + j] =
          0.25 * (A[(i - 1) * N + j] +
                  A[(i + 1) * N + j] +
                  A[i * N + (j - 1)] +
                  A[i * N + (j + 1)]) -
          A[i * N + j];
    }
  }
  double t3b_end = omp_get_wtime();

  /* Sanity: B and C must be bit-identical (same arithmetic, same order). */
  double max_diff = 0.0;
  #pragma omp parallel for reduction(max:max_diff) schedule(static)
  for (int k = 0; k < NN; ++k) {
    double d = B[k] - C[k];
    if (d < 0) d = -d;
    if (d > max_diff) max_diff = d;
  }
  if (max_diff != 0.0) {
    fprintf(stderr, "WARNING: stencil variants disagree, max |B-C| = %.3e\n", max_diff);
  }

  /* -----------------------------
   * Task 4: Total sum reduction (use B; B and C are identical)
   * ----------------------------- */
  double t4_start = omp_get_wtime();
  double total = 0.0;
  #pragma omp parallel for reduction(+:total) schedule(static)
  for (int k = 0; k < NN; ++k) {
    total += B[k];
  }
  double t4_end = omp_get_wtime();

  /* Sample outputs (unchanged format) */
  int si = (N > 500) ? 500 : (N / 2);
  int sj = (N > 500) ? 500 : (N / 2);
  double sampleA = A[si * N + sj];
  double sampleRow = rowSums[si];

  writeResults(outFile, sampleA, sampleRow, total);

  /* Per-task timings */
  double dt1  = t1 - t0;
  double dt2  = t2_end - t2_start;
  double dt3a = t3a_end - t3a_start;
  double dt3b = t3b_end - t3b_start;
  double dt4  = t4_end - t4_start;

  fprintf(stdout,
          "threads=%d  N=%d  t1_init=%.6fs  t2_rowsum=%.6fs  "
          "t3a_stencil_no_collapse=%.6fs  t3b_stencil_collapse2=%.6fs  "
          "t4_total=%.6fs\n",
          nthreads, N, dt1, dt2, dt3a, dt3b, dt4);

  if (csvFile) {
    FILE* g = fopen(csvFile, "a");
    if (g) {
      /* If file is empty, write header. */
      fseek(g, 0, SEEK_END);
      if (ftell(g) == 0) {
        fprintf(g,
                "threads,N,t1_init,t2_rowsum,t3a_stencil,t3b_stencil_collapse,t4_total\n");
      }
      fprintf(g, "%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f\n",
              nthreads, N, dt1, dt2, dt3a, dt3b, dt4);
      fclose(g);
    }
  }

  free(A);
  free(B);
  free(C);
  free(rowSums);
  return 0;
}
