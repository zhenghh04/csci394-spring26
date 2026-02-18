/* loopy.c  (SERIAL / non-parallel version) 
 * 
 * Tasks:
 * 1. Matrix Initialization
 * 2. Row Sum Calculation
 * 3. Stencil Computation
 * 4. Total Sum (serial)
 *
 * Usage: ./loopy --dim <N> --output <file.dat
 * Example: ./loopy --dim 2048 --output loopy.dat
 *
 * Output: loopy.dat with sample values and total sum.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void usage(const char* prog) {
  fprintf(stderr, "Usage: %s --dim <N> [--output <file>]\n", prog);
  fprintf(stderr, "Example: %s --dim 2048 --output loopy.dat\n", prog);
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
  int N = 2048; /* default */
  const char* outFile = "loopy.dat";

  /* Parse: --dim <N>, --output <file> */
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "--dim") == 0) {
      if (i + 1 >= argc) {
        usage(argv[0]);
        return 1;
      }
      N = atoi(argv[i + 1]);
      i++; /* skip value */
    } else if (strcmp(argv[i], "--output") == 0) {
      if (i + 1 >= argc) {
        usage(argv[0]);
        return 1;
      }
      outFile = argv[i + 1];
      i++; /* skip value */
    } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
      usage(argv[0]);
      return 0;
    } else {
      fprintf(stderr, "Unknown argument: %s\n", argv[i]);
      usage(argv[0]);
      return 1;
    }
  }

  if (N < 8) {
    fprintf(stderr, "Error: --dim must be >= 8\n");
    return 1;
  }

  /* WARNING: int overflow if N is huge. For class-scale N it's fine. */
  int NN = N * N;

  double* A = (double*)malloc((long long)NN * sizeof(double));
  double* B = (double*)malloc((long long)NN * sizeof(double));
  double* rowSums = (double*)malloc((long long)N * sizeof(double));

  if (!A || !B || !rowSums) {
    fprintf(stderr, "Error: malloc failed (try smaller --dim)\n");
    free(A); free(B); free(rowSums);
    return 1;
  }

  /* -----------------------------
   * Task 1: Matrix Initialization
   * ----------------------------- */
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      A[i * N + j] = 0.001 * i + 0.002 * j;
    }
  }

  /* -----------------------------
   * Task 2: Row Sum Calculation
   * ----------------------------- */
  for (int i = 0; i < N; ++i) {
    double sum = 0.0;
    for (int j = 0; j < N; ++j) {
      sum += A[i * N + j];
    }
    rowSums[i] = sum;
  }

  /* -----------------------------
   * Task 3: Stencil Computation
   * ----------------------------- */

  /* boundaries: copy */
  for (int i = 0; i < N; ++i) {
    B[i * N + 0] = A[i * N + 0];
    B[i * N + (N - 1)] = A[i * N + (N - 1)];
  }
  for (int j = 0; j < N; ++j) {
    B[0 * N + j] = A[0 * N + j];
    B[(N - 1) * N + j] = A[(N - 1) * N + j];
  }

  /* interior: 5-point stencil */
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

  /* -----------------------------
   * Task 4: Total Sum (serial)
   * ----------------------------- */
  double total = 0.0;
  for (int k = 0; k < NN; ++k) {
    total += B[k];
  }

  /* Sample outputs */
  int si = (N > 500) ? 500 : (N / 2);
  int sj = (N > 500) ? 500 : (N / 2);

  double sampleA = A[si * N + sj];
  double sampleRow = rowSums[si];

  writeResults(outFile, sampleA, sampleRow, total);

  free(A);
  free(B);
  free(rowSums);
  return 0;
}
