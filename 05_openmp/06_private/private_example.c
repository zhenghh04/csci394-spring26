#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static int count_mismatches(const double *a, const double *b, int n) {
    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (fabs(a[i] - b[i]) > 1e-9) {
            bad++;
        }
    }
    return bad;
}

int main(void) {
    const int n = 40000;
    double *serial = (double *)malloc((size_t)n * sizeof(double));
    double *parallel = (double *)malloc((size_t)n * sizeof(double));
    double *parallel_no_private = (double *)malloc((size_t)n * sizeof(double));

    if (!serial || !parallel || !parallel_no_private) {
        fprintf(stderr, "Allocation failed\n");
        free(serial);
        free(parallel);
        free(parallel_no_private);
        return 1;
    }

    // Case 1: serial baseline.
    for (int i = 0; i < n; i++) {
        int temp = i * i;
        serial[i] = (double)temp * 3.14;
    }

    // Case 2: parallel for WITHOUT private (shared temp -> race).
    int temp = 0;
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        temp = i * i;
        usleep(1);
        parallel_no_private[i] = (double)temp * 3.14;
    }
    int bad_no_private = count_mismatches(parallel, serial, n);

    // Case 3: parallel for WITH private temp (correct).
    temp = 0; 
    #pragma omp parallel for private(temp)
    for (int i = 0; i < n; i++) {
        temp = i * i;
        usleep(1);
        parallel[i] = (double)temp * 3.14;
    }

    int bad_private = count_mismatches(parallel, serial, n);

    printf("Compared to serial baseline (n=%d)\n", n);
    printf("parallel_for_without_private mismatches = %d\n", bad_no_private);
    printf("parallel_for_with_private    mismatches = %d\n", bad_private);
    printf("serial first 10 results:\n");
    for (int i = 0; i < 5; i++) {
        printf("  serial[%d] = %.2f, parallel[%d] = %.2f, parallel_no_private[%d] = %.2f\n", i, serial[i], i, parallel[i], i, parallel_no_private[i]);
    }

    free(serial);
    free(parallel);
    free(parallel_no_private);
    return 0;
}
