#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

static int count_mismatches(const double *arr, int n) {
    int bad = 0;
    for (int i = 0; i < n; i++) {
        double expected = (double)(i * i) * 3.14;
        if (fabs(arr[i] - expected) > 1e-9) {
            bad++;
        }
    }
    return bad;
}

int main(void) {
    const int n = 40000;
    double *results = (double *)malloc((size_t)n * sizeof(double));
    if (!results) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    volatile int temp = 0;

    // Bad: temp is shared, so this has a data race.
    #pragma omp parallel for schedule(dynamic, 1) shared(temp)
    for (int i = 0; i < n; i++) {
        temp = i * i;
        usleep(1);
        results[i] = (double)temp * 3.14;
    }
    int bad_shared = count_mismatches(results, n);

    // Good: each loop iteration has its own private temp.
    #pragma omp parallel for schedule(dynamic, 1)
    for (int i = 0; i < n; i++) {
        int temp_private = i * i;
        usleep(1);
        results[i] = (double)temp_private * 3.14;
    }
    int bad_private = count_mismatches(results, n);

    printf("Mismatches with shared temp : %d\n", bad_shared);
    printf("Mismatches with private temp: %d\n", bad_private);

    free(results);
    return 0;
}
