#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static int count_mismatches(const int *out, int n) {
    int bad = 0;
    for (int i = 0; i < n; i++) {
        if (out[i] != i) bad++;
    }
    return bad;
}

int main(void) {
    const int n = 1000000;
    int *out = (int *)malloc((size_t)n * sizeof(int));
    if (!out) {
        fprintf(stderr, "Allocation failed\n");
        return 1;
    }

    for (int i = 0; i < n; i++) out[i] = -1;

    int i, tmp;

    // Incorrect: tmp is shared, so multiple threads race on it.
    #pragma omp parallel
    {
        #pragma omp for
        for (i = 0; i < n; i++) {
            tmp = i;        // data race on tmp
            out[tmp] = i;   // may write to wrong index
        }
    }
    int bad_shared = count_mismatches(out, n);

    for (int i2 = 0; i2 < n; i2++) out[i2] = -1;

    // Correct: tmp is private, each thread gets its own copy.
    #pragma omp parallel private(tmp)
    {
        #pragma omp for
        for (i = 0; i < n; i++) {
            tmp = i;
            out[tmp] = i;
        }
    }
    int bad_private = count_mismatches(out, n);

    printf("Mismatches without private: %d\n", bad_shared);
    printf("Mismatches with    private: %d\n", bad_private);

    free(out);
    return 0;
}
