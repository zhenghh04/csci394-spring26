#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * Demonstrates OpenMP task dependencies:
 * stage 1: produce A[i]
 * stage 2: consume A[i], produce B[i]
 * stage 3: consume B[i]
 */
int main(void) {
    const int n = 12;
    int *a = (int *)malloc((size_t)n * sizeof(int));
    int *b = (int *)malloc((size_t)n * sizeof(int));
    int *c = (int *)malloc((size_t)n * sizeof(int));
    if (!a || !b || !c) {
        fprintf(stderr, "allocation failed\n");
        free(a);
        free(b);
        free(c);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        a[i] = 0;
        b[i] = 0;
        c[i] = 0;
    }

#pragma omp parallel
    {
#pragma omp single
        {
            for (int i = 0; i < n; i++) {
#pragma omp task depend(out : a[i]) firstprivate(i)
                {
                    a[i] = i + 1;
                    printf("T%d produce a[%d]=%d\n", omp_get_thread_num(), i, a[i]);
                }

#pragma omp task depend(in : a[i]) depend(out : b[i]) firstprivate(i)
                {
                    b[i] = a[i] * 10;
                    printf("T%d transform b[%d]=%d\n", omp_get_thread_num(), i, b[i]);
                }

#pragma omp task depend(in : b[i]) depend(out : c[i]) firstprivate(i)
                {
                    c[i] = b[i] + 7;
                    printf("T%d finalize c[%d]=%d\n", omp_get_thread_num(), i, c[i]);
                }
            }
        }
    }

    long long sum = 0;
    int ok = 1;
    for (int i = 0; i < n; i++) {
        const int expected = (i + 1) * 10 + 7;
        sum += c[i];
        if (c[i] != expected) {
            ok = 0;
            fprintf(stderr, "mismatch at i=%d: got %d expected %d\n", i, c[i], expected);
        }
    }

    printf("check: %s, sum(c)=%lld\n", ok ? "PASS" : "FAIL", sum);

    free(a);
    free(b);
    free(c);
    return ok ? 0 : 2;
}
