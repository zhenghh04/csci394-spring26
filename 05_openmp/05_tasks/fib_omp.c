#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static long long fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

int main(void) {
    const int lo = 20, hi = 40;
    long long *results = (long long *)malloc((hi + 1) * sizeof(long long));
    if (!results) return 1;

    double t0 = omp_get_wtime();
    #pragma omp parallel for schedule(static)
    for (int i = lo; i <= hi; i++) {
        results[i] = fib(i);
    }
    double t1 = omp_get_wtime();

    printf("fib_omp: fib(%d)=%lld fib(%d)=%lld\n", lo, results[lo], hi, results[hi]);
    printf("fib_omp_time_s=%.6f\n", t1 - t0);

    free(results);
    return 0;
}
