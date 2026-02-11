#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static long long fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

int main(void) {
    const int lo = 20, hi = 40;
    long long *results = (long long *)malloc((hi + 1) * sizeof(long long));
    if (!results) return 1;

    struct timespec t0, t1;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    for (int i = lo; i <= hi; i++) {
        results[i] = fib(i);
    }
    clock_gettime(CLOCK_MONOTONIC, &t1);

    double elapsed = (double)(t1.tv_sec - t0.tv_sec) +
                     1e-9 * (double)(t1.tv_nsec - t0.tv_nsec);
    printf("serial: fib(%d)=%lld fib(%d)=%lld\n", lo, results[lo], hi, results[hi]);
    printf("serial_time_s=%.6f\n", elapsed);

    free(results);
    return 0;
}
