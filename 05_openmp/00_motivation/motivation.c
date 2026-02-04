#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_sec(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

int main(int argc, char **argv) {
    long long n = 10000000000;
    if (argc > 1) {
        n = atoll(argv[1]);
    }

    // Serial work
    double t0 = now_sec();
    double sum = 0.0;
    for (long long i = 1; i <= n; i++) {
        sum += 1.0 / (double)i / (double)i;
    }
    double t1 = now_sec();

    // Parallel work
    double t2 = now_sec();
    double psum = 0.0;
    #pragma omp parallel for reduction(+:psum)
    for (long long i = 1; i <= n; i++) {
        psum += 1.0 / (double)i / (double)i;
    }
    double t3 = now_sec();

    double serial = t1 - t0;
    double parallel = t3 - t2;
    printf("sum=%.12f \n psum=%.12f\n", sum, psum);
    printf("n=%lld serial_s=%.6f parallel_s=%.6f speedup=%.2f\n",
           n, serial, parallel, serial / parallel);

    return 0;
}
