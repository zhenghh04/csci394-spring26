#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    long long n = 200000000;
    if (argc > 1) {
        n = atoll(argv[1]);
    }

    double start = omp_get_wtime();
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (long long i = 1; i <= n; i++) {
        sum += 1.0 / (double)i;
    }

    double end = omp_get_wtime();

    printf("n=%lld sum=%.6f time_s=%.6f\n", n, sum, end - start);
    return 0;
}
