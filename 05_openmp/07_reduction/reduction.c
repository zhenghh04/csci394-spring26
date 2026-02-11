#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int n = 100;
    const int expected = n * (n + 1) / 2;
    int sum_no_reduction = 0;
    int sum_with_reduction = 0;

    int sum = 0; 
    #pragma omp parallel for reduction(+:sum)
    for (int i = 1; i <= n; i++) {
        sum += i;
    }

    // Case 1: no reduction -> data race on sum_no_reduction.
    #pragma omp parallel for
    for (int i = 1; i <= n; i++) {
        sum_no_reduction += i;
    }

    // Case 2: reduction -> each thread has a private partial sum.
    #pragma omp parallel for reduction(+:sum_with_reduction)
    for (int i = 1; i <= n; i++) {
        sum_with_reduction += i;
    }

    printf("sum 1..%d expected: %d\n", n, expected);
    printf("  - w/o reduction: %d (race, may be wrong)\n", sum_no_reduction);
    printf("  - w/  reduction: %d (correct)\n", sum_with_reduction);
    return 0;
}
