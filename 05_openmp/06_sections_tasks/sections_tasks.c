#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

static long long fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

int main(void) {
    int a = 28, b = 30, c = 32;
    long long r1 = 0, r2 = 0, r3 = 0;

    #pragma omp parallel sections
    {
        #pragma omp section
        r1 = fib(a);

        #pragma omp section
        r2 = fib(b);

        #pragma omp section
        r3 = fib(c);
    }

    printf("sections: fib(%d)=%lld fib(%d)=%lld fib(%d)=%lld\n", a, r1, b, r2, c, r3);

    int n = 32;
    long long *results = (long long *)malloc((n + 1) * sizeof(long long));
    if (!results) return 1;

    #pragma omp parallel
    {
        #pragma omp single
        {
            for (int i = 0; i <= n; i++) {
                #pragma omp task firstprivate(i)
                {
                    results[i] = fib(i);
                }
            }
        }
    }

    printf("tasks: fib(%d)=%lld\n", n, results[n]);
    free(results);

    return 0;
}
