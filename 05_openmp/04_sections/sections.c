#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    const int N = 1000000;
    int *x = (int *)malloc((size_t)N * sizeof(int));
    long long sum = 0, sum2 = 0;
    long double sum3 = 0.0L;
    int upper = 0, lower = 0;
    const int divide = 20;
    if (!x) {
        fprintf(stderr, "Allocation failed for N=%d\n", N);
        return 1;
    }

    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        x[i] = i;
    }

    #pragma omp parallel shared(x, sum, sum2, sum3, upper, lower)
    {
        #pragma omp sections
        {
            #pragma omp section
            {
                for (int i = 0; i < N; i++) {
                    if (x[i] > divide) upper++;
                    if (x[i] <= divide) lower++;
                }
                printf("The number of points at or below %d in x is %d\n", divide, lower);
                printf("The number of points above %d in x is %d\n", divide, upper);
            }

            #pragma omp section
            {
                for (int i = 0; i < N; i++) {
                    sum = sum + x[i];
                }
                printf("Sum of x = %lld\n", sum);
            }

            #pragma omp section
            {
                for (int i = 0; i < N; i++) {
                    sum2 = sum2 + x[i] * x[i];
                }
                printf("Sum2 of x = %lld\n", sum2);
            }

            #pragma omp section
            {
                for (int i = 0; i < N; i++) {
                    sum3 = sum3 + (long double)x[i] * x[i] * x[i];
                }
                printf("Sum3 of x = %.0Lf\n", sum3);
            }
        }
    }

    free(x);
    return 0;
}
