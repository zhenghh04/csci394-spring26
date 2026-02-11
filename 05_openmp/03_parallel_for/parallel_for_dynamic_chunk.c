#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    const int N = 32;
    int chunk = 2;
    double a[N], b[N], result[N];

    if (argc > 1) {
        int parsed = atoi(argv[1]);
        if (parsed > 0) {
            chunk = parsed;
        }
    }

    for (int i = 0; i < N; i++) {
        a[i] = 1.0 * i;
        b[i] = 2.0 * i;
        result[i] = 0.0;
    }

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp for schedule(dynamic, chunk)
        for (int i = 0; i < N; i++) {
            result[i] = a[i] + b[i];
            printf("thread %d -> i=%d\n", tid, i);
        }
    }

    printf("chunk=%d, result[19]=%.1f\n", chunk, result[19]);
    return 0;
}
