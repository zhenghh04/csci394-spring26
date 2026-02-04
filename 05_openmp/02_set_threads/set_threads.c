#include <omp.h>
#include <stdio.h>

int main(void) {
    int requested = 4;

    omp_set_num_threads(requested);
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    printf("[Outside] Hello from thread %d of %d\n", tid, nthreads);
    printf("---------\n");
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        #pragma omp single
        printf("Requested threads=%d, actual threads=%d\n", requested, nthreads);
        printf("Hello from thread %d of %d\n", tid, nthreads);
    }

    return 0;
}
