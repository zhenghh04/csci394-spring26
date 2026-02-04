#include <omp.h>
#include <stdio.h>

int main(void) {
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    printf("[Outside] Hello from thread %d of %d\n", tid, nthreads);
    printf("---------\n");
    #pragma omp parallel
    {
        // tid and nthreads are inside parallel region as local variable to each thread
        int tid = omp_get_thread_num(); 
        int nthreads = omp_get_num_threads();
        printf("[Inside] Hello from thread %d of %d\n", tid, nthreads);
    }
    return 0;
}

// mistakes
/*
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    #pragma omp parallel
    {
        // tid and nthreads are outside parallel region as shared variable
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        printf("[Inside] Hello from thread %d of %d\n", tid, nthreads);
    }
*/