#include <omp.h>
#include <stdio.h>



int main(void) {
    int tid, nthreads;
    #pragma omp parallel
    {
     /* tid and nthreads are shared; this causes a race
    The issue is a data race on tid and nthreads.
    They’re declared outside the #pragma omp parallel, so they are shared by default.
    Every thread writes to the same two variables at the same time.
    The printed values are non‑deterministic, and the final [Outside] line just shows whatever thread happened to write last.
    */
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        printf("[Inside] Hello from thread %d of %d\n", tid, nthreads);
    }

    printf("[Outside] tid=%d nthreads=%d\n", tid, nthreads);
    return 0;
}

