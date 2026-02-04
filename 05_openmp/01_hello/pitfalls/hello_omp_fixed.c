#include <omp.h>
#include <stdio.h>

int main(void) {
    int tid, nthreads; 
    #pragma omp parallel private(tid, nthreads)
    {
        tid = omp_get_thread_num();
        nthreads = omp_get_num_threads();
        printf("[Inside] Hello from thread %d of %d\n", tid, nthreads);
    }

    return 0;
}
