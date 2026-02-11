#include <omp.h>
#include <stdio.h>

int main(void) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        #pragma omp single
        {
            printf("single executed by thread %d\n", tid);
        }

        #pragma omp master
        {
            printf("master executed by thread %d\n", tid);
        }
    }

    return 0;
}
