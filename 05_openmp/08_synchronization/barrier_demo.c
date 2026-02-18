#include <omp.h>
#include <stdio.h>

int main(void) {
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        printf("thread %d before barrier\n", tid);

        printf("thread %d after barrier\n", tid);
    }

    return 0;
}
    