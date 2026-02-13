#include <omp.h>
#include <stdio.h>

int main(void) {
    const int n = 10;
    int counter_atomic = 0;
    int counter_single = 0;
    int counter_critical = 0;
    int counter_reduction = 0;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        #pragma omp atomic
            counter_atomic += 1;
    }

    #pragma omp parallel reduction(+:counter_single)
    {
        counter_reduction += 1;
    }


    #pragma omp parallel reduction(+:counter_single)
    {
        #pragma omp single
        {
            counter_single += 1;
            printf("[Inside single] I am %d\n", omp_get_thread_num());
        }
        printf("[Outside single] I am %d\n", omp_get_thread_num());

    }
    printf("======\n");

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        #pragma omp critical
        {
            counter_critical += 1;
            printf("[Inside critical] I am %d\n", omp_get_thread_num());
        }
        printf("[Outside critical] I am %d\n", omp_get_thread_num());
    }

    printf("atomic   counter=%d (expected %d)\n", counter_atomic, n);
    printf("single   counter=%d (expected 1)\n", counter_single);
    printf("critical counter=%d (expected %d)\n", counter_critical, n);
    printf("reduction counter=%d (expected %d)\n", counter_reduction, n);
    return 0;
}
