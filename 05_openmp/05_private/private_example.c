#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(void) {
    double results[100];
    int temp;

    #pragma omp parallel for private(temp)
    for (int i = 0; i < 100; i++) {
        temp = i * i;       
        usleep(1*omp_get_thread_num());          // Each thread has its own temp.
        results[i] = temp * 3.14;     // No interference across threads.
    }

    for (int i = 0; i < 10; i++) {
        printf("results[%d] = %.2f\n", i, results[i]);
    }

    return 0;
}
