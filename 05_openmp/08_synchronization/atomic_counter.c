#include <omp.h>
#include <stdio.h>

int main(void) {
    const int n = 1000000;
    int counter = 0;

    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        #pragma omp atomic
        counter += 1;
    }

    printf("atomic counter=%d expected=%d\n", counter, n);
    return 0;
}
