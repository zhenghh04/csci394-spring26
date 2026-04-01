#include <omp.h>
#include <stdio.h>

int main(void) {
    int num_devices = omp_get_num_devices();
    int default_device = omp_get_default_device();
    int running_on_host = 1;

#pragma omp target map(from : running_on_host)
    {
        running_on_host = omp_is_initial_device();
    }

    printf("OpenMP target hello\n");
    printf("num_devices=%d default_device=%d\n", num_devices, default_device);
    printf("omp_is_initial_device=%d\n", running_on_host);
    if (running_on_host) {
        printf("Result: target region ran on the host fallback path.\n");
    } else {
        printf("Result: target region ran on an offload device.\n");
    }
    return 0;
}
