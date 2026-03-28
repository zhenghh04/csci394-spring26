#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("CUDA device query\n");
    printf("device_count=%d\n", device_count);
    printf("device_0_name=%s\n", prop.name);
    printf("compute_capability=%d.%d\n", prop.major, prop.minor);
    printf("global_mem_gb=%.2f\n", (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("multi_processor_count=%d\n", prop.multiProcessorCount);
    printf("max_threads_per_block=%d\n", prop.maxThreadsPerBlock);
    return 0;
}
