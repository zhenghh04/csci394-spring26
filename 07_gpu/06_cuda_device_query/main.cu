#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceProperties failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("CUDA device query\n");
    printf("device_count=%d\n", device_count);
    printf("device_0_name=%s\n", prop.name);
    printf("compute_capability=%d.%d\n", prop.major, prop.minor);
    printf("global_mem_gb=%.2f\n", (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("multi_processor_count=%d\n", prop.multiProcessorCount);
    return 0;
}
