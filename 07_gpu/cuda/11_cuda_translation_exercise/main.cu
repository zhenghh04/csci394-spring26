#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <cuda_runtime.h>

/*
In-class exercise: translate the CPU three-point stencil in main.c to CUDA.

What students should do:
1. Write the CUDA kernel.
2. Allocate device memory.
3. Copy the input array to the GPU.
4. Launch the kernel with a 1D grid.
5. Copy the result back to the host.

CPU formula:
    out[i] = in[i - 1] + 2.0 * in[i] + in[i + 1];

Only indices 1..n-2 should be updated.
*/

__global__ void stencil_kernel(const double *in, double *out, int n) {
    // TODO:
    // 1. Compute the global thread index.
    // 2. Guard the boundary points.
    // 3. Compute the three-point stencil.
}

int main(int argc, char **argv) {
    int n = 1 << 20;
    int threads_per_block = 256;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }
    if (argc > 2) {
        threads_per_block = std::atoi(argv[2]);
    }
    if (n < 3 || threads_per_block < 1) {
        std::fprintf(stderr, "Usage: %s [N>=3] [threads_per_block>=1]\n", argv[0]);
        return 1;
    }

    size_t bytes = (size_t)n * sizeof(double);
    double *host_in = (double *)std::malloc(bytes);
    double *host_out = (double *)std::calloc((size_t)n, sizeof(double));

    for (int i = 0; i < n; ++i) {
        host_in[i] = std::sin(0.001 * (double)i);
    }

    double *device_in = nullptr;
    double *device_out = nullptr;

    // TODO: allocate device arrays with cudaMalloc.

    // TODO: copy host_in to device_in with cudaMemcpy.

    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // TODO: launch stencil_kernel<<<num_blocks, threads_per_block>>>(...)

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // TODO: copy device_out back to host_out.

    std::printf("CUDA three-point stencil exercise\n");
    std::printf("N=%d threads_per_block=%d num_blocks=%d\n",
                n, threads_per_block, num_blocks);
    std::printf("gpu_time_ms=%.3f\n", gpu_ms);
    std::printf("out[1]=%.6f out[n/2]=%.6f out[n-2]=%.6f\n",
                host_out[1], host_out[n / 2], host_out[n - 2]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_in);
    cudaFree(device_out);
    std::free(host_in);
    std::free(host_out);
    return 0;
}
