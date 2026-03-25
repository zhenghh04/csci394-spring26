#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

static inline int idx(int i, int j, int n) {
    return i * n + j;
}

__global__ void laplacian2d_kernel(const double *u, double *lap, int n) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i <= 0 || i >= n - 1 || j <= 0 || j >= n - 1) {
        return;
    }

    int center = idx(i, j, n);
    lap[center] = u[idx(i - 1, j, n)] + u[idx(i + 1, j, n)] +
                  u[idx(i, j - 1, n)] + u[idx(i, j + 1, n)] -
                  4.0 * u[center];
}

int main(int argc, char **argv) {
    int n = 2048;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }
    if (n < 8) {
        std::fprintf(stderr, "Usage: %s [N>=8]\n", argv[0]);
        return 1;
    }

    const size_t count = (size_t)n * (size_t)n;
    const size_t bytes = count * sizeof(double);

    double *host_u = (double *)std::malloc(bytes);
    double *host_lap = (double *)std::calloc(count, sizeof(double));

    for (size_t i = 0; i < count; ++i) {
        host_u[i] = (double)(i % 100) * 0.01;
    }

    double *device_u = nullptr;
    double *device_lap = nullptr;
    cudaMalloc((void **)&device_u, bytes);
    cudaMalloc((void **)&device_lap, bytes);
    cudaMemcpy(device_u, host_u, bytes, cudaMemcpyHostToDevice);
    cudaMemset(device_lap, 0, bytes);

    dim3 block(16, 16);
    dim3 grid((unsigned)((n + block.x - 1) / block.x),
              (unsigned)((n + block.y - 1) / block.y));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    laplacian2d_kernel<<<grid, block>>>(device_u, device_lap, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    cudaMemcpy(host_lap, device_lap, bytes, cudaMemcpyDeviceToHost);

    std::printf("CUDA 2D Laplacian\n");
    std::printf("n=%d block=%u x %u grid=%u x %u\n", n, block.x, block.y, grid.x, grid.y);
    std::printf("gpu_kernel_time_ms=%.3f\n", gpu_ms);
    std::printf("lap center=%.6f\n", host_lap[idx(n / 2, n / 2, n)]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_u);
    cudaFree(device_lap);
    std::free(host_u);
    std::free(host_lap);
    return 0;
}
