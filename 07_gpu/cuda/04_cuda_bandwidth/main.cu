#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

int main(int argc, char **argv) {
    int n = 1 << 24;
    int iters = 10;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }
    if (argc > 2) {
        iters = std::atoi(argv[2]);
    }
    if (n <= 0 || iters <= 0) {
        std::fprintf(stderr, "Usage: %s [N>0] [iters>0]\n", argv[0]);
        return 1;
    }

    size_t bytes = (size_t)n * sizeof(float);
    float *host = nullptr;
    float *device = nullptr;

    cudaMallocHost((void **)&host, bytes);
    cudaMalloc((void **)&device, bytes);

    for (int i = 0; i < n; i++) {
        host[i] = (float)(i % 100) * 0.5f;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost);

    cudaEventRecord(start);
    for (int iter = 0; iter < iters; iter++) {
        cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float h2d_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms, start, stop);

    cudaEventRecord(start);
    for (int iter = 0; iter < iters; iter++) {
        cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float d2h_ms = 0.0f;
    cudaEventElapsedTime(&d2h_ms, start, stop);

    double total_gb = ((double)bytes * (double)iters) / 1.0e9;
    double h2d_bw = total_gb / ((double)h2d_ms / 1.0e3);
    double d2h_bw = total_gb / ((double)d2h_ms / 1.0e3);

    std::printf("CUDA data movement bandwidth example\n");
    std::printf("N=%d bytes=%zu iters=%d\n", n, bytes, iters);
    std::printf("H2D time=%.3f ms  bandwidth=%.3f GB/s\n", h2d_ms, h2d_bw);
    std::printf("D2H time=%.3f ms  bandwidth=%.3f GB/s\n", d2h_ms, d2h_bw);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device);
    cudaFreeHost(host);
    return 0;
}
