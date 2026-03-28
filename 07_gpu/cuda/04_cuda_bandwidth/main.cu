#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__,          \
                         __LINE__, #call, cudaGetErrorString(err__));         \
            return 1;                                                         \
        }                                                                     \
    } while (0)

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

    CUDA_CHECK(cudaMallocHost((void **)&host, bytes));
    CUDA_CHECK(cudaMalloc((void **)&device, bytes));

    for (int i = 0; i < n; i++) {
        host[i] = (float)(i % 100) * 0.5f;
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iters; iter++) {
        CUDA_CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float h2d_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, start, stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int iter = 0; iter < iters; iter++) {
        CUDA_CHECK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float d2h_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, start, stop));

    double total_gb = ((double)bytes * (double)iters) / 1.0e9;
    double h2d_bw = total_gb / ((double)h2d_ms / 1.0e3);
    double d2h_bw = total_gb / ((double)d2h_ms / 1.0e3);

    std::printf("CUDA data movement bandwidth example\n");
    std::printf("N=%d bytes=%zu iters=%d\n", n, bytes, iters);
    std::printf("H2D time=%.3f ms  bandwidth=%.3f GB/s\n", h2d_ms, h2d_bw);
    std::printf("D2H time=%.3f ms  bandwidth=%.3f GB/s\n", d2h_ms, d2h_bw);

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(device));
    CUDA_CHECK(cudaFreeHost(host));
    return 0;
}
