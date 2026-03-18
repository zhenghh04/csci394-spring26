#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

static inline void check_cuda(cudaError_t result, const char *what) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

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

    check_cuda(cudaMallocHost((void **)&host, bytes), "cudaMallocHost");
    check_cuda(cudaMalloc((void **)&device, bytes), "cudaMalloc");

    for (int i = 0; i < n; i++) {
        host[i] = (float)(i % 100) * 0.5f;
    }

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop), "cudaEventCreate(stop)");

    check_cuda(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice), "warmup H2D");
    check_cuda(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost), "warmup D2H");

    check_cuda(cudaEventRecord(start), "cudaEventRecord(start H2D)");
    for (int iter = 0; iter < iters; iter++) {
        check_cuda(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice),
                   "cudaMemcpy H2D");
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop H2D)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop H2D)");

    float h2d_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&h2d_ms, start, stop), "cudaEventElapsedTime H2D");

    check_cuda(cudaEventRecord(start), "cudaEventRecord(start D2H)");
    for (int iter = 0; iter < iters; iter++) {
        check_cuda(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost),
                   "cudaMemcpy D2H");
    }
    check_cuda(cudaEventRecord(stop), "cudaEventRecord(stop D2H)");
    check_cuda(cudaEventSynchronize(stop), "cudaEventSynchronize(stop D2H)");

    float d2h_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&d2h_ms, start, stop), "cudaEventElapsedTime D2H");

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
