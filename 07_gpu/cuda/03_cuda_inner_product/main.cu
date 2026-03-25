#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

__global__ void pointwise_product_kernel(const double *x, const double *y,
                                         double *prod, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        prod[gid] = x[gid] * y[gid];
    }
}

int main(int argc, char **argv) {
    int n = 1 << 20;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }
    if (n <= 0) {
        std::fprintf(stderr, "Usage: %s [N>0]\n", argv[0]);
        return 1;
    }

    size_t bytes = (size_t)n * sizeof(double);
    double *host_x = (double *)std::malloc(bytes);
    double *host_y = (double *)std::malloc(bytes);
    double *host_prod = (double *)std::malloc(bytes);
    if (!host_x || !host_y || !host_prod) {
        std::fprintf(stderr, "host allocation failed\n");
        std::free(host_x);
        std::free(host_y);
        std::free(host_prod);
        return 1;
    }

    for (int i = 0; i < n; i++) {
        host_x[i] = 1.0 + 0.001 * (double)(i % 100);
        host_y[i] = 2.0 - 0.002 * (double)(i % 100);
    }

    auto cpu_t0 = std::chrono::steady_clock::now();
    double cpu_dot = 0.0;
    for (int i = 0; i < n; i++) {
        cpu_dot += host_x[i] * host_y[i];
    }
    auto cpu_t1 = std::chrono::steady_clock::now();
    double cpu_time_s =
        std::chrono::duration<double>(cpu_t1 - cpu_t0).count();

    double *device_x = nullptr;
    double *device_y = nullptr;
    double *device_prod = nullptr;
    cudaMalloc((void **)&device_x, bytes);
    cudaMalloc((void **)&device_y, bytes);
    cudaMalloc((void **)&device_prod, bytes);

    cudaEvent_t start_total, stop_total, start_h2d, stop_h2d, start_kernel,
        stop_kernel, start_d2h, stop_d2h;
    cudaEventCreate(&start_total);
    cudaEventCreate(&stop_total);
    cudaEventCreate(&start_h2d);
    cudaEventCreate(&stop_h2d);
    cudaEventCreate(&start_kernel);
    cudaEventCreate(&stop_kernel);
    cudaEventCreate(&start_d2h);
    cudaEventCreate(&stop_d2h);

    cudaEventRecord(start_total);
    cudaEventRecord(start_h2d);
    cudaMemcpy(device_x, host_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, host_y, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);

    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    cudaEventRecord(start_kernel);
    pointwise_product_kernel<<<num_blocks, threads_per_block>>>(device_x, device_y,
                                                                device_prod, n);
    cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel);

    cudaEventRecord(start_d2h);
    cudaMemcpy(host_prod, device_prod, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    double gpu_dot = 0.0;
    for (int i = 0; i < n; i++) {
        gpu_dot += host_prod[i];
    }

    double abs_err = std::fabs(gpu_dot - cpu_dot);
    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    float total_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms, start_h2d, stop_h2d);
    cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
    cudaEventElapsedTime(&d2h_ms, start_d2h, stop_d2h);
    cudaEventElapsedTime(&total_ms, start_total, stop_total);

    std::printf("CUDA inner product example\n");
    std::printf("N=%d threads_per_block=%d num_blocks=%d\n", n, threads_per_block,
                num_blocks);
    std::printf("gpu_dot=%.12f cpu_dot=%.12f abs_err=%.3e\n", gpu_dot, cpu_dot,
                abs_err);
    std::printf("cpu_time=%.6f s\n", cpu_time_s);
    std::printf("gpu_h2d=%.3f ms gpu_kernel=%.3f ms gpu_d2h=%.3f ms gpu_total=%.3f ms\n",
                h2d_ms, kernel_ms, d2h_ms, total_ms);
    if (total_ms > 0.0f) {
        double gpu_total_s = (double)total_ms / 1.0e3;
        std::printf("speedup_cpu_vs_gpu_total=%.3f\n", cpu_time_s / gpu_total_s);
    }

    cudaEventDestroy(start_total);
    cudaEventDestroy(stop_total);
    cudaEventDestroy(start_h2d);
    cudaEventDestroy(stop_h2d);
    cudaEventDestroy(start_kernel);
    cudaEventDestroy(stop_kernel);
    cudaEventDestroy(start_d2h);
    cudaEventDestroy(stop_d2h);
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_prod);
    std::free(host_x);
    std::free(host_y);
    std::free(host_prod);
    return abs_err > 1.0e-9;
}
