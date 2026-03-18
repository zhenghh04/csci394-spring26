#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

static inline void check_cuda(cudaError_t result, const char *what) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

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

    double *device_x = nullptr;
    double *device_y = nullptr;
    double *device_prod = nullptr;
    check_cuda(cudaMalloc((void **)&device_x, bytes), "cudaMalloc(device_x)");
    check_cuda(cudaMalloc((void **)&device_y, bytes), "cudaMalloc(device_y)");
    check_cuda(cudaMalloc((void **)&device_prod, bytes), "cudaMalloc(device_prod)");

    check_cuda(cudaMemcpy(device_x, host_x, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D x");
    check_cuda(cudaMemcpy(device_y, host_y, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D y");

    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    pointwise_product_kernel<<<num_blocks, threads_per_block>>>(device_x, device_y,
                                                                device_prod, n);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check_cuda(cudaMemcpy(host_prod, device_prod, bytes, cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H prod");

    double gpu_dot = 0.0;
    double cpu_dot = 0.0;
    for (int i = 0; i < n; i++) {
        gpu_dot += host_prod[i];
        cpu_dot += host_x[i] * host_y[i];
    }

    double abs_err = std::fabs(gpu_dot - cpu_dot);

    std::printf("CUDA inner product example\n");
    std::printf("N=%d threads_per_block=%d num_blocks=%d\n", n, threads_per_block,
                num_blocks);
    std::printf("gpu_dot=%.12f cpu_dot=%.12f abs_err=%.3e\n", gpu_dot, cpu_dot,
                abs_err);

    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_prod);
    std::free(host_x);
    std::free(host_y);
    std::free(host_prod);
    return abs_err > 1.0e-9;
}
