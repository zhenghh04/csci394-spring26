#include <cmath>
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
    int n = 1 << 22;
    int max_gpus_to_use = 0;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }
    if (argc > 2) {
        max_gpus_to_use = std::atoi(argv[2]);
    }
    if (n <= 0) {
        std::fprintf(stderr, "Usage: %s [N>0] [num_gpus]\n", argv[0]);
        return 1;
    }

    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count <= 0) {
        std::fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }

    int num_gpus = device_count;
    if (max_gpus_to_use > 0 && max_gpus_to_use < num_gpus) {
        num_gpus = max_gpus_to_use;
    }
    if (num_gpus > n) {
        num_gpus = n;
    }

    size_t bytes = (size_t)n * sizeof(double);
    double *host_x = (double *)std::malloc(bytes);
    double *host_y = (double *)std::malloc(bytes);
    if (!host_x || !host_y) {
        std::fprintf(stderr, "host allocation failed\n");
        std::free(host_x);
        std::free(host_y);
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        host_x[i] = 1.0 + 0.001 * (double)(i % 100);
        host_y[i] = 2.0 - 0.002 * (double)(i % 100);
    }

    double cpu_dot = 0.0;
    for (int i = 0; i < n; ++i) {
        cpu_dot += host_x[i] * host_y[i];
    }

    double gpu_dot = 0.0;
    int threads_per_block = 256;

    std::printf("CUDA multi-GPU inner product example\n");
    std::printf("N=%d requested_gpus=%d visible_gpus=%d using_gpus=%d\n", n,
                max_gpus_to_use, device_count, num_gpus);

    for (int dev = 0; dev < num_gpus; ++dev) {
        int chunk_begin = (int)(((long long)dev * n) / num_gpus);
        int chunk_end = (int)(((long long)(dev + 1) * n) / num_gpus);
        int chunk_n = chunk_end - chunk_begin;
        size_t chunk_bytes = (size_t)chunk_n * sizeof(double);
        if (chunk_n <= 0) {
            continue;
        }

        cudaSetDevice(dev);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, dev);
        double *device_x = nullptr;
        double *device_y = nullptr;
        double *device_prod = nullptr;
        double *host_prod = (double *)std::malloc(chunk_bytes);

        cudaMalloc((void **)&device_x, chunk_bytes);
        cudaMalloc((void **)&device_y, chunk_bytes);
        cudaMalloc((void **)&device_prod, chunk_bytes);

        cudaMemcpy(device_x, host_x + chunk_begin, chunk_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(device_y, host_y + chunk_begin, chunk_bytes, cudaMemcpyHostToDevice);

        int num_blocks = (chunk_n + threads_per_block - 1) / threads_per_block;
        pointwise_product_kernel<<<num_blocks, threads_per_block>>>(device_x, device_y,
                                                                    device_prod, chunk_n);
        cudaDeviceSynchronize();

        cudaMemcpy(host_prod, device_prod, chunk_bytes, cudaMemcpyDeviceToHost);

        double device_partial = 0.0;
        for (int i = 0; i < chunk_n; ++i) {
            device_partial += host_prod[i];
        }
        gpu_dot += device_partial;

        std::printf("gpu_%d name=%s range=[%d,%d) chunk_n=%d num_blocks=%d partial=%.12f\n",
                    dev, prop.name, chunk_begin, chunk_end, chunk_n, num_blocks,
                    device_partial);

        cudaFree(device_x);
        cudaFree(device_y);
        cudaFree(device_prod);
        std::free(host_prod);
    }

    double abs_err = std::fabs(gpu_dot - cpu_dot);
    std::printf("multi_gpu_dot=%.12f cpu_dot=%.12f abs_err=%.3e\n", gpu_dot,
                cpu_dot, abs_err);

    std::free(host_x);
    std::free(host_y);
    return abs_err > 1.0e-9;
}
