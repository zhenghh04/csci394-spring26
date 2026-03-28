#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

__global__ void inner_product_reduce_kernel(const double *x, const double *y,
                                            double *partial, int n) {
    extern __shared__ double shared[];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0;
    if (gid < n) {
        value = x[gid] * y[gid];
    }
    shared[threadIdx.x] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        partial[blockIdx.x] = shared[0];
    }
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

    size_t x_bytes = (size_t)n * sizeof(double);
    double *host_x = (double *)std::malloc(x_bytes);
    double *host_y = (double *)std::malloc(x_bytes);

    for (int i = 0; i < n; ++i) {
        host_x[i] = 1.0 + 0.001 * (double)(i % 100);
        host_y[i] = 2.0 - 0.002 * (double)(i % 100);
    }

    double cpu_dot = 0.0;
    for (int i = 0; i < n; ++i) {
        cpu_dot += host_x[i] * host_y[i];
    }

    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    size_t partial_bytes = (size_t)num_blocks * sizeof(double);
    double *host_partial = (double *)std::malloc(partial_bytes);

    double *device_x = nullptr;
    double *device_y = nullptr;
    double *device_partial = nullptr;
    cudaError_t status = cudaMalloc((void **)&device_x, x_bytes);
    status = cudaMalloc((void **)&device_y, x_bytes);
    status = cudaMalloc((void **)&device_partial, partial_bytes);

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
    status = cudaMemcpy(device_x, host_x, x_bytes, cudaMemcpyHostToDevice);
    status = cudaMemcpy(device_y, host_y, x_bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stop_h2d);

    cudaEventRecord(start_kernel);
    inner_product_reduce_kernel<<<num_blocks, threads_per_block,
                                  (size_t)threads_per_block * sizeof(double)>>>(
        device_x, device_y, device_partial, n);

    status = cudaDeviceSynchronize();
    cudaEventRecord(stop_kernel);
    cudaEventRecord(start_d2h);
    status = cudaMemcpy(host_partial, device_partial, partial_bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stop_d2h);
    cudaEventRecord(stop_total);
    cudaEventSynchronize(stop_total);

    double gpu_dot = 0.0;
    for (int block = 0; block < num_blocks; ++block) {
        gpu_dot += host_partial[block];
    }

    double abs_err = std::fabs(gpu_dot - cpu_dot);
    double rel_err = abs_err / (std::fabs(cpu_dot) + 1.0e-30);

    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    float total_ms = 0.0f;
    cudaEventElapsedTime(&h2d_ms, start_h2d, stop_h2d);
    cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);
    cudaEventElapsedTime(&d2h_ms, start_d2h, stop_d2h);
    cudaEventElapsedTime(&total_ms, start_total, stop_total);

    std::printf("CUDA reduction example\n");
    std::printf("N=%d threads_per_block=%d num_blocks=%d\n",
                n, threads_per_block, num_blocks);
    std::printf("gpu_dot=%.12f cpu_dot=%.12f abs_err=%.3e rel_err=%.3e\n",
                gpu_dot, cpu_dot, abs_err, rel_err);
    std::printf("gpu_h2d=%.3f ms gpu_kernel=%.3f ms gpu_d2h=%.3f ms gpu_total=%.3f ms\n",
                h2d_ms, kernel_ms, d2h_ms, total_ms);

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
    cudaFree(device_partial);
    std::free(host_x);
    std::free(host_y);
    std::free(host_partial);
    return rel_err > 1.0e-12;
}
