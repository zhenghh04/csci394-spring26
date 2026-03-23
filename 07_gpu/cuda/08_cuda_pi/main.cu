#include <cmath>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

__global__ void pi_midpoint_reduce_kernel(int n, double dx, double *out_partial) {
    extern __shared__ double shared[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    double value = 0.0;
    if (gid < n) {
        double x = ((double)gid + 0.5) * dx;
        value = 4.0 / (1.0 + x * x);
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
        out_partial[blockIdx.x] = shared[0];
    }
}

__global__ void reduce_sum_kernel(const double *in, double *out, int n) {
    extern __shared__ double shared[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    shared[threadIdx.x] = (gid < n) ? in[gid] : 0.0;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out[blockIdx.x] = shared[0];
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
    if (n <= 0 || threads_per_block <= 0) {
        std::fprintf(stderr, "Usage: %s [num_intervals>0] [threads_per_block>0]\n",
                     argv[0]);
        return 1;
    }

    double dx = 1.0 / (double)n;
    if ((threads_per_block & (threads_per_block - 1)) != 0) {
        std::fprintf(stderr, "threads_per_block must be a power of two\n");
        return 1;
    }

    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    size_t partial_bytes = (size_t)num_blocks * sizeof(double);

    double *device_partial_a = nullptr;
    double *device_partial_b = nullptr;
    cudaError_t cuda_status = cudaMalloc((void **)&device_partial_a, partial_bytes);
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc device_partial_a failed: %s\n",
                     cudaGetErrorString(cuda_status));
        return 1;
    }
    cuda_status = cudaMalloc((void **)&device_partial_b, partial_bytes);
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc device_partial_b failed: %s\n",
                     cudaGetErrorString(cuda_status));
        cudaFree(device_partial_a);
        return 1;
    }

    pi_midpoint_reduce_kernel<<<num_blocks, threads_per_block,
                                (size_t)threads_per_block * sizeof(double)>>>(
        n, dx, device_partial_a);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "warmup kernel launch failed: %s\n",
                     cudaGetErrorString(cuda_status));
        cudaFree(device_partial_a);
        cudaFree(device_partial_b);
        return 1;
    }
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "warmup cudaDeviceSynchronize failed: %s\n",
                     cudaGetErrorString(cuda_status));
        cudaFree(device_partial_a);
        cudaFree(device_partial_b);
        return 1;
    }

    double gpu_sum = 0.0;
    auto gpu_t0 = std::chrono::steady_clock::now();
    pi_midpoint_reduce_kernel<<<num_blocks, threads_per_block,
                                (size_t)threads_per_block * sizeof(double)>>>(
        n, dx, device_partial_a);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "timed midpoint kernel launch failed: %s\n",
                     cudaGetErrorString(cuda_status));
        cudaFree(device_partial_a);
        cudaFree(device_partial_b);
        return 1;
    }
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "timed midpoint cudaDeviceSynchronize failed: %s\n",
                     cudaGetErrorString(cuda_status));
        cudaFree(device_partial_a);
        cudaFree(device_partial_b);
        return 1;
    }

    int current_n = num_blocks;
    double *current_in = device_partial_a;
    double *current_out = device_partial_b;
    while (current_n > 1) {
        int reduction_blocks = (current_n + threads_per_block - 1) / threads_per_block;
        reduce_sum_kernel<<<reduction_blocks, threads_per_block,
                            (size_t)threads_per_block * sizeof(double)>>>(
            current_in, current_out, current_n);
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            std::fprintf(stderr, "reduction kernel launch failed: %s\n",
                         cudaGetErrorString(cuda_status));
            cudaFree(device_partial_a);
            cudaFree(device_partial_b);
            return 1;
        }
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            std::fprintf(stderr, "reduction cudaDeviceSynchronize failed: %s\n",
                         cudaGetErrorString(cuda_status));
            cudaFree(device_partial_a);
            cudaFree(device_partial_b);
            return 1;
        }
        current_n = reduction_blocks;
        double *tmp = current_in;
        current_in = current_out;
        current_out = tmp;
    }

    cuda_status = cudaMemcpy(&gpu_sum, current_in, sizeof(double),
                             cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy final sum D2H failed: %s\n",
                     cudaGetErrorString(cuda_status));
        cudaFree(device_partial_a);
        cudaFree(device_partial_b);
        return 1;
    }
    auto gpu_t1 = std::chrono::steady_clock::now();
    double pi_gpu = gpu_sum * dx;
    double gpu_time_s =
        std::chrono::duration<double>(gpu_t1 - gpu_t0).count();

    double cpu_sum = 0.0;
    auto cpu_t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n; i++) {
        double x = ((double)i + 0.5) * dx;
        cpu_sum += 4.0 / (1.0 + x * x);
    }
    auto cpu_t1 = std::chrono::steady_clock::now();
    double pi_cpu = cpu_sum * dx;
    double cpu_time_s =
        std::chrono::duration<double>(cpu_t1 - cpu_t0).count();
    double abs_err_cpu = std::fabs(pi_gpu - pi_cpu);
    double abs_err_true = std::fabs(pi_gpu - std::acos(-1.0));

    std::printf("CUDA pi midpoint example\n");
    std::printf("num_intervals=%d threads_per_block=%d num_blocks=%d\n",
                n, threads_per_block, num_blocks);
    std::printf("pi_gpu=%.15f\n", pi_gpu);
    std::printf("pi_cpu=%.15f\n", pi_cpu);
    std::printf("gpu_time_s=%.6f\n", gpu_time_s);
    std::printf("cpu_time_s=%.6f\n", cpu_time_s);
    if (gpu_time_s > 0.0) {
        std::printf("speedup_cpu_vs_gpu=%.3f\n", cpu_time_s / gpu_time_s);
    }
    std::printf("abs_err_vs_cpu=%.6e\n", abs_err_cpu);
    std::printf("abs_err_vs_true_pi=%.6e\n", abs_err_true);

    cudaFree(device_partial_a);
    cudaFree(device_partial_b);
    return abs_err_cpu > 1.0e-12;
}
