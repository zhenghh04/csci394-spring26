#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

static int check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        return 0;
    }
    return 1;
}

__global__ void pi_midpoint_kernel(int n, double dx, double *out_terms) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        double x = ((double)gid + 0.5) * dx;
        out_terms[gid] = 4.0 / (1.0 + x * x);
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
    size_t bytes = (size_t)n * sizeof(double);

    double *host_terms = (double *)std::malloc(bytes);
    if (!host_terms) {
        std::fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    double *device_terms = nullptr;
    if (!check_cuda(cudaMalloc((void **)&device_terms, bytes), "cudaMalloc")) {
        std::free(host_terms);
        return 1;
    }

    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    pi_midpoint_kernel<<<num_blocks, threads_per_block>>>(n, dx, device_terms);
    if (!check_cuda(cudaGetLastError(), "kernel launch") ||
        !check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize") ||
        !check_cuda(cudaMemcpy(host_terms, device_terms, bytes, cudaMemcpyDeviceToHost),
                    "cudaMemcpy D2H")) {
        cudaFree(device_terms);
        std::free(host_terms);
        return 1;
    }

    double gpu_sum = 0.0;
    for (int i = 0; i < n; i++) {
        gpu_sum += host_terms[i];
    }
    double pi_gpu = gpu_sum * dx;

    double cpu_sum = 0.0;
    for (int i = 0; i < n; i++) {
        double x = ((double)i + 0.5) * dx;
        cpu_sum += 4.0 / (1.0 + x * x);
    }
    double pi_cpu = cpu_sum * dx;
    double abs_err_cpu = std::fabs(pi_gpu - pi_cpu);
    double abs_err_true = std::fabs(pi_gpu - std::acos(-1.0));

    std::printf("CUDA pi midpoint example\n");
    std::printf("num_intervals=%d threads_per_block=%d num_blocks=%d\n",
                n, threads_per_block, num_blocks);
    std::printf("pi_gpu=%.15f\n", pi_gpu);
    std::printf("pi_cpu=%.15f\n", pi_cpu);
    std::printf("abs_err_vs_cpu=%.6e\n", abs_err_cpu);
    std::printf("abs_err_vs_true_pi=%.6e\n", abs_err_true);

    cudaFree(device_terms);
    std::free(host_terms);
    return abs_err_cpu > 1.0e-12;
}
