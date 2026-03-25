#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

__global__ void lj_forces_kernel(const double *x, const double *y, double *fx, double *fy,
                                 int n, double sigma, double epsilon, double cutoff2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) {
        return;
    }

    double xi = x[i];
    double yi = y[i];
    double fxi = 0.0;
    double fyi = 0.0;
    const double sigma2 = sigma * sigma;

    for (int j = 0; j < n; ++j) {
        if (i == j) {
            continue;
        }
        double dx = xi - x[j];
        double dy = yi - y[j];
        double r2 = dx * dx + dy * dy;
        if (r2 > cutoff2) {
            continue;
        }

        double inv_r2 = 1.0 / r2;
        double sr2 = sigma2 * inv_r2;
        double sr6 = sr2 * sr2 * sr2;
        double sr12 = sr6 * sr6;
        double f = 24.0 * epsilon * inv_r2 * (2.0 * sr12 - sr6);
        fxi += f * dx;
        fyi += f * dy;
    }

    fx[i] = fxi;
    fy[i] = fyi;
}

int main(int argc, char **argv) {
    int n = 2000;
    int threads_per_block = 256;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }
    if (argc > 2) {
        threads_per_block = std::atoi(argv[2]);
    }
    if (n < 2 || threads_per_block < 1) {
        std::fprintf(stderr, "Usage: %s [n>=2] [threads_per_block>=1]\n", argv[0]);
        return 1;
    }

    const double sigma = 1.0;
    const double epsilon = 1.0;
    const double cutoff = 2.5 * sigma;
    const double cutoff2 = cutoff * cutoff;
    const size_t bytes = (size_t)n * sizeof(double);

    double *host_x = (double *)std::malloc(bytes);
    double *host_y = (double *)std::malloc(bytes);
    double *host_fx = (double *)std::malloc(bytes);
    double *host_fy = (double *)std::malloc(bytes);

    for (int i = 0; i < n; ++i) {
        host_x[i] = (double)(i % 50) * 1.1;
        host_y[i] = (double)(i / 50) * 1.1;
    }

    double *device_x = nullptr;
    double *device_y = nullptr;
    double *device_fx = nullptr;
    double *device_fy = nullptr;
    cudaMalloc((void **)&device_x, bytes);
    cudaMalloc((void **)&device_y, bytes);
    cudaMalloc((void **)&device_fx, bytes);
    cudaMalloc((void **)&device_fy, bytes);
    cudaMemcpy(device_x, host_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device_y, host_y, bytes, cudaMemcpyHostToDevice);

    dim3 block((unsigned)threads_per_block);
    dim3 grid((unsigned)((n + threads_per_block - 1) / threads_per_block));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);


    dim3 block((unsigned)threads_per_block);
    dim3 grid((unsigned)((n + threads_per_block - 1) / threads_per_block));
    lj_forces_kernel<<<grid, block>>>(device_x, device_y, device_fx, device_fy,
                                      n, sigma, epsilon, cutoff2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float gpu_ms = 0.0f;
    cudaEventElapsedTime(&gpu_ms, start, stop);
    cudaMemcpy(host_fx, device_fx, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host_fy, device_fy, bytes, cudaMemcpyDeviceToHost);

    std::printf("CUDA Lennard-Jones forces\n");
    std::printf("n=%d threads_per_block=%d blocks=%u\n", n, threads_per_block, grid.x);
    std::printf("gpu_kernel_time_ms=%.3f\n", gpu_ms);
    std::printf("fx[0]=%.3e fy[0]=%.3e\n", host_fx[0], host_fy[0]);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_fx);
    cudaFree(device_fy);
    std::free(host_x);
    std::free(host_y);
    std::free(host_fx);
    std::free(host_fy);
    return 0;
}
