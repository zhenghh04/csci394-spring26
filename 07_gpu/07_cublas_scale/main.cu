#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublas_v2.h>

static int check_cuda(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(err));
        return 1;
    }
    return 0;
}

static int check_cublas(cublasStatus_t status, const char *what) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "%s failed with cuBLAS status %d\n", what, (int)status);
        return 1;
    }
    return 0;
}

int main(void) {
    const int m = 5;
    const int n = 5;
    float *host_a = (float *)std::malloc((size_t)m * (size_t)n * sizeof(float));
    if (!host_a) {
        std::fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    const float pi = 4.0f * std::atan(1.0f);
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            host_a[row + col * m] = std::sin(0.5f * pi * (float)(row + col * m) / (float)(m * n));
        }
    }

    float *device_a = nullptr;
    if (check_cuda(cudaMalloc((void **)&device_a, (size_t)m * (size_t)n * sizeof(float)), "cudaMalloc")) {
        std::free(host_a);
        return 1;
    }

    cublasHandle_t handle;
    if (check_cublas(cublasCreate(&handle), "cublasCreate")) {
        cudaFree(device_a);
        std::free(host_a);
        return 1;
    }

    if (check_cublas(cublasSetMatrix(m, n, sizeof(float), host_a, m, device_a, m), "cublasSetMatrix")) {
        cublasDestroy(handle);
        cudaFree(device_a);
        std::free(host_a);
        return 1;
    }

    const float alpha = 10.0f;
    if (check_cublas(cublasSscal(handle, m, &alpha, device_a, 1), "cublasSscal")) {
        cublasDestroy(handle);
        cudaFree(device_a);
        std::free(host_a);
        return 1;
    }

    if (check_cublas(cublasGetMatrix(m, n, sizeof(float), device_a, m, host_a, m), "cublasGetMatrix")) {
        cublasDestroy(handle);
        cudaFree(device_a);
        std::free(host_a);
        return 1;
    }

    std::printf("cuBLAS first-column scale example\n");
    for (int row = 0; row < m; row++) {
        for (int col = 0; col < n; col++) {
            std::printf("%8.5f ", host_a[row + col * m]);
        }
        std::printf("\n");
    }

    cublasDestroy(handle);
    cudaFree(device_a);
    std::free(host_a);
    return 0;
}
