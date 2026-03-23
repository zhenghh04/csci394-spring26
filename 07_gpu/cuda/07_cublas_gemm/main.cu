#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <cublas_v2.h>

static void print_matrix(const char *name, const float *matrix, int rows, int cols) {
    std::printf("%s =\n", name);
    for (int row = 0; row < rows; row++) {
        for (int col = 0; col < cols; col++) {
            std::printf("%8.3f ", matrix[row + col * rows]);
        }
        std::printf("\n");
    }
}

int main(void) {
    const int m = 2;
    const int k = 3;
    const int n = 2;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Column-major storage:
    // A = [1 2 3; 4 5 6]
    // B = [7 8; 9 10; 11 12]
    const float host_a[m * k] = {
        1.0f, 4.0f,
        2.0f, 5.0f,
        3.0f, 6.0f
    };
    const float host_b[k * n] = {
        7.0f, 9.0f, 11.0f,
        8.0f, 10.0f, 12.0f
    };
    float host_c[m * n] = {0.0f, 0.0f, 0.0f, 0.0f};
    float host_c_ref[m * n] = {0.0f, 0.0f, 0.0f, 0.0f};

    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            float sum = 0.0f;
            for (int p = 0; p < k; p++) {
                sum += host_a[row + p * m] * host_b[p + col * k];
            }
            host_c_ref[row + col * m] = sum;
        }
    }

    float *device_a = nullptr;
    float *device_b = nullptr;
    float *device_c = nullptr;
    cudaError_t cuda_status = cudaMalloc((void **)&device_a, sizeof(host_a));
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(device_a) failed: %s\n",
                     cudaGetErrorString(cuda_status));
        return 1;
    }
    cuda_status = cudaMalloc((void **)&device_b, sizeof(host_b));
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(device_b) failed: %s\n",
                     cudaGetErrorString(cuda_status));
        cudaFree(device_a);
        return 1;
    }
    cuda_status = cudaMalloc((void **)&device_c, sizeof(host_c));
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc(device_c) failed: %s\n",
                     cudaGetErrorString(cuda_status));
        cudaFree(device_a);
        cudaFree(device_b);
        return 1;
    }

    cuda_status = cudaMemcpy(device_a, host_a, sizeof(host_a), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy H2D A failed: %s\n",
                     cudaGetErrorString(cuda_status));
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        return 1;
    }
    cuda_status = cudaMemcpy(device_b, host_b, sizeof(host_b), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy H2D B failed: %s\n",
                     cudaGetErrorString(cuda_status));
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        return 1;
    }
    cuda_status = cudaMemcpy(device_c, host_c, sizeof(host_c), cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy H2D C failed: %s\n",
                     cudaGetErrorString(cuda_status));
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        return 1;
    }

    cublasHandle_t handle;
    cublasStatus_t cublas_status = cublasCreate(&handle);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasCreate failed with cuBLAS status %d\n",
                     (int)cublas_status);
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        return 1;
    }

    cublas_status = cublasSgemm(handle,
                                CUBLAS_OP_N, CUBLAS_OP_N,
                                m, n, k,
                                &alpha,
                                device_a, m,
                                device_b, k,
                                &beta,
                                device_c, m);
    if (cublas_status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cublasSgemm failed with cuBLAS status %d\n",
                     (int)cublas_status);
        cublasDestroy(handle);
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        return 1;
    }

    cuda_status = cudaMemcpy(host_c, device_c, sizeof(host_c), cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy D2H C failed: %s\n",
                     cudaGetErrorString(cuda_status));
        cublasDestroy(handle);
        cudaFree(device_a);
        cudaFree(device_b);
        cudaFree(device_c);
        return 1;
    }

    float max_abs_err = 0.0f;
    for (int i = 0; i < m * n; i++) {
        float abs_err = std::fabs(host_c[i] - host_c_ref[i]);
        if (abs_err > max_abs_err) {
            max_abs_err = abs_err;
        }
    }

    std::printf("cuBLAS GEMM example: C = A * B\n");
    print_matrix("A", host_a, m, k);
    print_matrix("B", host_b, k, n);
    print_matrix("C_gpu", host_c, m, n);
    print_matrix("C_ref", host_c_ref, m, n);
    std::printf("max_abs_err = %.6e\n", max_abs_err);

    cublasDestroy(handle);
    cudaFree(device_a);
    cudaFree(device_b);
    cudaFree(device_c);
    return max_abs_err > 1.0e-6f;
}
