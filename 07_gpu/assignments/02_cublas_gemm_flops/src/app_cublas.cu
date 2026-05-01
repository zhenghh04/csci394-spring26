// cuBLAS FP32 GEMM and tensor-core (BF16-in, FP32-accum) GEMM throughput.
// Usage: ./app_cublas <mode> <n> <iters> <warmup>
//   mode in {fp32, tensor_core}
// Output:
//   RESULT,<mode>,n,iters,warmup,gemm_s,gflops,max_abs_err
//   DETAIL,<mode>,n,end_to_end_s=...,h2d_s=...,d2h_s=...

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    std::exit(2); } } while (0)

#define CHK_CUBLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    std::fprintf(stderr, "cuBLAS err %d @ %s:%d\n", (int)s, __FILE__, __LINE__); \
    std::exit(3); } } while (0)

static double wclock(void) {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

static void init_host(float *a, float *b, int n) {
    // Same pattern as project 1, reproducible
    for (size_t idx = 0; idx < (size_t)n * n; idx++) {
        a[idx] = (float)((idx % 13)) / 13.0f;
        b[idx] = (float)(((idx * 7) % 17)) / 17.0f;
    }
}

// Compute reference C[0:sample, :] = A[0:sample, :] * B  (row-major) on CPU
static float ref_max_err(const float *a, const float *b, const float *c_rowmajor,
                         int n, int sample) {
    float max_err = 0.0f;
    for (int i = 0; i < sample; i++) {
        for (int j = 0; j < sample; j++) {
            float s = 0.0f;
            for (int k = 0; k < n; k++) s += a[i*n + k] * b[k*n + j];
            float e = std::fabs(s - c_rowmajor[i*n + j]);
            if (e > max_err) max_err = e;
        }
    }
    return max_err;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s {fp32|tensor_core} n [iters=5] [warmup=1]\n", argv[0]);
        return 1;
    }
    std::string mode = argv[1];
    int n = std::atoi(argv[2]);
    int iters = (argc > 3) ? std::atoi(argv[3]) : 5;
    int warmup = (argc > 4) ? std::atoi(argv[4]) : 1;
    if (n <= 0 || iters <= 0) { std::fprintf(stderr, "bad args\n"); return 1; }

    size_t nn = (size_t)n * n;
    size_t fbytes = nn * sizeof(float);

    std::vector<float> a(nn), b(nn), c(nn);
    init_host(a.data(), b.data(), n);

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CHK(cudaMalloc(&dA, fbytes));
    CHK(cudaMalloc(&dB, fbytes));
    CHK(cudaMalloc(&dC, fbytes));

    double th0 = wclock();
    CHK(cudaMemcpy(dA, a.data(), fbytes, cudaMemcpyHostToDevice));
    CHK(cudaMemcpy(dB, b.data(), fbytes, cudaMemcpyHostToDevice));
    double th1 = wclock();

    cublasHandle_t handle;
    CHK_CUBLAS(cublasCreate(&handle));

    // For tensor_core mode we need BF16 buffers
    __nv_bfloat16 *dAbf = nullptr, *dBbf = nullptr;
    if (mode == "tensor_core") {
        CHK(cudaMalloc(&dAbf, nn * sizeof(__nv_bfloat16)));
        CHK(cudaMalloc(&dBbf, nn * sizeof(__nv_bfloat16)));
        // Convert on device with a tiny kernel via thrust-free approach:
        // do conversion on host then copy.
        std::vector<__nv_bfloat16> ahf(nn), bhf(nn);
        for (size_t i = 0; i < nn; i++) {
            ahf[i] = __float2bfloat16(a[i]);
            bhf[i] = __float2bfloat16(b[i]);
        }
        CHK(cudaMemcpy(dAbf, ahf.data(), nn * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(dBbf, bhf.data(), nn * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    }

    const float alpha = 1.0f;
    const float beta  = 0.0f;

    auto launch_gemm = [&]() {
        // We treat A and B as row-major n×n. cuBLAS is column-major. Trick:
        // (A·B)^T = B^T · A^T. So calling cublas with column-major B (which is
        // row-major B viewed as col-major B^T), then column-major A, computes
        // C^T in column-major == C in row-major. Use OP_N for both.
        if (mode == "fp32") {
            CHK_CUBLAS(cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                dB, n,
                dA, n,
                &beta,
                dC, n));
        } else {
            CHK_CUBLAS(cublasGemmEx(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, n, n,
                &alpha,
                dBbf, CUDA_R_16BF, n,
                dAbf, CUDA_R_16BF, n,
                &beta,
                dC,   CUDA_R_32F,  n,
                CUBLAS_COMPUTE_32F,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }
    };

    // warmup
    for (int w = 0; w < warmup; w++) {
        launch_gemm();
    }
    CHK(cudaDeviceSynchronize());

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    double t0 = wclock();
    cudaEventRecord(e0);
    for (int it = 0; it < iters; it++) launch_gemm();
    cudaEventRecord(e1);
    CHK(cudaEventSynchronize(e1));
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, e0, e1);
    double per_call_s = ((double)ms / 1.0e3) / iters;
    double e2e = wclock() - t0;

    double td0 = wclock();
    CHK(cudaMemcpy(c.data(), dC, fbytes, cudaMemcpyDeviceToHost));
    double td1 = wclock();

    int sample = n < 64 ? n : 64;
    float err = ref_max_err(a.data(), b.data(), c.data(), n, sample);

    double gflops = 2.0 * (double)n * n * n / per_call_s / 1.0e9;

    std::printf("RESULT,%s,%d,%d,%d,%.6e,%.3f,%.3e\n",
                mode.c_str(), n, iters, warmup, per_call_s, gflops, err);
    std::printf("DETAIL,%s,%d,end_to_end_s=%.6e,h2d_s=%.6e,d2h_s=%.6e\n",
                mode.c_str(), n, e2e, th1 - th0, td1 - td0);

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    if (dAbf) cudaFree(dAbf);
    if (dBbf) cudaFree(dBbf);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    cublasDestroy(handle);
    return 0;
}
