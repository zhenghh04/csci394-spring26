#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>

#include <cuda_runtime.h>

#include "common.h"

#define BX 16
#define BY 16

__global__ void gemm_kernel(const float *a, const float *b, float *c, int n) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || j >= n) return;
    float s = 0.0f;
    for (int k = 0; k < n; k++) s += a[i*n + k] * b[k*n + j];
    c[i*n + j] = s;
}

#define CHK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA err %s @ %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); \
    return 1; } } while (0)

static double wclock(void) {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double>(clk::now().time_since_epoch()).count();
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 512;
    int iters = (argc > 2) ? std::atoi(argv[2]) : 5;
    int warmup = (argc > 3) ? std::atoi(argv[3]) : 1;
    if (n <= 0 || iters <= 0) { std::fprintf(stderr, "bad args\n"); return 1; }

    size_t nn = (size_t)n * n;
    size_t bytes = nn * sizeof(float);
    float *a = (float *)std::malloc(bytes);
    float *b = (float *)std::malloc(bytes);
    float *c = (float *)std::malloc(bytes);
    if (!a || !b || !c) { std::fprintf(stderr, "alloc fail\n"); return 1; }
    init_matrices(a, b, c, n);

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    /* warmup */
    for (int w = 0; w < warmup; w++) {
        CHK(cudaMalloc(&da, bytes));
        CHK(cudaMalloc(&db, bytes));
        CHK(cudaMalloc(&dc, bytes));
        CHK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
        dim3 block(BX, BY);
        dim3 grid((n + BX - 1) / BX, (n + BY - 1) / BY);
        gemm_kernel<<<grid, block>>>(da, db, dc, n);
        CHK(cudaDeviceSynchronize());
        CHK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
        cudaFree(da); cudaFree(db); cudaFree(dc);
    }

    cudaEvent_t e_k0, e_k1;
    cudaEventCreate(&e_k0); cudaEventCreate(&e_k1);

    double sum_e2e = 0.0, sum_compute = 0.0, sum_h2d = 0.0, sum_d2h = 0.0;
    for (int it = 0; it < iters; it++) {
        std::memset(c, 0, bytes);
        double t0 = wclock();
        CHK(cudaMalloc(&da, bytes));
        CHK(cudaMalloc(&db, bytes));
        CHK(cudaMalloc(&dc, bytes));
        double th0 = wclock();
        CHK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
        CHK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));
        double th1 = wclock();
        dim3 block(BX, BY);
        dim3 grid((n + BX - 1) / BX, (n + BY - 1) / BY);
        cudaEventRecord(e_k0);
        gemm_kernel<<<grid, block>>>(da, db, dc, n);
        cudaEventRecord(e_k1);
        CHK(cudaEventSynchronize(e_k1));
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, e_k0, e_k1);
        double td0 = wclock();
        CHK(cudaMemcpy(c, dc, bytes, cudaMemcpyDeviceToHost));
        double td1 = wclock();
        cudaFree(da); cudaFree(db); cudaFree(dc);
        double t1 = wclock();
        sum_e2e += (t1 - t0);
        sum_compute += (double)ms / 1.0e3;
        sum_h2d += (th1 - th0);
        sum_d2h += (td1 - td0);
    }

    float err = cpu_reference_max_err(a, b, c, n);
    std::printf("RESULT,cuda,%d,%d,%d,%.6e,%.6e,%.3e\n",
                n, iters, warmup, sum_e2e/iters, sum_compute/iters, err);
    std::printf("DETAIL,cuda,%d,h2d_s=%.6e,d2h_s=%.6e\n",
                n, sum_h2d/iters, sum_d2h/iters);

    cudaEventDestroy(e_k0); cudaEventDestroy(e_k1);
    std::free(a); std::free(b); std::free(c);
    return 0;
}
