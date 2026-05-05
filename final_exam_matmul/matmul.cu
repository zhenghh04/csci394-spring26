#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define BLK         16        /* 16x16 = 256 threads per block */
#define NITER       10        /* number of timed iterations    */
#define PEAK_GFLOPS 19500.0   /* A100 FP32 peak (GFLOPS)       */

/* Step 1: Complete the kernel.
 * Each thread computes C[row][col] = sum_k A[row][k] * B[k][col].
 * A, B, C are n x n in row-major order: M[i][j] = M[i*n + j].
 * Guard against out-of-bounds threads. */
__global__ void matmul(const float *A, const float *B, float *C, int n)
{
    int row = blockIdx.y * BLK + threadIdx.y;
    int col = blockIdx.x * BLK + threadIdx.x;
    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++)  /* TODO: fill in the accumulation */
            sum += /* TODO */ 0.0f;
        C[row * n + col] = sum;
    }
}

void run(int n)
{
    size_t bytes = (size_t)n * n * sizeof(float);

    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    for (int i = 0; i < n * n; i++) { h_A[i] = 1.0f; h_B[i] = 1.0f; }

    /* Step 2: Allocate device memory for d_A, d_B, d_C */
    float *d_A, *d_B, *d_C;
    /* TODO: cudaMalloc */

    /* Step 3: Copy h_A and h_B from host to device */
    /* TODO: cudaMemcpy */

    /* Step 4: Set grid dimensions */
    dim3 block(BLK, BLK);
    dim3 grid(/* TODO */, /* TODO */);

    /* Warmup (not timed) */
    matmul<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaDeviceSynchronize();

    /* Timed iterations */
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int iter = 0; iter < NITER; iter++)
        matmul<<<grid, block>>>(d_A, d_B, d_C, n);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms_total = 0.0f;
    cudaEventElapsedTime(&ms_total, t0, t1);
    double ms = ms_total / NITER;   /* average time per iteration */

    /* Step 5: Compute achieved GFLOPS and efficiency.
     *   FLOPs per iteration = 2 * n^3
     *   achieved GFLOPS     = FLOPs / (ms * 1e6)
     *   efficiency (%)      = achieved GFLOPS / PEAK_GFLOPS * 100 */
    double gflops     = /* TODO */ 0.0;
    double efficiency = /* TODO */ 0.0;
    printf("n=%5d  time=%8.3f ms  GFLOPS=%8.1f  efficiency=%5.1f%%\n",
           n, ms, gflops, efficiency);

    /* Step 6: Copy d_C back to h_C and free device memory */
    /* TODO: cudaMemcpy + cudaFree */

    free(h_A); free(h_B); free(h_C);
}

int main(void)
{
    printf("%-7s  %-12s  %-14s  %s\n",
           "n", "time (ms)", "GFLOPS", "efficiency");
    printf("-------  ------------  --------------  ----------\n");
    int sizes[] = {16, 64, 256, 1024, 4096};
    int nsizes  = (int)(sizeof(sizes) / sizeof(sizes[0]));
    for (int i = 0; i < nsizes; i++)
        run(sizes[i]);
    return 0;
}
