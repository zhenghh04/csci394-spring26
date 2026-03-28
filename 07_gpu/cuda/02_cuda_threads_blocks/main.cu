#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess) {                                           \
            fprintf(stderr, "%s:%d: %s failed: %s\n", __FILE__, __LINE__,     \
                    #call, cudaGetErrorString(err__));                        \
            return 1;                                                         \
        }                                                                     \
    } while (0)

__global__ void map_threads_kernel(int n, int *out_global, int *out_block,
                                   int *out_thread) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        out_global[gid] = gid;
        out_block[gid] = blockIdx.x;
        out_thread[gid] = threadIdx.x;
    }
}

int main(int argc, char **argv) {
    int n = 16;
    int threads_per_block = 4;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    if (argc > 2) {
        threads_per_block = atoi(argv[2]);
    }
    if (n <= 0 || threads_per_block <= 0) {
        fprintf(stderr, "Usage: %s [N>0] [threads_per_block>0]\n", argv[0]);
        return 1;
    }

    int num_blocks = (n + threads_per_block - 1) / threads_per_block;

    int *h_global = (int *)malloc((size_t)n * sizeof(*h_global));
    int *h_block = (int *)malloc((size_t)n * sizeof(*h_block));
    int *h_thread = (int *)malloc((size_t)n * sizeof(*h_thread));
    if (!h_global || !h_block || !h_thread) {
        fprintf(stderr, "host allocation failed\n");
        free(h_global);
        free(h_block);
        free(h_thread);
        return 1;
    }

    int *d_global = NULL;
    int *d_block = NULL;
    int *d_thread = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_global, (size_t)n * sizeof(*d_global)));
    CUDA_CHECK(cudaMalloc((void **)&d_block, (size_t)n * sizeof(*d_block)));
    CUDA_CHECK(cudaMalloc((void **)&d_thread, (size_t)n * sizeof(*d_thread)));

    map_threads_kernel<<<num_blocks, threads_per_block>>>(n, d_global, d_block,
                                                          d_thread);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_global, d_global, (size_t)n * sizeof(*h_global),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_block, d_block, (size_t)n * sizeof(*h_block),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_thread, d_thread, (size_t)n * sizeof(*h_thread),
                          cudaMemcpyDeviceToHost));

    printf("CUDA threads and blocks mapping\n");
    printf("N=%d threads_per_block=%d num_blocks=%d\n", n, threads_per_block,
           num_blocks);
    printf("global_idx block_idx thread_idx\n");
    for (int i = 0; i < n; i++) {
        printf("%10d %9d %10d\n", h_global[i], h_block[i], h_thread[i]);
    }

    CUDA_CHECK(cudaFree(d_global));
    CUDA_CHECK(cudaFree(d_block));
    CUDA_CHECK(cudaFree(d_thread));
    free(h_global);
    free(h_block);
    free(h_thread);
    return 0;
}
