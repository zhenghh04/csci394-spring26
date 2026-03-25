#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

__global__ void saturation_kernel(float *out, int repeats) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    float x = 0.001f * (float)(gid % 251);

    for (int iter = 0; iter < repeats; ++iter) {
        x = 1.000001f * x + 0.0000001f;
    }

    out[gid] = x;
}

static int rounded_blocks(int sms, int numerator, int denominator) {
    int blocks = (sms * numerator + denominator - 1) / denominator;
    return (blocks < 1) ? 1 : blocks;
}

int main(int argc, char **argv) {
    int threads_per_block = 256;
    int repeats = 200000;
    if (argc > 1) {
        threads_per_block = std::atoi(argv[1]);
    }
    if (argc > 2) {
        repeats = std::atoi(argv[2]);
    }
    if (threads_per_block < 32 || repeats < 1) {
        std::fprintf(stderr, "Usage: %s [threads_per_block>=32] [repeats>=1]\n", argv[0]);
        return 1;
    }

    int device = 0;
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, device);
    int sms = prop.multiProcessorCount;

    int sweep_blocks[] = {
        rounded_blocks(sms, 1, 4),
        rounded_blocks(sms, 1, 2),
        rounded_blocks(sms, 1, 1),
        rounded_blocks(sms, 2, 1),
        rounded_blocks(sms, 4, 1),
        rounded_blocks(sms, 8, 1)
    };
    int num_cases = (int)(sizeof(sweep_blocks) / sizeof(sweep_blocks[0]));
    int max_blocks = sweep_blocks[num_cases - 1];
    size_t max_threads = (size_t)max_blocks * (size_t)threads_per_block;

    float *host_out = (float *)std::malloc(max_threads * sizeof(float));
    float *device_out = nullptr;
    if (!host_out) {
        std::fprintf(stderr, "host allocation failed\n");
        return 1;
    }

    cudaMalloc((void **)&device_out, max_threads * sizeof(float));

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    std::printf("CUDA saturation sweep\n");
    std::printf("device=%s\n", prop.name);
    std::printf("SMs=%d threads_per_block=%d repeats=%d\n", sms, threads_per_block, repeats);
    std::printf("%10s %12s %12s %14s\n",
                "blocks", "blocks/SM", "time_ms", "GFLOP/s");

    for (int case_id = 0; case_id < num_cases; ++case_id) {
        int num_blocks = sweep_blocks[case_id];
        size_t total_threads = (size_t)num_blocks * (size_t)threads_per_block;

        cudaEventRecord(start);
        saturation_kernel<<<num_blocks, threads_per_block>>>(device_out, repeats);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        double flops = 2.0 * (double)total_threads * (double)repeats;
        double gflops = flops / ((double)ms * 1.0e6);

        cudaMemcpy(host_out, device_out, total_threads * sizeof(float),
                   cudaMemcpyDeviceToHost);

        std::printf("%10d %12.2f %12.3f %14.3f\n",
                    num_blocks, (double)num_blocks / (double)sms, ms, gflops);
    }

    std::printf("\n");
    std::printf("Interpretation:\n");
    std::printf("  Increase blocks until GFLOP/s stops rising much.\n");
    std::printf("  That plateau is a practical sign of saturation for this kernel.\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(device_out);
    std::free(host_out);
    return 0;
}
