#include <cuda_runtime.h>
#include <stdio.h>

struct ArchInfo {
    const char* name;
    int fp32_per_sm;
    int int32_per_sm;
    int fp64_per_sm;
    int tensor_cores_per_sm;
    bool separate_int32_units;
};

static ArchInfo lookup_arch_info(int major, int minor) {
    const int cc = 10 * major + minor;
    switch (cc) {
        case 50:
        case 52:
        case 53:
            return {"Maxwell", 128, 128, 4, 0, false};
        case 60:
            return {"Pascal (GP100)", 64, 64, 32, 0, false};
        case 61:
        case 62:
            return {"Pascal", 128, 128, 4, 0, false};
        case 70:
            return {"Volta", 64, 64, 32, 8, true};
        case 72:
            return {"Volta/Xavier", 64, 64, 32, 8, true};
        case 75:
            return {"Turing", 64, 64, 2, 8, true};
        case 80:
            return {"Ampere (A100)", 64, 64, 32, 4, true};
        case 86:
        case 87:
            return {"Ampere", 128, 64, 2, 4, true};
        case 89:
            return {"Ada", 128, 64, 2, 4, true};
        case 90:
            return {"Hopper", 128, 64, 64, 4, true};
        default:
            return {"Unknown", 0, 0, 0, 0, false};
    }
}

static void print_estimates(const cudaDeviceProp& prop) {
    ArchInfo info = lookup_arch_info(prop.major, prop.minor);
    const int sms = prop.multiProcessorCount;

    printf("  Architecture: %s\n", info.name);
    if (info.fp32_per_sm == 0) {
        printf("  Estimated cores: no table entry for compute capability %d.%d\n",
               prop.major, prop.minor);
        printf("  Note: CUDA reports SM count directly, but per-type core counts\n");
        printf("        must be inferred from architecture-specific tables.\n");
        return;
    }

    printf("  Estimated FP32 lanes per SM: %d\n", info.fp32_per_sm);
    printf("  Estimated FP32 lanes total: %d\n", sms * info.fp32_per_sm);
    printf("  Estimated FP64 lanes per SM: %d\n", info.fp64_per_sm);
    printf("  Estimated FP64 lanes total: %d\n", sms * info.fp64_per_sm);

    if (info.separate_int32_units) {
        printf("  Estimated INT32 lanes per SM: %d\n", info.int32_per_sm);
        printf("  Estimated INT32 lanes total: %d\n", sms * info.int32_per_sm);
    } else {
        printf("  Estimated INT32 lanes: shared with FP32 issue slots on this architecture\n");
        printf("  Shared INT32/FP32 lanes per SM: %d\n", info.int32_per_sm);
        printf("  Shared INT32/FP32 lanes total: %d\n", sms * info.int32_per_sm);
    }

    if (info.tensor_cores_per_sm > 0) {
        printf("  Estimated Tensor Cores per SM: %d\n", info.tensor_cores_per_sm);
        printf("  Estimated Tensor Cores total: %d\n", sms * info.tensor_cores_per_sm);
    } else {
        printf("  Estimated Tensor Cores: none\n");
    }
}

int main(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    if (device_count == 0) {
        fprintf(stderr, "No CUDA-capable devices found.\n");
        return 1;
    }

    printf("CUDA query with architecture-based core estimates\n");
    printf("device_count=%d\n\n", device_count);

    for (int device = 0; device < device_count; ++device) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, device);
        if (err != cudaSuccess) {
            fprintf(stderr, "cudaGetDeviceProperties failed for device %d: %s\n",
                    device, cudaGetErrorString(err));
            return 1;
        }

        printf("Device %d: %s\n", device, prop.name);
        printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("  SM count: %d\n", prop.multiProcessorCount);
        printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("  Warp size: %d\n", prop.warpSize);
        printf("  Core clock: %.2f MHz\n", prop.clockRate / 1000.0);
        printf("  Global memory: %.2f GiB\n",
               (double)prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
        printf("  Memory clock: %.2f MHz\n", prop.memoryClockRate / 1000.0);
        printf("  Memory bus width: %d bits\n", prop.memoryBusWidth);
        print_estimates(prop);
        printf("\n");
    }

    printf("Notes:\n");
    printf("  - CUDA does not expose per-datatype core counts directly.\n");
    printf("  - FP32/INT32/FP64/Tensor counts above are architecture-based estimates.\n");
    printf("  - Older GPUs often share INT32 and FP32 execution resources.\n");
    return 0;
}
