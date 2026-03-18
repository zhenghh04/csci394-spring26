#include <assert.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

static inline void check_cuda(cudaError_t result, const char *what) {
    if (result != cudaSuccess) {
        std::fprintf(stderr, "%s failed: %s\n", what, cudaGetErrorString(result));
        std::exit(EXIT_FAILURE);
    }
}

__global__ void derivative_x_kernel(const double *f, double *df, int nx, int ny, int nz, double inv_2dx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;
    if (i >= nx || j >= ny || k >= nz) {
        return;
    }

    int idx = i + nx * (j + ny * k);
    int im1 = (i == 0) ? 0 : i - 1;
    int ip1 = (i == nx - 1) ? nx - 1 : i + 1;
    int idx_l = im1 + nx * (j + ny * k);
    int idx_r = ip1 + nx * (j + ny * k);
    df[idx] = (f[idx_r] - f[idx_l]) * inv_2dx;
}

int main(int argc, char **argv) {
    int n = 96;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }
    if (n < 8) {
        std::fprintf(stderr, "Usage: %s [N>=8]\n", argv[0]);
        return 1;
    }

    const int nx = n;
    const int ny = n;
    const int nz = n;
    const size_t count = (size_t)nx * (size_t)ny * (size_t)nz;
    const size_t bytes = count * sizeof(double);

    double minx = -10.0;
    double maxx = 10.0;
    double miny = -10.0;
    double maxy = 10.0;
    double minz = -10.0;
    double maxz = 10.0;
    double hx = (maxx - minx) / (double)(nx - 1);
    double hy = (maxy - miny) / (double)(ny - 1);
    double hz = (maxz - minz) / (double)(nz - 1);

    double *host_f = (double *)std::malloc(bytes);
    double *host_df = (double *)std::malloc(bytes);
    if (!host_f || !host_df) {
        std::fprintf(stderr, "host allocation failed\n");
        std::free(host_f);
        std::free(host_df);
        return 1;
    }

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                double x = minx + (double)i * hx;
                double y = miny + (double)j * hy;
                double z = minz + (double)k * hz;
                host_f[i + nx * (j + ny * k)] = std::sin(x) + std::sin(y) + std::sin(z);
            }
        }
    }

    double *device_f = nullptr;
    double *device_df = nullptr;
    check_cuda(cudaMalloc((void **)&device_f, bytes), "cudaMalloc(device_f)");
    check_cuda(cudaMalloc((void **)&device_df, bytes), "cudaMalloc(device_df)");
    check_cuda(cudaMemcpy(device_f, host_f, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    dim3 block(16, 8, 1);
    dim3 grid((unsigned)((nx + block.x - 1) / block.x),
              (unsigned)((ny + block.y - 1) / block.y),
              (unsigned)nz);
    derivative_x_kernel<<<grid, block>>>(device_f, device_df, nx, ny, nz, 0.5 / hx);
    check_cuda(cudaGetLastError(), "kernel launch");
    check_cuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
    check_cuda(cudaMemcpy(host_df, device_df, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    double max_error = 0.0;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 1; i < nx - 1; i++) {
                double x = minx + (double)i * hx;
                double ref = std::cos(x);
                double err = std::fabs(ref - host_df[i + nx * (j + ny * k)]);
                if (err > max_error) {
                    max_error = err;
                }
            }
        }
    }

    std::printf("CUDA finite difference derivative example\n");
    std::printf("grid=%d x %d x %d  block=%d x %d x %d\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    std::printf("N=%d max_error=%g\n", n, max_error);

    cudaFree(device_f);
    cudaFree(device_df);
    std::free(host_f);
    std::free(host_df);
    return 0;
}
