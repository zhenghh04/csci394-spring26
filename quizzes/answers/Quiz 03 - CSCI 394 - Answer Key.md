# Quiz 03 - CSCI 394 — Answer Key

**Dated:** April 8, 2026

## Multiple Choice (8 questions, 8 pts each = 64 pts)

1. The primary advantage of a GPU over a CPU for numerical workloads is:
   A. Lower clock frequency
   B. Higher single-thread performance with complex branch prediction
   **C. High throughput from many simple cores executing data-parallel work concurrently ✅**
   D. Larger L1 cache per core

2. In CUDA, the global thread index for a 1D launch is computed as:
   A. `threadIdx.x + gridDim.x`
   **B. `blockIdx.x * blockDim.x + threadIdx.x` ✅**
   C. `blockDim.x * threadIdx.x + blockIdx.x`
   D. `threadIdx.x * gridDim.x + blockIdx.x`

3. On NVIDIA GPUs, threads are scheduled in groups called:
   A. Blocks of 64 threads
   **B. Warps of 32 threads ✅**
   C. Wavefronts of 128 threads
   D. Teams of 16 threads

4. In a heterogeneous CPU-GPU system, which factor most commonly dominates the end-to-end runtime of a simple GPU kernel?
   A. Kernel compilation time
   **B. Host-to-device and device-to-host data transfer time ✅**
   C. The time to allocate registers on the GPU
   D. The time for the CPU to parse the source code

5. Which CUDA function is used to copy data from host memory to device memory?
   A. `cudaMalloc`
   B. `cudaFree`
   **C. `cudaMemcpy` ✅**
   D. `cudaDeviceSynchronize`

6. Which of the following GPU programming models gives the programmer the most explicit control over thread indexing, shared memory, and launch configuration?
   A. PyTorch
   B. OpenACC
   **C. CUDA ✅**
   D. NumPy

7. In a CUDA shared-memory reduction, why is `__syncthreads()` required between each halving step?
   A. It transfers data from host to device
   **B. It ensures all threads in the block have completed their writes to shared memory before the next read ✅**
   C. It launches a new kernel
   D. It deallocates shared memory

8. Which statement about the AXPY kernel (`y[i] = a * x[i] + y[i]`) on a GPU is most accurate?
   A. It is compute-bound because the multiply-add operation is expensive
   **B. It is typically bandwidth-bound because the arithmetic intensity is very low relative to data movement ✅**
   C. It cannot be parallelized on a GPU
   D. It requires shared memory and synchronization to be correct

## Short Answer (16 pts) — `scale_kernel` bug

```c
__global__ void scale_kernel(float *x, float alpha, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    x[gid] = alpha * x[gid];
}
```

**(1) The bug:** there is no bounds check on `gid`. The launch configuration rounds the thread count up to a multiple of `threads_per_block`, so for `n = 1000` and `threads_per_block = 256`, 1024 threads run but only 1000 array elements exist. Threads with `gid` ∈ [1000, 1023] read/write `x[gid]` past the end of the allocation → undefined behavior: illegal-memory access, silent corruption of neighbouring memory, or wrong results.

**(2) Corrected kernel:**
```c
__global__ void scale_kernel(float *x, float alpha, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        x[gid] = alpha * x[gid];
    }
}
```

**(3) With `n = 1000`, `threads_per_block = 256`:**
- `num_blocks = (1000 + 255) / 256 = 1255 / 256 = ` **4 blocks**
- Total threads launched = 4 × 256 = **1024**
- Threads doing no useful work = 1024 − 1000 = **24**

## GPU Kernel Question (20 pts) — `c[i] = (a[i] − b[i])²`

```c
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

/* Step 1: CUDA kernel */
__global__ void diff_sq_kernel(const float *a, const float *b, float *c, int n) {
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) {
        float d = a[gid] - b[gid];
        c[gid] = d * d;
    }
}

int main() {
    int n = 1048576;
    size_t bytes = n * sizeof(float);

    /* Assume h_a, h_b are allocated and initialized on the host */
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    /* Step 2: Declare device pointers and allocate device memory */
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, bytes);
    cudaMalloc((void **)&d_b, bytes);
    cudaMalloc((void **)&d_c, bytes);

    /* Step 3: Copy input data from host to device */
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

    /* Step 4: Set up launch configuration and launch the kernel */
    int threads_per_block = 256;
    int num_blocks = (n + threads_per_block - 1) / threads_per_block;
    diff_sq_kernel<<<num_blocks, threads_per_block>>>(d_a, d_b, d_c, n);
    cudaDeviceSynchronize();

    /* Step 5: Copy result back to host */
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

    /* Step 6: Free device memory */
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);
    return 0;
}
```

**Notes for grading / Polaris run:**
- `n = 1048576` is exactly 4096 blocks of 256 threads, so the bounds check isn't strictly needed for this `n`, but it's good practice and required if `n` isn't a multiple of `threads_per_block`.
- Compile with `nvcc diff_sq.cu -o diff_sq`. On Polaris, request a GPU node (e.g. `qsub -I -A <project> -q debug -l select=1 -l walltime=00:30:00 -l filesystems=home`) and run `./diff_sq`.
- This kernel is bandwidth-bound (3 floats moved per 2 FLOPs), consistent with the AXPY discussion in MC #8.
