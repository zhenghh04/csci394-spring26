# Quiz 03 - CSCI 394

**Name:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Dated:** April 8, 2026

## Multiple Choice (8 questions, 8 pts each = 64 pts) - Please highlight the correct answer

1.  The primary advantage of a GPU over a CPU for numerical workloads is:  
    A. Lower clock frequency  
    B. Higher single-thread performance with complex branch prediction  
    C. High throughput from many simple cores executing data-parallel work concurrently  
    D. Larger L1 cache per core

2.  In CUDA, the global thread index for a 1D launch is computed as:  
    A. threadIdx.x + gridDim.x  
    B. blockIdx.x \* blockDim.x + threadIdx.x  
    C. blockDim.x \* threadIdx.x + blockIdx.x  
    D. threadIdx.x \* gridDim.x + blockIdx.x

3.  On NVIDIA GPUs, threads are scheduled in groups called:  
    A. Blocks of 64 threads  
    B. Warps of 32 threads  
    C. Wavefronts of 128 threads  
    D. Teams of 16 threads

4.  In a heterogeneous CPU-GPU system, which factor most commonly dominates the end-to-end runtime of a simple GPU kernel?  
    A. Kernel compilation time  
    B. Host-to-device and device-to-host data transfer time  
    C. The time to allocate registers on the GPU  
    D. The time for the CPU to parse the source code

5.  Which CUDA function is used to copy data from host memory to device memory?  
    A. cudaMalloc  
    B. cudaFree  
    C. cudaMemcpy  
    D. cudaDeviceSynchronize

6.  Which of the following GPU programming models gives the programmer the most explicit control over thread indexing, shared memory, and launch configuration?  
    A. PyTorch  
    B. OpenACC  
    C. CUDA  
    D. NumPy

7.  In a CUDA shared-memory reduction, why is \_\_syncthreads() required between each halving step?  
    A. It transfers data from host to device  
    B. It ensures all threads in the block have completed their writes to shared memory before the next read  
    C. It launches a new kernel  
    D. It deallocates shared memory

8.  Which statement about the AXPY kernel (y\[i\] = a \* x\[i\] + y\[i\]) on a GPU is most accurate?  
    A. It is compute-bound because the multiply-add operation is expensive  
    B. It is typically bandwidth-bound because the arithmetic intensity is very low relative to data movement  
    C. It cannot be parallelized on a GPU  
    D. It requires shared memory and synchronization to be correct

## Short Answer (16 pts)

1.  Consider the following CUDA kernel and launch:

<!-- -->

    __global__ void scale_kernel(float *x, float alpha, int n) {
        int gid = blockIdx.x * blockDim.x + threadIdx.x;
        x[gid] = alpha * x[gid];
    }

    int main() {
        int n = 1000;
        float *d_x;
        cudaMalloc((void**)&d_x, n * sizeof(float));
        // assume d_x is initialized with valid data

        int threads_per_block = 256;
        int num_blocks = (n + threads_per_block - 1) / threads_per_block;
        scale_kernel<<<num_blocks, threads_per_block>>>(d_x, 2.0f, n);
        cudaDeviceSynchronize();
    }

1.  This kernel has a correctness bug. Identify the problem and explain what could go wrong.

2.  Write the corrected version of the kernel body (just the kernel function, not main).

3.  With n = 1000 and threads_per_block = 256, how many blocks are launched? How many total threads are launched? How many of those threads do no useful work?

## GPU Kernel Question (20 pts)

Write a complete CUDA kernel that computes the elementwise sum of two vectors: c\[i\] = (a\[i\] - b\[i\])^2 for i = 0, 1, ..., n-1.

Then write the host-side code that: 1. Allocates device memory for arrays a, b, and c 2. Copies input arrays a and b from host to device 3. Launches the kernel with an appropriate grid and block configuration 4. Copies the result c back to the host 5. Frees device memory

Use threads_per_block = 256. Assume the host arrays h_a, h_b, and h_c are already allocated and initialized, and n is known.

Please finish the implementation and run on Polaris:

    #include <cuda_runtime.h>
    #include <stdio.h>

    /* Step 1: Write the CUDA kernel here */




    int main() {
        int n = 1048576;
        size_t bytes = n * sizeof(float);

        /* Assume h_a, h_b are allocated and initialized on the host */
        float *h_a = (float *)malloc(bytes);
        float *h_b = (float *)malloc(bytes);
        float *h_c = (float *)malloc(bytes);

        /* Step 2: Declare device pointers and allocate device memory */




        /* Step 3: Copy input data from host to device */




        /* Step 4: Set up launch configuration and launch the kernel */
        int threads_per_block = 256;




        /* Step 5: Copy result back to host */




        /* Step 6: Free device memory */




        free(h_a);
        free(h_b);
        free(h_c);
        return 0;
    }
