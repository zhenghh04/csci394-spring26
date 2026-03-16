#!/usr/bin/env python3

import time

import torch

matrix_size = 8192
niter=10
warmup=2

def benchmark(device_name: str, warmup: int, niter: int) -> float:
    device = torch.device(device_name)

    a = torch.rand((matrix_size, matrix_size), device=device)
    b = torch.rand((matrix_size, matrix_size), device=device)

    if device_name == "cuda":
        torch.cuda.synchronize()

    # warmup
    for _ in range(warmup):
        c = torch.matmul(a, b)

    if device_name == "cuda":
        torch.cuda.synchronize()    

    # benchmark
    start = time.perf_counter()

    for _ in range(niter):
        c = torch.matmul(a, b)

    if device_name == "cuda":
        torch.cuda.synchronize()

    end = time.perf_counter()

    _ = c[0, 0].item()
    return (end - start)/niter


print(f"matrix size = {matrix_size} x {matrix_size}")

cpu_time = benchmark("cpu", warmup=warmup, niter=niter)
print(f"cpu time  = {cpu_time:.6f} s")

if torch.cuda.is_available():
    gpu_time = benchmark("cuda", warmup=warmup, niter=niter)
    print(f"gpu time  = {gpu_time:.6f} s")
    print(f"speedup   = {cpu_time / gpu_time:.2f}x")
else:
    print("CUDA not available")
