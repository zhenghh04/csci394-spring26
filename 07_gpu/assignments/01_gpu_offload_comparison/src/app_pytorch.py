"""PyTorch GEMM benchmark for the GPU offload comparison assignment.

Output: RESULT,pytorch,n,iters,warmup,end_to_end_s,compute_s,max_abs_err
"""
import argparse
import time

import torch


def init_matrices(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.arange(n * n, device=device, dtype=torch.float32)
    a = ((idx % 13) / 13.0).reshape(n, n)
    b = (((idx * 7) % 17) / 17.0).reshape(n, n)
    return a, b


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("n", type=int)
    p.add_argument("iters", type=int, nargs="?", default=5)
    p.add_argument("warmup", type=int, nargs="?", default=1)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n, iters, warmup = args.n, args.iters, args.warmup

    # Build host (CPU) source matrices to measure transfer time honestly.
    a_host, b_host = init_matrices(n, torch.device("cpu"))

    # Warmup
    for _ in range(warmup):
        a = a_host.to(device, non_blocking=False)
        b = b_host.to(device, non_blocking=False)
        c = a @ b
        if device.type == "cuda":
            torch.cuda.synchronize()
        _ = c.cpu()

    sum_e2e = 0.0
    sum_compute = 0.0
    sum_h2d = 0.0
    sum_d2h = 0.0

    for _ in range(iters):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        th0 = time.perf_counter()
        a = a_host.to(device)
        b = b_host.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        th1 = time.perf_counter()

        if device.type == "cuda":
            ev_start = torch.cuda.Event(enable_timing=True)
            ev_end = torch.cuda.Event(enable_timing=True)
            ev_start.record()
            c = a @ b
            ev_end.record()
            ev_end.synchronize()
            compute_s = ev_start.elapsed_time(ev_end) / 1.0e3
        else:
            tk0 = time.perf_counter()
            c = a @ b
            tk1 = time.perf_counter()
            compute_s = tk1 - tk0

        td0 = time.perf_counter()
        c_h = c.cpu()
        td1 = time.perf_counter()
        t1 = time.perf_counter()

        sum_e2e += (t1 - t0)
        sum_compute += compute_s
        sum_h2d += (th1 - th0)
        sum_d2h += (td1 - td0)

    # Correctness vs CPU reference (small block of full result)
    sample = min(n, 64)
    a_cpu = a_host[:sample]
    b_cpu = b_host
    ref = a_cpu @ b_cpu  # (sample, n)
    got = c_h[:sample]
    max_abs_err = float((ref - got).abs().max().item())

    print(f"RESULT,pytorch,{n},{iters},{warmup},"
          f"{sum_e2e/iters:.6e},{sum_compute/iters:.6e},{max_abs_err:.3e}")
    print(f"DETAIL,pytorch,{n},h2d_s={sum_h2d/iters:.6e},"
          f"d2h_s={sum_d2h/iters:.6e},device={device.type}")


if __name__ == "__main__":
    main()
