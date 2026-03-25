# CUDA Saturation Sweep

This example helps students see a practical saturation effect on a GPU.

Idea:

- keep the block size fixed
- sweep the number of launched thread blocks
- measure throughput
- look for the point where more blocks do not improve performance much

What this example does:

- queries the GPU `multiProcessorCount` value
- launches the same kernel with different total block counts
- reports:
  - total blocks
  - blocks per SM
  - kernel time
  - estimated GFLOP/s

Kernel:

- each thread performs the same repeated floating-point update in a loop
- this keeps the benchmark simple and makes the launch size the main variable

Concepts:

- SM count
- block count versus SM count
- `blocks / SM` as a rough occupancy-style teaching metric
- throughput plateau as a practical sign of saturation

How to read the output:

- when `blocks / SM < 1`, some SMs may be idle
- near `blocks / SM = 1`, each SM can usually get at least one block
- beyond that, throughput may continue to rise
- when GFLOP/s stops increasing much, the kernel is practically saturated

Important note:

- this does not measure absolute peak GPU performance
- it shows saturation behavior for this specific kernel
- real kernels also depend on registers, shared memory, memory traffic, and occupancy limits

Build:

```bash
make
```

If your driver is older than the CUDA toolkit, build for the exact GPU target:

```bash
make CUDA_ARCH=sm_80
```

Run:

```bash
./app
./app 256 200000
```

Arguments:

- `threads_per_block` default: `256`
- `repeats` default: `200000`

What students should learn:

- `#blocks >= #SMs` is a useful starting rule, not a full saturation rule
- one block per SM is often not enough for peak throughput
- performance usually levels off after enough blocks are available

Sample output on an NVIDIA A100 80GB PCIe:

```text
CUDA saturation sweep
device=NVIDIA A100 80GB PCIe
SMs=108 threads_per_block=256 repeats=200000
theoretical_fp32_peak_gflops=19491.840
warmup_blocks=108 (excluded from timing)
    blocks    blocks/SM      time_ms        GFLOP/s      %peak
        27         0.25        0.649       4258.675      21.85
        54         0.50        0.650       8510.219      43.66
       108         1.00        0.648      17060.769      87.53
       216         2.00        1.210      18276.044      93.76
       432         4.00        2.345      18861.540      96.77
       864         8.00        4.649      19031.885      97.64
```

How to interpret this sample:

- performance rises rapidly as the number of blocks approaches and then exceeds the SM count
- `1.0 blocks/SM` is better than `0.25` or `0.50`, but it is still not the best point
- the throughput begins to level off between `4.0` and `8.0 blocks/SM`
- that plateau is the practical saturation point for this kernel on this GPU
