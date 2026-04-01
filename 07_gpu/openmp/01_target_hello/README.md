# OpenMP Target Hello

This is the smallest OpenMP GPU-offload example in the module. It does not
perform a numerical kernel yet. Its purpose is to answer one question first:

- does OpenMP see a target device, and is a `target` region actually running there?

What this example does:

- queries the number of OpenMP target devices with `omp_get_num_devices()`
- queries the default target device with `omp_get_default_device()`
- enters a `#pragma omp target` region
- uses `omp_is_initial_device()` inside that region to detect whether the code
  executed on the host fallback path or on an offload device

Concepts:

- `#pragma omp target`
- `omp_get_num_devices`
- `omp_get_default_device`
- `omp_is_initial_device`

Why this matters:

- before showing larger offload examples, students need a way to verify that
  the environment is actually using a GPU
- many OpenMP implementations fall back to the CPU if no GPU offload target is
  available
- this example makes that fallback visible instead of hiding it

What to look for in the output:

- `num_devices`
  This is the number of target devices visible to OpenMP.
- `default_device`
  This is the device OpenMP will use unless told otherwise.
- `omp_is_initial_device`
  `1` means the target region ran on the host.
  `0` means the target region ran on an offload device.

Build:

```bash
make
```

Many compilers need explicit offload flags. Examples:

```bash
make OFFLOAD_FLAGS='-fopenmp-targets=nvptx64-nvidia-cuda'
make CC=nvc OFFLOAD_FLAGS='-mp=gpu'
```

Run:

```bash
./app
OMP_TARGET_OFFLOAD=MANDATORY ./app
```

Suggested class discussion:

- What does OpenMP mean by a "target device"?
- Why is host fallback convenient for portability but risky for benchmarking?
- Why might `num_devices=0` still allow the program to run?

What students should learn from this example:

- OpenMP offload is not just syntax; it also depends on compiler and runtime support
- `target` regions can run on the host if offload is unavailable
- the first step in debugging offload code is verifying where it actually ran
