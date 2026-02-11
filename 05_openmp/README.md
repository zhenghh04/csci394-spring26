# OpenMP examples

This module includes a few small OpenMP examples, from very basic to slightly
more advanced. Each subfolder contains a C source file and a simple Makefile.

## Contents
- `00_motivation/` – serial vs parallel loop timing to motivate OpenMP.
- `00_pitfalls/` – common OpenMP mistakes and fixes.
- `01_hello/` – minimal parallel region; threads print their IDs.
- `02_set_threads/` – set thread count programmatically with `omp_set_num_threads()`.
- `03_parallel_for/` – simple `parallel for` array addition.
- `04_reduction_timing/` – reduction + timing; shows scaling by changing threads.
- `05_private/` – `private` variable example (data race vs fixed).
- `06_sections/` – `sections` examples (`sum`, `sum2`, `sum3`, threshold counts).
- `07_tasks/` – OpenMP `task` examples with Fibonacci workload.
- `07_pi_for/` – `parallel for` loop to estimate pi.
- `08_physics_heat/` – 2D heat diffusion stencil kernel.
- `09_materials_lj/` – Lennard-Jones force kernel.
- `10_mechanics_laplacian/` – 2D Laplacian stencil kernel.


## Dependencies (OpenMP runtime)
macOS (Homebrew):
```bash
brew install libomp
```

Linux:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y gcc libomp-dev

# Fedora
sudo dnf install -y gcc libomp
```

Windows:
- Install MSYS2 or Visual Studio with C/C++ tools.
- MSYS2 (recommended for gcc + OpenMP):
  - https://www.msys2.org/
  - `pacman -S --needed base-devel mingw-w64-x86_64-toolchain`
  - Use the “MSYS2 MinGW 64-bit” shell and compile with `gcc -fopenmp`.


## Build (any folder)
```bash
make
```

## Run
```bash
# Example: 4 threads
OMP_NUM_THREADS=4 ./<program>
```

## Notes
- Set threads with `OMP_NUM_THREADS` or `omp_set_num_threads()`.
- On macOS, OpenMP requires `libomp` (`brew install libomp`).

## Example: `private`
`private` gives each thread its own copy of a variable.
```c
int i, tid;
#pragma omp parallel private(i, tid)
{
    tid = omp_get_thread_num();
    #pragma omp for
    for (i = 0; i < N; i++) {
        // use i and tid safely per-thread
        a[i] = i + tid;
    }
}
```
