# OpenMP examples

This module includes a few small OpenMP examples, from very basic to slightly
more advanced. Each subfolder contains a C source file and a simple Makefile.

Homework set for these examples:
- `HOMEWORK_00_09.md`

## Contents
- `assignments/` – assignment starter codes and specs for this module.
- `threads_examples.py` – Python threading demos (basic threads, queue, lock, barrier, thread pool, timing).
- `multithreads_io.py` – serial vs multithreaded file I/O benchmark in Python.
- `00_motivation/` – serial vs parallel loop timing to motivate OpenMP.
- `01_hello/` – minimal parallel region; threads print their IDs.
- `02_set_threads/` – set thread count programmatically with `omp_set_num_threads()`.
- `03_parallel_for/` – simple `parallel for` array addition.
- `04_sections/` – `sections` examples (`sum`, `sum2`, `sum3`, threshold counts).
- `05_tasks/` – OpenMP `task` examples with Fibonacci workload.
- `06_private/` – `private` variable example (data race vs fixed).
- `07_reduction/` – reduction examples.
- `08_synchronization/` – `atomic`, `critical`, `barrier`, and `single/master` examples.
- `09_pi_for/` – `parallel for` loop to estimate pi.
- `10_physics_heat/` – 2D heat diffusion stencil kernel.
- `11_materials_lj/` – Lennard-Jones force kernel.
- `12_mechanics_laplacian/` – 2D Laplacian stencil kernel.
- `13_ai_conv2d/` – Conv2D OpenMP example.
- `14_simd_completion/` – SIMD vs threads benchmark + thread completion imbalance study.


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

## How to build an OpenMP C program
General source file example: `file_omp.c`

Linux (GCC):
```bash
gcc -O2 -fopenmp file_omp.c -o file_omp
```

macOS (clang + Homebrew `libomp`):
```bash
clang -O2 -Xpreprocessor -fopenmp \
  -I/opt/homebrew/opt/libomp/include file_omp.c \
  -L/opt/homebrew/opt/libomp/lib -lomp -o file_omp
```

Run:
```bash
OMP_NUM_THREADS=4 ./file_omp
```

## Build (any folder)
```bash
make
```

## Run
```bash
# Example: 4 threads
OMP_NUM_THREADS=4 ./<program>

# Python threading examples (non-OpenMP)
python3 threads_examples.py --demo all

# Python I/O multithreading benchmark
python3 multithreads_io.py --files 20 --size-kb 128 --workers 6 --regenerate
```

## Notes
- Set threads with `OMP_NUM_THREADS` or `omp_set_num_threads()`.
- On macOS, OpenMP requires `libomp` (`brew install libomp`).
