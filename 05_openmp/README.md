# OpenMP examples

This module includes a few small OpenMP examples, from very basic to slightly
more advanced. Each subfolder contains a C source file and a simple Makefile.

## Contents
- `01_hello/` – minimal parallel region; threads print their IDs.
- `02_pi_for/` – `parallel for` loop to estimate pi.
- `03_reduction_timing/` – reduction + timing; shows scaling by changing threads.
- `04_sections_tasks/` – `sections` and a simple `task` example.


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
