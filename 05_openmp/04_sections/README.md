# OpenMP sections

`#pragma omp sections` is an OpenMP work-sharing construct for running a small set of distinct code blocks in parallel, where each block is marked with `#pragma omp section`. It is useful when you have different independent tasks (not just iterations of one loop), such as one block computing statistics while another computes transforms or I/O preprocessing. Use `sections` when the number and type of tasks are known ahead of time; use OpenMP `task` when work is generated dynamically at runtime.

Example pattern:
```c
#pragma omp parallel
{
    #pragma omp sections
    {
        #pragma omp section
        { /* task A */ }

        #pragma omp section
        { /* task B */ }

        #pragma omp section
        { /* task C */ }
    }
}
```

This folder focuses on `#pragma omp sections` with four independent section jobs:
- count values above/below a threshold
- sum
- sum of squares (`sum2`)
- sum of cubes (`sum3`)

Files:
- `sections.c` (clean version)
- `sections_trace.c` (writes Chrome trace JSON)

## Build
```bash
make
```

## Run
```bash
OMP_NUM_THREADS=4 ./sections
OMP_NUM_THREADS=4 ./sections_trace
```

## Trace output
To see what exactly each thread is doing. 
- `sections_trace.json`
- Open with https://ui.perfetto.dev
