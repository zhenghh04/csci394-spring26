# OpenMP parallel for

Examples of the same array-addition loop:
- `parallel_for_serial` (serial)
- `parallel_for_omp` (OpenMP `parallel for` + timing)
- `parallel_for_dist` (prints which thread handles each index)
- `parallel_for_static_chunk` (OpenMP `for schedule(static, chunk)`)
- `parallel_for_dynamic_chunk` (OpenMP `for schedule(dynamic, chunk)`)
- `parallel_for_schedule_bench` (benchmark with `schedule(runtime)`)

## Build
```bash
make
```

## Run
```bash
./parallel_for_serial
OMP_NUM_THREADS=4 ./parallel_for_omp
OMP_NUM_THREADS=4 ./parallel_for_dist
OMP_NUM_THREADS=4 ./parallel_for_static_chunk 2
OMP_NUM_THREADS=4 ./parallel_for_dynamic_chunk 2
OMP_NUM_THREADS=4 OMP_SCHEDULE="static,8" ./parallel_for_schedule_bench 100000000 5
```

## No-schedule distribution example
`parallel_for_dist.c` uses plain `#pragma omp for` (no `schedule(...)` clause):

```c
#pragma omp parallel
{
    int threadid = omp_get_thread_num();

    #pragma omp for
    for (int i = 0; i < N; i++) {
        printf("Thread id: %d working on index %d\\n", threadid, i);
    }
}
```

Run and sort by index to view distribution clearly:
```bash
OMP_NUM_THREADS=4 ./parallel_for_dist | sort -k7,7n
```

Schematic for `N=32`, `threads=4` (typical default behavior):

![No-schedule distribution](no_schedule_distribution.png)

Without an explicit `schedule(...)` clause, many OpenMP runtimes use a static-style partition for `omp for`, often giving each thread a contiguous block of iterations. That is why the no-schedule schematic looks similar to static block distribution. However, the exact default policy and chunk details are implementation-dependent, so if you need guaranteed behavior, set `schedule(static, ...)` or `schedule(dynamic, ...)` explicitly.

## Static chunk schematic
Example for `N=32`, `threads=4`, `chunk=2`:

![Static chunk distribution](static_chunk_distribution.png)

## Dynamic chunk schematic
Example for `N=32`, `threads=4`, `chunk=2` (one possible run):

![Dynamic chunk distribution](dynamic_chunk_distribution.png)

## What to notice
- The OpenMP version splits the loop across threads.
- Each iteration is independent, so it is safe to parallelize.
- `parallel_for_dist` shows how iterations are distributed.
- `parallel_for_static_chunk` shows fixed-size chunk distribution.
- `parallel_for_dynamic_chunk` shows dynamic chunk dispatch (`chunk=2` example above).

## Benchmark cases
Run each case for thread counts `1 2 4 6 8` and compare `avg_per_repeat_s`.

Case 1: `static` default
```bash
for t in 1 2 4 6 8; do OMP_NUM_THREADS=$t OMP_SCHEDULE="static" ./parallel_for_schedule_bench 100000000 5; done
```

Case 2: `static,1` (many tiny chunks)
```bash
for t in 1 2 4 6 8; do OMP_NUM_THREADS=$t OMP_SCHEDULE="static,1" ./parallel_for_schedule_bench 100000000 5; done
```

Case 3: `static,64` (larger chunks)
```bash
for t in 1 2 4 6 8; do OMP_NUM_THREADS=$t OMP_SCHEDULE="static,64" ./parallel_for_schedule_bench 100000000 5; done
```

Case 4: `dynamic,1` (highest scheduling overhead)
```bash
for t in 1 2 4 6 8; do OMP_NUM_THREADS=$t OMP_SCHEDULE="dynamic,1" ./parallel_for_schedule_bench 100000000 5; done
```

Case 5: `dynamic,64` (reduced dynamic overhead)
```bash
for t in 1 2 4 6 8; do OMP_NUM_THREADS=$t OMP_SCHEDULE="dynamic,64" ./parallel_for_schedule_bench 100000000 5; done
```
