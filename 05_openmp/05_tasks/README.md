# OpenMP tasks

`#pragma omp task` lets one thread create units of work dynamically at runtime, and idle threads in the team can execute those tasks. It is useful when work is irregular, recursive, or not known in advance (for example tree traversal, divide-and-conquer, or producer-generated jobs). Use `task` when dynamic scheduling is needed; use `sections` when you have a fixed set of known code blocks.

Example pattern:
```c
#pragma omp parallel
{
    #pragma omp single
    {
        for (int i = 0; i < n; i++) {
            #pragma omp task firstprivate(i)
            { /* task body */ }
        }
    }
}
```

In this pattern:
- `#pragma omp single` means only one thread executes the block, which avoids multiple threads creating duplicate tasks.
- `firstprivate(i)` gives each task its own private copy of `i`, initialized with the value at task creation time. Without it, tasks could read a changed loop variable value.

This folder focuses on OpenMP task creation with a Fibonacci workload.
Current task range in code: `i = 20..40`.

Files:
- `fib.c` (serial baseline, no OpenMP)
- `fib_omp.c` (`parallel for` baseline over the same range)
- `fib_omp_dynamic.c` (`parallel for schedule(dynamic,1)`)
- `fib_omp_trace.c` (`parallel for` + Chrome trace output)
- `fib_omp_dynamic_trace.c` (`parallel for schedule(dynamic,1)` + Chrome trace output)
- `tasks.c` (clean version)
- `tasks_trace.c` (timing + Chrome trace output)
- `depend_demo.c` (`task depend(in/out)` pipeline example)
- `dag_pipeline_demo.c` (realistic ETL/ML-style DAG with fork/join + ordered sink)

## Build
```bash
make
```

## Run
```bash
./fib
OMP_NUM_THREADS=4 ./fib_omp
OMP_NUM_THREADS=4 ./fib_omp_dynamic
OMP_NUM_THREADS=4 ./fib_omp_trace
OMP_NUM_THREADS=4 ./fib_omp_dynamic_trace
OMP_NUM_THREADS=4 ./tasks
OMP_NUM_THREADS=4 ./tasks_trace
OMP_NUM_THREADS=4 ./depend_demo
OMP_NUM_THREADS=4 ./dag_pipeline_demo
# optional args: batches dim
OMP_NUM_THREADS=4 ./dag_pipeline_demo 48 4096
```

## `depend` clause quick note
`depend` lets you express ordering through data dependencies instead of global barriers:
- `depend(out: x)` says this task produces `x`
- `depend(in: x)` says this task needs `x` first
- `depend(inout: x)` says this task both reads and writes `x`

`depend_demo.c` creates three tasks per index:
1. produce `a[i]`
2. transform `a[i] -> b[i]`
3. finalize `b[i] -> c[i]`

Different indices can run concurrently, while each index still preserves the required order.

`dag_pipeline_demo.c` is a DAG-style application:
- `load -> clean -> {feature_a, feature_b} -> merge -> infer -> write`
- `feature_a` and `feature_b` run in parallel for each batch (fork)
- `merge` waits on both branches (join)
- `write` uses `depend(inout: writer_token)` to serialize sink output order

## Trace output
- `fib_omp_trace.json`
- `fib_omp_dynamic_trace.json`
- `tasks_trace.json`
- Open with `chrome://tracing` or https://ui.perfetto.dev

## `parallel for` issue shown by trace
`fib_omp_trace.c` uses `#pragma omp parallel for schedule(static)`. For Fibonacci, iteration cost is highly non-uniform (`fib(40)` is much more expensive than `fib(20)`), so static contiguous assignment gives one thread most of the heavy iterations near the end of the range. In the trace, that thread runs much longer while others go idle, so total runtime is limited by the slowest thread (load imbalance / tail effect).

![OpenMP parallel-for trace](omp_trace.png)

This is why `tasks` scales better for this workload: work is picked up dynamically instead of being fixed up front by loop index.

`fib_omp_dynamic_trace.c` uses `#pragma omp parallel for schedule(dynamic,1)`. In this trace, heavy iterations are no longer stuck on one thread as in static scheduling; instead, iterations are pulled dynamically, so busy and idle periods are more balanced across threads and the long tail is reduced.

![OpenMP parallel-for dynamic(1) trace](omp_for_dynamic.png)


`tasks_trace.c` shows the task-based execution timeline for the same `fib(20..40)` workload. Compared to `dynamic,1`, it can be slightly faster at higher thread counts because the runtime schedules explicit work units from a task queue (work stealing), which can reduce tail idle time on irregular recursion-heavy jobs.

![OpenMP tasks trace](fib_tasks.png)

## Scaling study (1, 2, 4 threads)
We benchmarked three OpenMP variants on the same workload (`fib(20..40)`), with 5 runs per thread count:
- `tasks` (`tasks.c`)
- `parallel for (static)` (`fib_omp.c`)
- `parallel for (dynamic,1)` (`fib_omp_dynamic.c`)

Raw data:
- `scaling_static_dynamic_tasks.csv`

Plots:
- Mean bars (order: static, dynamic, tasks):
![Scaling comparison: static vs dynamic vs tasks](scaling_static_dynamic_tasks.svg)

Mean timing results (seconds):

| Variant | 1 thread | 2 threads | 4 threads |
|---|---:|---:|---:|
| `parallel for (static)` | 0.981870 | 0.965147 | 0.887515 |
| `parallel for (dynamic,1)` | 0.973128 | 0.613598 | 0.398490 |
| `tasks` | 0.976995 | 0.678197 | 0.381157 |

Interpretation:
- `parallel for (static)` scales poorly because work per iteration is highly uneven, creating load imbalance.
- `parallel for (dynamic,1)` greatly improves balance by letting threads fetch work one iteration at a time.
- `tasks` and `dynamic,1` both scale well for this irregular workload; in this run `tasks` is slightly faster at 4 threads.
- Why `tasks` can be slightly faster than `dynamic,1` here:
  - task scheduling can reduce end-of-loop imbalance (shorter tail) on non-uniform recursive work;
  - `dynamic,1` still pays per-iteration loop-scheduling overhead and relies on loop chunk dispatch;
  - small differences also depend on runtime implementation and run-to-run noise, so this gap is workload/system specific.
