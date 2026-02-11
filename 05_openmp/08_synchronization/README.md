# OpenMP synchronization

This folder contains small examples of common OpenMP synchronization constructs.

Files:
- `atomic_counter.c` – uses `#pragma omp atomic` for thread-safe counter increments.
- `critical_counter.c` – uses `#pragma omp critical` to protect a shared counter.
- `barrier_demo.c` – uses `#pragma omp barrier` so all threads wait at a synchronization point.
- `single_master_demo.c` – compares `#pragma omp single` and `#pragma omp master`.

## What each construct means
`atomic`:
`#pragma omp atomic` protects one simple memory update (for example `x += 1`) so it happens safely without races.  
Use it for lightweight synchronization on a single variable update. It is usually faster than `critical` for this narrow case.

`critical`:
`#pragma omp critical` allows only one thread at a time to execute a protected block.  
Use it when you need to protect multiple statements or more complex shared-state updates that `atomic` cannot express directly.

`barrier`:
`#pragma omp barrier` forces all threads in the team to wait until everyone reaches that point.  
Use it when later work depends on all threads finishing an earlier phase.

`single`:
`#pragma omp single` means exactly one thread (any thread) executes the block once.  
Use it for one-time actions inside a parallel region (for example task creation, initialization, or one print/report step).

`master`:
`#pragma omp master` means only thread 0 executes the block.  
Use it when a step must be done specifically by the master thread (for example code that assumes `tid==0` ownership).

`single` vs `master`:
- `single` can be executed by any thread and has an implicit barrier at the end (unless `nowait` is used).
- `master` is always thread 0 and has no implicit barrier.

## Quick comparison: `atomic` vs `critical` vs `single`
- `atomic`: thread-safe update of one memory location (simple operation like increment/add). Best for lightweight counters/accumulators.
- `critical`: mutual exclusion for an arbitrary code block. Use for complex shared-state updates; higher overhead than `atomic`.
- `single`: run a block once by one thread in a parallel region (others skip). Use for one-time setup/task creation, not per-iteration shared updates.

## Build
```bash
make
```

## Run
```bash
OMP_NUM_THREADS=4 ./atomic_counter
OMP_NUM_THREADS=4 ./critical_counter
OMP_NUM_THREADS=4 ./barrier_demo
OMP_NUM_THREADS=4 ./single_master_demo
```

## Notes
- `atomic` is usually lighter-weight than `critical` for simple updates like `x += 1`.
- `critical` is more general and can protect larger code blocks.
- `single` is executed by one arbitrary thread (with an implicit barrier at end unless `nowait`).
- `master` is executed only by thread 0 (no implicit barrier).
