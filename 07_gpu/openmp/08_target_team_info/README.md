# OpenMP Target Team and Thread Inspection

This example lets students observe what `target teams distribute parallel for`
actually created at runtime.

It answers questions such as:

- how many OpenMP teams were launched?
- how many threads were in the team that handled iteration `i = 0`?
- which team and thread handled each loop iteration?
- what changes if we request `num_teams(...)` or `thread_limit(...)`?

What the code records:

- `omp_get_num_teams()`
- `omp_get_team_num()`
- `omp_get_num_threads()`
- `omp_get_thread_num()`
- `omp_is_initial_device()`

Why the example stores data in arrays instead of printing inside the target loop:

- device-side `printf` support varies by compiler and runtime
- copying small arrays back to the host is more reliable for classroom use

Build:

```bash
make
```

Many compilers need explicit offload flags. Examples:

```bash
make OFFLOAD_FLAGS='-fopenmp-targets=nvptx64-nvidia-cuda'
make CC=nvc OFFLOAD_FLAGS='-mp=gpu'
```

Validated A100 build and run sequence with NVIDIA HPC SDK:

```bash
make clean
make CC=nvc OFFLOAD_FLAGS='-mp=gpu'
OMP_TARGET_OFFLOAD=MANDATORY ./app
```

Run with default runtime decisions:

```bash
./app
./app 32
```

Run while requesting a launch shape:

```bash
./app 32 4 8
```

Meaning:

- first argument: `n` loop iterations
- second argument: requested `num_teams`
- third argument: requested `thread_limit`

Example interpretation:

```text
i  team  thread  threads_in_team
0     0       0                8
1     0       1                8
...
8     1       0                8
```

This suggests:

- iteration `0` was handled by team `0`, thread `0`
- the runtime created `8` threads in that team
- later iterations moved to the next team

Important teaching note:

- `num_teams(...)` and `thread_limit(...)` are requests, not absolute guarantees
- the runtime may choose smaller values depending on the device and compiler
- OpenMP teams and threads are the language-level abstraction; on GPUs they often
  map roughly to blocks and threads, but not as a strict language guarantee

Suggested student experiments:

1. Run `./app 32` and inspect the ownership table.
2. Run `./app 32 2 4` and compare the mapping.
3. Run `./app 64 8 32` and see whether the runtime honors the request.
4. Run with `OMP_TARGET_OFFLOAD=MANDATORY` to force true offload or fail visibly.
