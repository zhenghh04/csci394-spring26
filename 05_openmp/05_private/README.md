# OpenMP `private` example

This example shows how `private` gives each thread its own copy of variables
inside a parallel region.

## Build
```bash
make
```

## Run
```bash
# Example: 4 threads
OMP_NUM_THREADS=4 ./private_example
OMP_NUM_THREADS=4 ./private_race_demo
```

`private_race_demo` intentionally uses a shared `temp` first, then a private
`temp` version. You should see many mismatches in the shared case and zero in
the private case.
