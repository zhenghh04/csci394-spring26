# OpenMP `private` example

`private` means each thread gets its own independent copy of a variable inside
a parallel region. Use it when a variable is temporary per-thread state (for
example loop-local scratch values) and should not be shared across threads.
This avoids data races caused by multiple threads writing the same variable.

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
```

`private_example` runs three cases:
- serial baseline
- parallel `for` without `private` (shared temp, race-prone)
- parallel `for` with `private` (correct)

Example output (`OMP_NUM_THREADS=4`):
```text
Compared to serial baseline (n=40000)
parallel_for_without_private mismatches = 39999
parallel_for_with_private    mismatches = 0
serial first 10 results:
  serial[0] = 0.00, parallel[0] = 0.00, parallel_no_private[0] = 2826000000.00
  serial[1] = 3.14, parallel[1] = 3.14, parallel_no_private[1] = 2826376812.56
  serial[2] = 12.56, parallel[2] = 12.56, parallel_no_private[2] = 2826565228.26
  serial[3] = 28.26, parallel[3] = 28.26, parallel_no_private[3] = 1256502450.24
  serial[4] = 50.24, parallel[4] = 50.24, parallel_no_private[4] = 314251250.24
```
