# `parallel loop` SAXPY

This is the first OpenACC example for class.

Concepts:

- `#pragma acc parallel loop`
- implicit host-to-device and device-to-host copies
- one loop, one kernel, easy correctness check

Build:

```bash
make ACCFLAGS='-acc'
```

Run:

```bash
./app
./app 4000000
```

The program now reports:

- `kernel_plus_transfer_time`: time around the OpenACC region, including the
  implicit data transfers in the pragma
- `total_time`: total wall-clock time for allocation, initialization, kernel,
  and validation

This makes it easy to compare small and large `N` runs in class.
