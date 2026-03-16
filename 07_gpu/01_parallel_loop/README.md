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
