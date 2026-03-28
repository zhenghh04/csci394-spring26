# OpenACC Parallel AXPY

This example is the first complete numerical OpenACC offload kernel in the
sequence.

Mathematical operation:

```c
out[i] = a * x[i] + y[i]
```

Concepts:

- `#pragma acc parallel loop`
- automatic data motion with `copyin` and `copyout`
- first offloaded 1D numerical loop

Build:

```bash
make ACCFLAGS='-acc -Minfo=accel'
```

Run:

```bash
./app
./app 4000000
```
