# OpenACC Data Region

This example separates data movement from computation using an explicit
OpenACC data region.

Mathematical operation:

```c
out[i] = x[i] * x[i] + y[i] * y[i]
```

Concepts:

- `#pragma acc data`
- `present(...)`
- explicit lifetime for device data
- separating setup time from kernel time

Build:

```bash
make ACCFLAGS='-acc -Minfo=accel'
```

Run:

```bash
./app
./app 4000000
```
