# `kernels` matrix-vector multiply

This example shows the Chapter 17 idea that `kernels` asks the compiler to find
parallel loop nests for you.

Concepts:

- `#pragma acc kernels`
- compiler-driven kernel generation
- dense matrix-vector multiply as a clear nested-loop example

Build:

```bash
make ACCFLAGS='-acc -Minfo=accel'
```

Run:

```bash
./app
./app 1024
```
