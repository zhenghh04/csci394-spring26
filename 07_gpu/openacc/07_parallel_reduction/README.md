# OpenACC Parallel Reduction

This example is the direct OpenACC counterpart to the OpenMP target reduction
lesson.

Mathematical operation:

```c
sum = sum_i x[i] * x[i]
```

Concepts:

- `#pragma acc parallel loop`
- `reduction(+:...)`
- scalar result from an accelerator kernel

Build:

```bash
make ACCFLAGS='-acc -Minfo=accel'
```
