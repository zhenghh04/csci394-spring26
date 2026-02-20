# OpenMP `depend` demo

This module isolates a minimal task-dependency example using:
- `#pragma omp task depend(out: ...)`
- `#pragma omp task depend(in: ...)`

`depend_demo.c` builds a 3-stage per-index pipeline:
1. produce `a[i]`
2. transform `a[i] -> b[i]`
3. finalize `b[i] -> c[i]`

Different indices can run concurrently, while dependency order is preserved per index.

`depend_demo_trace.c` is the same dependency pipeline with trace instrumentation.
It emits `depend_demo_trace.json`, which can be opened in Perfetto.

## Build
```bash
make
```

## Run
```bash
OMP_NUM_THREADS=4 ./depend_demo
OMP_NUM_THREADS=4 ./depend_demo_trace
```

## Trace
- Output file: `depend_demo_trace.json`
- Open with: https://ui.perfetto.dev
