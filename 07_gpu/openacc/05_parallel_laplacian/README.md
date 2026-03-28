# OpenACC 2D Laplacian

This example mirrors the OpenMP target Laplacian lesson with OpenACC.

Concepts:

- `#pragma acc parallel loop collapse(2)`
- 2D five-point stencil
- boundary handling with interior loops only

Build:

```bash
make ACCFLAGS='-acc -Minfo=accel'
```
