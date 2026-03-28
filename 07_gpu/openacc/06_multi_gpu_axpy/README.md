# OpenACC Multi-GPU AXPY

This example mirrors the OpenMP multi-GPU AXPY lesson with OpenACC.

Concepts:

- explicit device selection
- manual vector partitioning
- one chunk per accelerator device

Build:

```bash
make ACCFLAGS='-acc -Minfo=accel'
```
