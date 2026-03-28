# OpenACC Collapse(2)

This example is the OpenACC counterpart to the 2D OpenMP target loop with
`collapse(2)`.

Concepts:

- `#pragma acc parallel loop collapse(2)`
- 2D loop nests on an accelerator
- matrix-style elementwise work

Build:

```bash
make ACCFLAGS='-acc -Minfo=accel'
```
