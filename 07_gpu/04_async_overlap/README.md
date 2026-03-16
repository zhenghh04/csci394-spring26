# Async overlap

This example shows host work overlapping with accelerated work.

Concepts:

- `async`
- `wait`
- reduction on the device
- CPU work proceeding before the accelerator result is consumed

Build:

```bash
make ACCFLAGS='-acc'
```

Run:

```bash
./app
./app 8000000
```
