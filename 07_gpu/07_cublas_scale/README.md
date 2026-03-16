# cuBLAS column scaling

This Chapter 18 example follows the cuBLAS idea from the textbook: use a vendor
library rather than writing a custom kernel.

Concepts:

- `cudaMalloc`
- `cublasCreate`
- `cublasSetMatrix`
- `cublasSscal`
- `cublasGetMatrix`

Build:

```bash
make
```

Run:

```bash
./app
```
