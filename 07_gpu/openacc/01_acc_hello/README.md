# OpenACC Hello

This is the smallest OpenACC offload sanity check in the sequence.

What this example does:

- enters a tiny OpenACC parallel region
- writes one scalar value inside the region
- prints whether the program was compiled with OpenACC support

Concepts:

- `#pragma acc parallel`
- smallest possible offload example
- checking whether OpenACC support is enabled at compile time

Build:

```bash
make ACCFLAGS='-acc -Minfo=accel'
```

Run:

```bash
./app
```
