# Loop and reduction

This example maps directly to the Chapter 17 discussion of `loop` and
`reduction`.

Concepts:

- `#pragma acc parallel`
- `#pragma acc loop independent`
- `reduction(+:...)`

Build:

```bash
make ACCFLAGS='-acc -Minfo=accel'
```

Run:

```bash
./app
./app 1000000
```
