# Explicit data movement

This example focuses on the Chapter 17 data clauses.

Concepts:

- `copyin`
- `copyout`
- `create`
- keeping temporary arrays on the device only

Build:

```bash
make ACCFLAGS='-acc -Minfo=accel'
```

Run:

```bash
./app
./app 2000000
```
