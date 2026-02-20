# 03 Send/Recv

First point-to-point message:
- rank 0 sends an integer token to rank 1
- rank 1 receives it and prints it

## Build
```bash
make
```

## Run
```bash
mpiexec -n 2 ./app
```
