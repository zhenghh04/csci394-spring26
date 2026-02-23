# 03 Send/Recv

Point-to-point fan-in example:
- every odd rank sends an integer token to rank 0
- rank 0 receives one token from each odd rank and prints them

## Build
```bash
make
```

## Run
```bash
mpiexec -n 4 ./app
# try more ranks
mpiexec -n 8 ./app
```
