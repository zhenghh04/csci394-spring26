# Simple MPI Code Example

This is a minimal MPI program in C that starts MPI, gets each process rank, prints a message, and then shuts MPI down.

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    printf("Hello from rank %d of %d\n", rank, size);

    MPI_Finalize();
    return 0;
}
```

Compile and run:

```bash
mpicc main.c -o app
mpiexec -n 4 ./app
```

Expected idea:

- `MPI_Init` starts the MPI environment.
- `MPI_COMM_WORLD` is the default communicator containing all ranks.
- `MPI_Comm_rank` gives the ID of the current process.
- `MPI_Comm_size` gives the total number of processes.
- `MPI_Finalize` shuts MPI down cleanly.

Typical output:

```text
Hello from rank 0 of 4
Hello from rank 1 of 4
Hello from rank 2 of 4
Hello from rank 3 of 4
```

The print order is not guaranteed.
