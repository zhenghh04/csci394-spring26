#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int next = (rank + 1) % size;
    int prev = (rank - 1 + size) % size;

    int value = 0;
    if (rank == 0) value = 1;

    MPI_Sendrecv_replace(&value, 1, MPI_INT, next, 0, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("rank %d got value=%d from rank %d\n", rank, value, prev);

    MPI_Finalize();
    return 0;
}
