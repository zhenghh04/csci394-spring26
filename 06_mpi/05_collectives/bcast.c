#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int value = 0;
    if (rank == 0) {
        value = 42;
    }

    MPI_Bcast(&value, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("rank %d received bcast value=%d\n", rank, value);

    MPI_Finalize();
    return 0;
}
