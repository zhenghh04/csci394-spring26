#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Run with at least 2 ranks.\n");
        }
        MPI_Finalize();
        return 1;
    }

    int value = 0;
    const int forward_tag = 100;

    if (rank == 0) {
        value = 12;
        printf("rank 0 starts with value=%d\n", value);
        MPI_Send(&value, 1, MPI_INT, 1, forward_tag, MPI_COMM_WORLD);
    } else if (rank < size - 1) {
        MPI_Recv(&value, 1, MPI_INT, rank - 1, forward_tag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        printf("rank %d received value=%d and forwards to rank %d\n",
               rank, value, rank + 1);
        MPI_Send(&value, 1, MPI_INT, rank + 1, forward_tag, MPI_COMM_WORLD);
    } else {
        MPI_Recv(&value, 1, MPI_INT, rank - 1, forward_tag, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
        printf("rank %d received final value=%d\n", rank, value);
    }

    MPI_Finalize();
    return 0;
}
