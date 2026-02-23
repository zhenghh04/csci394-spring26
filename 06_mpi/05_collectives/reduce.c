#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int local = rank + 1;
    int global_sum = 0;

    MPI_Reduce(&local, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("reduce sum of (rank+1) = %d\n", global_sum);
    }

    MPI_Finalize();
    return 0;
}
