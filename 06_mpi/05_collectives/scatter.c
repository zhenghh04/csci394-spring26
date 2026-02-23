#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *sendbuf = NULL;
    int recv = -1;

    if (rank == 0) {
        sendbuf = (int *)malloc((size_t)size * sizeof(int));
        if (!sendbuf) {
            fprintf(stderr, "allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < size; ++i) {
            sendbuf[i] = 100 + i;
        }
    }

    MPI_Scatter(sendbuf, 1, MPI_INT, &recv, 1, MPI_INT, 0, MPI_COMM_WORLD);
    printf("rank %d received scatter value=%d\n", rank, recv);

    free(sendbuf);
    MPI_Finalize();
    return 0;
}
