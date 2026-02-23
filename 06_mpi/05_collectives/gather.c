#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int send = rank * 10;
    int *recvbuf = NULL;

    if (rank == 0) {
        recvbuf = (int *)malloc((size_t)size * sizeof(int));
        if (!recvbuf) {
            fprintf(stderr, "allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Gather(&send, 1, MPI_INT, recvbuf, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("rank 0 gathered:");
        for (int i = 0; i < size; ++i) {
            printf(" %d", recvbuf[i]);
        }
        printf("\n");
    }

    free(recvbuf);
    MPI_Finalize();
    return 0;
}
