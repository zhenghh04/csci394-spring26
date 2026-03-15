#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int count = 1024;
    if (argc >= 2) {
        count = atoi(argv[1]);
    }
    if (count <= 0) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s [count>0]\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    int *send_right = (int *)malloc((size_t)count * sizeof(int));
    int *recv_left = (int *)malloc((size_t)count * sizeof(int));
    if (!send_right || !recv_left) {
        fprintf(stderr, "rank %d allocation failed for count=%d\n", rank, count);
        free(send_right);
        free(recv_left);
        MPI_Finalize();
        return 1;
    }
    for (int i = 0; i < count; i++) {
        send_right[i] = rank;
        recv_left[i] = -1;
    }

    MPI_Request reqs[2];
    MPI_Irecv(recv_left, count, MPI_INT, left, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(send_right, count, MPI_INT, right, 0, MPI_COMM_WORLD, &reqs[1]);
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    printf("rank %d received %d ints from left neighbor %d (first=%d last=%d)\n",
           rank, count, left, recv_left[0], recv_left[count - 1]);

    free(send_right);
    free(recv_left);

    MPI_Finalize();
    return 0;
}
