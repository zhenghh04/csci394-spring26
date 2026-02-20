#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    int send_right = rank;
    int recv_left = -1;

    MPI_Request reqs[2];
    MPI_Irecv(&recv_left, 1, MPI_INT, left, 0, MPI_COMM_WORLD, &reqs[0]);
    MPI_Isend(&send_right, 1, MPI_INT, right, 0, MPI_COMM_WORLD, &reqs[1]);
    MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE);

    printf("rank %d received %d from left neighbor %d\n", rank, recv_left, left);

    MPI_Finalize();
    return 0;
}
