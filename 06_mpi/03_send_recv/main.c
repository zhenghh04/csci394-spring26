#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) fprintf(stderr, "Run with at least 2 ranks.\n");
        MPI_Finalize();
        return 1;
    }

    const int tag = 0;
    int token = 0;

    if (rank == 0) {
        MPI_Status st;
        for (int src = 1; src < size; src += 1) {
            MPI_Recv(&token, 1, MPI_INT, src, tag, MPI_COMM_WORLD, &st);
            printf("rank 0 received token=%d from rank %d\n", token, st.MPI_SOURCE);
        }
    } else {
        token = 1000 + rank;
        MPI_Send(&token, 1, MPI_INT, 0, tag, MPI_COMM_WORLD);
        printf("rank %d sent token=%d to rank 0\n", rank, token);
    }

    MPI_Finalize();
    return 0;
}
