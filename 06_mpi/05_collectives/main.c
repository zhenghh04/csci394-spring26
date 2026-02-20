#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = size;
    int *sendbuf = NULL;
    int my_value = 0;

    if (rank == 0) {
        sendbuf = (int *)malloc((size_t)n * sizeof(int));
        if (!sendbuf) {
            fprintf(stderr, "allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        for (int i = 0; i < n; i++) sendbuf[i] = i + 10;
    }

    MPI_Scatter(sendbuf, 1, MPI_INT, &my_value, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int bcast_val = 0;
    if (rank == 0) bcast_val = 5;
    MPI_Bcast(&bcast_val, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local = my_value + bcast_val;

    int sum = 0;
    MPI_Reduce(&local, &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    int *gathered = NULL;
    if (rank == 0) {
        gathered = (int *)malloc((size_t)size * sizeof(int));
        if (!gathered) {
            fprintf(stderr, "allocation failed\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Gather(&local, 1, MPI_INT, gathered, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("collectives demo\n");
        printf("gathered:");
        for (int i = 0; i < size; i++) printf(" %d", gathered[i]);
        printf("\nreduced_sum=%d\n", sum);
    }

    free(sendbuf);
    free(gathered);
    MPI_Finalize();
    return 0;
}
