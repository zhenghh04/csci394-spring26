#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n_local = 100000;
    if (argc >= 2) n_local = atoi(argv[1]);
    if (n_local <= 0) {
        if (rank == 0) fprintf(stderr, "Usage: %s [n_local>0]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    double *x = (double *)malloc((size_t)n_local * sizeof(double));
    if (!x) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    for (int i = 0; i < n_local; i++) {
        x[i] = 1.0 + 0.001 * (double)((rank + i) % 1000);
    }

    double local_sq = 0.0;
    for (int i = 0; i < n_local; i++) {
        local_sq += x[i] * x[i];
    }

    double global_sq = 0.0;
    MPI_Allreduce(&local_sq, &global_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    double global_norm = sqrt(global_sq);
    if (rank == 0) {
        printf("allreduce norm demo\n");
        printf("n_local=%d ranks=%d global_l2_norm=%.6f\n", n_local, size, global_norm);
    }

    free(x);
    MPI_Finalize();
    return 0;
}
