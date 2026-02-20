#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long n = 10000000;
    if (argc >= 2) n = atoll(argv[1]);
    if (n <= 0) {
        if (rank == 0) fprintf(stderr, "Usage: %s [N>0]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    double h = 1.0 / (double)n;
    double local = 0.0;

    double t0 = MPI_Wtime();
    for (long long i = rank; i < n; i += size) {
        double x = (i + 0.5) * h;
        local += 4.0 / (1.0 + x * x);
    }

    double global = 0.0;
    MPI_Reduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        double pi = h * global;
        printf("pi_reduce\n");
        printf("N=%lld ranks=%d pi=%.12f elapsed_s=%.6f\n", n, size, pi, t1 - t0);
    }

    MPI_Finalize();
    return 0;
}
