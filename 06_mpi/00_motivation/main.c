#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long n = 100000000;
    if (argc >= 2) n = atoll(argv[1]);
    if (n <= 0) {
        if (rank == 0) fprintf(stderr, "Usage: %s [N>0]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    long long chunk = n / size;
    long long rem = n % size;
    long long start = rank * chunk + (rank < rem ? rank : rem);
    long long count = chunk + (rank < rem ? 1 : 0);
    long long end = start + count;

    double t0 = MPI_Wtime();
    double local = 0.0;
    for (long long i = start; i < end; i++) {
        double x = (double)(i + 1);
        local += 1.0 / (x * x);
    }

    double global = 0.0;
    MPI_Reduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    double t1 = MPI_Wtime();

    if (rank == 0) {
        printf("MPI motivation: distributed sum of 1/i^2\n");
        printf("N=%lld ranks=%d\n", n, size);
        printf("sum=%.12f elapsed_s=%.6f\n", global, t1 - t0);
    }

    MPI_Finalize();
    return 0;
}
