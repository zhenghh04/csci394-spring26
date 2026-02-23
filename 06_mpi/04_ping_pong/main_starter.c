#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Run with at least 2 ranks.\n");
        }
        MPI_Finalize();
        return 1;
    }

    // TODO 1: parse args (msg_bytes, iters, warmup) with defaults.
    int msg_bytes = 1 << 20;  // 1 MiB
    int iters = 100;
    int warmup = 10;

    // TODO 2: allocate and initialize buffer.
    char *buf = (char *)malloc((size_t)msg_bytes);

    // TODO 3: warmup ping-pong (tag 0) between rank 0 and rank 1.

    MPI_Barrier(MPI_COMM_WORLD);

    // TODO 4: timed ping-pong (tag 1), measure with MPI_Wtime().
    double t0 = MPI_Wtime();

    // ... timed loop here ...

    double t1 = MPI_Wtime();

    // TODO 5: on rank 0, compute and print:
    // round_trip_us = (t1 - t0) / iters * 1e6
    // bw_gbps = (2 * msg_bytes * iters) / (t1 - t0) / 1e9

    free(buf);
    MPI_Finalize();
    return 0;
}
