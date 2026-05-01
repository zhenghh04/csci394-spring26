#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2 || size % 2 != 0) {
        if (rank == 0) {
            fprintf(stderr, "Error: need an even number of ranks (got %d).\n", size);
        }
        MPI_Finalize();
        return 1;
    }

    int msg_bytes = 1048576;  /* 1 MiB default */
    int iters     = 200;
    int warmup    = 50;

    if (argc >= 2) msg_bytes = atoi(argv[1]);
    if (argc >= 3) iters     = atoi(argv[2]);
    if (argc >= 4) warmup    = atoi(argv[3]);

    if (msg_bytes <= 0 || iters <= 0 || warmup < 0) {
        if (rank == 0) {
            fprintf(stderr,
                    "Usage: %s [message_bytes>0] [iterations>0] [warmup>=0]\n",
                    argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    int half = size / 2;
    int partner;
    if (rank < half) {
        partner = rank + half;
    } else {
        partner = rank - half;
    }

    char *buf = (char *)malloc((size_t)msg_bytes);
    if (!buf) {
        fprintf(stderr, "rank %d: allocation failed for %d bytes\n", rank, msg_bytes);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < msg_bytes; i++) {
        buf[i] = (char)(i & 0x7F);
    }

    /* ---------- warmup ---------- */
    for (int w = 0; w < warmup; w++) {
        if (rank < half) {
            MPI_Send(buf, msg_bytes, MPI_CHAR, partner, 0, MPI_COMM_WORLD);
            MPI_Recv(buf, msg_bytes, MPI_CHAR, partner, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(buf, msg_bytes, MPI_CHAR, partner, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Send(buf, msg_bytes, MPI_CHAR, partner, 0, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    /* ---------- measured iterations ---------- */
    double t0 = MPI_Wtime();

    for (int i = 0; i < iters; i++) {
        if (rank < half) {
            MPI_Send(buf, msg_bytes, MPI_CHAR, partner, 1, MPI_COMM_WORLD);
            MPI_Recv(buf, msg_bytes, MPI_CHAR, partner, 1, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
        } else {
            MPI_Recv(buf, msg_bytes, MPI_CHAR, partner, 1, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            MPI_Send(buf, msg_bytes, MPI_CHAR, partner, 1, MPI_COMM_WORLD);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t1 = MPI_Wtime();
    double local_elapsed = t1 - t0;

    /* bottleneck time across all ranks */
    double max_elapsed = 0.0;
    MPI_Reduce(&local_elapsed, &max_elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);

    if (rank == 0) {
        /*
         * Each pair exchanges 2 * msg_bytes per iteration (send + recv).
         * There are 'half' independent pairs.
         */
        double bw_GBps = (2.0 * (double)msg_bytes * (double)iters * (double)half)
                         / max_elapsed / 1e9;
        printf("INJECTION system=crux nodes=2 total_ranks=%d message_bytes=%d "
               "iters=%d warmup=%d max_elapsed_s=%.9f bw_GBps=%.6f\n",
               size, msg_bytes, iters, warmup, max_elapsed, bw_GBps);
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
