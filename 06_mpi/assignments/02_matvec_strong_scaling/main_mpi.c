#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void fill_inputs(double *a, double *x, int n) {
    for (int i = 0; i < n; ++i) {
        x[i] = 1.0 / (double)(i + 1);
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            a[(size_t)i * (size_t)n + (size_t)j] = 1.0 / (double)(i + j + 1);
        }
    }
}

static double checksum(const double *v, int n) {
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += v[i];
    }
    return s;
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank, nproc;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if (argc < 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <n> [iters] [warmup]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const int n      = atoi(argv[1]);
    const int iters  = (argc >= 3) ? atoi(argv[2]) : 5;
    const int warmup = (argc >= 4) ? atoi(argv[3]) : 1;

    if (n <= 0 || iters <= 0 || warmup < 0) {
        if (rank == 0)
            fprintf(stderr, "Invalid args: n>0, iters>0, warmup>=0 required.\n");
        MPI_Finalize();
        return 1;
    }

    if (n % nproc != 0) {
        if (rank == 0)
            fprintf(stderr, "Error: n (%d) must be divisible by nproc (%d).\n",
                    n, nproc);
        MPI_Finalize();
        return 1;
    }

    int rows_local = n / nproc;

    /* Root allocates full A, x, y; all ranks allocate local buffers */
    double *A_full = NULL;
    double *x_full = (double *)malloc((size_t)n * sizeof(double));
    double *y_full = NULL;

    double *A_local = (double *)malloc((size_t)rows_local * (size_t)n * sizeof(double));
    double *y_local = (double *)malloc((size_t)rows_local * sizeof(double));

    if (!x_full || !A_local || !y_local) {
        fprintf(stderr, "rank %d: allocation failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (rank == 0) {
        A_full = (double *)malloc((size_t)n * (size_t)n * sizeof(double));
        y_full = (double *)malloc((size_t)n * sizeof(double));
        if (!A_full || !y_full) {
            fprintf(stderr, "rank 0: allocation failed for full arrays\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        fill_inputs(A_full, x_full, n);
    }

    /* Accumulators for per-phase timing */
    double t_scatter_sum  = 0.0;
    double t_bcast_sum    = 0.0;
    double t_compute_sum  = 0.0;
    double t_gather_sum   = 0.0;
    double t_total_sum    = 0.0;

    int total_iters = warmup + iters;

    for (int it = 0; it < total_iters; it++) {
        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        /* Scatter rows of A */
        double t0 = MPI_Wtime();
        MPI_Scatter(A_full, rows_local * n, MPI_DOUBLE,
                    A_local, rows_local * n, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        double dt_scatter = t1 - t0;

        /* Broadcast x */
        t0 = MPI_Wtime();
        MPI_Bcast(x_full, n, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        double dt_bcast = t1 - t0;

        /* Local matvec */
        t0 = MPI_Wtime();
        for (int i = 0; i < rows_local; i++) {
            const double *row = A_local + (size_t)i * (size_t)n;
            double acc = 0.0;
            for (int j = 0; j < n; j++) {
                acc += row[j] * x_full[j];
            }
            y_local[i] = acc;
        }
        t1 = MPI_Wtime();
        double dt_compute = t1 - t0;

        /* Gather y */
        t0 = MPI_Wtime();
        MPI_Gather(y_local, rows_local, MPI_DOUBLE,
                   y_full, rows_local, MPI_DOUBLE,
                   0, MPI_COMM_WORLD);
        t1 = MPI_Wtime();
        double dt_gather = t1 - t0;

        double t_end = MPI_Wtime();
        double dt_total = t_end - t_start;

        /* Only accumulate measured iterations (skip warmup) */
        if (it >= warmup) {
            /* Use max across ranks for each phase */
            double max_scatter, max_bcast, max_compute, max_gather, max_total;
            MPI_Reduce(&dt_scatter, &max_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&dt_bcast,   &max_bcast,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&dt_compute, &max_compute,  1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&dt_gather,  &max_gather,   1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&dt_total,   &max_total,    1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                t_scatter_sum += max_scatter;
                t_bcast_sum   += max_bcast;
                t_compute_sum += max_compute;
                t_gather_sum  += max_gather;
                t_total_sum   += max_total;
            }
        }
    }

    if (rank == 0) {
        double cs = checksum(y_full, n);
        printf("MATVEC_MPI n=%d p=%d rows_local=%d iters=%d warmup=%d "
               "scatter_s=%.9f bcast_s=%.9f compute_s=%.9f gather_s=%.9f "
               "total_s=%.9f time_per_iter_s=%.9f checksum=%.12e\n",
               n, nproc, rows_local, iters, warmup,
               t_scatter_sum / (double)iters,
               t_bcast_sum   / (double)iters,
               t_compute_sum / (double)iters,
               t_gather_sum  / (double)iters,
               t_total_sum   / (double)iters,
               t_total_sum   / (double)iters,
               cs);
    }

    free(A_local);
    free(y_local);
    free(x_full);
    if (rank == 0) {
        free(A_full);
        free(y_full);
    }

    MPI_Finalize();
    return 0;
}
