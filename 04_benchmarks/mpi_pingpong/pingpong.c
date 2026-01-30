#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t parse_size(const char *s) {
    if (!s || !*s) return 0;
    char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (end == s) return 0;
    if (*end == '\0') return (size_t)v;
    if (end[1] != '\0') return 0;
    switch (*end) {
        case 'k':
        case 'K':
            return (size_t)(v * 1024ULL);
        case 'm':
        case 'M':
            return (size_t)(v * 1024ULL * 1024ULL);
        case 'g':
        case 'G':
            return (size_t)(v * 1024ULL * 1024ULL * 1024ULL);
        default:
            return 0;
    }
}

static void usage(const char *prog, int rank) {
    if (rank != 0) return;
    fprintf(stderr,
        "Usage: %s [--min SIZE] [--max SIZE] [--iters N] [--warmup N]\n"
        "  SIZE supports suffix K/M/G (e.g., 1K, 4M).\n"
        "  Default: --min 1 --max 8M --iters 1000 --warmup 100\n",
        prog);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires exactly 2 MPI ranks.\n");
        }
        MPI_Finalize();
        return 1;
    }

    size_t min_msg = 1;
    size_t max_msg = 8 * 1024 * 1024;
    int iters = 1000;
    int warmup = 100;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--min") == 0 && i + 1 < argc) {
            min_msg = parse_size(argv[++i]);
        } else if (strcmp(argv[i], "--max") == 0 && i + 1 < argc) {
            max_msg = parse_size(argv[++i]);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
        } else {
            usage(argv[0], rank);
            MPI_Finalize();
            return 1;
        }
    }

    if (min_msg == 0 || max_msg == 0 || min_msg > max_msg || iters <= 0 || warmup < 0) {
        usage(argv[0], rank);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("# MPI ping-pong test (2 ranks)\n");
        printf("# size_bytes latency_us bandwidth_MBps\n");
    }

    for (size_t msg = min_msg; msg <= max_msg; msg *= 2) {
        char *buf = (char *)malloc(msg);
        if (!buf) {
            if (rank == 0) fprintf(stderr, "Failed to allocate %zu bytes\n", msg);
            MPI_Finalize();
            return 1;
        }
        memset(buf, 0xAB, msg);

        for (int w = 0; w < warmup; w++) {
            if (rank == 0) {
                MPI_Send(buf, (int)msg, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, (int)msg, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(buf, (int)msg, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf, (int)msg, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = 0.0, t1 = 0.0;
        if (rank == 0) t0 = MPI_Wtime();
        for (int it = 0; it < iters; it++) {
            if (rank == 0) {
                MPI_Send(buf, (int)msg, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, (int)msg, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(buf, (int)msg, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf, (int)msg, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }
        if (rank == 0) {
            t1 = MPI_Wtime();
            double total = t1 - t0; // round-trip time for iters
            double one_way = total / (double)iters / 2.0;
            double latency_us = one_way * 1e6;
            double bw = (double)msg / one_way / (1024.0 * 1024.0);
            printf("%zu %.3f %.3f\n", msg, latency_us, bw);
            fflush(stdout);
        }

        free(buf);
        MPI_Barrier(MPI_COMM_WORLD);

        if (msg > max_msg / 2) break;
    }

    MPI_Finalize();
    return 0;
}
