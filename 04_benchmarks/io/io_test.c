#include <mpi.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

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
        "Usage: %s [--mb N] [--iters N] [--dir PATH] [--chunk-kb N] [--fsync] [--keep]\n"
        "  --mb N        Total MB written per rank per iteration (default 256).\n"
        "  --iters N     Number of iterations (default 1).\n"
        "  --dir PATH    Output directory (default ./io_out).\n"
        "  --chunk-kb N  Write chunk size in KB (default 1024).\n"
        "  --fsync       Call fsync() before close.\n"
        "  --keep        Keep files after test (default: delete).\n",
        prog);
}

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    size_t total_mb = 256;
    int iters = 1;
    const char *out_dir = "./io_out";
    size_t chunk_kb = 1024;
    int do_fsync = 0;
    int keep_files = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--mb") == 0 && i + 1 < argc) {
            total_mb = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--iters") == 0 && i + 1 < argc) {
            iters = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--dir") == 0 && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (strcmp(argv[i], "--chunk-kb") == 0 && i + 1 < argc) {
            chunk_kb = (size_t)strtoull(argv[++i], NULL, 10);
        } else if (strcmp(argv[i], "--fsync") == 0) {
            do_fsync = 1;
        } else if (strcmp(argv[i], "--keep") == 0) {
            keep_files = 1;
        } else {
            usage(argv[0], rank);
            MPI_Finalize();
            return 1;
        }
    }

    if (total_mb == 0 || iters <= 0 || chunk_kb == 0) {
        usage(argv[0], rank);
        MPI_Finalize();
        return 1;
    }

    size_t total_bytes = total_mb * 1024ULL * 1024ULL;
    size_t chunk_bytes = chunk_kb * 1024ULL;
    if (chunk_bytes > total_bytes) chunk_bytes = total_bytes;

    if (rank == 0) {
        if (mkdir(out_dir, 0777) != 0 && errno != EEXIST) {
            fprintf(stderr, "Failed to create output dir %s: %s\n", out_dir, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);

    char *buf = (char *)malloc(chunk_bytes);
    if (!buf) {
        fprintf(stderr, "Rank %d failed to allocate %zu bytes\n", rank, chunk_bytes);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    memset(buf, 0xCD, chunk_bytes);

    double total_time = 0.0;

    for (int it = 0; it < iters; it++) {
        char path[512];
        snprintf(path, sizeof(path), "%s/io_rank%06d_iter%03d.dat", out_dir, rank, it);

        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        int fd = open(path, O_CREAT | O_TRUNC | O_WRONLY, 0666);
        if (fd < 0) {
            fprintf(stderr, "Rank %d failed to open %s: %s\n", rank, path, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        size_t written = 0;
        while (written < total_bytes) {
            size_t to_write = chunk_bytes;
            if (total_bytes - written < chunk_bytes) to_write = total_bytes - written;
            ssize_t rc = write(fd, buf, to_write);
            if (rc < 0) {
                fprintf(stderr, "Rank %d write failed: %s\n", rank, strerror(errno));
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            written += (size_t)rc;
        }

        if (do_fsync) {
            if (fsync(fd) != 0) {
                fprintf(stderr, "Rank %d fsync failed: %s\n", rank, strerror(errno));
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
        }

        if (close(fd) != 0) {
            fprintf(stderr, "Rank %d close failed: %s\n", rank, strerror(errno));
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();
        total_time += (t1 - t0);

        if (!keep_files) {
            unlink(path);
        }
    }

    double min_t = 0.0, max_t = 0.0, sum_t = 0.0;
    MPI_Reduce(&total_time, &min_t, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_time, &sum_t, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double bytes_per_rank = (double)total_bytes * (double)iters;
        double agg_bw = (bytes_per_rank * (double)size) / max_t / (1024.0 * 1024.0);
        double avg_t = sum_t / (double)size;
        printf("# IO test (file-per-rank write)\n");
        printf("# ranks=%d bytes_per_rank=%.0f iters=%d fsync=%s keep=%s\n",
               size, bytes_per_rank, iters, do_fsync ? "yes" : "no", keep_files ? "yes" : "no");
        printf("# time_min_s time_avg_s time_max_s agg_bw_MBps\n");
        printf("%.6f %.6f %.6f %.3f\n", min_t, avg_t, max_t, agg_bw);
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
