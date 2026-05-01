#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int value = 0;
    const int tag = 100;

    if (size < 2) {
        if (rank == 0) {
            fprintf(stderr, "Run with at least 2 ranks.\n");
        }
        MPI_Finalize();
        return 1;
    }

    /*
     * Goal:
     * - rank 0 starts with value 12
     * - rank 0 sends to rank 1
     * - intermediate ranks receive from rank-1 and send to rank+1
     * - rank size-1 receives and prints the final value
     */

    if (rank == 0) {
        /* TODO: initialize value and send it to rank 1 */
    } else if (rank < size - 1) {
        /* TODO: receive from rank-1 and forward to rank+1 */
    } else {
        /* TODO: receive from rank-1 and print final value */
    }

    MPI_Finalize();
    return 0;
}
