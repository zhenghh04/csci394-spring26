# Quiz 02 - CSCI 394 — Answer Key

**Dated:** March 18, 2026

## Multiple Choice (8 questions, 5 pts each = 40 pts)

1. In the OpenMP programming model used in class, parallel work is primarily executed by:
   A. Independent distributed-memory processes on different nodes
   **B. Threads in a shared-memory address space ✅**
   C. GPU kernels only
   D. File-system daemons

2. Which OpenMP directive is commonly used to create a team of threads?
   **A. `#pragma omp parallel` ✅**
   B. `#pragma omp critical`
   C. `#pragma omp atomic`
   D. `#pragma omp flush`

3. Which clause is the best match for safely summing values across threads in a loop?
   A. `shared(sum)`
   B. `master`
   **C. `reduction(+:sum)` ✅**
   D. `sections`

4. Which statement about `single` and `master` is correct?
   A. Both must be executed by every thread
   **B. `single` can be executed by any one thread, while `master` is executed only by thread 0 ✅**
   C. `master` always creates a new parallel region
   D. `single` is only valid inside an MPI communicator

5. A race condition occurs when:
   **A. Multiple threads access shared data and at least one access is a write without proper synchronization ✅**
   B. A program uses more than one `for` loop
   C. The compiler generates vector instructions
   D. The loop bound is greater than the thread count

6. In MPI, the rank of a process is:
   A. The number of threads in an OpenMP team
   **B. A unique process ID within a communicator ✅**
   C. The number of messages already sent
   D. The index of a GPU on the node

7. Which MPI collective operation gives every rank the final reduced result?
   A. `MPI_Gather`
   B. `MPI_Bcast`
   C. `MPI_Reduce`
   **D. `MPI_Allreduce` ✅**

8. Which statement best describes MPI?
   A. It is a shared-memory threading model based on compiler pragmas
   **B. It is a specification for message-passing libraries used on distributed-memory systems ✅**
   C. It is only used for file I/O on clusters
   D. It can be used only by rank 0

## Short Answer (2 questions, 15 pts each = 30 pts)

### 1. Race in OpenMP loop

```c
double sum = 0.0;
double tmp;

#pragma omp parallel for
for (int i = 0; i < n; i++) {
    tmp = a[i] * b[i];
    sum += tmp;
}
```

**(1) Race conditions:**
- `tmp` is declared **outside** the parallel region, so it is shared by default. Every thread writes the same `tmp` location → values clobber each other.
- `sum += tmp` is an unsynchronized read-modify-write on the shared `sum`.

**(2) Corrected pragma:**
```c
#pragma omp parallel for private(tmp) reduction(+:sum)
```
(Equivalently, declare `double tmp;` *inside* the loop body and use `reduction(+:sum)`.)

**(3) Why dynamic schedule / tasks help with irregular workloads:**
With irregular per-iteration cost, a `static` schedule pre-assigns chunks at compile time, so threads that finish their light iterations sit idle while others still grind through heavy ones. `schedule(dynamic, chunk)` (or OpenMP tasks) hands work out at runtime — as soon as a thread finishes a chunk it grabs the next one, so the load is balanced and idle time is reduced.

### 2. Load balance with 4 threads, costs {1, 9, 2, 8}

**(1) Why `schedule(static)` is poor:** With 4 iterations and 4 threads, each thread gets one iteration: T0→1, T1→9, T2→2, T3→8. Threads 0 and 2 finish in ~1–2 units and then idle, while threads 1 and 3 keep running for 8–9 units. Wall time ≈ 9 units, with utilization only (1+9+2+8)/(4·9) ≈ 56%.

**(2) Why `schedule(dynamic, 1)` improves utilization:** Iterations are placed in a runtime queue. Whichever thread is free pulls the next iteration. So after T0 finishes the 1-unit iteration, it can pick up another piece of work instead of idling. The fast-finishing threads keep helping until the queue is empty, balancing the total work (~20 units) across 4 threads.

**(3) Shared vs private variables:**
- **Shared**: a single memory location visible to and writable by all threads in the team.
- **Private**: each thread gets its own (uninitialized) copy that is invisible to other threads; lifetime is the parallel region.

## MPI Question 1 (15 pts)

With 4 ranks, `local = rank + 1` gives values **1, 2, 3, 4**.

**(1) `MPI_Allreduce` (SUM):** every rank ends with `global_sum = 1+2+3+4 = `**`10`**.

**(2) `MPI_Allgather`:** every rank's `gathered` array is **`{1, 2, 3, 4}`** (in rank order).

**(3) `MPI_Scatter` from rank 0 of `{100, 200, 300, 400}`:**
| Rank | `recv` |
|------|-------|
| 0    | 100   |
| 1    | 200   |
| 2    | 300   |
| 3    | 400   |

**(4) Communication patterns:**
- **`MPI_Allreduce`** — all-to-all reduction: every rank contributes a value, an operator (e.g. SUM) combines them, and the same scalar result is delivered to every rank.
- **`MPI_Allgather`** — all-to-all gather (no reduction): every rank's value is concatenated into an array delivered to every rank in rank order.
- **`MPI_Scatter`** — one-to-many: a single root rank splits an array into chunks and sends a different chunk to each rank.

## MPI Question 2 (15 pts) — Chain send

```c
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    int rank = 0, size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int value = 12;
    const int tag = 100;

    if (size < 2) {
        if (rank == 0) fprintf(stderr, "Run with at least 2 ranks.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        /* start of the chain */
        MPI_Send(&value, 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
    } else {
        MPI_Status status;
        MPI_Recv(&value, 1, MPI_INT, rank - 1, tag, MPI_COMM_WORLD, &status);
        value += 1;                               /* add 1 as it passes through */
        if (rank < size - 1) {
            MPI_Send(&value, 1, MPI_INT, rank + 1, tag, MPI_COMM_WORLD);
        } else {
            /* last rank prints the final value */
            printf("Rank %d received final value: %d\n", rank, value);
        }
    }

    MPI_Finalize();
    return 0;
}
```

**Expected output (with N=4):** `Rank 3 received final value: 15` (12 incremented once at each of ranks 1, 2, 3).
