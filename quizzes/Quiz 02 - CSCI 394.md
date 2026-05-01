**Name:** \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_

**Dated:** March 18, 2026

# Multiple Choice (8 questions, 8 pts each = 64 pts) - Please highlight the correct answer

1.  In the OpenMP programming model used in class, parallel work is primarily executed by:  
    > A. Independent distributed-memory processes on different nodes  
    > B. Threads in a shared-memory address space  
    > C. GPU kernels only  
    > D. File-system daemons

2.  Which OpenMP directive is commonly used to create a team of threads?  
    > A. \#pragma omp parallel  
    > B. \#pragma omp critical  
    > C. \#pragma omp atomic  
    > D. \#pragma omp flush

3.  Which clause is the best match for safely summing values across threads in a loop?  
    > A. shared(sum)  
    > B. master  
    > C. reduction(+:sum)  
    > D. sections

4.  Which statement about single and master is correct?  
    > A. Both must be executed by every thread  
    > B. single can be executed by any one thread, while master is executed only by thread 0  
    > C. master always creates a new parallel region  
    > D. single is only valid inside an MPI communicator

5.  A race condition occurs when:  
    > A. Multiple threads access shared data and at least one access is a write without proper synchronization  
    > B. A program uses more than one for loop  
    > C. The compiler generates vector instructions  
    > D. The loop bound is greater than the thread count

6.  In MPI, the rank of a process is:  
    > A. The number of threads in an OpenMP team  
    > B. A unique process ID within a communicator  
    > C. The number of messages already sent  
    > D. The index of a GPU on the node

7.  Which MPI collective operation gives every rank the final reduced result?  
    > A. MPI_Gather  
    > B. MPI_Bcast  
    > C. MPI_Reduce  
    > D. MPI_Allreduce

8.  Which statement best describes MPI?  
    > A. It is a shared-memory threading model based on compiler pragmas  
    > B. It is a specification for message-passing libraries used on distributed-memory systems  
    > C. It is only used for file I/O on clusters

# Short Answer (3 questions)

**1. (6%)Consider the following OpenMP code fragment:**

> double sum = 0.0;
>
> double tmp;
>
> \#pragma omp parallel for
>
> for (int i = 0; i \< n; i++) {
>
> tmp = a\[i\] \* b\[i\];
>
> sum += tmp;
>
> }

**Identify the race-condition problem or problems in this code. Rewrite the pragma line so the loop is correct using OpenMP clauses discussed in class. The new pragma should be: \_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_\_**

2\. (12%) A parallel loop runs with 2 OpenMP threads. The iterations have the following approximate costs:

> Iteration 0 -\> 1 second
>
> Iteration 1 -\> 2 seconds
>
> Iteration 2 -\> 9 seconds
>
> Iteration 3 -\> 8 seconds

What would be the total time:

1)  Using \#pragam omp parallel for schedule(static). \_\_\_\_

2)  Using \#pragma omp parallel for schedule(dynamic,1) \_\_\_\_\_

**3. (15%) Assume an MPI program is launched with 4 ranks, and each rank starts with:**

int local = rank + 1;

The program then executes the following collective operations:

int global_sum = 0;

MPI_Allreduce(&local, &global_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

int gathered\[4\] = {-1, -1, -1, -1};

MPI_Allgather(&local, 1, MPI_INT, gathered, 1, MPI_INT, MPI_COMM_WORLD);

int sendbuf\[4\] = {100, 200, 300, 400};

int recv = -1;

MPI_Scatter(sendbuf, 1, MPI_INT, &recv, 1, MPI_INT, 0, MPI_COMM_WORLD);

Answer the following:

1.  After the MPI_Allreduce call, what value should global_sum contain on each rank?

2.  After the MPI_Allgather call, what should the array gathered contain on every rank?

3.  After the MPI_Scatter call, what value should recv contain on ranks 0, 1, 2, and 3?

# Coding exercise (18 pts)

Write MPI code for the following communication pattern:

- Rank 0 starts with the integer value 12.

- Rank 0 sends the value to rank 1.

- Each intermediate rank receives the value from rank - 1 and adds the value by 1, and then forwards the results to rank + 1.

- Rank N-1 receives the value, adds the value by 1 and prints it.

Use blocking MPI_Send and MPI_Recv calls,

MPI_Send(buf, count, datatype, dest, tag, comm);

MPI_Recv(buf, count, datatype, source, tag, comm, &status);

And finish the implementation:

\#include \<mpi.h\>

\#include \<stdio.h\>

int main(int argc, char \*\*argv) {

MPI_Init(&argc, &argv);

int rank = 0;

int size = 0;

MPI_Comm_rank(MPI_COMM_WORLD, &rank);

MPI_Comm_size(MPI_COMM_WORLD, &size);

int value = 12;

const int tag = 100;

if (size \< 2) {

if (rank == 0) {

fprintf(stderr, "Run with at least 2 ranks.\n");

}

MPI_Finalize();

return 1;

}

/\* Please finish the implementation of the code here:

\* Goal:

\* - rank 0 starts with value 12

\* - rank 0 sends to rank 1

\* - intermediate ranks receive from rank-1 and add 1 to the value then send to rank+1

\* - rank size-1 receives and add one to the value and prints it out

\*/

MPI_Finalize();

return 0;

}
