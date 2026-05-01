# Quiz 01 - CSCI 394 — Answer Key

**Dated:** Feb 7, 2026

## Multiple Choice (8 questions, 8 pts each)

1. Which is the slide's definition of architecture in HPC?
   A. The physical layout of racks and cables
   **B. The organization and functionality of constituent components and the logical ISA presented to programs ✅**
   C. The programming model used by applications
   D. The performance ranking of the system

2. In HPC, "performance" is generally measured as:
   A. Peak clock frequency
   **B. Work accomplished per unit time ✅**
   C. Number of nodes in the cluster
   D. Size of RAM per node

3. Which best describes Peak FLOPS vs Sustained FLOPS?
   A. Peak is measured on real apps; sustained is theoretical
   **B. Peak is theoretical maximum; sustained is what apps achieve ✅**
   C. Both are identical if MPI is used
   D. Sustained is always higher than peak

4. Which is NOT listed in the SLOWER performance model?
   A. Starvation
   B. Latency
   C. Overhead
   **D. Accuracy ✅**
   *(SLOWER = Starvation, Latency, Overhead, Waiting, Execution, Resource contention)*

5. Which is a throughput metric?
   A. Wall-clock time
   **B. FLOPS ✅**
   C. Cache miss rate
   D. Serial fraction

6. Strong scaling means:
   A. Problem size increases with core count
   **B. Execution time decreases as you add processors for a fixed problem size ✅**
   C. Execution time stays constant as you add processors for a fixed problem size
   D. Only GPU count increases, not CPU

7. Weak scaling means:
   A. Fixed total problem size, more processors
   **B. Problem size grows with core count, aiming for constant time ✅**
   C. Time must decrease linearly with processors
   D. Only applies to I/O benchmarks

8. HPL (High Performance LINPACK) primarily measures performance on:
   A. Sparse matrix-vector multiply
   **B. Dense linear system solve (Ax=b) ✅**
   C. File metadata operations
   D. MPI ping-pong latency only

## Short Answer (2 questions, 18 pts each)

### 1. Theoretical peak performance

System: 8192 nodes × 2 sockets × 64 cores × 2.5 GHz × 32 FLOP/cycle.

**(1) Peak performance**

Per node:
$$2 \times 64 \times 2.5 \times 10^9 \times 32 = 1.024 \times 10^{13}\ \text{FLOP/s} = 10.24\ \text{TFLOP/s}$$

System total:
$$8192 \times 10.24\ \text{TFLOP/s} = 8.389 \times 10^{16}\ \text{FLOP/s} \approx \mathbf{83.89\ PFLOP/s}$$

**(2) Two reasons sustained < peak (any two):**
- Memory-bandwidth bottleneck — the cores can't be fed operands fast enough to issue an FMA every cycle.
- Network/communication and synchronization overhead between nodes.
- Load imbalance across processes/threads.
- Cache misses, branch mispredictions, pipeline stalls.
- Real codes contain non-FMA instructions (loads, integer ops, control flow).
- I/O and OS jitter.

### 2. Amdahl's Law, f = 0.08, p = 16

Amdahl: $S(p) = \dfrac{1}{f + (1-f)/p}$

**Speedup:**
$$S(16) = \frac{1}{0.08 + 0.92/16} = \frac{1}{0.08 + 0.0575} = \frac{1}{0.1375} \approx \mathbf{7.27}$$

**Parallel efficiency:**
$$E(16) = \frac{S(16)}{16} \approx \frac{7.27}{16} \approx \mathbf{0.454\ (45.4\%)}$$

**Maximum speedup as p → ∞:**
$$S_\infty = \frac{1}{f} = \frac{1}{0.08} = \mathbf{12.5}$$
