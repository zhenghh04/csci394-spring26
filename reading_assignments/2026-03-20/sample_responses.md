# Sample Responses

These are sample responses for the March 20 reading assignment. They are written as model answers, not as quoted textbook text.

## 1. What is the three-world model of HPC?

The three-world model of HPC describes high-performance computing as the interaction of the application world, the mathematical or algorithmic world, and the computer-system world. The application world contains the scientific, engineering, or data problem we care about. The mathematical world turns that problem into models, approximations, algorithms, and data structures. The computer-system world is the actual hardware and software stack that runs the computation.

This model is useful because progress in HPC rarely comes from only one world. A faster machine helps, but only if the algorithm exposes enough parallelism and the application can make use of the results. Likewise, a better model or algorithm matters only if it can be implemented efficiently on real memory systems, networks, CPUs, GPUs, or other accelerators. The future of HPC therefore depends on co-design: applications, algorithms, and machines must evolve together.

## 2. Why do the authors think the von Neumann model limits future HPC systems?

The von Neumann model separates memory from processing: data and instructions are stored in memory and moved to the processor when needed. This design has been extremely successful, but it creates a bottleneck for modern HPC because moving data is often slower and more energy-intensive than doing arithmetic. Many large simulations and machine-learning workloads now spend a major fraction of their time waiting for data from memory or moving data between levels of the memory hierarchy and across the network.

The limitation becomes more serious as processors gain more cores and accelerators. Peak FLOP rates can grow faster than memory bandwidth, so the machine may have enormous arithmetic capability that cannot be fully used. Future systems therefore need designs that reduce data movement, improve locality, use heterogeneous accelerators, or place computation closer to the data. In that sense, the traditional von Neumann separation between memory and compute is not well matched to the most data-intensive future workloads.

## 3. Why is cloud computing described as a business model rather than a new computing technology?

Cloud computing is described as a business model because most of its technical ingredients already existed: servers, storage systems, networks, virtualization, containers, schedulers, and distributed software. What changed was the way those resources are packaged and sold. Instead of buying and maintaining local machines, users rent computing capacity on demand and pay for what they use.

This does not mean cloud computing is unimportant. Its value is elasticity, convenience, and operational scale. A user can provision resources quickly, scale up for a temporary workload, and avoid owning idle hardware. But at the hardware and systems level, cloud computing is still built from conventional computing technologies. Its main novelty is the service model and economic organization of computing resources.

## 4. What problem is memory-centric computing trying to solve?

Memory-centric computing tries to solve the "memory wall" problem: the growing gap between how fast processors can compute and how fast data can be supplied to them. In many modern workloads, especially large simulations, graph analytics, and AI, performance is limited less by arithmetic and more by memory capacity, bandwidth, latency, and data movement energy. A processor may be capable of doing many operations per cycle, but that capability is wasted if the needed data cannot arrive quickly enough.

The idea of memory-centric computing is to organize systems around data movement rather than treating memory as a passive storage area behind the CPU. This can include high-bandwidth memory, larger memory pools, persistent memory, near-memory processing, in-memory computing, and architectures that reduce copying between storage, memory, and processors. The goal is not just to make memory bigger, but to make the whole system more efficient for data-intensive computation.

## 5. What makes quantum computing powerful in theory, and what makes it difficult in practice?

Quantum computing is powerful in theory because qubits can use superposition, entanglement, and interference. These properties allow certain algorithms to explore computational states in ways that classical bits cannot directly match. For selected problems, such as integer factoring, unstructured search speedups, and simulation of quantum physical systems, quantum algorithms can offer major theoretical advantages over the best known classical approaches.

In practice, quantum computing is difficult because useful quantum states are fragile. Qubits are affected by noise, decoherence, imperfect gates, measurement errors, and interactions with the environment. Correcting these errors requires many physical qubits to produce a smaller number of reliable logical qubits, which greatly increases the engineering burden. Quantum computers also do not speed up every problem, so the practical challenge is both hardware-related and algorithmic: build stable machines and find workloads where they clearly outperform classical HPC systems.
