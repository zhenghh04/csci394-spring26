# Sample Responses

These are sample responses for the March 20 reading assignment. They are based on Chapter 23, "Computing in the Future: A Personal Perspective," from the course textbook, but they are paraphrased model answers rather than quoted textbook text.

## 1. What is the three-world model of HPC?

In this chapter, the three-world model describes the uneven distribution of real HPC capability, especially as seen through systems such as those on the Top500 list. One "world" contains the very small number of leading post-exascale or near-exascale machines. A second world contains the small group of top-tier, hyperperformance systems below those leaders. The third and much larger world contains the systems that most users are more likely to encounter: machines in the low-petaflop to roughly ten-petaflop range.

The point is that saying HPC has entered the post-exascale era can be misleading for ordinary users. Exascale machines may exist, but most scientists, engineers, students, and institutions do not have routine access to them. The practical challenge for the field is therefore not only building the fastest possible machine, but also raising the level of performance available to the broader HPC user community. The same programming skills may carry upward to systems such as Frontier or Aurora, but access and available capability remain very uneven.

## 2. Why do the authors think the von Neumann model limits future HPC systems?

The authors do not dismiss the von Neumann model; they present it as a historically successful design that shaped modern computing. Their argument is that several assumptions inherited from that model now constrain future HPC. One assumption is that arithmetic units are the precious resource that the rest of the machine should work to keep busy. That made sense when arithmetic hardware was large and costly, but today the arithmetic logic is only a small part of a processor core, while much of the surrounding hardware exists to feed and manage it.

Two other limits are especially important for future systems. First, processor logic and main memory are still physically separated, even though both are now built from semiconductor technologies; this separation creates the familiar von Neumann bottleneck in data movement. Second, conventional systems still preserve sequential instruction-fetch and memory-ordering assumptions even when running highly parallel workloads. For decades, Moore's law helped hide the cost of these choices. As those scaling gains weaken, HPC needs more radical architecture ideas rather than only incremental improvements to the same model.

## 3. Why is cloud computing described as a business model rather than a new computing technology?

The chapter describes cloud computing as a business model because the underlying computing technologies are not fundamentally new. Cloud systems still use conventional servers, networks, storage, operating systems, accelerators, and data-center infrastructure. What changes is the arrangement between the provider and the user: instead of buying, operating, staffing, powering, cooling, and eventually replacing an in-house HPC facility, an organization rents capacity and pays according to usage.

This model can reduce up-front cost and make resource spending more flexible, but the authors are cautious about its fit for serious HPC. Many HPC jobs depend on specific accelerators, low-latency placement of nodes, high-performance interconnects, predictable storage, and repeatable large-scale resource availability. Cloud providers are designed for a much broader market, so their systems may not be optimized for tightly coupled parallel jobs. The cloud can expand access to computing, but it is not automatically a replacement for purpose-built HPC facilities.

## 4. What problem is memory-centric computing trying to solve?

Memory-centric computing tries to solve the growing mismatch between processor-centered architectures and the actual cost of accessing data. Traditional HPC systems place processor cores at the center and attach memory through memory hierarchies, networks, and message passing. At scale, this creates delays from latency, software overhead, contention for shared resources, and waiting while data moves to the arithmetic units. As device scaling slows, simply making processors faster is no longer enough.

The textbook's memory-centric alternative reverses the emphasis: instead of treating arithmetic units as scarce and memory as passive storage, it tries to place computation and memory much closer together. Designs such as active memory architecture aim to reduce memory-access latency, reduce contention for memory and network bandwidth, avoid starvation caused by serial instruction flow, and support a new execution model for scalable parallelism. The difficulty is that these systems could disrupt decades of existing applications, libraries, and programming interfaces, so they may first appear as specialized accelerators rather than immediate replacements for current machines.

## 5. What makes quantum computing powerful in theory, and what makes it difficult in practice?

Quantum computing is powerful in theory because it replaces ordinary two-state bits with qubits. A classical bit stores either 0 or 1, while a qubit can represent a superposition of possible states. For certain classes of problems, this could allow a quantum computer with enough reliable qubits to achieve dramatic speedups over conventional supercomputers. The chapter points to cryptography, especially factoring with Shor's algorithm, as a major example of why the idea attracts so much attention.

The practical difficulty is that useful quantum states are extremely fragile. Decoherence, noise, imperfect control, and interaction with the surrounding environment can cause qubits to collapse into ordinary 0 or 1 states before a computation finishes. Researchers are exploring several physical ways to build qubits, including superconducting, optical, trapped-particle, and spin-based approaches, and some systems already operate with dozens or hundreds of qubits. Even so, the textbook presents practical quantum computing as still experimental: promising in principle, but not yet a commercially mature solution for general HPC workloads.
