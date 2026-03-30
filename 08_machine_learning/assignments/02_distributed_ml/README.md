# Assignment 02: Distributed Machine Learning

Extend a single-process MNIST training code to run in distributed mode.

Minimum requirements:

1. launch with multiple processes
2. partition or shard the training data by rank
3. synchronize model updates correctly
4. report timing and final accuracy
5. include one scaling study

Recommended study:

- `1` process
- `2` processes
- `4` processes if hardware allows

Questions to answer:

1. What speedup do you observe?
2. Where does scaling begin to flatten out?
3. How does communication cost affect efficiency?
4. Does distributed execution change model accuracy?

Deliverables:

- source code
- run commands
- raw timing logs
- one short report with a scaling plot

Notes:

- a single-node multi-process study is acceptable if multi-node resources are unavailable
- explain your hardware and software environment clearly
