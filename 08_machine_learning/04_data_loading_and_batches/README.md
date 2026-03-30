# 04 Data Loading and Batches: Input Pipeline Basics

This lesson isolates the data path from the model so students can see that
machine-learning performance is not only about matrix math.

Topics to cover:

- dataset objects
- batching
- shuffling
- train/test split
- why data-loading overhead can limit throughput

Learning goals:

- understand how batch size changes memory use and performance
- see why reproducible shuffling and splits matter
- connect machine-learning input pipelines to the broader HPC idea of data movement

Suggested exercise:

- measure one epoch time for several batch sizes
- compare shuffled versus unshuffled loading
- record throughput in samples per second

Suggested run:

```bash
python3 main.py --batch-size 64
python3 main.py --batch-size 256
python3 main.py --batch-size 1024
```
