# Exercise Answers — CSCI394 Distributed Deep Learning

Written answers to the three exercises under `../`:

| Exercise | Folder | Topic |
|---|---|---|
| Warmup | [warmup/](warmup/) | DDP linear LR scaling and warmup on MNIST |
| Tensor parallel | [tensor_parallel/](tensor_parallel/) | K-sharded matmul with `all_reduce(SUM)` |
| 3D parallel | [3d_parallel/](3d_parallel/) | TP × PP × DP transformer benchmark |

Each folder contains:

- `report.md` — answers to the README questions, with reasoning grounded in the source code.
- `run_polaris.sh` — a runnable PBS submission script for the recommended sweep on Polaris.
- (tensor_parallel only) `matmul_tensor_parallel_ext.py` — extension that accepts `M, K, N` from the CLI and supports a sweep.

Numerical results (CSVs / plots) are produced when the run scripts are executed on Polaris;
the answers are written so that the qualitative conclusions can be verified against
those CSVs without re-deriving the analysis.
