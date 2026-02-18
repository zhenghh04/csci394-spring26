# OpenMP Assignment: Loopy Kernel

## Objective
Parallelize `loopy_assignment.c` with OpenMP and evaluate scaling on Crux (single compute node, up to 64 threads).

## Required Work
Parallelize and time these four regions using `omp_get_wtime()`:
1. Task 1: Matrix initialization
2. Task 2: Row-sum computation
3. Task 3: 2D stencil computation
4. Task 4: Total sum reduction

For Task 3, compare both versions:
1. without `collapse(2)`
2. with `collapse(2)`

## Build Requirement
Create your own `Makefile` in this folder.

Minimum requirement:
1. target `loopy_assignment`
2. builds `loopy_assignment.c` with OpenMP flags

## Run Plan (Crux, Single Node)
Use PBS and run at least these thread counts:
1. `OMP_NUM_THREADS=1` (baseline)
2. `OMP_NUM_THREADS=2`
3. `OMP_NUM_THREADS=4`
4. `OMP_NUM_THREADS=8`
5. continue up to `OMP_NUM_THREADS=64` if available

Example commands:
```bash
cd $HOME/csci394-spring26/05_openmp/assignments
make loopy_assignment

OMP_NUM_THREADS=1 ./loopy_assignment --dim 2048 --output loopy_t1.dat
OMP_NUM_THREADS=2 ./loopy_assignment --dim 2048 --output loopy_t2.dat
OMP_NUM_THREADS=4 ./loopy_assignment --dim 2048 --output loopy_t4.dat
OMP_NUM_THREADS=8 ./loopy_assignment --dim 2048 --output loopy_t8.dat
...
OMP_NUM_THREADS=64 ./loopy_assignment --dim 2048 --output loopy_t64.dat
```

## Submission Checklist
1. Modified `loopy_assignment.c`
2. Your `Makefile`
3. Your PBS submission script
4. Report (`report_loopy_parallel.pdf`) including:
   - correctness checks (sample values and total sum vs 1-thread result)
   - per-task timing table by thread count
   - speedup table (`T1 / Tp`) per task and overall
   - stencil comparison: with vs without `collapse(2)`
   - what you learned from these experiments

## Notes
1. Keep math and output format unchanged.
2. Verify numerical consistency before performance analysis.
