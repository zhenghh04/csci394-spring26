#!/bin/bash
module use /soft/modulefiles
module load conda
conda activate
mpiexec -np 8 --ppn 4 --cpu-bind depth -d 16 ./launcher.sh python3 train_transformer_3d_parallel.py --tp-size 1 --pp-size 1
mpiexec -np 8 --ppn 4 --cpu-bind depth -d 16 ./launcher.sh python3 train_transformer_3d_parallel.py --tp-size 2 --pp-size 1
mpiexec -np 8 --ppn 4 --cpu-bind depth -d 16 ./launcher.sh python3 train_transformer_3d_parallel.py --tp-size 2 --pp-size 4
mpiexec -np 8 --ppn 4 --cpu-bind depth -d 16 ./launcher.sh python3 train_transformer_3d_parallel.py --tp-size 1 --pp-size 8
