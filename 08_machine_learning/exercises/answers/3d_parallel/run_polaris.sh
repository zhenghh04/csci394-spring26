#!/bin/bash -l
#PBS -N transformer_3d_sweep
#PBS -l select=2:ncpus=64:ngpus=4
#PBS -l walltime=00:45:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A datascience
#PBS -j oe

# Sweep TP × PP combinations on 8 GPUs (2 nodes × 4) for the 3D-parallel
# transformer benchmark. Submit from exercises/answers/3d_parallel/.

cd "$PBS_O_WORKDIR"

EX_DIR="$(realpath ../../3d_parallel)"
cd "$EX_DIR"

module use /soft/modulefiles
module load conda
conda activate

run() {
  local tp=$1
  local pp=$2
  echo "===== TP=${tp}  PP=${pp} ====="
  mpiexec -np 8 --ppn 4 --cpu-bind depth -d 16 \
    ./launcher.sh python3 train_transformer_3d_parallel.py \
      --tp-size "${tp}" --pp-size "${pp}"
}

#  TP  PP   DP
run 1   1   # 8
run 2   1   # 4
run 4   1   # 2
run 1   2   # 4
run 2   2   # 2
run 4   2   # 1
run 1   8   # 1
run 2   4   # 1

echo
echo "Sweep complete. CSV: ${EX_DIR}/results_3d_parallel/scaling_results.csv"
