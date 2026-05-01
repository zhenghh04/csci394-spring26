#!/bin/bash -l
#PBS -N mnist_ddp_warmup
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A datascience
#PBS -j oe

# Submit from exercises/answers/warmup/ ; runs the full sweep table from
# the warmup README (1, 2, 4 GPUs × {0, 2, 5} warmup epochs).
# Results land in ../../warmup/results/scaling_results.csv.

cd "$PBS_O_WORKDIR"

EX_DIR="$(realpath ../../warmup)"
cd "$EX_DIR"

module load conda
conda activate

EPOCHS=5
BATCH=64
LR=0.01

run() {
  local nproc=$1
  local warmup=$2
  echo "===== nproc=${nproc}  warmup=${warmup} ====="
  mpiexec -np "${nproc}" --ppn "${nproc}" --cpu-bind depth -d 16 \
    ./launcher.sh python3 train_mnist_ddp.py \
      --epochs "${EPOCHS}" --batch-size "${BATCH}" --lr "${LR}" \
      --warmup-epochs "${warmup}"
}

run 1 0
run 2 0
run 4 0
run 4 2
run 4 5

echo
echo "Sweep complete. CSV: ${EX_DIR}/results/scaling_results.csv"
