#!/bin/bash -l
#PBS -N tp_matmul_sweep
#PBS -l select=1:ncpus=64:ngpus=4
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -q debug
#PBS -A datascience
#PBS -j oe

# Sweep matrix size and process count for the K-shard tensor-parallel matmul.
# Submit from exercises/answers/tensor_parallel/.

cd "$PBS_O_WORKDIR"

EXT=/lus/$(realpath ./matmul_tensor_parallel_ext.py)
EXT="$(realpath ./matmul_tensor_parallel_ext.py)"

module load conda
conda activate

mkdir -p results

for NPROC in 1 2 4; do
  for SZ in 1024 2048 4096 8192 16384; do
    echo "===== nproc=${NPROC}  size=${SZ} ====="
    if [ "$NPROC" = 1 ]; then
      python3 "$EXT" --size "$SZ" --repeats 5 --csv results/sweep.csv
    else
      torchrun --standalone --nproc_per_node="$NPROC" "$EXT" \
        --size "$SZ" --repeats 5 --csv results/sweep.csv
    fi
  done
done

echo
echo "Sweep complete. CSV: results/sweep.csv"
