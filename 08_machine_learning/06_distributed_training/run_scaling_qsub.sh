#!/bin/bash
#PBS -A DLIO
#PBS -q debug
#PBS -l select=2:system=polaris
#PBS -l walltime=00:30:00
#PBS -l filesystems=home:eagle
#PBS -l place=scatter
#PBS -N cifar10_ddp_scale
#PBS -j oe

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RESULTS_DIR="${SCRIPT_DIR}/results"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_cifar10_ddp.py"
NGPUS_PER_NODE=4

cd "${PBS_O_WORKDIR:-$SCRIPT_DIR}"

module use /soft/modulefiles
module load conda
conda activate

mkdir -p "${RESULTS_DIR}"

NODEFILE_UNIQ=$(mktemp)
sort -u "${PBS_NODEFILE}" > "${NODEFILE_UNIQ}"
MASTER_ADDR=$(head -n 1 "${NODEFILE_UNIQ}")
NUM_NODES=$(wc -l < "${NODEFILE_UNIQ}")

echo "=============================================="
echo "CIFAR-10 DDP scaling study on Polaris"
echo "Job ID: ${PBS_JOBID}"
echo "Working directory: ${SCRIPT_DIR}"
echo "Allocated nodes: ${NUM_NODES}"
echo "Master address: ${MASTER_ADDR}"
echo "Node list:"
cat "${NODEFILE_UNIQ}"
echo "Python: $(which python)"
echo "Torch:  $(python -c 'import torch; print(torch.__version__, torch.__file__)')"
echo "=============================================="

run_case() {
    local ngpus=$1
    local ppn=$2
    local nodes_used=$(( (ngpus + NGPUS_PER_NODE - 1) / NGPUS_PER_NODE ))
    local master_port=$((29500 + ngpus))

    echo
    echo "----------------------------------------------"
    echo "Running ${ngpus} GPU(s)"
    echo "Processes: ${ngpus}"
    echo "GPUs per node: ${ppn}"
    echo "Nodes used: ${nodes_used}"
    echo "MASTER_PORT: ${master_port}"
    echo "----------------------------------------------"

    mpiexec -n "${ngpus}" --ppn "${ppn}" \
        --hostfile "${NODEFILE_UNIQ}" \
        --cpu-bind depth --depth 8 \
        --envall \
        bash -lc "
            cd '${SCRIPT_DIR}'
            unset PYTHONPATH
            export PYTHONNOUSERSITE=1
            export MASTER_ADDR='${MASTER_ADDR}'
            export MASTER_PORT='${master_port}'
            export WORLD_SIZE='${ngpus}'
            export RANK=\${PMI_RANK}
            export LOCAL_RANK=\${PMI_LOCAL_RANK}
            python '${TRAIN_SCRIPT}' --epochs 3 --batch-size 64 --results-dir '${RESULTS_DIR}'
        "
}

run_case 1 1
run_case 2 2
run_case 4 4
run_case 8 4

echo
echo "Scaling runs complete."
echo "Results file: ${RESULTS_DIR}/scaling_results.csv"

rm -f "${NODEFILE_UNIQ}"
