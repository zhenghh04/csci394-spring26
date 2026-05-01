#!/bin/bash -l
# Project 02: cuBLAS FP32 + tensor-core GEMM throughput on Polaris (A100)
#
# polaris-services orchestrator: write PBS script, qsub, wait, plot.
# Keep the orchestrator absolutely minimal — no conda activate.

WORK_DIR="/eagle/datascience/hzheng/jobs/polaris/csci394_gpu_p2"
cd "$WORK_DIR" || exit 1

echo "=== orchestrator host: $(hostname)  date: $(date) ==="

PBS_SCRIPT="$WORK_DIR/pbs_run.sh"
cat > "$PBS_SCRIPT" << 'EOF'
#!/bin/bash -l
#PBS -A AmSC_Demos
#PBS -q debug
#PBS -l select=1:system=polaris
#PBS -l walltime=00:30:00
#PBS -l filesystems=eagle:home
#PBS -N csci394_p2
#PBS -j oe
#PBS -o /eagle/datascience/hzheng/jobs/polaris/csci394_gpu_p2/pbs.out

set -e
WORK=/eagle/datascience/hzheng/jobs/polaris/csci394_gpu_p2
cd "$WORK"

source /etc/profile 2>/dev/null || true
echo "host: $(hostname)  date: $(date)"

module use /soft/modulefiles
module load cudatoolkit-standalone
which nvcc && nvcc --version | grep release
nvidia-smi -L | head -2

make NVCC=nvcc CUDA_ARCH=sm_80 all

CSV="$WORK/results.csv"
LOG="$WORK/raw.log"
echo "mode,n,iters,warmup,gemm_s,gflops,max_abs_err" > "$CSV"
: > "$LOG"

ITERS=10
WARMUP=2
SIZES=(256 512 1024 2048 4096 8192)

for n in "${SIZES[@]}"; do
    for mode in fp32 tensor_core; do
        echo "--- $mode n=$n ---" | tee -a "$LOG"
        set +e
        out=$(./app_cublas $mode $n $ITERS $WARMUP 2>&1)
        rc=$?
        set -e
        echo "$out" | tee -a "$LOG"
        if [ $rc -ne 0 ]; then echo "FAILED rc=$rc" | tee -a "$LOG"; continue; fi
        echo "$out" | grep -E "^RESULT," | sed 's/^RESULT,//' >> "$CSV"
    done
done

echo "=== CSV ==="
cat "$CSV"

# Plot using miniconda Python directly (skip conda activate)
PY=/soft/applications/conda/2024-04-29/mconda3/bin/python
WORK=$WORK $PY - << 'PYEOF' || echo "(plot skipped)"
import csv, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
work = os.environ["WORK"]
csv_path = os.path.join(work, "results.csv")
xs = defaultdict(list); ys = defaultdict(list)
with open(csv_path) as f:
    for row in csv.DictReader(f):
        xs[row["mode"]].append(int(row["n"]))
        ys[row["mode"]].append(float(row["gflops"]))
plt.figure(figsize=(7, 4.5))
for m in sorted(xs):
    pairs = sorted(zip(xs[m], ys[m]))
    plt.plot([p[0] for p in pairs], [p[1] for p in pairs], "-o", label=m)
plt.xlabel("matrix size n"); plt.ylabel("GFLOP/s")
plt.title("Polaris A100 cuBLAS GEMM throughput")
plt.xscale("log"); plt.yscale("log"); plt.grid(True, which="both", alpha=0.3); plt.legend()
plt.tight_layout()
out = os.path.join(work, "gflops_vs_size.png")
plt.savefig(out, dpi=130); print("plot:", out)
PYEOF
echo "=== compute-node DONE ==="
EOF
chmod +x "$PBS_SCRIPT"

# qsub from /opt/pbs/bin (always in PATH on login)
job_id=$(/opt/pbs/bin/qsub "$PBS_SCRIPT" 2>&1)
echo "PBS Job: $job_id"

# Strip newlines / spaces
JID=$(echo "$job_id" | tr -d '[:space:]')

while true; do
    sleep 30
    state=$(/opt/pbs/bin/qstat -f "$JID" 2>/dev/null | grep "job_state" | awk '{print $3}')
    if [ -z "$state" ] || [ "$state" = "F" ]; then
        echo "Job done"
        break
    fi
    echo "PBS state: $state"
done

echo "=== orchestrator DONE ==="
ls -la "$WORK_DIR"
