#!/bin/bash -l
# Project 01: GPU offload comparison on Polaris (A100)
# polaris-services orchestrator -> qsub -> wait -> plot.

WORK_DIR="/eagle/datascience/hzheng/jobs/polaris/csci394_gpu_p1"
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
#PBS -N csci394_p1
#PBS -j oe
#PBS -o /eagle/datascience/hzheng/jobs/polaris/csci394_gpu_p1/pbs.out

set -e
WORK=/eagle/datascience/hzheng/jobs/polaris/csci394_gpu_p1
cd "$WORK"

source /etc/profile 2>/dev/null || true
echo "host: $(hostname)  date: $(date)"

module use /soft/modulefiles
# Polaris default PrgEnv-nvidia provides nvc/nvcc through the cc/CC wrapper.
# Use the cc compiler wrapper as $CC to compile NVHPC/OpenACC/OpenMP-target.
module load cudatoolkit-standalone
which cc nvcc
cc --version | head -1
nvcc --version | grep release

# Use ALCF conda Python directly for the PyTorch run (no conda activate)
PYBIN=/soft/applications/conda/2024-04-29/mconda3/bin/python
$PYBIN -c "import torch; print('torch', torch.__version__, 'cuda?', torch.cuda.is_available())"

echo "=== build ==="
# `cc` wrapper on Polaris picks up nvc; works for our -mp / -acc flags.
make CC=cc NVCC=nvcc all

CSV="$WORK/results.csv"
LOG="$WORK/raw.log"
echo "version,n,iters,warmup,end_to_end_s,compute_s,max_abs_err" > "$CSV"
: > "$LOG"

ITERS=5
WARMUP=1
SIZES=(256 512 1024 2048 4096)

run_one() {
    local label=$1; local cmd=$2; local n=$3
    echo "--- $label n=$n ---" | tee -a "$LOG"
    set +e
    out=$(eval "$cmd" 2>&1)
    rc=$?
    set -e
    echo "$out" | tee -a "$LOG"
    if [ $rc -ne 0 ]; then echo "FAILED $label n=$n rc=$rc" | tee -a "$LOG"; return; fi
    line=$(echo "$out" | grep -E "^RESULT," | head -1)
    [ -n "$line" ] && echo "$line" | sed 's/^RESULT,//' >> "$CSV"
}

export OMP_NUM_THREADS=32
export OMP_PLACES=cores
export OMP_PROC_BIND=close

for n in "${SIZES[@]}"; do
    if [ "$n" -le 2048 ]; then
        run_one cpu        "./app_cpu $n $ITERS $WARMUP"        $n
    else
        run_one cpu        "./app_cpu $n 2 1"                   $n
    fi
    run_one openacc    "./app_openacc $n $ITERS $WARMUP"    $n
    run_one omp_target "./app_omp_target $n $ITERS $WARMUP" $n
    run_one cuda       "./app_cuda $n $ITERS $WARMUP"       $n
    run_one pytorch    "$PYBIN src/app_pytorch.py $n $ITERS $WARMUP" $n
done

echo "=== CSV ==="
cat "$CSV"

WORK=$WORK $PYBIN - << 'PYEOF' || echo "(plot skipped)"
import csv, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

work = os.environ["WORK"]
csv_path = os.path.join(work, "results.csv")
data_e2e = defaultdict(list); data_comp = defaultdict(list); sizes = defaultdict(list)
with open(csv_path) as f:
    for row in csv.DictReader(f):
        v = row["version"]; n = int(row["n"])
        sizes[v].append(n); data_e2e[v].append(float(row["end_to_end_s"])); data_comp[v].append(float(row["compute_s"]))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for v in sorted(data_e2e):
    pairs = sorted(zip(sizes[v], data_e2e[v]))
    axes[0].loglog([p[0] for p in pairs], [p[1] for p in pairs], "-o", label=v)
for v in sorted(data_comp):
    pairs = sorted(zip(sizes[v], data_comp[v]))
    axes[1].loglog([p[0] for p in pairs], [p[1] for p in pairs], "-o", label=v)
for ax, title in zip(axes, ["End-to-end time (s)", "Compute-only time (s)"]):
    ax.set_xlabel("matrix size n"); ax.set_ylabel("time (s)")
    ax.set_title(title); ax.grid(True, which="both", alpha=0.3); ax.legend()
plt.tight_layout()
out = os.path.join(work, "runtime_vs_size.png")
plt.savefig(out, dpi=130); print("plot:", out)
PYEOF

echo "=== compute-node DONE ==="
EOF
chmod +x "$PBS_SCRIPT"

job_id=$(/opt/pbs/bin/qsub "$PBS_SCRIPT" 2>&1)
echo "PBS Job: $job_id"
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
