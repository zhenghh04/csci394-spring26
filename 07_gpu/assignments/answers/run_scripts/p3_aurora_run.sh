#!/bin/bash -l
# Project 03: Intel XPU SYCL GEMM on Aurora (FP32 + XMX joint_matrix BF16->FP32)
# Run via aurora-services (login node) -> qsub child PBS job -> wait -> plot.

WORK_DIR=/lus/flare/projects/Aurora_deployment/hzheng/jobs/aurora/csci394_gpu_p3
cd "$WORK_DIR" || exit 1

echo "=== orchestrator host: $(hostname)  date: $(date) ==="

PBS_SCRIPT="$WORK_DIR/pbs_run.sh"
cat > "$PBS_SCRIPT" << 'EOF'
#!/bin/bash -l
#PBS -A CSCI394-HPC
#PBS -q debug
#PBS -l select=1
#PBS -l walltime=00:30:00
#PBS -l filesystems=flare
#PBS -N csci394_p3
#PBS -j oe
#PBS -o /lus/flare/projects/Aurora_deployment/hzheng/jobs/aurora/csci394_gpu_p3/pbs.out

set -e
WORK=/lus/flare/projects/Aurora_deployment/hzheng/jobs/aurora/csci394_gpu_p3
cd "$WORK"

source /etc/profile 2>/dev/null || true
echo "host: $(hostname)  date: $(date)"

module load oneapi
which icpx && icpx --version | head -1

export ONEAPI_DEVICE_SELECTOR="level_zero:gpu"

make CXX=icpx all

CSV="$WORK/results.csv"
LOG="$WORK/raw.log"
echo "mode,n,iters,warmup,time_s,gflops,max_abs_err" > "$CSV"
: > "$LOG"

ITERS=5
WARMUP=2
SIZES=(256 512 1024 2048 4096)

for n in "${SIZES[@]}"; do
    for mode in fp32 xmx_bf16_fp32; do
        echo "--- $mode n=$n ---" | tee -a "$LOG"
        set +e
        out=$(./app_xpu $mode $n $ITERS $WARMUP 2>&1)
        rc=$?
        set -e
        echo "$out" | tee -a "$LOG"
        if [ $rc -ne 0 ]; then echo "FAILED rc=$rc" | tee -a "$LOG"; continue; fi
        echo "$out" | grep -E "^RESULT," | sed 's/^RESULT,//' >> "$CSV"
    done
done

echo "=== CSV ==="
cat "$CSV"
echo "=== compute-node DONE ==="
EOF
chmod +x "$PBS_SCRIPT"

job_id=$(/opt/pbs/bin/qsub "$PBS_SCRIPT" 2>&1)
echo "PBS Job: $job_id"
JID=$(echo "$job_id" | tr -d '[:space:]')

while true; do
    sleep 30
    state=$(/opt/pbs/bin/qstat -f "$JID" 2>/dev/null | grep "job_state" | awk '{print $3}')
    if [ -z "$state" ] || [ "$state" = "F" ]; then echo "Job done"; break; fi
    echo "PBS state: $state"
done

echo "=== orchestrator DONE ==="
ls -la "$WORK_DIR"
