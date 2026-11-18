#!/bin/bash -l
#SBATCH --job-name=saion_preflight
# Fallback only: scripts/saion/submit_saion.sh overrides this with whichever of
# gpu-p100/gpu-v100 starts first (a comma list is rejected for this account).
#SBATCH --partition=gpu-p100
#SBATCH --gres=gpu:2
# A quarter-node share, matching the production job (9 cores + 1 GPU), so the
# capacity probe measures the shape that will actually run. The 36-worker
# sweep is already done -- preflight 4724497 on a P100: 36 workers, 0 errors,
# 15,135 MiB of 16,384 (92%), safe cap ~31, and total throughput flat at ~58
# CCM sweeps/s for 1/9/18/36 alike. See docs/no-kEDM-gpu-optimization-progress.md.
#SBATCH --cpus-per-task=9
#SBATCH --mem=120G
#SBATCH --time=0-02:00:00
# Absolute path under /home/j/j-tominaga (the guide's recommendation); the directory must
# already exist -- Slurm opens it before this script runs. See the runbook.
#SBATCH --output=/home/j/j-tominaga/saion_slurm_logs/%x-%j.out
#SBATCH --mail-user=j-tominaga@oist.jp
#SBATCH --mail-type=END,FAIL

# =============================================================================
# Saion preflight for the parallel-MDE GPU pipeline
# =============================================================================
# Submit from the Saion login node, from /home/j/j-tominaga:
#     cd /home/j/j-tominaga && /home/j/j-tominaga/pao_rotation/scripts/saion/submit_saion.sh saion-preflight.sbatch
#
# Checks, on a real restricted-partition GPU node, everything
# saion-no-kedm-optimization-parallelized-mde--tsls-mde-gmn.sbatch depends on,
# before a multi-day allocation is spent finding out:
#   1. node / GPU / CUDA: nvidia-smi, torch sees the device, a CUDA op works
#   2. venv: every pipeline import + sitecustomize.py (pyEDM Pool fix)
#   3. storage rules: /work/FukaiU/j-tominaga writable,
#      /bucket/FukaiU/tominaga/pao_rotation readable (and read-only here)
#   4. stage-out: passwordless ssh to the login node, a real round-trip
#      write into /bucket/FukaiU/tominaga/pao_rotation through it, then
#      removal of the test file
#   5. tests: the two GPU-vs-CPU equivalence suites must PASS, not SKIP --
#      they have never run on real GPU hardware -- plus fork safety and the
#      parallel-MDE network-equivalence test
#   6. GPU capacity: do 1/9/18/36 concurrent spawn workers fit on ONE GPU
#      (scripts/saion/gpu_capacity_probe.py) -- the measurement the
#      multi-GPU decision is waiting on
#
# Only the report is kept (next to this job's .out in /home/j/j-tominaga/saion_slurm_logs);
# its scratch dir inside /work/FukaiU/j-tominaga is always removed. Exit status is non-zero if any
# REQUIRED check failed; the capacity verdict is reported but not required
# (it decides job sizing, it doesn't block).
# =============================================================================

set -uo pipefail

# Defines the scoped roots PROJ_DIR, SLURM_LOG_DIR, WORK_BASE, BUCKET_PROJECT.
# shellcheck source=../saion/saion_env.sh
source "/home/j/j-tominaga/pao_rotation-exp-neuro/scripts/saion/saion_env.sh"
VENV_PY="$PROJ_DIR/.venv-gpu/bin/python3"
REPORT_DIR="$SLURM_LOG_DIR/preflight_${SLURM_JOB_ID}"
mkdir -p "$REPORT_DIR"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLBACKEND=Agg

declare -a RESULTS=()
FAILED=0
check() {   # check REQUIRED|INFO "name" command...
    local level="$1" name="$2"; shift 2
    local log="$REPORT_DIR/$(echo "$name" | tr -c 'A-Za-z0-9_-' '_').log"
    echo "---- [$level] $name"
    if "$@" >"$log" 2>&1; then
        RESULTS+=("PASS  [$level] $name")
    else
        RESULTS+=("FAIL  [$level] $name  (see $log)")
        [ "$level" = REQUIRED ] && FAILED=1
    fi
    tail -n 25 "$log"
}

mkdir -p "$WORK_BASE" 2>/dev/null
scratch="$(mktemp -d "$WORK_BASE/preflight.${SLURM_JOB_ID}.XXXXXX" 2>/dev/null || true)"
trap '[ -n "${scratch:-}" ] && safe_rm_work "$scratch"' EXIT
if [ -n "$scratch" ]; then
    # Temp files stay inside this job's scratch dir, not the node-wide /tmp.
    mkdir -p "$scratch/tmp" && export TMPDIR="$scratch/tmp" MPLCONFIGDIR="$scratch/tmp/matplotlib"
fi

echo "host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} gpus=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-?}} cpus=${SLURM_CPUS_PER_TASK:-?}"
echo "commit=$(git -C "$PROJ_DIR" rev-parse --short HEAD 2>/dev/null || echo '?') branch=$(git -C "$PROJ_DIR" branch --show-current 2>/dev/null || echo '?')"

# ---- 1. node / GPU / CUDA ----------------------------------------------------
check REQUIRED "nvidia-smi" nvidia-smi
check REQUIRED "torch CUDA" "$VENV_PY" -c "
import torch
assert torch.cuda.is_available(), 'torch.cuda.is_available() is False'
print('torch', torch.__version__, '| device', torch.cuda.get_device_name(0), '| capability', torch.cuda.get_device_capability(0))
x = torch.randn(2000, 2000, device='cuda'); print('matmul', float((x @ x).sum()))
assert torch.__version__.endswith('+cu126'), 'expected a cu126 torch build'
"

# ---- 2. venv -------------------------------------------------------------------
check REQUIRED "pipeline imports + sitecustomize" "$VENV_PY" -c "
import sys; sys.path.insert(0, '$PROJ_DIR/scripts/src')
import sitecustomize; print('sitecustomize:', sitecustomize.__file__)
import gmn, pyEDM, networkx, sklearn, matplotlib, pydiffmap, pandas, numpy
from apps.CreateNetwork import CreateNetwork
import cleaned_helper_functions, gpu_edm, gpu_budget
# The pipeline itself: importing pipeline pulls in every stage module, and
# importing network_parallel covers the builder this job actually uses. This is
# the check that catches a package the sbatch forgot to stage, or a module that
# no longer imports, before a 7-day job dies at startup.
import mde_gmn.pipeline, mde_gmn.network_parallel
assert gpu_edm.is_available(), 'gpu_edm.is_available() is False'
print('all imports OK; gpu_edm.is_available() = True')
# gpu_budget sizes a per-device admission budget from what the card
# reports. Printing it here is the only place the numbers a real run
# will queue against are visible BEFORE the run commits days to them.
print(gpu_budget.env_summary())
import multiprocessing as _mp
_b = gpu_budget.create_budgets(_mp.get_context('spawn'),
                               workers=int('${SLURM_CPUS_PER_TASK:-1}'))
assert _b, 'gpu_budget.create_budgets() produced no budget despite a visible GPU'
"

# ---- 3. storage rules ------------------------------------------------------------
check REQUIRED "work dir writable ($WORK_BASE)" bash -c "[ -n '$scratch' ] && [ -d '$scratch' ] && echo ok > '$scratch/probe' && cat '$scratch/probe' && df -h '$WORK_BASE'"
check REQUIRED "bucket project readable ($BUCKET_PROJECT)" ls -la "$BUCKET_PROJECT"
check INFO "bucket project is read-only on this node (write must fail)" bash -c "! touch '$BUCKET_PROJECT/.preflight_write_test_${SLURM_JOB_ID}' 2>/dev/null"
check REQUIRED "input data readable on Bucket" ls -la "$BUCKET_DATA_DIR/J1620210605_epoch_8_freq_40_Hz.csv"

# ---- 4. stage-out through the login node --------------------------------------------
check REQUIRED "ssh to $STAGE_HOST (BatchMode, no password)" remote_run "hostname; command -v rsync || echo 'no rsync on login node (tar fallback will be used)'"
stage_test() {
    local src="$scratch/stage_test" dest="$BUCKET_PROJECT/.preflight_stage_test_${SLURM_JOB_ID}"
    mkdir -p "$src/sub" && echo "preflight $SLURM_JOB_ID $(date -Iseconds)" > "$src/sub/hello.txt"
    stage_dir "$src" "$dest" || return 1
    remote_run "cat '$dest/sub/hello.txt'" || return 1
    path_under "$BUCKET_PROJECT" "$dest" || return 1
    remote_run "rm -rf '$dest'"
}
check REQUIRED "stage-out round trip into $BUCKET_PROJECT" stage_test

# ---- 5. tests -------------------------------------------------------------------------
run_test() {   # run_test file require_no_cuda_skip
    local out rc
    [ -n "$scratch" ] || { echo "no scratch dir in $WORK_BASE"; return 1; }
    out="$(cd "$scratch" && "$VENV_PY" "$PROJ_DIR/tests/$1" 2>&1)"; rc=$?
    echo "$out"
    [ $rc -eq 0 ] || return 1
    echo "$out" | grep -q 'ALL PASS' || return 1
    if [ "$2" = yes ] && echo "$out" | grep -q 'SKIP (no CUDA device visible)'; then
        echo '>>> equivalence was SKIPPED although this is a GPU node -- counts as FAIL'
        return 1
    fi
}
check REQUIRED "test_gpu_ccm_equivalence (must not skip)" run_test test_gpu_ccm_equivalence.py yes
check REQUIRED "test_gpu_multi_ccm_equivalence (must not skip)" run_test test_gpu_multi_ccm_equivalence.py yes
check REQUIRED "test_gpu_fork_safety" run_test test_gpu_fork_safety.py no
check REQUIRED "test_parallel_mde_equivalence" run_test test_parallel_mde_equivalence.py no

# ---- 6. GPU capacity (decides job sizing, not required) --------------------------------------
check INFO "GPU capacity: ${SLURM_CPUS_PER_TASK} spawn workers on one GPU" \
    bash -c "cd '$scratch' && '$VENV_PY' '$PROJ_DIR/scripts/saion/gpu_capacity_probe.py' --json '$REPORT_DIR/gpu_capacity.json'"

# ---- summary ----------------------------------------------------------------------------
{
    echo "=============================================================="
    echo "SAION PREFLIGHT job ${SLURM_JOB_ID} on $(hostname) (${SLURM_JOB_PARTITION:-?})"
    echo "=============================================================="
    printf '%s\n' "${RESULTS[@]}"
    echo
    echo "GPU capacity summary:"
    sed -n '/=== SUMMARY ===/,$p' "$REPORT_DIR"/GPU_capacity*.log 2>/dev/null
    echo
    if [ "$FAILED" -eq 0 ]; then
        echo "OVERALL: PASS -- required checks OK. Size the pipeline job from the GPU VERDICT above."
    else
        echo "OVERALL: FAIL -- fix the FAIL [REQUIRED] items before submitting the pipeline job."
    fi
} | tee "$REPORT_DIR/SUMMARY.txt"

exit "$FAILED"
