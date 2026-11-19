#!/bin/bash -l
#SBATCH --job-name=spk_gen
# exp/spike-conversion-general ablation: ONE event->continuous method per job,
# on a quarter of a Saion GPU node (1 GPU for the CCM gate, 9 cores).
# Saion's CPU-only `intel` partition was the first choice, but on 2026-11-19 it
# was saturated by other users and refused any request above ~3 cores.
#
# Submit (Saion login node, from /home/j/j-tominaga):
#   ABLATION_LABEL=... SPIKE_TRANSFORM=... \
#     /home/j/j-tominaga/pao_rotation-exp-general/scripts/saion/submit_saion.sh \
#     saion-spike-general.sbatch no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py
# or scripts/experiments/spike_general/submit_grid.sh, which sets the variables
# and records label / job / commit in docs/experiments/spike-conversion-general/jobs.tsv.
#
# Data path is the same as the parent saion sbatch: data read directly from
# /bucket, code staged to /work, results and cache staged back to /bucket
# through the login node at exit (and the cache hourly).
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=9
#SBATCH --mem=110G
#SBATCH --time=3-00:00:00
#SBATCH --signal=B:USR1@1800
#SBATCH --output=/home/j/j-tominaga/saion_slurm_logs/%x-%j.out
#SBATCH --error=/home/j/j-tominaga/saion_slurm_logs/%x-%j.err
#SBATCH --mail-user=j-tominaga@oist.jp
#SBATCH --mail-type=END,FAIL

# ==============================================================================
# MODIFIABLE PARAMETERS
# ==============================================================================
# Every experiment value can be overridden from the submitting environment.
ABLATION_LABEL="${ABLATION_LABEL:-unlabelled}"

# ---- the representation under test (method selector) ----------------------
SPIKE_TRANSFORM="${SPIKE_TRANSFORM:-sawtooth}"     # sawtooth | gamma | lmu | logsig
GAMMA_ORDER="${GAMMA_ORDER:-3}"                    # m; the response is C^(m-1)
GAMMA_TAUS_MS="${GAMMA_TAUS_MS:-250}"              # comma list = multi-scale sum; or auto:<k>
GAMMA_LOG_KAPPA="${GAMMA_LOG_KAPPA:-0}"            # 0 = off; else log(1 + kappa x / mean rate)
LMU_THETA_MS="${LMU_THETA_MS:-1000}"
LMU_D="${LMU_D:-3}"
LMU_DISCRETIZATION="${LMU_DISCRETIZATION:-impulse}"  # impulse (exact for events) | zoh
LMU_PRESMOOTH_ORDER="${LMU_PRESMOOTH_ORDER:-0}"
LMU_REDUCTION="${LMU_REDUCTION:-lmu_norm}"
LOGSIG_W_MS="${LOGSIG_W_MS:-1000}"
LOGSIG_DEPTH="${LOGSIG_DEPTH:-2}"
LOGSIG_PRESMOOTH_ORDER="${LOGSIG_PRESMOOTH_ORDER:-0}"
LOGSIG_REDUCTION="${LOGSIG_REDUCTION:-rms_norm}"

# ---- the updated evaluation (same for every method) ------------------------
N_MIN_EVENTS="${N_MIN_EVENTS:-100}"       # observability filter, training-split event bins
WARMUP_ROWS=400                           # common leading trim (10 s), every method
EMBED_TAU="${EMBED_TAU:-auto}"            # Takens lag from AMI of the transformed signals
EMBED_TAU_METHOD="ami"
EMBED_TAU_MAX=40
METRIC="correlation"                      # neurons scored on the whole trajectory, not reset events
TARGET_METRIC="correlation"
GAMMA_DELTA=2
TARGET_RANKING_WINDOW_ROWS=4000           # one contiguous 100-s window for EVERY node
MIN_PRED_EVENTS=20                        # unused under METRIC=correlation
MAX_WINDOW_ROWS=30000                     # unused under METRIC=correlation
DELAY_SOURCE="tau_count"                  # Stage 2 lags = -tau, -2tau (tau = resolved Takens lag)
LAG_STEP=1                                # replaced by the resolved Takens lag at run time
LAG_COUNT="${LAG_COUNT:-2}"
LAG_MODE="generation_length_adjusted"
LAG_DENSITY="strided"
DELAY_SET="0"

# ---- the free run ----------------------------------------------------------
EXTERNAL_INPUT_MODE="self_simulate"       # input-free: zero-driver nodes simulate themselves
TP=1
GMN_TP="match"
GENERATE_LENGTH=400                       # 10 s
GENERATE_NOISE="none"
GENERATE_NOISE_SEED=0
GENERATE_START_MIN_OFFSET=0               # one start for the parent's own sweep; the
GENERATE_START_MAX_OFFSET=1               # rollout ensemble below reports every start
GENERATE_START_STEP=1
ROLLOUT_N_STARTS="${ROLLOUT_N_STARTS:-24}"
ROLLOUT_LENGTH=400
ROLLOUT_HORIZONS_S="2,5,10"
ROLLOUT_WORKERS="${ROLLOUT_WORKERS:-8}"
PH_LANDMARKS=80
GMN_EXCLUSION_RADIUS=0
FORECAST_PLOT_RANGE="windowed"

# ---- scale -----------------------------------------------------------------
REGION_CHANNEL_CAP="${REGION_CHANNEL_CAP:-30}"   # first 30 per region = 90 candidates before filtering
MDE_D="${MDE_D:-6}"
NESTED_M="${NESTED_M:-2}"
THETA_RANGE="0.1,0.5,1,2,4,8"

# ---- unchanged from the parent saion sbatch --------------------------------
DATA_FILE="J1620210605_epoch_8_freq_40_Hz.csv"
MODE="MDE-GMN"
GMN_MODE="combined"
TARGETS="head_position_x,head_position_y"
MDE_MIN_GAIN="${MDE_MIN_GAIN:-0}"
EMBED_DIM_RHO_MIN=0.2
CROSS_MAP_RHO_MIN=0.2
EMBED_DIM_RHO_MIN_F1=0.15
CROSS_MAP_RHO_MIN_F1=0.15
CCM_PREFILTER_RHO_MIN=0.2
CCM_PREFILTER_E=7
LIBSIZE_RATIOS="0.05,0.10,0.20,0.40,0.6,0.8,0.9"
MULTI_MDE_E=8
MULTI_MDE_TARGET_AGGREGATE="mean"
EXCLUSION_RADIUS=0
PARALLEL_MDE_SCOPE="staged"
LIB_RATIO=0.7
MDE_VALIDATION_RATIO=0.5
E_MIN=3; E_MAX=13; E_STEP=4
TAU_MIN=-9; TAU_MAX=0; TAU_STEP=4
MAX_TP=20
GRID_RHO_THRESHOLD=0.9
RUN_MDE_DIAGNOSTICS=false
RUN_MDE_EVALUATE=false
# The parent sized its CCM windows for a Takens lag of 1 row. Here the lag is
# resolved from the data and is ~40 rows (1 s) for every method, so an 800-row
# window would leave ~500 embedded rows; both windows follow the 4000-row
# ranking window instead (same for every method).
CCM_WINDOW_ROWS=4000
CCM_LIB_SIZE_PCTS="2,3,5,8,12,20,30,50"
CCM_SAMPLE=10
CCM_SLOPE=0.01
CCM_LIB_SIZES_ROWS="40,160,400,800,1600,3200"
REUSE_BEST_PARAMS=true                    # keys include the preprocessing digest: resume only
# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================
# Deliberately no `set -e`: every failure path below must still reach the
# EXIT trap's stage-out, and the steps that must not fail check explicitly.
set -uo pipefail

# Defines the scoped roots PROJ_DIR, SLURM_LOG_DIR, WORK_BASE, BUCKET_PROJECT.
# shellcheck source=../saion/saion_env.sh
source "/home/j/j-tominaga/pao_rotation-exp-general/scripts/saion/saion_env.sh"
PY_NAME="no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py"
VENV_DIR="$PROJ_DIR/.venv-gpu"

die() { echo "FATAL: $*" >&2; exit 2; }
[ -n "${SLURM_JOB_ID:-}" ] || die "submit with sbatch, not bash"
[ -d "$(dirname "$WORK_BASE")" ] || die "$(dirname "$WORK_BASE") not found -- this script is for Saion compute nodes only"
[ -x "$VENV_DIR/bin/python3" ] || die "$VENV_DIR missing -- run scripts/saion/setup_gpu_venv.sh on the login node first"

# No `module load python` -- the venv's interpreter is uv-managed under /home/j/j-tominaga.
# No `module load cuda` -- the cu126 torch wheel bundles its own CUDA runtime;
# only the node's driver is used.
source "${VENV_DIR}/bin/activate"
export MPLBACKEND=Agg
# Workers are processes (TSLS_MDE_GMN_CORES); keep BLAS/OpenMP single-threaded
# inside each, so N workers don't each spin up N threads and oversubscribe the
# cores this job was actually given.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
# Let the CUDA caching allocator grow one segment instead of many fixed
# blocks. torch's own OOM message in job 4741044 recommended exactly this;
# it does not create capacity, it stops capacity being lost to
# fragmentation between the several workers sharing each card.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Bucket targets, all inside $BUCKET_PROJECT (written only through the login
# node; stage_dir refuses anything outside it).
CACHE_ROOT="$BUCKET_PROJECT/tsls_mde_gmn_cache"
OUTPUT_ARCHIVE_ROOT="$BUCKET_PROJECT/tsls_mde_gmn_outputs"
CACHE_DEST="$CACHE_ROOT/job_${SLURM_JOB_ID}"

# Per-job scratch inside $WORK_BASE (guide: unique mktemp dir, deleted at the end).
mkdir -p "$WORK_BASE" || die "cannot create $WORK_BASE"
jobdir="$(mktemp -d "$WORK_BASE/pmde.${SLURM_JOB_ID}.XXXXXX")" || die "mktemp in $WORK_BASE failed"
OUT_DIR="$jobdir/results"
RUN_CACHE_DIR="$jobdir/cache"
STAGE_SRC="$jobdir/src"
mkdir -p "$OUT_DIR" "$RUN_CACHE_DIR" "$STAGE_SRC" "$jobdir/tmp" || die "cannot populate $jobdir"
# Temp files (Python tempfile, mktemp, matplotlib/torch scratch) stay inside
# this job's dir instead of the node-wide /tmp, and are removed with it.
export TMPDIR="$jobdir/tmp"
export MPLCONFIGDIR="$jobdir/tmp/matplotlib"

SLURM_OUT="$SLURM_LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
SLURM_ERR="$SLURM_LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"

PY_PID=""
PY_STATUS=""
STOP_REASON=""
BG_PIDS=()

# ==============================================================================
# STOP / OUTCOME / STAGE-OUT
# ==============================================================================
stop_python() {
    [ -n "$PY_PID" ] && kill -0 "$PY_PID" 2>/dev/null || return 0
    # Spawn workers are children of the Python process -- stop them too.
    pkill -TERM -P "$PY_PID" 2>/dev/null
    kill -TERM "$PY_PID" 2>/dev/null
    for _ in $(seq 60); do
        kill -0 "$PY_PID" 2>/dev/null || return 0
        sleep 2
    done
    pkill -KILL -P "$PY_PID" 2>/dev/null
    kill -KILL "$PY_PID" 2>/dev/null
}

on_usr1() {
    STOP_REASON="time limit approaching (or scancel --signal=USR1): stopped cleanly to stage out"
    echo "$(date -Iseconds) USR1 received -- ${STOP_REASON}" >&2
    stop_python
}
trap on_usr1 USR1
trap 'STOP_REASON="${STOP_REASON:-SIGTERM received (scancel or node shutdown)}"; stop_python; exit 143' TERM

cleanup() {
    trap - EXIT USR1 TERM
    stop_python
    for pid in "${BG_PIDS[@]}"; do
        pkill -P "$pid" 2>/dev/null
        kill "$pid" 2>/dev/null
    done

    local slurm_state="" gpu_errors budget_advisories networks_skipped mde_failures ccm_dead_ends status reason dest stage_ok=1
    for _ in 1 2 3; do
        slurm_state="$(sacct -j "$SLURM_JOB_ID" --format=State -P -n 2>/dev/null | head -1 | tr -d ' ')"
        [ -n "$slurm_state" ] && break
        sleep 2
    done
    # 'memory allocation failed with OOM' is the CUDACachingAllocator's own
    # spelling and matched NONE of the other patterns -- job 4741044 logged
    # 47 of those lines and this check saw zero of them, so the early
    # warning that the card was saturating never reached STATUS.txt.
    #
    # 'UNDER-ESTIMATED' is deliberately NOT in this list (it was, 2026-10-05
    # to 2026-10-22). gpu_budget.reserve() prints it AFTER a reservation has
    # been granted and the call has run, to say the shape formula predicted
    # less than the allocator went on to hold. Nothing failed. Counting it
    # here stamped job 4744294 FAIL on 93 of those advisories with zero real
    # CUDA errors in the whole 19h run, which buried the one thing that did
    # go wrong. It gets its own line below instead.
    gpu_errors="$(cat "$SLURM_OUT" "$SLURM_ERR" 2>/dev/null \
        | grep -c -E 'CUDA out of memory|OutOfMemoryError|CUDA error|CUBLAS_STATUS|cuDNN error|memory allocation failed with OOM')"
    budget_advisories="$(cat "$SLURM_OUT" "$SLURM_ERR" 2>/dev/null | grep -c 'UNDER-ESTIMATED')"
    # A network whose Generate() never rebuilt produces NO plots, fidelity or
    # output, and the pipeline says so and carries on to exit 0. Job 4744294
    # lost head_position_x_head_position_y exactly that way and reported
    # success. Counted, and FAILed on below, so it cannot pass unnoticed again.
    networks_skipped="$(cat "$SLURM_OUT" 2>/dev/null | grep -c 'no successful Generate() rebuild')"
    mde_failures="$(cat "$SLURM_OUT" 2>/dev/null | grep -c 'MDE call FAILED')"
    # Candidates that were genuinely evaluated and found wanting (short
    # data, no convergence). These are FINDINGS, not failures -- counted
    # so they stop being indistinguishable from the GPU-memory drops that
    # used to print through the same clause.
    ccm_dead_ends="$(cat "$SLURM_OUT" 2>/dev/null | grep -c 'CCM dead end:')"

    if [ "${PY_STATUS:-unset}" = "0" ] && [ -z "$STOP_REASON" ] && [ "$gpu_errors" -eq 0 ] \
            && [ "$networks_skipped" -eq 0 ]; then
        status=COMPLETED; reason="pipeline exited 0"
        dest="$OUTPUT_ARCHIVE_ROOT/job_${SLURM_JOB_ID}"
    else
        status=FAIL
        if [ -n "$STOP_REASON" ]; then reason="$STOP_REASON"
        elif [ "${PY_STATUS:-unset}" = "0" ] && [ "$gpu_errors" -gt 0 ]; then reason="pipeline exited 0 but the log has ${gpu_errors} CUDA error line(s) -- MDE nodes that hit them were silently recorded as dead ends"
        elif [ "${PY_STATUS:-unset}" = "0" ]; then reason="pipeline exited 0 but ${networks_skipped} network(s) produced no output (Generate() never rebuilt) -- see the '*: FINAL Generate() rebuild FAILED' line(s) in the log"
        elif [ -z "${PY_STATUS:-}" ]; then reason="job ended before the pipeline started (setup failure)"
        else reason="pipeline exited with status ${PY_STATUS}"
        fi
        dest="$OUTPUT_ARCHIVE_ROOT/job_${SLURM_JOB_ID}_FAIL"
    fi

    {
        echo "STATUS: $status"
        echo "REASON: $reason"
        echo "PY_STATUS: ${PY_STATUS:-unset}"
        echo "SLURM_STATE_AT_EXIT: ${slurm_state:-unknown (check: sacct -j $SLURM_JOB_ID)}"
        echo "HOST: $(hostname)  PARTITION: ${SLURM_JOB_PARTITION:-?}  GPUS: ${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-?}}"
        echo "CUDA_ERROR_LINES: $gpu_errors"
        echo "GPU_BUDGET_ADVISORIES: $budget_advisories  (UNDER-ESTIMATED; informational, not failures)"
        echo "NETWORKS_SKIPPED: $networks_skipped"
        echo "CCM_DEAD_ENDS: $ccm_dead_ends"
        echo "MDE_CALL_FAILED_LINES: $mde_failures"
        echo "ENDED: $(date -Iseconds)"
        echo "CACHE_ARCHIVE: $CACHE_DEST"
    } > "$OUT_DIR/STATUS.txt"
    [ -f "$OUT_DIR/progress.log" ] && \
        echo "$(date -Iseconds) JOB ENDED: $status -- $reason (PY_STATUS=${PY_STATUS:-unset} SLURM_STATE=${slurm_state:-unknown})" \
        >> "$OUT_DIR/progress.log"
    # cp, not mv: Slurm is still appending to these until the job exits.
    [ -f "$SLURM_OUT" ] && cp -f "$SLURM_OUT" "$OUT_DIR"/
    [ -f "$SLURM_ERR" ] && cp -f "$SLURM_ERR" "$OUT_DIR"/
    [ -f "$0" ] && cp -f "$0" "$OUT_DIR/submitted.sbatch"

    echo "========================================================"
    echo "Job ${SLURM_JOB_ID}: $status -- $reason"
    echo "Staging out through ${STAGE_HOST}:"
    echo "  $OUT_DIR -> $dest"
    echo "  $RUN_CACHE_DIR -> $CACHE_DEST"
    stage_dir "$OUT_DIR" "$dest" || { stage_ok=0; echo "WARNING: output stage-out FAILED" >&2; }
    stage_dir "$RUN_CACHE_DIR" "$CACHE_DEST" || { stage_ok=0; echo "WARNING: cache stage-out FAILED" >&2; }

    if [ "$stage_ok" -eq 1 ]; then
        safe_rm_work "$jobdir" && echo "Stage-out OK; removed $jobdir."
    else
        cat >&2 <<EOF
WARNING: stage-out incomplete -- NOT deleting $jobdir (it holds the only copy).
Recover from the Saion LOGIN node (it can write $BUCKET_PROJECT directly):
    mkdir -p "$dest" "$CACHE_DEST"
    rsync -a "$OUT_DIR"/ "$dest"/
    rsync -a "$RUN_CACHE_DIR"/ "$CACHE_DEST"/
    rm -rf "$jobdir"
EOF
    fi
    echo "========================================================"
}
trap cleanup EXIT

# ==============================================================================
# JOB DESCRIPTION
# ==============================================================================
# Same JOB_LOG.md entry as the Deigo pipelines (no-op when submitted via
# submit.sh, which already wrote it). Non-fatal: provenance, not the run.
"$PROJ_DIR/scripts/batch/generate_job_description.sh" \
    "$PROJ_DIR" "$SLURM_JOB_ID" \
    "$0" \
    "$PROJ_DIR/scripts/src/$PY_NAME" \
    || echo "WARNING: generate_job_description.sh failed (non-fatal)" >&2

# ==============================================================================
# STAGE IN: code + helper modules -> $jobdir (inside $WORK_BASE)
# ==============================================================================
# gpu_edm.py must be staged: the pipeline puts its own directory at the head
# of sys.path, and a missing gpu_edm would turn every GPU MDE call into an
# ImportError-recorded dead end (an empty network, not a crash).
# gpu_budget.py likewise, and it fails harder: cleaned_helper_functions
# imports it on the GPU path and again from both CCM except-clauses, so a
# missing copy is an unhandled ImportError mid-run rather than a quiet
# degradation. It is a separate module from gpu_edm for the same reason
# gpu_edm is separate -- it must import cleanly with no torch on Deigo.
# The mde_gmn/ PACKAGE is staged too, not just the entry script: the pipeline
# is split across its modules (see scripts/src/mde_gmn/__init__.py). Without
# it the entry script fails at import.
cp -a \
    "$PROJ_DIR/scripts/src/$PY_NAME" \
    "$PROJ_DIR/scripts/src/cleaned_helper_functions.py" \
    "$PROJ_DIR/scripts/src/gpu_edm.py" \
    "$PROJ_DIR/scripts/src/gpu_budget.py" \
    "$STAGE_SRC"/
cp -a "$PROJ_DIR/scripts/src/mde_gmn" "$STAGE_SRC"/ || die "staging source files failed"
[ -r "$BUCKET_DATA_DIR/${DATA_FILE}" ] || die "input data not found on Bucket: $BUCKET_DATA_DIR/${DATA_FILE}"
cd "$STAGE_SRC" || die "cd $STAGE_SRC failed"

echo "host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} cpus=${SLURM_CPUS_PER_TASK} gpus=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-?}}"
staged_commit="$(git -C "$PROJ_DIR" rev-parse HEAD 2>/dev/null || echo '?')"
echo "commit=${staged_commit:0:12} jobdir=$jobdir"
# SCOPE_PAIR_COMMIT is set by submit_saion_scope_pair.sh at submission time. The
# sources above were copied out of $PROJ_DIR just now, not when this job was
# queued, so a job that waited days behind another may have staged different
# code than its partner -- which would quietly invalidate a staged-vs-union
# comparison. Warn rather than die: the runbook's resume path legitimately
# resubmits after a code fix, and dying here would throw away the queue wait.
if [ -n "${SCOPE_PAIR_COMMIT:-}" ] && [ "$SCOPE_PAIR_COMMIT" != "$staged_commit" ]; then
    echo "WARNING: staged sources are at ${staged_commit:0:12} but this job was" >&2
    echo "         submitted expecting ${SCOPE_PAIR_COMMIT:0:12}. The repository changed" >&2
    echo "         while this job was queued, so it is NOT running the same code as the" >&2
    echo "         other job in its pair. Results are not directly comparable." >&2
fi

# ==============================================================================
# BACKGROUND: GPU usage log + hourly cache sync
# ==============================================================================
(
    while sleep 3600; do
        stage_dir "$RUN_CACHE_DIR" "$CACHE_DEST" >/dev/null 2>&1 \
            || echo "$(date -Iseconds) WARNING: hourly cache sync to $CACHE_DEST failed (will retry)" >&2
    done
) &
BG_PIDS+=($!)

# ==============================================================================
# RUN
# ==============================================================================
export TSLS_MDE_GMN_GPU_ENABLED=1

export TSLS_MDE_GMN_DATA_DIR="$BUCKET_DATA_DIR"
export TSLS_MDE_GMN_DATA_FILE="${DATA_FILE}"
export TSLS_MDE_GMN_RESULTS_DIR="$OUT_DIR"
export TSLS_MDE_GMN_MODE="${MODE}"
export TSLS_MDE_GMN_GMN_MODE="${GMN_MODE}"
export TSLS_MDE_GMN_TARGETS="${TARGETS}"
export TSLS_MDE_GMN_D="${MDE_D}"
export TSLS_MDE_GMN_TP="${TP}"
export TSLS_MDE_GMN_GMN_TP="${GMN_TP}"
export TSLS_MDE_GMN_MDE_MIN_GAIN="${MDE_MIN_GAIN}"
export TSLS_MDE_GMN_EMBED_DIM_RHO_MIN="${EMBED_DIM_RHO_MIN}"
export TSLS_MDE_GMN_CROSS_MAP_RHO_MIN="${CROSS_MAP_RHO_MIN}"
export TSLS_MDE_GMN_EMBED_DIM_RHO_MIN_F1="${EMBED_DIM_RHO_MIN_F1}"
export TSLS_MDE_GMN_CROSS_MAP_RHO_MIN_F1="${CROSS_MAP_RHO_MIN_F1}"
export TSLS_MDE_GMN_CCM_PREFILTER_RHO_MIN="${CCM_PREFILTER_RHO_MIN}"
export TSLS_MDE_GMN_CCM_PREFILTER_E="${CCM_PREFILTER_E}"
export TSLS_MDE_GMN_LIBSIZE_RATIOS="${LIBSIZE_RATIOS}"
export TSLS_MDE_GMN_MULTI_MDE_E="${MULTI_MDE_E}"
export TSLS_MDE_GMN_MULTI_MDE_TARGET_AGGREGATE="${MULTI_MDE_TARGET_AGGREGATE}"
export TSLS_MDE_GMN_EXCLUSION_RADIUS="${EXCLUSION_RADIUS}"
export TSLS_MDE_GMN_NESTED_M="${NESTED_M}"
export TSLS_MDE_GMN_PARALLEL_MDE_SCOPE="${PARALLEL_MDE_SCOPE}"
export TSLS_MDE_GMN_REGION_CHANNEL_CAP="${REGION_CHANNEL_CAP}"
export TSLS_MDE_GMN_DELAY_SOURCE="${DELAY_SOURCE}"
export TSLS_MDE_GMN_LAG_MODE="${LAG_MODE}"
export TSLS_MDE_GMN_LAG_DENSITY="${LAG_DENSITY}"
export TSLS_MDE_GMN_LAG_STEP="${LAG_STEP}"
export TSLS_MDE_GMN_LAG_COUNT="${LAG_COUNT}"
export TSLS_MDE_GMN_DELAY_SET="${DELAY_SET}"
export TSLS_MDE_GMN_LIB_RATIO="${LIB_RATIO}"
export TSLS_MDE_GMN_MDE_VALIDATION_RATIO="${MDE_VALIDATION_RATIO}"
export TSLS_MDE_GMN_GENERATE_LENGTH="${GENERATE_LENGTH}"
export TSLS_MDE_GMN_GENERATE_NOISE="${GENERATE_NOISE}"
export TSLS_MDE_GMN_GENERATE_NOISE_SEED="${GENERATE_NOISE_SEED}"
export GMN_EXCLUSION_RADIUS="${GMN_EXCLUSION_RADIUS}"
export TSLS_MDE_GMN_FORECAST_PLOT_RANGE="${FORECAST_PLOT_RANGE}"
export TSLS_MDE_GMN_GENERATE_START_MIN_OFFSET="${GENERATE_START_MIN_OFFSET}"
export TSLS_MDE_GMN_GENERATE_START_MAX_OFFSET="${GENERATE_START_MAX_OFFSET}"
export TSLS_MDE_GMN_GENERATE_START_STEP="${GENERATE_START_STEP}"
export TSLS_MDE_GMN_E_MIN="${E_MIN}"
export TSLS_MDE_GMN_E_MAX="${E_MAX}"
export TSLS_MDE_GMN_E_STEP="${E_STEP}"
export TSLS_MDE_GMN_TAU_MIN="${TAU_MIN}"
export TSLS_MDE_GMN_TAU_MAX="${TAU_MAX}"
export TSLS_MDE_GMN_TAU_STEP="${TAU_STEP}"
export TSLS_MDE_GMN_THETA_RANGE="${THETA_RANGE}"
export TSLS_MDE_GMN_MAX_TP="${MAX_TP}"
export TSLS_MDE_GMN_GRID_RHO_THRESHOLD="${GRID_RHO_THRESHOLD}"
export TSLS_MDE_GMN_RUN_MDE_DIAGNOSTICS="${RUN_MDE_DIAGNOSTICS}"
export TSLS_MDE_GMN_RUN_MDE_EVALUATE="${RUN_MDE_EVALUATE}"
export TSLS_MDE_GMN_CORES="${SLURM_CPUS_PER_TASK}"
export TSLS_MDE_GMN_METRIC="${METRIC}"
export TSLS_MDE_GMN_TARGET_METRIC="${TARGET_METRIC}"
export TSLS_MDE_GMN_GAMMA_DELTA="${GAMMA_DELTA}"
export TSLS_MDE_GMN_CCM_WINDOW_ROWS="${CCM_WINDOW_ROWS}"
export TSLS_MDE_GMN_CCM_LIB_SIZE_PCTS="${CCM_LIB_SIZE_PCTS}"
export TSLS_MDE_GMN_CCM_SAMPLE="${CCM_SAMPLE}"
export TSLS_MDE_GMN_CCM_SLOPE="${CCM_SLOPE}"
export TSLS_MDE_GMN_TARGET_RANKING_WINDOW_ROWS="${TARGET_RANKING_WINDOW_ROWS}"
export TSLS_MDE_GMN_MIN_PRED_EVENTS="${MIN_PRED_EVENTS}"
export TSLS_MDE_GMN_MAX_WINDOW_ROWS="${MAX_WINDOW_ROWS}"
export TSLS_MDE_GMN_CCM_LIB_SIZES_ROWS="${CCM_LIB_SIZES_ROWS}"
export TSLS_MDE_GMN_REUSE_BEST_PARAMS="${REUSE_BEST_PARAMS}"
# Lookups scan the durable bucket pool (read-only is all a lookup needs);
# this run's own new entries go to $jobdir/cache and are staged out to CACHE_DEST.
export TSLS_MDE_GMN_PREV_RUNS_DIR="$CACHE_ROOT"
export TSLS_MDE_GMN_RUN_CACHE_DIR="$RUN_CACHE_DIR"

export TSLS_MDE_GMN_SPIKE_TRANSFORM="${SPIKE_TRANSFORM}"
export TSLS_MDE_GMN_GAMMA_ORDER="${GAMMA_ORDER}"
export TSLS_MDE_GMN_GAMMA_TAUS_MS="${GAMMA_TAUS_MS}"
export TSLS_MDE_GMN_GAMMA_LOG_KAPPA="${GAMMA_LOG_KAPPA}"
export TSLS_MDE_GMN_LMU_THETA_MS="${LMU_THETA_MS}"
export TSLS_MDE_GMN_LMU_D="${LMU_D}"
export TSLS_MDE_GMN_LMU_DISCRETIZATION="${LMU_DISCRETIZATION}"
export TSLS_MDE_GMN_LMU_PRESMOOTH_ORDER="${LMU_PRESMOOTH_ORDER}"
export TSLS_MDE_GMN_LMU_REDUCTION="${LMU_REDUCTION}"
export TSLS_MDE_GMN_LOGSIG_W_MS="${LOGSIG_W_MS}"
export TSLS_MDE_GMN_LOGSIG_DEPTH="${LOGSIG_DEPTH}"
export TSLS_MDE_GMN_LOGSIG_PRESMOOTH_ORDER="${LOGSIG_PRESMOOTH_ORDER}"
export TSLS_MDE_GMN_LOGSIG_REDUCTION="${LOGSIG_REDUCTION}"
export TSLS_MDE_GMN_N_MIN_EVENTS="${N_MIN_EVENTS}"
export TSLS_MDE_GMN_WARMUP_ROWS="${WARMUP_ROWS}"
export TSLS_MDE_GMN_EMBED_TAU="${EMBED_TAU}"
export TSLS_MDE_GMN_EMBED_TAU_METHOD="${EMBED_TAU_METHOD}"
export TSLS_MDE_GMN_EMBED_TAU_MAX="${EMBED_TAU_MAX}"
export TSLS_MDE_GMN_EXTERNAL_INPUT_MODE="${EXTERNAL_INPUT_MODE}"
export TSLS_MDE_GMN_ROLLOUT_N_STARTS="${ROLLOUT_N_STARTS}"
export TSLS_MDE_GMN_ROLLOUT_LENGTH="${ROLLOUT_LENGTH}"
export TSLS_MDE_GMN_ROLLOUT_HORIZONS_S="${ROLLOUT_HORIZONS_S}"
export TSLS_MDE_GMN_ROLLOUT_WORKERS="${ROLLOUT_WORKERS}"
export TSLS_MDE_GMN_PH_LANDMARKS="${PH_LANDMARKS}"
echo "ABLATION_LABEL=${ABLATION_LABEL} SPIKE_TRANSFORM=${SPIKE_TRANSFORM}"
python3 -V

# Python runs in the background so the USR1 trap can fire while it runs
# (bash defers traps until a FOREGROUND child exits). `wait` returns early
# when a trap fires (status > 128 while Python is still alive), hence the
# loop: keep waiting until the process is really gone, and keep the status
# of the wait that saw it go.
python3 -u "$PY_NAME" &
PY_PID=$!
while :; do
    wait "$PY_PID"
    rc=$?
    if ! kill -0 "$PY_PID" 2>/dev/null; then
        PY_STATUS=$rc
        break
    fi
done
echo "$PY_NAME exited with status ${PY_STATUS}"
# Outcome, STATUS.txt, stage-out and $jobdir cleanup: the EXIT trap above.
exit "$PY_STATUS"
