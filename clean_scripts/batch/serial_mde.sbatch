#!/bin/bash -l
# =============================================================================
# EDIT FOR YOUR ACCOUNT -- the #SBATCH block below
# =============================================================================
# Slurm parses #SBATCH directives itself, before any shell runs, so NOTHING in
# this block can be a variable: --output, --error and --mail-user have to be
# literal. They are the only account-specific strings in this file; everything
# else comes from ../saion/cluster_env.sh, which is configurable by
# environment variable. --output/--error must name the same directory as that
# file's PAO_SLURM_LOG_DIR, and the directory must exist before you submit.
# =============================================================================
#SBATCH --job-name=serial_mde
#SBATCH --output=/home/j/j-tominaga/saion_slurm_logs/%x-%j.out
#SBATCH --error=/home/j/j-tominaga/saion_slurm_logs/%x-%j.err
#SBATCH --mail-user=j-tominaga@oist.jp
#SBATCH --mail-type=END,FAIL
#
# Resources. No --gres: this pipeline runs on the CPU. See
# ../docs/deployment.md for how the memory and time were derived, and what to
# change if the job is killed -- in short: if it is OOM-killed, lower
# --cpus-per-task (= worker count), not MDE_D.
#SBATCH --partition=intel
#SBATCH --cpus-per-task=18
#SBATCH --mem=240G
#SBATCH --time=6-00:00:00
# 30 min before the time limit, Slurm sends USR1 to this batch shell only. The
# handler below stops Python cleanly so there is time to stage results and the
# MDE cache out to durable storage before the hard kill.
#SBATCH --signal=B:USR1@1800

# =============================================================================
# serial-MDE pipeline (CPU)
# =============================================================================
# Runs ../src/run_serial_mde.py -- the SERIAL network builder, one MDE call at
# a time in breadth-first waves. Every parameter it reads is a TSLS_MDE_GMN_*
# environment variable exported at the bottom of this file, and every one of
# them means the same thing as in saion_parallel_mde.sbatch, minus
# PARALLEL_MDE_SCOPE (which only the parallel builder has).
#
# WHEN TO USE THIS RATHER THAN THE PARALLEL ONE. The serial builder is the
# reference implementation: it is what the parallel one's 'staged' scope is
# asserted equal to. Use it to check a result, or on a CPU-only machine. For a
# production run on a GPU node, use saion_parallel_mde.sbatch -- the serial
# builder cannot use more workers than the current BFS wave has nodes, which
# is often two, so most of the machine idles at every wave barrier.
#
#   DATA FLOW
#     $BUCKET_DATA_DIR/$DATA_FILE   read directly, never copied to home
#     ../src (code only) --copy--> $WORK_BASE/pmde.<job>.XXXX/
#         src/      pipeline + helper modules
#         results/  TSLS_MDE_GMN_RESULTS_DIR   (all pipeline output)
#         cache/    TSLS_MDE_GMN_RUN_CACHE_DIR (this run's cache entries)
#     $BUCKET_PROJECT/tsls_mde_gmn_cache   TSLS_MDE_GMN_PREV_RUNS_DIR, READ-ONLY
#     cache/   --hourly + at exit, via login node--> .../tsls_mde_gmn_cache/job_<id>/
#     results/ --at exit, via login node-->          .../tsls_mde_gmn_outputs/job_<id>/
#                                                    (or job_<id>_FAIL/ on failure)
#     The scratch dir is removed only after BOTH copies succeed, and every
#     mkdir/rm/stage-out is guarded to stay inside those roots (path_under /
#     safe_rm_work in ../saion/cluster_env.sh).
#
#   OUTCOME  results/STATUS.txt says COMPLETED or FAIL plus the reason, and
#     reports CCM_DEAD_ENDS separately -- candidates genuinely evaluated and
#     rejected are FINDINGS, not failures.
#
#   RESUMING  Every finished MDE node is cached immediately and synced hourly,
#     so resubmitting this same file after a timeout or crash skips every
#     completed node (REUSE_BEST_PARAMS=true). Only nodes in flight at the
#     kill are recomputed.
#
#   GRACEFUL CANCEL  `scancel --batch --signal=USR1 <jobid>` stops the run AND
#     stages everything out. Plain `scancel` leaves only ~30s, so rely on the
#     hourly cache sync and the manual recovery command it prints.
#
# Submit with ../saion/submit_saion.sh (which picks the partition, then goes
# through submit.sh so the job gets a JOB_LOG.md entry). Full procedure and
# all parameter rationale: ../docs/deployment.md and ../docs/parameters.md.
# =============================================================================

# =============================================================================
# MODIFIABLE PARAMETERS
# =============================================================================
# One line each. ../docs/parameters.md has the calibration evidence behind
# every value that has any, and says which ones are guesses.

# ---- Data -------------------------------------------------------------------
DATA_FILE="J1620210605_epoch_8_freq_40_Hz.csv"   # the 40 Hz (25 ms-bin) spike-train file

# ---- Pipeline shape ---------------------------------------------------------
MODE="MDE-GMN"                # MDE-GMN: two-stage nested MDE. GMN: skip MDE, build an interaction matrix instead
GMN_MODE="combined"           # combined: one network + one joint MDE across all targets. separate: one of each per target
TARGETS="head_position_x,head_position_y"   # target columns (never TSLS-converted)

# ---- MDE: variable selection (unused when MODE=GMN) ------------------------
MDE_D=10                      # drivers per node; also the ceiling on per-candidate auto-selected E. Half of the documented 20
TP=20                         # prediction interval every candidate is ranked against
EMBED_DIM_RHO_MIN=0.2         # embedding-dimension skill gate -- CORRELATION-scored nodes (the behavioural targets)
CROSS_MAP_RHO_MIN=0.2         # cross-map skill gate -- CORRELATION-scored nodes
EMBED_DIM_RHO_MIN_F1=0.15     # the same gate for f1-scored (neuronal) nodes; see saion_parallel_mde.sbatch for why it differs
CROSS_MAP_RHO_MIN_F1=0.15     # the same gate for f1-scored (neuronal) nodes
CCM_PREFILTER_RHO_MIN=0.2     # drop observables cross-mapping no target above this. 0 disables the pre-filter
CCM_PREFILTER_E=7             # the pre-filter's own static E -- NOT a reuse of MULTI_MDE_E; see parameters.md
LIBSIZE_RATIOS="0.05,0.10,0.20,0.40,0.6,0.8,0.9"   # pre-filter library sizes, as fractions of its window
MULTI_MDE_E=8                 # ceiling for the joint call's per-candidate auto-selected E (GMN_MODE=combined only)
MULTI_MDE_TARGET_AGGREGATE="mean"   # how per-target scores combine: mean (don't penalise uneven drivers) or min (demand both)
EXCLUSION_RADIUS=0            # Theiler-window FLOOR for MDE; proper_mde raises the effective value per dimension on top of it
NESTED_M="${NESTED_M:-0}"     # overridable; 0 = uncapped. additional nested-MDE rounds beyond each target's own selection
REGION_CHANNEL_CAP="${REGION_CHANNEL_CAP:-78}"   # overridable; first N raw channels per region (CA1_/mPFC_/OFC_). 78x3 = 234 of 468. 999 = uncapped

# ---- Stage 2: the lag set --------------------------------------------------
# Three ways to define it, differing only in which quantity you pin -- a lag
# set is a stride and how far back it reaches, so fixing two of
# {stride, count, depth} determines the third:
#   lag_mode   pins the DEPTH, count follows (LAG_MODE/LAG_DENSITY/LAG_STEP)
#   tau_count  pins the COUNT, depth follows (LAG_STEP as tau x LAG_COUNT)
#   explicit   pins EVERY VALUE (DELAY_SET)
# Stage 2 offers MDE one candidate column per (observable x lag) and Stage 2 is
# where essentially all the runtime goes, so the COUNT is a compute budget.
# Reach for tau_count when you need to hold that budget directly; under
# lag_mode it is a side effect of GENERATE_LENGTH, which means something else
# entirely. With the values below, lag_mode resolves to 50 lags per observable.
DELAY_SOURCE="${DELAY_SOURCE:-lag_mode}"   # lag_mode | tau_count | explicit
LAG_MODE="generation_length_adjusted"      # depth = GENERATE_LENGTH. Other: every_lag_possible (as deep as the library allows)
LAG_DENSITY="strided"                      # sample that depth every LAG_STEP rows. Other: dense (every integer lag)
LAG_STEP="${LAG_STEP:-8}"                  # the stride (lag_mode) / tau (tau_count)
LAG_COUNT="${LAG_COUNT:-10}"               # how many lags -- tau_count only
DELAY_SET="${DELAY_SET:-0,-1,-3,-5,-10}"   # explicit only. DELAY_SET=0 resolves to an EMPTY lag set: no Stage 2 picks at all
# NOTE: STAGE2_ALLOW_REAL_OBSERVABLES is retired. Stage 2 considers lagged
# columns only and never recurses, so there was nothing for it to select.
# load_config() REJECTS the variable rather than ignoring it.

# ---- Lib/pred split --------------------------------------------------------
# Three-way split of the TSLS-converted, burn-in-trimmed row count (which is
# data-driven -- it depends on which channel fires slowest):
#   [0, A)    MDE library        LIB_RATIO of the usable rows
#   [A, B)    MDE validation     MDE_VALIDATION_RATIO of the remainder
#   [B, end)  Generate/Forecast validation
# Network construction and the reported fidelity therefore never share rows.
LIB_RATIO=0.7
MDE_VALIDATION_RATIO=0.5
GENERATE_LENGTH=400          # steps each Generate() free-runs for. 0 = the whole held-out window
GMN_EXCLUSION_RADIUS=0       # Theiler window for GMN's OWN Generate()/Forecast() search -- a separate EDM context from EXCLUSION_RADIUS
FORECAST_PLOT_RANGE="windowed"   # x-axis of the final diagnostic figure: full | windowed

# ---- Generate() start-point sweep ------------------------------------------
# Try starting Generate() at several rows and keep whichever best matches the
# held-out data. MIN=0/MAX=1 tries only offset 0. Each offset is one full
# Generate() of the whole network; see saion_parallel_mde.sbatch before widening.
GENERATE_START_MIN_OFFSET="${GENERATE_START_MIN_OFFSET:-0}"    # overridable
GENERATE_START_MAX_OFFSET="${GENERATE_START_MAX_OFFSET:-20}"   # overridable
GENERATE_START_STEP="${GENERATE_START_STEP:-1}"                # overridable

# ---- E/tau search (MODE=GMN only; THETA_RANGE is used by both modes) -------
E_MIN=3      # inclusive lower bound of the E search; must stay >= 2
E_MAX=13     # EXCLUSIVE upper bound (python range())
E_STEP=4
TAU_MIN=-9   # inclusive
TAU_MAX=0    # EXCLUSIVE
TAU_STEP=4
THETA_RANGE="0.01,0.1,0.3,0.5,0.75,1,1.5,2,3,4,5,6,7,8,9"   # the only thing searched per node

# ---- TSLS-aware scoring ----------------------------------------------------
METRIC="f1"                  # how NEURON predictions are scored: f1 | gamma | correlation | mae. f1 credits spike TIMING
TARGET_METRIC="correlation"  # how TARGET predictions are scored. Targets are continuous: f1/gamma on them yields NaN
GAMMA_DELTA=2                # coincidence window (bins), METRIC=gamma only
CCM_SAMPLE=10                # random library resamples averaged per libSize. Also the GPU batch dimension: halving it buys time AND memory
CCM_SLOPE=0.01               # convergence-slope threshold for the CCM gate
CCM_WINDOW_ROWS=800          # MODE=GMN only -- bounds the interaction matrix's CCM window
CCM_LIB_SIZE_PCTS="2,3,5,8,12,20,30,50"   # MODE=GMN only -- its library sweep, as % of CCM_WINDOW_ROWS

# ---- Window sizing (MODE=MDE-GMN only) -------------------------------------
# Separates two things one global window used to conflate: how much data a node
# is SCORED over, and which library sizes the CCM gate sweeps.
TARGET_RANKING_WINDOW_ROWS=1000   # the depth-0 seed's own fixed window (behavioural targets have no event concept)
MIN_PRED_EVENTS=20                # reset events a neuron's window must contain before f1 is signal. Calibrated; see parameters.md
MAX_WINDOW_ROWS=30000             # compute ceiling on that search -- a backstop, not a tuned value
CCM_LIB_SIZES_ROWS="16,40,160,400,800"   # the CCM gate's library sweep, ABSOLUTE rows (never a % of the window above)

# ---- Diagnostics -----------------------------------------------------------
MAX_TP=20                    # max horizon for the Tp-vs-rho sweep
GRID_RHO_THRESHOLD=0.9       # highlight threshold on the univariate E/tau grid figure
RUN_MDE_DIAGNOSTICS=true     # the MDE-only figures
RUN_MDE_EVALUATE=false       # the PCA/diffusion-map baseline. Defaults OFF: the diffusion-map fit can HANG; see parameters.md

# ---- Search reuse ----------------------------------------------------------
REUSE_BEST_PARAMS=true       # look for an exact-match cached result before computing. Always safe to leave on


# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================
# Deliberately no `set -e`: every failure path must still reach the EXIT trap's
# stage-out. The steps that must not fail check explicitly instead.
set -uo pipefail

# Defines PROJ_DIR, SLURM_LOG_DIR, WORK_BASE, BUCKET_PROJECT, BUCKET_DATA_DIR,
# STAGE_HOST and the stage_dir/path_under/safe_rm_work helpers.
# shellcheck source=../saion/cluster_env.sh
source "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../saion/cluster_env.sh"

PY_NAME="run_serial_mde.py"
VENV_DIR="${PAO_VENV_DIR:-$(dirname "$PROJ_DIR")/.venv}"

die() { echo "FATAL: $*" >&2; exit 2; }
[ -n "${SLURM_JOB_ID:-}" ] || die "submit with sbatch, not bash"
[ -d "$(dirname "$WORK_BASE")" ] || die "$(dirname "$WORK_BASE") not found -- compute nodes only"
[ -x "$VENV_DIR/bin/python3" ] || die "$VENV_DIR missing -- create it on the login node first"

# No `module load python` -- the venv's interpreter is self-contained.
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
export MPLBACKEND=Agg
# Workers are processes (TSLS_MDE_GMN_CORES), so keep BLAS/OpenMP
# single-threaded inside each -- otherwise N workers each spin up N threads and
# oversubscribe the cores this job was actually given.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
CACHE_ROOT="$BUCKET_PROJECT/tsls_mde_gmn_cache"
OUTPUT_ARCHIVE_ROOT="$BUCKET_PROJECT/tsls_mde_gmn_outputs"
CACHE_DEST="$CACHE_ROOT/job_${SLURM_JOB_ID}"

# Per-job scratch: a unique mktemp dir, deleted at the end.
mkdir -p "$WORK_BASE" || die "cannot create $WORK_BASE"
jobdir="$(mktemp -d "$WORK_BASE/smde.${SLURM_JOB_ID}.XXXXXX")" || die "mktemp in $WORK_BASE failed"
OUT_DIR="$jobdir/results"
RUN_CACHE_DIR="$jobdir/cache"
STAGE_SRC="$jobdir/src"
mkdir -p "$OUT_DIR" "$RUN_CACHE_DIR" "$STAGE_SRC" "$jobdir/tmp" || die "cannot populate $jobdir"
# Temp files stay inside this job's dir rather than the node-wide /tmp, and go
# away with it.
export TMPDIR="$jobdir/tmp"
export MPLCONFIGDIR="$jobdir/tmp/matplotlib"

SLURM_OUT="$SLURM_LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
SLURM_ERR="$SLURM_LOG_DIR/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"

PY_PID=""
PY_STATUS=""
STOP_REASON=""
BG_PIDS=()

# =============================================================================
# STOP / OUTCOME / STAGE-OUT
# =============================================================================
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

# count_lines PATTERN FILE... -- how many lines match, 0 if the files are gone.
count_lines() { local pat="$1"; shift; cat "$@" 2>/dev/null | grep -c -E "$pat"; }

cleanup() {
    trap - EXIT USR1 TERM
    stop_python
    for pid in "${BG_PIDS[@]}"; do
        pkill -P "$pid" 2>/dev/null
        kill "$pid" 2>/dev/null
    done

    local slurm_state="" status reason dest stage_ok=1
    for _ in 1 2 3; do
        slurm_state="$(sacct -j "$SLURM_JOB_ID" --format=State -P -n 2>/dev/null | head -1 | tr -d ' ')"
        [ -n "$slurm_state" ] && break
        sleep 2
    done

    # What the log says went wrong, in three counts, because they mean three
    # different things. (The GPU sibling has two more, for CUDA errors and
    # allocator advisories; GPU_ENABLED=0 here, so neither can occur.)
    #
    #   networks_skipped  a network whose Generate() never rebuilt produces no
    #                     plots, fidelity or output -- and the pipeline says so
    #                     and carries on to exit 0. FAILed on below so it
    #                     cannot pass unnoticed.
    #   mde_failures      per-node MDE calls that raised and were recorded as
    #                     dead ends.
    #   ccm_dead_ends     candidates genuinely evaluated and rejected. FINDINGS,
    #                     not failures, and reported separately for exactly
    #                     that reason.
    local networks_skipped mde_failures ccm_dead_ends
    networks_skipped="$(count_lines 'no successful Generate\(\) rebuild' "$SLURM_OUT")"
    mde_failures="$(count_lines 'MDE call FAILED' "$SLURM_OUT")"
    ccm_dead_ends="$(count_lines 'CCM dead end:' "$SLURM_OUT")"

    if [ "${PY_STATUS:-unset}" = "0" ] && [ -z "$STOP_REASON" ] \
            && [ "$networks_skipped" -eq 0 ]; then
        status=COMPLETED; reason="pipeline exited 0"
        dest="$OUTPUT_ARCHIVE_ROOT/job_${SLURM_JOB_ID}"
    else
        status=FAIL
        if   [ -n "$STOP_REASON" ]; then reason="$STOP_REASON"
        elif [ "${PY_STATUS:-unset}" != "0" ] && [ -n "${PY_STATUS:-}" ]; then
            reason="pipeline exited with status ${PY_STATUS}"
        elif [ -z "${PY_STATUS:-}" ]; then
            reason="job ended before the pipeline started (setup failure)"
        else
            reason="pipeline exited 0 but ${networks_skipped} network(s) produced no output (Generate() never rebuilt) -- see the 'FINAL Generate() rebuild FAILED' line(s)"
        fi
        dest="$OUTPUT_ARCHIVE_ROOT/job_${SLURM_JOB_ID}_FAIL"
    fi

    {
        echo "STATUS: $status"
        echo "REASON: $reason"
        echo "PY_STATUS: ${PY_STATUS:-unset}"
        echo "SLURM_STATE_AT_EXIT: ${slurm_state:-unknown (check: sacct -j $SLURM_JOB_ID)}"
        echo "HOST: $(hostname)  PARTITION: ${SLURM_JOB_PARTITION:-?}"
        echo "NETWORKS_SKIPPED: $networks_skipped"
        echo "CCM_DEAD_ENDS: $ccm_dead_ends  (evaluated and rejected -- findings, not failures)"
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
Recover from the LOGIN node (it can write $BUCKET_PROJECT directly):
    mkdir -p "$dest" "$CACHE_DEST"
    rsync -a "$OUT_DIR"/ "$dest"/
    rsync -a "$RUN_CACHE_DIR"/ "$CACHE_DEST"/
    rm -rf "$jobdir"
EOF
    fi
    echo "========================================================"
}
trap cleanup EXIT

# =============================================================================
# JOB DESCRIPTION -- provenance, not the run. Non-fatal.
# =============================================================================
# A no-op when submitted via submit.sh, which already wrote the entry.
"$PROJ_DIR/batch/generate_job_description.sh" \
    "$PROJ_DIR" "$SLURM_JOB_ID" "$0" "$PROJ_DIR/src/$PY_NAME" \
    || echo "WARNING: generate_job_description.sh failed (non-fatal)" >&2

# =============================================================================
# STAGE IN: code -> $jobdir
# =============================================================================
# edm.py is the EDM engine -- the pipeline is nothing without it -- and
# mde_gmn/ IS the pipeline, so a missing package is an import failure, not a
# degradation. gpu_edm.py/gpu_budget.py are staged too even though
# GPU_ENABLED=0 never reaches them: mde.is_gpu_memory_failure() imports
# gpu_budget lazily and treats ImportError as "no GPU path here", so their
# absence would be silent, and staging them costs nothing and keeps this list
# identical to the GPU sibling's.
cp -a \
    "$PROJ_DIR/src/$PY_NAME" \
    "$PROJ_DIR/src/edm.py" \
    "$PROJ_DIR/src/gpu_edm.py" \
    "$PROJ_DIR/src/gpu_budget.py" \
    "$STAGE_SRC"/ || die "staging source files failed"
cp -a "$PROJ_DIR/src/mde_gmn" "$STAGE_SRC"/ || die "staging the mde_gmn package failed"
[ -r "$BUCKET_DATA_DIR/${DATA_FILE}" ] || die "input data not found: $BUCKET_DATA_DIR/${DATA_FILE}"
cd "$STAGE_SRC" || die "cd $STAGE_SRC failed"

echo "host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} cpus=${SLURM_CPUS_PER_TASK}"
staged_commit="$(git -C "$PROJ_DIR" rev-parse HEAD 2>/dev/null || echo '?')"
echo "commit=${staged_commit:0:12} jobdir=$jobdir"

# =============================================================================
# BACKGROUND: hourly cache sync
# =============================================================================
(
    while sleep 3600; do
        stage_dir "$RUN_CACHE_DIR" "$CACHE_DEST" >/dev/null 2>&1 \
            || echo "$(date -Iseconds) WARNING: hourly cache sync to $CACHE_DEST failed (will retry)" >&2
    done
) &
BG_PIDS+=($!)

# =============================================================================
# RUN
# =============================================================================
# No GPU here: the CPU engines are used throughout, and the worker pools fork
# rather than spawn (CUDA cannot fork, which is the only reason spawn exists).
export TSLS_MDE_GMN_GPU_ENABLED=0
export TSLS_MDE_GMN_DATA_DIR="$BUCKET_DATA_DIR"
export TSLS_MDE_GMN_DATA_FILE="$DATA_FILE"
export TSLS_MDE_GMN_RESULTS_DIR="$OUT_DIR"
export TSLS_MDE_GMN_CORES="$SLURM_CPUS_PER_TASK"
# Lookups scan the durable cache pool read-only; this run's new entries go to
# $jobdir/cache and are staged out from there.
export TSLS_MDE_GMN_PREV_RUNS_DIR="$CACHE_ROOT"
export TSLS_MDE_GMN_RUN_CACHE_DIR="$RUN_CACHE_DIR"

# Every remaining parameter, exported under its TSLS_MDE_GMN_ name. The loop
# replaces ~60 near-identical export lines; `export VAR` on a shell variable
# that is already set is exactly what each of those lines did, and the two
# names that do not follow the prefix convention are handled beside it.
for _p in MODE GMN_MODE TARGETS TP EMBED_DIM_RHO_MIN CROSS_MAP_RHO_MIN \
          EMBED_DIM_RHO_MIN_F1 CROSS_MAP_RHO_MIN_F1 \
          CCM_PREFILTER_RHO_MIN CCM_PREFILTER_E LIBSIZE_RATIOS MULTI_MDE_E \
          MULTI_MDE_TARGET_AGGREGATE EXCLUSION_RADIUS NESTED_M \
          REGION_CHANNEL_CAP DELAY_SOURCE LAG_MODE \
          LAG_DENSITY LAG_STEP LAG_COUNT DELAY_SET LIB_RATIO \
          MDE_VALIDATION_RATIO GENERATE_LENGTH FORECAST_PLOT_RANGE \
          GENERATE_START_MIN_OFFSET GENERATE_START_MAX_OFFSET \
          GENERATE_START_STEP E_MIN E_MAX E_STEP TAU_MIN TAU_MAX TAU_STEP \
          THETA_RANGE MAX_TP GRID_RHO_THRESHOLD RUN_MDE_DIAGNOSTICS \
          RUN_MDE_EVALUATE METRIC TARGET_METRIC GAMMA_DELTA CCM_WINDOW_ROWS \
          CCM_LIB_SIZE_PCTS CCM_SAMPLE CCM_SLOPE TARGET_RANKING_WINDOW_ROWS \
          MIN_PRED_EVENTS MAX_WINDOW_ROWS CCM_LIB_SIZES_ROWS \
          REUSE_BEST_PARAMS; do
    # The ${VAR?...} form fails loudly if a parameter above was deleted but
    # not removed from this list, instead of exporting an empty string.
    export "TSLS_MDE_GMN_$_p=${!_p?parameter $_p is not set above}"
done
unset _p
# MDE_D is TSLS_MDE_GMN_D, and GMN_EXCLUSION_RADIUS is deliberately unprefixed
# to stay visually distinct from the unrelated MDE-side EXCLUSION_RADIUS.
export TSLS_MDE_GMN_D="$MDE_D"
export GMN_EXCLUSION_RADIUS

python3 -V

# Python runs in the BACKGROUND so the USR1 trap can fire while it runs -- bash
# defers traps until a FOREGROUND child exits. `wait` returns early when a trap
# fires (status > 128 while Python is still alive), hence the loop: keep
# waiting until the process is really gone, and keep the status of the wait
# that saw it go.
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
# Outcome, STATUS.txt, stage-out and scratch cleanup: the EXIT trap above.
exit "$PY_STATUS"
