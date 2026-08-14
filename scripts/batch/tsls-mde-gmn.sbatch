#!/bin/bash
#SBATCH --job-name=run_tsls_mde_gmn
#SBATCH --partition=compute
#SBATCH --time=2-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=32
# ADJUST: must be a literal absolute path, not $HOME/$PROJ_DIR -- Slurm
# parses #SBATCH directives itself before this script (or any shell)
# runs, so no shell variable expansion happens here. Pointed directly at
# results/ (a fixed path that already exists) rather than results/job_%j/
# because Slurm opens these files before the job starts, i.e. before
# $SLURM_JOB_ID is known and before this script's own `mkdir -p
# "$OUT_DIR"` below could create that subdirectory -- a per-job
# subdirectory in this path would fail with "couldn't open output file"
# since Slurm doesn't create missing directories. The EXIT trap further
# down (search SLURM LOG RELOCATION) moves these into $OUT_DIR once it
# exists, so they still end up alongside the rest of that job's output.
# Same fix as simple-mde-gmn.sbatch -- its original %x-%j.out/.err
# (relative to whatever directory `sbatch` was invoked from, not
# $OUT_DIR) is why old logs ended up scattered across the repo root and
# scripts/batch/ instead of under results/.
#SBATCH --output=/home/j/j-tominaga/pao_rotation/results/%x-%j.out
#SBATCH --error=/home/j/j-tominaga/pao_rotation/results/%x-%j.err
#SBATCH --mail-user=j-tominaga@oist.jp
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ==============================================================================
# MODIFIABLE PARAMETERS
# ==============================================================================
#
# NOTE ON --time/--mem/--cpus-per-task ABOVE: this pipeline is
# simple-mde-gmn.sbatch's own MDE+GMN pipeline fed by a TSLS-converted
# 40 Hz recording instead of a pre-binned 1 Hz one -- a MUCH larger usable
# row count by default (REGION_CHANNEL_CAP=999/LIB_RATIO=0.7 against the
# full 47999-row 40 Hz file leaves roughly 39000 rows after the TSLS
# burn-in trim, vs. simple-mde-gmn.sbatch's ~1100). As of the rewrite
# documented in scripts/src/tsls-mde-gmn.py's module docstring, this
# pipeline no longer uses dimx.MDE or apps.InteractionMatrix at all (both
# replaced by cleaned_helper_functions.py's proper_mde/proper_ccm, fixing
# an empty-GMN-network bug those tools' own CCM convergence gates hit at
# this pipeline's row counts -- see CCM_WINDOW_ROWS below), which also
# removes the old dimx/apps.InteractionMatrix multiprocessing-pool-nesting
# concern (proper_mde/proper_ccm have no multiprocessing pool of their
# own at all -- see proper_mde's own single-threaded-cost note in
# scripts/src/tsls_example.py) at the cost of losing dimx.MDE's own
# cores-based parallelism for the forward-selection step: a very large
# REGION_CHANNEL_CAP candidate pool will now take proportionally longer
# wall-clock than it used to on a multi-core node.
#
# STRONGLY RECOMMENDED: before an uncapped (REGION_CHANNEL_CAP=999) run,
# submit a small calibration job first -- REGION_CHANNEL_CAP=10-20,
# MDE_D=3-5 -- and time it end-to-end before scaling up. The
# --time/--mem/--cpus-per-task below are simple-mde-gmn.sbatch's own
# starting points, not validated against this pipeline's much larger
# default row count.

# ---- Data ----------------------------------------------------------------
DATA_FILE="J1620210605_epoch_8_freq_40_Hz.csv"   # default: the 40 Hz (25ms-bin) spike-train file

# ---- Overall pipeline shape ------------------------------------------------
# MODE=MDE-GMN (default): proper_mde forward-selects each target's TSLS-
#   encoded drivers first, then GMN is built from TARGETS + the union of
#   MDE-selected columns.
# MODE=GMN: skip MDE; GMN's interaction matrix is built directly from the
#   full REGION_CHANNEL_CAP-limited, TSLS-converted observable set.
MODE="MDE-GMN"                # default: MDE-GMN. Other value: GMN
GMN_MODE="combined"           # default: combined. Other value: separate
TARGETS="head_position_x,head_position_y"   # comma-separated target column names (never TSLS-converted)

# ---- MDE (variable selection; unused when MODE=GMN) ------------------------
MDE_D=4                       # VALIDATION RUN (see NOTE above) -- default: 20, forward-selection dimension ceiling
EMBED_DIM_RHO_MIN=0.2         # default: 0.2, embedding-dimension skill gate
CROSS_MAP_RHO_MIN=0.2         # default: 0.2, cross-map skill gate
# First N *raw* channels per region (CA1_/mPFC_/OFC_), TSLS-converted and
# offered as MDE candidates (or, when MODE=GMN, used directly as the
# interaction-matrix's full node set). 999 is effectively uncapped
# (matches notebooks/tsls-mde-gmn.ipynb's own full-468-channel-set
# decision). Cap this down for a calibration job -- see NOTE above.
REGION_CHANNEL_CAP=15         # VALIDATION RUN (see NOTE above) -- default: 999

# ---- Lib/pred split ----------------------------------------------------------
# Fraction of the TSLS-converted, burn-in-trimmed row count used as
# library (see this pipeline's .py module docstring: TSLS is NaN before
# each channel's own first spike, so the usable window starts only once
# every included channel has fired at least once -- that trim happens
# inside the script, before this ratio is applied). The whole
# post-trim dataset is used; there is no separate row-count cap.
LIB_RATIO=0.7                  # default: 0.7, fraction of usable rows used as library

# ---- E/tau/theta search ------------------------------------------------------
# Controls the GMN node generator's own (E, tau[, theta]) search -- see
# simple-mde-gmn.sbatch for the full rationale (identical here). E_MIN
# must stay >= 2 (SMap's own predictor-exclusion convention). Also feeds
# the interaction-matrix's own (E, tau) grid search (target_E_tau_rho_grid
# in build_network()) -- keep at least a few candidates wide, not a single
# point, so that step isn't locked to one possibly-unlucky (E,tau) guess.
E_MIN=3
E_MAX=13
E_STEP=4
TAU_MIN=-9
TAU_MAX=0
TAU_STEP=4
# 'coarse_retune' (recommended): len(E)*len(tau) full Generate() calls.
# 'joint': len(E)*len(tau)*15 full Generate() calls -- ~15x the cost.
SMAP_SEARCH_MODE="coarse_retune"

# ---- TSLS-aware scoring -------------------------------------------------------
# See scripts/src/tsls-mde-gmn.py's load_config()'s METRIC docstring for
# the full rationale. 'f1' (recommended for TSLS/spike-timing data) scores
# predictions by decoding both true and predicted series back to discrete
# spike/reset events (via a near-zero threshold calibrated from the true
# series itself) and matching them within an adaptive per-event tolerance
# window -- i.e. it specifically credits getting spike *timing* right,
# not just overall curve shape (which plain correlation over the whole
# TSLS sawtooth would reward even for the uninformative ramp segments
# between spikes). Other values: 'gamma' (Kistler coincidence factor,
# alternative event-timing metric), 'correlation' (plain Pearson rho,
# pyEDM's own original default), 'mae'.
METRIC="f1"
# TARGETS (head_position_x/y) are continuous behavioral traces with no
# genuine spike/reset structure -- 'f1'/'gamma' event-decoding on them
# produced NaN for every candidate in production. Always 'correlation'
# (or 'mae'); this is NOT the knob to change for TSLS-aware scoring, use
# METRIC above for that.
TARGET_METRIC="correlation"
GAMMA_DELTA=2                  # coincidence window (bins), only used when METRIC=gamma
# Bounds the row count fed into every CCM convergence gate in this
# pipeline (proper_mde's own CCM step, and build_interaction_matrix()'s)
# -- CRITICAL for calibration, not just a speed knob. Both gates require
# skill to visibly RISE as library size grows from small to large; too
# large a CCM_WINDOW_ROWS means even the smallest tested library
# (CCM_LIB_SIZE_PCTS' lowest percentage of this value) is already past
# where real relationships in this data saturate, so the gate finds no
# rise regardless of whether a real link exists -- confirmed as the
# direct cause of two separate production failures (apps.InteractionMatrix
# reporting ~all-zero interactions at job 35513363; proper_mde selecting
# zero drivers for either target at job 35525431). Recalibrated by testing
# genuinely small (tens of rows) library sizes directly against real data:
# a real, smooth, directionally-asymmetric convergence curve appeared in
# the ~15-400 row range. CCM_WINDOW_ROWS=800 with the percentages below
# reproduces that validated range (~16/24/40/64/96/160/240/400 rows) --
# don't widen this back toward "percentage of the full ~30-45k-row
# dataset" without re-validating against real small-libSize data first,
# and keep the smallest resulting libSize >= (max E in E_RANGE)+2.
CCM_WINDOW_ROWS=800
CCM_LIB_SIZE_PCTS="2,3,5,8,12,20,30,50"  # -> ~16,24,40,64,96,160,240,400 rows at CCM_WINDOW_ROWS=800
CCM_SAMPLE=20                    # random-library resamples averaged per libSize -- higher = less noisy, slower
CCM_SLOPE=0.01                   # convergence-slope threshold (endpoint-improved AND fitted slope > this), used by both proper_mde's own CCM gate and build_interaction_matrix()

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

module purge
# ADJUST: only needed if the cluster doesn't expose Python 3 in PATH by
# default on compute nodes.
module load python/3.11

# ADJUST: absolute path to your uv venv directory.
VENV_DIR="$HOME/pao_rotation/.venv"
source "${VENV_DIR}/bin/activate"

export MPLBACKEND=Agg
# dimx.MDE / apps.InteractionMatrix / pyEDM's EmbedDimension and
# PredictNonlinear already parallelize across SLURM_CPUS_PER_TASK worker
# *processes* (passed through as TSLS_MDE_GMN_CORES below); leaving
# BLAS/OpenMP free to also multithread within each worker oversubscribes
# the node, so pin those to 1 (same reasoning as simple-mde-gmn.sbatch).
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ADJUST: project checkout location and lab/group allocation on this cluster.
PROJ_DIR="$HOME/pao_rotation"
OUT_DIR="$PROJ_DIR/results/job_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# ==============================================================================
# SLURM LOG RELOCATION
# ==============================================================================
# The #SBATCH --output/--error above write to $PROJ_DIR/results/ (flat,
# not $OUT_DIR -- see the ADJUST comment on those directives for why).
# Move them into $OUT_DIR once it exists, via a single combined EXIT trap
# (merged with the /flash tempdir cleanup trap below, since bash only
# honors the most recently registered EXIT trap -- a second `trap ...
# EXIT` later in this script would silently replace this one rather than
# stacking). Runs on every exit path, including failure/timeout/cancel,
# which is exactly when you most want the log sitting next to the rest
# of that run's output instead of loose in results/. Same fix as
# simple-mde-gmn.sbatch.
SLURM_OUT="$PROJ_DIR/results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
SLURM_ERR="$PROJ_DIR/results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
cleanup() {
    [ -n "${tempdir:-}" ] && rm -rf "$tempdir"
    [ -f "$SLURM_OUT" ] && mv -f "$SLURM_OUT" "$OUT_DIR"/
    [ -f "$SLURM_ERR" ] && mv -f "$SLURM_ERR" "$OUT_DIR"/
    return 0
}
trap cleanup EXIT

# ==============================================================================
# JOB DESCRIPTION
# ==============================================================================
# Diffs this run's tsls-mde-gmn.py + tsls-mde-gmn.sbatch against the
# rolling snapshot cache of the last tsls-mde-gmn submission
# ($PROJ_DIR/results/.snapshots/ -- keyed by filename, so this never gets
# compared against a simple-mde-gmn job by mistake; see
# generate_job_description.sh's own module docstring), inserting a new
# entry into $PROJ_DIR/JOB_LOG.md (parameter-block changes as a table,
# everything else as unified diffs), then updates that cache to this
# run's files. Runs before the pipeline itself so the entry is written
# even if the run later fails or times out.
#
# sbatch_src is "$0", not a hardcoded path to this canonical file -- same
# reasoning as simple-mde-gmn.sbatch: sbatch copies whatever file it's
# given verbatim into the running job, and $0 reflects that actual
# submitted path, which matters if the live canonical file at
# scripts/batch/tsls-mde-gmn.sbatch gets edited again while this job sits
# queued.
"$PROJ_DIR/scripts/batch/generate_job_description.sh" \
    "$PROJ_DIR" "$OUT_DIR" "$SLURM_JOB_ID" \
    "$0" \
    "$PROJ_DIR/scripts/src/tsls-mde-gmn.py"

# ==============================================================================
# /flash scratch setup
# ==============================================================================

umask 002

# ADJUST: <GROUP> to this lab's actual flash allocation.
FLASH_BASE="/flash/FukaiU/${USER}"
mkdir -p "${FLASH_BASE}"
chmod 2755 "${FLASH_BASE}"

tempdir="$(mktemp -d "${FLASH_BASE}/run_tsls_mde_gmn.${SLURM_JOB_ID}.XXXXXX")"
# Cleanup on exit (including this tempdir) is handled by the combined
# `cleanup` EXIT trap registered in the SLURM LOG RELOCATION section
# above -- deliberately not a second `trap ... EXIT` here, which would
# silently replace that one instead of stacking.

# NOTE: the E/tau/theta search checkpoints to disk after every candidate
# so a resubmitted job can resume instead of restarting a multi-hour
# search from scratch -- but $tempdir lives under $FLASH_BASE and is
# deleted by the `cleanup` trap above the moment this script exits,
# including on a SLURM timeout/cancellation (SIGTERM), *before* results
# get copied to $OUT_DIR. Same caveat as simple-mde-gmn.sbatch: if your
# search might not finish within --time, checkpoints will be lost across
# resubmissions under this scratch layout.

# ==============================================================================
# COPY SOURCE FILE + HELPER MODULE + INPUT DATA
# ==============================================================================

cp -av \
    "$PROJ_DIR/scripts/src/tsls-mde-gmn.py" \
    "$PROJ_DIR/scripts/src/cleaned_helper_functions.py" \
    "$tempdir"/

mkdir -p "$tempdir/data"
cp -av \
    "$PROJ_DIR/data/${DATA_FILE}" \
    "$tempdir/data/"

cd "$tempdir"

# ==============================================================================
# RUN
# ==============================================================================

export TSLS_MDE_GMN_DATA_DIR="$tempdir/data"
export TSLS_MDE_GMN_DATA_FILE="${DATA_FILE}"
# TEMP DIAGNOSTIC: pointed directly at $OUT_DIR (not $tempdir, which the
# `cleanup` trap deletes on every exit including a crash) so intermediate
# outputs -- especially the CCM interaction-matrix CSV -- survive if this
# run fails the same way jobs 35332887/35347307 did, instead of being lost
# with the rest of $tempdir before anyone can inspect them. Revert to
# "$tempdir/tsls_mde_gmn_outputs" once this is resolved -- writing directly
# under $OUT_DIR (not local /flash scratch) is slower I/O for every
# intermediate file this pipeline writes, acceptable for one diagnostic run
# but not a permanent tradeoff.
export TSLS_MDE_GMN_RESULTS_DIR="$OUT_DIR/live_debug"
export TSLS_MDE_GMN_MODE="${MODE}"
export TSLS_MDE_GMN_GMN_MODE="${GMN_MODE}"
export TSLS_MDE_GMN_TARGETS="${TARGETS}"
export TSLS_MDE_GMN_D="${MDE_D}"
export TSLS_MDE_GMN_EMBED_DIM_RHO_MIN="${EMBED_DIM_RHO_MIN}"
export TSLS_MDE_GMN_CROSS_MAP_RHO_MIN="${CROSS_MAP_RHO_MIN}"
export TSLS_MDE_GMN_REGION_CHANNEL_CAP="${REGION_CHANNEL_CAP}"
export TSLS_MDE_GMN_LIB_RATIO="${LIB_RATIO}"
export TSLS_MDE_GMN_E_MIN="${E_MIN}"
export TSLS_MDE_GMN_E_MAX="${E_MAX}"
export TSLS_MDE_GMN_E_STEP="${E_STEP}"
export TSLS_MDE_GMN_TAU_MIN="${TAU_MIN}"
export TSLS_MDE_GMN_TAU_MAX="${TAU_MAX}"
export TSLS_MDE_GMN_TAU_STEP="${TAU_STEP}"
export TSLS_MDE_GMN_SMAP_SEARCH_MODE="${SMAP_SEARCH_MODE}"
export TSLS_MDE_GMN_CORES="${SLURM_CPUS_PER_TASK}"
export TSLS_MDE_GMN_METRIC="${METRIC}"
export TSLS_MDE_GMN_TARGET_METRIC="${TARGET_METRIC}"
export TSLS_MDE_GMN_GAMMA_DELTA="${GAMMA_DELTA}"
export TSLS_MDE_GMN_CCM_WINDOW_ROWS="${CCM_WINDOW_ROWS}"
export TSLS_MDE_GMN_CCM_LIB_SIZE_PCTS="${CCM_LIB_SIZE_PCTS}"
export TSLS_MDE_GMN_CCM_SAMPLE="${CCM_SAMPLE}"
export TSLS_MDE_GMN_CCM_SLOPE="${CCM_SLOPE}"

python3 -V

python3 -u tsls-mde-gmn.py

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================

# TEMP DIAGNOSTIC: TSLS_MDE_GMN_RESULTS_DIR above already points at
# $OUT_DIR/live_debug (not a relative $tempdir path), so there's nothing
# to copy from $tempdir on success -- flatten live_debug/ into $OUT_DIR
# directly instead, matching this pipeline's normal final layout (files
# directly under $OUT_DIR, not nested). Revert alongside the
# TSLS_MDE_GMN_RESULTS_DIR change above once resolved.
cp -av "$OUT_DIR/live_debug/." "$OUT_DIR"/

echo "========================================================"
echo "Job ${SLURM_JOB_ID} complete. Outputs saved to:"
echo "  ${OUT_DIR}"
echo "========================================================"
