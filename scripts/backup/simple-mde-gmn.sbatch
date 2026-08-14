#!/bin/bash
#SBATCH --job-name=run_simple_mde_gmn
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
#SBATCH --output=/home/j/j-tominaga/pao_rotation/results/%x-%j.out
#SBATCH --error=/home/j/j-tominaga/pao_rotation/results/%x-%j.err
#SBATCH --mail-user=j-tominaga@oist.jp
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ==============================================================================
# MODIFIABLE PARAMETERS
# ==============================================================================
#
# NOTE ON --time/--mem/--cpus-per-task ABOVE: this pipeline now includes a
# data-driven CCM interaction-matrix step and an (E, tau, per-node theta)
# search that each run a full network Generate() per candidate -- both
# can legitimately take hours, not minutes (the old 1-hour budget this
# file used to have was sized for a much simpler fixed-parameter
# pipeline). These are starting points, not maxed-out requests against
# this partition's limits (40-128 cores/node, 500G mem/node, 4-day cap):
#   --cpus-per-task=32: most of this pipeline's parallel steps (dimx.MDE,
#     EmbedDimension, PredictNonlinear, gmn.GMN) scale close to linearly
#     with cores with no nesting issue. The one exception is the CCM
#     interaction-matrix step -- see INTERACTION_MATRIX_OUTER_CORES and
#     KNOWN CAVEAT below for why *that* step doesn't benefit from a
#     larger --cpus-per-task past a point, and why raising it further is
#     mostly wasted allocation rather than wasted time.
#   --mem=48G: this pipeline's actual data (a few hundred channels x a
#     few thousand rows) is a few MB; memory use is driven by per-process
#     overhead (~100-150MB/worker observed) at whatever process count
#     --cpus-per-task and INTERACTION_MATRIX_OUTER_CORES produce, not by
#     data size. 48G has comfortable headroom without requesting enough
#     to slow down scheduling (larger --mem requests generally mean fewer
#     eligible nodes and a longer queue wait).
#   --time=2-00:00:00: a generous but not maxed (4-day cap) starting
#     point. Tighten it once you've timed an actual run of your chosen
#     ranges below -- e.g. via a cheap calibration job with a small
#     REGION_CHANNEL_CAP and a 1x1 E/tau search, timed end-to-end, then
#     scaled up to your real channel count / search grid size.
# See KNOWN CAVEAT further down for why wall time won't scale linearly
# with --cpus-per-task regardless.

# ---- Data ----------------------------------------------------------------
DATA_FILE="J1620210605_epoch_8_freq_1_Hz.csv"   # default: J1620210605_epoch_8_freq_4_Hz.csv

# ---- Overall pipeline shape ------------------------------------------------
# MODE=MDE-GMN: dimx.MDE forward-selects each target's drivers first, then
#   GMN is built from TARGETS + the union of MDE-selected columns.
# MODE=GMN: skip MDE; GMN's interaction matrix is built directly from the
#   full REGION_CHANNEL_CAP-limited observable set (Park et al. 2026
#   Section 2.3's standard construction). Much more expensive -- see
#   KNOWN CAVEAT below.
MODE="GMN"                # default: MDE-GMN. Other value: GMN
# GMN_MODE=combined: one network spanning every TARGETS entry + the union
#   of their drivers. GMN_MODE=separate: one independent network per target.
GMN_MODE="combined"           # default: combined. Other value: separate
TARGETS="head_position_x,head_position_y"   # comma-separated target column names

# ---- MDE (variable selection; unused when MODE=GMN) ------------------------
MDE_D=30                      # default: 20, max # of driver vars MDE can forward-select per target
EMBED_DIM_RHO_MIN=0.2         # default: 0.2, CCM embedding-dimension rho gate
CROSS_MAP_RHO_MIN=0.2         # default: 0.2, simplex cross-map rho gate
# First N channels per region (CA1_/mPFC_/OFC_) offered as MDE candidates,
# or (when MODE=GMN) used directly as the interaction-matrix's full node
# set. 999 is effectively uncapped (matches the full recording). Cap this
# down for a fast test job -- see KNOWN CAVEAT below for why an uncapped
# MODE=GMN run is an O(channels^2) interaction matrix.
REGION_CHANNEL_CAP=999

# ---- GMN Generate() window --------------------------------------------------
# Lib/pred split as a fraction of the full dataset (no more separate
# start-row/duration row counts): LIB_RATIO=0.7 means the first 70% of
# rows are library, the remaining 30% are the held-out pred/Generate()
# window. The entire input file is used -- row counts are derived inside
# the script from this ratio and the loaded data's length.
LIB_RATIO=0.5                  # default: 0.5, fraction of rows used as library

# ---- E/tau/theta search ------------------------------------------------------
# CCM interaction-matrix E/tau and per-node driver counts are data-driven
# (grid-searched / EmbedDimension-estimated inside the script, not fixed
# parameters here anymore). These ranges instead control the GMN node
# generator's own (E, tau[, theta]) search -- every node uses SMap
# uniformly, and this search picks the network-wide E/tau (+ per-node
# theta) that maximizes free-running Generate() fidelity against the real
# held-out targets. E_MIN must stay >= 2 (SMap's own predictor-exclusion
# convention needs E >= 2 even for leaf/undriven nodes).
E_MIN=3      # lower bound (inclusive) of embedding-dimension E search; must stay >= 2
E_MAX=16     # upper bound (EXCLUSIVE -- python range()) of E search; 16 itself is never tried
E_STEP=2     # step between candidate E values, e.g. 3,5,7,...,15 with the defaults above
TAU_MIN=-15  # lower bound (inclusive) of time-delay tau (in samples) search
TAU_MAX=0    # upper bound (EXCLUSIVE -- python range()) of tau search; 0 itself is never tried
TAU_STEP=2   # step between candidate tau values, e.g. -15,-13,...,-1 with the defaults above
# 'coarse_retune' (recommended): re-tune theta per node at each (E, tau)
#   candidate with a coarse 5-value grid, then one full Generate() per
#   candidate -- len(E)*len(tau) full Generate() calls total.
# 'joint': also sweeps the full 15-value theta grid inside the loop --
#   len(E)*len(tau)*15 full Generate() calls. Only use this deliberately;
#   it is roughly 15x the cost of coarse_retune.
SMAP_SEARCH_MODE="coarse_retune" # what the hell is this?

# ---- Parallelism -------------------------------------------------------------
# Kept deliberately small and separate from --cpus-per-task -- see KNOWN
# CAVEAT below for why raising this to match --cpus-per-task backfires
# (it multiplies against, rather than shares, pyEDM.CCM()'s own per-pair
# worker pool). 2-4 is a reasonable range; there's rarely a reason to go
# higher regardless of how many cores you request.
INTERACTION_MATRIX_OUTER_CORES=2

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

module purge
# ADJUST: only needed if the cluster doesn't expose Python 3 in PATH by
# default on compute nodes. If your uv venv bundles its own Python
# interpreter (created with `uv python install` / `uv venv --python ...`)
# you may not need this line at all -- try commenting it out if you get
# "module not found" errors.
module load python/3.11

# ADJUST: absolute path to your uv venv directory.
VENV_DIR="$HOME/pao_rotation/.venv"
source "${VENV_DIR}/bin/activate"

export MPLBACKEND=Agg
# dimx.MDE / apps.InteractionMatrix / pyEDM's EmbedDimension and
# PredictNonlinear already parallelize across SLURM_CPUS_PER_TASK worker
# *processes* (passed through as MDE_GMN_CORES below); leaving BLAS/OpenMP
# free to also multithread within each worker oversubscribes the node, so
# pin those to 1 rather than mirroring SLURM_CPUS_PER_TASK the way a
# single-process job would.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# FIXED CAVEAT (found by reading pyEDM/CCM.py and apps/InteractionMatrix.py
# directly, after observing ~90 live processes from an 8-core request on a
# 10-core test machine, and root-caused after several full jobs each ran
# ~2 days regardless of dataset size, MODE, or MDE_D): apps.InteractionMatrix's
# own outer worker pool (sized by MDE_GMN_INTERACTION_MATRIX_OUTER_CORES
# below) calls pyEDM.CCM() once per channel pair (dimx.MDE similarly calls
# it once per candidate column), and CCM() used to create a *brand-new*
# multiprocessing pool sized to the full visible cpu_count() on *every
# single call* for work that only ever amounts to 4-8 tiny KDTree tasks --
# apps.InteractionMatrix/dimx.MDE never pass CCM's own `parallel=`
# override, so this couldn't be tuned away via their CLIs. The two pools
# didn't coordinate, so peak concurrent processes was (outer cores) x
# cpu_count(), not just cpu_count(); with REGION_CHANNEL_CAP effectively
# uncapped (~470 real channels), that's ~100k+ CCM() calls each paying a
# full spawn/forkserver Pool creation + heavy reimport teardown cost --
# process-spawn overhead dominating actual compute by orders of magnitude,
# which is why cutting DATA_FILE resolution or MDE_D didn't help wall time.
# scripts/src/simple-mde-gmn.py now monkeypatches pyEDM.CCM's inner pool
# away entirely (`_sequential_ccm_workers`) rather than just avoiding the
# multiplication, so this is no longer a live caveat -- kept here for
# provenance. MDE_GMN_INTERACTION_MATRIX_OUTER_CORES (default 2) is still
# a separate knob from --cpus-per-task since it's the one thing sizing
# real parallelism across column pairs now.

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
# of that run's output instead of loose in results/.
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
# Diffs this run's simple-mde-gmn.py + simple-mde-gmn.sbatch against the
# rolling snapshot cache of the last simple-mde-gmn submission
# ($PROJ_DIR/results/.snapshots/ -- not a per-job copy under $OUT_DIR;
# see generate_job_description.sh's own module docstring for why),
# inserting a new entry into $PROJ_DIR/JOB_LOG.md (parameter-block changes
# as a table, everything else as unified diffs), then updates that cache
# to this run's files. Runs before the pipeline itself so the entry is
# written even if the run later fails or times out.
#
# sbatch_src is "$0", not a hardcoded path to this canonical file: sbatch
# copies whatever file it's given verbatim into the running job, and $0
# reflects that actual submitted path (confirmed via `scontrol show job`'s
# Command= field) -- this matters if the live canonical file at
# scripts/batch/simple-mde-gmn.sbatch gets edited again while this job
# sits queued. A hardcoded canonical path here would silently diff
# whatever that file happens to contain *right now* instead of what this
# job actually launched with.
"$PROJ_DIR/scripts/batch/generate_job_description.sh" \
    "$PROJ_DIR" "$OUT_DIR" "$SLURM_JOB_ID" \
    "$0" \
    "$PROJ_DIR/scripts/src/simple-mde-gmn.py"

# ==============================================================================
# /flash scratch setup
# ==============================================================================

umask 002

# ADJUST: <GROUP> to this lab's actual flash allocation.
FLASH_BASE="/flash/FukaiU/${USER}"
mkdir -p "${FLASH_BASE}"
chmod 2755 "${FLASH_BASE}"

tempdir="$(mktemp -d "${FLASH_BASE}/run_simple_mde_gmn.${SLURM_JOB_ID}.XXXXXX")"
# Cleanup on exit (including this tempdir) is handled by the combined
# `cleanup` EXIT trap registered in the SLURM LOG RELOCATION section
# above -- deliberately not a second `trap ... EXIT` here, which would
# silently replace that one instead of stacking.

# NOTE: the E/tau/theta search below checkpoints to disk after every
# candidate so a resubmitted job can resume instead of restarting a
# multi-hour search from scratch (see the script's run_smap_search()) --
# but $tempdir lives under $FLASH_BASE and is deleted by the `trap` above
# the moment this script exits, including on a SLURM timeout/cancellation
# (SIGTERM), *before* results get copied to $OUT_DIR. If your search
# might not finish within --time, checkpoints will be lost across
# resubmissions under this scratch layout -- consider pointing
# MDE_GMN_RESULTS_DIR directly at a subdirectory of $OUT_DIR instead of
# $tempdir/simple_mde_gmn_outputs if that matters for your run (trades
# some I/O speed on /flash for surviving a timeout).

# ==============================================================================
# COPY SOURCE FILE + INPUT DATA
# ==============================================================================

cp -av \
    "$PROJ_DIR/scripts/src/simple-mde-gmn.py" \
    "$tempdir"/

mkdir -p "$tempdir/data"
cp -av \
    "$PROJ_DIR/data/${DATA_FILE}" \
    "$tempdir/data/"

cd "$tempdir"

# ==============================================================================
# RUN
# ==============================================================================

export MDE_GMN_DATA_DIR="$tempdir/data"
export MDE_GMN_DATA_FILE="${DATA_FILE}"
export MDE_GMN_RESULTS_DIR="$tempdir/simple_mde_gmn_outputs"
export MDE_GMN_MODE="${MODE}"
export MDE_GMN_GMN_MODE="${GMN_MODE}"
export MDE_GMN_TARGETS="${TARGETS}"
export MDE_GMN_D="${MDE_D}"
export MDE_GMN_EMBED_DIM_RHO_MIN="${EMBED_DIM_RHO_MIN}"
export MDE_GMN_CROSS_MAP_RHO_MIN="${CROSS_MAP_RHO_MIN}"
export MDE_GMN_REGION_CHANNEL_CAP="${REGION_CHANNEL_CAP}"
export MDE_GMN_LIB_RATIO="${LIB_RATIO}"
export MDE_GMN_E_MIN="${E_MIN}"
export MDE_GMN_E_MAX="${E_MAX}"
export MDE_GMN_E_STEP="${E_STEP}"
export MDE_GMN_TAU_MIN="${TAU_MIN}"
export MDE_GMN_TAU_MAX="${TAU_MAX}"
export MDE_GMN_TAU_STEP="${TAU_STEP}"
export MDE_GMN_SMAP_SEARCH_MODE="${SMAP_SEARCH_MODE}"
export MDE_GMN_CORES="${SLURM_CPUS_PER_TASK}"
export MDE_GMN_INTERACTION_MATRIX_OUTER_CORES="${INTERACTION_MATRIX_OUTER_CORES}"

python3 -V

python3 -u simple-mde-gmn.py

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================

cp -av simple_mde_gmn_outputs/. "$OUT_DIR"/

echo "========================================================"
echo "Job ${SLURM_JOB_ID} complete. Outputs saved to:"
echo "  ${OUT_DIR}"
echo "========================================================"
