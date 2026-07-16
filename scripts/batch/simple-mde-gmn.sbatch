#!/bin/bash
#SBATCH --job-name=run_simple_mde_gmn
#SBATCH --partition=compute
#SBATCH --time=2-00:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=32
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

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
DATA_FILE="J1620210605_epoch_8_freq_40_Hz.csv"   # default: J1620210605_epoch_8_freq_4_Hz.csv

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
MDE_D=20                      # default: 20, forward-selection dimension ceiling
EMBED_DIM_RHO_MIN=0.2         # default: 0.2, CCM embedding-dimension rho gate
CROSS_MAP_RHO_MIN=0.2         # default: 0.2, simplex cross-map rho gate
# First N channels per region (CA1_/mPFC_/OFC_) offered as MDE candidates,
# or (when MODE=GMN) used directly as the interaction-matrix's full node
# set. 999 is effectively uncapped (matches the full recording). Cap this
# down for a fast test job -- see KNOWN CAVEAT below for why an uncapped
# MODE=GMN run is an O(channels^2) interaction matrix.
REGION_CHANNEL_CAP=999

# ---- GMN Generate() window --------------------------------------------------
PREDICTION_START=3500          # default: 3500, library size (rows) / Generate() start row
PREDICTION_LENGTH=300          # default: 300, held-out rows to generate

# ---- E/tau/theta search ------------------------------------------------------
# CCM interaction-matrix E/tau and per-node driver counts are data-driven
# (grid-searched / EmbedDimension-estimated inside the script, not fixed
# parameters here anymore). These ranges instead control the GMN node
# generator's own (E, tau[, theta]) search -- every node uses SMap
# uniformly, and this search picks the network-wide E/tau (+ per-node
# theta) that maximizes free-running Generate() fidelity against the real
# held-out targets. E_MIN must stay >= 2 (SMap's own predictor-exclusion
# convention needs E >= 2 even for leaf/undriven nodes).
E_MIN=3 # what the hell is this?
E_MAX=16
E_STEP=2
TAU_MIN=-15
TAU_MAX=0
TAU_STEP=2
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

# KNOWN CAVEAT (found by reading pyEDM/CCM.py and apps/InteractionMatrix.py
# directly, after observing ~90 live processes from an 8-core request on a
# 10-core test machine): apps.InteractionMatrix's own outer worker pool
# (sized by MDE_GMN_INTERACTION_MATRIX_OUTER_CORES below) calls
# pyEDM.CCM() once per channel pair, and CCM() creates a *brand-new*
# multiprocessing pool sized to the full visible cpu_count() on *every
# single call* -- apps.InteractionMatrix never passes CCM's own
# `parallel=` override, so this can't be tuned away via its CLI. The two
# pools don't coordinate, so peak concurrent processes is
# (outer cores) x cpu_count(), not just cpu_count() -- setting the outer
# count equal to --cpus-per-task therefore *squares* your core request
# into a process count. The script keeps this outer count decoupled and
# small (MDE_GMN_INTERACTION_MATRIX_OUTER_CORES, default 2) specifically
# to avoid that multiplication; don't raise it to match --cpus-per-task.
# This is not a hang (confirmed via per-process CPU-time deltas: real work
# keeps accumulating), but every column pair still pays a fresh pool
# creation/teardown cost regardless of core counts -- so wall time for the
# interaction-matrix step won't scale linearly with --cpus-per-task either
# way. If a job runs far longer than --time allows, check actual CPU
# utilization (e.g. `srun --jobid=<id> top`) before assuming the pipeline
# itself is broken.

# ADJUST: project checkout location and lab/group allocation on this cluster.
PROJ_DIR="$HOME/pao_rotation"
OUT_DIR="$PROJ_DIR/results/job_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# ==============================================================================
# /flash scratch setup
# ==============================================================================

umask 002

# ADJUST: <GROUP> to this lab's actual flash allocation.
FLASH_BASE="/flash/FukaiU/${USER}"
mkdir -p "${FLASH_BASE}"
chmod 2755 "${FLASH_BASE}"

tempdir="$(mktemp -d "${FLASH_BASE}/run_simple_mde_gmn.${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf "$tempdir"' EXIT

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
export MDE_GMN_PREDICTION_START="${PREDICTION_START}"
export MDE_GMN_PREDICTION_LENGTH="${PREDICTION_LENGTH}"
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
