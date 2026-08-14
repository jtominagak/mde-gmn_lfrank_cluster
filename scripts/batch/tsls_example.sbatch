#!/bin/bash
#SBATCH --job-name=run_tsls_example
#SBATCH --partition=compute
#SBATCH --time=04:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --mail-user=j-tominaga@oist.jp
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# ==============================================================================
# MODIFIABLE PARAMETERS
# ==============================================================================
#
# NOTE ON --time/--mem/--cpus-per-task ABOVE: unlike simple-mde-gmn.sbatch,
# this pipeline's cleaned_helper_functions.py has no multiprocessing of its
# own (confirmed by grepping it for "multiprocessing"/"Pool(") and its
# KDTree nearest-neighbor queries never pass scipy's `workers=` kwarg
# (confirmed by grepping for ".query("), so they default to workers=1 --
# single-threaded, start to finish. --cpus-per-task=4 here is just modest
# headroom for pandas/OS overhead, not a real parallelism budget; raising
# it will not speed this job up. What DOES cost time: WINDOW_ROWS (linear)
# and SCAN_POOL_SIZE (quadratic -- Section 1's pairwise correlation scan is
# O(SCAN_POOL_SIZE^2) proper_simplex calls). The defaults below match the
# notebook's own demo-sized values (WINDOW_ROWS=6000, SCAN_POOL_SIZE=28);
# --time=4h has comfortable headroom over that but will need raising for a
# substantially wider window/pool -- time a small increase first rather
# than guessing.

# ---- Data ----------------------------------------------------------------
DATA_FILE="J1620210605_epoch_8_freq_40_Hz.csv"   # default: 40 Hz file (TSLS is for true spike trains)
WINDOW_ROWS=6000                                  # default: 6000 (~150s at 40Hz)

# ---- Candidate neuron pool (Section 1) ------------------------------------
OCC_MIN=0.015                 # default: 0.015, min per-bin firing occupancy to be a scan candidate
OCC_MAX=0.10                  # default: 0.10, max per-bin firing occupancy to be a scan candidate
SCAN_POOL_SIZE=28             # default: 28 neurons -- see cost note above (O(n^2) pairwise scan)
NUM_WORKING_NEURONS=6         # default: 6, size of the working set built from the pairwise scan's top pairs

# ---- Simplex/CCM demo pair (Sections 2-3) ---------------------------------
DEMO_E=3                      # default: 3
DEMO_TAU=-1                   # default: -1
DEMO_TP=1                     # default: 1
CCM_SAMPLE=20                 # default: 20, CCM subsamples per library size
CCM_LIB_SIZE_PCTS="10,30,50,70,90"   # default: percentiles of the working set's row count
CCM_SEED=0                    # default: 0

# ---- MDE target/candidate discovery (Section 4) ---------------------------
MDE_TARGET_POOL_SIZE=8        # default: 8, candidate MDE targets drawn from the pairwise scan
MDE_DRIVERS_PER_TARGET=5      # default: 5, top-correlated partners considered per candidate target
PROBE_LIB_SIZE_PCTS="20,50,80"       # default: percentiles used by the cheap t2c-convergence probe
PROBE_SAMPLE=8                # default: 8, CCM subsamples used by the same probe

# ---- proper_mde runs (Section 4) ------------------------------------------
MDE_D=3                              # default: 3, forward-selection dimension ceiling
MDE_RANK_ONLY_CROSS_MAP_RHO_MIN=0.3  # default: 0.3, noCCM=True run's rho gate
MDE_FULL_SAMPLE=15                   # default: 15, noCCM=False run's CCM subsamples
MDE_FULL_P_LIB_SIZES="20,40,60,80"   # default: noCCM=False run's percentile library sizes
MDE_FULL_CROSS_MAP_RHO_MIN=0.3       # default: 0.3, noCCM=False run's simplex cross-map rho gate
MDE_FULL_EMBED_DIM_RHO_MIN=0.3       # default: 0.3, noCCM=False run's CCM embedding-dimension rho gate

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

module purge
# ADJUST: only needed if the cluster doesn't expose Python 3 in PATH by
# default on compute nodes. If your uv venv bundles its own Python
# interpreter you may not need this line at all.
module load python/3.11

# ADJUST: absolute path to your uv venv directory.
VENV_DIR="$HOME/pao_rotation/.venv"
source "${VENV_DIR}/bin/activate"

export MPLBACKEND=Agg
# Unlike simple-mde-gmn.sbatch, this pipeline has no outer worker pool to
# protect from oversubscription (see cost note above -- everything here is
# single-threaded), so BLAS/OpenMP threading is left at its default rather
# than pinned to 1.

# ADJUST: project checkout location and lab/group allocation on this cluster.
PROJ_DIR="$HOME/pao_rotation"
OUT_DIR="$PROJ_DIR/results/job_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# ==============================================================================
# JOB DESCRIPTION
# ==============================================================================
# Snapshots this run's tsls_example.py + tsls_example.sbatch into $OUT_DIR
# and diffs them against the most recent earlier tsls_example job's
# snapshot, writing $OUT_DIR/JOB_DESCRIPTION.md. Runs before the pipeline
# itself so the description is written even if the run later fails.
"$PROJ_DIR/scripts/batch/generate_job_description.sh" \
    "$PROJ_DIR" "$OUT_DIR" "$SLURM_JOB_ID" \
    "$PROJ_DIR/scripts/batch/tsls_example.sbatch" \
    "$PROJ_DIR/scripts/src/tsls_example.py"

# ==============================================================================
# /flash scratch setup
# ==============================================================================

umask 002

# ADJUST: <GROUP> to this lab's actual flash allocation.
FLASH_BASE="/flash/FukaiU/${USER}"
mkdir -p "${FLASH_BASE}"
chmod 2755 "${FLASH_BASE}"

tempdir="$(mktemp -d "${FLASH_BASE}/run_tsls_example.${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf "$tempdir"' EXIT

# ==============================================================================
# COPY SOURCE FILE + HELPER MODULE + INPUT DATA
# ==============================================================================

cp -av \
    "$PROJ_DIR/scripts/src/tsls_example.py" \
    "$PROJ_DIR/external/TSLS/cleaned_helper_functions.py" \
    "$tempdir"/

mkdir -p "$tempdir/data"
cp -av \
    "$PROJ_DIR/data/${DATA_FILE}" \
    "$tempdir/data/"

cd "$tempdir"

# ==============================================================================
# RUN
# ==============================================================================

export TSLS_EXAMPLE_DATA_DIR="$tempdir/data"
export TSLS_EXAMPLE_DATA_FILE="${DATA_FILE}"
export TSLS_EXAMPLE_RESULTS_DIR="$tempdir/tsls_example_outputs"
export TSLS_EXAMPLE_WINDOW_ROWS="${WINDOW_ROWS}"
export TSLS_EXAMPLE_OCC_MIN="${OCC_MIN}"
export TSLS_EXAMPLE_OCC_MAX="${OCC_MAX}"
export TSLS_EXAMPLE_SCAN_POOL_SIZE="${SCAN_POOL_SIZE}"
export TSLS_EXAMPLE_NUM_WORKING_NEURONS="${NUM_WORKING_NEURONS}"
export TSLS_EXAMPLE_DEMO_E="${DEMO_E}"
export TSLS_EXAMPLE_DEMO_TAU="${DEMO_TAU}"
export TSLS_EXAMPLE_DEMO_TP="${DEMO_TP}"
export TSLS_EXAMPLE_CCM_SAMPLE="${CCM_SAMPLE}"
export TSLS_EXAMPLE_CCM_LIB_SIZE_PCTS="${CCM_LIB_SIZE_PCTS}"
export TSLS_EXAMPLE_CCM_SEED="${CCM_SEED}"
export TSLS_EXAMPLE_MDE_TARGET_POOL_SIZE="${MDE_TARGET_POOL_SIZE}"
export TSLS_EXAMPLE_MDE_DRIVERS_PER_TARGET="${MDE_DRIVERS_PER_TARGET}"
export TSLS_EXAMPLE_PROBE_LIB_SIZE_PCTS="${PROBE_LIB_SIZE_PCTS}"
export TSLS_EXAMPLE_PROBE_SAMPLE="${PROBE_SAMPLE}"
export TSLS_EXAMPLE_MDE_D="${MDE_D}"
export TSLS_EXAMPLE_MDE_RANK_ONLY_CROSS_MAP_RHO_MIN="${MDE_RANK_ONLY_CROSS_MAP_RHO_MIN}"
export TSLS_EXAMPLE_MDE_FULL_SAMPLE="${MDE_FULL_SAMPLE}"
export TSLS_EXAMPLE_MDE_FULL_P_LIB_SIZES="${MDE_FULL_P_LIB_SIZES}"
export TSLS_EXAMPLE_MDE_FULL_CROSS_MAP_RHO_MIN="${MDE_FULL_CROSS_MAP_RHO_MIN}"
export TSLS_EXAMPLE_MDE_FULL_EMBED_DIM_RHO_MIN="${MDE_FULL_EMBED_DIM_RHO_MIN}"

python3 -V

python3 -u tsls_example.py

# ==============================================================================
# SAVE OUTPUTS
# ==============================================================================

cp -av tsls_example_outputs/. "$OUT_DIR"/

echo "========================================================"
echo "Job ${SLURM_JOB_ID} complete. Outputs saved to:"
echo "  ${OUT_DIR}"
echo "========================================================"
