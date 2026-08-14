#!/bin/bash
#SBATCH --job-name=run_simple_mde_gmn
#SBATCH --partition=compute
#SBATCH --time=00-01:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

set -euo pipefail

# ==============================================================================
# MODIFIABLE PARAMETERS
# ==============================================================================

# ---- Data ----------------------------------------------------------------
DATA_FILE="J1620210605_epoch_8_freq_4_Hz.csv"   # default: J1620210605_epoch_8_freq_4_Hz.csv
TARGET="head_position_x"                        # default: head_position_x

# ---- MDE (variable selection) ---------------------------------------------
MDE_D=20                      # default: 20, forward-selection dimension ceiling
EMBED_DIM_RHO_MIN=0.1         # default: 0.1, CCM embedding-dimension rho gate
CROSS_MAP_RHO_MIN=0.1         # default: 0.1, simplex cross-map rho gate

# ---- Interaction matrix / network (GMN) ------------------------------------
INTERACTION_E=5                # default: 5, CCM embedding dimension for InteractionMatrix
TAU=-3                         # default: -3, embedding delay shared by InteractionMatrix and GMN
NUM_DRIVERS=4                  # default: 4, apps.CreateNetwork -d (numDrivers per node)

# ---- GMN Generate() window --------------------------------------------------
PREDICTION_START=3500          # default: 3500, row index Generate() starts forecasting from
PREDICTION_LENGTH=300          # default: 300, number of rows to generate

# ==============================================================================
# ENVIRONMENT SETUP
# ==============================================================================

module purge
# ADJUST: module name for conda/miniforge on this cluster.
module load miniconda3

CONDA_ENV_NAME="mde-gmn-lfrank"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export MPLBACKEND=Agg
# dimx.MDE / apps.InteractionMatrix already parallelize across
# SLURM_CPUS_PER_TASK worker *processes* (passed through as MDE_GMN_CORES
# below); leaving BLAS/OpenMP free to also multithread within each worker
# oversubscribes the node, so pin those to 1 rather than mirroring
# SLURM_CPUS_PER_TASK the way a single-process job would.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ADJUST: project checkout location and lab/group allocation on this cluster.
PROJ_DIR="$HOME/mde-gmn_lfrank"
OUT_DIR="$PROJ_DIR/results/job_${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"

# ==============================================================================
# /flash scratch setup
# ==============================================================================

umask 002

# ADJUST: <GROUP> to this lab's actual flash allocation.
FLASH_BASE="/flash/<GROUP>/${USER}"
mkdir -p "${FLASH_BASE}"
chmod 2755 "${FLASH_BASE}"

tempdir="$(mktemp -d "${FLASH_BASE}/run_simple_mde_gmn.${SLURM_JOB_ID}.XXXXXX")"
trap 'rm -rf "$tempdir"' EXIT

# ==============================================================================
# COPY SOURCE FILE + INPUT DATA
# ==============================================================================

cp -av \
    "$PROJ_DIR/src/mde_gmn_lfrank/simple-mde-gmn.py" \
    "$tempdir"/

mkdir -p "$tempdir/data"
cp -av \
    "$PROJ_DIR/data/transfer_35162_files_7d441a8f/${DATA_FILE}" \
    "$tempdir/data/"

cd "$tempdir"

# ==============================================================================
# RUN
# ==============================================================================

export MDE_GMN_DATA_DIR="$tempdir/data"
export MDE_GMN_DATA_FILE="${DATA_FILE}"
export MDE_GMN_RESULTS_DIR="$tempdir/simple_mde_gmn_outputs"
export MDE_GMN_TARGET="${TARGET}"
export MDE_GMN_D="${MDE_D}"
export MDE_GMN_EMBED_DIM_RHO_MIN="${EMBED_DIM_RHO_MIN}"
export MDE_GMN_CROSS_MAP_RHO_MIN="${CROSS_MAP_RHO_MIN}"
export MDE_GMN_INTERACTION_E="${INTERACTION_E}"
export MDE_GMN_TAU="${TAU}"
export MDE_GMN_NUM_DRIVERS="${NUM_DRIVERS}"
export MDE_GMN_PREDICTION_START="${PREDICTION_START}"
export MDE_GMN_PREDICTION_LENGTH="${PREDICTION_LENGTH}"
export MDE_GMN_CORES="${SLURM_CPUS_PER_TASK}"

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
