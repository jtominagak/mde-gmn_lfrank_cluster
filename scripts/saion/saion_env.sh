# shellcheck shell=bash
# =============================================================================
# saion_env.sh -- shared Saion paths + stage-out helpers
# =============================================================================
# SOURCED (not executed) by scripts/batch/saion-*.sbatch. One place for the
# storage rules from https://scda-guide.oist.jp/saion-system.html so the
# preflight job and the pipeline job can never disagree about them:
#
#   /work/FukaiU/j-tominaga  read-write on compute nodes. Job scratch ONLY --
#                 every job creates its own mktemp dir here and deletes it.
#   /bucket/FukaiU/tominaga/pao_rotation  READ-ONLY on compute nodes. Caches
#                 may be read from it directly; anything written back must go
#                 through the login node (the guide's `scp ... saion:/bucket/...`).
#   /flash        does not exist on Saion (only a read-only /deigo_flash).
#   /home/j/j-tominaga  shared with Deigo, slow, 50GB -- code, venv, Slurm logs only.
#
# Stage-out needs passwordless SSH from a compute node to the login node. See
# docs/saion-parallelized-mde-runbook.md step 1 for the one-time key setup;
# scripts/batch/saion-preflight.sbatch verifies it before any real job relies
# on it.
# =============================================================================

# ---- Scoped roots (literal on purpose -- no $USER/$HOME/unit indirection and
# no environment overrides). Every directory this project creates, copies into
# or deletes on Saion must be INSIDE one of these; path_under() below enforces
# it for the destructive/remote operations.
PROJ_DIR="/home/j/j-tominaga/pao_rotation-exp-general"
SLURM_LOG_DIR="/home/j/j-tominaga/saion_slurm_logs"   # must match the #SBATCH -o paths
WORK_BASE="/work/FukaiU/j-tominaga"
# Same bucket tree the Deigo jobs already archive to, so caches written by
# either cluster are visible to both (cache entries are fingerprinted against
# code + installed package versions, so a cross-cluster hit is still exact).
BUCKET_PROJECT="/bucket/FukaiU/tominaga/pao_rotation"
# Input data lives on Bucket and is read from there directly (guide: "Keep
# your data on Bucket ... Read data directly from Bucket"), not from home.
BUCKET_DATA_DIR="/bucket/FukaiU/tominaga/pao_rotation/data"
# `saion` is the ~/.ssh/config alias the runbook sets up (HostName
# saion.oist.jp + the dedicated key). A host name, not a path.
STAGE_HOST="${SAION_STAGE_HOST:-saion}"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new
          -o ConnectTimeout=30 -o ServerAliveInterval=60)

# path_under ROOT PATH -- true only if PATH is strictly inside ROOT (not ROOT
# itself, no `..` components). Guards every mkdir/rm/stage-out below so a
# mis-set variable can never touch /work, /bucket or another user's tree.
path_under() {
    local root="$1" path="$2"
    case "$path" in
        *..*) return 1 ;;
        "$root"/?*) return 0 ;;
        *) return 1 ;;
    esac
}

# safe_rm_work DIR -- rm -rf, but only for a directory inside WORK_BASE.
safe_rm_work() {
    if ! path_under "$WORK_BASE" "$1"; then
        echo "REFUSING to delete '$1': not inside $WORK_BASE" >&2
        return 1
    fi
    rm -rf -- "$1"
}

remote_run() {
    ssh "${SSH_OPTS[@]}" "$STAGE_HOST" "$@"
}

# stage_dir SRC DEST -- copy the CONTENTS of local dir SRC into DEST (a
# /bucket path) through the login node. Returns non-zero on any failure so
# callers can refuse to delete the only copy. rsync when both ends have it
# (incremental -- matters for the hourly cache sync), otherwise a tar stream
# over the same ssh connection (same transport as the guide's scp, but copes
# with dotfiles/empty dirs without glob tricks).
stage_dir() {
    local src="$1" dest="$2"
    [ -d "$src" ] || return 0
    if ! path_under "$BUCKET_PROJECT" "$dest"; then
        echo "REFUSING to stage into '$dest': not inside $BUCKET_PROJECT" >&2
        return 1
    fi
    remote_run "mkdir -p '$dest'" || return 1
    if command -v rsync >/dev/null 2>&1 && remote_run 'command -v rsync' >/dev/null 2>&1; then
        rsync -a -e "ssh ${SSH_OPTS[*]}" "$src"/ "${STAGE_HOST}:${dest}/"
    else
        tar -C "$src" -cf - . | remote_run "tar -C '$dest' -xf -"
    fi
}
