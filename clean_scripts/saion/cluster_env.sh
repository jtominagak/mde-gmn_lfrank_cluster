# shellcheck shell=bash
# =============================================================================
# cluster_env.sh -- shared cluster paths and stage-out helpers
# =============================================================================
# SOURCED, not executed, by the sbatch scripts in ../batch/. One place for the
# storage rules so every job agrees about them.
#
# THE STORAGE RULES this encodes (OIST Saion; adjust for another site):
#   $WORK_BASE      read-write on compute nodes. Job scratch ONLY -- every job
#                   makes its own mktemp dir here and deletes it at the end.
#   $BUCKET_PROJECT READ-ONLY on compute nodes. Caches may be read directly;
#                   anything written back goes through the login node.
#   $PROJ_DIR       home storage: slow, small quota. Code, venv and Slurm logs
#                   only -- never run output.
#
# Stage-out needs passwordless SSH from a compute node to the login node; see
# ../docs/deployment.md for the one-time key setup.
#
# -----------------------------------------------------------------------------
# CONFIGURING THIS FOR YOUR ACCOUNT
# -----------------------------------------------------------------------------
# Every path below is an environment variable with a default, so nothing here
# has to be edited to move this to another account: export the ones that differ
# before submitting, or edit the defaults once.
#
#   PAO_WORK_BASE        job scratch root            (default: /work/FukaiU/$USER)
#   PAO_BUCKET_PROJECT   durable project root        (default: the one below)
#   PAO_SLURM_LOG_DIR    where Slurm .out/.err go    (default: ~/saion_slurm_logs)
#   PAO_STAGE_HOST       ssh alias for the login node (default: saion)
#
# The #SBATCH --output/--error/--mail-user lines in the batch scripts CANNOT
# read these: Slurm parses its directives before any shell runs, so no variable
# expansion happens there. Those few literals are grouped at the top of each
# sbatch under an EDIT FOR YOUR ACCOUNT banner, and PAO_SLURM_LOG_DIR has to
# agree with what they say.
# =============================================================================

# PROJ_DIR is derived from this file's own location rather than configured --
# it is by definition the checkout this script is part of, so there is nothing
# for a reader to get wrong and nothing to keep in sync.
PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

WORK_BASE="${PAO_WORK_BASE:-/work/FukaiU/$USER}"
BUCKET_PROJECT="${PAO_BUCKET_PROJECT:-/bucket/FukaiU/tominaga/pao_rotation}"
SLURM_LOG_DIR="${PAO_SLURM_LOG_DIR:-$HOME/saion_slurm_logs}"

# Input data is read straight off the durable store, never copied to home.
BUCKET_DATA_DIR="${PAO_BUCKET_DATA_DIR:-$BUCKET_PROJECT/data}"

# An ssh alias (see ~/.ssh/config), not a path.
STAGE_HOST="${PAO_STAGE_HOST:-saion}"
SSH_OPTS=(-o BatchMode=yes -o StrictHostKeyChecking=accept-new
          -o ConnectTimeout=30 -o ServerAliveInterval=60)

# -----------------------------------------------------------------------------
# Root validation
# -----------------------------------------------------------------------------
# These roots used to be hardcoded literals, deliberately, because path_under()
# below is what keeps a mis-set variable from reaching outside the intended
# tree with an rm -rf. Making them configurable does not weaken that -- the
# guard validates against whatever the roots ARE, so a wrongly-set WORK_BASE
# can still only ever delete inside itself -- but it does open one new failure
# mode the literals could not have: a root that is empty, relative, or "/".
# So each is checked once, here, at source time.
for _root_name in WORK_BASE BUCKET_PROJECT PROJ_DIR; do
    _root_value="${!_root_name}"
    case "$_root_value" in
        ''|/|//) echo "FATAL: $_root_name is ${_root_value:-empty} -- refusing to run" >&2
                 return 1 2>/dev/null || exit 2 ;;
        /*)      ;;
        *)       echo "FATAL: $_root_name must be an absolute path, got '$_root_value'" >&2
                 return 1 2>/dev/null || exit 2 ;;
    esac
done
unset _root_name _root_value

# path_under ROOT PATH -- true only if PATH is strictly inside ROOT (not ROOT
# itself, no '..' components). Guards every mkdir/rm/stage-out below.
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

# stage_dir SRC DEST -- copy the CONTENTS of local dir SRC into DEST (a path
# under $BUCKET_PROJECT) through the login node. Returns non-zero on any
# failure, so callers can refuse to delete the only copy. rsync when both ends
# have it (incremental, which matters for the hourly cache sync), otherwise a
# tar stream over the same ssh connection.
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
