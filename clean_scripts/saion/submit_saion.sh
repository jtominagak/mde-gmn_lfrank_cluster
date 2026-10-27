#!/usr/bin/env bash
# =============================================================================
# submit_saion.sh -- submit a Saion job to whichever of gpu-p100 / gpu-v100
# can start it first
# =============================================================================
# Why this exists: Slurm's own "either partition" syntax
# (`#SBATCH --partition=gpu-p100,gpu-v100`) is REJECTED for this account:
#
#   sbatch: error: Multiple partition job request not supported when a
#   partition is set in the association
#
# (the allocation is attached to each partition separately -- check yours with
# `sacctmgr show assoc user=$USER`). So the scheduler is
# asked instead: `sbatch --test-only -p <partition> <file>` for each candidate
# returns the estimated start time without submitting, and the job goes to the
# earliest one. The .venv-gpu cu126 torch build runs unchanged on both.
#
# Rules of the pick:
#   - --test-only ignores the per-partition allocation (36 CPUs / 4 GPUs). If
#     this job would push your own running+pending jobs on a partition past
#     it, that partition's start is taken as the latest expected end of those
#     jobs instead (and as "unknown", ranked last, if Slurm has no end yet).
#   - Estimates within TIE_S of each other are a tie; ties go to the first
#     entry of CANDIDATES.
#   - The pick happens once, at submission. A pending job does not move if the
#     other partition frees up later -- `scancel` and rerun this to re-pick.
#
# Usage (from a login node, in your home directory):
#   submit_saion.sh <batch/ sbatch basename> [<src/ py basename>]
#     with a .py  -> submitted through batch/submit.sh (writes a JOB_LOG.md entry)
#     without one -> plain sbatch
#   SAION_PARTITION=gpu-p100 submit_saion.sh ...   skip the pick, force one
#   SAION_PICK_ONLY=1 submit_saion.sh ...          print the pick, submit nothing
#
# e.g.  submit_saion.sh saion_parallel_mde.sbatch run_parallel_mde.py
# =============================================================================
set -euo pipefail

# Derived from this script's own location, so there is nothing to keep in sync.
PROJ_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"

# Configurable for another account or site. CANDIDATES is in preference order:
# ties go to the first entry.
read -r -a CANDIDATES <<< "${PAO_GPU_PARTITIONS:-gpu-v100 gpu-p100}"
ALLOC_CPUS="${PAO_ALLOC_CPUS:-36}"      # your per-partition core allocation
ALLOC_GPUS="${PAO_ALLOC_GPUS:-4}"       # your per-partition GPU allocation
TIE_S="${PAO_TIE_SECONDS:-600}"         # estimates within this are a tie

[ "$#" -ge 1 ] || { sed -n '2,38p' "$0"; exit 1; }
SBATCH_NAME="$1"
PY_NAME="${2:-}"
SBATCH_SCRIPT="$PROJ_DIR/batch/$SBATCH_NAME"
[ -f "$SBATCH_SCRIPT" ] || { echo "No such sbatch script: $SBATCH_SCRIPT" >&2; exit 1; }
[ "$PWD" = "$HOME" ] || echo "NOTE: the site guide says to submit from your home directory ($HOME); current dir is $PWD" >&2

# This job's own request, read from its #SBATCH lines.
req_cpus="$(sed -n 's/^#SBATCH[[:space:]]\+--cpus-per-task=\([0-9]\+\).*/\1/p' "$SBATCH_SCRIPT" | head -1)"
req_gpus="$(sed -n 's/^#SBATCH[[:space:]]\+--gres=gpu:\([0-9]\+\).*/\1/p' "$SBATCH_SCRIPT" | head -1)"
req_cpus="${req_cpus:-1}"
req_gpus="${req_gpus:-0}"

pick() {
    local now part out start epoch used_cpus used_gpus over ends end_epoch note
    local best="" best_epoch="" best_over=""
    now="$(date +%s)"
    for part in "${CANDIDATES[@]}"; do
        if ! out="$(sbatch --test-only -p "$part" "$SBATCH_SCRIPT" 2>&1)"; then
            echo "  $part: not usable -- $out" >&2
            continue
        fi
        start="$(sed -n 's/.* to start at \([0-9T:-]\+\) .*/\1/p' <<<"$out" | head -1)"
        if [ -z "$start" ] || ! epoch="$(date -d "$start" +%s 2>/dev/null)"; then
            echo "  $part: could not parse --test-only output: $out" >&2
            continue
        fi
        [ "$epoch" -lt "$now" ] && epoch="$now"

        # Your own running + pending jobs on this partition (CPUs, and GPUs from
        # the gres/tres-per-node field, e.g. "gres/gpu:1" or "gres:gpu:1").
        read -r used_cpus used_gpus < <(
            squeue -h -u "$(id -un)" -p "$part" -t RUNNING,PENDING -o '%C %b' 2>/dev/null \
            | awk '{c+=$1; if (match($2, /gpu(:[a-z0-9]+)?:[0-9]+/)) {n=substr($2,RSTART,RLENGTH); sub(/.*:/,"",n); g+=n}}
                   END {print c+0, g+0}')
        over=0
        note=""
        if [ $((used_cpus + req_cpus)) -gt "$ALLOC_CPUS" ] || [ $((used_gpus + req_gpus)) -gt "$ALLOC_GPUS" ]; then
            # Waits for your own jobs there: start no earlier than the last of
            # them is expected to end (%e; pending jobs may have no estimate).
            ends="$(squeue -h -u "$(id -un)" -p "$part" -t RUNNING,PENDING -o '%e' 2>/dev/null)"
            if grep -qvE '^[0-9]{4}-[0-9]{2}-[0-9]{2}T' <<<"$ends"; then
                over=1
                note=" -- exceeds your allocation until your jobs there end (no end estimate yet)"
            else
                end_epoch="$(while read -r e; do date -d "$e" +%s; done <<<"$ends" | sort -n | tail -1)"
                [ "$end_epoch" -gt "$epoch" ] && epoch="$end_epoch"
                note=" -- exceeds your allocation until your jobs there end, so start >= their end"
            fi
        fi
        printf '  %-9s est. start %s  (your jobs there: %s CPUs, %s GPUs%s)\n' \
            "$part" "$([ "$over" -eq 1 ] && echo unknown || date -d "@$epoch" '+%F %T')" \
            "$used_cpus" "$used_gpus" "$note" >&2

        if [ -z "$best" ] \
            || [ "$over" -lt "$best_over" ] \
            || { [ "$over" -eq "$best_over" ] && [ "$epoch" -lt $((best_epoch - TIE_S)) ]; }; then
            best="$part"; best_epoch="$epoch"; best_over="$over"
        fi
    done
    [ -n "$best" ] || { echo "No candidate partition accepted the job." >&2; return 1; }
    echo "$best"
}

if [ -n "${SAION_PARTITION:-}" ]; then
    partition="$SAION_PARTITION"
    echo "Partition forced by SAION_PARTITION: $partition" >&2
else
    echo "Estimating start time for $SBATCH_NAME (${req_cpus} CPUs, ${req_gpus} GPU) on: ${CANDIDATES[*]}" >&2
    partition="$(pick)"
    echo "Picked: $partition" >&2
fi
[ -n "${SAION_PICK_ONLY:-}" ] && exit 0

# sbatch reads SBATCH_PARTITION from the environment; it overrides the
# file's own #SBATCH --partition line (submit.sh calls plain sbatch).
export SBATCH_PARTITION="$partition"
if [ -n "$PY_NAME" ]; then
    exec "$PROJ_DIR/batch/submit.sh" "$SBATCH_NAME" "$PY_NAME"
else
    job_id="$(sbatch --parsable "$SBATCH_SCRIPT")"
    echo "Submitted batch job $job_id ($SBATCH_NAME) to $partition"
fi
