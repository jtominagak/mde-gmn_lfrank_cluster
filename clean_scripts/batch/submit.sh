#!/usr/bin/env bash
# =============================================================================
# submit.sh -- submit a pipeline and write its JOB_LOG.md entry immediately
# =============================================================================
# Usage, from anywhere:
#     submit.sh <sbatch basename> [<src .py basename>]
#
#     submit.sh saion_parallel_mde.sbatch run_parallel_mde.py
#     submit.sh serial_mde.sbatch         run_serial_mde.py
#
# The .py argument is optional and defaults to <sbatch basename>.py, which is
# the convention for an sbatch named after its pipeline. The batch scripts here
# are named for the CLUSTER SHAPE rather than the pipeline, so both of them
# need it stated.
#
# WHY THIS EXISTS. generate_job_description.sh writes a JOB_LOG.md entry
# comparing this job's parameters and code against the most recent earlier job
# from the SAME pipeline. Running it at submission time rather than at job
# start means you can read what a job is about to do while it is still queued,
# instead of waiting for it to reach the front. The job also calls that script
# itself on startup, and it is idempotent, so the two do not conflict and
# anything you add to the entry in between survives.
#
# Plain `sbatch <file>` still works; it just skips the log entry.
# =============================================================================
set -euo pipefail

PROJ_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"

[ "$#" -ge 1 ] || { sed -n '2,25p' "$0"; exit 1; }
SBATCH_NAME="$1"
PY_NAME="${2:-${SBATCH_NAME%.sbatch}.py}"

SBATCH_SCRIPT="$PROJ_DIR/batch/$SBATCH_NAME"
PY_SRC="$PROJ_DIR/src/$PY_NAME"
[ -f "$SBATCH_SCRIPT" ] || { echo "No such sbatch script: $SBATCH_SCRIPT" >&2; exit 1; }
[ -f "$PY_SRC" ] || {
    echo "No such pipeline source: $PY_SRC (pass its basename as the 2nd argument)" >&2
    exit 1
}

job_id="$(sbatch --parsable "$SBATCH_SCRIPT")"
echo "Submitted batch job $job_id ($SBATCH_NAME)"

"$PROJ_DIR/batch/generate_job_description.sh" \
    "$PROJ_DIR" "$job_id" "$SBATCH_SCRIPT" "$PY_SRC"

echo "Job log: $(dirname "$PROJ_DIR")/JOB_LOG.md (entry: ## Job ${job_id})"
