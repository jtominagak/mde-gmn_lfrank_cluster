#!/usr/bin/env bash
# Submits a pipeline's .sbatch script (default: simple-mde-gmn.sbatch;
# pass another scripts/batch/*.sbatch basename as $1 -- e.g.
# `submit.sh tsls-mde-gmn.sbatch` -- to submit a different pipeline) and
# immediately generates that job's entry in JOB_LOG.md (parameter table +
# code/design diffs against the most recent earlier job from the SAME
# pipeline -- generate_job_description.sh keys that on the source
# filenames, so different pipelines' jobs never get diffed against each
# other), instead of waiting for the job to reach the front of the queue
# to find out what it's about to run. The corresponding .py source is
# derived from the sbatch basename (scripts/src/<name>.py), matching the
# naming convention every pipeline in this repo follows.
#
# Intended to be invoked by Claude Code on the user's behalf: after this
# script prints the log path, Claude reads the diff and adds a
# "### Plain-language summary" section to this job's entry in JOB_LOG.md
# explaining the parameter/code changes in context --
# generate_job_description.sh is idempotent, so that addition survives
# the job's own startup call to the same script untouched. Manual
# `sbatch <name>.sbatch` still works fine on its own; it just skips the
# prose summary, since nothing is around to write it.
set -euo pipefail

PROJ_DIR="$HOME/pao_rotation"
SBATCH_NAME="${1:-simple-mde-gmn.sbatch}"
SBATCH_SCRIPT="$PROJ_DIR/scripts/batch/$SBATCH_NAME"
PY_SRC="$PROJ_DIR/scripts/src/${SBATCH_NAME%.sbatch}.py"

job_id="$(sbatch --parsable "$SBATCH_SCRIPT")"
echo "Submitted batch job $job_id ($SBATCH_NAME)"

out_dir="$PROJ_DIR/results/job_${job_id}"
mkdir -p "$out_dir"

"$PROJ_DIR/scripts/batch/generate_job_description.sh" \
    "$PROJ_DIR" "$out_dir" "$job_id" \
    "$SBATCH_SCRIPT" \
    "$PY_SRC"

echo "Job log: $PROJ_DIR/JOB_LOG.md (entry: ## Job ${job_id})"
