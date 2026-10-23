#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# push-to-cluster.sh -- RUN ON YOUR LOCAL MACHINE.
#
# Pushes source to the cluster checkout so a job can be submitted against it.
# Code only: data/, results/ and the venvs stay where they are, and the
# authoritative copies of the data live on /bucket, not here.
#
#   scripts/dev/push-to-cluster.sh              # dry run
#   scripts/dev/push-to-cluster.sh --go         # push
#   CLUSTER=saion scripts/dev/push-to-cluster.sh --go
#
# JOB_LOG.md IS NEVER PUSHED. generate_job_description.sh writes it ON THE
# CLUSTER at job start, so the cluster copy is ahead of yours whenever a job
# has run. Pull it instead:  scripts/dev/cluster-run.sh pull-log
# ---------------------------------------------------------------------------
set -euo pipefail

CLUSTER="${CLUSTER:-deigo}"
case "$CLUSTER" in
    deigo) HOST="${DEIGO_HOST:-deigo.oist.jp}" ;;
    saion) HOST="${SAION_HOST:-saion.oist.jp}" ;;
    *) echo "CLUSTER must be deigo or saion (got: $CLUSTER)" >&2; exit 1 ;;
esac
USER_REMOTE="${REMOTE_USER:-j-tominaga}"
REMOTE_DIR="${REMOTE_DIR:-/home/j/j-tominaga/pao_rotation}"

# Repo root = two levels up from this script.
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

FLAGS=(--dry-run); LABEL="DRY RUN"
[ "${1:-}" = "--go" ] && { FLAGS=(); LABEL="PUSHING"; }

echo "=========================================================="
echo "$LABEL  $LOCAL_DIR  ->  ${CLUSTER}:${REMOTE_DIR}"
echo "=========================================================="

# --delete is scoped to these code subtrees only, so a file you deleted
# locally also disappears on the cluster -- but nothing outside them is ever
# considered. data/, results/ and the venvs are not in this list at all.
for sub in scripts tests docs; do
    [ -d "$LOCAL_DIR/$sub" ] || continue
    echo "--- $sub/"
    rsync -a -h --delete "${FLAGS[@]}" \
        --exclude '__pycache__/' --exclude '*.py[cod]' \
        --exclude '_scratch_*/' \
        -e ssh \
        "$LOCAL_DIR/$sub/" "${USER_REMOTE}@${HOST}:${REMOTE_DIR}/$sub/"
done

echo "--- top-level files"
rsync -a -h "${FLAGS[@]}" -e ssh \
    --include 'CLAUDE.md' --include 'README.md' --include 'pyproject.toml' \
    --include 'uv.lock' --include '.python-version' --include 'main.py' \
    --exclude '*' \
    "$LOCAL_DIR/" "${USER_REMOTE}@${HOST}:${REMOTE_DIR}/"

echo
if [ "${1:-}" = "--go" ]; then
    echo "Pushed. Remember: sbatch stages scripts/src/*.py AND scripts/src/mde_gmn/"
    echo "at job start -- do not submit while a rename is half-finished."
else
    echo "Nothing was written. Re-run with --go."
fi
