#!/usr/bin/env bash
# submit_grid.sh STAGES MAX_IN_FLIGHT [LABEL ...]
#
# Submits the ablation grid (grid.py) one job per configuration through the
# project's own Saion submitter, never more than MAX_IN_FLIGHT of this user's
# spike_abl jobs queued or running at once -- Slurm's GrpSubmitJobs=20 is
# account-wide and shared with the lab. Waits (sleeping on the login node, no
# computation) for a slot before each submission. Labels already in the
# manifest are skipped, so rerunning is safe. Optional LABELs restrict the
# run to those entries.
#
# Manifest: docs/experiments/spike-conversion-neuro/jobs.tsv
#   label  job_id  commit  submitted_at
set -euo pipefail
EXP=/home/j/j-tominaga/pao_rotation-exp-neuro
STAGES="${1:?stages, e.g. I or I,II}"
MAX="${2:?max in flight}"
shift 2
ONLY=" $* "
MANIFEST="$EXP/docs/experiments/spike-conversion-neuro/jobs.tsv"
[ -f "$MANIFEST" ] || printf 'label\tjob_id\tcommit\tsubmitted_at\n' > "$MANIFEST"
PY=/home/j/j-tominaga/pao_rotation/.venv/bin/python3

[ "$(git -C "$EXP" rev-parse --abbrev-ref HEAD)" = exp/spike-conversion-neuro ] \
    || { echo "worktree is not on exp/spike-conversion-neuro" >&2; exit 1; }
if [ -n "$(git -C "$EXP" status --porcelain scripts/src scripts/batch scripts/saion)" ]; then
    echo "uncommitted pipeline changes in $EXP -- commit first so each job's commit= is exact" >&2
    exit 1
fi
commit="$(git -C "$EXP" rev-parse --short=12 HEAD)"

cd /home/j/j-tominaga
"$PY" -I "$EXP/scripts/experiments/spike_conversion/grid.py" "$STAGES" | while read -r label assigns; do
    if [ "$ONLY" != "  " ] && [[ "$ONLY" != *" $label "* ]]; then continue; fi
    if cut -f1 "$MANIFEST" | grep -qx "$label"; then
        echo "skip $label (already in manifest)"; continue
    fi
    while [ "$(squeue -h -u "$(id -un)" -n spike_abl -t RUNNING,PENDING | wc -l)" -ge "$MAX" ]; do
        sleep 300
    done
    out="$(env ABLATION_LABEL="$label" $assigns \
        "$EXP/scripts/saion/submit_saion.sh" saion-spike-ablation.sbatch \
        no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py 2>&1)"
    echo "$out" | tail -3
    job="$(sed -n 's/^Submitted batch job \([0-9]\+\).*/\1/p' <<<"$out" | head -1)"
    [ -n "$job" ] || { echo "submission of $label failed" >&2; exit 1; }
    printf '%s\t%s\t%s\t%s\n' "$label" "$job" "$commit" "$(date -Iseconds)" >> "$MANIFEST"
    sleep 5
done
