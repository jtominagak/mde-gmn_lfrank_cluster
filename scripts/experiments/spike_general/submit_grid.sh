#!/usr/bin/env bash
# Submit Stage II (full pipeline + autonomous evaluation) jobs for
# exp/spike-conversion-general, one per line of a grid file, and record
# label / job id / commit in docs/experiments/spike-conversion-general/jobs.tsv.
#
# Usage (Saion login node):
#   scripts/experiments/spike_general/submit_grid.sh <grid.tsv> [label ...]
# grid.tsv: one setting per line, "<label><TAB>VAR=value VAR=value ...";
# '#' lines are comments. With labels given, only those lines are submitted.
# SAION_PARTITION is passed through to submit_saion.sh (e.g. intel); unset,
# submit_saion.sh picks gpu-v100 / gpu-p100.
set -euo pipefail
P=/home/j/j-tominaga/pao_rotation-exp-general
GRID="$1"; shift
WANT=" $* "
LOG="$P/docs/experiments/spike-conversion-general/jobs.tsv"
[ -f "$LOG" ] || printf 'label\tjob_id\tcommit\tsubmitted_at\tsettings\n' > "$LOG"
commit="$(git -C "$P" rev-parse --short=12 HEAD)"
if [ -n "$(git -C "$P" status --porcelain -- scripts/src scripts/batch/saion-spike-general.sbatch)" ]; then
    echo "Refusing to submit: uncommitted changes under scripts/src or the sbatch -- commit first," >&2
    echo "so the commit recorded for each job is the code it runs." >&2
    exit 1
fi
cd /home/j/j-tominaga
while IFS=$'\t' read -r label settings; do
    [ -z "$label" ] && continue
    case "$label" in \#*) continue ;; esac
    if [ "$WANT" != "  " ] && [[ "$WANT" != *" $label "* ]]; then continue; fi
    out="$(env ABLATION_LABEL="$label" $settings \
        "$P/scripts/saion/submit_saion.sh" saion-spike-general.sbatch \
        no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py 2>&1)" || { echo "$out"; exit 1; }
    job="$(grep -oE 'Submitted batch job [0-9]+' <<<"$out" | grep -oE '[0-9]+' | head -1)"
    echo "$label -> job ${job:-?}"
    printf '%s\t%s\t%s\t%s\t%s\n' "$label" "${job:-?}" "$commit" "$(date -Iseconds)" "$settings" >> "$LOG"
done < "$GRID"
