#!/usr/bin/env bash
# =============================================================================
# submit_saion_scope_pair.sh -- submit the staged job and the union job as a
# chain: the second starts only after the first has finished
# =============================================================================
# WHAT THIS RUNS. Both jobs are the same sbatch file and the same pipeline
# (scripts/batch/saion-no-kedm-optimization-parallelized-mde--tsls-mde-gmn.sbatch
# -> scripts/src/no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py). They
# differ in one environment variable, PARALLEL_MDE_SCOPE:
#
#   staged  Stage 1 (real observables, recursive) for the whole node universe,
#           then Stage 2 (lagged columns, attached as leaf nodes) for exactly
#           Stage 1's visited set. Builds the same network as the serial CPU
#           pipeline; two barriers.
#   union   ONE MDE per node over real AND lagged candidates together, whole
#           universe, one barrier. A DIFFERENT network -- real and lagged
#           candidates compete inside the same forward selection -- and a much
#           more expensive call per node, since every node's pool is
#           (1 + number of lags)x wider from the first selection round.
#
# WHY A CHAIN AND NOT TWO SUBMISSIONS. Each job asks for 18 cores + 2 GPUs
# (raised from 9 + 1 on 2026-10-05, see the sbatch's own header) and the
# allocation is 36 cores + 4 GPUs per partition, so the two together are
# EXACTLY the allocation and Slurm would happily run both at once. `--dependency=afterany:<first>` is what holds the second back.
# afterany, not afterok: the union job is an independent experiment, and with
# afterok a first job that merely hit its 6-day limit would leave the second
# stuck in DependencyNeverSatisfied until cancelled by hand.
#
# Usage, from the Saion LOGIN node, in /home/j/j-tominaga:
#   pao_rotation/scripts/saion/submit_saion_scope_pair.sh
#
# Optional environment overrides (both jobs get the same values, so the two
# networks stay comparable):
#   LAG_OVERRIDE=1            use DELAY_SOURCE=tau_count with the LAG_* values
#                             below instead of the sbatch's own lag_mode
#                             setting. See the SIZING WARNING printed at the
#                             end of a dry run.
#   LAG_STEP=8 LAG_COUNT=25   the tau and the number of lags, with LAG_OVERRIDE
#   SCOPE_PAIR_ORDER=union,staged   run union first instead
#   SCOPE_PAIR_DRY_RUN=1      run every check, print what would be submitted,
#                             submit nothing
#   SAION_PARTITION=gpu-p100  skip the start-time estimate, force a partition
# =============================================================================
set -euo pipefail

PROJ_DIR="/home/j/j-tominaga/pao_rotation-exp-general"
BRANCH_REQUIRED="no-kEDM-gpu-optimization"
SBATCH_NAME="saion-no-kedm-optimization-parallelized-mde--tsls-mde-gmn.sbatch"
PY_NAME="no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py"
SBATCH_SCRIPT="$PROJ_DIR/scripts/batch/$SBATCH_NAME"
SLURM_LOG_DIR="/home/j/j-tominaga/saion_slurm_logs"

IFS=',' read -r SCOPE_FIRST SCOPE_SECOND <<<"${SCOPE_PAIR_ORDER:-staged,union}"
DRY_RUN="${SCOPE_PAIR_DRY_RUN:-}"

die() { echo "ERROR: $*" >&2; exit 1; }
ok()  { printf '  ok    %s\n' "$*"; }

echo "=== Preflight checks (nothing is submitted until all of them pass) ==="

# 1. Login node, not a compute node, and not Deigo.
case "$(hostname -s)" in
    saion*) ok "on a Saion node ($(hostname -s))" ;;
    *) die "hostname is $(hostname -s), which is not a Saion node. ssh j-tominaga@saion.oist.jp first." ;;
esac
[ -z "${SLURM_JOB_ID:-}" ] || die "this is running inside Slurm job ${SLURM_JOB_ID}; submit from the login node shell."

# 2. The SCDA guide wants submissions from the home directory, and
#    submit_saion.sh warns about it otherwise.
[ "$PWD" = "/home/j/j-tominaga" ] || die "run this from /home/j/j-tominaga (currently $PWD)."

# 3. The right branch, because the sbatch copies scripts/src/ out of the live
#    working tree at JOB START -- the parallel entry script only exists here.
branch="$(git -C "$PROJ_DIR" rev-parse --abbrev-ref HEAD)"
[ "$branch" = "$BRANCH_REQUIRED" ] \
    || die "branch is '$branch'; need '$BRANCH_REQUIRED' (git -C $PROJ_DIR checkout $BRANCH_REQUIRED)."
ok "branch $branch"
[ -f "$PROJ_DIR/scripts/src/$PY_NAME" ] || die "missing $PROJ_DIR/scripts/src/$PY_NAME"
[ -f "$SBATCH_SCRIPT" ] || die "missing $SBATCH_SCRIPT"
ok "pipeline and sbatch present"

# The checked-out code has to be new enough for this workflow, and BOTH of the
# things below fail SILENTLY if it is not -- which is why they are checked here
# rather than left to be noticed in the results.
#
#   1. An sbatch that assigns PARALLEL_MDE_SCOPE outright ignores --export, so
#      both jobs would run the same scope while the submission log claims
#      otherwise.
#   2. Without builder_kwargs in the network fingerprint, the second job matches
#      the first job's cached network, skips the whole MDE phase and reports the
#      other algorithm's result -- again while printing its own scope.
# Anchored, and matching the ASSIGNMENT only: the unanchored substring also
# occurs in `export TSLS_MDE_GMN_PARALLEL_MDE_SCOPE="${PARALLEL_MDE_SCOPE}"`
# further down, which is present in the old file too -- so an unanchored grep
# passes on exactly the version this is meant to reject.
grep -qE '^PARALLEL_MDE_SCOPE="\$\{PARALLEL_MDE_SCOPE' "$SBATCH_SCRIPT" \
    || die "$SBATCH_NAME assigns PARALLEL_MDE_SCOPE outright instead of honouring an inherited value, so --export would be ignored and BOTH jobs would run the same scope. This working tree predates commit fdb848d."
grep -qF 'builder_kwargs' "$PROJ_DIR/scripts/src/mde_gmn/caching.py" \
    || die "mde_gmn/caching.py does not put the builder's options in the network fingerprint, so the second job would reuse the first job's cached network and never run its own algorithm. This working tree predates commit 69849ec."
ok "sbatch honours PARALLEL_MDE_SCOPE; cache fingerprint records it"

# 4. The sources the jobs stage must not move under them. The second job copies
#    scripts/src/ out of this directory at ITS OWN start, days from now, so an
#    edit in between silently makes the two halves of the comparison different
#    code. That is a hard failure.
#
#    Scoped to scripts/src/ rather than the whole tree deliberately. The sbatch
#    file is snapshotted by sbatch at submission, so later edits to it cannot
#    reach a queued job; and JOB_LOG.md is written by THIS script, so a
#    repo-wide clean requirement would block every resubmission after the first
#    one -- a check that fails on its own side effects trains you to bypass it.
if [ -n "$(git -C "$PROJ_DIR" status --porcelain -- scripts/src)" ]; then
    git -C "$PROJ_DIR" status --short -- scripts/src >&2
    die "scripts/src/ has uncommitted changes. Commit or stash them -- job 2 stages that directory when it starts, not now, so the two jobs would otherwise run different code."
fi
commit="$(git -C "$PROJ_DIR" rev-parse HEAD)"
ok "scripts/src/ clean at ${commit:0:12}"
if [ -n "$(git -C "$PROJ_DIR" status --porcelain)" ]; then
    echo "  note  other files are modified; they are not staged by the jobs, so this is not blocking:"
    git -C "$PROJ_DIR" status --short | sed 's/^/          /'
fi

# 5. The stage-out key (runbook step 1). Without it a job computes for days and
#    then cannot write its results to /bucket.
ssh -o BatchMode=yes -o ConnectTimeout=30 saion hostname >/dev/null 2>&1 \
    || die "passwordless ssh to 'saion' failed -- do runbook step 1 (docs/saion-parallelized-mde-runbook.md) first. Without it both jobs lose their results at stage-out."
ok "passwordless stage-out ssh works"

# 6. Directories the #SBATCH -o/-e lines and the job scratch need.
[ -d "$SLURM_LOG_DIR" ] || die "missing $SLURM_LOG_DIR (mkdir -p it; the #SBATCH -o path needs it to exist)."
[ -d /work/FukaiU/j-tominaga ] || die "missing /work/FukaiU/j-tominaga (job scratch)."
ok "log and scratch directories exist"

# 7. The GPU venv.
[ -x "$PROJ_DIR/.venv-gpu/bin/python3" ] || die "missing $PROJ_DIR/.venv-gpu -- runbook step 3."
ok ".venv-gpu present"

echo
echo "=== What will be submitted ==="
scope_env() {
    local scope="$1"
    local -a pairs=("PARALLEL_MDE_SCOPE=$scope")
    if [ -n "${LAG_OVERRIDE:-}" ]; then
        pairs+=("DELAY_SOURCE=tau_count" "LAG_STEP=${LAG_STEP:-8}" "LAG_COUNT=${LAG_COUNT:-25}")
    fi
    printf '%s\n' "${pairs[*]}"
}
printf '  job 1  scope=%-7s %s\n' "$SCOPE_FIRST" "$(scope_env "$SCOPE_FIRST")"
printf '  job 2  scope=%-7s %s   (--dependency=afterany:<job 1>)\n' "$SCOPE_SECOND" "$(scope_env "$SCOPE_SECOND")"
printf '  commit %s\n' "$commit"
if [ -z "${LAG_OVERRIDE:-}" ]; then
    cat <<'WARN'

  SIZING WARNING -- read before submitting a union job at these settings.
  The sbatch leaves the lag set on DELAY_SOURCE=lag_mode, which with
  GENERATE_LENGTH=400 and LAG_STEP=8 resolves to 50 lags per observable.
  'union' gives every node a pool 51x wider than staged's Stage 1 pool, from
  the first selection round, for the whole node universe -- where staged pays
  that cost only on its backbone nodes. Measured reference: Deigo job 40684253
  (101 lags, staged-shaped) had single Stage 2 nodes take up to 36.7 h. No
  union run has ever completed, so 6 days is a guess, not a budget.
  To make both jobs narrower and still comparable, resubmit with e.g.
      LAG_OVERRIDE=1 LAG_STEP=8 LAG_COUNT=25 <this script>
  Either way the per-node cache is synced hourly, so a timeout costs a
  resubmission rather than the work.
WARN
fi

if [ -n "$DRY_RUN" ]; then
    echo
    echo "SCOPE_PAIR_DRY_RUN set -- stopping here, nothing submitted."
    exit 0
fi

# Partition: ask the same question submit_saion.sh asks, once, and use the
# answer for both jobs. A pick made now is meaningless for a job that starts in
# six days, so this is about the first job; the second inherits it so the two
# run on comparable hardware (P100 and V100 differ in speed).
if [ -n "${SAION_PARTITION:-}" ]; then
    partition="$SAION_PARTITION"
    echo "Partition forced by SAION_PARTITION: $partition"
else
    partition="$(SAION_PICK_ONLY=1 "$PROJ_DIR/scripts/saion/submit_saion.sh" "$SBATCH_NAME" "$PY_NAME" 2>&1 \
                 | sed -n 's/^Picked: //p' | tail -1)"
    [ -n "$partition" ] || die "could not determine a partition; rerun with SAION_PARTITION=gpu-p100 or gpu-v100."
    echo "Partition picked for both jobs: $partition"
fi

submit_one() {
    local scope="$1" dependency="${2:-}"
    local -a args=(--parsable --partition="$partition")
    [ -n "$dependency" ] && args+=(--dependency="afterany:$dependency")
    local export_list="ALL,PARALLEL_MDE_SCOPE=$scope,SCOPE_PAIR_COMMIT=$commit"
    if [ -n "${LAG_OVERRIDE:-}" ]; then
        export_list="$export_list,DELAY_SOURCE=tau_count,LAG_STEP=${LAG_STEP:-8},LAG_COUNT=${LAG_COUNT:-25}"
    fi
    args+=(--export="$export_list")
    sbatch "${args[@]}" "$SBATCH_SCRIPT"
}

echo
first_id="$(submit_one "$SCOPE_FIRST")"
echo "Submitted job $first_id  scope=$SCOPE_FIRST  partition=$partition"
second_id="$(submit_one "$SCOPE_SECOND" "$first_id")"
echo "Submitted job $second_id  scope=$SCOPE_SECOND  partition=$partition  starts after $first_id finishes"

# JOB_LOG.md entries, the same ones submit.sh would have written -- plus a line
# naming the scope. That line is not decoration: generate_job_description.sh
# diffs each job against the previous job from the same pipeline, and these two
# share one byte-identical sbatch file (the scope arrives via --export, not the
# file), so the generated entry would show NO difference between them. Without
# this, JOB_LOG.md could not tell you which job ran which algorithm.
record_scope_note() {
    local id="$1" scope="$2" partner="$3" note tmp
    note="**Scope-pair note:** ran \`PARALLEL_MDE_SCOPE=$scope\`"
    if [ -n "${LAG_OVERRIDE:-}" ]; then
        note="$note with \`DELAY_SOURCE=tau_count LAG_STEP=${LAG_STEP:-8} LAG_COUNT=${LAG_COUNT:-25}\`"
    else
        note="$note with the sbatch's own lag settings"
    fi
    note="$note. Submitted by submit_saion_scope_pair.sh at commit ${commit:0:12},"
    note="$note paired with job $partner. The sbatch file is identical for both jobs,"
    note="$note so the parameter diff below does not show the scope."
    tmp="$(mktemp)"
    awk -v heading="## Job $id" -v note="$note" \
        '{print} $0 == heading {print ""; print note}' \
        "$PROJ_DIR/JOB_LOG.md" > "$tmp" && mv "$tmp" "$PROJ_DIR/JOB_LOG.md"
}

for pair in "$first_id:$SCOPE_FIRST:$second_id" "$second_id:$SCOPE_SECOND:$first_id"; do
    IFS=':' read -r id scope partner <<<"$pair"
    # Both jobs are already queued by this point, so nothing here may abort the
    # script under `set -e` -- the summary below is how you learn the job ids.
    if "$PROJ_DIR/scripts/batch/generate_job_description.sh" \
            "$PROJ_DIR" "$id" "$SBATCH_SCRIPT" "$PROJ_DIR/scripts/src/$PY_NAME"; then
        record_scope_note "$id" "$scope" "$partner" \
            || echo "WARNING: could not add the scope note for $id to JOB_LOG.md." >&2
    else
        echo "WARNING: JOB_LOG.md entry for $id failed; the job is still queued." >&2
    fi
done

cat <<EOF

=== Submitted ===
  $first_id   scope=$SCOPE_FIRST
  $second_id   scope=$SCOPE_SECOND   (afterany:$first_id)

The scope each job actually used is recorded in its own output, which is the
only place to confirm it -- grep for it once each starts:
  grep -h "PARALLEL_MDE_SCOPE\|^LAG_SET:" $SLURM_LOG_DIR/saion_pmde_half-{$first_id,$second_id}.out

Watch:        squeue -u \$(id -un) -o '%.10i %.9P %.8T %.10M %.12L %R'
Cancel both:  scancel $first_id $second_id
Cancel job 2 only (keep job 1 running):  scancel $second_id

DO NOT edit $PROJ_DIR/scripts/src or switch branches until job $second_id has
started: it copies its sources out of that directory when it starts, not now,
so an edit in between makes the two halves of this comparison different code.
Each job logs the commit it staged ("commit=..."); $commit is what both should
report.
EOF
