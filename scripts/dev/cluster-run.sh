#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# cluster-run.sh -- RUN ON YOUR LOCAL MACHINE. Drives the cluster over SSH so
# local development never has to hold the cluster's rules in your head.
#
#   scripts/dev/cluster-run.sh test [<test file>]   srun the test suite
#   scripts/dev/cluster-run.sh submit <name>.sbatch [<name>.py]
#   scripts/dev/cluster-run.sh queue                your jobs
#   scripts/dev/cluster-run.sh tail <jobid>         follow a running job's log
#   scripts/dev/cluster-run.sh pull-log             JOB_LOG.md cluster -> local
#   scripts/dev/cluster-run.sh results <jobid>      fetch archived results
#   scripts/dev/cluster-run.sh shell                interactive login shell
#
# CLUSTER=saion selects the GPU cluster (default: deigo).
# ---------------------------------------------------------------------------
set -euo pipefail

CLUSTER="${CLUSTER:-deigo}"
case "$CLUSTER" in
    deigo) HOST="${DEIGO_HOST:-deigo.oist.jp}"; LOGDIR="/flash/FukaiU/j-tominaga/pao_rotation_results" ;;
    saion) HOST="${SAION_HOST:-saion.oist.jp}"; LOGDIR="/home/j/j-tominaga/saion_slurm_logs" ;;
    *) echo "CLUSTER must be deigo or saion (got: $CLUSTER)" >&2; exit 1 ;;
esac
USER_REMOTE="${REMOTE_USER:-j-tominaga}"
REMOTE_DIR="${REMOTE_DIR:-/home/j/j-tominaga/pao_rotation}"
BUCKET="/bucket/FukaiU/tominaga/pao_rotation"
SSH=(ssh -o ServerAliveInterval=60 "${USER_REMOTE}@${HOST}")
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

cmd="${1:-}"; shift || true

case "$cmd" in
  test)
    # Never on a login node, even for a 2-second test: always via srun.
    target="${1:-tests/test_lagged_leaf_nodes.py}"
    echo "srun on $CLUSTER: $target"
    "${SSH[@]}" "cd '$REMOTE_DIR' && srun -p short -t 0:30:00 -c 4 --mem=8G \
        .venv/bin/python3 '$target'"
    ;;

  submit)
    [ $# -ge 1 ] || { echo "usage: submit <name>.sbatch [<name>.py]" >&2; exit 1; }
    # deigo and saion have different wrappers: submit_saion.sh additionally
    # picks whichever restricted partition can start the job first.
    if [ "$CLUSTER" = "saion" ]; then wrapper="scripts/saion/submit_saion.sh"
    else wrapper="scripts/batch/submit.sh"; fi
    echo "$wrapper $*"
    "${SSH[@]}" "cd '$REMOTE_DIR' && $wrapper $*"
    echo
    echo "Now write the plain-language summary into JOB_LOG.md for this job."
    echo "Pull it first:  CLUSTER=$CLUSTER scripts/dev/cluster-run.sh pull-log"
    ;;

  queue)
    "${SSH[@]}" "squeue -u '$USER_REMOTE' -o '%.10i %.12P %.28j %.2t %.11M %.6D %R'"
    ;;

  tail)
    [ $# -ge 1 ] || { echo "usage: tail <jobid>" >&2; exit 1; }
    "${SSH[@]}" "f=\$(ls -t $LOGDIR/*-$1.out 2>/dev/null | head -1); \
        [ -n \"\$f\" ] || { echo 'no log for job $1 in $LOGDIR' >&2; exit 1; }; \
        echo \"== \$f\"; tail -f \"\$f\""
    ;;

  pull-log)
    # Cluster -> local, one direction only. generate_job_description.sh writes
    # this file on the cluster at job start, so the cluster copy is the newer one.
    echo "JOB_LOG.md: $CLUSTER -> local"
    rsync -a -h --progress -e ssh \
        "${USER_REMOTE}@${HOST}:${REMOTE_DIR}/JOB_LOG.md" "$LOCAL_DIR/JOB_LOG.md"
    ;;

  results)
    [ $# -ge 1 ] || { echo "usage: results <jobid>" >&2; exit 1; }
    # Archived results live on /bucket, which both clusters can read.
    dest="$LOCAL_DIR/results/job_$1"
    mkdir -p "$dest"
    echo "fetching $BUCKET/tsls_mde_gmn_outputs/job_$1/ -> $dest/"
    rsync -a -h --progress -e ssh \
        "${USER_REMOTE}@${HOST}:$BUCKET/tsls_mde_gmn_outputs/job_$1/" "$dest/"
    ;;

  shell)
    exec "${SSH[@]}" -t "cd '$REMOTE_DIR' && exec bash -l"
    ;;

  *)
    sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
