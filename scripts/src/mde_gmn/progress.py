"""Progress/heartbeat logging for long pipeline stages.

A job can spend hours inside one MDE call, so every stage appends a dated
line to RESULTS_DIR/progress.log. That file is the only way to tell "still
working" from "stuck" while a job runs, and the sbatch script appends the
job's final outcome to it as well.

Depends on: nothing in this package (stdlib only).
Imported by: mde, network, network_parallel, theta, generate, pipeline.
"""

import os
from datetime import datetime


HEARTBEAT_SECONDS = 300

def _log_progress(log_path, message):
    '''Appends one timestamped line to the progress-monitoring log (see
    load_config()'s own PROGRESS_LOG_PATH -- flash, not bucket, so it's
    readable while a job is still running, and survives independently of
    the bucket-archival step). Every write is flushed and fsynced
    immediately -- this is a diagnostic log for a pipeline with a real
    history of runs that never finish (see JOB_LOG.md), so a line must
    survive a crash/OOM/timeout the instant after it's written, not just
    on a clean close. `log_path=None` is a silent no-op (callers that
    don't have one, e.g. tests, never need to guard every call site with
    an `if`).

    Only ever called from the main process -- Stage 1/Stage 2 worker
    processes (run inside ProcessPoolExecutor) are never given this path,
    so there is no concurrent-writer question to solve here (see this
    module's own docstring section on the monitoring log for why timing
    is measured from the orchestrating process's own submit/as_completed
    calls instead of from inside proper_mde itself).'''
    if log_path is None:
        return
    with open(log_path, 'a') as f:
        f.write(f'{datetime.now().isoformat()} {message}\n')
        f.flush()
        os.fsync(f.fileno())

def _format_eta(durations, n_remaining, per_unit=None, unit_remaining=None):
    '''Rough remaining-time estimate for a monitoring-log heartbeat/summary
    line, from durations already observed *in this same run* (never across
    jobs -- see load_config()'s own PROGRESS_LOG_PATH docstring for why).

    Two modes:
    - Flat per-task average: pass `durations` (list of completed task
      durations, seconds) and `n_remaining` (how many more of that same
      task type are left) -- estimate is `mean(durations) * n_remaining`.
    - Per-window-row rate (used for Stage 1/Stage 2 nodes, whose own cost
      scales roughly with resolved window size -- see
      _resolve_target_ranking_window()): pass `per_unit` (list of
      observed seconds-per-window-row, one entry per completed node) and
      `unit_remaining` (total window rows still queued across pending
      nodes) instead of/alongside durations -- estimate is
      `mean(per_unit) * unit_remaining`. Falls back to the flat estimate
      if `per_unit` has no observations yet (e.g. the very first node in
      a wave).

    Returns '' (not an estimate) when there is nothing completed yet to
    base one on -- an ETA fabricated from zero observations would be
    actively misleading, not just uninformative.'''
    if per_unit and unit_remaining is not None and unit_remaining > 0:
        rate = sum(per_unit) / len(per_unit)
        return (f' (~{rate * unit_remaining:.0f}s remaining, based on {len(per_unit)} '
                f'completed node(s) averaging {rate * 1000:.2f}ms/window-row)')
    if durations and n_remaining and n_remaining > 0:
        mean_s = sum(durations) / len(durations)
        return (f' (~{mean_s * n_remaining:.0f}s remaining, based on {len(durations)} '
                f'completed averaging {mean_s:.1f}s each)')
    return ''
