# `/bucket` read-only outage — 2026-08-30

A fresh recurrence of the same `/bucket` read-only failure mode documented in
`docs/bucket-outage-2026-08-28.md` (and the earlier 2026-08-23/08-25 incidents in
`JOB_LOG.md`) — unrelated to any code change; multiple real jobs completed
cleanly between then and now (e.g. job 40143104 ran Stage 1/2, theta search,
Generate/Forecast, and R7 successfully before crashing on an unrelated R19 bug).
Same signature, at job startup, before the Python pipeline runs:

```
mkdir: cannot create directory '/bucket/FukaiU/tominaga/pao_rotation/tsls_mde_gmn_cache/job_<id>':
Read-only file system
```

Per the user's own standing instruction from the 2026-08-28 incident ("Do
periodical retry, and keep record of what worked and didn't work for future
reference"), applying the same automated retry loop (15 min spacing, up to 48
attempts, stops on the first attempt that gets past the `mkdir` or fails
differently) without re-asking, since this is exactly the scenario that
instruction covers.

This occurrence surfaced while confirming that `RUN_MDE_EVALUATE=false`
lets the post-window-sizing smoke test reach "Pipeline complete" (job 40143104
crashed inside R19's diffusion-map baseline; see `docs/review-findings.md`'s
"Diffusion-map crash and hang in `run_mde_evaluate()`" section) — purely a
scheduling/infrastructure blocker on that confirmation, not a finding about
the fix itself.

## Attempts

- 2026-08-30T23:02:54+09:00 — manual — job 40385403 on deigo010908 — FAILED (bucket read-only)
- 2026-08-30T23:03:0X+09:00 — manual — job 40385408 on deigo010908 — FAILED (bucket read-only)
- 2026-08-30T23:04:51+09:00 -- attempt 1/48 -- job 40385413 on deigo010931 -- FAILED (bucket read-only, same known issue)
- 2026-08-30T23:20:06+09:00 -- attempt 2/48 -- job 40385481 on deigo010913 -- FAILED (bucket read-only, same known issue)
- 2026-08-30T23:35:21+09:00 -- attempt 3/48 -- job 40385555 on deigo010931 -- FAILED (bucket read-only, same known issue)
- 2026-08-30T23:50:37+09:00 -- attempt 4/48 -- job 40385630 on deigo011224 -- state=RUNNING (not an immediate FAILED) -- looks past the bucket mkdir -- STOPPING retry loop

## Resolution

Resolved after 4 attempts (~48 min, 2026-08-30T23:02:54 to 23:50:37) — job
40385630 got past the `mkdir` and is running normally. Its own `progress.log`
confirms the smoke configuration is genuinely progressing well past R19's
diffusion-map baseline (the step that crashed job 40143104): depth-0 seed,
Stage 1, Stage 2, and theta retuning for both networks all completed with
real per-node timing, matching design. Reached `PIPELINE COMPLETE -- 1089.8s
total` at 2026-08-31T00:09:22 — the smoke-scale acceptance criterion is
confirmed (see `docs/review-findings.md` and `JOB_LOG.md`'s own entry for
this job).

## Recurrence, ~36 minutes after the resolution above

Job 40386298 (the first mid-scale, `REGION_CHANNEL_CAP=15` calibration run,
submitted 2026-08-31T00:26:03) failed at 2026-08-31T00:26:58 with the
identical `mkdir` signature, ~36 minutes after job 40385630 got past it.
Either the mount flapped back to read-only again, or the "resolution" above
was a narrower window than it looked — either way, same known issue, same
established remedy: restarting the automated periodic retry loop (15 min
spacing, up to 48 attempts, stops on the first attempt that gets past the
`mkdir` or fails differently), unchanged from the 2026-08-28/08-30 design.

### Attempts (recurrence)

- 2026-08-31T00:26:58+09:00 — manual (mid-scale run, not the retry script) — job 40386298 on deigo010934 — FAILED (bucket read-only)
- 2026-08-31T00:28:28+09:00 -- attempt 1/48 -- job 40386309 on deigo011106 -- state=RUNNING (not an immediate FAILED) -- looks past the bucket mkdir -- STOPPING retry loop

### Resolution (recurrence)

Resolved on the very first retry (~90s, 2026-08-31T00:26:58 to 00:28:28) --
much faster than the original 08-30 occurrence's 4 attempts/~48 min,
consistent with this being a brief flap rather than a sustained outage.
Job 40386309 (the actual mid-scale `REGION_CHANNEL_CAP=15` calibration run)
is running normally; its own `progress.log` shows `PIPELINE START` logged.
Given this run's own 24-hour walltime budget, completion (or a timeout) is
not expected soon -- tracked in `JOB_LOG.md` and via a `Monitor` on this
job's own `progress.log`, not in this file.

## Second recurrence, ~6h20m after the previous one

Job 40418907 (a second mid-scale run, `tsls-mde-gmn.sbatch`'s config
further hand-tuned since -- `GMN_MODE=combined`, `LAG_STEP=4`, `TP=20`,
`GENERATE_LENGTH=200`, `NESTED_M=3` -- submitted 2026-08-31T07:48:xx while
job 40386309 above is still running) failed at 2026-08-31T07:48:24 with the
identical `mkdir` signature. This is now a recurring pattern (4 distinct
occurrences: 2026-08-28, 2026-08-30, and two on 2026-08-31) rather than a
one-off -- worth escalating to infrastructure/IT if it keeps happening at
this frequency, but the immediate remedy stays the same: restarting the
automated periodic retry loop (15 min spacing, up to 48 attempts).

### Attempts (second recurrence)

- 2026-08-31T07:48:24+09:00 — manual (second mid-scale run, not the retry script) — job 40418907 on deigo010924 — FAILED (bucket read-only)
- 2026-08-31T07:49:02+09:00 -- attempt 1/48 -- job 40419017 on deigo010924 -- FAILED (bucket read-only, same known issue)
- 2026-08-31T08:04:17+09:00 -- attempt 2/48 -- job 40420963 on deigo010903 -- FAILED (bucket read-only, same known issue)
- 2026-08-31T08:19:33+09:00 -- attempt 3/48 -- job 40421505 on deigo011102 -- state=RUNNING (not an immediate FAILED) -- looks past the bucket mkdir -- STOPPING retry loop

### Resolution (second recurrence)

Resolved after 3 attempts (~31 min, 2026-08-31T07:48:24 to 08:19:33). Job
40421505 -- the `LAG_STEP=4` config (down from the first mid-scale run's
`LAG_STEP=1`, specifically to shrink Stage 2's own candidate pool after
the diagnosis in `docs/review-findings.md`'s per-node-observable-count
investigation) -- is running normally; `progress.log` shows `PIPELINE
START`. Job 40386309 (the original `LAG_STEP=1` mid-scale run) is still
running concurrently in parallel, untouched.

## Third recurrence, ~14h after the previous one

Job 40561419 (same mid-scale config as 40421505, re-run to pick up the
R22 Forecast-window-alignment fix and the interactive 3D-network-plot
redesign, `docs/review-findings.md`'s R22) failed at 2026-08-31T21:29:29
with the identical `mkdir: cannot create directory .../tsls_mde_gmn_cache/
job_40561419: Read-only file system` signature (5th distinct occurrence:
2026-08-28, 2026-08-30, two on 2026-08-31 morning, and this one).

### Attempts (third recurrence)

- 2026-08-31T21:29:29+09:00 — manual (medium-scale re-run, not the retry script) — job 40561419 — FAILED (bucket read-only)
- 2026-08-31T21:3x — manual write test (`touch` under `tsls_mde_gmn_cache/`) succeeded a few minutes later, from the login node — misleadingly suggested bucket was already writable again
- 2026-08-31T21:33:12+09:00 — manual (second retry, not the retry script) — job 40561770 — FAILED (bucket read-only, identical signature) — the login-node write test above did not predict this: the read-only state is evidently per-compute-node (or still genuinely flapping), not a simple global up/down, so a login-node touch isn't a reliable predictor of whether a given compute node's own mount will work
- 2026-08-31T21:3x — restarted the same generic periodic retry loop (`retry_midscale.sh`, 15 min spacing, up to 48 attempts) used for the second recurrence, since two manual attempts on two different nodes both hit the same failure

- 2026-08-31T21:34:30+09:00 -- attempt 1/48 -- job 40561813 on deigo010928 -- FAILED (bucket read-only, same known issue)
- 2026-08-31T21:49:45+09:00 -- attempt 2/48 -- job 40563312 on deigo011114 -- state=RUNNING (not an immediate FAILED) -- looks past the bucket mkdir -- STOPPING retry loop

### Resolution (third recurrence)

Resolved after 2 attempts (~20 min, 2026-08-31T21:29:29 to 21:49:45) --
faster than the second recurrence (3 attempts/~31 min), consistent with
these being brief flaps rather than a sustained outage. Job 40563312 (the
medium-scale run carrying the R22 fix, the interactive 3D-network-plot
redesign, and `multi_proper_embed_dimension`) is running normally.

## Fourth recurrence

Job 40577382 (a larger-scale re-run -- `REGION_CHANNEL_CAP` 15->30,
`MDE_D` 4->6, `NESTED_M` 3->4, `MULTI_MDE_E` 3->8, `GENERATE_LENGTH`
200->400, manually adjusted in `tsls-mde-gmn.sbatch`, submitted while
job 40563312 above is still running) failed with the identical `mkdir`
signature (6th distinct occurrence overall). Restarted the same generic
periodic retry loop (`retry_midscale.sh`, 15 min spacing, up to 48
attempts) -- it just resubmits whatever's currently in
`tsls-mde-gmn.sbatch`, so it picks up this larger config automatically,
no script changes needed.

### Attempts (fourth recurrence)

- (manual) job 40577382 -- FAILED (bucket read-only)
- 2026-08-31T22:56:35+09:00 -- attempt 1/48 -- job 40577533 on deigo010917 -- FAILED (bucket read-only, same known issue)
- 2026-08-31T23:11:51+09:00 -- attempt 2/48 -- job 40580126 on deigo010930 -- FAILED (bucket read-only, same known issue)
- 2026-08-31T23:27:07+09:00 -- attempt 3/48 -- job 40582445 on deigo011115 -- state=RUNNING (not an immediate FAILED) -- looks past the bucket mkdir -- STOPPING retry loop

### Resolution (fourth recurrence)

Resolved after 3 attempts (~31 min, 2026-08-31T22:56:35 to 23:27:07),
similar to the second recurrence's own 3-attempt/~31 min resolution. Job
40582445 (the larger-scale `REGION_CHANNEL_CAP=30`/`MDE_D=6`/`NESTED_M=4`
run) is running normally. Job 40563312 (the medium-scale run from the
third recurrence) is still running concurrently, untouched.
- 2026-08-31T22:56:35+09:00 -- attempt 1/48 -- job 40577533 on deigo010917 -- FAILED (bucket read-only, same known issue)
- 2026-08-31T23:11:51+09:00 -- attempt 2/48 -- job 40580126 on deigo010930 -- FAILED (bucket read-only, same known issue)
- 2026-08-31T23:27:07+09:00 -- attempt 3/48 -- job 40582445 on deigo011115 -- state=RUNNING (not an immediate FAILED) -- looks past the bucket mkdir -- STOPPING retry loop

## Fifth recurrence

Job 40664457 (a repeat of job 40563312's own medium-scale config --
`REGION_CHANNEL_CAP=15`/`MDE_D=4`/`NESTED_M=3`/`MULTI_MDE_E=3`/
`GENERATE_LENGTH=200` -- with resources bumped to `--time=48:00:00`/
`--mem=64G`/`--cpus-per-task=32`, submitted 2026-09-01 while jobs 40563312
[complete] and 40582445 [still running] exist) failed with the identical
`mkdir` signature (7th distinct occurrence overall). Restarted the same
generic periodic retry loop.

### Attempts (fifth recurrence)

- (manual) job 40664457 -- FAILED (bucket read-only)

- 2026-09-01T08:13:25+09:00 -- attempt 1/48 -- job 40664609 on deigo010905 -- FAILED (bucket read-only, same known issue)
- 2026-09-01T08:28:41+09:00 -- attempt 2/48 -- job 40665921 on deigo011135 -- state=RUNNING (not an immediate FAILED) -- looks past the bucket mkdir -- STOPPING retry loop

### Resolution (fifth recurrence)

Resolved after 2 attempts (~15 min, 2026-09-01T08:13:25 to 08:28:41). Job
40665921 (the resource-bumped repeat of job 40563312's own medium-scale
config -- `--time=48:00:00`/`--mem=64G`/`--cpus-per-task=32`) is running
normally. Jobs 40563312 (complete) and 40582445 (still running,
large-scale) are unaffected.
