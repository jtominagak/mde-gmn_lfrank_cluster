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
real per-node timing, matching design. A `Monitor` is watching this job's
`progress.log` for `PIPELINE COMPLETE` or a crash/`JOB ENDED` line to confirm
the final, still-pending acceptance criterion (reaches `"Pipeline complete"`
with `RUN_MDE_EVALUATE=false`).
