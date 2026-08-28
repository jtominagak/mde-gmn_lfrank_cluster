# `/bucket` read-only outage — 2026-08-28

Tracking record for the `/bucket` read-only outage encountered while submitting the
post-window-sizing smoke test (`scripts/batch/tsls-mde-gmn.sbatch`, `REGION_CHANNEL_CAP=3
/MDE_D=2/NESTED_M=2`). Every attempt fails identically, at job startup, before the Python
pipeline itself ever runs:

```
mkdir: cannot create directory '/bucket/FukaiU/tominaga/pao_rotation/tsls_mde_gmn_cache/job_<id>':
Read-only file system
```

Matches the same failure mode documented in `JOB_LOG.md`'s 2026-08-23 outage and its
2026-08-25 partial recurrence (job_39655804/39656077) — not a bug in this session's own
pipeline changes. See that history for the established (and, so far, only known) resolution:
resubmit until scheduling lands a job on a healthy node.

An automated retry loop (`retry_smoke_test.sh`, launched in the background) resubmits every
15 minutes, up to 48 attempts (~12 hours), and appends one line per attempt below. It stops
automatically on the first attempt that gets past the `mkdir` (success or a *different*
failure worth investigating on its own) or after exhausting its attempt budget.

## Attempts

- 2026-08-28T21:50:57+09:00 — manual — job 40142455 on deigo010901 — FAILED (bucket read-only)
- 2026-08-28T21:53:28+09:00 — manual — job 40142907 on deigo010919 — FAILED (bucket read-only)
- 2026-08-28T22:00:21+09:00 — manual — job 40142937 on deigo010933 — FAILED (bucket read-only)
- 2026-08-28T22:05:26+09:00 -- attempt 1/48 -- job 40143041 on deigo010933 -- FAILED (bucket read-only, same known issue)
- 2026-08-28T22:20:41+09:00 -- attempt 2/48 -- job 40143067 on deigo011222 -- state=RUNNING (not an immediate FAILED) -- looks past the bucket mkdir -- STOPPING retry loop

## Resolution

**Resolved after 2 automated retries** (2/48 attempts used, ~30 minutes total). Job 40143067
is genuinely running on `deigo011222` — confirmed via real output already present under
`/flash/FukaiU/j-tominaga/pao_rotation_results/job_40143067/` (network pickles, MDE stage
1/2 output, theta-search configs), well past the point the `/bucket` mkdir used to fail at.
No node exclusion or other intervention was needed beyond what this file's own retry loop
already did automatically. For future reference: **2 retries at ~15 min spacing was enough
this time** — considerably shorter than the 2026-08-23 outage (4+ hours) or its 2026-08-25
partial recurrence, so this looks like a brief, localized blip rather than a repeat of
either past incident.

Note: job 40143067 started (and copied its own snapshot of `tsls-mde-gmn.py`) before this
session's progress-monitoring-log feature was finished, so it will not have a `progress.log`
of its own — that lands starting with the next fresh submission.
