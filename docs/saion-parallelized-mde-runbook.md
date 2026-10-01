# Runbook: parallel-MDE GPU pipeline on Saion (`gpu-p100` / `gpu-v100`)

How to run `scripts/src/no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py` on Saion's
restricted GPU partitions within the cluster rules
(<https://scda-guide.oist.jp/saion-system.html>, `/saion-restrict`). Written 2026-09-16.

| File | Role |
|---|---|
| `scripts/saion/saion_env.sh` | Scoped Saion paths (`/work/FukaiU/j-tominaga`, `/bucket/FukaiU/tominaga/pao_rotation`, `/home/j/j-tominaga/...`), path guards and the stage-out helper, sourced by both job scripts |
| `scripts/saion/setup_gpu_venv.sh` | Completes `.venv-gpu` (gpu-p100 compute node via `srun`) |
| `scripts/saion/submit_saion.sh` | Submits a job to whichever of `gpu-p100` / `gpu-v100` has the earlier estimated start |
| `scripts/batch/saion-preflight.sbatch` | 2 h check job: CUDA, imports, storage, stage-out, GPU tests, one-GPU capacity |
| `scripts/saion/gpu_capacity_probe.py` | Measures how many spawn workers fit on one GPU (run by the preflight) |
| `scripts/batch/saion-no-kedm-optimization-parallelized-mde--tsls-mde-gmn.sbatch` | The pipeline job (half production scale) |

## The rules this follows

- **Allocation:** 36 cores + 4 GPUs *per partition*, 7-day maximum. The pipeline job asks for a
  balanced quarter of that (9 cores + 1 GPU + 120 GB) rather than a whole node's cores alongside a
  single GPU, which would strand the other three. Nodes: 36 cores, 512 GB,
  4 × 16 GB GPUs. P100 nodes support CUDA ≤ 12.8, so `.venv-gpu` uses the cu126 torch build,
  which also runs on V100, so a job can go to either partition.
- **Partition choice:** Slurm's `-p gpu-p100,gpu-v100` is **rejected** for this account ("Multiple
  partition job request not supported when a partition is set in the association"; checked
  2026-09-17). `submit_saion.sh` asks `sbatch --test-only` for each partition's estimated start
  and submits to the earlier one. It also accounts for your own jobs already filling a
  partition's 36-core/4-GPU allocation. The choice is fixed at submission. To re-pick, `scancel`
  the pending job and run it again. `SAION_PARTITION=gpu-v100` forces a partition;
  `SAION_PICK_ONLY=1` only prints the choice.
- **Login node:** submit jobs, move data, install packages. **No computation.**
- **Storage:**
  - `/bucket` is **read-only on compute nodes**. This project only reads and stages into `/bucket/FukaiU/tominaga/pao_rotation`.
  - `/work/FukaiU/j-tominaga` is this project's job scratch space. Each job uses its own subdirectory there and deletes it.
  - `/flash` does not exist on Saion.
  - `/home/j/j-tominaga` is slow and capped at 50 GB: code, venv and Slurm logs only.
- **Scope guard:** every directory the scripts create, delete or stage into must be inside one of
  those three roots. `saion_env.sh` hardcodes them with no environment overrides and refuses any
  `rm`/stage-out outside them. Temp files (`TMPDIR`) also stay inside the job's own directory.
- **Data flow:** job reads the input CSV directly from `/bucket/FukaiU/tominaga/pao_rotation/data/`,
  copies code to `/work/FukaiU/j-tominaga/pmde.<jobid>.XXXX/` → writes results and cache there → `scp`/`rsync` through the `saion` login node into
  `/bucket/FukaiU/tominaga/pao_rotation/` → deletes its `pmde.<jobid>.XXXX` directory.
- **Batch scripts:** start with `#!/bin/bash -l`. Submit from `/home/j/j-tominaga`. Use absolute `-o` paths.

---

## One-time setup (Saion login node)

```bash
ssh j-tominaga@saion.oist.jp
```

### 1. SSH key for compute node → login node stage-out

Jobs copy results into `/bucket/FukaiU/tominaga/pao_rotation` by ssh-ing to the login node, so this must work with no
password prompt. `/home/j/j-tominaga` is shared with Deigo, so the key is visible there too.

Checked from a gpu-p100 node on 2026-09-17: **this step is required**. Compute nodes don't resolve
the bare name `saion`, and host-based auth to `saion.oist.jp` is refused. Compute nodes and the
login nodes share `10.210.16.0/20` (e.g. `saion-gpu11` = 10.210.28.31, `saion.oist.jp` =
10.210.28.18). The `from=` option below makes the passwordless key usable only from that subnet,
and the other options disable forwarding and terminals for it.

```bash
ssh-keygen -t ed25519 -N "" -C "saion compute-node stage-out" -f /home/j/j-tominaga/.ssh/id_ed25519_saion
printf 'from="10.210.16.0/20",no-agent-forwarding,no-port-forwarding,no-X11-forwarding,no-pty %s\n' \
    "$(cat /home/j/j-tominaga/.ssh/id_ed25519_saion.pub)" >> /home/j/j-tominaga/.ssh/authorized_keys
chmod 700 /home/j/j-tominaga/.ssh && chmod 600 /home/j/j-tominaga/.ssh/authorized_keys /home/j/j-tominaga/.ssh/id_ed25519_saion

cat >> /home/j/j-tominaga/.ssh/config <<'EOF'

Host saion
    HostName saion.oist.jp
    User j-tominaga
    IdentityFile /home/j/j-tominaga/.ssh/id_ed25519_saion
    IdentitiesOnly yes
EOF
chmod 600 /home/j/j-tominaga/.ssh/config

# Must print a hostname with no password prompt. The first run records the host key:
ssh -o StrictHostKeyChecking=accept-new saion hostname
ssh -o BatchMode=yes saion hostname
```

### 2. Log and scratch directories

```bash
mkdir -p /home/j/j-tominaga/saion_slurm_logs   # the #SBATCH -o paths point here; must exist before sbatch
mkdir -p /work/FukaiU/j-tominaga && ls -ld /work/FukaiU/j-tominaga
```

If `mkdir` fails, `/work/FukaiU` isn't your unit's directory. Ask SCDA (ask-scda@oist.jp) for the
right one and change `WORK_BASE` in `scripts/saion/saion_env.sh`; it is deliberately a literal.

### 3. Complete the GPU venv (on a gpu-p100 compute node, as the guide requires for builds)

```bash
cd /home/j/j-tominaga
srun -p gpu-p100 -t 0-1 -c 4 --mem=16G bash -l /home/j/j-tominaga/pao_rotation/scripts/saion/setup_gpu_venv.sh
```

Compute nodes do have internet access (PyPI and download.pytorch.org, checked 2026-09-17).
**Done 2026-09-17** on a gpu-p100 node; rerun only if `.venv`'s packages change.

It must end with `OK -- .venv-gpu is complete`. It pins every package to the CPU `.venv`'s
versions, installs `torch==2.14.0+cu126`, and copies the pyEDM Pool-fix `sitecustomize.py`.
Before this, `.venv-gpu` lacked `gmn`, `dimx`, `pydiffmap`, `statsmodels` and the Pool fix, so
the pipeline would have failed to import.

---

## Step 4 — Preflight (every time the venv, code or hardware choice changes)

```bash
cd /home/j/j-tominaga
/home/j/j-tominaga/pao_rotation/scripts/saion/submit_saion.sh saion-preflight.sbatch
squeue -u j-tominaga
```

When it finishes (usually well under 1 h once running):

```bash
cat /home/j/j-tominaga/saion_slurm_logs/preflight_<jobid>/SUMMARY.txt
```

- Any **`FAIL [REQUIRED]`**: fix it before going further. Each check's full output is in
  `/home/j/j-tominaga/saion_slurm_logs/preflight_<jobid>/<check>.log`.
  - *ssh / stage-out:* redo step 1. The bare name `saion` resolves only through the
    `~/.ssh/config` alias, so that block must exist.
  - *equivalence test SKIPPED:* torch can't see the GPU. Check the `torch CUDA` log.
- **GPU capacity verdict.** Already measured, and no longer the binding constraint: preflight
  4724497 on a P100 ran 36 workers with 0 errors at 15,135 MiB of 16,384 (92%, safe cap ~31), and
  9 workers at 3,785 MiB (23%). The job requests 9, so expect `FITS` with large margin.
  - `DOES NOT FIT ... about N workers per GPU` → set `#SBATCH --cpus-per-task=N` in the pipeline
    sbatch and scale `--mem` to about `N × 8G + 20G`.
  - The `slowdown` column is the more interesting number, and it already answered the multi-GPU
    question: per-sweep time scales *exactly* with worker count (1.00× / 9.05× / 18.11× / 36.27×),
    so total throughput is flat at ~58 sweeps/s however many workers share the card. One card is
    saturated by one worker. More CPU workers buy no GPU throughput — only `--gres=gpu:4` plus
    per-worker GPU assignment would, and that code does not exist yet (deferred 2026-09-16).

## Step 5 — Submit the pipeline

Optional: commit first, so the job's `JOB_LOG.md` entry records a clean commit.

```bash
cd /home/j/j-tominaga
/home/j/j-tominaga/pao_rotation/scripts/saion/submit_saion.sh \
    saion-no-kedm-optimization-parallelized-mde--tsls-mde-gmn.sbatch \
    no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py
```

This picks the partition, submits the job through `scripts/batch/submit.sh`, and writes its
`JOB_LOG.md` entry. Ask Claude to add the plain-language
summary to that entry.

Scale ("half production"; the reasoning is in the sbatch's SIZING NOTE):

| Parameter | Production default | This job |
|---|---|---|
| `REGION_CHANNEL_CAP` | 999 (468 channels) | **78** (234 channels) |
| `MDE_D` | 20 | **10** |
| `NESTED_M` | 1 | 1 |
| `GENERATE_LENGTH` | 0 (full window, ~6,700) | **400**, deliberately not halved |
| `PARALLEL_MDE_SCOPE` | union | **staged** — the CPU pipeline's own algorithm without its barriers, so results stay comparable; `union` is now a different algorithm (one combined MDE per node) |
| Resources | – | `gpu-p100` or `gpu-v100` (picked at submission), **9 cores, 1 GPU, 120 G**, 6 days — a quarter-node share, so the other 3 GPUs stay usable by others |

## Step 5b — Submit the staged **and** union jobs as a chain

To run both `PARALLEL_MDE_SCOPE` values one after the other, rather than one scope alone:

```bash
cd /home/j/j-tominaga
git -C pao_rotation status --porcelain      # must print nothing (see below)
SCOPE_PAIR_DRY_RUN=1 pao_rotation/scripts/saion/submit_saion_scope_pair.sh   # check, submit nothing
pao_rotation/scripts/saion/submit_saion_scope_pair.sh                        # submit both
```

Both jobs run the same sbatch file and the same pipeline; only `PARALLEL_MDE_SCOPE` differs
(the sbatch honours an inherited value, so there is no second copy of the file to keep in sync).
The second job carries `--dependency=afterany:<first>`, which is what stops Slurm running them
together — each asks for 9 cores + 1 GPU and the allocation is 36 + 4, so there is room for both.
`afterany` rather than `afterok`: the union job is an independent experiment, and under `afterok`
a first job that merely hit its 6-day limit would leave the second stuck in
`DependencyNeverSatisfied` until cancelled by hand.

Options, all applied to **both** jobs so the two networks stay comparable:

| Variable | Effect |
|---|---|
| `SCOPE_PAIR_DRY_RUN=1` | run every check, print what would be submitted, submit nothing |
| `LAG_OVERRIDE=1 LAG_STEP=4 LAG_COUNT=25` | use `DELAY_SOURCE=tau_count` instead of the sbatch's `lag_mode` — see the sizing warning below |
| `SCOPE_PAIR_ORDER=union,staged` | run union first |
| `SAION_PARTITION=gpu-p100` | skip the start-time estimate and force a partition |

### Two things that will invalidate the comparison

**`union` at the default lag setting may well not finish.** `lag_mode` with `GENERATE_LENGTH=400`
and `LAG_STEP=4` resolves to 100 lags per observable. `union` gives *every* node in the universe a
pool `1 + 100 = 101x` wider than staged's Stage 1 pool, from the first selection round, where
staged pays the wide-pool cost only on its backbone nodes. Deigo job 40684253 (101 lags,
staged-shaped) had single Stage 2 nodes take up to 36.7 h. No union run has ever completed, so
6 days is a guess rather than a budget. `LAG_OVERRIDE=1 LAG_COUNT=25` narrows both jobs together;
a timeout otherwise costs a resubmission, not the work, because the per-node cache syncs hourly.

**Do not touch `scripts/src/` or switch branches until the second job has started.** The sbatch
copies the sources out of the live working tree at *job start*, not at submission — so the second
job, starting up to six days later, would stage whatever is there then. The script refuses to
submit from a dirty tree for this reason, records the commit it expected, and each job warns in
its own output if what it staged does not match.

### Confirming each job really used its own scope

This is worth checking once, because until 2026-10-01 it was not true: the network cache key did
not include `parallel_mde_scope`, so the second job would have matched the first job's cached
network, skipped the whole MDE phase, and reported the other algorithm's result while still
printing its own scope in the config dump. Fixed, and pinned by
`tests/test_network_cache_builder_kwargs.py`. In each job's `.out`:

```bash
grep -h "PARALLEL_MDE_SCOPE\|^LAG_SET:\|matched cached network" \
    /home/j/j-tominaga/saion_slurm_logs/saion_pmde_half-<jobid>.out
```

The union job should say "no matching cached network found ... building from scratch". If it says
"matched cached network", stop and check why — that is the failure above.

### Where each job's results land

Nothing is shared, and nothing is overwritten — every destination is keyed on the job id:

| What | Where |
|---|---|
| Scratch while running | `/work/FukaiU/j-tominaga/pmde.<jobid>.XXXXXX` (`mktemp`, deleted after a successful stage-out) |
| Results on Bucket | `.../tsls_mde_gmn_outputs/job_<jobid>` — or `job_<jobid>_FAIL` if the job failed, so a failure never lands where a success would |
| Cache entries on Bucket | `.../tsls_mde_gmn_cache/job_<jobid>` (also written hourly while running) |
| Slurm logs | `saion_slurm_logs/saion_pmde_half-<jobid>.{out,err}`, and copied into the results directory |
| `JOB_LOG.md` | one `## Job <jobid>` entry each, plus a line naming the scope |

The cache root is *shared for reading* — that is the point of it, so a resubmission resumes from
finished nodes — but each job only ever **writes** into its own `job_<jobid>` subdirectory. That
read-sharing is exactly why the fingerprint has to distinguish the scopes, which is the fix above.

The scope line in `JOB_LOG.md` is written by the pair script rather than by
`generate_job_description.sh`, because that script diffs each job against the previous job from the
same pipeline — and these two jobs use one byte-identical sbatch file, with the scope arriving via
`--export`. Its parameter diff therefore shows no difference between them.

## Step 6 — Monitor (login node; reading only)

```bash
squeue -u j-tominaga
tail -f /home/j/j-tominaga/saion_slurm_logs/saion_pmde_half-<jobid>.out
d=$(ls -d /work/FukaiU/j-tominaga/pmde.<jobid>.*)   # this job's scratch dir
tail -f $d/results/progress.log                   # stage/node timeline, heartbeats, ETAs
tail $d/results/gpu_usage.csv                     # GPU util + memory, every 60 s
sstat -j <jobid>.batch --format=MaxRSS,AveCPU     # live memory high-water mark
```

What to watch for:

- `parallel MDE precompute [...]: N MDE call(s) -- W parallel worker(s)` near the start.
- `MDE call FAILED: ... CUDA` in the `.out` file means GPU OOM: cancel gracefully (below) and
  resize.
- `MaxRSS` approaching 400 G means the next run needs fewer workers.

## Step 7 — When it ends

```bash
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqMem,AllocCPUS
ls /bucket/FukaiU/tominaga/pao_rotation/tsls_mde_gmn_outputs/ | grep <jobid>
cat /bucket/FukaiU/tominaga/pao_rotation/tsls_mde_gmn_outputs/job_<jobid>*/STATUS.txt
```

- `job_<jobid>/` with `STATUS: COMPLETED`: done.
- `job_<jobid>_FAIL/` with `STATUS: FAIL` and a `REASON:`. Partial results, `progress.log`,
  the Slurm logs and the GPU logs are all inside.
- In both cases the MDE cache is in `tsls_mde_gmn_cache/job_<jobid>/`, and the job's
  `/work/FukaiU/j-tominaga/pmde.<jobid>.*` directory was removed.
- If the `.out` file ends with `WARNING: stage-out incomplete`, run the `mkdir`/`rsync`/`rm`
  commands it printed, **on the login node**. That leaves `/work/FukaiU/j-tominaga` clean again.

## Resuming after a timeout or crash

Resubmit exactly as in step 5. With `REUSE_BEST_PARAMS=true`, every MDE node finished by an
earlier attempt is loaded from `/bucket/FukaiU/tominaga/pao_rotation/tsls_mde_gmn_cache` (`REUSE_BEST_PARAMS matched cached MDE selection ...` lines in the
log). Only nodes that were running when the job stopped are recomputed.

**Limitation:** a single Stage 2 node that alone needs more than 7 days can never finish by
resubmitting, because it restarts from zero each time. If `progress.log` shows the same node
in flight across two attempts, the scale has to come down (or that node's cost has to be
investigated).

## Cancelling

```bash
scancel --batch --signal=USR1 <jobid>   # graceful: stops Python, stages everything out, removes its pmde.<jobid>.* dir
scancel <jobid>                         # hard: ~30 s grace; relies on the hourly cache sync
```

After a hard cancel, check `ls /work/FukaiU/j-tominaga/`. Copy anything worth keeping to
`/bucket/FukaiU/tominaga/pao_rotation/` from the login node, then `rm -rf` the leftover `pmde.<jobid>.*` directory.
