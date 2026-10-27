# Running this on a cluster

How a job gets from a checkout to a result, and what to do when it does not.
Parameter values and their evidence are in [parameters.md](parameters.md).

---

## 1. One-time setup

### Storage

Three roots, with different rules. They are defined in
`saion/cluster_env.sh`, each overridable by environment variable.

| Root | Default | Rule |
|---|---|---|
| `WORK_BASE` | `/work/FukaiU/$USER` | read-write on compute nodes. **Job scratch only** — every job makes its own `mktemp` dir and deletes it |
| `BUCKET_PROJECT` | `/bucket/FukaiU/tominaga/pao_rotation` | **read-only** on compute nodes. Caches may be read directly; anything written back goes via the login node |
| `PROJ_DIR` | derived from the checkout | home storage: slow, small quota. **Code, venv and Slurm logs only** — never run output |

Input data is read straight off `BUCKET_DATA_DIR` and never copied to home.

Every `mkdir`/`rm`/stage-out is guarded by `path_under()` to stay inside those
roots, and `safe_rm_work()` refuses to delete anything outside `WORK_BASE`.
Each root is also validated at source time against being empty, relative, or
`/`.

### Slurm log directory

`#SBATCH --output`/`--error` **cannot** be variables — Slurm parses its
directives before any shell runs. They are grouped at the top of each sbatch
under an `EDIT FOR YOUR ACCOUNT` banner, and must name the same directory as
`PAO_SLURM_LOG_DIR`. Create it before submitting; Slurm will not.

### Passwordless SSH to the login node

Stage-out copies results from a compute node to `BUCKET_PROJECT`, which compute
nodes cannot write, so it goes through the login node over SSH. That needs a
key the compute node can use without a passphrase, and a `~/.ssh/config` host
alias matching `PAO_STAGE_HOST` (default `saion`).

### The GPU venv

```bash
srun -p gpu-p100 -t 0-1 -c 4 --mem=16G --pty bash -l
bash <checkout>/clean_scripts/saion/setup_gpu_venv.sh
```

Build on a **GPU compute node**, and on a **P100** node specifically: P100s top
out at CUDA 12.8 and cu128+ wheels dropped Pascal (sm_60), so the script pins a
cu126 torch build. That same wheel runs on the V100 nodes too, which is what
lets `submit_saion.sh` send a job to either partition.

If that node cannot reach the internet, run the same command on the login node
instead — installing pre-built wheels is data movement, not compilation.

The script pins every package to the exact version in the CPU venv, except
torch/triton/nvidia-*/cuda-*, and copies `sitecustomize.py` across. That last
one matters: it patches pyEDM's per-call `Pool` spawning, and without it jobs
slow down by an order of magnitude.

---

## 2. Submitting

```bash
cd ~
<checkout>/clean_scripts/saion/submit_saion.sh saion_parallel_mde.sbatch run_parallel_mde.py
```

`submit_saion.sh` picks the GPU partition that can start the job soonest, then
hands off to `batch/submit.sh`, which submits and immediately writes the job's
`JOB_LOG.md` entry.

### Why the partition is picked by a script

Slurm's own "either partition" syntax (`--partition=gpu-p100,gpu-v100`) is
rejected when an allocation is attached to each partition separately:

```
sbatch: error: Multiple partition job request not supported when a partition
is set in the association
```

So the scheduler is asked instead — `sbatch --test-only -p <partition>` returns
an estimated start time without submitting — and the job goes to the earliest.
Two caveats: `--test-only` ignores your own per-partition allocation, so the
script separately checks whether this job would push your running+pending jobs
past it (and if so takes the latest expected end of those jobs as the start
estimate); and the pick happens **once, at submission**. A pending job does not
move if the other partition frees up later — `scancel` and rerun to re-pick.

`SAION_PARTITION=gpu-p100` forces one; `SAION_PICK_ONLY=1` prints the pick
without submitting.

### Comparing the two parallel scopes

`saion/submit_saion_scope_pair.sh` submits `staged` and `union` as one chain
from the **same sbatch file**. Keeping it one file matters: the two scopes must
otherwise agree on every other knob, or the comparison is not one.

It also refuses to submit if `src/` has uncommitted changes, because the second
job stages `src/` at *its own* start — days from now — not at submission time.
The job itself warns (rather than dies) if the staged commit differs from what
its partner expected: the resume path below legitimately resubmits after a code
fix, and dying there would throw away the queue wait.

---

## 3. What a job does

```
BUCKET_DATA_DIR/<DATA_FILE>     read directly
src/ (code only) --copy-->      WORK_BASE/pmde.<job>.XXXX/
                                  src/      pipeline + helper modules
                                  results/  all pipeline output
                                  cache/    this run's cache entries
BUCKET_PROJECT/tsls_mde_gmn_cache   read-only lookups of past runs
cache/   --hourly + at exit-->  .../tsls_mde_gmn_cache/job_<id>/
results/ --at exit-->           .../tsls_mde_gmn_outputs/job_<id>/
                                  (or job_<id>_FAIL/ if the run failed)
```

The scratch directory is removed **only after both copies succeed**. If
stage-out fails it is kept, and the job prints the exact `rsync` commands to
recover it from the login node.

Four modules are staged, not just the entry script, and each for its own
reason:

- `edm.py` — the EDM engine. The pipeline is nothing without it.
- `mde_gmn/` — the package IS the pipeline; a missing copy fails at import.
- `gpu_edm.py` — missing, every GPU MDE call becomes an `ImportError` recorded
  as a dead end: an empty network, not a crash.
- `gpu_budget.py` — missing, an unhandled `ImportError` mid-run, because
  `edm.py` imports it on the GPU path and from both CCM except-clauses.

---

## 4. Reading the outcome

`results/STATUS.txt`:

```
STATUS: COMPLETED | FAIL
REASON: <what happened>
CUDA_ERROR_LINES: <n>          real CUDA failures
GPU_BUDGET_ADVISORIES: <n>     UNDER-ESTIMATED notices -- informational
NETWORKS_SKIPPED: <n>          Generate() never rebuilt -> no output
CCM_DEAD_ENDS: <n>             evaluated and rejected -- FINDINGS
MDE_CALL_FAILED_LINES: <n>
```

The last three distinctions are the ones that matter, and each was learned the
hard way:

- **`CCM_DEAD_ENDS` are findings, not failures.** A candidate evaluated and
  found non-causal is a result. These are counted separately so they stop being
  indistinguishable from GPU-memory drops, which used to print through the same
  code path.
- **`GPU_BUDGET_ADVISORIES` are not errors.** `gpu_budget.reserve()` prints
  `UNDER-ESTIMATED` *after* a reservation was granted and the call ran, to say
  the shape formula predicted less than the allocator went on to hold. Nothing
  failed. Counting these as errors once stamped a 19-hour run FAIL on 93
  advisories with zero real CUDA errors, burying the one thing that had actually
  gone wrong.
- **`NETWORKS_SKIPPED` is a silent failure made loud.** A network whose
  `Generate()` never rebuilt produces no plots, no fidelity and no output — and
  the pipeline says so and carries on to exit 0. One job lost its only network
  that way and reported success. The outcome check now FAILs on it.

`results/progress.log` is the heartbeat: every stage appends a timestamped
line, flushed and fsynced, so a line survives a crash or an OOM kill the instant
after it is written. It is the only way to tell "still working" from "stuck"
while a job runs — a job once timed out after 4+ hours on a single wave of 3
nodes, doing real work the whole time, with nothing in the log to say which node
was slow or how close to done it was.

---

## 5. Stopping and resuming

### Resuming after a timeout or crash

Every finished MDE node is cached immediately and synced to durable storage
hourly, so **resubmitting the same sbatch skips every completed node**
(`REUSE_BEST_PARAMS=true`). Only nodes that were in flight at the kill are
recomputed.

This is the real protection against exceeding the wall clock: exceeding it costs
a resubmission, not the work.

### Cancelling

```bash
scancel --batch --signal=USR1 <jobid>    # stops the run AND stages out
scancel <jobid>                          # ~30s only; rely on the hourly sync
```

The `USR1` handler stops Python cleanly, so the EXIT trap has time to write
`STATUS.txt` and stage everything out. Slurm also sends `USR1` itself 30 minutes
before the time limit (`--signal=B:USR1@1800`), which is why a timeout still
produces a complete archive.

Python runs in the **background** inside the sbatch, deliberately: bash defers
traps until a foreground child exits, so a foreground `python3` would make the
`USR1` handler unable to fire until Python was already gone.

---

## 6. When something goes wrong

| Symptom | Likely cause | What to change |
|---|---|---|
| OOM-killed | too many workers, each holding its own copy of the Stage 2 frame | lower `--cpus-per-task`, **not** `MDE_D` |
| `CUDA out of memory` in the log | the per-call estimate came in low, or too many workers per card | raise `--gres=gpu:N`, lower `--cpus-per-task`, or reduce `MAX_WINDOW_ROWS`/`CCM_SAMPLE` |
| `UNDER-ESTIMATED` advisories | `crossmap_bytes()`'s headroom factor is too small for these shapes | nothing broke; raise that factor if it is persistent |
| Job exceeds the wall clock | Stage 2 is heavy-tailed — single nodes have run 36 h | resubmit; the per-node cache resumes it |
| Zero drivers for a target | gates too strict, or too few candidate observables | check `EMBED_DIM_RHO_MIN`/`CROSS_MAP_RHO_MIN`/`CCM_PREFILTER_RHO_MIN` before working around it |
| `NETWORKS_SKIPPED` ≥ 1 | `Generate()` diverged or an embedding was empty | the `FINAL Generate() rebuild FAILED` line names the exception |
| Stage 2 taking far longer than expected | the lag count is larger than you think | `LAG_SET:` is printed on its own line at startup; switch to `DELAY_SOURCE=tau_count` to state it directly |
| Everything recomputes despite `REUSE_BEST_PARAMS=true` | a real code change invalidated the fingerprint | expected; see the cache note in `../README.md` |

### Never compute on a login node

Including test suites and quick timings:

```bash
srun -p <partition> -t 0:30:00 -c 4 --mem=8G <command>
```
