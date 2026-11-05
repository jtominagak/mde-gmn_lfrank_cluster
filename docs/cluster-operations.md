# Cluster operations — what a local development machine needs to know

As of 2026-10-23 this project is developed **locally** and executed **only on
the cluster**. Nothing in this repo runs end-to-end on a laptop (see
[Why the pipeline cannot run locally](#why-the-pipeline-cannot-run-locally)),
so every change you make locally has to be reasoned about in terms of a
machine you are not sitting at. This file is the reference for that: the two
clusters, the storage tiers, how data actually moves, and how a job is
submitted and recovered.

The companion docs stay authoritative for their own subjects:
`mde-gmn-architecture.md` (pipeline internals) and, on the
`no-kEDM-gpu-optimization` branch only, `saion-parallelized-mde-runbook.md`
(the step-by-step GPU procedure). This file is the operational envelope around
them, and lives on `main` because it is not GPU-specific.

---

## 1. The two clusters

| | **deigo** | **saion** |
|---|---|---|
| Login | `deigo.oist.jp` (`deigo-login[1-4]`) | `saion.oist.jp` |
| Role | CPU pipelines | GPU pipelines (`no-kEDM-gpu-optimization`) |
| Slurm | `/opt/shared/deigo/slurm/bin` | same tooling |
| Partitions used | `compute`, `short`, `test-gpu` | `gpu-p100`, `gpu-v100` |
| Other partitions | `largejob`, `largemem`, `bigmem`, `datacp`, `dcv` | — |
| Your allocation | standard | **36 cores + 4 GPUs _per partition_, 7-day max** |

Node shapes on deigo, from `sinfo`: `compute` 40+ cores / 512 GB (446 nodes),
`short` 40+ / 512 GB (622 nodes), `largejob` 128 / 512 GB (150),
`largemem` 36+ / 512 GB+ (58), `bigmem` 24+ / 1.5 TB (2).

### Saion has two rules that deigo does not

1. **You cannot ask for two partitions at once.** `#SBATCH --partition=gpu-p100,gpu-v100`
   is rejected for this account — *"Multiple partition job request not supported
   when a partition is set in the association"*, because the allocation is
   attached per partition. `scripts/saion/submit_saion.sh` works around this by
   running `sbatch --test-only -p <part>` against each candidate, parsing the
   estimated start time, and submitting to whichever would start first.
2. **`--test-only` does not know about your allocation.** It ignores the
   36-core / 4-GPU cap, so `submit_saion.sh` additionally subtracts your own
   RUNNING+PENDING usage on that partition and, if the new job would exceed the
   cap, treats the partition's start as the latest expected end of those jobs.

---

## 2. Storage tiers

This is the part that most often causes a job to fail hours in, because the
rules differ **per cluster** for the same path.

| Path | Size | deigo | saion | What belongs there |
|---|---|---|---|---|
| `/home/j/j-tominaga` | 50 TB Isilon, shared, **slow** | rw | rw | Code, venvs, Slurm logs. Never bulk I/O. |
| `/bucket/FukaiU/tominaga/pao_rotation` | 56 PB Lustre | rw | **READ-ONLY on compute nodes** | Durable archive: input data, cross-job cache, archived results |
| `/flash/FukaiU/j-tominaga` | 483 TB | rw, **scratch** | **does not exist** | deigo per-job scratch + result staging |
| `/work/FukaiU/j-tominaga` | — | **not mounted** | rw on compute | saion per-job scratch only |

Consequences worth internalising:

- **`/flash` is deigo-only; `/work` is saion-only.** A path that works in one
  sbatch file is a guaranteed failure in the other. This is why the saion
  variants are separate files rather than a flag.
- **On saion, a compute node cannot write to `/bucket`.** Anything that must
  persist is pushed **through the saion login node** over SSH. That is what
  `stage_dir()` in `scripts/saion/saion_env.sh` does, and why the runbook has a
  one-time passwordless-key step: a compute node must be able to `ssh saion`.
- **`/flash` is scratch, not storage.** It is subject to cleanup. Results live
  there only until the job archives them to `/bucket`.
- `/home` is shared between deigo and saion, which is why the venvs and Slurm
  logs can be referenced by both.
- **`/bucket` on deigo is `rw`, but intermittently fails read-only.** A job
  can die at startup on `mkdir: cannot create directory
  '/bucket/.../job_<id>': Read-only file system` even though the table above
  says `rw`, and even though ~33 earlier `compute` jobs archived there fine.
  It is a **per-node mount fault**, not a policy: the same submission fails on
  one node and succeeds on another minutes later (failing: `deigo010901`,
  `deigo010919`, `deigo010933`, `deigo010924`, `deigo010903`; working:
  `deigo011222`, `deigo011224`, `deigo011102`). It recurred at least six times
  between 2026-08-23 and 08-31. **Remedy: resubmit until the scheduler lands a
  healthy node** — a 15-minute retry loop has always cleared it in ~30–50
  minutes. Full records: `bucket-outage-2026-08-28.md`,
  `bucket-outage-2026-08-30.md`. Escalation to OIST IT was flagged as worth
  doing if the frequency continued, and was never done.

### The safety guards in `saion_env.sh`

Scoped roots are **literal strings** — no `$USER`, `$HOME` or environment
override — and `path_under()` gates every destructive or remote operation:

- `safe_rm_work DIR` refuses to `rm -rf` anything not strictly inside `/work/FukaiU/j-tominaga`.
- `stage_dir SRC DEST` refuses to stage into anything not strictly inside
  `/bucket/FukaiU/tominaga/pao_rotation`, and returns non-zero on any failure
  so callers never delete the only copy.
- Paths containing `..` are rejected outright.

Keep this property when editing locally: a mis-set variable must not be able
to reach `/work`, `/bucket` or another user's tree.

---

## 3. How data moves during a job

### deigo (CPU, e.g. `tsls-mde-gmn.sbatch`)

```
/bucket/.../data/<DATA_FILE>  ──copied──►  $tempdir/data   (TSLS_MDE_GMN_DATA_DIR)
/home/.../pao_rotation        ──staged──►  /flash/FukaiU/j-tominaga/pao_rotation_results/
                                             └─ OUT_DIR    (TSLS_MDE_GMN_RESULTS_DIR)
/bucket/.../tsls_mde_gmn_cache ──read────►  TSLS_MDE_GMN_PREV_RUNS_DIR (cross-job cache)
OUT_DIR  ──on clean exit, archived──────►  /bucket/.../tsls_mde_gmn_outputs/job_<id>/
```

On a crash, timeout or OOM the results are **left in place on `/flash`** and
*not* archived, deliberately, so there is something to post-mortem. The EXIT
trap only clears the `/flash` copy after a verified successful archive.

### saion (GPU, `saion-*.sbatch`)

```
/bucket/.../data/<DATA_FILE>   ──read DIRECTLY──►  TSLS_MDE_GMN_DATA_DIR
/home/.../pao_rotation         ──code only, cp -a──►  /work/FukaiU/j-tominaga/pmde.<job>.XXXX/
                                                        ├─ src/     staged entry scripts + mde_gmn/
                                                        ├─ results/ TSLS_MDE_GMN_RESULTS_DIR
                                                        └─ cache/   TSLS_MDE_GMN_RUN_CACHE_DIR
/bucket/.../tsls_mde_gmn_cache ──read-only──────►  TSLS_MDE_GMN_PREV_RUNS_DIR
cache/   ──hourly + at exit, VIA LOGIN NODE────►  /bucket/.../tsls_mde_gmn_cache/job_<id>/
results/ ──at exit, VIA LOGIN NODE─────────────►  /bucket/.../tsls_mde_gmn_outputs/job_<id>/
```

The hourly cache sync is what makes a timeout survivable: resubmitting the same
file after a crash reuses everything already pushed to `/bucket`.

**Both clusters write into the same bucket tree on purpose.** Cache entries are
fingerprinted against code + installed package versions, so a cache hit written
by deigo and read by saion is still exact.

### Graceful shutdown

`#SBATCH --signal=B:USR1@1800` sends `USR1` to the batch shell 30 minutes
before the wall-clock limit. The handler stops Python cleanly so there is time
to stage results and the cache out to `/bucket` before the hard kill. If you
change the stage-out cost, revisit that 1800 s.

---

## 4. Staging: the rule that bites

`sbatch` copies `scripts/src/*.py` **and `scripts/src/mde_gmn/`** into the job
directory at job start.

- **Never submit while source files are half-moved.** The job stages at start,
  not at submit, so a rename in flight is picked up mid-copy.
- **A new sbatch file must include the `mde_gmn/` package line** or the job dies
  at import. `cp -a` of the entry scripts alone is not enough.
- Because `cleaned_helper_functions.py` lives outside the package on purpose,
  it is staged explicitly too.

---

## 5. Submitting a job

Always through the wrapper, never bare `sbatch`:

```bash
# deigo
scripts/batch/submit.sh <name>.sbatch [<name>.py]

# saion (picks the partition that can start first)
scripts/saion/submit_saion.sh <name>.sbatch [<name>.py]
```

The second argument exists because `saion-<pipeline>.sbatch` runs
`<pipeline>.py` — the names deliberately differ.

`submit.sh` submits, then immediately runs `generate_job_description.sh`, which
creates that job's `JOB_LOG.md` entry: a parameter diff table plus code/design
diffs **against the most recent earlier job of the same pipeline** (keyed on
source filenames, so pipelines are never diffed against each other). You then
add a `### Plain-language summary` section explaining the change in context.
`generate_job_description.sh` is idempotent, so your prose survives the job's
own startup call to the same script.

A `JOB_LOG.md` entry looks like: `## Job <id>`, generation timestamp, git commit
at submission (flagged if the tree was dirty), what it was compared against,
the plain-language summary, then parameter / batch-design / pipeline diffs.

### Never compute on a login node

Including test suites and quick timings:

```bash
srun -p short -t 0:30:00 -c 4 --mem=8G python3 tests/test_lagged_leaf_nodes.py
```

Tests are plain scripts — there is no pytest in this venv.

---

## 6. The precomputed-result cache

Expensive deterministic stages are cached across jobs, so a resubmitted job
reuses them instead of recomputing. Implemented in `mde_gmn/caching.py`.

### Reads come from Bucket. Writes go to scratch, then get staged to Bucket.

This asymmetry is the thing to get right:

| | Path | Env var |
|---|---|---|
| **Lookup (read)** | `/bucket/FukaiU/tominaga/pao_rotation/tsls_mde_gmn_cache` | `TSLS_MDE_GMN_PREV_RUNS_DIR` |
| **Write, deigo** | `/flash/.../pao_rotation_results/<job>` + run cache | `TSLS_MDE_GMN_RESULTS_DIR`, `TSLS_MDE_GMN_RUN_CACHE_DIR` |
| **Write, saion** | `/work/FukaiU/j-tominaga/pmde.<job>.XXXX/{results,cache}` | same two |
| **Then staged to** | `.../tsls_mde_gmn_cache/job_<id>/` | — |

So **a job never reads its cache from `/flash` or `/work`** — those hold only
what the current job just wrote. `find_cached()` globs
`prev_runs_dir/**/<filename>` recursively, newest mtime first, and accepts the
first entry whose fingerprint compares **exactly equal**. Writes go to
`results_dir` *and* `run_cache_dir` (deduplicated), and the stage-out is what
makes them visible to the next job.

Consequence: a result computed by a job that crashed before stage-out is not
in the cache, even though it is sitting on `/flash`. On saion the cache syncs
hourly, so a timed-out job's work is mostly preserved; on deigo stage-out is
at exit, so a hard kill loses that job's cache contribution.

Both clusters share the same bucket tree deliberately — fingerprints are exact,
so a cross-cluster hit is safe.

### What the fingerprint contains

Every stage's key combines its own parameters with `build_code_fingerprint()`:

- **Code** — SHA-256 over **every `mde_gmn/*.py` plus the running entry
  script**, each parsed to an AST with **docstrings stripped**. So comment and
  docstring edits reuse the cache; any real change to a function body anywhere
  in the package invalidates it. The entry script is included by name and
  content, which is why the serial and parallel builders never share entries.
- **Environment** — `importlib.metadata.version()` of `gmn`, `pyEDM`,
  `networkx`, `numpy`, `pandas`, plus **content hashes** of
  `cleaned_helper_functions` and `apps.CreateNetwork` (not installed packages,
  so they are hashed by bytes). Current values are recorded in
  `scripts/env/README.md`. **Bumping any of those five invalidates the whole
  cache.**
- **Data** — the data file name *and* its SHA-256. A re-exported dataset with
  identical contents still hits; one changed value misses.
- **Stage parameters** — e.g. `builder_kwargs` for the network,
  node/candidate pool/`mde_d`/`tp`/rho thresholds for MDE selection.

### Cached stages

Per-node MDE selection, multi-target MDE selection, network construction,
the theta search (`best_params_<network_id>.json`), and the generate-start
sweep.

### Working on this locally

- Assume any substantive edit means a **cold run**, and that the cost is the
  whole pipeline, not one stage.
- A docs-only or comment-only commit is free. This is worth exploiting
  deliberately when you only want to re-describe something.
- Do not "tidy" a dependency version to a newer one casually. It is a
  days-of-compute decision, not a hygiene one.

---

## 7. Why the pipeline cannot run locally

`pyproject.toml` pins resolution to Linux x86_64 only:

```toml
[tool.uv]
environments = ["sys_platform == 'linux' and platform_machine == 'x86_64'"]
```

and depends on `dimx==1.1.3`, `gmn==1.5.1`, `pyedm==2.5.0` plus a
CPU/cu126 `torch`. On macOS (especially Apple Silicon) `uv sync` will not
produce a working environment. The cluster venvs are not portable either:
`pao_rotation/.venv` is 3.2 GB and `.venv-gpu` is 8.6 GB of Linux binaries.

The two things inside those venvs that are *not* reproducible from
`uv.lock` — `sitecustomize.py` (the pyEDM `Pool` fix and the warning-flood
filter) and the exact installed versions the cache fingerprint reads — are
preserved in `scripts/env/`. Read `scripts/env/README.md` before rebuilding
any venv.

**So local work is: read, edit, reason, and run the fake-backed tests.** Per
`CLAUDE.md`, tests reach the tsls pipeline through a namespace assembled from
the package (`load_pipeline_module`) and install MDE fakes with
`install_fake_mde()` — never by assigning to the loaded module. Anything that
needs real `gmn`/`dimx`/`pyEDM` has to go to the cluster via `srun`.

Also note: `pyEDM`'s CCM `parallel=True` is **~1400x slower** than
`parallel=False` once `dimx`/`gmn` are imported, and a per-call `Pool` spawn bug
is worked around by a `sitecustomize.py` in the cluster venv — which is another
reason the cluster environment is not reproducible by `uv sync` alone.

---

## 8. The local → cluster loop

```
edit locally  ─►  scripts/dev/push-to-cluster.sh   (rsync code only)
              ─►  scripts/dev/cluster-run.sh test  (srun the test suite)
              ─►  scripts/dev/cluster-run.sh submit <name>.sbatch
              ─►  write the JOB_LOG.md summary
              ─►  scripts/dev/cluster-run.sh queue / tail
```

`data/` and `results/` are gitignored, so they never travel through git — only
through rsync, and the authoritative copies live on `/bucket`.

Branch discipline is unchanged: `main` is the CPU pipeline, shared changes are
made on `main` and merged into `no-kEDM-gpu-optimization`, and GPU-only code
(`gpu_edm.py`, `network_parallel.py`) never goes to `main`.
