# pao_rotation — orientation for Claude Code

## Read this first
**`../CLUSTER.md`** — the cross-project cluster contract (storage tiers, the
intermittent `/bucket` read-only fault, the `tominaga` vs `j-tominaga` path
split). This project's own `docs/cluster-operations.md` is more detailed and
**wins on any conflict**; `../CLUSTER.md` exists so the other projects here
share the same baseline.

**`docs/mde-gmn-architecture.md`** — what the pipeline is made of, the stage
order, the dependency diagram, and the three things that surprise people
(gmn is patched on import; MDE is called through the module object; the cache
key includes the entry script's path).

The algorithm contract the pipelines must obey — lagged drivers are leaf nodes,
cycle-closing candidates are excluded before MDE, Stage 2 is lagged-only — is
explained in `scripts/src/mde_gmn/network.py`'s module docstring and pinned by
`tests/test_lagged_leaf_nodes.py`. Keep that test passing.

**`docs/cluster-operations.md`** — read this before anything that touches a
job, a path or a dataset. Development is local; execution is cluster-only.
It covers the two clusters, the storage tiers (`/flash` is deigo-only,
`/work` is saion-only, `/bucket` is read-only from saion compute nodes), how
data actually moves during a job, and why the pipeline cannot run on a laptop.

## Layout
- `scripts/src/mde_gmn/` — the pipeline, one module per stage. `__init__.py`
  has the module map and the dependency direction.
- `scripts/src/tsls-mde-gmn.py` — entry script, serial builder (~45 lines).
- `scripts/src/no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py` — entry
  script, parallel builder (`no-kEDM-gpu-optimization` branch only).
- `scripts/src/cleaned_helper_functions.py` — the TSLS-aware EDM engine
  (proper_mde, proper_ccm); outside the package on purpose.
- `scripts/src/simple-mde-gmn.py` — sibling 1 Hz pipeline, still one file.
  **Out of scope** for current work; it keeps the older algorithm deliberately.
- `scripts/batch/*.sbatch` — one per pipeline; `scripts/saion/` — Saion (GPU
  cluster) helpers, with `docs/saion-parallelized-mde-runbook.md`.
- `tests/` — plain scripts, no pytest in this venv: `python3 tests/test_x.py`.

## Local development, cluster execution
Since 2026-10-23 this repo is edited locally and run only on the cluster.

- **Nothing here runs end-to-end locally.** `pyproject.toml` pins resolution to
  `sys_platform == 'linux' and platform_machine == 'x86_64'`, and the pipeline
  needs `dimx`, `gmn`, `pyedm` and a cu126 `torch`. Do not try to `uv sync` on
  macOS and do not "fix" that pin.
- Locally you can read, edit, reason, and run the **fake-backed** tests. Any
  test that needs real `gmn`/`dimx`/`pyEDM` goes to the cluster via `srun`.
- The helpers in `scripts/dev/` exist so cluster rules do not have to be
  remembered: `push-to-cluster.sh` (code only, never `JOB_LOG.md`) and
  `cluster-run.sh` (`test`, `submit`, `queue`, `tail`, `pull-log`, `results`).
  `CLUSTER=saion` switches them to the GPU cluster.
- **`JOB_LOG.md` is written on the cluster**, by `generate_job_description.sh`
  at job start. It syncs cluster → local only
  (`scripts/dev/cluster-run.sh pull-log`); pushing it from local clobbers
  entries that jobs wrote.
- `data/` and `results/` are gitignored. They never travel through git, and
  the authoritative copies live on `/bucket`.

## Branches
- `main` — CPU pipeline.
- `no-kEDM-gpu-optimization` — adds the GPU CCM path (`gpu_edm.py`) and
  `network_parallel.py`. Shared changes are made on `main` and merged in;
  GPU-only code never goes to `main`.

## Rules that are easy to get wrong
- **Never compute on a login node.** Use `srun -p short -t 0:30:00 -c 4
  --mem=8G ...` even for test suites and quick timings.
- Submit pipeline jobs with `scripts/batch/submit.sh <name>.sbatch [<name>.py]`
  on deigo, or `scripts/saion/submit_saion.sh` on saion (it picks whichever
  restricted partition can start first — saion rejects a multi-partition
  request for this account). Then add the plain-language summary to that
  job's `JOB_LOG.md` entry.
- Don't run jobs while source files are half-moved: sbatch stages
  `scripts/src/*.py` **and `scripts/src/mde_gmn/`** at job start. A new sbatch
  needs that package line or the job dies at import.
- Caches are keyed on a hash of the whole package plus the entry script
  (docstring-stripped AST), the data file's SHA-256, the stage's parameters,
  and the versions of `gmn`/`pyEDM`/`networkx`/`numpy`/`pandas`: real code
  changes miss the cache, comment-only changes reuse it, and bumping one of
  those five dependencies invalidates **all** of it.
- **The cache is read from `/bucket`, never from `/flash` or `/work`.** Scratch
  holds only what the running job just wrote; stage-out is what makes a result
  visible to the next job. See `docs/cluster-operations.md` §6.
- `scripts/env/` holds the two things the cluster venvs contain that `uv.lock`
  cannot rebuild: `sitecustomize.py` (the pyEDM per-call `Pool` fix and the
  `FindNeighbors` warning-flood filter — it only works when installed into the
  venv's `site-packages`) and the frozen version manifests. Read
  `scripts/env/README.md` before rebuilding a venv.
- Tests reach the tsls pipeline through a namespace assembled from the package
  (`load_pipeline_module`), because its entry script is thin. Install MDE fakes
  with `install_fake_mde()`, never by assigning to the loaded module.
