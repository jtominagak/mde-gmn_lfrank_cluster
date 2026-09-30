# pao_rotation — orientation for Claude Code

## Read this first
**`docs/mde-gmn-architecture.md`** — what the pipeline is made of, the stage
order, the dependency diagram, and the three things that surprise people
(gmn is patched on import; MDE is called through the module object; the cache
key includes the entry script's path).

The algorithm contract the pipelines must obey — lagged drivers are leaf nodes,
cycle-closing candidates are excluded before MDE, Stage 2 is lagged-only — is
explained in `scripts/src/mde_gmn/network.py`'s module docstring and pinned by
`tests/test_lagged_leaf_nodes.py`. Keep that test passing.

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

## Branches
- `main` — CPU pipeline.
- `no-kEDM-gpu-optimization` — adds the GPU CCM path (`gpu_edm.py`) and
  `network_parallel.py`. Shared changes are made on `main` and merged in;
  GPU-only code never goes to `main`.

## Rules that are easy to get wrong
- **Never compute on a login node.** Use `srun -p short -t 0:30:00 -c 4
  --mem=8G ...` even for test suites and quick timings.
- Submit pipeline jobs with `scripts/batch/submit.sh <name>.sbatch [<name>.py]`,
  then add the plain-language summary to that job's `JOB_LOG.md` entry.
- Don't run jobs while source files are half-moved: sbatch stages
  `scripts/src/*.py` **and `scripts/src/mde_gmn/`** at job start. A new sbatch
  needs that package line or the job dies at import.
- Caches are keyed on a hash of the whole package plus the entry script: real
  code changes miss the cache, comment-only changes reuse it.
- Tests reach the tsls pipeline through a namespace assembled from the package
  (`load_pipeline_module`), because its entry script is thin. Install MDE fakes
  with `install_fake_mde()`, never by assigning to the loaded module.
