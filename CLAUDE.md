# pao_rotation — orientation for Claude Code

## Work in progress (read this first)
A refactor is mid-flight: **`docs/phase2-refactor-handoff.md`** — Phase 2 step 1 is committed and
green, steps 2–5 are not started. It carries the module boundaries, the branch workflow, the
traps, and the verification command. `scripts/dev/split_module.py` is the tool it uses.

The algorithm contract the pipelines must obey (lagged drivers are leaf nodes, cycle-closing
candidates excluded before MDE, Stage 2 lagged-only) is in that doc's §1 and pinned by
`tests/test_lagged_leaf_nodes.py`.

## Layout
- `scripts/src/mde_gmn/` — the pipeline package; `__init__.py` has the module map and the
  dependency direction. Start there.
- `scripts/src/tsls-mde-gmn.py` — entry script, serial network builder (branch `main`).
- `scripts/src/no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py` — entry script, parallel
  builder (branch `no-kEDM-gpu-optimization` only).
- `scripts/src/cleaned_helper_functions.py` — the TSLS-aware EDM engine (proper_mde, proper_ccm);
  outside the package on purpose.
- `scripts/src/simple-mde-gmn.py` — sibling 1 Hz pipeline. **Out of scope** for current work;
  it keeps the older algorithm on purpose.
- `scripts/batch/*.sbatch` — one per pipeline; `scripts/saion/` — Saion (GPU cluster) helpers.
- `tests/` — plain scripts, no pytest in this venv: `python3 tests/test_x.py`.

## Branches
- `main` — CPU pipeline.
- `no-kEDM-gpu-optimization` — adds the GPU CCM path (`gpu_edm.py`) and the parallel-MDE fork.
  Shared changes are made on `main` and merged in; GPU-only code never goes to `main`.

## Rules that are easy to get wrong
- **Never compute on a login node.** Use `srun -p short -t 0:30:00 -c 4 --mem=8G ...` even for
  test suites and quick timings.
- Submit pipeline jobs with `scripts/batch/submit.sh <name>.sbatch [<name>.py]`, then add the
  plain-language summary to that job's `JOB_LOG.md` entry.
- Don't run jobs while source files are half-moved: sbatch stages `scripts/src/*.py` at job start.
- Caches are keyed on a hash of the whole package plus the entry script; real code changes miss
  the cache, comment-only changes reuse it.
