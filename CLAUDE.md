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

## Every change must be a feature you can switch off (standing rule, since 2026-11-18)
This applies to every agent and subagent, every round.

The basic pipeline is built. Work from now on means changing, adding or
removing parts of it, then comparing runs with and without each change. So
every change the user asks for is built as a feature with an on/off switch.
**With the switch off, the pipeline must behave exactly as it did before that
change.** This must still hold after many more rounds: a feature added several
rounds ago must still switch off cleanly, whatever has been added since.

**What needs a switch.** Any change to what the pipeline computes, selects or
outputs: a new stage, method, filter, metric or gate, or a change to one that
exists. "Remove X" needs one too: add a switch that turns X off, and leave X's
code in place. No switch is needed for fixes to crashes or to results that
break the algorithm contract above, for refactors that change no behaviour, or
for logging and docs. If you decide a change doesn't need a switch, say so and
why. If unsure, add the switch.

**How to build the switch**
1. **One switch per feature, in `scripts/src/mde_gmn/config.py`.** Read it from
   `TSLS_MDE_GMN_<FEATURE>` with the existing `_env_*` helpers. Start its
   comment with `# FEATURE TOGGLE (YYYY-MM-DD):` and say what OFF restores.
   Then `grep -n "FEATURE TOGGLE" scripts/src/mde_gmn/config.py` lists every
   switch. Add it to the MODIFIABLE PARAMETERS block of every sbatch that runs
   the affected pipeline. The default is what the user asked for. If they
   didn't say, default to OFF and tell them.
2. **Keep the new code in one place.** Put the feature in its own function or
   module, and check the switch at as few call sites as possible (ideally one).
   Don't scatter `if cfg[...]` checks through shared code. When changing
   existing behaviour, keep the old code as the OFF branch; don't edit it in
   place. With the switch OFF, the original path must run unchanged: no new
   imports, required parameters, random draws, reordering or changed defaults
   on shared functions.
3. **Features are independent.** Each switch must work whatever the others are
   set to. If feature B only makes sense with feature A on, `config.py` raises
   a clear error when B is on and A is off (the same "raise, don't ignore"
   convention retired settings use). Never build B by changing A's internals
   in a way that stops A from switching off, and never remove or bypass an
   earlier feature's switch.
4. **Add the switch to the cache key.** Stage fingerprints are built from
   explicit per-stage parameters (`build_*_fingerprint` in `caching.py`). If
   the switch changes a stage's output, add it to that stage's fingerprint, or
   ON and OFF runs will reuse each other's cached results.

**The extra check every implementation needs.** Do this before you report the
work as done:
- Add a test under `tests/` that shows **OFF matches pre-change behaviour**
  and **ON does the new thing**. Before editing, record a baseline from the
  unchanged code to compare OFF against. Use fakes where possible. Tests that
  need real `gmn`/`dimx`/`pyEDM` go to the cluster.
- Run the **whole** `tests/` suite, not just the new test, on a compute node
  via `srun`. Then every earlier feature's OFF test is checked again. Toggle
  tests stay in the suite for good: never delete or loosen an earlier
  feature's toggle test to get a new one to pass.
- Port to `clean_scripts/` as usual, with the switch included.
- In your final message, give the switch's name, its default, what OFF
  restores, which tests ran and where (locally or on the cluster), and any
  check you could not run.

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

## clean_scripts/ must follow scripts/
`clean_scripts/` (this branch only) is the shareable copy of the pipeline:
behaviour-identical to `scripts/`, restructured to be readable. Every change
to `scripts/` or `tests/` gets ported there, never copied verbatim. Use the
`/sync-clean-scripts` skill. `scripts/dev/clean_sync.py status` reports what
is unported. A Stop hook in `.claude/settings.json` blocks finishing a turn
while code drift remains, and `scripts/dev/git-hooks/pre-commit` warns (it
never blocks). If a new file in `scripts/` or `tests/` has no clean
counterpart, add it to `IGNORE` in `clean_sync.py` and give the reason. Set
`CLEAN_SYNC_HOOK=off` to pause the hooks.

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
