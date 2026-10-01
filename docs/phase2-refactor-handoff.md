# Phase 2 refactor — DONE

All five steps are complete on both branches. This file is kept as the record of
what was done and what it changed; for orientation, read
**`docs/mde-gmn-architecture.md`** instead.

**Goal (user's words):** the codebase is messy from accumulated iterative edits.
Every file should read as clear logic, file dependencies should be explicit in
concise header comments, and a non-expert should be able to read the algorithm.
No behaviour change.

**Scope:** `scripts/src/tsls-mde-gmn.py` and the parallel-MDE fork only.
`simple-mde-gmn.py` was deliberately left alone, including its test
expectations — it keeps the older algorithm on purpose.

## What changed

| | before | after |
|---|---|---|
| `scripts/src/tsls-mde-gmn.py` | 5,076 lines | 44 |
| the parallel-MDE fork | 5,415 lines | 50 |
| shared code | duplicated between them | one `mde_gmn` package |

Of the fork's 61 definitions, 48 were byte-identical copies of the parent's.
Those are gone. The 8 that are genuinely its own live in
`mde_gmn/network_parallel.py`; the rest is shared.

`main()` takes the network builder as an argument, which is the only thing the
two pipelines disagree about, plus `entry_path` (the cache key includes it) and
`builder_config_keys` (options only one builder understands).

Every extraction was done with `scripts/dev/split_module.py`, which moves
definitions verbatim, so each commit's diff is a move plus an import line.

## Two real bugs this surfaced

1. **The fork never got Phase 1's data-export fix.** `fa804e8` taught
   `build_network` to export one data column per lagged leaf, because
   `partition_network` classifies leaves as external inputs and gmn feeds those
   from the exported frame BY NODE NAME. `eb2a42c` ported Phase 1's algorithm
   changes to the fork but not that export, so the parallel pipeline created
   lagged leaf nodes and handed gmn a frame with no columns for them — broken
   for any network containing a lagged pick. Sharing `build_network` fixed it.

2. **Comments that contradicted the code.** `gmn_build.py` and
   `network_parallel.py` both still said Stage 2's real-observable selections
   recurse. Phase 1 made Stage 2 lagged-only and non-recursive, so both would
   have told a reader the opposite of what the code does.

## The traps, for whoever works here next

1. **Tests fake MDE by assigning to the module.** `network.py` therefore calls
   it as `mde.run_mde_for_target(...)` through the module object. A
   `from mde_gmn.mde import run_mde_for_target` would bind the real function at
   import time and the tests would silently run real MDE against real pyEDM.
   `install_fake_mde()` follows the builder to wherever it is defined and
   raises if a module ever has both bindings.
2. **`build_code_fingerprint(entry_path)`** hashes every module in the package
   plus the entry script. Adding a module is covered automatically. Do not
   reintroduce single-file hashing, and do not let `main()` pass its own
   `__file__` — that would merge the two pipelines' cache namespaces.
3. **Every tsls-family sbatch stages `scripts/src/mde_gmn/`.** A new staging
   site or a new sbatch needs that line, or the job dies at import inside
   `/work`. `saion-preflight.sbatch` now imports the package, so a missing
   stage is caught in preflight rather than days into a job.
4. **`gmn` is patched when `gmn_embedding` is imported.** Import order matters:
   anything that constructs `gmn` objects must import it first.
5. **Don't run jobs mid-refactor.** The sbatch copies source at job start, so a
   job beginning while files are half-moved stages broken code.
6. **`cleaned_helper_functions.py` and `gpu_edm.py` stay outside the package** —
   the sbatch scripts stage them as individual files and the environment
   fingerprint hashes `cleaned_helper_functions` by content.

## Verification

Never on a login node:

```bash
srun -p short -t 0:30:00 -c 4 --mem=8G bash -c 'cd /home/j/j-tominaga/pao_rotation && \
  fail=0; for t in tests/test_*.py; do .venv/bin/python3 "$t" >/dev/null 2>&1 || \
  { fail=1; echo "FAIL $t"; }; done; echo "SUITE-EXIT=$fail"'
```

17 files on `main`, 21 on `no-kEDM-gpu-optimization`, all passing. Beyond the
suite, each entry script was run with a nonexistent `DATA_FILE` and reaches
`pd.read_csv` inside `main()`, which exercises the whole import graph, the gmn
patching and `load_config()`.

## Still open (not refactor work)

1. Saion job shape: 9 cores + 120 GB (a fair quarter-node share) versus the
   current 36 cores + 400 GB, which occupies a whole node and strands 3 GPUs.
   Needed before the half-production run.
2. `docs/tsls-mde-gmn-pipeline.tex` still describes the pre-Phase-1 design
   (lagged drivers as delay-tagged edges out of the real node, Stage 2
   recursing into real observables). It is now the last place that description
   survives.
3. Saion: `.venv-gpu` is built and preflight 4724497 passed on a P100 -- every
   check except the stage-out ssh round trip, which needs the key from runbook
   step 1 (the user's action; see `docs/saion-parallelized-mde-runbook.md`).
   The pipeline job itself has not been submitted.
