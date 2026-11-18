---
name: sync-clean-scripts
description: Port changes made in scripts/ or tests/ into clean_scripts/, the shareable cleaned-up copy of the pipeline, keeping it behaviour-identical but readable. Use when the clean-sync Stop hook reports drift, when the user asks to sync or update clean_scripts, or after any edit to scripts/src, scripts/batch, scripts/saion, scripts/env or tests/.
---

# Sync clean_scripts/ with scripts/

`clean_scripts/` is the copy of the pipeline that gets shared. It must **compute
exactly what `scripts/` computes**, but it is restructured to be readable, so a
change is *ported*, never copied. `scripts/dev/clean_sync.py` tracks which
changes have not been ported yet; this skill is how to port them.

## 1. See what drifted

```bash
python3 scripts/dev/clean_sync.py status      # what, and where each one goes
python3 scripts/dev/clean_sync.py diff        # the exact change since the last port
```

The status tags are CODE, TEXT, NEW, DELETED and UNCLASSIFIED; the docstring of
`clean_sync.py` defines each one. An UNCLASSIFIED file has to go into `MAPPING`
or `IGNORE` in `clean_sync.py` first. Don't port anything into `clean_scripts/`
that `IGNORE` says is out of scope; the 1 Hz `simple-mde-gmn.py` pipeline never
goes there.

## 2. Read the counterpart before editing it

Clean code is not laid out like `scripts/`. Find where the changed logic lives
in the counterpart named in the status output. Usually it is a function with
the same name. The known relocations:

| In scripts/ | In clean_scripts/ |
|---|---|
| `cleaned_helper_functions.py`, imported as `tsls` | `src/edm.py`, imported as `edm` |
| entry scripts `tsls-mde-gmn.py`, `no-kedm-...-tsls-mde-gmn.py` | `src/run_serial_mde.py`, `src/run_parallel_mde.py` |
| pool/heartbeat loops in `network.py` and `network_parallel.py` | `mde_gmn/waves.py` `run_mde_batch()` |
| `_failed_mde_result`, `_embedding_spec_for`, `_generalized_embed`, `_simulated_nodes`, `_blocked_observables`, `_GPU_MEMORY_STOP_ADVICE` | same names, without the leading underscore |
| `_ccm_converged` and the inline convergence checks | `edm.converged()`. `proper_mde`'s own gate is deliberately separate, see README |
| `if/elif` over gmn function types in `gmn_embedding.py` | one function per type, plus the `_FORECAST`/`_GENERATE` tables |
| `pd.read_csv(net_result['DATA_AFTER_MDE'])` | `tsls_data.network_data_frame(net_result)` |
| depth-0 seed: two branches in `pipeline.main` | one loop with `if joint:` |
| `saion_env.sh` | `saion/cluster_env.sh` (no account-specific paths) |
| saion parallel sbatch | `batch/saion_parallel_mde.sbatch`, **and** its CPU twin `batch/serial_mde.sbatch` |

## 3. Port it, in clean_scripts' style

- **Same behaviour.** Use the same values, gates, ordering of RNG draws, and
  cache payload keys. If the clean tree has removed redundant compute on that
  path (see README "What changed"), keep it removed and apply the change to the
  code that is left.
- **Clean style.** Use the clean names and module boundaries. Write docstrings
  that explain the idea in the clean tree's own words. Don't carry over job
  archaeology ("job 4774136 showed…"): that goes in `docs/parameters.md` or
  `docs/deployment.md`, organised by topic. Don't bake in account-specific
  paths; use environment variables with working defaults.
- **Parameters.** A new or changed `TSLS_MDE_GMN_*` variable goes in
  `config.py`, in **both** sbatch files' parameter blocks and `for _p in …`
  export loops, and in `docs/parameters.md`. Shared values in the two sbatch
  files stay identical; only GPU-specific knobs differ.
- **Tests.** Port the test too: switch to clean paths and module names, and
  install MDE fakes the way `clean_scripts/tests/_pipeline_test_utils.py`
  does.
- **Fixes vs. differences.** If `scripts/` has a bug the clean copy already
  avoids, don't reintroduce it. List it under README "Behaviours that differ
  from the original, on purpose", and tell the user so they can fix
  `scripts/`. Any change to `scripts/` code invalidates the whole cache, so
  that is their decision.
- **TEXT-only drift.** Port the explanation only if a clean-tree reader needs
  it.

## 4. Verify

- `bash -n` every sbatch/sh you touched.
- Run both suites on a compute node, **never on the login node**:

```bash
srun -p gpu-p100 -t 0:45:00 -c 4 --mem=16G --gres=gpu:1 bash -c '
  cd <repo>; for tree in tests clean_scripts/tests; do fail=0
    for t in $tree/test_*.py; do timeout 900 .venv-gpu/bin/python3 "$t" >/dev/null 2>&1 \
      && echo "PASS $t" || { fail=1; echo "FAIL $t"; }; done; echo "SUITE $tree EXIT=$fail"; done'
```

  (On deigo: `-p short`, no `--gres`, `.venv/bin/python3`; the GPU tests skip.)
  A test that fails in `tests/` too was not broken by the port. Say so rather
  than mark it as a regression.

## 5. Record it

```bash
python3 scripts/dev/clean_sync.py mark-synced <each ported scripts/ or tests/ path>
python3 scripts/dev/clean_sync.py status        # must say "in sync"
```

To deliberately not mirror a change, record why instead:
`mark-synced PATH --note 'why'`.

Then report to the user which `scripts/` changes went where, any deliberate
difference, and the test results. Commit only if they ask, and commit
`scripts/dev/clean_sync_state.json` together with the `clean_scripts/`
changes.
