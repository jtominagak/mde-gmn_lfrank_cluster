# Phase 2 refactor — handoff (steps 2–5)

Written 2026-09-30 to let a fresh session pick this up. Step 1 is committed and green;
steps 2–5 are not started.

**Goal (user's words):** the codebase is messy from accumulated iterative edits. Every file
should read as clear logic, file dependencies should be explicit in concise header comments, and
a non-expert should be able to read the algorithm. No behaviour change.

**Scope:** `scripts/src/tsls-mde-gmn.py` and `scripts/src/no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py`
(the parallel-MDE fork) only. `simple-mde-gmn.py` is deliberately OUT of scope — do not touch it,
and do not "fix" tests that assert its older behaviour.

---

## 1. Where things stand

| | |
|---|---|
| `main` | `369362f` — Phase 2 step 1 done, 18/18 tests pass |
| `no-kEDM-gpu-optimization` | `baa4e77` — same step 1 plus the GPU/fork work, 21/21 tests pass |

Phase 1 (algorithm correctness) finished earlier the same day: lagged drivers are leaf nodes,
cycle-closing candidates are excluded before MDE (with a serial recompute when a precomputed or
same-wave result violates it), Stage 2 is lagged-only, `STAGE2_ALLOW_REAL_OBSERVABLES` retired.
`tests/test_lagged_leaf_nodes.py` is the executable spec — keep it passing.

Step 1 created `scripts/src/mde_gmn/` with `progress.py`, `columns.py`, `tsls_data.py`,
`caching.py`, plus `__init__.py` holding the module map and dependency direction. Read
`__init__.py` first: it is the plan for the remaining modules.

## 2. Branch workflow (matters — get this right)

The package is branch-neutral (verified: step 1's modules extracted from either branch are
byte-identical). The GPU branch's entry scripts carry `use_gpu`/`mp_start_method` arguments that
must NOT reach `main`.

1. Do each shared extraction on **`main`** first, against `main`'s own entry script.
2. `git checkout no-kEDM-gpu-optimization && git merge main` — expect small conflicts only where
   a moved function differs by a GPU argument. Resolve by keeping the GPU-branch signature and
   the new (main) structure, exactly as the earlier merges did.
3. Fork-only work (`network_parallel.py`, the fork entry script) happens on the GPU branch only.

## 3. The tool

`scripts/dev/split_module.py` does one extraction mechanically: moves the named top-level
definitions VERBATIM into `mde_gmn/<module>.py`, derives the imports the moved code needs,
deletes them from the entry script, and adds `from mde_gmn.<module> import ...` for whatever the
entry still uses. It refuses to run if the moved code depends on definitions that would be left
behind — that means the boundary is wrong or something must move first.

```bash
cat > /tmp/hdr.txt <<'EOF'
"""One line saying what this module is for.

A short paragraph of what it actually does, in plain language.

Depends on: <package modules / external>.
Imported by: <modules>.
"""
EOF
.venv/bin/python3 scripts/dev/split_module.py --module mde \
    --entry scripts/src/tsls-mde-gmn.py \
    --names run_mde_for_target,run_multi_mde_for_targets,... \
    --header-file /tmp/hdr.txt --dry-run      # then re-run without --dry-run
```

Dry-run output for the planned `mde` module (already checked): 8 definitions, 658 lines, nothing
stranded, entry 5076 → 4451 lines.

## 4. Remaining module boundaries

Function lists come from the inventory of `main`'s entry script. Extract in this order; each
module only needs the ones above it.

**Step 2 — the big win.** The fork duplicates ~5,800 lines of the parent and differs by only 7
functions plus one builder, so this is where the duplication collapses.

- `mde.py` — `run_mde_for_target`, `run_multi_mde_for_targets`, `plot_mde_rho_vs_dimension`,
  `plot_3d_mde_projection`, `_failed_mde_result`, `_failed_multi_mde_result`,
  `ccm_prefilter_observables`, `_ccm_prefilter_pair`
- `network.py` — `_accept_mde_result_edges`, `_real_candidate_pool`, `build_nested_mde_network`,
  `partition_network`, `_simulated_nodes`, `report_partition`
- `network_parallel.py` (GPU branch only) — `mde_node_universe`, `_mde_stage_spec`,
  `_mde_call_arguments`, `_run_one_mde`, `_mde_result_for_node`, `_blocked_observables`,
  `precompute_all_mde`, and the fork's own `build_nested_mde_network`. It should import the
  shared pieces from `network.py` rather than keep copies. When this lands, the fork entry
  script should shrink to roughly the parent's entry plus a different builder argument.

**Step 3**

- `gmn_embedding.py` — `_embedding_spec_for`, `network_max_embedding_offset`,
  `_node_embedding_spec`, `_generalized_embed`, `_generalized_node_init`,
  `_generalized_node_forecast`, `_generalized_node_generate` **and the module-level
  `gmn.Node.__init__ = ...` monkeypatch assignments that follow them**. Those run at import
  time; whoever imports this module patches `gmn`. Say so in the header.
- `gmn_build.py` — `_lib_train_test_ranges`, `target_E_tau_rho_grid`, `estimate_node_dimension`,
  `_ccm_converged`, `build_interaction_matrix`, `build_network`, `build_gmn_config`
- `theta.py` — `proper_predict_nonlinear`, `retune_theta_for_all_nodes`,
  `evaluate_smap_candidate`, `run_theta_search`
- `generate.py` — `sweep_generate_start`, `rebuild_at_best_params`, `run_forecast_diagnostic`,
  `gmn_fidelity_rho`, `plot_fidelity` (`build_start_sweep_fingerprint` already lives in
  `caching.py`)
- `plots.py` — `draw_colored_network`, `plot_3d_network`
- `diagnostics.py` — `run_tp_vs_rho_sweep`, `run_univariate_e_tau_grid`, `_cae`,
  `_score_baseline`, `MDE_EVALUATE_BASELINE_TIMEOUT_SECONDS`, `_pca_baseline_worker`,
  `_dmap_baseline_worker`, `_run_baseline_with_timeout`, `run_mde_evaluate`

**Step 4**

- `config.py` — `_env_bool`, `_env_list`, `_env_float_list`, `_env_int_list`, `load_config`
  (516 lines, mostly comments explaining each knob — good; keep them)
- `pipeline.py` — `target_groups_for_mode`, `main`. `main` should take the network builder as an
  argument so both entry scripts share it.
- Entry scripts end up thin: set `sys.path`, import the package, pass the builder, call `main`.

**Step 5**

- Tests: see the trap in §5 — this is the fiddly part.
- Add the dependency diagram to the docs (`__init__.py` has the text; a figure would help).
- Rewrite leftover change-history comments ("R12", "R14", "ported unchanged from
  simple-mde-gmn.py", "see JOB_LOG.md") into explanations of what the code does. Move the history
  into `docs/` if it is worth keeping. This is the bulk of the "accessible to non-experts" ask
  and can be done module by module after the moves.

## 5. Traps (each of these has already bitten or nearly bitten)

1. **Tests fake `run_mde_for_target` by assigning to the module they loaded**
   (`module.run_mde_for_target = fake`, see `tests/_pipeline_test_utils.py`). That works while
   the caller lives in the same module as the name. Once `build_nested_mde_network` moves into
   `network.py`, `network.py` must call it through the module object:
   `from mde_gmn import mde` … `mde.run_mde_for_target(...)`. Then tests patch
   `mde_gmn.mde.run_mde_for_target`. If instead `network.py` does
   `from mde_gmn.mde import run_mde_for_target`, patching cannot reach it and the tests will
   silently run the real MDE. Same for `ProcessPoolExecutor`, which tests replace with
   `FakeProcessPoolExecutor`.
2. **`build_code_fingerprint(entry_path)`** hashes every module in the package plus the entry
   script. Adding a module is automatically covered; do not reintroduce single-file hashing, and
   keep the entry path argument so the two pipelines keep separate cache namespaces.
3. **Every tsls-family sbatch stages `scripts/src/mde_gmn/`.** Any new staging site (or a new
   sbatch) needs that line, or the job dies at import inside `/work`.
4. **`gmn` monkeypatching happens at import** of the embedding module (`gmn.Node.__init__`,
   `Forecast`, `Generate`). Import order matters: anything that constructs `gmn` objects must
   import that module first.
5. **Don't run jobs mid-refactor.** The sbatch copies `scripts/src/*.py` at job start, so a job
   beginning while files are half-moved stages broken code. Finish a step, run the suite, commit,
   and only then submit.
6. **`cleaned_helper_functions.py` and `gpu_edm.py` stay outside the package** — the sbatch
   scripts stage them as individual files and `cleaned_helper_functions` is content-hashed by the
   environment fingerprint.

## 6. Verification (after every step)

Never on a login node:

```bash
srun -p short -t 0:30:00 -c 4 --mem=8G bash -c 'cd /home/j/j-tominaga/pao_rotation && \
  fail=0; for t in tests/test_*.py; do .venv/bin/python3 "$t" >/dev/null 2>&1 || \
  { fail=1; echo "FAIL $t"; }; done; echo "SUITE-EXIT=$fail"'
```

Expect `SUITE-EXIT=0` with 18 files on `main`, 21 on the GPU branch. `test_parallel_mde_equivalence.py`
(GPU branch) is the strongest check: it asserts the fork's `staged` scope builds the same network
as the parent, edge for edge in order.

Commit after each module with a message saying what moved and what, if anything, changed
behaviourally (normally: nothing).

## 7. Open questions for the user (not blocking the refactor)

1. Saion job shape: 9 cores + 120 GB (a fair quarter-node share) vs the current 36 cores + 400 GB,
   which occupies a whole node and strands 3 GPUs. Needed before the half-production run.
2. `docs/tsls-mde-gmn-pipeline.tex` still describes the pre-Phase-1 design (lagged drivers as
   delay-tagged edges out of the real node). Rewrite as part of Phase 2?
3. Nothing has ever run on Saion: the venv setup, preflight and A/B are all still pending
   (`docs/saion-parallelized-mde-runbook.md`).
