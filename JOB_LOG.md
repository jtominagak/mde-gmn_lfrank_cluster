# Job Log

Running record of parameter and pipeline/design changes across job
submissions (see scripts/batch/submit.sh), newest entry first.

<!-- new entries inserted below this line -->

## Job 36969861

Generated: 2026-08-10T01:21:32+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_36969708**

### Plain-language summary

**10 of 10 in the MDE_D x Tp sweep** -- see job_36969445's entry for the full rationale. The main idea for the sweep is that MDE_D must not be capped arbitrarily, and Tp must be set for longer prediction horizon because in the generative mode, we also care about long term dynamics. If Tp is only 1, the generative mode will likely only generate short-term dynamics, and die off quickly. This might suggest a need for multi-target MDE.
MDE_D=999 (unchanged), Tp raised 20 -> 30 -- last combo in the sweep. Once all 10
complete, compare each network's generative fidelity (`gmn_fidelity_head_position_*.png`
/ the `fidelity rho` log lines) across the grid to see whether MDE_D, Tp, or neither
actually explains job 36786285's poor generative performance.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_36969708 | job_36969861 |
|---|---|---|
| TP | 20 | 30 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---


## Job 36969708

Generated: 2026-08-10T01:21:14+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_36969703**

### Plain-language summary

**9 of 10 in the MDE_D x Tp sweep** -- see job_36969445's entry for the full rationale.
MDE_D=999 (unchanged), Tp raised 10 -> 20.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_36969703 | job_36969708 |
|---|---|---|
| TP | 10 | 20 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---


## Job 36969703

Generated: 2026-08-10T01:20:53+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_36969695**

### Plain-language summary

**8 of 10 in the MDE_D x Tp sweep** -- see job_36969445's entry for the full rationale.
MDE_D=999 (unchanged), Tp raised 5 -> 10.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_36969695 | job_36969703 |
|---|---|---|
| TP | 5 | 10 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---


## Job 36969695

Generated: 2026-08-10T01:20:34+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_36969688**

### Plain-language summary

**7 of 10 in the MDE_D x Tp sweep** -- see job_36969445's entry for the full rationale.
MDE_D=999 (unchanged), Tp raised 1 -> 5.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_36969688 | job_36969695 |
|---|---|---|
| TP | 1 | 5 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---


## Job 36969688

Generated: 2026-08-10T01:20:12+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_36969674**

### Plain-language summary

**6 of 10 in the MDE_D x Tp sweep** -- see job_36969445's entry for the full rationale.
MDE_D raised 20 -> 999 (effectively uncapped: `effective_mde_dimension()` clamps this
down to however many real observables each node's own candidate pool actually has, so
this doesn't literally attempt 999 dimensions -- it means "let MDE's own
crossMapRhoMin/embedDimRhoMin/ccmSlope convergence gates decide", per that function's
docstring), Tp reset to 1 -- first of the 5 combos now sweeping Tp again at the
uncapped MDE_D. `--time` raised 3-00:00:00 -> 4-00:00:00 (the partition's stated cap)
for this and the remaining 4 jobs -- removing the MDE_D ceiling is much more open-ended
than the 20-vs-5 step above, so this and the next 4 jobs are budgeted at the max
available rather than a guessed intermediate value.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_36969674 | job_36969688 |
|---|---|---|
| MDE_D | 20 | 999 |
| TP | 30 | 1 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

```diff
--- job_36969674/simple-mde-gmn.sbatch
+++ job_36969688/simple-mde-gmn.sbatch
@@ -1,7 +1,7 @@
 #!/bin/bash
 #SBATCH --job-name=run_simple_mde_gmn
 #SBATCH --partition=compute
-#SBATCH --time=3-00:00:00
+#SBATCH --time=4-00:00:00
 #SBATCH --mem=200G
 #SBATCH --cpus-per-task=32
 # ADJUST: must be a literal absolute path, not $HOME/$PROJ_DIR -- Slurm
```

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---


## Job 36969674

Generated: 2026-08-10T01:19:34+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_36969670**

### Plain-language summary

**5 of 10 in the MDE_D x Tp sweep** -- see job_36969445's entry for the full rationale.
MDE_D=20 (unchanged), Tp raised 20 -> 30 -- last combo at MDE_D=20; the remaining 5
switch MDE_D to 999 (effectively uncapped -- see effective_mde_dimension()'s docstring).

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_36969670 | job_36969674 |
|---|---|---|
| TP | 20 | 30 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---


## Job 36969670

Generated: 2026-08-10T01:19:14+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_36969664**

### Plain-language summary

**4 of 10 in the MDE_D x Tp sweep** -- see job_36969445's entry for the full rationale.
MDE_D=20 (unchanged), Tp raised 10 -> 20.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_36969664 | job_36969670 |
|---|---|---|
| TP | 10 | 20 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---


## Job 36969664

Generated: 2026-08-10T01:18:32+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_36969525**

### Plain-language summary

**3 of 10 in the MDE_D x Tp sweep** -- see job_36969445's entry for the full rationale.
MDE_D=20 (unchanged), Tp raised 5 -> 10.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_36969525 | job_36969664 |
|---|---|---|
| TP | 5 | 10 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---


## Job 36969525

Generated: 2026-08-10T01:18:09+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_36969445**

### Plain-language summary

**2 of 10 in the MDE_D x Tp sweep** -- see job_36969445's entry for the full rationale
(job 36786285's near-zero generative fidelity despite R~0.98 1-step MDE skill, and why
MDE_D and Tp are the two suspects). This job: MDE_D=20 (unchanged), Tp raised 1 -> 5 --
first step into the Tp sweep at the pipeline's now-documented default MDE_D.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_36969445 | job_36969525 |
|---|---|---|
| TP | 1 | 5 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---


## Job 36969445

Generated: 2026-08-10T01:17:14+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_36786285**

### Plain-language summary

**1 of 10 in an MDE_D x Tp sweep, investigating job 36786285's poor generative
performance.** That job (MDE_D=5, Tp=1 -- Tp wasn't even a configurable parameter yet,
just dimx.MDE's own hardcoded default) completed the full pipeline cleanly (the NaN fix
from job 36533849 held), but the actual targets' generative fidelity was essentially
nil: `head_position_x` fidelity rho = 0.0284, `head_position_y` = -0.1269 (and Generate()
rho at the theta-search stage was -0.202 / 0.151) -- this despite the MDE-selected
embedding itself scoring R~0.98 on 1-step-ahead skill (`Eval.Run()`'s Simplex-MDE step:
`CAE 899.02 RMSE 4.832 R 0.983` / `R 0.985`). Two candidate explanations, not mutually
exclusive:

1. **MDE_D=5 is a severe bottleneck.** It caps every node (and both targets) to at most
   5 distinct real observables out of ~460 recorded channels -- confirmed from
   `edges_nestedMDE_*.csv`: in-degree (drivers/node) maxed out at exactly 25 (= 5 x the 5
   DELAY_SET entries) for a large fraction of nodes, meaning MDE_D was the actual
   binding constraint, not MDE's own convergence gates, for much of the network. The
   pipeline's own documented default is 20 -- this run used a quarter of that.
2. **Tp=1 optimizes for the wrong horizon.** Every dimx.MDE forward-selection call (both
   the top-level target search and every nested-MDE per-node search) ranks/validates
   candidate drivers purely on 1-step-ahead cross-map skill (`CrossMapColumns(...,
   Tp=a.Tp, ...)`, dimx's own default `Tp=1`, never overridden anywhere in this
   pipeline before now). Nothing in driver selection ever rewarded stability over the
   300-step recursive `Generate()` run (`GENERATE_LENGTH=300`) fidelity is actually
   scored against -- a classic short-horizon-skill-vs-long-horizon-stability mismatch.

This sweep varies MDE_D in {20, 999} x Tp in {1, 5, 10, 20, 30} (10 jobs total) to see
which knob, if either, actually moves generative fidelity. MDE_D=999 relies on
`effective_mde_dimension()`'s existing clamp-to-candidate-pool-size behavior (see that
function's docstring) to mean "let MDE's own crossMapRhoMin/embedDimRhoMin/ccmSlope
gates decide" rather than literally attempting 999 dimensions. Tp is a newly added
parameter (`MDE_GMN_TP`/`cfg['TP']`, threaded through `run_mde_for_target()`'s
`dx.MDE(..., Tp=tp, ...)` call and every cache fingerprint that depends on it) --
before this job it wasn't configurable at all.

**This job**: MDE_D=20, Tp=1 (Tp at its previous implicit default, MDE_D raised to the
pipeline's own documented default -- the smallest step off job 36786285's config,
isolating MDE_D's effect on its own before combining with a swept Tp).

**Resourcing**: `--time` kept at 3-00:00:00 (job 36786285 took 20h48m at MDE_D=5; MDE_D=20
lets more nodes search deeper before hitting their own convergence gates, so this is a
buffer, not a precise estimate). `--mem`/`--cpus-per-task` left unchanged at 200G/32 --
these are driven by parallel worker count, not search depth, and 200G already has
comfortable headroom over job 36786285's observed ~61GB peak RSS. The MDE_D=999 combos
later in this sweep (jobs 6-10) get `--time=4-00:00:00`, the partition's stated cap --
removing the MDE_D ceiling entirely is much more open-ended, so budgeted at the max
rather than guessed. If any of these 10 still time out, the durable per-stage cache on
bucket storage (see the module docstring's SEARCH REUSE section) means nothing already
completed is lost -- a resubmission would pick up from whatever nodes/networks that run
finished.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_36786285 | job_36969445 |
|---|---|---|
| MDE_D | 5 | 20 |
| TP | _(unset)_ | 1 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

```diff
--- job_36786285/simple-mde-gmn.sbatch
+++ job_36969445/simple-mde-gmn.sbatch
@@ -1,7 +1,7 @@
 #!/bin/bash
 #SBATCH --job-name=run_simple_mde_gmn
 #SBATCH --partition=compute
-#SBATCH --time=2-00:00:00
+#SBATCH --time=3-00:00:00
 #SBATCH --mem=200G
 #SBATCH --cpus-per-task=32
 # ADJUST: must be a literal absolute path, not $HOME/$PROJ_DIR -- Slurm
@@ -205,6 +205,7 @@
 export MDE_GMN_GMN_MODE="${GMN_MODE}"
 export MDE_GMN_TARGETS="${TARGETS}"
 export MDE_GMN_D="${MDE_D}"
+export MDE_GMN_TP="${TP}"
 export MDE_GMN_EMBED_DIM_RHO_MIN="${EMBED_DIM_RHO_MIN}"
 export MDE_GMN_CROSS_MAP_RHO_MIN="${CROSS_MAP_RHO_MIN}"
 export MDE_GMN_EXCLUSION_RADIUS="${EXCLUSION_RADIUS}"
```

### Python pipeline changes (simple-mde-gmn.py)

```diff
--- job_36786285/simple-mde-gmn.py
+++ job_36969445/simple-mde-gmn.py
@@ -923,6 +923,16 @@
     # all of them, capped down rather than inflating D past what the pool
     # can supply.
     cfg['MDE_D'] = int(os.environ.get('MDE_GMN_D', 20))
+    # Prediction interval every dimx.MDE forward-selection call (both the
+    # top-level per-target search and every nested-MDE per-node search --
+    # see run_mde_for_target()) ranks/validates candidate drivers against.
+    # dimx.MDE's own default is 1 -- unset before this parameter existed,
+    # every driver in this pipeline was being selected purely for 1-step-
+    # ahead skill, with nothing in the selection ever rewarding stability
+    # over a long recursive Generate() run. See simple-mde-gmn.sbatch's TP
+    # comment and JOB_LOG.md's job 36786285 diagnosis for the full
+    # rationale.
+    cfg['TP'] = int(os.environ.get('MDE_GMN_TP', 1))
     cfg['EMBED_DIM_RHO_MIN'] = float(os.environ.get('MDE_GMN_EMBED_DIM_RHO_MIN', 0.2))
     cfg['CROSS_MAP_RHO_MIN'] = float(os.environ.get('MDE_GMN_CROSS_MAP_RHO_MIN', 0.2))
     # 'auto' (default): dynamic per-dimx.Run()-call exclusion radius --
@@ -1151,14 +1161,17 @@
 
 def run_mde_for_target(target, df, observable_columns, prediction_start, prediction_length,
                         mde_d, embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
-                        cores, results_dir, delay_set,
+                        cores, results_dir, delay_set, tp=1,
                         reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                         data_file=None, data_file_sha256=None, code_fingerprint=None):
     '''Run dimx.MDE's forward selection for a single target variable.
     `mde_d` here is already the *effective* dimension count (see
     effective_mde_dimension()) -- callers compute that from the
     real-observable-count MDE_D before calling this, since only they know
-    each call's own candidate pool / DELAY_SET. `exclusion_radius` is
+    each call's own candidate pool / DELAY_SET. `tp` (default 1, matching
+    dimx.MDE's own default) is the prediction interval every candidate
+    driver is ranked/validated against -- see load_config()'s TP
+    docstring for why this is swept alongside MDE_D. `exclusion_radius` is
     only actually used as-is when MDE_GMN_EXCLUSION_RADIUS is a fixed
     integer -- when it's 'auto' (the default), the module-level
     _EXCLUSION_RADIUS_AUTO-gated monkeypatches (see that flag's docstring
@@ -1198,7 +1211,7 @@
     cache_filename = f'mde_selection_{target}.pkl'
     if reuse_cache:
         fingerprint = build_mde_selection_fingerprint(
-            target, observable_columns, mde_d, embed_dim_rho_min, cross_map_rho_min,
+            target, observable_columns, mde_d, tp, embed_dim_rho_min, cross_map_rho_min,
             exclusion_radius, delay_set, prediction_start, prediction_length,
             data_file, data_file_sha256, code_fingerprint,
         )
@@ -1229,6 +1242,7 @@
                 removeColumns = [target],
                 lib  = [warmup + 1, prediction_start],
                 pred = [prediction_start + 1, prediction_start + prediction_length],
+                Tp = tp,
                 embedDimRhoMin = embed_dim_rho_min,
                 crossMapRhoMin = cross_map_rho_min,
                 exclusionRadius = exclusion_radius,
@@ -1359,7 +1373,7 @@
 def build_nested_mde_network(targets, expanded_df, raw_columns, expanded_columns,
                               prediction_start, prediction_length, nested_n, nested_m,
                               embed_dim_rho_min, cross_map_rho_min, exclusion_radius, cores,
-                              results_dir, mde_results, delay_set,
+                              results_dir, mde_results, delay_set, tp=1,
                               reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                               data_file=None, data_file_sha256=None, code_fingerprint=None):
     '''Nested-MDE network construction -- MODE='MDE-GMN' only. MODE='GMN'
@@ -1570,7 +1584,7 @@
                         run_mde_for_target, node, expanded_df, candidate_pool,
                         prediction_start, prediction_length, effective_d,
                         embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
-                        inner_cores, results_dir, delay_set,
+                        inner_cores, results_dir, delay_set, tp,
                         reuse_cache, prev_runs_dir, run_cache_dir, data_file, data_file_sha256,
                         code_fingerprint,
                     )
@@ -1643,7 +1657,7 @@
                    mode, e_range, tau_range, results_dir, cores, interaction_matrix_outer_cores,
                    observable_columns=None, nested_n=None, nested_m=None,
                    embed_dim_rho_min=None, cross_map_rho_min=None, exclusion_radius=None,
-                   expanded_df=None, expanded_observable_columns=None, delay_set=None,
+                   expanded_df=None, expanded_observable_columns=None, delay_set=None, tp=1,
                    threshold=0, fallback_node_dimension=4, max_node_dimension=15,
                    reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                    data_file=None, data_file_sha256=None, code_fingerprint=None):
@@ -1732,7 +1746,7 @@
     cache_hit = None
     if reuse_cache:
         fingerprint = build_network_fingerprint(
-            mode, targets, observable_columns, nested_n, nested_m,
+            mode, targets, observable_columns, nested_n, nested_m, tp,
             embed_dim_rho_min, cross_map_rho_min, exclusion_radius, delay_set,
             e_range, tau_range, prediction_start, prediction_length,
             data_file, data_file_sha256, code_fingerprint,
@@ -1754,7 +1768,7 @@
                 targets, expanded_df, observable_columns, expanded_observable_columns,
                 prediction_start, prediction_length,
                 nested_n, nested_m, embed_dim_rho_min, cross_map_rho_min, exclusion_radius, cores,
-                results_dir, mde_results, delay_set,
+                results_dir, mde_results, delay_set, tp,
                 reuse_cache, prev_runs_dir, run_cache_dir, data_file, data_file_sha256,
                 code_fingerprint,
             )
@@ -2342,7 +2356,7 @@
     }
 
 
-def build_network_fingerprint(mode, targets, observable_columns, nested_n, nested_m,
+def build_network_fingerprint(mode, targets, observable_columns, nested_n, nested_m, tp,
                                embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                                delay_set, e_range, tau_range, prediction_start,
                                prediction_length, data_file, data_file_sha256, code_fingerprint):
@@ -2368,6 +2382,7 @@
         'observable_columns': sorted(observable_columns),
         'mde_d': nested_n,
         'nested_m': nested_m,
+        'tp': tp,
         'embed_dim_rho_min': embed_dim_rho_min,
         'cross_map_rho_min': cross_map_rho_min,
         'exclusion_radius': exclusion_radius,
@@ -2383,7 +2398,7 @@
     }
 
 
-def build_mde_selection_fingerprint(node, candidate_pool, mde_d, embed_dim_rho_min,
+def build_mde_selection_fingerprint(node, candidate_pool, mde_d, tp, embed_dim_rho_min,
                                      cross_map_rho_min, exclusion_radius, delay_set,
                                      prediction_start, prediction_length,
                                      data_file, data_file_sha256, code_fingerprint):
@@ -2402,6 +2417,7 @@
         'node': node,
         'candidate_pool': list(candidate_pool),
         'mde_d': mde_d,
+        'tp': tp,
         'embed_dim_rho_min': embed_dim_rho_min,
         'cross_map_rho_min': cross_map_rho_min,
         'exclusion_radius': exclusion_radius,
@@ -3317,6 +3333,7 @@
                 target, expanded_df, candidate_pool, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                 effective_d, cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
                 cfg['EXCLUSION_RADIUS_VALUE'], cfg['CORES'], cfg['RESULTS_DIR'], cfg['DELAY_SET'],
+                tp=cfg['TP'],
                 reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
                 run_cache_dir=cfg['RUN_CACHE_DIR'],
                 data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
@@ -3350,7 +3367,7 @@
             embed_dim_rho_min=cfg['EMBED_DIM_RHO_MIN'], cross_map_rho_min=cfg['CROSS_MAP_RHO_MIN'],
             exclusion_radius=cfg['EXCLUSION_RADIUS_VALUE'],
             expanded_df=expanded_df, expanded_observable_columns=expanded_obs_cols,
-            delay_set=cfg['DELAY_SET'],
+            delay_set=cfg['DELAY_SET'], tp=cfg['TP'],
             reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
             run_cache_dir=cfg['RUN_CACHE_DIR'],
             data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
```

---


## Job 36786285

Generated: 2026-08-08T23:57:57+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_36533849**

### Plain-language summary

Resubmission of job 36533849 after fixing the bug that killed it: that job ran 23h14m,
completed MDE selection, nested-MDE network construction, theta search, and Generate()/
Forecast()/fidelity for both targets, then crashed in the very last stage
(`run_mde_evaluate`'s PCA step) on `ValueError: Input X contains NaN` -- delay-expanded
columns (e.g. `CA1_137::delay-10`) carry NaN in their first `abs(delay)` rows, and the
MDE library window started at row 1, right through that warm-up. Fixed in
`run_mde_for_target()` by starting `lib` after `max(abs(d) for d in delay_set)` rows
instead.

Also adds a durable cross-job cache (see the module docstring's expanded SEARCH REUSE
section): MDE forward-selection, network construction, theta search, and the
generate-start sweep now each fingerprint their own inputs -- config knobs, a hash of
the data file, and a code+environment fingerprint (this file's own AST with docstrings
stripped, plus installed dimx/gmn/pyEDM/etc. versions) -- and sync their result to
`/bucket/FukaiU/tominaga/pao_rotation/mde_gmn_cache/job_<id>/` immediately on
completing, not just at job end. Since job 36533849 never reached a successful exit,
none of its own (real) work ever reached persistent storage even for the one stage
(theta search) that already had a cache mechanism before this change -- this job is a
cold start that will populate the bucket cache pool for the first time. No config/param
changes from job 36533849 -- same DELAY_SET/MDE_D/NESTED_M/targets/etc., so this is a
direct rerun under the fixed code.

### Parameter changes (simple-mde-gmn.sbatch)

_No parameter changes._

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

```diff
--- job_36533849/simple-mde-gmn.sbatch
+++ job_36786285/simple-mde-gmn.sbatch
@@ -86,6 +86,23 @@
 OUT_DIR="$PROJ_DIR/results/job_${SLURM_JOB_ID}"
 mkdir -p "$OUT_DIR"
 
+# ADJUST: durable cross-job cache pool for expensive, fingerprint-reusable
+# pipeline stages (MDE forward-selection, nested-MDE/CCM network
+# construction, theta search, generate-start sweep -- see the module
+# docstring's SEARCH REUSE section and each stage's own docstring).
+# Deliberately NOT under $PROJ_DIR/results (/home) -- that quota is meant
+# for this project's own human-facing per-job output, not a growing pool
+# of machine-only cache artifacts across every run ever submitted. Bucket
+# storage has the headroom for that instead. CACHE_ROOT is the read side
+# (searched recursively, across every past job's subdirectory, by
+# find_cached_best_params()-style lookups); RUN_CACHE_DIR is this job's
+# own write side -- each cacheable stage syncs its result here
+# *immediately* upon completing (not just at job end like $OUT_DIR),
+# so it survives even if this job later crashes or times out.
+CACHE_ROOT="/bucket/FukaiU/tominaga/pao_rotation/mde_gmn_cache"
+RUN_CACHE_DIR="$CACHE_ROOT/job_${SLURM_JOB_ID}"
+mkdir -p "$RUN_CACHE_DIR"
+
 # ==============================================================================
 # SLURM LOG RELOCATION
 # ==============================================================================
@@ -212,10 +229,12 @@
 export MDE_GMN_REUSE_BEST_PARAMS="${REUSE_BEST_PARAMS}"
 # Unlike MDE_GMN_RESULTS_DIR (pointed at $tempdir on /flash scratch,
 # deleted at job exit -- see that export above / the /flash scratch
-# setup section's own comment), this must point at the persistent
-# per-job output location so a later job can actually find a past run's
-# best_params_<network_id>.json to reuse.
-export MDE_GMN_PREV_RUNS_DIR="$PROJ_DIR/results"
+# setup section's own comment), these must point at the durable bucket
+# cache pool set up above so a later job can actually find a past run's
+# cached results to reuse, and so this run's own results survive even a
+# crash partway through (see CACHE_ROOT/RUN_CACHE_DIR's own comment).
+export MDE_GMN_PREV_RUNS_DIR="$CACHE_ROOT"
+export MDE_GMN_RUN_CACHE_DIR="$RUN_CACHE_DIR"
 
 python3 -V
 
```

### Python pipeline changes (simple-mde-gmn.py)

```diff
--- job_36533849/simple-mde-gmn.py
+++ job_36786285/simple-mde-gmn.py
@@ -81,30 +81,55 @@
 outer one provides parallelism -- see that function's and
 build_network()'s docstrings for the full evidence trail.
 
-SEARCH REUSE (MDE_GMN_REUSE_BEST_PARAMS, default off): the theta search
-(step 4 above) is deterministic given the same network and data, so
-resubmitting identical settings pays for it twice for no reason. When
-enabled, main() builds a fingerprint per network (every config knob that
-shapes its topology/search space, a hash of the input data file, and its
-own realized node set) and looks for a past job's
-best_params_<network_id>.json under MDE_GMN_PREV_RUNS_DIR with a
-matching fingerprint -- a hit skips that network's theta retuning and
-reuses its cached per-node theta; any mismatch (different dataset,
-different MDE_D/NESTED_M/REGION_CHANNEL_CAP/DELAY_SET/etc., or the first
-run with these settings) just runs the real search, so this is always a
-safe, redundant-compute optimization, never a correctness shortcut. See
-build_search_fingerprint() and find_cached_best_params(). Before this
-pipeline's generalized (per-driver-delay) embedding existed, this same
-mechanism also cached a searched network-wide (E, tau) -- see
+SEARCH REUSE (MDE_GMN_REUSE_BEST_PARAMS, default off): every genuinely
+expensive stage -- MDE forward-selection (run_mde_for_target(), step 1),
+network construction (build_network()/build_nested_mde_network(),
+step 2), theta search (run_theta_search(), step 4), and the generate-
+start sweep (sweep_generate_start(), step 5) -- is deterministic given
+the same inputs, so resubmitting identical settings pays for all of them
+twice for no reason. When enabled, each stage builds its own fingerprint
+(every config knob that shapes its outcome, plus a hash of the input data
+file -- build_search_fingerprint()/build_network_fingerprint()/
+build_mde_selection_fingerprint()/build_start_sweep_fingerprint()) and
+looks for a past job's matching cache file (best_params_<network_id>.json
+/ network_cache_<network_id>.pkl / mde_selection_<node>.pkl /
+generate_start_sweep_<network_id>.json) under MDE_GMN_PREV_RUNS_DIR via
+find_cached() -- a hit skips that stage's own expensive work entirely and
+reuses the cached result; any mismatch (different dataset, different
+MDE_D/NESTED_M/REGION_CHANNEL_CAP/DELAY_SET/etc., or the first run with
+these settings) just runs the real computation, so this is always a safe,
+redundant-compute optimization, never a correctness shortcut.
+
+MDE_GMN_PREV_RUNS_DIR/MDE_GMN_RUN_CACHE_DIR deliberately point at a
+durable bucket-storage pool (see simple-mde-gmn.sbatch's CACHE_ROOT/
+RUN_CACHE_DIR comment), not RESULTS_DIR -- RESULTS_DIR lives on a /flash
+scratch tempdir deleted the instant this script exits, including on a
+crash, before its contents ever reach persistent storage (see that
+script's /flash scratch setup section). Every save_json_to_cache()/
+save_pickle_to_cache() call therefore writes to RUN_CACHE_DIR
+*immediately*, right after its stage finishes, in addition to the
+ordinary results_dir write -- so a stage's result is durable the moment
+it completes, not contingent on the whole job finishing successfully.
+This is what actually fixes a 23-hour job losing everything to a crash in
+its very last step (see JOB_LOG.md job 36533849): before this, only
+theta search had a cache at all, and even that cache's write only ever
+reached persistent storage via RESULTS_DIR's own end-of-job copy, so it
+never survived a crash either.
+
+Before this pipeline's generalized (per-driver-delay) embedding existed,
+this same mechanism also cached a searched network-wide (E, tau) -- see
 scripts/backup/simple-mde-gmn_pre-delay-embedding.py for that version,
 kept for reference/comparison per explicit direction.
 '''
+import ast
 import hashlib
+import importlib.metadata
 import json
 import multiprocessing as mp
 import os
 import pickle
 import subprocess
+import types
 from concurrent.futures import ProcessPoolExecutor, as_completed
 from configparser import ConfigParser
 from pathlib import Path
@@ -1010,13 +1035,22 @@
     # rationale; short version: reuse a past job's E/tau/theta search
     # result instead of redoing it, when settings match exactly.
     cfg['REUSE_BEST_PARAMS'] = _env_bool('MDE_GMN_REUSE_BEST_PARAMS', False)
-    # Where to look for previous jobs' best_params_*.json files. Default
-    # mirrors DATA_DIR/RESULTS_DIR's own '../../data'-style fallback, but
-    # the sbatch script overrides this to the persistent $PROJ_DIR/results
-    # -- unlike RESULTS_DIR, which the sbatch script points at a /flash
-    # scratch tempdir deleted at job exit, PREV_RUNS_DIR needs to point at
-    # wherever past jobs' outputs actually still exist.
+    # Where to look for previous jobs' cache artifacts (best_params_*.json,
+    # mde_selection_*.pkl, Network_*.pkl, generate_start_*.json -- see the
+    # module docstring's SEARCH REUSE section). Default mirrors DATA_DIR/
+    # RESULTS_DIR's own '../../data'-style fallback, but the sbatch script
+    # overrides this to the durable bucket cache pool -- unlike RESULTS_DIR
+    # (a /flash scratch tempdir deleted at job exit), PREV_RUNS_DIR needs
+    # to point at wherever past jobs' cache output actually still exists.
     cfg['PREV_RUNS_DIR'] = Path(os.environ.get('MDE_GMN_PREV_RUNS_DIR', '../../results'))
+    # This job's own write destination for those same cache artifacts,
+    # synced immediately after each cacheable stage completes (not just at
+    # job end like RESULTS_DIR->$OUT_DIR) -- see save_to_cache()'s
+    # docstring. Falls back to RESULTS_DIR (no eager off-scratch sync) for
+    # ad hoc/local runs outside the sbatch script, where durability across
+    # a crash isn't a concern.
+    cfg['RUN_CACHE_DIR'] = Path(os.environ.get('MDE_GMN_RUN_CACHE_DIR', cfg['RESULTS_DIR']))
+    cfg['RUN_CACHE_DIR'].mkdir(parents=True, exist_ok=True)
 
     cfg['TARGET_COLORS'] = {t: plt.get_cmap('tab10')(i % 10) for i, t in enumerate(cfg['TARGETS'])}
 
@@ -1117,7 +1151,9 @@
 
 def run_mde_for_target(target, df, observable_columns, prediction_start, prediction_length,
                         mde_d, embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
-                        cores, results_dir):
+                        cores, results_dir, delay_set,
+                        reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
+                        data_file=None, data_file_sha256=None, code_fingerprint=None):
     '''Run dimx.MDE's forward selection for a single target variable.
     `mde_d` here is already the *effective* dimension count (see
     effective_mde_dimension()) -- callers compute that from the
@@ -1132,12 +1168,66 @@
     harmless placeholder (0) in that case. See
     notebooks/simple-mde-gmn.ipynb cell 2's run_mde_for_target for the
     original single-delay rationale (identical logic otherwise, just
-    cores threaded through).'''
+    cores threaded through).
+
+    CACHE (reuse_cache, gated by MDE_GMN_REUSE_BEST_PARAMS same as every
+    other stage -- see the module docstring's SEARCH REUSE section):
+    every downstream consumer of the returned 'm' object only ever reads
+    m.MDEcolumns / m.MDErho / m.MDEOut / m.args.lib / m.args.pred /
+    m.args.tau / m.args.Tp (confirmed by exhaustive search across this
+    file) -- never anything else, never a second .Run(). So a cache hit
+    reconstructs a types.SimpleNamespace exposing exactly those seven
+    attributes instead of a live dx.MDE object, and skips calling
+    dx.MDE(...).Run() entirely. Called both from main()'s top-level
+    per-target loop and from every per-node wave inside
+    build_nested_mde_network()'s ProcessPoolExecutor, so caching here
+    covers both at full per-node granularity for free.'''
     mde_df = df[[target] + observable_columns].iloc[:prediction_start + prediction_length]
 
+    # expand_delay_columns() shifts each observable forward by abs(delay)
+    # rows, leaving that many leading rows NaN (nothing precedes the start
+    # of the recording to shift in). Starting `lib` at row 1 regardless
+    # used to let those NaNs into the library window -- harmless for MDE's
+    # own NaN-tolerant pyEDM calls, but fatal once mde_df's library slice
+    # reaches run_mde_evaluate()'s PCA fit (sklearn has zero NaN
+    # tolerance). Skipping the warm-up rows keeps every delay column fully
+    # populated for the entire library range.
+    warmup = max((abs(d) for d in delay_set), default=0)
+
+    fingerprint = None
+    cache_filename = f'mde_selection_{target}.pkl'
+    if reuse_cache:
+        fingerprint = build_mde_selection_fingerprint(
+            target, observable_columns, mde_d, embed_dim_rho_min, cross_map_rho_min,
+            exclusion_radius, delay_set, prediction_start, prediction_length,
+            data_file, data_file_sha256, code_fingerprint,
+        )
+        hit = find_cached(prev_runs_dir, cache_filename, fingerprint,
+                           load_one=_load_pickle_cache_entry)
+        if hit is not None:
+            entry, source = hit
+            print(f'{target}: MDE_GMN_REUSE_BEST_PARAMS matched cached MDE selection '
+                  f'{source} -- skipping forward-selection entirely', flush=True)
+            m = types.SimpleNamespace(
+                MDEcolumns=entry['MDEcolumns'], MDErho=entry['MDErho'], MDEOut=entry['MDEOut'],
+                args=types.SimpleNamespace(lib=entry['lib'], pred=entry['pred'],
+                                            tau=entry['tau'], Tp=entry['Tp']),
+            )
+            m.MDEOut.to_csv(results_dir / f'mde_output_{target}.csv', index=False)
+            selected_columns = list(m.MDEcolumns)
+            print(f'MDE selected columns for {target}:', selected_columns, flush=True)
+            return {
+                'target': target,
+                'mde_df': mde_df,
+                'm': m,
+                'SELECTED_COLUMNS': selected_columns,
+            }
+        print(f'{target}: MDE_GMN_REUSE_BEST_PARAMS set but no matching cached MDE selection '
+              f'found under {prev_runs_dir} -- running forward-selection.', flush=True)
+
     m = dx.MDE( mde_df, target = target, D = mde_d,
                 removeColumns = [target],
-                lib  = [1, prediction_start],
+                lib  = [warmup + 1, prediction_start],
                 pred = [prediction_start + 1, prediction_start + prediction_length],
                 embedDimRhoMin = embed_dim_rho_min,
                 crossMapRhoMin = cross_map_rho_min,
@@ -1151,6 +1241,17 @@
     selected_columns = list(m.MDEcolumns)
     print(f'MDE selected columns for {target}:', selected_columns, flush=True)
 
+    if fingerprint is not None:
+        save_pickle_to_cache(
+            cache_filename,
+            {
+                'fingerprint': fingerprint,
+                'MDEcolumns': m.MDEcolumns, 'MDErho': m.MDErho, 'MDEOut': m.MDEOut,
+                'lib': m.args.lib, 'pred': m.args.pred, 'tau': m.args.tau, 'Tp': m.args.Tp,
+            },
+            results_dir, run_cache_dir,
+        )
+
     return {
         'target': target,
         'mde_df': mde_df,
@@ -1258,7 +1359,9 @@
 def build_nested_mde_network(targets, expanded_df, raw_columns, expanded_columns,
                               prediction_start, prediction_length, nested_n, nested_m,
                               embed_dim_rho_min, cross_map_rho_min, exclusion_radius, cores,
-                              results_dir, mde_results):
+                              results_dir, mde_results, delay_set,
+                              reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
+                              data_file=None, data_file_sha256=None, code_fingerprint=None):
     '''Nested-MDE network construction -- MODE='MDE-GMN' only. MODE='GMN'
     still uses the original CCM-across-the-full-observable-pool path in
     build_network(), untouched.
@@ -1467,7 +1570,9 @@
                         run_mde_for_target, node, expanded_df, candidate_pool,
                         prediction_start, prediction_length, effective_d,
                         embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
-                        inner_cores, results_dir,
+                        inner_cores, results_dir, delay_set,
+                        reuse_cache, prev_runs_dir, run_cache_dir, data_file, data_file_sha256,
+                        code_fingerprint,
                     )
                     futures[fut] = node
                 for fut in as_completed(futures):
@@ -1539,7 +1644,9 @@
                    observable_columns=None, nested_n=None, nested_m=None,
                    embed_dim_rho_min=None, cross_map_rho_min=None, exclusion_radius=None,
                    expanded_df=None, expanded_observable_columns=None, delay_set=None,
-                   threshold=0, fallback_node_dimension=4, max_node_dimension=15):
+                   threshold=0, fallback_node_dimension=4, max_node_dimension=15,
+                   reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
+                   data_file=None, data_file_sha256=None, code_fingerprint=None):
     '''Build one network spanning `targets` and their driver candidates,
     via one of two unrelated constructions depending on `mode`:
       - mode == 'MDE-GMN': build_nested_mde_network()'s own graph IS the
@@ -1610,13 +1717,47 @@
     ccm_E = ccm_tau = None
     node_dimensions = {}
 
-    if mode == 'MDE-GMN':
-        graph, edges_df, _ = build_nested_mde_network(
-            targets, expanded_df, observable_columns, expanded_observable_columns,
-            prediction_start, prediction_length,
-            nested_n, nested_m, embed_dim_rho_min, cross_map_rho_min, exclusion_radius, cores,
-            results_dir, mde_results,
+    # CACHE (reuse_cache, gated by MDE_GMN_REUSE_BEST_PARAMS same as every
+    # other stage -- see the module docstring's SEARCH REUSE section):
+    # this is the single most expensive stage in the whole pipeline
+    # (build_nested_mde_network()'s BFS-over-hundreds-of-nodes for
+    # MODE='MDE-GMN', or the apps.InteractionMatrix CCM subprocess for
+    # MODE='GMN'), so a fingerprint match here skips it entirely rather
+    # than just resuming partway through. Computed *before* construction
+    # (unlike build_search_fingerprint(), which needs the already-built
+    # network for its node-set sanity check) since the whole point is
+    # deciding whether to build it at all.
+    fingerprint = None
+    cache_filename = f'network_cache_{network_id}.pkl'
+    cache_hit = None
+    if reuse_cache:
+        fingerprint = build_network_fingerprint(
+            mode, targets, observable_columns, nested_n, nested_m,
+            embed_dim_rho_min, cross_map_rho_min, exclusion_radius, delay_set,
+            e_range, tau_range, prediction_start, prediction_length,
+            data_file, data_file_sha256, code_fingerprint,
         )
+        hit = find_cached(prev_runs_dir, cache_filename, fingerprint, load_one=_load_pickle_cache_entry)
+        if hit is not None:
+            cache_hit, source = hit
+            print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS matched cached network {source} '
+                  f'-- skipping network construction entirely', flush=True)
+        else:
+            print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS set but no matching cached network '
+                  f'found under {prev_runs_dir} -- building from scratch.', flush=True)
+
+    if mode == 'MDE-GMN':
+        if cache_hit is not None:
+            graph = cache_hit['graph']
+        else:
+            graph, _, _ = build_nested_mde_network(
+                targets, expanded_df, observable_columns, expanded_observable_columns,
+                prediction_start, prediction_length,
+                nested_n, nested_m, embed_dim_rho_min, cross_map_rho_min, exclusion_radius, cores,
+                results_dir, mde_results, delay_set,
+                reuse_cache, prev_runs_dir, run_cache_dir, data_file, data_file_sha256,
+                code_fingerprint,
+            )
         # 0 is forced into self_delay_set regardless of DELAY_SET: gmn's
         # own SMap target-lookup convention (Parameters.target + '(t-0)',
         # mirrored by this pipeline's _generalized_node_forecast/Generate)
@@ -1626,9 +1767,22 @@
         # configures a DELAY_SET without 0 in it.
         graph.graph['self_delay_set'] = sorted(set(delay_set) | {0}, reverse=True)
         selected_union = [n for n in graph.nodes if n not in targets]
+        # Rebuilt from the graph itself (rather than build_nested_mde_
+        # network()'s own returned edges_df) so this works identically on
+        # both the cache-hit and freshly-built path -- every edge that
+        # function's own BFS discovers and keeps is exactly what ends up
+        # in `graph` (cycle-rejected edges are never added in the first
+        # place), so the two are equivalent in content either way.
         edges_file = results_dir / f'edges_nestedMDE_N{nested_n}_M{nested_m}_{network_id}.csv'
-        edges_df.to_csv(edges_file, index=False)
+        edges_rows = [{'driver': u, 'node': v, 'delay': d['delay'], 'weight': d['weight']}
+                      for u, v, d in graph.edges(data=True)]
+        pd.DataFrame(edges_rows, columns=['driver', 'node', 'delay', 'weight']).to_csv(
+            edges_file, index=False)
     else:  # mode == 'GMN'
+        if cache_hit is not None:
+            network = cache_hit['network']
+            ccm_E, ccm_tau, node_dimensions = (
+                cache_hit['ccm_E'], cache_hit['ccm_tau'], cache_hit['node_dimensions'])
         selected_union = []
         for t in targets:
             for c in mde_results[t]['SELECTED_COLUMNS']:
@@ -1653,84 +1807,104 @@
         # see this function's docstring for why apps.CreateNetwork() adds
         # nothing here. 'Map' is written only because CreateNetwork's own
         # pickle format includes it and downstream code below prints from
-        # it; gmn itself never reads it (see docstring).
+        # it; gmn itself never reads it (see docstring). Rebuilt from
+        # `graph` unconditionally (not skipped on a cache hit) since it's
+        # a trivial dict comprehension -- not worth a second code path.
         network_dict = {node: list(graph.predecessors(node)) for node in graph.nodes}
         network = {'Graph': graph, 'Map': network_dict}
         network_file = results_dir / f'Network_nestedMDE_N{nested_n}_M{nested_m}_{network_id}.pkl'
         with open(network_file, 'wb') as f:
             pickle.dump(network, f)
+
+        if fingerprint is not None and cache_hit is None:
+            save_pickle_to_cache(cache_filename, {'fingerprint': fingerprint, 'graph': graph},
+                                  results_dir, run_cache_dir)
     else:  # mode == 'GMN'
-        e_vals, tau_vals = list(e_range), list(tau_range)
-        target_grids = [target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals) for t in targets]
-        mean_grid = np.nanmean(np.stack(target_grids), axis=0)
-        best_i, best_j = np.unravel_index(np.nanargmax(mean_grid), mean_grid.shape)
-        ccm_E, ccm_tau = e_vals[best_i], tau_vals[best_j]
-        print(f'{network_id}: CCM E={ccm_E}, tau={ccm_tau} '
-              f'(grid-search rho={mean_grid[best_i, best_j]:.3f})', flush=True)
-
-        interaction_prefix = results_dir / f'iMatrix_E{ccm_E}_tau{ccm_tau}_{network_id}'
-        # apps.InteractionMatrix is a CLI-only tool (no in-memory API) --
-        # invoked via subprocess instead of the notebook's IPython
-        # get_ipython().system(), which only exists inside a live kernel.
-        # MPLBACKEND=Agg keeps -P from trying to open a GUI window.
-        subprocess_env = dict(os.environ, MPLBACKEND='Agg')
-        subprocess.run(
-            [
-                'python', '-m', 'apps.InteractionMatrix',
-                '-d', str(data_after_mde_lib_file),
-                '-oc', str(interaction_prefix),
-                '-ccm', '-E', str(ccm_E), '-t', str(ccm_tau),
-                '-l', '1', str(prediction_start), '-p', '1', str(prediction_start),
-                '-P', '-cr', str(interaction_matrix_outer_cores),
-            ],
-            check=True, env=subprocess_env,
-        )
+        if cache_hit is None:
+            e_vals, tau_vals = list(e_range), list(tau_range)
+            target_grids = [target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals)
+                             for t in targets]
+            mean_grid = np.nanmean(np.stack(target_grids), axis=0)
+            best_i, best_j = np.unravel_index(np.nanargmax(mean_grid), mean_grid.shape)
+            ccm_E, ccm_tau = e_vals[best_i], tau_vals[best_j]
+            print(f'{network_id}: CCM E={ccm_E}, tau={ccm_tau} '
+                  f'(grid-search rho={mean_grid[best_i, best_j]:.3f})', flush=True)
+
+            interaction_prefix = results_dir / f'iMatrix_E{ccm_E}_tau{ccm_tau}_{network_id}'
+            # apps.InteractionMatrix is a CLI-only tool (no in-memory API)
+            # -- invoked via subprocess instead of the notebook's IPython
+            # get_ipython().system(), which only exists inside a live
+            # kernel. MPLBACKEND=Agg keeps -P from trying to open a GUI
+            # window.
+            subprocess_env = dict(os.environ, MPLBACKEND='Agg')
+            subprocess.run(
+                [
+                    'python', '-m', 'apps.InteractionMatrix',
+                    '-d', str(data_after_mde_lib_file),
+                    '-oc', str(interaction_prefix),
+                    '-ccm', '-E', str(ccm_E), '-t', str(ccm_tau),
+                    '-l', '1', str(prediction_start), '-p', '1', str(prediction_start),
+                    '-P', '-cr', str(interaction_matrix_outer_cores),
+                ],
+                check=True, env=subprocess_env,
+            )
 
-        interaction_csv = Path(f'{interaction_prefix}_CCM.csv')
-        iMatrix = pd.read_csv(interaction_csv, index_col=0)
+            interaction_csv = Path(f'{interaction_prefix}_CCM.csv')
+            iMatrix = pd.read_csv(interaction_csv, index_col=0)
 
-        node_dimensions = {
-            node: estimate_node_dimension(data_after_mde_lib, node, cores,
-                                           max_e=max_node_dimension, fallback=fallback_node_dimension)
-            for node in iMatrix.index
-        }
-        print(f'{network_id}: node N_D =', node_dimensions, flush=True)
+            node_dimensions = {
+                node: estimate_node_dimension(data_after_mde_lib, node, cores,
+                                               max_e=max_node_dimension, fallback=fallback_node_dimension)
+                for node in iMatrix.index
+            }
+            print(f'{network_id}: node N_D =', node_dimensions, flush=True)
+
+            # apps.CreateNetwork() keeps each node's *weakest* above-
+            # threshold drivers instead of the strongest (confirmed
+            # against the installed apps/CreateNetwork.py:110-116 -- it
+            # sorts each row ascending, then truncates to numDrivers).
+            # Worked around by pre-selecting the strongest N_D drivers
+            # ourselves, then calling CreateNetwork with numDrivers set to
+            # the full column count so its buggy truncation never has
+            # anything left to remove.
+            strongest_iMatrix = iMatrix.copy()
+            for node in strongest_iMatrix.index:
+                row = strongest_iMatrix.loc[node].drop(node)
+                n_drivers = max(1, min(node_dimensions[node], len(row)))
+                keep = row[row > threshold].sort_values(ascending=False).index[:n_drivers]
+                drop = row.index.difference(keep)
+                strongest_iMatrix.loc[node, drop] = threshold
+
+            network_file = results_dir / f'Network_E{ccm_E}_tau{ccm_tau}_CCM_{network_id}.pkl'
+            network = CreateNetwork(
+                interactionMatrix = strongest_iMatrix,
+                targetCols        = list(targets),
+                threshold         = threshold,
+                numDrivers        = len(strongest_iMatrix.columns),
+                outputFile        = str(network_file),
+                plotNetwork       = False,
+            )
+        else:
+            network_file = results_dir / f'Network_E{ccm_E}_tau{ccm_tau}_CCM_{network_id}.pkl'
 
-        # apps.CreateNetwork() keeps each node's *weakest* above-threshold
-        # drivers instead of the strongest (confirmed against the installed
-        # apps/CreateNetwork.py:110-116 -- it sorts each row ascending, then
-        # truncates to numDrivers). Worked around by pre-selecting the
-        # strongest N_D drivers ourselves, then calling CreateNetwork with
-        # numDrivers set to the full column count so its buggy truncation
-        # never has anything left to remove.
-        strongest_iMatrix = iMatrix.copy()
-        for node in strongest_iMatrix.index:
-            row = strongest_iMatrix.loc[node].drop(node)
-            n_drivers = max(1, min(node_dimensions[node], len(row)))
-            keep = row[row > threshold].sort_values(ascending=False).index[:n_drivers]
-            drop = row.index.difference(keep)
-            strongest_iMatrix.loc[node, drop] = threshold
-
-        network_file = results_dir / f'Network_E{ccm_E}_tau{ccm_tau}_CCM_{network_id}.pkl'
-        network = CreateNetwork(
-            interactionMatrix = strongest_iMatrix,
-            targetCols        = list(targets),
-            threshold         = threshold,
-            numDrivers        = len(strongest_iMatrix.columns),
-            outputFile        = str(network_file),
-            plotNetwork       = False,
-        )
         # gmn.Network.__init__ reloads the graph fresh from network_file's
         # own pickle (gmn/Network.py:43-46), not from this in-memory
-        # `network` dict -- CreateNetwork already wrote that file above,
-        # so fallback_E/fallback_tau have to be added to the graph and
-        # the file re-pickled here, or _node_embedding_spec()'s plain-
-        # DiGraph fallback (see its docstring) would never see them.
+        # `network` dict -- CreateNetwork already wrote that file above
+        # (fresh path only), so fallback_E/fallback_tau have to be added
+        # to the graph and the file (re-)pickled here regardless of
+        # cache-hit/fresh, or _node_embedding_spec()'s plain-DiGraph
+        # fallback (see its docstring) would never see them.
         network['Graph'].graph['fallback_E'] = ccm_E
         network['Graph'].graph['fallback_tau'] = ccm_tau
         with open(network_file, 'wb') as f:
             pickle.dump(network, f)
 
+        if fingerprint is not None and cache_hit is None:
+            save_pickle_to_cache(cache_filename, {
+                'fingerprint': fingerprint, 'network': network,
+                'ccm_E': ccm_E, 'ccm_tau': ccm_tau, 'node_dimensions': node_dimensions,
+            }, results_dir, run_cache_dir)
+
     for t in targets:
         print(f'{t} parents:', network['Map'][t], flush=True)
 
@@ -2036,7 +2210,88 @@
     return h.hexdigest()
 
 
-def build_search_fingerprint(cfg, net_result, data_file_sha256):
+def _strip_docstrings(tree):
+    '''Removes every module/class/function docstring (the first statement
+    of a body, when it's a bare string-literal expression) from an AST in
+    place. Used by _pipeline_code_fingerprint() below so that hash is
+    insensitive to prose -- by far the most common kind of edit to this
+    file (nearly every function here carries a long "why" docstring that
+    gets revised/expanded on its own, unrelated to any change in what the
+    function actually computes) -- while still changing on any real
+    change to a function's statements/expressions. Comments need no
+    equivalent handling: they're never part of the AST at all.'''
+    for node in ast.walk(tree):
+        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
+            body = node.body
+            if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
+                    and isinstance(body[0].value.value, str)):
+                body.pop(0)
+    return tree
+
+
+def _pipeline_code_fingerprint():
+    '''SHA-256 of this file's own normalized AST (docstrings stripped via
+    _strip_docstrings(); comments already excluded from the AST). Folded
+    into every cache fingerprint (see build_code_fingerprint() and the
+    module docstring's SEARCH REUSE section) because a config-knob-only
+    fingerprint has no way to tell "the code that computes this stage's
+    result changed" from "nothing changed" -- without this, editing the
+    actual computation (fixing a bug, changing a formula, reordering a
+    calculation) would silently keep matching old cached results computed
+    under the previous, different logic. A pure docstring/comment edit --
+    demonstrably the dominant kind of change to this file -- does NOT
+    change this hash and so does not cost a cache miss; any real change
+    to a statement or expression does, whole-file (not per-function): the
+    simpler, safer choice over curating a per-stage dependency list that
+    could silently go stale as this file grows -- see the SEARCH REUSE
+    section for the "occasional unnecessary miss, never an incorrect hit"
+    tradeoff this accepts, same as every other fingerprint field here.'''
+    tree = ast.parse(Path(__file__).read_text())
+    _strip_docstrings(tree)
+    return hashlib.sha256(ast.dump(tree).encode()).hexdigest()
+
+
+def _environment_fingerprint():
+    '''Captures the installed numerical/algorithmic dependencies this
+    pipeline's cached stages actually run through -- an upgraded
+    dimx/gmn/pyEDM/etc. version could change a cached stage's numerical
+    output just as much as an edit to this file would, and neither a
+    config-knob fingerprint nor _pipeline_code_fingerprint() (this file's
+    own source only) can notice that. importlib.metadata.version() covers
+    every properly-installed package; `apps` (apps.CreateNetwork /
+    apps.InteractionMatrix -- see build_network()'s docstring, MODE='GMN'
+    only) ships without package metadata in this venv, so its two
+    actually-used modules are content-hashed directly instead, same
+    approach as _hash_file(). Always includes every entry regardless of
+    MODE, same "simpler than branching, costs only an occasional
+    unnecessary miss" rationale as build_search_fingerprint()'s own
+    docstring.'''
+    versions = {}
+    for pkg in ('dimx', 'gmn', 'pyEDM', 'networkx', 'numpy', 'pandas'):
+        try:
+            versions[pkg] = importlib.metadata.version(pkg)
+        except importlib.metadata.PackageNotFoundError:
+            versions[pkg] = None
+    for mod_name in ('apps.CreateNetwork', 'apps.InteractionMatrix'):
+        mod = importlib.import_module(mod_name)
+        versions[mod_name] = _hash_file(Path(mod.__file__))
+    return versions
+
+
+def build_code_fingerprint():
+    '''Combines _pipeline_code_fingerprint() and _environment_fingerprint()
+    into the single value every cache fingerprint embeds (see the module
+    docstring's SEARCH REUSE section). Computed once per job in main()
+    (gated by MDE_GMN_REUSE_BEST_PARAMS, same as data_file_sha256 --
+    parsing this file's AST and querying installed package versions has a
+    real, if small, cost not worth paying when caching is off) and
+    threaded down as a plain argument rather than recomputed per call --
+    run_mde_for_target() alone can run 400+ times in one job (once per
+    nested-MDE node), so recomputing this per call would add up.'''
+    return {'source': _pipeline_code_fingerprint(), 'environment': _environment_fingerprint()}
+
+
+def build_search_fingerprint(cfg, net_result, data_file_sha256, code_fingerprint):
     '''Everything that determines one network's theta search outcome
     (run_theta_search()), for MDE_GMN_REUSE_BEST_PARAMS's cache-lookup
     below: the knobs that shape that network's topology (MODE/GMN_MODE/
@@ -2082,49 +2337,193 @@
         'prediction_start': cfg['PREDICTION_START'],
         'prediction_length': cfg['PREDICTION_LENGTH'],
         'network_nodes': sorted(graph.nodes),
+        'code_fingerprint': code_fingerprint,
+        'stage': 'theta_search',
     }
 
 
-def find_cached_best_params(prev_runs_dir, network_id, fingerprint):
-    '''Scans prev_runs_dir (recursively, so it works whether prev_runs_dir
-    is the results/ root -- one level of job_<id>/ subdirectories -- or
-    already a single job's output dir) for best_params_<network_id>.json
-    files, newest (by mtime) first, and returns the first one whose
-    stored fingerprint exactly equals `fingerprint`. Returns None if
-    prev_runs_dir doesn't exist yet (e.g. a project's very first run) or
-    nothing matches -- both are ordinary "no cache hit" outcomes, not
-    errors, so the caller falls through to running the real search.'''
+def build_network_fingerprint(mode, targets, observable_columns, nested_n, nested_m,
+                               embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
+                               delay_set, e_range, tau_range, prediction_start,
+                               prediction_length, data_file, data_file_sha256, code_fingerprint):
+    '''Everything that determines one group's network-construction outcome
+    (build_nested_mde_network(), or the CCM/apps.CreateNetwork path for
+    MODE='GMN') -- unlike build_search_fingerprint() above, computable
+    *before* that construction runs, since the whole point is deciding
+    whether to skip it. Takes build_network()'s own already-resolved
+    parameters directly (e.g. `observable_columns` -- REGION_CHANNEL_CAP's
+    effect is already fully captured by exactly which channels ended up
+    in that list, so the raw cap number doesn't need its own field; same
+    for LIB_RATIO via prediction_start/prediction_length) rather than raw
+    cfg knobs, so this needs no `cfg` dependency. Skips
+    build_search_fingerprint()'s network_nodes sanity check (nothing
+    built yet to check) and GMN_MODE (irrelevant to one already-resolved
+    `targets` group's own topology -- see target_groups_for_mode()).
+    'stage' is what keeps this from ever being confused with a
+    theta_search/mde_selection fingerprint even if the remaining fields
+    happened to collide.'''
+    return {
+        'mode': mode,
+        'targets': sorted(targets),
+        'observable_columns': sorted(observable_columns),
+        'mde_d': nested_n,
+        'nested_m': nested_m,
+        'embed_dim_rho_min': embed_dim_rho_min,
+        'cross_map_rho_min': cross_map_rho_min,
+        'exclusion_radius': exclusion_radius,
+        'delay_set': list(delay_set) if delay_set is not None else None,
+        'e_vals': list(e_range),
+        'tau_vals': list(tau_range),
+        'prediction_start': prediction_start,
+        'prediction_length': prediction_length,
+        'data_file': data_file,
+        'data_file_sha256': data_file_sha256,
+        'code_fingerprint': code_fingerprint,
+        'stage': 'network',
+    }
+
+
+def build_mde_selection_fingerprint(node, candidate_pool, mde_d, embed_dim_rho_min,
+                                     cross_map_rho_min, exclusion_radius, delay_set,
+                                     prediction_start, prediction_length,
+                                     data_file, data_file_sha256, code_fingerprint):
+    '''Everything that determines one run_mde_for_target() call's outcome,
+    computable before that call runs. `candidate_pool` is the exact
+    (possibly delay-expanded) column list dx.MDE searched over, kept in
+    order rather than sorted -- dx.MDE's forward selection can tie-break
+    on column order, so the same columns in a different order aren't
+    guaranteed to select the same result. `exclusion_radius` is recorded
+    as passed (not normalized to 'auto' the way build_network_fingerprint
+    does) since run_mde_for_target() itself has no visibility into
+    EXCLUSION_RADIUS_AUTO -- see that function's own docstring on why the
+    argument is a harmless placeholder in AUTO mode; at worst this costs
+    an occasional unnecessary cache miss, never an incorrect hit.'''
+    return {
+        'node': node,
+        'candidate_pool': list(candidate_pool),
+        'mde_d': mde_d,
+        'embed_dim_rho_min': embed_dim_rho_min,
+        'cross_map_rho_min': cross_map_rho_min,
+        'exclusion_radius': exclusion_radius,
+        'delay_set': list(delay_set),
+        'prediction_start': prediction_start,
+        'prediction_length': prediction_length,
+        'data_file': data_file,
+        'data_file_sha256': data_file_sha256,
+        'code_fingerprint': code_fingerprint,
+        'stage': 'mde_selection',
+    }
+
+
+def _load_json_cache_entry(path):
+    with open(path) as f:
+        return json.load(f)
+
+
+def _load_pickle_cache_entry(path):
+    with open(path, 'rb') as f:
+        return pickle.load(f)
+
+
+def find_cached(prev_runs_dir, filename, fingerprint, load_one=_load_json_cache_entry):
+    '''Generic cross-job cache lookup, shared by every cacheable stage
+    (theta search, MDE forward-selection, network construction, the
+    generate-start sweep -- see the module docstring's SEARCH REUSE
+    section). Scans prev_runs_dir recursively -- works whether
+    prev_runs_dir is the cache pool's root (one level of job_<id>/
+    subdirectories, see simple-mde-gmn.sbatch's CACHE_ROOT/RUN_CACHE_DIR
+    comment) or already a single job's own cache dir -- for files named
+    `filename`, newest (by mtime) first, and returns the first
+    (entry, path) whose entry['fingerprint'] exactly equals `fingerprint`.
+    `load_one(path)` does the actual deserialization: JSON by default for
+    small metadata (theta, MDE column lists), or _load_pickle_cache_entry
+    for DataFrames/graphs too structured for JSON (see
+    save_pickle_to_cache()). Returns None if prev_runs_dir doesn't exist
+    yet (e.g. this project's very first run) or nothing matches -- both
+    ordinary "no cache hit" outcomes, not errors; every caller falls
+    through to computing the real result on None.'''
     if not prev_runs_dir.exists():
         return None
     candidates = sorted(
-        prev_runs_dir.glob(f'**/best_params_{network_id}.json'),
+        prev_runs_dir.glob(f'**/{filename}'),
         key=lambda p: p.stat().st_mtime, reverse=True,
     )
     for path in candidates:
         try:
-            with open(path) as f:
-                entry = json.load(f)
-        except (OSError, json.JSONDecodeError):
+            entry = load_one(path)
+        except (OSError, EOFError, json.JSONDecodeError, pickle.UnpicklingError):
             continue
         if entry.get('fingerprint') == fingerprint:
-            return entry['best_node_theta'], path
+            return entry, path
     return None
 
 
-def save_best_params(results_dir, network_id, fingerprint, best_node_theta, source):
+def _save_to_cache(filename, results_dir, run_cache_dir, write_one):
+    '''Calls write_one(path) once per unique directory in
+    {results_dir, run_cache_dir} (deduplicated, order-preserving, so the
+    RUN_CACHE_DIR==RESULTS_DIR fallback for ad hoc runs outside the sbatch
+    script -- see load_config()'s docstring -- writes exactly once, not
+    twice). results_dir is this run's ordinary per-job record, unchanged
+    in behavior from before this cache existed: it still only reaches
+    persistent storage at job end, on success, via simple-mde-gmn.sbatch's
+    final `cp -av`. run_cache_dir is the durable bucket pool
+    (MDE_GMN_RUN_CACHE_DIR) a future job's find_cached() searches --
+    writing there *immediately*, right after this stage finishes, is what
+    lets its result survive even if this job crashes or times out later in
+    the pipeline, instead of being lost with the rest of the /flash
+    scratch tempdir (see the sbatch script's /flash scratch setup
+    section's own comment on exactly this gap). Returns the results_dir
+    path, for callers that log it.'''
+    written = []
+    for directory in dict.fromkeys([results_dir, run_cache_dir]):
+        path = directory / filename
+        write_one(path)
+        written.append(path)
+    return written[0]
+
+
+def save_json_to_cache(filename, payload, results_dir, run_cache_dir):
+    '''JSON variant of _save_to_cache() -- `payload` must already carry
+    its own 'fingerprint' key (see find_cached()).'''
+    def write_one(path):
+        with open(path, 'w') as f:
+            json.dump(payload, f, indent=2)
+    return _save_to_cache(filename, results_dir, run_cache_dir, write_one)
+
+
+def save_pickle_to_cache(filename, payload, results_dir, run_cache_dir):
+    '''Pickle variant of _save_to_cache(), for artifacts too large/
+    structured for JSON (a network graph, an MDE run's DataFrame output).
+    `payload` must already carry its own 'fingerprint' key (see
+    find_cached()).'''
+    def write_one(path):
+        with open(path, 'wb') as f:
+            pickle.dump(payload, f)
+    return _save_to_cache(filename, results_dir, run_cache_dir, write_one)
+
+
+def find_cached_best_params(prev_runs_dir, network_id, fingerprint):
+    '''Theta search's own cache lookup -- thin find_cached() wrapper kept
+    as a named function since callers want (best_node_theta, path), not
+    the raw cache entry dict.'''
+    hit = find_cached(prev_runs_dir, f'best_params_{network_id}.json', fingerprint)
+    if hit is None:
+        return None
+    entry, path = hit
+    return entry['best_node_theta'], path
+
+
+def save_best_params(results_dir, run_cache_dir, network_id, fingerprint, best_node_theta, source):
     '''Writes this run's own best_params_<network_id>.json -- the artifact
     a future job's find_cached_best_params() looks for -- regardless of
     whether this run computed it fresh or itself reused a cache hit, so
     the provenance chain (source: 'computed' vs. 'reused_from:<path>')
     stays visible in whichever job's output this file ends up in.'''
-    path = results_dir / f'best_params_{network_id}.json'
-    with open(path, 'w') as f:
-        json.dump({
-            'fingerprint': fingerprint,
-            'best_node_theta': best_node_theta,
-            'source': source,
-        }, f, indent=2)
-    return path
+    return save_json_to_cache(
+        f'best_params_{network_id}.json',
+        {'fingerprint': fingerprint, 'best_node_theta': best_node_theta, 'source': source},
+        results_dir, run_cache_dir,
+    )
 
 
 def retune_theta_for_all_nodes(data_lib, graph, nodes, theta_range, cores, min_pred_rows=20):
@@ -2208,7 +2607,7 @@
 
 def run_theta_search(gmn_runs, prediction_start, prediction_length, theta_range, results_dir,
                       cores, generate_length=None, reuse_best_params=False, prev_runs_dir=None,
-                      fingerprints=None, exclusion_radius=0):
+                      run_cache_dir=None, fingerprints=None, exclusion_radius=0):
     '''Retunes each network's per-node theta once
     (retune_theta_for_all_nodes()), then scores the result with a single
     Generate() call -- replacing this pipeline's old combined (E, tau,
@@ -2248,7 +2647,7 @@
                 theta_search_results[network_id] = {
                     'best_node_theta': best_node_theta, 'reused_from': str(source),
                 }
-                save_best_params(results_dir, network_id, fingerprint, best_node_theta,
+                save_best_params(results_dir, run_cache_dir, network_id, fingerprint, best_node_theta,
                                   source=f'reused_from:{source}')
                 continue
             print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS set but no matching cached run '
@@ -2276,7 +2675,8 @@
         theta_search_results[network_id] = {'best_node_theta': best_node_theta}
 
         if fingerprint is not None:
-            save_best_params(results_dir, network_id, fingerprint, best_node_theta, source='computed')
+            save_best_params(results_dir, run_cache_dir, network_id, fingerprint, best_node_theta,
+                              source='computed')
 
     return theta_search_results
 
@@ -2285,8 +2685,31 @@
 # Generate()/Forecast() start-point sweep
 # =============================================================================
 
+def build_start_sweep_fingerprint(network_fingerprint, node_theta, start_offsets,
+                                   generate_length, exclusion_radius):
+    '''Everything that determines one network's start-point sweep outcome
+    (sweep_generate_start()): the upstream network+theta identity
+    (network_fingerprint -- see build_search_fingerprint(), None if
+    MDE_GMN_REUSE_BEST_PARAMS wasn't on for that stage, in which case this
+    sweep's own cache is skipped too, since there'd be nothing to pin the
+    "which model" half of this fingerprint to) plus node_theta itself
+    (two different theta results for the very same network fingerprint --
+    e.g. a nondeterministic PredictNonlinear retune -- must not share a
+    sweep cache entry) and the sweep's own knobs.'''
+    return {
+        'network_fingerprint': network_fingerprint,
+        'node_theta': {k: node_theta[k] for k in sorted(node_theta)},
+        'start_offsets': list(start_offsets),
+        'generate_length': generate_length,
+        'exclusion_radius': exclusion_radius,
+        'stage': 'start_sweep',
+    }
+
+
 def sweep_generate_start(gmn_runs, theta_search_results, prediction_start, prediction_length,
-                          results_dir, cores, generate_length, start_offsets, exclusion_radius=0):
+                          results_dir, cores, generate_length, start_offsets, exclusion_radius=0,
+                          reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
+                          fingerprints=None):
     '''Finds, per network, which row Generate() should start free-running
     from to best match the real held-out data -- at the per-node theta
     run_theta_search() already chose, not re-searched here (E and each
@@ -2304,7 +2727,13 @@
     -- see that function's docstring. Checkpoints to disk after every
     offset (same resume pattern as run_theta_search's cache) since each
     candidate is its own full Generate() call and this shouldn't have to
-    restart from scratch after an interrupted job.
+    restart from scratch after an interrupted job -- that checkpoint is
+    intra-job only (results_dir), though; reuse_cache/prev_runs_dir/
+    run_cache_dir/fingerprints add the same cross-job fingerprinted cache
+    every other stage has (see the module docstring's SEARCH REUSE
+    section) via build_start_sweep_fingerprint(), so a full resubmission
+    after a crash can skip the whole sweep, not just resume within one
+    still-running job.
 
     A single-offset `start_offsets` (the default -- see load_config()'s
     GENERATE_START_MIN/MAX_OFFSET) means the sweep is disabled: that one
@@ -2322,6 +2751,36 @@
 
         net_result = run['net_result']
         node_theta = theta_search_results[network_id]['best_node_theta']
+        network_fingerprint = fingerprints[network_id] if fingerprints else None
+
+        sweep_fingerprint = None
+        cache_filename = f'generate_start_sweep_{network_id}.json'
+        if reuse_cache and network_fingerprint is not None:
+            sweep_fingerprint = build_start_sweep_fingerprint(
+                network_fingerprint, node_theta, start_offsets, generate_length, exclusion_radius,
+            )
+            hit = find_cached(prev_runs_dir, cache_filename, sweep_fingerprint)
+            if hit is not None:
+                entry, source = hit
+                scores = {int(k): v for k, v in entry['scores'].items()}
+                best_offset = entry['best_offset']
+                best_starts[network_id] = prediction_start + best_offset
+                print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS matched cached start sweep '
+                      f'{source} -- skipping sweep entirely (best offset {best_offset}, '
+                      f'score={scores.get(best_offset, float("nan")):.4f})', flush=True)
+
+                fig, ax = plt.subplots(figsize=(7, 4))
+                ax.plot(start_offsets, [scores.get(o, np.nan) for o in start_offsets], marker='o')
+                ax.axvline(best_offset, color='gray', ls='--', lw=0.8)
+                ax.set_xlabel('Start offset from PREDICTION_START (rows)')
+                ax.set_ylabel('Generate() rho (mean over targets)')
+                ax.set_title(f'Generate() start-point sweep -- {network_id} (cached)')
+                fig.tight_layout()
+                fig.savefig(results_dir / f'generate_start_sweep_{network_id}.png', dpi=300)
+                plt.close(fig)
+                continue
+            print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS set but no matching cached start '
+                  f'sweep found under {prev_runs_dir} -- running sweep.', flush=True)
 
         checkpoint_file = results_dir / f'generate_start_checkpoint_{network_id}.pkl'
         scores = {}
@@ -2366,6 +2825,19 @@
         print(f'{network_id}: best generate start = row {best_starts[network_id]} '
               f'(offset {best_offset}, score={scores.get(best_offset, float("nan")):.4f})', flush=True)
 
+        if sweep_fingerprint is not None:
+            # JSON object keys must be strings -- offsets round-trip back
+            # to int in the cache-hit branch above.
+            save_json_to_cache(
+                cache_filename,
+                {
+                    'fingerprint': sweep_fingerprint,
+                    'scores': {str(k): v for k, v in scores.items()},
+                    'best_offset': best_offset,
+                },
+                results_dir, run_cache_dir,
+            )
+
         fig, ax = plt.subplots(figsize=(7, 4))
         ax.plot(start_offsets, [scores.get(o, np.nan) for o in start_offsets], marker='o')
         ax.axvline(best_offset, color='gray', ls='--', lw=0.8)
@@ -2812,6 +3284,19 @@
 
     obs_cols = observable_columns(df, cfg['REGION_CHANNEL_CAP'])
 
+    # Computed once, up front, and threaded through every cacheable stage
+    # below (MDE forward-selection, network construction, theta search,
+    # the generate-start sweep -- see the module docstring's SEARCH REUSE
+    # section) rather than per-stage, so they all key off the exact same
+    # data/code/environment identity. Only actually paid for when caching
+    # is enabled -- skip the cost otherwise. code_fingerprint in
+    # particular is what lets a cache safely survive a purely cosmetic
+    # edit to this file (a docstring, say) while still invalidating on any
+    # real change to the actual computation -- see build_code_fingerprint()
+    # and _pipeline_code_fingerprint()'s docstrings.
+    data_file_sha256 = _hash_file(cfg['DATA_DIR'] / cfg['DATA_FILE']) if cfg['REUSE_BEST_PARAMS'] else None
+    code_fingerprint = build_code_fingerprint() if cfg['REUSE_BEST_PARAMS'] else None
+
     # ---- MDE (or its full-channel-pool stand-in for MODE='GMN') -------
     mde_results = {}
     expanded_df = expanded_obs_cols = None  # MDE-GMN only -- see below
@@ -2831,7 +3316,11 @@
             mde_results[target] = run_mde_for_target(
                 target, expanded_df, candidate_pool, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                 effective_d, cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
-                cfg['EXCLUSION_RADIUS_VALUE'], cfg['CORES'], cfg['RESULTS_DIR'],
+                cfg['EXCLUSION_RADIUS_VALUE'], cfg['CORES'], cfg['RESULTS_DIR'], cfg['DELAY_SET'],
+                reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
+                run_cache_dir=cfg['RUN_CACHE_DIR'],
+                data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
+                code_fingerprint=code_fingerprint,
             )
 
         if cfg['RUN_MDE_DIAGNOSTICS']:
@@ -2862,6 +3351,10 @@
             exclusion_radius=cfg['EXCLUSION_RADIUS_VALUE'],
             expanded_df=expanded_df, expanded_observable_columns=expanded_obs_cols,
             delay_set=cfg['DELAY_SET'],
+            reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
+            run_cache_dir=cfg['RUN_CACHE_DIR'],
+            data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
+            code_fingerprint=code_fingerprint,
         )
         draw_colored_network(
             net_result, cfg['TARGET_COLORS'],
@@ -2871,9 +3364,8 @@
 
     fingerprints = None
     if cfg['REUSE_BEST_PARAMS']:
-        data_file_sha256 = _hash_file(cfg['DATA_DIR'] / cfg['DATA_FILE'])
         fingerprints = {
-            network_id: build_search_fingerprint(cfg, run['net_result'], data_file_sha256)
+            network_id: build_search_fingerprint(cfg, run['net_result'], data_file_sha256, code_fingerprint)
             for network_id, run in gmn_runs.items()
         }
 
@@ -2882,6 +3374,7 @@
         cfg['RESULTS_DIR'], cfg['CORES'],
         generate_length=cfg['GENERATE_LENGTH'],
         reuse_best_params=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
+        run_cache_dir=cfg['RUN_CACHE_DIR'],
         fingerprints=fingerprints, exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
     )
 
@@ -2889,6 +3382,8 @@
         gmn_runs, theta_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
         cfg['RESULTS_DIR'], cfg['CORES'], cfg['GENERATE_LENGTH'], cfg['GENERATE_START_OFFSETS'],
         exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
+        reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
+        run_cache_dir=cfg['RUN_CACHE_DIR'], fingerprints=fingerprints,
     )
 
     rebuild_at_best_params(
```

---


## Job 36533849

Generated: 2026-08-07T00:09:39+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_36533672**

### Plain-language summary

Further `--mem` increase, 112G -> 200G. Job 36515744 (48G) was OOM-killed
at ~49.75G RSS while still inside nested-MDE network *construction*
(depth-3 wave, 32 parallel workers) -- it never reached
`run_theta_search()`'s per-node retuning + `Generate()` scoring, which
this pipeline's own comments flag as its most expensive stage. So the
112G resubmission (job 36533672, cancelled while still queued and never
actually run) only had real memory headroom validated against the
construction phase, not against that later, unmeasured, likely-heavier
stage. 200G gives more margin against that unknown while staying well
under this partition's 500G/node cap. No other parameters or code
changed.

### Parameter changes (simple-mde-gmn.sbatch)

_No parameter changes._

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

```diff
--- job_36533672/simple-mde-gmn.sbatch
+++ job_36533849/simple-mde-gmn.sbatch
@@ -2,7 +2,7 @@
 #SBATCH --job-name=run_simple_mde_gmn
 #SBATCH --partition=compute
 #SBATCH --time=2-00:00:00
-#SBATCH --mem=112G
+#SBATCH --mem=200G
 #SBATCH --cpus-per-task=32
 # ADJUST: must be a literal absolute path, not $HOME/$PROJ_DIR -- Slurm
 # parses #SBATCH directives itself before this script (or any shell)
```

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---


## Job 36533672

Generated: 2026-08-07T00:05:18+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_36515744**

### Plain-language summary

Resubmission of job 36515744, which was OOM-killed after ~5h42m (MaxRSS
~49.75G against a 48G request). Only change: `--mem` raised 48G -> 112G
to give real headroom above observed usage; no other parameters or code
touched, so this run continues testing the same
REGION_CHANNEL_CAP=999/NESTED_M=3/MDE_D=5 configuration, just with
enough memory to actually finish.

### Parameter changes (simple-mde-gmn.sbatch)

_No parameter changes._

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

```diff
--- job_36515744/simple-mde-gmn.sbatch
+++ job_36533672/simple-mde-gmn.sbatch
@@ -2,7 +2,7 @@
 #SBATCH --job-name=run_simple_mde_gmn
 #SBATCH --partition=compute
 #SBATCH --time=2-00:00:00
-#SBATCH --mem=48G
+#SBATCH --mem=112G
 #SBATCH --cpus-per-task=32
 # ADJUST: must be a literal absolute path, not $HOME/$PROJ_DIR -- Slurm
 # parses #SBATCH directives itself before this script (or any shell)
```

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---


## Job 36515744

Generated: 2026-08-06T17:23:55+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_35998003**

### Plain-language summary

job_35998003 was a deliberately shrunk smoke test (see its own entry) --
the table/diffs below are noisy because they're against that tiny
config, not against the last *real* run (job_35928213). Relative to
job_35928213, the sbatch parameter values themselves are actually
unchanged (MDE_D=5, NESTED_M=3, REGION_CHANNEL_CAP=999,
GENERATE_LENGTH=300, full E/tau grid, 2-day time limit) -- this is the
first full-scale run since then, now with REUSE_BEST_PARAMS=true so a
matching past network/search combination (if any) skips its theta
retuning instead of repeating it.

What's actually new this run is entirely pipeline-code work from this
session, all opt-in/backward-compatible with job_35928213's own results
except where noted:

- MDE_D now counts distinct *real* observables, not delay-expanded
  candidate columns -- internally scaled up by len(DELAY_SET) before
  being handed to dimx.MDE as D (effective_mde_dimension()), so the same
  MDE_D=5 here selects up to 5 real observables per round rather than up
  to 5 columns that could all be different delays of the same one.
  Changes this run's actual node/edge selection vs. job_35928213 even
  though the MDE_D number itself didn't change.
- NESTED_M=0 is now a sentinel for "uncapped" nesting (expand until real
  observables are exhausted) -- unused this run (NESTED_M=3, unchanged).
- New MDE_GMN_EXCLUSION_RADIUS='auto' (default): dimx.MDE's own internal
  CrossMapColumns/EmbedDimension/CCM calls now use a dynamically
  recomputed Theiler-window exclusion radius (dimension x largest
  DELAY_SET delay actually selected so far) instead of the previous
  always-0 (no exclusion). Also changes this run's MDE selections vs.
  job_35928213.
- New GMN_EXCLUSION_RADIUS=0 (unchanged default): same exclusion-radius
  mechanism, now exposed for gmn's own Generate()/Forecast() SMap calls
  -- 0 this run, so no behavior change there.
- New FORECAST_PLOT_RANGE=windowed (previously not a parameter -- always
  "full"): the final Generate()-vs-Forecast()-vs-actual diagnostic plot
  now windows to 2x GENERATE_LENGTH on each side of generate_start
  instead of always starting at row 0, and the actual/lib trace is now
  drawn with alpha=0.6 for easier visual comparison against the
  overlaid Generate()/Forecast() lines. Diagnostic-only -- doesn't
  affect any computed rho/network/search result.
- Divergence-guarding fix: rebuild_at_best_params()'s final Generate()
  call and run_forecast_diagnostic()'s Forecast() call were previously
  the only two SMap calls in this pipeline with no try/except around
  them (every other Generate()/Forecast()/Simplex call already was) --
  this is very likely what actually killed job_35998003 ("generative
  step failed", per that job's own newly-added Outcome note) and, before
  that, job_35886791. Both are now guarded the same way as everywhere
  else: a divergence is logged and that one network is skipped (no
  plots/fidelity/output for it), rather than crashing the whole job and
  losing every other, unrelated network's results too -- relevant here
  since GMN_MODE=separate with 2 TARGETS builds two independent networks
  in the same job.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_35998003 | job_36515744 |
|---|---|---|
| CROSS_MAP_RHO_MIN | 0.2 | 0.1 |
| DELAY_SET | _(unset)_ | 0,-1,-3,-5,-10 |
| EMBED_DIM_RHO_MIN | 0.2 | 0.1 |
| EXCLUSION_RADIUS | _(unset)_ | auto |
| E_MAX | 7 | 16 |
| FORECAST_PLOT_RANGE | _(unset)_ | windowed |
| GENERATE_LENGTH | 50 | 300 |
| GENERATE_START_MAX_OFFSET | _(unset)_ | 20 |
| GENERATE_START_MIN_OFFSET | _(unset)_ | 0 |
| GENERATE_START_STEP | _(unset)_ | 2 |
| GMN_EXCLUSION_RADIUS | _(unset)_ | 0 |
| INTERACTION_MATRIX_OUTER_CORES | 2 | 4 |
| LIB_RATIO | 0.5 | 0.75 |
| MDE_D | 4 | 5 |
| NESTED_M | 1 | 3 |
| REGION_CHANNEL_CAP | 15 | 999 |
| REUSE_BEST_PARAMS | _(unset)_ | true |
| SMAP_SEARCH_MODE | coarse_retune | _(unset)_ |
| TAU_MIN | -5 | -15 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

```diff
--- job_35998003/simple-mde-gmn.sbatch
+++ job_36515744/simple-mde-gmn.sbatch
@@ -1,7 +1,7 @@
 #!/bin/bash
 #SBATCH --job-name=run_simple_mde_gmn
 #SBATCH --partition=compute
-#SBATCH --time=00:30:00
+#SBATCH --time=2-00:00:00
 #SBATCH --mem=48G
 #SBATCH --cpus-per-task=32
 # ADJUST: must be a literal absolute path, not $HOME/$PROJ_DIR -- Slurm
@@ -49,29 +49,37 @@
 export MKL_NUM_THREADS=1
 export OPENBLAS_NUM_THREADS=1
 
-# FIXED CAVEAT (found by reading pyEDM/CCM.py and apps/InteractionMatrix.py
-# directly, after observing ~90 live processes from an 8-core request on a
-# 10-core test machine, and root-caused after several full jobs each ran
-# ~2 days regardless of dataset size, MODE, or MDE_D): apps.InteractionMatrix's
+# FIXED CAVEAT (root-caused after several full jobs each ran ~2 days
+# regardless of dataset size, MODE, or MDE_D): apps.InteractionMatrix's
 # own outer worker pool (sized by MDE_GMN_INTERACTION_MATRIX_OUTER_CORES
-# below) calls pyEDM.CCM() once per channel pair (dimx.MDE similarly calls
-# it once per candidate column), and CCM() used to create a *brand-new*
-# multiprocessing pool sized to the full visible cpu_count() on *every
-# single call* for work that only ever amounts to 4-8 tiny KDTree tasks --
-# apps.InteractionMatrix/dimx.MDE never pass CCM's own `parallel=`
-# override, so this couldn't be tuned away via their CLIs. The two pools
-# didn't coordinate, so peak concurrent processes was (outer cores) x
-# cpu_count(), not just cpu_count(); with REGION_CHANNEL_CAP effectively
-# uncapped (~470 real channels), that's ~100k+ CCM() calls each paying a
-# full spawn/forkserver Pool creation + heavy reimport teardown cost --
-# process-spawn overhead dominating actual compute by orders of magnitude,
-# which is why cutting DATA_FILE resolution or MDE_D didn't help wall time.
+# below) calls pyEDM.CCM() once per channel pair (dimx.MDE similarly once
+# per candidate column), and CCM() used to spawn its *own* fresh
+# multiprocessing pool sized to the full visible cpu_count() on every
+# call, for work that only ever amounts to 4-8 tiny KDTree tasks -- an
+# 8-core request could produce ~90 live processes, not 8, since the two
+# pools didn't coordinate (peak processes was outer-cores x cpu_count()).
+# With REGION_CHANNEL_CAP uncapped (~470 real channels) that's ~100k+
+# CCM() calls each paying a full spawn-and-reimport cost -- process-spawn
+# overhead dominating actual compute by orders of magnitude, which is why
+# cutting DATA_FILE resolution or MDE_D didn't help wall time.
 # scripts/src/simple-mde-gmn.py now monkeypatches pyEDM.CCM's inner pool
-# away entirely (`_sequential_ccm_workers`) rather than just avoiding the
-# multiplication, so this is no longer a live caveat -- kept here for
-# provenance. MDE_GMN_INTERACTION_MATRIX_OUTER_CORES (default 2) is still
-# a separate knob from --cpus-per-task since it's the one thing sizing
-# real parallelism across column pairs now.
+# away entirely (`_sequential_ccm_workers`), so this is no longer live --
+# kept here for provenance. MDE_GMN_INTERACTION_MATRIX_OUTER_CORES
+# (default 2) is still a separate knob from --cpus-per-task since it's
+# the one thing sizing real parallelism across column pairs now.
+#
+# REMAINING CAVEAT (unfixed, just made avoidable): even with the above
+# fixed, the per-node theta search in run_theta_search() -- one
+# GENERATE_LENGTH-step recursive Generate() call across every node -- is
+# still the single most expensive part of this pipeline. That cost is
+# unavoidable the *first* time a given network/data combination is
+# searched, but resubmitting the identical combination used to redo it
+# every time regardless. REUSE_BEST_PARAMS below skips it on an exact
+# repeat -- see that parameter's own comment. (This used to also include
+# a len(E) x len(tau) grid on top of the per-node theta retuning --
+# removed per explicit direction once E and each driver's delay started
+# coming from nested-MDE's own network instead of a separate search; see
+# scripts/backup/simple-mde-gmn_pre-delay-embedding.py for that version.)
 
 # ADJUST: project checkout location and lab/group allocation on this cluster.
 PROJ_DIR="$HOME/pao_rotation"
@@ -142,10 +150,10 @@
 # above -- deliberately not a second `trap ... EXIT` here, which would
 # silently replace that one instead of stacking.
 
-# NOTE: the E/tau/theta search below checkpoints to disk after every
-# candidate so a resubmitted job can resume instead of restarting a
-# multi-hour search from scratch (see the script's run_smap_search()) --
-# but $tempdir lives under $FLASH_BASE and is deleted by the `trap` above
+# NOTE: the theta search below checkpoints to disk after every network
+# so a resubmitted job can resume instead of restarting a multi-hour
+# search from scratch (see the script's run_theta_search()) -- but
+# $tempdir lives under $FLASH_BASE and is deleted by the `trap` above
 # the moment this script exits, including on a SLURM timeout/cancellation
 # (SIGTERM), *before* results get copied to $OUT_DIR. If your search
 # might not finish within --time, checkpoints will be lost across
@@ -182,19 +190,32 @@
 export MDE_GMN_D="${MDE_D}"
 export MDE_GMN_EMBED_DIM_RHO_MIN="${EMBED_DIM_RHO_MIN}"
 export MDE_GMN_CROSS_MAP_RHO_MIN="${CROSS_MAP_RHO_MIN}"
+export MDE_GMN_EXCLUSION_RADIUS="${EXCLUSION_RADIUS}"
 export MDE_GMN_NESTED_M="${NESTED_M}"
 export MDE_GMN_REGION_CHANNEL_CAP="${REGION_CHANNEL_CAP}"
+export MDE_GMN_DELAY_SET="${DELAY_SET}"
 export MDE_GMN_LIB_RATIO="${LIB_RATIO}"
 export MDE_GMN_GENERATE_LENGTH="${GENERATE_LENGTH}"
+export GMN_EXCLUSION_RADIUS="${GMN_EXCLUSION_RADIUS}"
+export MDE_GMN_FORECAST_PLOT_RANGE="${FORECAST_PLOT_RANGE}"
+export MDE_GMN_GENERATE_START_MIN_OFFSET="${GENERATE_START_MIN_OFFSET}"
+export MDE_GMN_GENERATE_START_MAX_OFFSET="${GENERATE_START_MAX_OFFSET}"
+export MDE_GMN_GENERATE_START_STEP="${GENERATE_START_STEP}"
 export MDE_GMN_E_MIN="${E_MIN}"
 export MDE_GMN_E_MAX="${E_MAX}"
 export MDE_GMN_E_STEP="${E_STEP}"
 export MDE_GMN_TAU_MIN="${TAU_MIN}"
 export MDE_GMN_TAU_MAX="${TAU_MAX}"
 export MDE_GMN_TAU_STEP="${TAU_STEP}"
-export MDE_GMN_SMAP_SEARCH_MODE="${SMAP_SEARCH_MODE}"
 export MDE_GMN_CORES="${SLURM_CPUS_PER_TASK}"
 export MDE_GMN_INTERACTION_MATRIX_OUTER_CORES="${INTERACTION_MATRIX_OUTER_CORES}"
+export MDE_GMN_REUSE_BEST_PARAMS="${REUSE_BEST_PARAMS}"
+# Unlike MDE_GMN_RESULTS_DIR (pointed at $tempdir on /flash scratch,
+# deleted at job exit -- see that export above / the /flash scratch
+# setup section's own comment), this must point at the persistent
+# per-job output location so a later job can actually find a past run's
+# best_params_<network_id>.json to reuse.
+export MDE_GMN_PREV_RUNS_DIR="$PROJ_DIR/results"
 
 python3 -V
 
```

### Python pipeline changes (simple-mde-gmn.py)

```diff
--- job_35998003/simple-mde-gmn.py
+++ job_36515744/simple-mde-gmn.py
@@ -3,82 +3,108 @@
 MDE_GMN_* environment variable (set by the sbatch script's MODIFIABLE
 PARAMETERS block) rather than edited here directly.
 
-This mirrors the notebook cell-for-cell (see that file for the narrative
-explanation of *why* each step works the way it does -- docstrings below
-are trimmed to what's specific to running headless/non-interactively):
-
-  1. MDE forward-selection (skipped entirely if MODE='GMN'): top-level
-     per-target driver selection (dimx.MDE), used both for the MDE-only
-     diagnostics below and as the depth-0 seed of step 2's nested
-     expansion when MODE='MDE-GMN'.
-  2. GMN network construction. Two entirely different ways to build the
-     interaction matrix depending on MODE -- everything downstream (node
-     generation, search, rebuild, diagnostics) is identical either way:
-       - MODE='MDE-GMN': nested dimx.MDE expansion
-         (build_nested_mde_network()) -- each MDE-selected driver becomes
-         a new MDE target in turn for MDE_GMN_NESTED_M further rounds,
-         eagerly rejecting any edge that would create a cycle or
-         self-loop as the graph is discovered. No CCM / no
-         apps.InteractionMatrix involved at all.
+Pipeline steps (see the notebook for the narrative "why" behind each one
+-- docstrings here cover only what's specific to running headless):
+
+  1. MDE forward-selection (skipped if MODE='GMN'): per-target driver
+     selection (dimx.MDE). Feeds both the MDE-only diagnostics in step 7
+     and, when MODE='MDE-GMN', step 2's depth-0 seed.
+  2. Network construction -- two unrelated methods depending on MODE;
+     everything after this step is identical either way:
+       - MODE='MDE-GMN': build_nested_mde_network() repeatedly runs
+         dimx.MDE over a delay-expanded candidate pool (expand_delay_
+         columns(), MDE_GMN_DELAY_SET -- each observable offered at every
+         configured delay as its own candidate column, a generalized/
+         non-uniform Takens embedding per explicit direction), turning
+         each *contemporaneous* (delay=0) selected driver into a new MDE
+         target for MDE_GMN_NESTED_M further rounds -- a driver selected
+         at a nonzero delay becomes a terminal graph edge instead, never
+         itself re-MDE'd (see that function's docstring for why),
+         rejecting any edge that would create a cycle or self-loop as
+         it's proposed. No CCM, no apps.InteractionMatrix, no
+         apps.CreateNetwork -- the resulting MultiDiGraph (parallel
+         edges let the same observable drive one node at several delays
+         at once) IS the network handed to gmn.GMN directly (see
+         build_network()'s docstring for why CreateNetwork would only
+         be redundant here).
        - MODE='GMN': the original CCM interaction matrix (data-driven
-         E/tau grid search) across the full REGION_CHANNEL_CAP-limited
-         observable pool, via apps.InteractionMatrix. Unchanged.
-     Both paths feed the same per-node driver-count estimate
-     (EmbedDimension) + apps.CreateNetwork call, with the same
-     driver-ranking bug workaround.
-  3. GMN node generation: SMap uniformly for every node (not Simplex, not
-     a per-node mix -- see build_gmn_config()'s docstring for why)
-  4. Empirical (E, tau, per-node theta) search maximizing free-running
-     Generate() fidelity against the real held-out targets
-  5. Final GMN rebuild at the winning parameters + Forecast() diagnostic
-     + Fig-4e-style generative fidelity metric
-  6. MDE-only diagnostics (Evaluate vs PCA/Diffusion Map, Tp-vs-rho,
-     univariate E/tau grid) -- skipped if MODE='GMN', same as the notebook
-
-Differences from the notebook, all a consequence of running headless
-rather than in a live Jupyter kernel:
-  - apps.InteractionMatrix is invoked via subprocess.run instead of
-    get_ipython().system() (get_ipython() only exists inside a real
-    kernel and raises NameError otherwise).
-  - Every plt.show() is replaced with plt.close(fig) after the
-    corresponding savefig -- there's no display to show anything to, and
-    under MPLBACKEND=Agg plt.show() is already a documented no-op, but
-    closing explicitly avoids accumulating open figures across a long run.
-  - All output (data CSVs, .cfg files, checkpoints, figures) goes under
-    one flat RESULTS_DIR rather than the notebook's separate
-    results/vs/figures/ split -- simpler for the sbatch script's
-    `cp -av simple_mde_gmn_outputs/. "$OUT_DIR"/` to copy back verbatim.
-  - CORES is threaded through everywhere a *_cores/numProcess parameter
-    exists (dimx.MDE, apps.InteractionMatrix -cr, pyEDM.EmbedDimension/
-    PredictNonlinear's numProcess, gmn.GMN's cores -- the last one is
-    currently a no-op inside the installed gmn package, confirmed by
-    grepping its source for args.cores; passed anyway in case a future
-    gmn version starts using it) instead of relying on each tool's own
-    hardcoded default, which doesn't know how many cores SLURM actually
-    allocated.
-
-FIXED CAVEAT (found empirically running the MODE='GMN' interaction-matrix
-step against the real dataset, root-caused after several jobs each ran
-~2 days regardless of dataset size or MODE): apps.InteractionMatrix's own
-ProcessPoolExecutor (sized by -cr/CORES) and pyEDM.CCM()'s own *internal*
-multiprocessing nested inside each other -- with CORES=8 this produced
-~90 concurrent processes, not 8, and every one of pyEDM.CCM()'s ~100k+
-per-channel-pair (or per-MDE-candidate) calls paid a full `spawn`/
-`forkserver` Pool creation + heavy reimport (numpy/scipy/pandas/sklearn/
-pyEDM/gmn) teardown cost for work that's only 4-8 tiny KDTree tasks --
-process-spawn overhead dominating actual compute by orders of magnitude,
-which is why cutting dataset resolution or capping MDE_D didn't help.
-Now patched: `_sequential_ccm_workers` below forces pyEDM.CCM() onto its
-in-process sequential path everywhere in this pipeline, eliminating the
-inner pool entirely so only the (already-present, actually useful) outer
-pool provides parallelism. See `_sequential_ccm_workers`'s and
+         E/tau grid search over the REGION_CHANNEL_CAP-limited
+         observable pool, via apps.InteractionMatrix), then a per-node
+         driver-count estimate (EmbedDimension) and apps.CreateNetwork
+         with its driver-ranking-bug workaround.
+  3. Node generation config: SMap uniformly for every node, never Simplex
+     or a per-node mix -- see build_gmn_config()'s docstring for why.
+     Each node's actual embedding is no longer a shared network-wide
+     (E, tau): gmn's own Embed(E, tau, columns) call is replaced by this
+     pipeline's own per-driver-delay embedding, built straight from the
+     network graph -- see _embedding_spec_for()'s and
+     _generalized_embed()'s docstrings, and the gmn.Node.__init__/
+     Forecast/Generate monkeypatches right below this docstring.
+  4. Theta search: retunes each node's own S-map nonlinearity/
+     localization parameter once per network (run_theta_search(),
+     retune_theta_for_all_nodes()), then scores the result with one
+     Generate() call. E and every driver's delay are never searched --
+     they come directly from step 2's network -- so this replaces what
+     used to be a combined (E, tau, per-node theta) grid search; see the
+     SEARCH REUSE paragraph below and run_theta_search()'s docstring.
+  5. Start-point sweep: at the winning theta, tries several rows for
+     Generate()/Forecast() to launch from and keeps whichever matches
+     the real data best -- see sweep_generate_start()'s docstring.
+  6. Final rebuild at the winning theta and start row, plus the
+     Forecast() diagnostic and a Fig-4e-style generative fidelity metric.
+  7. MDE-only diagnostics (Evaluate vs PCA/Diffusion Map, Tp-vs-rho,
+     univariate E/tau grid) -- skipped if MODE='GMN'.
+
+Differences from the notebook, all consequences of running headless
+instead of inside a live Jupyter kernel:
+  - apps.InteractionMatrix runs via subprocess.run instead of
+    get_ipython().system(), which only exists inside a real kernel.
+  - plt.show() calls are replaced with plt.close(fig) after the matching
+    savefig, so figures don't accumulate over a long run.
+  - All output goes under one flat RESULTS_DIR instead of the notebook's
+    separate results/vs/figures/ split, so the sbatch script can copy it
+    back with a single `cp -av`.
+  - CORES is threaded through every *_cores/numProcess parameter (dimx.MDE,
+    apps.InteractionMatrix's -cr, pyEDM's numProcess, gmn.GMN's cores --
+    the last is currently a no-op in the installed gmn package, passed
+    anyway in case a future version uses it) instead of each tool's own
+    hardcoded default, which has no way to know how many cores SLURM
+    actually allocated.
+
+FIXED: apps.InteractionMatrix's own worker pool and pyEDM.CCM()'s
+internal multiprocessing used to nest inside each other -- CORES=8 could
+spawn ~90 live processes, not 8, with each of CCM()'s ~100k+ calls paying
+a full process-spawn-and-reimport cost for only 4-8 tiny KDTree tasks.
+This is why early jobs took ~2 days regardless of dataset size.
+`_sequential_ccm_workers` below forces pyEDM.CCM() onto its in-process
+path everywhere in this pipeline, removing the inner pool so only the
+outer one provides parallelism -- see that function's and
 build_network()'s docstrings for the full evidence trail.
+
+SEARCH REUSE (MDE_GMN_REUSE_BEST_PARAMS, default off): the theta search
+(step 4 above) is deterministic given the same network and data, so
+resubmitting identical settings pays for it twice for no reason. When
+enabled, main() builds a fingerprint per network (every config knob that
+shapes its topology/search space, a hash of the input data file, and its
+own realized node set) and looks for a past job's
+best_params_<network_id>.json under MDE_GMN_PREV_RUNS_DIR with a
+matching fingerprint -- a hit skips that network's theta retuning and
+reuses its cached per-node theta; any mismatch (different dataset,
+different MDE_D/NESTED_M/REGION_CHANNEL_CAP/DELAY_SET/etc., or the first
+run with these settings) just runs the real search, so this is always a
+safe, redundant-compute optimization, never a correctness shortcut. See
+build_search_fingerprint() and find_cached_best_params(). Before this
+pipeline's generalized (per-driver-delay) embedding existed, this same
+mechanism also cached a searched network-wide (E, tau) -- see
+scripts/backup/simple-mde-gmn_pre-delay-embedding.py for that version,
+kept for reference/comparison per explicit direction.
 '''
+import hashlib
+import json
 import multiprocessing as mp
 import os
 import pickle
 import subprocess
-import time
 from concurrent.futures import ProcessPoolExecutor, as_completed
 from configparser import ConfigParser
 from pathlib import Path
@@ -91,43 +117,168 @@
 import numpy as np
 import pandas as pd
 
+from collections import defaultdict
+
 import dimx as dx
 import gmn
 from gmn.Common import FunctionType
 from apps.CreateNetwork import CreateNetwork
 from pyEDM import (
-    Simplex, SMap, CCM, ComputeError, Embed, EmbedDimension,
+    Simplex, ComputeError, EmbedDimension,
     PredictNonlinear, PredictInterval,
 )
 
 
-def _fixed_node_forecast(self):
+def _embedding_spec_for(graph, node_name):
+    '''Builds one node's (column, delay) embedding coordinates directly
+    from the network graph -- a pure function, reusable both by
+    _node_embedding_spec() (which caches this on a live gmn.Node object,
+    for gmn's own monkeypatched Forecast()/Generate()) and by
+    retune_theta_for_all_nodes() (which builds embeddings directly with
+    pandas/pyEDM outside gmn's Node/Network machinery entirely, before
+    any gmn.GMN object exists).
+
+    MultiDiGraph (MODE='MDE-GMN', built by build_nested_mde_network()):
+    self always gets the full self_delay_set (graph.graph['self_delay_set'],
+    set once in build_network()) regardless of how many external drivers
+    nested-MDE found for this node -- the same unconditional fallback a
+    driverless leaf node already needed, just now built from the same
+    delay machinery as everything else rather than a special case. Each
+    real in-edge contributes one coordinate at that edge's own
+    discovered delay (see build_nested_mde_network()'s docstring for why
+    the same driver can appear more than once, at different delays).
+
+    Plain DiGraph (MODE='GMN', from apps.CreateNetwork -- no per-edge
+    delay data exists in that mode at all): falls back to the original
+    uniform E-lagged-copies-of-every-column scheme, using
+    graph.graph['fallback_E']/['fallback_tau'] (set in build_network()
+    from that mode's own CCM E/tau grid search) in place of a per-node
+    nested-MDE result it doesn't have.
+
+    Either way, this is what replaced gmn's own Embed(E, tau, columns)
+    call -- see _generalized_embed()'s docstring -- since a single
+    uniform (E, tau) can't express a per-driver, per-delay embedding.'''
+    if isinstance(graph, nx.MultiDiGraph):
+        self_delays = graph.graph.get('self_delay_set', [0])
+        spec = [(node_name, d) for d in self_delays]
+        spec += [(u, data['delay']) for u, v, data in graph.in_edges(node_name, data=True)]
+    else:
+        E = graph.graph.get('fallback_E', 3)
+        tau = graph.graph.get('fallback_tau', -1)
+        cols = list(graph.predecessors(node_name)) + [node_name]
+        spec = [(c, k * tau) for c in cols for k in range(E)]
+    return spec
+
+
+def network_max_embedding_offset(graph):
+    '''Largest per-node embedding offset across the whole network --
+    since nodes can each have their own distinct embedding now (unlike
+    the old shared (E, tau)), this is used by build_gmn_config()'s
+    Forecast mode to size a single, network-wide lib/pred window that
+    stays valid for every node (see that function's docstring for the
+    row-alignment tradeoff this implies for lower-offset nodes).'''
+    if isinstance(graph, nx.MultiDiGraph):
+        self_delays = graph.graph.get('self_delay_set', [0])
+        self_max = max(abs(d) for d in self_delays)
+        edge_max = max((abs(data['delay']) for _, _, data in graph.edges(data=True)), default=0)
+        return max(self_max, edge_max)
+    E = graph.graph.get('fallback_E', 3)
+    tau = graph.graph.get('fallback_tau', -1)
+    return (E - 1) * abs(tau)
+
+
+def _node_embedding_spec(node):
+    '''Cached (on node._embedding_spec) wrapper around _embedding_spec_for()
+    for a live gmn.Node object -- see that function's docstring for the
+    full construction.'''
+    if not hasattr(node, '_embedding_spec'):
+        node._embedding_spec = _embedding_spec_for(node.Network.Graph, node.name)
+    return node._embedding_spec
+
+
+def _generalized_embed(data, embedding_spec):
+    '''Builds a dataframe with one column per (col, delay) pair in
+    embedding_spec, each holding that column's value abs(delay) rows in
+    the past (every delay is <= 0). Named to match pyEDM.Embed()'s own
+    "col(t-k)" convention (k = abs(delay)) so downstream code that looks
+    up Parameters.target + '(t-0)' for the current-value column keeps
+    working unchanged. Returns (embedded_df, offset) with a fresh
+    0-based index -- offset is the largest lag magnitude across the
+    whole spec, i.e. how many leading rows are dropped as NaN once every
+    (col, delay) pair has been shifted into place (mirrors the
+    (E-1)*abs(tau)-style offset gmn's own Embed()-based code computed,
+    generalized to a non-uniform delay set).'''
+    columns = {
+        f'{col}(t-{abs(delay)})': data[col].shift(-delay).to_numpy()
+        for col, delay in embedding_spec
+    }
+    embedded = pd.DataFrame(columns, index=data.index)
+    offset = max(abs(delay) for _, delay in embedding_spec)
+    return embedded.iloc[offset:].reset_index(drop=True), offset
+
+
+_real_node_init = gmn.Node.__init__
+
+
+def _generalized_node_init(self, args, Network, nodeName):
+    '''Wraps gmn.Node.__init__ to correct self.libEnd_i (the EDM library
+    end index) for SMap nodes after construction -- the installed
+    package computes it as self.data.shape[0] - Parameters.E *
+    abs(Parameters.tau), which assumes one shared, uniform embedding;
+    the generalized embedding this pipeline builds instead (see
+    _node_embedding_spec()'s docstring) has its own, generally
+    different, per-node offset. Recomputed exactly once here, right
+    after __init__ sets the node's pristine self.data, since libEnd_i is
+    meant to reflect that original library size before any values get
+    appended during a Generate() run (gmn/Node.py's own docstring: "the
+    data library is limited to Parameters.predictionStart... does not
+    grow") -- not something to recompute on every later call.'''
+    _real_node_init(self, args, Network, nodeName)
+    if self.FunctionType == FunctionType.SMap:
+        spec = _node_embedding_spec(self)
+        offset = max(abs(delay) for _, delay in spec)
+        self.libEnd_i = self.data.shape[0] - offset
+
+
+gmn.Node.__init__ = _generalized_node_init
+
+
+def _generalized_node_forecast(self):
     '''Corrected copy of gmn.Node.Forecast (gmn/Forecast.py), fixing two
-    bugs found empirically in the installed package's SMap branch -- the
-    only branch this pipeline exercises, since every node uses SMap
-    uniformly (see build_gmn_config's docstring):
-
-    (1) that module calls Embed() without ever importing pyEDM.Embed
-    (confirmed: gmn/Forecast.py only does `from .Common import *`, while
-    gmn/Generate.py's identical SMap branch correctly imports it) --
-    raises NameError on the very first Forecast() call against any SMap
-    node.
-
-    (2) `val` is assigned as a 2-tuple (time, predictions) and then
-    immediately checked via `val.max() > 1E15` for divergence -- tuples
-    have no .max() method, so this raises AttributeError unconditionally,
-    not just on genuine divergence (confirmed: this fires as soon as (1)
-    is worked around, on ordinary non-diverging output). Fixed to check
-    the predictions Series specifically (val[1]).
-
-    Every other branch is copied verbatim from the installed
-    gmn/Forecast.py (unused by this pipeline, since every node is SMap,
-    but kept intact rather than trimmed so this patch doesn't silently
-    change behavior for any future non-SMap use). Monkeypatched onto
-    gmn.Node (== gmn.Node.Node -- gmn/__init__.py's `from .Node import
-    Node` shadows the submodule name with the class in gmn's own
-    namespace) rather than editing the installed package, to keep this
-    script self-contained on other machines/the cluster.'''
+    pre-existing bugs (below) and replacing its SMap branch's
+    Embed(E, tau, columns) call with _generalized_embed() -- the
+    per-driver-delay embedding built from the network graph (see
+    _node_embedding_spec()'s docstring) instead of one uniform (E, tau)
+    applied to every predictor. This is the only branch this pipeline
+    exercises, since every node uses SMap (see build_gmn_config()'s
+    docstring).
+
+    (1) gmn/Forecast.py calls Embed() without importing pyEDM.Embed (it
+    only does `from .Common import *`, unlike gmn/Generate.py's identical
+    SMap branch, which imports it correctly) -- raised NameError on the
+    first Forecast() call against any SMap node. Moot now that this
+    branch no longer calls pyEDM's Embed() at all, but noted since the
+    original bug is what first justified monkeypatching this method.
+
+    (2) `val` is a 2-tuple (time, predictions), then checked for
+    divergence via `val.max() > 1E15` -- tuples have no .max(), so this
+    raised AttributeError unconditionally, not just on genuine
+    divergence. Fixed to check the predictions Series directly (val[1]).
+
+    The time column is inserted explicitly from data's own offset-sliced,
+    reset-index time values (data[data.columns[0]].iloc[offset:].reset_
+    index(drop=True)) rather than gmn's own `df.insert(0, ...,
+    data.iloc[:,0])`, which relied on pandas' implicit index-alignment
+    between an un-reset Embed()+dropna() frame and the untouched `data`
+    -- since _generalized_embed() already resets its own index to 0-based,
+    that implicit alignment would silently line up the wrong rows here.
+
+    Every other branch is copied verbatim from gmn/Forecast.py (unused
+    here, since every node is SMap, but kept intact so this patch can't
+    silently change behavior for a future non-SMap node). Monkeypatched
+    onto gmn.Node (gmn/__init__.py's `from .Node import Node` shadows the
+    submodule name with the class) rather than editing the installed
+    package, so this script stays self-contained on other machines.'''
     if self.args.DEBUG :
         print( '-----> Node:Forecast() : ',self.FunctionType.name, self.name)
         print( self.data.tail( 2 ), flush = True );
@@ -167,12 +318,10 @@
         val = ( S.iloc[:,0], S['Predictions'] )
 
     elif self.FunctionType.value == FunctionType.SMap.value :
-        df = Embed( dataFrame = data, E = Parameters.E,
-                    tau = Parameters.tau, columns = Parameters.columns )
-
-        df = df.dropna()
+        spec = _node_embedding_spec(self)
+        df, offset = _generalized_embed(data, spec)
 
-        df.insert( 0, data.columns[0], data.iloc[:,0] )
+        df.insert( 0, data.columns[0], data[data.columns[0]].iloc[offset:].reset_index(drop=True) )
 
         if len( Parameters.lib ) and not Parameters.lib.isspace():
             lib = Parameters.lib
@@ -187,10 +336,10 @@
         S = self.Function( dataFrame       = df,
                            lib             = lib,
                            pred            = pred,
-                           E               = Parameters.E,
+                           E               = len(spec),
                            Tp              = Parameters.Tp,
                            knn             = Parameters.knn,
-                           tau             = Parameters.tau,
+                           tau             = 1,
                            theta           = Parameters.theta,
                            exclusionRadius = Parameters.exclusionRadius,
                            columns         = df.columns[1:-1],
@@ -229,7 +378,191 @@
     return val
 
 
-gmn.Node.Forecast = _fixed_node_forecast
+gmn.Node.Forecast = _generalized_node_forecast
+
+
+def _generalized_node_generate(self, lastDataOut):
+    '''Corrected copy of gmn.Node.Generate (gmn/Generate.py), replacing
+    its SMap branch's Embed(E, tau, columns) call with
+    _generalized_embed() -- see _generalized_node_forecast()'s and
+    _node_embedding_spec()'s docstrings for why. Unlike Forecast()'s
+    SMap branch, gmn's own Generate() code never tried to reuse real
+    time values for its inserted time column (just a fresh synthetic
+    0-based arange()), so there's no index-alignment subtlety to correct
+    here -- _generalized_embed()'s own 0-based index is used as-is.
+    Every other branch is copied verbatim (unused here, since every node
+    is SMap). Not a pre-existing gmn bug fix like the Forecast patch --
+    added because gmn/Generate.py's SMap branch needed the same
+    Embed()-call replacement, and this pipeline had no reason to patch
+    Node.Generate at all before this pipeline's generalized-embedding
+    feature existed.'''
+    if self.args.DEBUG :
+        print( '-----> Node:Generate() : ', self.FunctionType.name, self.name )
+        print( self.data.tail( 2 ) );
+        print( 'lastDataOut:' ); print( lastDataOut, flush = True )
+
+    if not ( lastDataOut is None ):
+        self.data = pd.concat( [ self.data, lastDataOut ] )
+
+    Parameters = self.Parameters
+    data       = self.data
+
+    if self.args.DEBUG :
+        print( '  Appended data : shape: ', data.shape );
+        print( data.tail( 3 ) ); print('', flush = True)
+
+    if self.FunctionType.value == FunctionType.Simplex.value :
+        if len( Parameters.lib ) and not Parameters.lib.isspace():
+            lib = Parameters.lib
+        else:
+            lib = "1 %d" % self.libEnd_i
+
+        if len( Parameters.pred ) and not Parameters.pred.isspace():
+            pred = Parameters.pred
+        else:
+            pred = "%d %d" % ( data.shape[0] - 1, data.shape[0] )
+
+        S = self.Function( dataFrame       = data,
+                           lib             = lib,
+                           pred            = pred,
+                           E               = Parameters.E,
+                           Tp              = Parameters.Tp,
+                           knn             = Parameters.knn,
+                           tau             = Parameters.tau,
+                           exclusionRadius = Parameters.exclusionRadius,
+                           columns         = Parameters.columns,
+                           target          = Parameters.target,
+                           embedded        = Parameters.embedded,
+                           validLib        = Parameters.validLib,
+                           generateSteps   = Parameters.generateSteps )
+
+        val = S['Predictions'].iloc[-1]
+
+    elif self.FunctionType.value == FunctionType.SMap.value :
+        spec = _node_embedding_spec(self)
+        df, offset = _generalized_embed(data, spec)
+
+        timeColumn = np.arange(df.shape[0])
+        df.insert(0, 'time', timeColumn, allow_duplicates=True)
+
+        if len( Parameters.lib ) and not Parameters.lib.isspace():
+            lib = Parameters.lib
+        else :
+            lib  = "1 %d" % self.libEnd_i
+
+        if len( Parameters.pred ) and not Parameters.pred.isspace():
+            pred = Parameters.pred
+        else:
+            pred = "%d %d" % ( df.shape[0] - 1, df.shape[0] )
+
+        S = self.Function( dataFrame       = df,
+                           lib             = lib,
+                           pred            = pred,
+                           E               = len(spec),
+                           Tp              = Parameters.Tp,
+                           knn             = Parameters.knn,
+                           tau             = 1,
+                           theta           = Parameters.theta,
+                           exclusionRadius = Parameters.exclusionRadius,
+                           columns         = df.columns[1:-1],
+                           target          = Parameters.target + '(t-0)',
+                           solver          = None,
+                           embedded        = Parameters.embedded,
+                           validLib        = Parameters.validLib,
+                           generateSteps   = Parameters.generateSteps )
+
+        val = S['predictions']['Predictions'].iloc[-1]
+
+        if abs( val ) > 1E15 :
+            raise RuntimeError( "Generate:SMap() Node: " + self.name +\
+                                "   Divergence detected (>1E15)" )
+
+    elif self.FunctionType.value == FunctionType.kedmSimplex.value :
+        data_ = data[ Parameters.columns ] # DataFrame; includes target
+
+        targetLen = Parameters.E * abs( Parameters.tau )
+        pred    = data_.iloc[ -targetLen : ]
+        library = data_.iloc[ : self.libEnd_i ]
+        target  = library[ Parameters.target ]
+
+        S = self.Function( lib    = library.to_numpy(),
+                           pred   = pred.to_numpy(),
+                           target = target.to_numpy(),
+                           E      = Parameters.E,
+                           tau    = abs( Parameters.tau ),
+                           Tp     = Parameters.Tp )
+
+        val = S[-1]
+
+    elif self.FunctionType.value == FunctionType.kedmSMap.value :
+        data_ = data[ self.name ] # Series
+
+        targetLen = Parameters.E * abs( Parameters.tau )
+        pred    = data_.iloc[ -targetLen : ]
+        library = data_.iloc[ : self.libEnd_i ]
+        target  = library
+
+        S = self.Function( lib    = library.to_numpy(),
+                           pred   = pred.to_numpy(),
+                           target = target.to_numpy(),
+                           E      = Parameters.E,
+                           tau    = abs( Parameters.tau ),
+                           Tp     = Parameters.Tp,
+                           theta  = Parameters.theta )
+
+        val = S[-1]
+
+    elif self.FunctionType.value == FunctionType.Linear.value :
+        X = data[ Parameters.columns ]
+        X = X.to_numpy()
+        y = data[ Parameters.target ].to_numpy()
+
+        lr = self.Function()
+        lr.fit( X, y )
+
+        nextX = self.FindNextX( X )
+        pred  = lr.predict( nextX )
+        val   = pred[ 0 ]
+
+    elif self.FunctionType.value == FunctionType.SVR.value :
+        X = data[ Parameters.columns ]
+        X = X.to_numpy()
+        y = data[ Parameters.target ].to_numpy()
+
+        svr = self.Function( kernel = 'rbf', gamma = 'scale', tol = 0.001,
+                             epsilon = 0.1, shrinking = True, cache_size = 200,
+                             verbose = False, max_iter = -1 )
+        svr.fit( X, y )
+
+        nextX = self.FindNextX( X )
+        pred  = svr.predict( nextX )
+        val   = pred[ 0 ]
+
+    elif self.FunctionType.value == FunctionType.knn.value :
+        X = data[ Parameters.columns ]
+
+        if len( Parameters.columns ) > 1 :
+            X  = X.drop( self.name, axis = 'columns' )
+
+        X = X.to_numpy()
+        y = data[ Parameters.target ].to_numpy()
+
+        knn = self.Function( n_neighbors = 10, weights = 'distance' )
+        knn.fit( X, y )
+
+        nextX = self.FindNextX( X )
+        pred  = knn.predict( nextX )
+        val   = pred[ 0 ]
+
+    if self.args.DEBUG :
+        print( self.name, "val:", val )
+        print( '<----- Node:Generate() : ',self.FunctionType.name,self.name )
+        print('', flush = True)
+
+    return val
+
+
+gmn.Node.Generate = _generalized_node_generate
 
 
 def _sequential_ccm_workers(parallel):
@@ -237,39 +570,27 @@
     CCM().Project() call in this process onto its sequential in-process
     path instead of spawning a fresh multiprocessing Pool.
 
-    This is the actual fix for the nested-pool blowup documented in this
-    module's KNOWN CAVEAT and in build_network()'s docstring below: every
-    single pyEDM.CCM() call -- one per channel pair in
-    apps.InteractionMatrix, one per MDE candidate column in dimx.Run --
-    creates its *own* fresh `spawn`/`forkserver` Pool sized to
-    cpu_count() (CCM.py's `_resolve_workers`, called from `Project()`),
-    on top of whichever outer pool (apps.InteractionMatrix's
-    ProcessPoolExecutor, dimx's CrossMapColumns Pool) is already
-    dispatching that call. Neither apps.InteractionMatrix nor dimx.Run
-    ever pass `parallel=` through to CCM(), so this can't be tuned via
-    their CLIs -- confirmed empirically (see the "No module named
-    'kedm'" line in a job's .out, printed once per spawned worker
-    process re-importing gmn from scratch: hundreds of thousands of
-    occurrences on a single run) and by reading pyEDM/CCM.py directly.
-
-    Forcing sequential execution here is strictly a win, not a
-    parallelism/wall-time tradeoff: every CCM() call in this pipeline
-    resolves to only 4-8 tiny tasks (2 directions x 2-4 libSizes -- see
-    apps/InteractionMatrix.py's libSizes=[libMin,libMax] and dimx's
-    default pLibSizes=[10,15,85,90]), each a KDTree build+query over a
-    handful of embedding rows. Running those 4-8 tasks in-process is
-    faster than the Pool spawn + full heavy-import (numpy/scipy/pandas/
-    sklearn/pyEDM/gmn) teardown cost alone, before any actual work
-    happens -- and the real parallelism across channel pairs/candidates
-    already comes from the outer pool, which this doesn't touch.
-    Monkeypatched rather than `parallel=False` at each call site because
-    apps.InteractionMatrix constructs CCM() internally and doesn't expose
-    that argument at all (dimx.Run's own CCM() call gets a second,
-    independent belt-and-suspenders patch below -- see
-    _sequential_ccm_api -- that does force parallel=False explicitly;
-    this _resolve_workers override is the one thing that still protects
-    apps.InteractionMatrix's own CCM() calls, and is kept as the first
-    line of defense for dimx.Run too).'''
+    This is the fix for the nested-pool blowup described in the module
+    docstring: every pyEDM.CCM() call (one per channel pair in
+    apps.InteractionMatrix, one per candidate column in dimx.Run) used to
+    spawn its own fresh Pool sized to cpu_count(), on top of whichever
+    outer pool was already dispatching that call -- confirmed by reading
+    pyEDM/CCM.py and by the "No module named 'kedm'" line each spawned
+    worker printed on reimport, hundreds of thousands of times in one run.
+    Neither apps.InteractionMatrix nor dimx.Run ever pass `parallel=`
+    through to CCM(), so this couldn't be tuned via their CLIs.
+
+    Forcing sequential execution is a strict win, not a tradeoff: every
+    CCM() call here only ever resolves to 4-8 tiny KDTree tasks (2
+    directions x 2-4 libSizes), so a fresh Pool's spawn-and-reimport cost
+    dwarfs the actual work. Real parallelism across channel pairs/
+    candidates already comes from the outer pool, which this doesn't
+    touch. Monkeypatched (rather than passing parallel=False at each call
+    site) because apps.InteractionMatrix constructs CCM() internally with
+    no way to pass that argument in; dimx.Run's own CCM() call gets a
+    second, independent patch below (_sequential_ccm_api) that does force
+    parallel=False explicitly, so this override is the sole protection
+    for apps.InteractionMatrix and a belt-and-suspenders one for dimx.'''
     return 1
 
 
@@ -278,53 +599,155 @@
 
 import dimx.Run as _dimx_run_module
 _real_pyedm_ccm_api = _dimx_run_module.CCM
+_real_cross_map_columns_api = _dimx_run_module.CrossMapColumns
+_real_embed_dimension_api = _dimx_run_module.EmbedDimension
+
+# Set from load_config() (MDE_GMN_EXCLUSION_RADIUS=auto, the default) --
+# gates the three dynamic-exclusion-radius wrappers below. Module-level
+# rather than threaded through call args because dimx.Run's own internal
+# CrossMapColumns/EmbedDimension/CCM calls are what's being intercepted
+# here, not something this pipeline calls directly. A plain global is
+# safe across the nested-MDE wave pool: that pool is always created with
+# mp_context=mp.get_context('fork') (see build_nested_mde_network()),
+# which clones this process's memory -- including this flag's already-
+# resolved value -- rather than re-executing imports the way forkserver
+# would.
+_EXCLUSION_RADIUS_AUTO = False
+
+
+def _delay_from_column_name(name):
+    '''DELAY_MARKER/_parse_delayed_column_name are defined later in this
+    file (expand_delay_columns() section) -- fine as a forward reference
+    since this is only ever called at runtime, long after the whole
+    module has finished loading. Delay=0 (contemporaneous) columns have
+    no ::delay suffix at all (see expand_delay_columns()'s docstring), so
+    absence of DELAY_MARKER means delay 0, not "not a candidate column"
+    -- every name reaching here (from dimx.Run's own `columns` args) is
+    already known to be one.'''
+    if DELAY_MARKER in name:
+        return _parse_delayed_column_name(name)[1]
+    return 0
+
+
+def _dynamic_exclusion_radius(columns):
+    '''Auto exclusion-radius (MDE_GMN_EXCLUSION_RADIUS=auto, the
+    default): approximates the classic Theiler-window guideline
+    exclusionRadius >= E*tau for dimx.MDE's forward selection over this
+    pipeline's *generalized* (multi-delay, DELAY_SET-driven) embedding --
+    where E and tau are confounded, per explicit direction, because
+    dimx.Run() grows a genuinely non-uniform embedding one selected
+    column at a time, and each column can carry its own independent
+    DELAY_SET-encoded delay rather than one lag shared network-wide.
+    Adding a new delayed observable always grows E by exactly 1, but the
+    embedding's own effective "tau" is whatever the single largest delay
+    among ALL its dimensions happens to be -- not a fixed per-step
+    increment -- so a single static E*tau computed once upfront can't
+    track it correctly.
+
+    Rather than a fixed a-priori bound, this is called fresh at every
+    dimx.Run() internal CrossMapColumns()/EmbedDimension()/CCM() call
+    (from the monkeypatched wrappers below, replacing dimx.Run's own
+    module-level names -- same pattern as _sequential_ccm_api above) and
+    reads dimension/delay directly off whatever `columns` argument that
+    specific call was made with:
+      - CrossMapColumns' `columns` is dimx.Run's own nested list of
+        [candidate] + already-selected-MDEcolumns lists (one per
+        candidate still being ranked at the current forward-selection
+        step) -- exactly "the multivariate embedding as it's being built
+        up", so this naturally grows across dimx.Run()'s d=1..D loop as
+        MDEcolumns accumulates: always the largest tau *actually selected
+        so far*, not a worst-case bound.
+      - EmbedDimension/CCM's `columns` is a single already-chosen
+        candidate column name (dimx.Run validates one column at a time
+        here) -- treated as its own 1-dimensional embedding.
+    Delays are parsed straight from each column's own ::delay<N> suffix
+    (_delay_from_column_name()) -- exactly the same DELAY_SET-driven
+    lag every column already carries via expand_delay_columns(), not a
+    separately-modeled quantity.
+
+    Returns dim * max_abs_delay, where dim is the size of the largest
+    embedding evaluated in this call and max_abs_delay is the largest
+    delay (in rows) among every column in it.'''
+    if isinstance(columns, str):
+        groups = [[columns]]
+    elif columns and isinstance(columns[0], str):
+        groups = [list(columns)]
+    else:
+        groups = [list(group) for group in columns]
+
+    max_dim = max((len(group) for group in groups), default=0)
+    max_delay = max(
+        (abs(_delay_from_column_name(c)) for group in groups for c in group),
+        default=0,
+    )
+    return max_dim * max_delay
+
+
+def _auto_exclusion_radius_cross_map_columns(*args, **kwargs):
+    '''Replaces dimx.Run's CrossMapColumns binding -- see
+    _dynamic_exclusion_radius()'s docstring. A no-op passthrough when
+    MDE_GMN_EXCLUSION_RADIUS is a fixed integer rather than 'auto' (the
+    ordinary case dimx.MDE's own exclusionRadius kwarg already handles
+    correctly on its own).'''
+    if _EXCLUSION_RADIUS_AUTO and 'columns' in kwargs:
+        kwargs['exclusionRadius'] = _dynamic_exclusion_radius(kwargs['columns'])
+    return _real_cross_map_columns_api(*args, **kwargs)
+
+
+def _auto_exclusion_radius_embed_dimension(*args, **kwargs):
+    '''Replaces dimx.Run's EmbedDimension binding -- see
+    _auto_exclusion_radius_cross_map_columns()'s docstring, same
+    reasoning, applied to the single-column E-search dimx.Run() runs on
+    each forward-selection step's winning candidate.'''
+    if _EXCLUSION_RADIUS_AUTO and 'columns' in kwargs:
+        kwargs['exclusionRadius'] = _dynamic_exclusion_radius(kwargs['columns'])
+    return _real_embed_dimension_api(*args, **kwargs)
 
 
 def _sequential_ccm_api(*args, **kwargs):
-    '''Second, independent layer on top of _sequential_ccm_workers above:
-    forces parallel=False as an explicit argument to pyEDM's top-level
-    CCM() API function (kwargs['parallel'] -- CCM.py's own
-    _resolve_workers(parallel) unconditionally returns 1 when
-    parallel is False, no override needed for that half). Patches the
-    `CCM` name inside dimx.Run's own namespace directly (dimx/Run.py does
-    `from pyEDM import ... CCM ...`, so `dimx.Run.CCM` is the exact
-    binding dimx.MDE.Run() actually calls) rather than only the
-    pyEDM.CCM._resolve_workers indirection above, in case that patch
-    doesn't reach every call the same way in every process.'''
+    '''Second, independent layer on top of _sequential_ccm_workers: forces
+    parallel=False on pyEDM's top-level CCM() function directly. Patches
+    the `CCM` name inside dimx.Run's own namespace (dimx/Run.py imports it
+    as `from pyEDM import ... CCM ...`, so `dimx.Run.CCM` is the exact
+    binding dimx.MDE.Run() calls) rather than relying only on the
+    _resolve_workers indirection above, in case that patch doesn't reach
+    every call the same way in every process. Also applies the dynamic
+    exclusion radius (see _auto_exclusion_radius_cross_map_columns()
+    above) to dimx.Run's own CCM() causality-validation call, same gating
+    on _EXCLUSION_RADIUS_AUTO.'''
+    if _EXCLUSION_RADIUS_AUTO and 'columns' in kwargs:
+        kwargs['exclusionRadius'] = _dynamic_exclusion_radius(kwargs['columns'])
     kwargs['parallel'] = False
     return _real_pyedm_ccm_api(*args, **kwargs)
 
 
 _dimx_run_module.CCM = _sequential_ccm_api
+_dimx_run_module.CrossMapColumns = _auto_exclusion_radius_cross_map_columns
+_dimx_run_module.EmbedDimension = _auto_exclusion_radius_embed_dimension
 
 
 def _mde_wave_worker_init():
-    '''ProcessPoolExecutor(initializer=...) hook for build_nested_mde_network's
-    per-wave worker pool -- reapplies both CCM patches above explicitly
-    inside every worker process, rather than assuming either carries over
-    from this module's own top-level monkeypatch lines.
-
-    That assumption doesn't hold reliably on this cluster, and the
-    failure mode is nasty, not just a silent perf regression: a worker
-    whose dimx.MDE -> CCM() call ends up on the unpatched path tries to
-    create CCM's own internal multiprocessing Pool *from inside an
-    already-pooled worker process*, and that nested Pool creation fails
-    outright with "ChildProcessError: No child processes" rather than
-    merely being slow -- confirmed by direct diagnostics (printing
-    _resolve_workers' identity and a live test call immediately before
-    m.Run(), in the same worker process about to crash). The exact
-    trigger was never fully pinned down (diagnostics showed the
-    _resolve_workers override alone correctly returning 1 when checked
-    directly, yet some workers still hit the unpatched path deeper inside
-    Project()'s own call chain) -- what's confirmed empirically is that
-    applying *both* independent patches (the _resolve_workers override
-    AND the dimx.Run.CCM parallel=False wrapper) together, freshly inside
-    every worker via this initializer, reliably avoids it across repeated
-    test runs, while either alone did not.'''
+    '''ProcessPoolExecutor(initializer=...) hook for
+    build_nested_mde_network()'s per-wave worker pool -- reapplies all
+    three dimx.Run patches above (CCM parallelism + exclusion radius,
+    CrossMapColumns/EmbedDimension exclusion radius) inside every worker
+    process, rather than assuming they survive from this module's own
+    top-level monkeypatch lines (they don't reliably on this cluster).
+
+    The failure mode when it doesn't is a hard crash, not just a slowdown:
+    a worker whose dimx.MDE -> CCM() call lands on the unpatched path
+    tries to create CCM's own Pool from inside an already-pooled worker
+    process, which fails outright with "ChildProcessError: No child
+    processes". The exact trigger was never fully pinned down, but
+    applying these patches together, fresh inside every worker via this
+    initializer, reliably avoided it across repeated test runs, while
+    applying only a subset did not.'''
     import pyEDM.CCM as _ccm
     _ccm._resolve_workers = _sequential_ccm_workers
     import dimx.Run as _dimx_run
     _dimx_run.CCM = _sequential_ccm_api
+    _dimx_run.CrossMapColumns = _auto_exclusion_radius_cross_map_columns
+    _dimx_run.EmbedDimension = _auto_exclusion_radius_embed_dimension
 
 
 # =============================================================================
@@ -354,6 +777,13 @@
     return [float(x.strip()) for x in val.split(',') if x.strip()]
 
 
+def _env_int_list(name, default):
+    val = os.environ.get(name)
+    if val is None:
+        return list(default)
+    return [int(x.strip()) for x in val.split(',') if x.strip()]
+
+
 def load_config():
     cfg = {}
 
@@ -375,46 +805,129 @@
     cfg['TARGETS'] = _env_list('MDE_GMN_TARGETS', ['head_position_x', 'head_position_y'])
     cfg['SECOND_TARGET'] = cfg['TARGETS'][1] if len(cfg['TARGETS']) > 1 else None
 
-    # Lib/pred split, expressed as the fraction of the full dataset used as
-    # library -- e.g. LIB_RATIO=0.5 means the first 50% of rows are lib,
-    # the remaining 50% are the held-out pred/Generate() window. The whole
-    # dataset is used (no separate row-count cap); PREDICTION_START/
-    # PREDICTION_LENGTH (row counts derived from this ratio and the loaded
-    # data's length) are filled in by main() once df is loaded, and are
-    # what the rest of the pipeline actually reads (MDE lib/pred split,
-    # CCM/EmbedDimension calls, and the held-out window Generate() results
-    # are validated against).
+    # Lib/pred split as a fraction of the full dataset -- e.g. LIB_RATIO=0.5
+    # means the first 50% of rows are library, the remaining 50% are the
+    # held-out pred/Generate() window. PREDICTION_START/PREDICTION_LENGTH
+    # (row counts derived from this ratio and the data's length) are filled
+    # in by main() once the data is loaded, and are what the rest of the
+    # pipeline reads: the MDE lib/pred split, CCM/EmbedDimension calls, and
+    # the held-out window Generate() results are validated against.
     cfg['LIB_RATIO'] = float(os.environ.get('MDE_GMN_LIB_RATIO', 0.5))
 
-    # GMN Generate() step count, decoupled from PREDICTION_LENGTH above --
-    # every Generate() call (both inside the E/tau/theta search loop and
-    # the final rebuild) always still *starts* at PREDICTION_START, but
-    # only runs this many steps rather than the entire held-out window.
-    # Generate() is the most expensive step in the search loop (one full
-    # network simulation per (E, tau[, theta]) candidate), so this lets
-    # you shrink that cost independently of LIB_RATIO/MDE's own held-out
-    # window, which the causality gating (CCM, EmbedDimension) still needs
-    # to be a meaningful size. 0 (default) means "use the full
-    # PREDICTION_LENGTH", i.e. unchanged pre-existing behavior; a value
-    # larger than PREDICTION_LENGTH is clamped down to it in main() once
-    # the dataset is loaded and PREDICTION_LENGTH is known (a longer
-    # Generate() than there is real held-out data to compare against isn't
-    # meaningful -- there'd be nothing to score the extra steps against).
+    # Step count for each Generate() call, decoupled from PREDICTION_LENGTH:
+    # every Generate() call (in the theta search, the start sweep, and
+    # the final rebuild) runs this many steps, not the entire held-out
+    # window. Generate() is the single most expensive step in this
+    # pipeline (one full network simulation per call), so this shrinks that
+    # cost independently of LIB_RATIO/MDE's own held-out window, which the
+    # causality gating (CCM, EmbedDimension) still needs to be a meaningful
+    # size. Also caps the Forecast() diagnostic's pred window (see
+    # build_gmn_config()'s docstring) so actual/Generate()/Forecast() all
+    # cover the same window on that plot. 0 (default) means "use the full
+    # PREDICTION_LENGTH"; a larger value is clamped down to it in main()
+    # once PREDICTION_LENGTH is known (there'd be no real data to score any
+    # extra steps against).
     cfg['GENERATE_LENGTH'] = int(os.environ.get('MDE_GMN_GENERATE_LENGTH', 0))
 
+    # Theiler-window / neighbor-exclusion radius for gmn's own recursive
+    # Generate()/Forecast() nearest-neighbor search (SMap), passed through
+    # as Parameters.exclusionRadius by build_gmn_config() -- independent
+    # of MDE_GMN_EXCLUSION_RADIUS above, which only affects dimx.MDE's own
+    # forward-selection search. Named without the MDE_GMN_ prefix
+    # (deliberately, per explicit direction) to keep it visually distinct
+    # from that unrelated MDE-side parameter. Static -- unlike MDE's
+    # 'auto' mode, there's no equivalent generalized/non-uniform embedding
+    # confound here since every node uses one shared theta-driven SMap
+    # embedding (see build_gmn_config()'s function='SMap' docstring), so
+    # 0 (default, matches this pipeline's pre-existing hardcoded
+    # behavior) is a reasonable starting point -- change it to explore
+    # sensitivity to temporal-neighbor exclusion in the generative step.
+    cfg['GMN_EXCLUSION_RADIUS'] = int(os.environ.get('GMN_EXCLUSION_RADIUS', 0))
+
+    # Plotted x-axis range for the final Generate()-vs-Forecast()-vs-
+    # actual diagnostic figure (run_forecast_diagnostic()) -- see that
+    # function's plot_range_mode docstring for the full behavior of each
+    # option. 'full' (default, unchanged pre-existing behavior): actual/
+    # lib plotted from row 0 through generate_start+GENERATE_LENGTH,
+    # Forecast() computes exactly GENERATE_LENGTH steps. 'windowed': the
+    # window narrows to 2x GENERATE_LENGTH on each side of generate_start
+    # (clamped to available data) -- actual/lib fills the whole window
+    # (cheap, real data), Forecast() is run out to that same 2x length
+    # (cheap, a direct per-row SMap prediction against real held-out
+    # data, not a recursive simulation), Generate() itself is left
+    # unchanged. Only affects this one diagnostic plot/Forecast() call --
+    # the theta search, start-point sweep, and Generate() rebuild all use
+    # GENERATE_LENGTH exactly as before regardless of this setting.
+    cfg['FORECAST_PLOT_RANGE'] = os.environ.get('MDE_GMN_FORECAST_PLOT_RANGE', 'full').strip().lower()
+    if cfg['FORECAST_PLOT_RANGE'] not in ('full', 'windowed'):
+        raise ValueError(
+            f"MDE_GMN_FORECAST_PLOT_RANGE must be 'full' or 'windowed', "
+            f"got {cfg['FORECAST_PLOT_RANGE']!r}"
+        )
+
+    # Candidate starting rows for the final Generate()/Forecast() diagnostic,
+    # as offsets (rows) from PREDICTION_START -- e.g. MIN=0, MAX=200,
+    # STEP=50 tries offsets 0, 50, 100, 150 (MAX exclusive, like E_MAX/
+    # TAU_MAX below). sweep_generate_start() runs one Generate() call per
+    # offset at the already-chosen best theta (E and every driver's delay
+    # come from nested-MDE itself -- see the DELAY_SET comment below --
+    # not re-searched) and keeps whichever start best matches the real
+    # data: Generate() is a recursive nonlinear simulation, so which row
+    # it launches from can change how well it tracks reality, independent
+    # of how good the embedding/theta are. Offsets must be >= 0 (never
+    # before PREDICTION_START) so every candidate's comparison data stays
+    # genuinely out-of-sample. Default MIN=0, MAX=1 tries only offset 0
+    # (always PREDICTION_START, unchanged pre-existing behavior) -- widen
+    # MAX to actually sweep. Offsets that would leave too little data to
+    # score GENERATE_LENGTH steps against are dropped once
+    # PREDICTION_LENGTH is known (main()).
+    cfg['GENERATE_START_MIN_OFFSET'] = int(os.environ.get('MDE_GMN_GENERATE_START_MIN_OFFSET', 0))
+    cfg['GENERATE_START_MAX_OFFSET'] = int(os.environ.get('MDE_GMN_GENERATE_START_MAX_OFFSET', 1))
+    cfg['GENERATE_START_STEP'] = int(os.environ.get('MDE_GMN_GENERATE_START_STEP', 50))
+
     # ---- MDE (variable selection; unused when MODE='GMN') -------------
+    # MDE_D counts *real* observables, not delay-expanded candidate
+    # columns -- e.g. with DELAY_SET holding 5 delays, MDE_D=5 means "up
+    # to 5 distinct real observables", not "5 candidate columns" (which
+    # could all be different delays of the very same observable).
+    # effective_mde_dimension() converts this into the actual D passed to
+    # dimx.MDE (MDE_D * len(DELAY_SET), capped at however many real
+    # observables that call's own candidate pool actually has) -- see
+    # that function's docstring. Setting MDE_D >= the real observable
+    # count (e.g. 999 against 500 real observables) makes MDE consider
+    # all of them, capped down rather than inflating D past what the pool
+    # can supply.
     cfg['MDE_D'] = int(os.environ.get('MDE_GMN_D', 20))
     cfg['EMBED_DIM_RHO_MIN'] = float(os.environ.get('MDE_GMN_EMBED_DIM_RHO_MIN', 0.2))
     cfg['CROSS_MAP_RHO_MIN'] = float(os.environ.get('MDE_GMN_CROSS_MAP_RHO_MIN', 0.2))
+    # 'auto' (default): dynamic per-dimx.Run()-call exclusion radius --
+    # see _dynamic_exclusion_radius()'s docstring near this file's
+    # monkeypatch block (just above _sequential_ccm_api) for the full
+    # E*tau-confounded-by-DELAY_SET rationale. Set an explicit integer
+    # instead to disable that and use dimx.MDE's ordinary fixed
+    # exclusionRadius on every call.
+    _raw_exclusion_radius = os.environ.get('MDE_GMN_EXCLUSION_RADIUS', 'auto').strip()
+    cfg['EXCLUSION_RADIUS_AUTO'] = _raw_exclusion_radius.lower() == 'auto'
+    cfg['EXCLUSION_RADIUS_VALUE'] = (
+        0 if cfg['EXCLUSION_RADIUS_AUTO'] else int(_raw_exclusion_radius)
+    )
+    global _EXCLUSION_RADIUS_AUTO
+    _EXCLUSION_RADIUS_AUTO = cfg['EXCLUSION_RADIUS_AUTO']
     # Nested-MDE network construction (MODE='MDE-GMN' only -- see
     # build_nested_mde_network()'s docstring for the full algorithm).
     # Reuses MDE_D as the per-node driver cap N at every nesting level
     # (deliberately not a separate parameter -- MDE_D already means "max
-    # # of driver vars MDE can forward-select", which is exactly N here
-    # too). NESTED_M is the number of *additional* nested-MDE rounds
+    # # of real observables MDE can forward-select", which is exactly N
+    # here too). NESTED_M is the number of *additional* nested-MDE rounds
     # beyond the initial per-target selection -- total graph depth from
-    # each target is 1 + NESTED_M levels. Combinatorially expensive: a
-    # naive upper bound on nodes is roughly MDE_D**(1+NESTED_M) before
+    # each target is 1 + NESTED_M levels. NESTED_M=0 is a sentinel for
+    # "uncapped": nesting keeps going, round after round, until it runs
+    # out on its own (every reachable real observable already visited, or
+    # dimx.MDE/cycle-rejection naturally dead-ends every remaining
+    # branch) instead of stopping at a fixed round count -- see
+    # build_nested_mde_network()'s docstring. Combinatorially expensive: a
+    # naive upper bound on nodes is roughly MDE_D**(1+NESTED_M) (or, when
+    # NESTED_M=0, bounded only by the real observable pool size) before
     # dedup/cycle-rejection collapses it back down to the real (capped)
     # observable pool size -- keep MDE_D and NESTED_M small for a first
     # test run.
@@ -425,10 +938,37 @@
     # down for a fast test job.
     cfg['REGION_CHANNEL_CAP'] = int(os.environ.get('MDE_GMN_REGION_CHANNEL_CAP', 999))
 
-    # ---- E/tau/theta search ranges -------------------------------------
-    # E >= 2 always: every GMN node uses SMap uniformly (see
-    # build_gmn_config's docstring), and SMap's predictor-exclusion
-    # convention needs E >= 2 even for leaf/undriven nodes.
+    # Delay set nested-MDE searches per observable (MODE='MDE-GMN' only):
+    # each candidate observable is offered to dimx.MDE's forward selection
+    # as len(DELAY_SET) separate columns, one per delay -- e.g. 200
+    # observables x 5 delays = 1000 candidate columns -- so MDE picks
+    # specific (observable, delay) pairs rather than being stuck with one
+    # fixed contemporaneous (delay=0) copy of every observable. This is a
+    # generalized/non-uniform Takens embedding (Deyle & Sugihara 2011):
+    # different driving variables, and even repeats of the *same* driver,
+    # can each carry their own delay rather than sharing one embedding
+    # depth/lag network-wide. Values must be <= 0 (0 = contemporaneous,
+    # negative = that many rows in the past) -- a positive delay would
+    # need future data that doesn't exist during a real free-running
+    # Generate() call. See build_nested_mde_network()'s docstring for how
+    # a node's per-driver delay is recorded and how nesting only recurses
+    # through delay=0 selections. 0 is always included in every node's
+    # own self-embedding regardless of whether it's listed here (see
+    # build_network()'s self_delay_set comment) -- a node's present
+    # value is the anchor SMap predicts from/to, not an optional driver
+    # delay choice, so it can't be omitted even from a DELAY_SET that
+    # doesn't otherwise include 0.
+    cfg['DELAY_SET'] = _env_int_list('MDE_GMN_DELAY_SET', [0, -1, -3, -5, -10])
+    if any(d > 0 for d in cfg['DELAY_SET']):
+        raise ValueError(f"MDE_GMN_DELAY_SET must be <= 0 (no future data): {cfg['DELAY_SET']!r}")
+
+    # ---- E/tau search range (MODE='GMN' only) --------------------------
+    # Used exclusively by build_network()'s mode == 'GMN' branch (the CCM
+    # interaction-matrix E/tau grid search, via target_E_tau_rho_grid()) --
+    # MODE='MDE-GMN' no longer has a network-wide E/tau at all (E and each
+    # driver's delay come from nested-MDE/DELAY_SET above instead; see
+    # this module's SEARCH REUSE section and run_theta_search()'s
+    # docstring for what replaced the old combined E/tau/theta search).
     cfg['E_RANGE'] = range(
         int(os.environ.get('MDE_GMN_E_MIN', 3)),
         int(os.environ.get('MDE_GMN_E_MAX', 16)),
@@ -439,12 +979,14 @@
         int(os.environ.get('MDE_GMN_TAU_MAX', 0)),
         int(os.environ.get('MDE_GMN_TAU_STEP', 2)),
     )
+
+    # Theta (SMap's nonlinearity/localization parameter -- see
+    # retune_theta_for_all_nodes()'s docstring) search range, used by
+    # run_theta_search() for both MODE='MDE-GMN' and MODE='GMN'.
     cfg['THETA_RANGE'] = _env_float_list(
         'MDE_GMN_THETA_RANGE',
         [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],
     )
-    cfg['COARSE_THETA_RANGE'] = cfg['THETA_RANGE'][::3]
-    cfg['SMAP_SEARCH_MODE'] = os.environ.get('MDE_GMN_SMAP_SEARCH_MODE', 'coarse_retune')  # 'coarse_retune' | 'joint'
 
     # ---- MDE-only diagnostics (skipped when MODE='GMN') ----------------
     cfg['MAX_TP'] = int(os.environ.get('MDE_GMN_MAX_TP', 20))
@@ -455,7 +997,7 @@
     # On HPC this should be SLURM_CPUS_PER_TASK, not left at each tool's
     # own package default (dimx: 5, apps.InteractionMatrix: 8, pyEDM
     # EmbedDimension/PredictNonlinear: their own internal default) -- see
-    # module docstring's KNOWN CAVEAT for why this doesn't scale linearly.
+    # module docstring's FIXED section for why this doesn't scale linearly.
     cfg['CORES'] = int(os.environ.get('MDE_GMN_CORES', os.cpu_count() or 1))
     # Deliberately NOT the same as CORES -- see build_network()'s
     # docstring: apps.InteractionMatrix's -cr and pyEDM.CCM()'s own
@@ -463,6 +1005,19 @@
     # stays small regardless of how large CORES is.
     cfg['INTERACTION_MATRIX_OUTER_CORES'] = int(os.environ.get('MDE_GMN_INTERACTION_MATRIX_OUTER_CORES', 2))
 
+    # ---- Search reuse ---------------------------------------------------
+    # See the module docstring's SEARCH REUSE section for the full
+    # rationale; short version: reuse a past job's E/tau/theta search
+    # result instead of redoing it, when settings match exactly.
+    cfg['REUSE_BEST_PARAMS'] = _env_bool('MDE_GMN_REUSE_BEST_PARAMS', False)
+    # Where to look for previous jobs' best_params_*.json files. Default
+    # mirrors DATA_DIR/RESULTS_DIR's own '../../data'-style fallback, but
+    # the sbatch script overrides this to the persistent $PROJ_DIR/results
+    # -- unlike RESULTS_DIR, which the sbatch script points at a /flash
+    # scratch tempdir deleted at job exit, PREV_RUNS_DIR needs to point at
+    # wherever past jobs' outputs actually still exist.
+    cfg['PREV_RUNS_DIR'] = Path(os.environ.get('MDE_GMN_PREV_RUNS_DIR', '../../results'))
+
     cfg['TARGET_COLORS'] = {t: plt.get_cmap('tab10')(i % 10) for i, t in enumerate(cfg['TARGETS'])}
 
     return cfg
@@ -485,11 +1040,99 @@
     )
 
 
+DELAY_MARKER = '::delay'
+
+
+def _delayed_column_name(observable, delay):
+    return f'{observable}{DELAY_MARKER}{delay}'
+
+
+def _parse_delayed_column_name(name):
+    observable, _, delay_str = name.rpartition(DELAY_MARKER)
+    return observable, int(delay_str)
+
+
+def observable_and_delay(column_name, raw_columns):
+    '''Maps a (possibly delay-expanded) candidate column name back to its
+    underlying (observable, delay) pair. Delay=0 candidates are just the
+    raw observable name itself (see expand_delay_columns()'s docstring
+    for why that delay isn't also given a suffixed alias) -- checked
+    against `raw_columns` rather than by string shape alone, in case a
+    real channel name were ever to collide with the delay-suffix marker.
+    '''
+    if column_name in raw_columns:
+        return column_name, 0
+    return _parse_delayed_column_name(column_name)
+
+
+def expand_delay_columns(df, columns, delay_set):
+    '''Adds one new column per (observable, delay) pair in `columns` x
+    `delay_set` (delay=0 excluded -- the observable's own raw column
+    already IS that pair, no separate alias needed) to a copy of `df`,
+    each holding that observable's value abs(delay) rows in the past
+    (pandas .shift() -- every delay is <= 0, see load_config()'s
+    DELAY_SET docstring). This is what lets dimx.MDE's forward selection
+    treat each (observable, delay) combination as its own independent
+    candidate driver, instead of being stuck with one fixed lag applied
+    uniformly to every column -- see build_nested_mde_network()'s
+    docstring for how the selected pairs become graph edges.
+
+    Returns (expanded_df, expanded_columns) -- expanded_columns is
+    `columns` (representing delay=0) plus one new name per nonzero delay,
+    in the format observable_and_delay() parses back.'''
+    expanded = df.copy()
+    expanded_columns = list(columns)
+    for observable in columns:
+        for delay in delay_set:
+            if delay == 0:
+                continue
+            name = _delayed_column_name(observable, delay)
+            expanded[name] = df[observable].shift(-delay)
+            expanded_columns.append(name)
+    return expanded, expanded_columns
+
+
+def effective_mde_dimension(mde_d, candidate_columns, num_delays):
+    '''MDE_D names the desired count of distinct *real* observables
+    dimx.MDE should be able to select -- not delay-expanded columns. Each
+    real observable occupies `num_delays` separate candidate columns (one
+    per DELAY_SET entry, see expand_delay_columns()), so the D actually
+    handed to dimx.MDE is scaled up by num_delays: even in the worst case
+    of every selection slot landing on a different delay of the same
+    observable before forward selection moves on to the next one, mde_d
+    distinct real observables are still guaranteed to be reachable.
+
+    Capped at the number of real observables actually present in this
+    call's own candidate pool (`len(candidate_columns) // num_delays` --
+    exact, not approximate: expand_delay_columns() always emits exactly
+    num_delays columns per real observable, and candidate-pool filtering
+    upstream (observable_and_delay()-based) only ever drops a real
+    observable's full column group at once, never a partial one) -- so
+    setting MDE_D larger than the pool's real observable count (e.g. 999
+    against a 500-observable pool) just means "select all of them" rather
+    than inflating D past what the pool can supply.'''
+    num_real_observables_available = len(candidate_columns) // num_delays
+    return min(mde_d, num_real_observables_available) * num_delays
+
+
 def run_mde_for_target(target, df, observable_columns, prediction_start, prediction_length,
-                        mde_d, embed_dim_rho_min, cross_map_rho_min, cores, results_dir):
+                        mde_d, embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
+                        cores, results_dir):
     '''Run dimx.MDE's forward selection for a single target variable.
-    See notebooks/simple-mde-gmn.ipynb cell 2's run_mde_for_target for the
-    full rationale (identical logic, just cores threaded through).'''
+    `mde_d` here is already the *effective* dimension count (see
+    effective_mde_dimension()) -- callers compute that from the
+    real-observable-count MDE_D before calling this, since only they know
+    each call's own candidate pool / DELAY_SET. `exclusion_radius` is
+    only actually used as-is when MDE_GMN_EXCLUSION_RADIUS is a fixed
+    integer -- when it's 'auto' (the default), the module-level
+    _EXCLUSION_RADIUS_AUTO-gated monkeypatches (see that flag's docstring
+    near the top of this file) override whatever value dimx.MDE is
+    called with here on every one of its own internal
+    CrossMapColumns/EmbedDimension/CCM calls, so this argument is a
+    harmless placeholder (0) in that case. See
+    notebooks/simple-mde-gmn.ipynb cell 2's run_mde_for_target for the
+    original single-delay rationale (identical logic otherwise, just
+    cores threaded through).'''
     mde_df = df[[target] + observable_columns].iloc[:prediction_start + prediction_length]
 
     m = dx.MDE( mde_df, target = target, D = mde_d,
@@ -498,6 +1141,7 @@
                 pred = [prediction_start + 1, prediction_start + prediction_length],
                 embedDimRhoMin = embed_dim_rho_min,
                 crossMapRhoMin = cross_map_rho_min,
+                exclusionRadius = exclusion_radius,
                 cores = cores )
 
     m.Run()
@@ -530,11 +1174,18 @@
     plt.close(fig)
 
 
-def plot_3d_mde_projection(target, second_target, mde_results, df,
+def plot_3d_mde_projection(target, second_target, mde_results, df, raw_columns,
                             prediction_start, prediction_length, save_path):
     from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection
 
-    obs = mde_results[target]['SELECTED_COLUMNS'][:2]
+    # SELECTED_COLUMNS can now be delay-suffixed names (see
+    # expand_delay_columns()'s docstring) that don't exist in the raw
+    # `df` this function indexes into -- mapped back to their underlying
+    # raw observable (the delay itself isn't meaningful for this plot,
+    # a visual trajectory of the real, undelayed driver values).
+    obs = list(dict.fromkeys(
+        observable_and_delay(c, raw_columns)[0] for c in mde_results[target]['SELECTED_COLUMNS']
+    ))[:2]
     if len(obs) < 2:
         print(f'{target}: fewer than 2 MDE-selected columns, skipping 3D projection.', flush=True)
         return
@@ -604,144 +1255,170 @@
         return fallback
 
 
-def build_nested_mde_network(targets, df, observable_columns, prediction_start, prediction_length,
-                              nested_n, nested_m, embed_dim_rho_min, cross_map_rho_min, cores,
+def build_nested_mde_network(targets, expanded_df, raw_columns, expanded_columns,
+                              prediction_start, prediction_length, nested_n, nested_m,
+                              embed_dim_rho_min, cross_map_rho_min, exclusion_radius, cores,
                               results_dir, mde_results):
-    '''Nested-MDE interaction-matrix construction -- MODE='MDE-GMN' only.
-    MODE='GMN' still uses the original CCM-across-the-full-observable-pool
-    path in build_network() below, entirely untouched.
-
-    This has no notebook precedent: it replaces the original "CCM over the
-    union of each target's MDE-selected columns" step with a from-scratch
-    algorithm built purely from repeated dimx.MDE calls (no CCM / no
-    apps.InteractionMatrix anywhere in this function):
-
-      Step 1 (depth 0): for each target in `targets`, dimx.MDE selects up
-        to `nested_n` (== MDE_D) driver columns from `observable_columns`.
-        Reuses the caller's already-computed top-level mde_results entry
-        when available -- main() always runs this once per cfg['TARGETS']
-        up front (for the pre-existing MDE-only diagnostics), so there's
-        no reason to pay for it twice.
-      Step 2 (depth 1..nested_m): each newly discovered driver becomes a
-        new MDE target in turn, again selecting up to `nested_n` drivers
-        from the same `observable_columns` pool (excluding itself) --
-        repeated for `nested_m` further rounds beyond step 1, so the
-        total graph depth from any target is 1 + nested_m levels.
-
-    Nesting along a given branch stops before reaching nested_m whenever
-    one of two things happens, both handled as natural BFS termination
-    rather than requiring any special-casing beyond the depth counter:
-      - A node's own dimx.MDE call selects zero drivers (the causality
-        gates -- embedDimRhoMin/crossMapRhoMin/the CCM slope gate --
-        rejected every candidate) -- its driver loop below is simply
-        empty, so nothing gets enqueued for the next wave.
-      - A selected driver's marginal MDE rho gain comes out NaN (a
-        degenerate correlation computation, not the same as dimx.MDE's
-        own gates rejecting it -- see the NaN check in the driver loop
-        below) -- that specific edge is rejected as a dead end and not
-        nested into further, since it could never pass
-        strongest_iMatrix's keep-filter downstream regardless (NaN
-        comparisons are always False), making any deeper exploration
-        through it wasted computation. If every driver a node selected
-        comes out NaN, that node is a terminal/"edge" node with no
-        further nesting from it at all -- the same empty-enqueue outcome
-        as the zero-drivers case above, just discovered per-edge instead
-        of by the selection being empty to begin with.
-
-    Cycle/self-loop guarding happens eagerly, during discovery, not as a
-    post-hoc filter: every proposed edge (driver -> node) is added to a
-    networkx.DiGraph built incrementally across this whole call (shared
-    by every target in `targets`, so combined-mode targets can share
-    drivers), then nx.is_directed_acyclic_graph() is checked immediately
-    -- if the edge would create a cycle it's removed and rejected (the
-    driver is simply not linked to that particular node; it may still
-    exist elsewhere in the graph via a different, non-cyclic edge). This
-    mirrors apps.CreateNetwork's own incremental accept/reject loop (see
-    that module's docstring), but has to happen here too: unlike the
-    original CCM path, there is no precomputed full pairwise interaction
-    matrix for CreateNetwork to rank and select from -- dimx.MDE only
-    ever sees one candidate target at a time and has no notion of the
-    network being built around it, so a cycle-producing edge would
-    otherwise end up baked into the interaction matrix itself rather than
-    just mis-ranked (CreateNetwork's own check only protects against
-    cycles it's shown; it can't fix a matrix that never should have had
-    that entry). Self-loop exclusion (driver == node) is redundant with
-    dimx.MDE's own removeColumns=[target] gate in the ordinary case, kept
-    anyway as an explicit guard for defense in depth.
-
-    Each unique node is only ever MDE'd once (memoized via `visited`) --
-    once discovered, it can still gain additional incoming edges from
-    other nodes later in the BFS (ordinary DAG driver-sharing across
-    branches), it just won't be re-selected as an MDE *target* again.
+    '''Nested-MDE network construction -- MODE='MDE-GMN' only. MODE='GMN'
+    still uses the original CCM-across-the-full-observable-pool path in
+    build_network(), untouched.
+
+    No notebook precedent: replaces "CCM over the union of each target's
+    MDE-selected columns" with an algorithm built purely from repeated
+    dimx.MDE calls (no CCM, no apps.InteractionMatrix here). Per explicit
+    direction, this searches a *generalized* (non-uniform, multi-delay)
+    embedding rather than a single contemporaneous one: `expanded_df`
+    (built by expand_delay_columns() in main(), from DELAY_SET) offers
+    each raw observable at every delay in DELAY_SET as its own separate
+    candidate column, so dimx.MDE's forward selection can pick specific
+    (observable, delay) pairs -- e.g. CA1_55 at delay -5 and CA1_78 at
+    delay -1 as two independent drivers of the same node, rather than
+    every driver sharing one embedding depth/lag network-wide (Deyle &
+    Sugihara 2011's generalized embedding theorem). `raw_columns` is the
+    unexpanded observable list, needed to map a selected (possibly
+    delay-suffixed) column name back to its underlying observable via
+    observable_and_delay(), and to exclude a node's own raw+delayed
+    variants from its own candidate pool.
+
+      Step 1 (depth 0): for each target, dimx.MDE selects up to
+        `nested_n` (== MDE_D) distinct *real* observables from
+        `expanded_columns` (excluding the target's own raw+delayed
+        variants) -- the actual D handed to dimx.MDE is
+        effective_mde_dimension(nested_n, candidate_pool, num_delays), not
+        nested_n itself, since each real observable spans num_delays
+        candidate columns; see that function's docstring. Reuses the
+        caller's precomputed mde_results entry when available -- main()
+        already ran this once per target for the MDE-only diagnostics
+        (against the same expanded pool), so it isn't paid for twice.
+      Step 2 (depth 1..nested_m): each driver selected *at delay 0* (see
+        below for why only these) becomes a new MDE target in turn,
+        again selecting up to `nested_n` real observables from the same
+        expanded pool (excluding itself), for `nested_m` further rounds
+        -- total graph depth from any target is 1 + nested_m levels.
+        nested_m == 0 is a sentinel for "uncapped": the depth check is
+        skipped entirely and nesting continues until it runs out on its
+        own (every reachable real observable already visited, or one of
+        the ordinary BFS termination conditions below), rather than
+        stopping after a fixed number of rounds.
+
+    Per explicit direction, only delay=0 selections continue the nested
+    search -- a driver selected at a nonzero delay is added to the graph
+    as a terminal edge, never itself MDE'd as a new target. A delayed
+    coordinate stands in for an unobserved dimension of the true state
+    space (a "placeholder" for something the recording didn't directly
+    measure, per generalized embedding theory), not a physical variable
+    whose own drivers are worth discovering -- so nesting into it would
+    search for the causal history of a lag, not of a real signal. Nodes
+    reached only via delay!=0 edges still simulate fine in gmn: they
+    always get an unconditional self-embedding (see build_network()'s
+    self_delay_set docstring) even with zero further discovered drivers,
+    the same fallback every driverless leaf node already relies on.
+
+    A branch's nesting stops before nested_m (or, when nested_m == 0,
+    stops at all), as ordinary BFS termination, when:
+      - A node's own dimx.MDE call selects zero drivers (embedDimRhoMin/
+        crossMapRhoMin/the CCM slope gate rejected every candidate) --
+        nothing gets enqueued for the next wave.
+      - A selected driver's marginal MDE rho gain is NaN (a degenerate
+        correlation, not the same as dimx.MDE's own gates rejecting it --
+        it already passed those, or it wouldn't be in `cols`). Since this
+        function's graph IS the final network (build_network() hands it
+        to gmn.GMN directly -- no downstream re-filtering by weight), an
+        edge with no trustworthy strength must be rejected here or it
+        becomes a real edge with a nonsensical weight; nesting deeper
+        through it would also be wasted work. `driver` can still enter
+        the graph later via a different, valid edge. If every driver a
+        node selected is NaN, that node is terminal -- the same
+        empty-enqueue outcome as above.
+      - Every driver a node selected was picked at a nonzero delay (the
+        new rule above) -- also the same empty-enqueue outcome.
+
+    Cycle/self-loop guarding happens eagerly during discovery: each
+    proposed edge is added to a networkx.MultiDiGraph shared across this
+    whole call, then checked with nx.is_directed_acyclic_graph()
+    immediately -- a cycle-causing edge is removed by its specific key
+    (MultiDiGraph.remove_edge() removes an arbitrary parallel edge if no
+    key is given, which would risk removing a different, already-valid
+    delay edge between the same driver/node pair). This is the only such
+    guard for MODE='MDE-GMN' (no apps.CreateNetwork call follows), and
+    dimx.MDE itself has no notion of the network being built around it,
+    so a cycle has to be caught here or never. Self-loop exclusion is
+    redundant with dimx.MDE's own removeColumns=[target]-equivalent
+    exclusion (observable_and_delay()-filtered candidate pool) in the
+    ordinary case, kept as an explicit guard anyway. A MultiDiGraph is
+    used (rather than DiGraph) specifically so the same observable can
+    drive one node at multiple delays simultaneously, each its own edge
+    -- per explicit direction, this is expected and kept, not deduped.
+
+    Each unique node is MDE'd at most once (memoized via `visited`); it
+    can still gain further incoming edges later (ordinary DAG
+    driver-sharing), just isn't re-selected as an MDE target.
 
     Parallelization: every node at the same BFS depth is MDE-independent
-    of every other node at that depth -- each dimx.MDE call only reads
-    the static `df` + `observable_columns` pool, with no shared mutable
-    state between nodes. So rather than one dimx.MDE call at a time, this
-    processes one whole "wave" (all not-yet-visited nodes newly
-    discovered at a given depth) with their MDE calls submitted
-    concurrently via ProcessPoolExecutor, one process per node in the
-    wave. Only the edge-acceptance / cycle-check step stays sequential,
-    run once a wave's MDE results are all back -- that step is cheap
-    graph bookkeeping (not EDM computation) and genuinely needs a single
-    consistent DiGraph to check against, so there's nothing to gain by
-    parallelizing it.
-
-    Outer (per-node-in-wave) and inner (each node's own dx.MDE ->
-    CrossMapColumns) parallelism are deliberately NOT both left at
-    `cores` -- that would reproduce the exact outer x inner pool-squaring
-    blowup this project already hit once with apps.InteractionMatrix /
-    pyEDM.CCM (see this module's KNOWN CAVEAT and
-    INTERACTION_MATRIX_OUTER_CORES). Instead the two are split adaptively
-    per wave: outer_workers = min(len(wave), cores), inner_cores =
-    max(1, cores // outer_workers) -- a wide wave (many nodes at once)
-    leans on outer parallelism with each node's own MDE call running
-    single-threaded internally; a narrow wave (e.g. one node) gives that
-    one call the full remaining core budget for its own CrossMapColumns
-    pool instead of leaving cores idle. Either way outer_workers *
-    inner_cores never exceeds `cores`. dimx.MDE's own internal CCM
-    validation step is separately forced sequential by
-    `_sequential_ccm_workers` above, same as everywhere else in this
-    module -- but that patch does NOT reliably survive into a
-    ProcessPoolExecutor worker on its own (confirmed empirically: this
-    cluster's 'forkserver' start method forks workers from a daemon that
-    snapshots imports at its own boot time, which can predate this
-    module's own monkeypatch line), so the wave pool below passes
-    `initializer=_mde_wave_worker_init` to reapply it explicitly inside
-    every worker -- see that function's docstring for the failure mode
-    this avoids.
-
-    Edge weight stored in the returned interaction matrix is the
-    *marginal* MDE rho gain at the step that driver was selected
-    (gain_d = MDErho[d] - MDErho[d-1], MDErho[0] := 0) -- not the raw
-    MDEOut['rho'], which is the *cumulative* multivariate cross-map rho
-    of the whole d-dimensional embedding built so far (dimx/Run.py calls
-    CrossMapColumns with columns=[candidate] + already-selected columns,
-    i.e. the joint embedding), and is expected to trend upward with d
-    regardless of any individual driver's own importance -- using it
-    directly would make later-ranked (weaker, marginal) drivers look
-    equal-or-stronger than the first-ranked (most informative) one. The
-    marginal gain isolates what that one driver specifically contributed
-    when greedily added on top of the previous d-1, which is what "edge
-    strength" should mean for an interaction matrix.
-
-    Returns (graph, iMatrix, nested_mde_results):
-      graph   -- networkx.DiGraph, guaranteed acyclic with no self-loops
-      iMatrix -- NxN DataFrame (index==columns==graph.nodes), the format
-                 apps.CreateNetwork's interactionMatrix= expects: row =
-                 driven node, column = driver, value = edge weight (0
-                 where no edge exists)
+    of every other (each dimx.MDE call only reads the static
+    expanded_df/expanded_columns pool). So each "wave" (not-yet-visited
+    nodes newly discovered at one depth) is submitted concurrently via
+    ProcessPoolExecutor, one process per node. Edge-acceptance/cycle-
+    checking stays sequential once a wave's results are back -- cheap
+    graph bookkeeping that genuinely needs one consistent MultiDiGraph,
+    nothing to gain from parallelizing.
+
+    Outer (per-node) and inner (each node's own CrossMapColumns)
+    parallelism are split adaptively per wave rather than both left at
+    `cores` -- that would reproduce the outer x inner pool-squaring
+    blowup this project hit once already (see the module docstring's
+    FIXED section). outer_workers = min(len(wave), cores), inner_cores =
+    max(1, cores // outer_workers): a wide wave leans on outer
+    parallelism with each MDE call single-threaded; a narrow wave gives
+    that one call the full remaining core budget. outer_workers *
+    inner_cores never exceeds `cores`. dimx.MDE's own CCM validation is
+    forced sequential by `_sequential_ccm_workers`, but that patch
+    doesn't reliably survive into a ProcessPoolExecutor worker on its own
+    (this cluster's 'forkserver' start method can predate the patch), so
+    the wave pool passes `initializer=_mde_wave_worker_init` to reapply
+    it inside every worker -- see that function's docstring for the
+    crash this avoids.
+
+    Edge weight is the *marginal* MDE rho gain at the step that driver
+    was selected (gain_d = MDErho[d] - MDErho[d-1], MDErho[0] := 0) --
+    not the raw cumulative MDEOut['rho'] (dimx/Run.py cross-maps
+    columns=[candidate]+already-selected, so cumulative rho trends
+    upward with d regardless of any one driver's importance, which would
+    make later, weaker drivers look equal-or-stronger than the first).
+    The marginal gain isolates what that driver specifically added on
+    top of the previous d-1 -- what "edge strength" should mean here.
+
+    Returns (graph, edges_df, nested_mde_results):
+      graph      -- networkx.MultiDiGraph, acyclic, no self-loops. Every
+                 edge has 'delay' (<= 0) and 'weight' attributes. The
+                 actual network build_network() hands to gmn.GMN (see
+                 that function's docstring) -- apps.CreateNetwork is not
+                 involved.
+      edges_df   -- DataFrame with columns ['driver', 'node', 'delay',
+                 'weight'], one row per edge -- an NxN interaction matrix
+                 can't represent a node having multiple edges from the
+                 same driver at different delays, so this replaces that
+                 format. Exported to CSV by build_network() as a
+                 human-readable record alongside the graph; not fed to
+                 apps.CreateNetwork.
       nested_mde_results -- {node: run_mde_for_target() result} for every
-                 node this call ran dimx.MDE on (targets included). Kept
-                 separate from the caller's own mde_results dict rather
-                 than mutating it in place, so repeated separate-mode
-                 calls (one per target group) can't cross-contaminate
-                 results for a shared intermediate channel name.
+                 node this call ran dimx.MDE on. Kept separate from the
+                 caller's own mde_results so repeated separate-mode calls
+                 can't cross-contaminate a shared intermediate channel
+                 name.
     '''
-    graph = nx.DiGraph()
+    graph = nx.MultiDiGraph()
     graph.add_nodes_from(targets)
 
-    edge_weight = {}          # (driver, node) -> weight
+    # expanded_columns is raw_columns with num_delays-1 extra columns
+    # appended per real observable (see expand_delay_columns()) -- exact,
+    # not approximate, so this recovers num_delays without needing
+    # delay_set threaded through as its own parameter. Used by
+    # effective_mde_dimension() below to convert nested_n (a real-
+    # observable count) into the actual D handed to dimx.MDE.
+    num_delays = len(expanded_columns) // len(raw_columns)
+
+    edges = []                # list of {driver, node, delay, weight} dicts
     nested_mde_results = {}   # node -> run_mde_for_target() result, this call only
     visited = set()
     current_wave = [(t, 0) for t in targets]
@@ -783,11 +1460,14 @@
                                       initializer=_mde_wave_worker_init) as executor:
                 futures = {}
                 for node, depth in to_compute:
-                    candidate_pool = [c for c in observable_columns if c != node]
+                    candidate_pool = [c for c in expanded_columns
+                                       if observable_and_delay(c, raw_columns)[0] != node]
+                    effective_d = effective_mde_dimension(nested_n, candidate_pool, num_delays)
                     fut = executor.submit(
-                        run_mde_for_target, node, df, candidate_pool,
-                        prediction_start, prediction_length, nested_n,
-                        embed_dim_rho_min, cross_map_rho_min, inner_cores, results_dir,
+                        run_mde_for_target, node, expanded_df, candidate_pool,
+                        prediction_start, prediction_length, effective_d,
+                        embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
+                        inner_cores, results_dir,
                     )
                     futures[fut] = node
                 for fut in as_completed(futures):
@@ -802,109 +1482,120 @@
             cols = result['SELECTED_COLUMNS']
             gains = np.diff(np.concatenate(([0.0], result['m'].MDErho)))
 
-            for driver, gain in zip(cols, gains):
+            for col, gain in zip(cols, gains):
+                driver, delay = observable_and_delay(col, raw_columns)
+
                 if driver == node:
                     continue  # explicit self-loop guard (belt-and-suspenders)
 
                 if np.isnan(gain):
-                    # Degenerate marginal-gain computation (e.g. a
-                    # near-constant embedding dimension producing an
-                    # undefined correlation) -- not the same as dimx.MDE
-                    # itself rejecting the candidate (it already passed
-                    # embedDimRhoMin/crossMapRhoMin/the CCM slope gate,
-                    # or it wouldn't be in `cols` at all), but a NaN edge
-                    # weight can never pass strongest_iMatrix's `> threshold`
-                    # keep-filter downstream regardless of what we do here
-                    # (NaN comparisons are always False), so CreateNetwork
-                    # would never select it as a kept driver either way.
-                    # Treat this edge as a dead end: don't add it to the
-                    # graph via this edge, and don't nest further from it
-                    # -- exploring deeper from a driver we can't assign a
-                    # trustworthy strength to is pure wasted computation.
-                    # `driver` can still enter the graph normally later via
-                    # a different, valid edge from another node (ordinary
-                    # DAG driver-sharing) -- only this specific edge is
-                    # rejected.
-                    print(f'nested MDE: rejected {driver} -> {node} '
+                    # Degenerate marginal-gain computation -- dead-end
+                    # this specific edge without nesting further through
+                    # it (see this function's docstring for why NaN
+                    # weights can't just be added anyway).
+                    print(f'nested MDE: rejected {driver} (delay {delay}) -> {node} '
                           f'(marginal MDE rho gain is NaN -- treating as a dead-end edge, '
                           f'no further nesting from this driver via this path)', flush=True)
                     continue
 
                 graph.add_node(driver)
-                graph.add_edge(driver, node)
+                key = graph.add_edge(driver, node, delay=delay, weight=gain)
 
                 if not nx.is_directed_acyclic_graph(graph):
-                    graph.remove_edge(driver, node)
+                    graph.remove_edge(driver, node, key=key)
                     if graph.degree(driver) == 0:
                         graph.remove_node(driver)
-                    print(f'nested MDE: rejected {driver} -> {node} (would create a cycle)', flush=True)
+                    print(f'nested MDE: rejected {driver} (delay {delay}) -> {node} '
+                          f'(would create a cycle)', flush=True)
                     continue
 
-                edge_weight[(driver, node)] = max(gain, edge_weight.get((driver, node), 0.0))
+                edges.append({'driver': driver, 'node': node, 'delay': delay, 'weight': gain})
 
-                if depth < nested_m and driver not in visited:
+                # Only a delay=0 (contemporaneous) selection continues the
+                # nested search -- see this function's docstring for why a
+                # delayed coordinate is a terminal placeholder, not a new
+                # MDE target. nested_m == 0 is the "uncapped" sentinel --
+                # see this function's docstring -- so the depth check is
+                # skipped entirely in that case and nesting only stops via
+                # the other ordinary BFS termination conditions (empty
+                # wave once every reachable real observable is visited,
+                # zero-driver/NaN/nonzero-delay/cycle rejections).
+                if delay == 0 and (nested_m == 0 or depth < nested_m) and driver not in visited:
                     next_wave.append((driver, depth + 1))
 
         current_wave = next_wave
 
-    all_nodes = list(graph.nodes)
-    iMatrix = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)
-    for (driver, node), weight in edge_weight.items():
-        iMatrix.loc[node, driver] = weight
+    edges_df = pd.DataFrame(edges, columns=['driver', 'node', 'delay', 'weight'])
 
-    print(f'nested MDE: {len(all_nodes)} nodes, {len(edge_weight)} edges '
-          f'(N={nested_n}, M={nested_m})', flush=True)
+    nested_m_display = 'unlimited' if nested_m == 0 else nested_m
+    print(f'nested MDE: {len(graph.nodes)} nodes, {len(edges_df)} edges '
+          f'(N={nested_n}, M={nested_m_display})', flush=True)
 
-    return graph, iMatrix, nested_mde_results
+    return graph, edges_df, nested_mde_results
 
 
 def build_network(targets, mde_results, df, prediction_start, prediction_length,
                    mode, e_range, tau_range, results_dir, cores, interaction_matrix_outer_cores,
                    observable_columns=None, nested_n=None, nested_m=None,
-                   embed_dim_rho_min=None, cross_map_rho_min=None,
+                   embed_dim_rho_min=None, cross_map_rho_min=None, exclusion_radius=None,
+                   expanded_df=None, expanded_observable_columns=None, delay_set=None,
                    threshold=0, fallback_node_dimension=4, max_node_dimension=15):
-    '''Build one interaction matrix + strongest-driver network spanning
-    `targets` and their driver candidates. The interaction matrix itself
-    comes from one of two entirely different constructions depending on
-    `mode`:
-      - mode == 'MDE-GMN': build_nested_mde_network() -- nested dimx.MDE
-        expansion, no CCM involved. observable_columns/nested_n/nested_m/
-        embed_dim_rho_min/cross_map_rho_min are required in this mode.
+    '''Build one network spanning `targets` and their driver candidates,
+    via one of two unrelated constructions depending on `mode`:
+      - mode == 'MDE-GMN': build_nested_mde_network()'s own graph IS the
+        final network, handed to gmn.GMN directly -- no
+        apps.CreateNetwork() call. Per explicit direction: nested-MDE
+        already does the network-discovery job CreateNetwork exists for
+        (BFS out from targets, rejecting cycles as edges are proposed),
+        using real MDE-driven edge selection, so re-deriving a network
+        from an interaction matrix afterward would just be redundant
+        work. Safe because gmn.Network only ever reads the pickled
+        network file's 'Graph' entry (gmn/Network.py:43-46) via
+        Graph.predecessors(node) (gmn/Node.py:104) -- the 'Map' entry
+        CreateNetwork also writes is explicitly unused (that file's own
+        comment) -- so a {'Graph': graph, 'Map': ...} pickle built
+        directly from build_nested_mde_network()'s MultiDiGraph is
+        exactly what CreateNetwork would have handed back anyway.
+        observable_columns/nested_n/nested_m/embed_dim_rho_min/
+        cross_map_rho_min/expanded_df/expanded_observable_columns/
+        delay_set are required in this mode -- expanded_df/
+        expanded_observable_columns are built once in main() via
+        expand_delay_columns() (see build_nested_mde_network()'s
+        docstring for why the search needs a delay-expanded pool rather
+        than the raw `df`/`observable_columns`, which are still used
+        as-is below for the real data every node actually simulates
+        from). delay_set is stored on the returned graph
+        (graph.graph['self_delay_set']) so gmn's own generalized-
+        embedding patches (see the module docstring) know which delays
+        to always give each node's own self-history, regardless of
+        what nested-MDE discovered for its external drivers.
+        node_dimensions is NOT computed here (see gmn_fidelity_rho()'s
+        default_E fallback for its one remaining consumer).
       - mode == 'GMN': the original CCM interaction matrix (data-driven
-        E/tau grid search) across the full REGION_CHANNEL_CAP-limited
-        observable pool, via apps.InteractionMatrix -- unchanged from the
-        pre-nested-MDE version of this pipeline.
-    Everything after the interaction matrix is computed (node-dimension
-    estimate, the CreateNetwork driver-ranking-bug workaround, the actual
-    CreateNetwork call) is identical either way.
+        E/tau grid search over the REGION_CHANNEL_CAP-limited observable
+        pool, via apps.InteractionMatrix), then a node-dimension estimate
+        and apps.CreateNetwork with its driver-ranking-bug workaround --
+        unchanged, since there's no MDE-discovered graph to substitute
+        and CreateNetwork's own construction is still needed.
 
     Mirrors notebooks/simple-mde-gmn.ipynb cell 7's build_network() for
-    the mode == 'GMN' branch. See that cell's docstring for the
-    CreateNetwork driver-ranking-bug workaround rationale (identical
-    here).
-
-    interaction_matrix_outer_cores is deliberately a *separate* knob from
-    `cores`, not just reused: apps.InteractionMatrix's own -cr sizes an
-    outer ProcessPoolExecutor across column *pairs*. Its internal
-    pyEDM.CCM() call (one per pair) used to create its *own* fresh
-    multiprocessing pool every time, sized to the full visible
-    cpu_count() by default -- confirmed by reading pyEDM/CCM.py
-    (`_resolve_workers`) and apps/InteractionMatrix.py's InteractFunc
-    (which calls CCM() without ever passing `parallel=`, so it couldn't
-    be overridden via the CLI). Neither tool coordinated with the other,
-    so peak concurrent processes was -cr * cpu_count(), not -cr --
-    setting -cr = cores (the seemingly natural choice) squared your core
-    request into a process count (confirmed empirically: -cr=cores on a
-    10-core machine produced ~90 live processes, not ~10).
-
-    This module now monkeypatches pyEDM.CCM's inner pool away entirely
-    (see `_sequential_ccm_workers` near the top of this file) rather than
-    just avoiding the multiplication by keeping this knob low -- every
-    CCM() call here only ever resolves to 4-8 tiny tasks, so the inner
-    pool was pure spawn/reimport overhead with no parallelism benefit to
-    trade away. interaction_matrix_outer_cores stays a separate knob
-    regardless, since it's still the one thing sizing real parallelism
-    across column pairs.
+    the mode == 'GMN' branch.
+
+    interaction_matrix_outer_cores is a *separate* knob from `cores`:
+    apps.InteractionMatrix's -cr sizes an outer pool across column pairs,
+    and its internal pyEDM.CCM() call used to spawn its *own* fresh pool
+    per pair sized to cpu_count() (confirmed by reading pyEDM/CCM.py and
+    apps/InteractionMatrix.py, neither of which passes `parallel=`).
+    Peak concurrent processes was -cr * cpu_count(), not -cr -- setting
+    -cr = cores (the natural-seeming choice) squared your core request
+    into a process count (-cr=cores on a 10-core machine produced ~90
+    live processes). This module now monkeypatches pyEDM.CCM's inner
+    pool away entirely (`_sequential_ccm_workers` near the top of this
+    file) rather than just avoiding the multiplication -- every CCM()
+    call here only ever resolves to 4-8 tiny tasks, so the inner pool was
+    pure overhead. interaction_matrix_outer_cores stays a separate knob
+    regardless, since it's still what sizes real parallelism across
+    column pairs.
     '''
     network_id = '_'.join(targets)
     # Distinguishes MODE='GMN' network artifacts from MODE='MDE-GMN'
@@ -917,16 +1608,26 @@
         network_id = f'gmn_{network_id}'
 
     ccm_E = ccm_tau = None
+    node_dimensions = {}
 
     if mode == 'MDE-GMN':
-        graph, iMatrix, _ = build_nested_mde_network(
-            targets, df, observable_columns, prediction_start, prediction_length,
-            nested_n, nested_m, embed_dim_rho_min, cross_map_rho_min, cores,
+        graph, edges_df, _ = build_nested_mde_network(
+            targets, expanded_df, observable_columns, expanded_observable_columns,
+            prediction_start, prediction_length,
+            nested_n, nested_m, embed_dim_rho_min, cross_map_rho_min, exclusion_radius, cores,
             results_dir, mde_results,
         )
+        # 0 is forced into self_delay_set regardless of DELAY_SET: gmn's
+        # own SMap target-lookup convention (Parameters.target + '(t-0)',
+        # mirrored by this pipeline's _generalized_node_forecast/Generate)
+        # requires a node's own present-value coordinate to exist in its
+        # embedding -- this isn't a driver delay choice, it's the anchor
+        # point being predicted, so it can't be left out even if a user
+        # configures a DELAY_SET without 0 in it.
+        graph.graph['self_delay_set'] = sorted(set(delay_set) | {0}, reverse=True)
         selected_union = [n for n in graph.nodes if n not in targets]
-        interaction_prefix = results_dir / f'iMatrix_nestedMDE_N{nested_n}_M{nested_m}_{network_id}'
-        iMatrix.to_csv(f'{interaction_prefix}_MDE.csv')
+        edges_file = results_dir / f'edges_nestedMDE_N{nested_n}_M{nested_m}_{network_id}.csv'
+        edges_df.to_csv(edges_file, index=False)
     else:  # mode == 'GMN'
         selected_union = []
         for t in targets:
@@ -947,7 +1648,18 @@
     data_after_mde_lib_file = results_dir / f'data_after_mde_lib_{network_id}.csv'
     data_after_mde_lib.to_csv(data_after_mde_lib_file, index=False)
 
-    if mode == 'GMN':
+    if mode == 'MDE-GMN':
+        # build_nested_mde_network()'s graph already IS the network --
+        # see this function's docstring for why apps.CreateNetwork() adds
+        # nothing here. 'Map' is written only because CreateNetwork's own
+        # pickle format includes it and downstream code below prints from
+        # it; gmn itself never reads it (see docstring).
+        network_dict = {node: list(graph.predecessors(node)) for node in graph.nodes}
+        network = {'Graph': graph, 'Map': network_dict}
+        network_file = results_dir / f'Network_nestedMDE_N{nested_n}_M{nested_m}_{network_id}.pkl'
+        with open(network_file, 'wb') as f:
+            pickle.dump(network, f)
+    else:  # mode == 'GMN'
         e_vals, tau_vals = list(e_range), list(tau_range)
         target_grids = [target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals) for t in targets]
         mean_grid = np.nanmean(np.stack(target_grids), axis=0)
@@ -977,40 +1689,47 @@
         interaction_csv = Path(f'{interaction_prefix}_CCM.csv')
         iMatrix = pd.read_csv(interaction_csv, index_col=0)
 
-    node_dimensions = {
-        node: estimate_node_dimension(data_after_mde_lib, node, cores,
-                                       max_e=max_node_dimension, fallback=fallback_node_dimension)
-        for node in iMatrix.index
-    }
-    print(f'{network_id}: node N_D =', node_dimensions, flush=True)
+        node_dimensions = {
+            node: estimate_node_dimension(data_after_mde_lib, node, cores,
+                                           max_e=max_node_dimension, fallback=fallback_node_dimension)
+            for node in iMatrix.index
+        }
+        print(f'{network_id}: node N_D =', node_dimensions, flush=True)
 
-    # apps.CreateNetwork() keeps each node's *weakest* above-threshold
-    # drivers instead of the strongest (confirmed against the installed
-    # apps/CreateNetwork.py:110-116 -- it sorts each row ascending, then
-    # truncates to numDrivers). Worked around by pre-selecting the
-    # strongest N_D drivers ourselves, then calling CreateNetwork with
-    # numDrivers set to the full column count so its buggy truncation
-    # never has anything left to remove.
-    strongest_iMatrix = iMatrix.copy()
-    for node in strongest_iMatrix.index:
-        row = strongest_iMatrix.loc[node].drop(node)
-        n_drivers = max(1, min(node_dimensions[node], len(row)))
-        keep = row[row > threshold].sort_values(ascending=False).index[:n_drivers]
-        drop = row.index.difference(keep)
-        strongest_iMatrix.loc[node, drop] = threshold
+        # apps.CreateNetwork() keeps each node's *weakest* above-threshold
+        # drivers instead of the strongest (confirmed against the installed
+        # apps/CreateNetwork.py:110-116 -- it sorts each row ascending, then
+        # truncates to numDrivers). Worked around by pre-selecting the
+        # strongest N_D drivers ourselves, then calling CreateNetwork with
+        # numDrivers set to the full column count so its buggy truncation
+        # never has anything left to remove.
+        strongest_iMatrix = iMatrix.copy()
+        for node in strongest_iMatrix.index:
+            row = strongest_iMatrix.loc[node].drop(node)
+            n_drivers = max(1, min(node_dimensions[node], len(row)))
+            keep = row[row > threshold].sort_values(ascending=False).index[:n_drivers]
+            drop = row.index.difference(keep)
+            strongest_iMatrix.loc[node, drop] = threshold
 
-    if mode == 'MDE-GMN':
-        network_file = results_dir / f'Network_nestedMDE_N{nested_n}_M{nested_m}_{network_id}.pkl'
-    else:
         network_file = results_dir / f'Network_E{ccm_E}_tau{ccm_tau}_CCM_{network_id}.pkl'
-    network = CreateNetwork(
-        interactionMatrix = strongest_iMatrix,
-        targetCols        = list(targets),
-        threshold         = threshold,
-        numDrivers        = len(strongest_iMatrix.columns),
-        outputFile        = str(network_file),
-        plotNetwork       = False,
-    )
+        network = CreateNetwork(
+            interactionMatrix = strongest_iMatrix,
+            targetCols        = list(targets),
+            threshold         = threshold,
+            numDrivers        = len(strongest_iMatrix.columns),
+            outputFile        = str(network_file),
+            plotNetwork       = False,
+        )
+        # gmn.Network.__init__ reloads the graph fresh from network_file's
+        # own pickle (gmn/Network.py:43-46), not from this in-memory
+        # `network` dict -- CreateNetwork already wrote that file above,
+        # so fallback_E/fallback_tau have to be added to the graph and
+        # the file re-pickled here, or _node_embedding_spec()'s plain-
+        # DiGraph fallback (see its docstring) would never see them.
+        network['Graph'].graph['fallback_E'] = ccm_E
+        network['Graph'].graph['fallback_tau'] = ccm_tau
+        with open(network_file, 'wb') as f:
+            pickle.dump(network, f)
 
     for t in targets:
         print(f'{t} parents:', network['Map'][t], flush=True)
@@ -1028,25 +1747,69 @@
 
 
 def draw_colored_network(net_result, target_colors, save_path):
+    '''Draws the network graph, nodes colored by which target ultimately
+    depends on them (unchanged). Edges are colored by delay, with a
+    legend mapping color -> delay, whenever the graph carries per-edge
+    delay data (a MultiDiGraph built by build_nested_mde_network() for
+    MODE='MDE-GMN' -- see that function's docstring on why the same
+    observable can drive one node via several parallel edges, each its
+    own delay). MODE='GMN' networks (a plain DiGraph from
+    apps.CreateNetwork, no delay concept at all) fall back to the
+    original by-target edge coloring instead.
+
+    Edges are drawn individually (nx.draw_networkx_edges per edge, not
+    one nx.draw_networkx call) with a curvature that increases per
+    parallel edge -- otherwise multiple edges between the same two nodes
+    (the whole point of a MultiDiGraph here) would overlap into one
+    indistinguishable line.'''
     graph = net_result['network']['Graph']
     targets = net_result['targets']
+    has_delays = isinstance(graph, nx.MultiDiGraph)
 
     node_colors = [target_colors.get(n, 'lightgray') for n in graph.nodes]
-    edge_colors = [target_colors.get(v, 'gray') for _, v in graph.edges]
 
     pos = nx.arf_layout(graph)
     fig, ax = plt.subplots(figsize=(10, 10))
-    nx.draw_networkx(
-        graph, pos=pos, ax=ax, with_labels=True, arrowsize=10,
-        node_size=300, node_color=node_colors, edge_color=edge_colors,
-        width=1.2, font_size=9, font_weight='bold',
-    )
-    handles = [
+    nx.draw_networkx_nodes(graph, pos=pos, ax=ax, node_size=300, node_color=node_colors)
+    nx.draw_networkx_labels(graph, pos=pos, ax=ax, font_size=9, font_weight='bold')
+
+    if has_delays:
+        all_delays = sorted({data['delay'] for _, _, data in graph.edges(data=True)})
+        cmap = plt.get_cmap('viridis')
+        delay_color = {d: cmap(i / max(1, len(all_delays) - 1)) for i, d in enumerate(all_delays)}
+
+        parallel_count = defaultdict(int)
+        for u, v, data in graph.edges(data=True):
+            i = parallel_count[(u, v)]
+            parallel_count[(u, v)] += 1
+            rad = 0.15 * ((i + 1) // 2) * (1 if i % 2 == 0 else -1)
+            nx.draw_networkx_edges(
+                graph, pos=pos, ax=ax, edgelist=[(u, v)], arrowsize=10, width=1.2,
+                edge_color=[delay_color[data['delay']]],
+                connectionstyle=f'arc3,rad={rad}',
+            )
+
+        edge_handles = [
+            plt.Line2D([0], [0], color=delay_color[d], lw=2, label=f'delay={d}')
+            for d in all_delays
+        ]
+        legend_title = 'Edge delay / target'
+    else:
+        for u, v in graph.edges():
+            nx.draw_networkx_edges(
+                graph, pos=pos, ax=ax, edgelist=[(u, v)], arrowsize=10, width=1.2,
+                edge_color=[target_colors.get(v, 'gray')],
+            )
+        edge_handles = []
+        legend_title = 'Target'
+
+    node_handles = [
         plt.Line2D([0], [0], marker='o', color='w', markersize=12,
                    markerfacecolor=target_colors[t], label=t)
         for t in targets
     ]
-    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 1.0))
+    ax.legend(handles=edge_handles + node_handles, loc='upper left',
+              bbox_to_anchor=(1.0, 1.0), title=legend_title)
     ax.set_title(f"GMN network ({net_result['network_id']})")
     fig.tight_layout()
     fig.savefig(save_path, dpi=300)
@@ -1058,41 +1821,72 @@
 # =============================================================================
 
 def build_gmn_config(net_result, prediction_start, prediction_length, results_dir,
-                      mode='Generate', function='SMap', E=3, tau=-3, theta=3,
-                      node_theta_overrides=None, variant='', generate_length=None):
+                      mode='Generate', function='SMap', theta=3,
+                      node_theta_overrides=None, variant='', generate_length=None,
+                      generate_start=None, max_offset=None, exclusion_radius=0):
     '''Stage the .cfg file gmn.GMN's configFile= constructor expects.
 
-    generate_length (mode='Generate' only): number of steps gmn.GMN
-    free-runs for, independent of prediction_length -- the latter is also
-    the MDE lib/pred split's held-out length (see load_config()'s
-    LIB_RATIO), which drives MDE variable selection, CCM/EmbedDimension
-    calls, and the search's own per-candidate theta retuning, none of
-    which have anything to do with how many steps a single Generate()
-    call should actually simulate. A full Generate() call over the whole
-    held-out window can be the single most expensive step in the search
-    loop (one call per (E, tau[, theta]) candidate), so decoupling this
-    lets you shrink it (MDE_GMN_GENERATE_LENGTH) without also shrinking
-    the held-out window MDE/CCM validate against. Defaults to
-    prediction_length when None (unchanged pre-existing behavior).
-    generate_length always still starts at prediction_start -- only the
-    *length* is decoupled, not the start point, since "start generating
-    right where the library ends" is what makes the comparison against
-    real held-out data meaningful. Ignored for mode='Forecast', which
-    still uses the full prediction_start/prediction_length (a one-step
-    boundary-accuracy diagnostic, not a free-running simulation length).
+    exclusion_radius: written into the [EDM] section below and read
+    straight through by gmn's own (monkeypatched) Node.Forecast/Generate
+    as `Parameters.exclusionRadius`, passed to every pyEDM Simplex/SMap
+    call gmn itself makes (see this pipeline's _generalized_node_generate/
+    Node.Forecast monkeypatches near the top of this file) -- the same
+    Theiler-window neighbor-exclusion mechanism as dimx.MDE's own
+    MDE_GMN_EXCLUSION_RADIUS (see that parameter's docstring for the
+    general rationale), just applied to GMN's own recursive Generate()/
+    Forecast() nearest-neighbor search rather than MDE's forward
+    selection -- these are two independent EDM contexts with their own
+    separate embeddings, so there's no single shared "auto" value that
+    would make sense for both; GMN_EXCLUSION_RADIUS is a plain static
+    knob, not dynamically recomputed the way MDE's is.
+
+    E/tau are no longer parameters here -- every node's real embedding
+    now comes from _node_embedding_spec()/_embedding_spec_for()
+    (per-driver delays off the network graph, plus self_delay_set or
+    fallback_E/fallback_tau -- see those functions' docstrings), not a
+    shared network-wide (E, tau) written into this .cfg. Placeholder
+    'E'/'tau' values are still written into the [EDM] section below
+    purely because gmn.Node.__init__'s own (unpatched) arithmetic reads
+    them before _generalized_node_init immediately corrects
+    self.libEnd_i -- see that function's docstring; their actual values
+    are otherwise inert.
+
+    generate_start (both modes): row index where Generate() starts
+    free-running / where Forecast()'s pred window starts, independent of
+    prediction_start -- the latter is the fixed MDE lib/pred boundary
+    (see load_config()'s LIB_RATIO) that MDE variable selection and
+    CCM/EmbedDimension validate against, and stays fixed regardless of
+    what row a diagnostic Generate()/Forecast() call happens to start at.
+    Defaults to prediction_start when None (unchanged pre-existing
+    behavior). Sweeping this is how sweep_generate_start() finds the
+    best-matching starting point for a fixed, already-chosen theta --
+    see that function's docstring.
+
+    generate_length (both modes): number of steps simulated/predicted
+    from generate_start, independent of prediction_length -- the latter
+    is also the MDE lib/pred split's held-out length, which drives MDE
+    variable selection and CCM/EmbedDimension calls, none of which have
+    anything to do with how many steps one Generate()/Forecast() call
+    should cover. A full Generate() call over the whole held-out window
+    can be the single most expensive step in the search loop (one call
+    per candidate), so decoupling this lets you shrink it
+    (MDE_GMN_GENERATE_LENGTH) without also shrinking the held-out window
+    MDE/CCM validate against. Defaults to prediction_length when None.
+    mode='Forecast' honors this same cap (rather than always spanning the
+    full held-out prediction_length) so its window matches Generate()'s --
+    run_forecast_diagnostic() plots both on the same axes, and a
+    Forecast() line running past where Generate() stopped would have
+    nothing to visually compare against.
 
     function is fixed to 'SMap' for every node in this pipeline -- gmn's
     Simplex branch never calls pyEDM.Embed() before predicting (confirmed
     by reading gmn/Generate.py directly), so it can't do real Takens
     delay embedding: embedded=True forces pyEDM to overwrite E with the
     raw predecessor-column count, silently discarding tau. gmn's SMap
-    branch does call Embed(E, tau, columns) first -- a genuine delay
-    embedding. Per explicit direction, this pipeline uses SMap uniformly
-    for every node rather than mixing generators per node or patching
-    Simplex -- see notebooks/simple-mde-gmn.ipynb's build_gmn_config
-    docstring for the full evidence trail. Consequence: every node
-    (including leaf/undriven ones) needs E >= 2, or SMap's own
-    predictor-column-exclusion convention leaves it with zero predictors.
+    branch does call Embed(E, tau, columns) first (now replaced by this
+    pipeline's own generalized embedding -- see the module docstring).
+    Per explicit direction, this pipeline uses SMap uniformly for every
+    node rather than mixing generators per node or patching Simplex.
 
     node_theta_overrides: optional {nodeName: theta} dict -- written as
     per-node .cfg files (gmn/Node.py's nodeConfigPath + f'{name}.cfg'
@@ -1101,46 +1895,56 @@
     concatenation path lookup (confirmed empirically: without it, every
     override is silently ignored).
 
-    Forecast mode's lib/pred offset: gmn/Forecast.py's SMap branch calls
-    Embed(E, tau, columns) then dropna() before predicting, which drops
-    the leading (E-1)*abs(tau) NaN rows the time-shift introduces --
-    shrinking the dataframe SMap actually sees relative to node.data's
-    full row count, which is what `prediction_start`/`n_rows` below are
-    expressed in. Since every node uses SMap and E >= 2 always (see
-    docstring above), this offset is never zero, and an unadjusted
-    lib/pred computed in original-row terms overflows the shrunk frame
-    (confirmed empirically: pyEDM.SMap's CreateIndices() raises
-    "prediction index N exceeds the number of data rows M" with M
-    matching n_rows - offset exactly). gmn/Generate.py's own SMap branch
-    sidesteps this by building a fresh 0-based time index after its own
-    equivalent offset slice; gmn/Forecast.py instead reuses the node's
-    original (now index-shifted) time column, so the fix here is to shift
-    our own explicit lib/pred bounds back by the same offset instead.'''
+    max_offset (mode='Forecast' only): the network's largest per-node
+    embedding offset (network_max_embedding_offset()), used to size a
+    single, network-wide lib/pred window valid for every node's own
+    embedding. Since nodes can each have a different real offset now
+    (unlike the old shared (E, tau)), a node whose own offset is smaller
+    than max_offset gets evaluated a few rows further into the window
+    than intended -- accepted as reasonable slop for this
+    diagnostic-only step (run_forecast_diagnostic()). Generate()'s own
+    per-step recursive simulation, used everywhere else, has no such
+    issue: libEnd_i is computed correctly per node in
+    _generalized_node_init(), not derived from a shared cfg string. This
+    mirrors gmn/Forecast.py's SMap branch, which calls Embed(E, tau,
+    columns) then dropna() before predicting, dropping (E-1)*abs(tau)
+    leading NaN rows the time-shift introduces and shrinking the
+    dataframe SMap actually sees relative to node.data's full row count
+    -- confirmed empirically that an unadjusted lib/pred computed in
+    original-row terms overflows the shrunk frame (pyEDM.SMap's
+    CreateIndices() raises "prediction index N exceeds the number of
+    data rows M" with M matching forecast_length - offset exactly), so
+    this function's own lib/pred bounds are shifted back by max_offset
+    to compensate.'''
     if mode not in ('Generate', 'Forecast'):
         raise ValueError(f"Unknown mode: {mode!r}")
-
-    n_rows = prediction_start + prediction_length
     Tp = 1
+    generate_start = generate_start if generate_start is not None else prediction_start
 
     if mode == 'Generate':
-        gmn_prediction_start = str(prediction_start)
+        gmn_prediction_start = str(generate_start)
         gmn_prediction_length = str(generate_length if generate_length is not None else prediction_length)
         lib_range = ''
         pred_range = ''
     else:
         gmn_prediction_start = '1'
         gmn_prediction_length = '1'
-        offset = (E - 1) * abs(tau)
-        if offset >= prediction_start or offset >= prediction_length - Tp:
+        offset = max_offset if max_offset is not None else 0
+        # Capped to generate_length (same window Generate() free-runs
+        # for), not the full held-out prediction_length -- see this
+        # function's docstring for why the two modes' windows need to
+        # match for run_forecast_diagnostic()'s combined plot to be
+        # meaningful.
+        forecast_length = generate_length if generate_length is not None else prediction_length
+        if offset >= generate_start or offset >= forecast_length - Tp:
             raise ValueError(
                 f'build_gmn_config(mode=Forecast): embedding offset {offset} '
-                f'(E={E}, tau={tau}) leaves too few rows in prediction_start='
-                f'{prediction_start}/prediction_length={prediction_length} for '
-                f'a valid lib/pred split after SMap\'s Embed()+dropna() shrinks '
-                f'the frame -- shrink E/|tau| or grow the prediction window.'
+                f'leaves too few rows in generate_start={generate_start}/'
+                f'forecast_length={forecast_length} for a valid lib/pred split '
+                f'-- shrink MDE_GMN_DELAY_SET or grow the prediction window.'
             )
-        lib_range = f'1 {prediction_start - offset}'
-        pred_range = f'{prediction_start - offset} {n_rows - Tp - offset}'
+        lib_range = f'1 {generate_start - offset}'
+        pred_range = f'{generate_start - offset} {generate_start + forecast_length - Tp - offset}'
 
     node_cfg_dir = results_dir / f"node_cfgs_{net_result['network_id']}_{variant or mode.lower()}"
 
@@ -1172,12 +1976,12 @@
             'EDM': {
                 'lib': lib_range,
                 'pred': pred_range,
-                'E': str(E),
+                'E': '2',    # placeholder, inert -- see this function's docstring
                 'Tp': str(Tp),
                 'knn': '0',
-                'tau': str(tau),
+                'tau': '-1',  # placeholder, inert -- see this function's docstring
                 'theta': str(node_theta),
-                'exclusionRadius': '0',
+                'exclusionRadius': str(exclusion_radius),
                 'columns': '',
                 'target': '',
                 'solver': '',
@@ -1218,15 +2022,130 @@
 # E/tau/theta search (mirrors notebook cell 10)
 # =============================================================================
 
-def retune_theta_for_candidate(data_lib, graph, nodes, E, tau, theta_range, cores,
-                                min_pred_rows=20):
+def _hash_file(path, chunk_size=1 << 20):
+    '''sha256 of a file's actual bytes, not its name/size/mtime -- used to
+    detect a same-named DATA_FILE whose content silently changed between
+    job submissions, which name/size/mtime alone can miss (mtime survives
+    a `cp`, size collisions are rare but real for CSVs of similar shape).
+    Streamed in chunks since this is not IPC-bound, just avoiding loading
+    an unnecessarily large file fully into memory at once.'''
+    h = hashlib.sha256()
+    with open(path, 'rb') as f:
+        for chunk in iter(lambda: f.read(chunk_size), b''):
+            h.update(chunk)
+    return h.hexdigest()
+
+
+def build_search_fingerprint(cfg, net_result, data_file_sha256):
+    '''Everything that determines one network's theta search outcome
+    (run_theta_search()), for MDE_GMN_REUSE_BEST_PARAMS's cache-lookup
+    below: the knobs that shape that network's topology (MODE/GMN_MODE/
+    targets/MDE_D/NESTED_M/REGION_CHANNEL_CAP/the causality gates/
+    LIB_RATIO/DELAY_SET -- the last only actually matters for
+    MODE='MDE-GMN', and e_vals/tau_vals only for MODE='GMN', but
+    including all of them regardless of mode is simpler than branching
+    the fingerprint by mode, and only costs an occasional unnecessary
+    cache miss, never an incorrect hit), the knobs that shape the search
+    itself (the theta range, GENERATE_LENGTH), a hash of the actual
+    input data (see _hash_file), and the network's own realized node set
+    as a cheap extra check -- if every upstream knob and the data
+    genuinely match, the reconstructed network (already built by the
+    time this runs, cost already paid) should have exactly the same
+    nodes; disagreement here would mean some other nondeterminism crept
+    in upstream that the knob list alone didn't catch, and is a signal
+    to NOT trust the cache rather than silently reuse a stale search
+    result.
+
+    All values here are already JSON-safe (plain str/int/float/list) so
+    the returned dict can be dumped/compared directly with `==` after a
+    json round-trip, with no custom (de)serialization needed.'''
+    graph = net_result['network']['Graph']
+    return {
+        'mode': cfg['MODE'],
+        'gmn_mode': cfg['GMN_MODE'],
+        'targets': sorted(net_result['targets']),
+        'mde_d': cfg['MDE_D'],
+        'embed_dim_rho_min': cfg['EMBED_DIM_RHO_MIN'],
+        'cross_map_rho_min': cfg['CROSS_MAP_RHO_MIN'],
+        'exclusion_radius': 'auto' if cfg['EXCLUSION_RADIUS_AUTO'] else cfg['EXCLUSION_RADIUS_VALUE'],
+        'gmn_exclusion_radius': cfg['GMN_EXCLUSION_RADIUS'],
+        'nested_m': cfg['NESTED_M'],
+        'region_channel_cap': cfg['REGION_CHANNEL_CAP'],
+        'lib_ratio': cfg['LIB_RATIO'],
+        'generate_length': cfg['GENERATE_LENGTH'],
+        'delay_set': list(cfg['DELAY_SET']),
+        'e_vals': list(cfg['E_RANGE']),
+        'tau_vals': list(cfg['TAU_RANGE']),
+        'theta_vals': list(cfg['THETA_RANGE']),
+        'data_file': cfg['DATA_FILE'],
+        'data_file_sha256': data_file_sha256,
+        'prediction_start': cfg['PREDICTION_START'],
+        'prediction_length': cfg['PREDICTION_LENGTH'],
+        'network_nodes': sorted(graph.nodes),
+    }
+
+
+def find_cached_best_params(prev_runs_dir, network_id, fingerprint):
+    '''Scans prev_runs_dir (recursively, so it works whether prev_runs_dir
+    is the results/ root -- one level of job_<id>/ subdirectories -- or
+    already a single job's output dir) for best_params_<network_id>.json
+    files, newest (by mtime) first, and returns the first one whose
+    stored fingerprint exactly equals `fingerprint`. Returns None if
+    prev_runs_dir doesn't exist yet (e.g. a project's very first run) or
+    nothing matches -- both are ordinary "no cache hit" outcomes, not
+    errors, so the caller falls through to running the real search.'''
+    if not prev_runs_dir.exists():
+        return None
+    candidates = sorted(
+        prev_runs_dir.glob(f'**/best_params_{network_id}.json'),
+        key=lambda p: p.stat().st_mtime, reverse=True,
+    )
+    for path in candidates:
+        try:
+            with open(path) as f:
+                entry = json.load(f)
+        except (OSError, json.JSONDecodeError):
+            continue
+        if entry.get('fingerprint') == fingerprint:
+            return entry['best_node_theta'], path
+    return None
+
+
+def save_best_params(results_dir, network_id, fingerprint, best_node_theta, source):
+    '''Writes this run's own best_params_<network_id>.json -- the artifact
+    a future job's find_cached_best_params() looks for -- regardless of
+    whether this run computed it fresh or itself reused a cache hit, so
+    the provenance chain (source: 'computed' vs. 'reused_from:<path>')
+    stays visible in whichever job's output this file ends up in.'''
+    path = results_dir / f'best_params_{network_id}.json'
+    with open(path, 'w') as f:
+        json.dump({
+            'fingerprint': fingerprint,
+            'best_node_theta': best_node_theta,
+            'source': source,
+        }, f, indent=2)
+    return path
+
+
+def retune_theta_for_all_nodes(data_lib, graph, nodes, theta_range, cores, min_pred_rows=20):
+    '''Finds each node's own best theta (S-map's nonlinearity/
+    localization parameter -- see build_gmn_config()'s module-level
+    context for what it controls) via PredictNonlinear on a 70/30
+    lib/pred split of that node's own generalized embedding
+    (_embedding_spec_for()/_generalized_embed() -- built directly with
+    pandas/pyEDM here, not through gmn's Node/Network machinery, since
+    this runs once per network before any gmn.GMN object exists yet).
+    Theta is the only thing this pipeline still searches per node (see
+    run_theta_search()'s docstring for why E and each driver's delay no
+    longer need searching at all). Falls back to theta=3 (a reasonable
+    generic default) whenever there's too little data for the split
+    (fewer than min_pred_rows pred rows) or PredictNonlinear itself
+    fails, rather than letting one bad node abort the whole search.'''
     best_theta_by_node = {}
 
     for node in nodes:
-        columns = list(graph.predecessors(node)) + [node]
-        embedded = Embed(dataFrame=data_lib, E=E, tau=tau, columns=columns)
-        offset = (E - 1) * abs(tau)
-        embedded = embedded.iloc[offset:].reset_index(drop=True)
+        spec = _embedding_spec_for(graph, node)
+        embedded, offset = _generalized_embed(data_lib, spec)
         embedded.insert(0, 'Time', np.arange(len(embedded)))
         embedded[node] = data_lib[node].to_numpy()[offset:]
 
@@ -1254,13 +2173,21 @@
 
 
 def evaluate_smap_candidate(net_result, prediction_start, prediction_length,
-                             E, tau, node_theta, variant, results_dir, cores,
-                             generate_length=None):
+                             node_theta, variant, results_dir, cores,
+                             generate_length=None, generate_start=None, exclusion_radius=0):
+    '''Runs one Generate() call and scores it as mean rho, across
+    `net_result['targets']` only (not every driver node), between the
+    simulated and real held-out series -- this is "generative skill" as
+    this pipeline defines it everywhere: the theta search
+    (run_theta_search) and the start-point sweep (sweep_generate_start)
+    both call this same function, at the same objective, so a search
+    over either axis is comparable to the other.'''
+    generate_start = generate_start if generate_start is not None else prediction_start
     config_file = build_gmn_config(
         net_result, prediction_start, prediction_length, results_dir,
-        mode='Generate', E=E, tau=tau,
-        node_theta_overrides=node_theta,
+        mode='Generate', node_theta_overrides=node_theta,
         variant=variant, generate_length=generate_length,
+        generate_start=generate_start, exclusion_radius=exclusion_radius,
     )
     G = gmn.GMN( configFile = str(config_file), cores = cores )
     G.Generate()
@@ -1271,7 +2198,7 @@
 
     target_rhos = {}
     for target in net_result['targets']:
-        actual_vals = actual_df[target].to_numpy()[prediction_start: prediction_start + n_gen]
+        actual_vals = actual_df[target].to_numpy()[generate_start: generate_start + n_gen]
         gen_vals = generated[target].to_numpy()
         target_rhos[target] = ComputeError(actual_vals, gen_vals)['rho']
 
@@ -1279,147 +2206,177 @@
     return score, target_rhos, G
 
 
-def run_smap_search(gmn_runs, prediction_start, prediction_length, e_range, tau_range,
-                     theta_range, coarse_theta_range, search_mode, results_dir, cores,
-                     generate_length=None):
-    smap_search_results = {}
+def run_theta_search(gmn_runs, prediction_start, prediction_length, theta_range, results_dir,
+                      cores, generate_length=None, reuse_best_params=False, prev_runs_dir=None,
+                      fingerprints=None, exclusion_radius=0):
+    '''Retunes each network's per-node theta once
+    (retune_theta_for_all_nodes()), then scores the result with a single
+    Generate() call -- replacing this pipeline's old combined (E, tau,
+    theta) grid search (see the module docstring's SEARCH REUSE section,
+    and scripts/backup/simple-mde-gmn_pre-delay-embedding.py for that
+    implementation, kept for reference/comparison per explicit
+    direction). E and each driver's delay are no longer search
+    parameters at all: for MODE='MDE-GMN' they come directly from
+    nested-MDE's own discovered network (MultiDiGraph edges' delay
+    attribute plus self_delay_set -- see build_nested_mde_network()'s
+    and _embedding_spec_for()'s docstrings); for MODE='GMN' they still
+    come from that mode's own CCM E/tau grid search
+    (fallback_E/fallback_tau, set in build_network()). Either way there
+    is nothing left to grid-search over except theta -- a genuine
+    reduction in search cost, not just a relocation of it, since the old
+    search's dominant cost (len(E) x len(tau) full Generate() calls) is
+    now a single Generate() call per network.
+
+    reuse_best_params/prev_runs_dir/fingerprints: the same cross-job
+    reuse mechanism as before (see the module docstring's SEARCH REUSE
+    section) -- a network whose fingerprint matches a past job's cached
+    result skips even this one retuning-and-scoring step entirely.'''
+    theta_search_results = {}
     for network_id, run in gmn_runs.items():
         net_result = run['net_result']
         graph = net_result['network']['Graph']
         all_nodes = list(graph.nodes)
         data_after_mde_lib = pd.read_csv(net_result['DATA_AFTER_MDE']).iloc[:prediction_start]
 
-        e_vals, tau_vals = list(e_range), list(tau_range)
-        theta_vals = list(theta_range) if search_mode == 'joint' else [None]
-        n_candidates = len(e_vals) * len(tau_vals) * len(theta_vals)
+        fingerprint = fingerprints[network_id] if fingerprints else None
+        if reuse_best_params and fingerprint is not None:
+            cached = find_cached_best_params(prev_runs_dir, network_id, fingerprint)
+            if cached is not None:
+                best_node_theta, source = cached
+                print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS matched {source} -- '
+                      f'skipping theta retuning entirely', flush=True)
+                theta_search_results[network_id] = {
+                    'best_node_theta': best_node_theta, 'reused_from': str(source),
+                }
+                save_best_params(results_dir, network_id, fingerprint, best_node_theta,
+                                  source=f'reused_from:{source}')
+                continue
+            print(f'{network_id}: MDE_GMN_REUSE_BEST_PARAMS set but no matching cached run '
+                  f'found under {prev_runs_dir} -- retuning theta.', flush=True)
 
-        checkpoint_file = results_dir / f'smap_search_checkpoint_{network_id}_{search_mode}.pkl'
-        score_grid = np.full((len(e_vals), len(tau_vals), len(theta_vals)), np.nan)
-        if checkpoint_file.exists():
-            with open(checkpoint_file, 'rb') as f:
-                ckpt = pickle.load(f)
-            # generate_length is part of the validity check alongside
-            # e_vals/tau_vals/theta_vals -- each cell's score was computed
-            # against a specific Generate() step count, and a resumed run
-            # with a different MDE_GMN_GENERATE_LENGTH would otherwise
-            # silently mix scores from two different-length simulations
-            # into the same grid.
-            if (ckpt.get('e_vals') == e_vals and ckpt.get('tau_vals') == tau_vals
-                    and ckpt.get('theta_vals') == theta_vals
-                    and ckpt.get('generate_length') == generate_length):
-                score_grid = ckpt['score_grid']
-                print(f'{network_id}: resumed checkpoint, '
-                      f'{int(np.sum(~np.isnan(score_grid)))}/{score_grid.size} candidates already done',
-                      flush=True)
-
-        print(f'{network_id}: {search_mode} search over {n_candidates} (E, tau) candidates '
-              f'{"x " + str(len(theta_vals)) + " theta " if search_mode == "joint" else ""}'
-              f'(mode={search_mode})', flush=True)
-
-        start_t = time.monotonic()
-        done_at_start = int(np.sum(~np.isnan(score_grid)))
-        done = done_at_start
+        print(f'{network_id}: retuning theta for {len(all_nodes)} node(s)', flush=True)
+        best_node_theta = retune_theta_for_all_nodes(
+            data_after_mde_lib, graph, all_nodes, theta_range, cores,
+        )
 
-        for i, E in enumerate(e_vals):
-            for j, tau in enumerate(tau_vals):
-                if np.all(~np.isnan(score_grid[i, j])):
-                    continue
+        try:
+            score, _, _ = evaluate_smap_candidate(
+                net_result, prediction_start, prediction_length, best_node_theta,
+                'theta_search', results_dir, cores, generate_length=generate_length,
+                exclusion_radius=exclusion_radius,
+            )
+            print(f'{network_id}: theta search score={score:.4f}', flush=True)
+        except Exception as exc:
+            # gmn.Generate raises RuntimeError on SMap divergence (blown-up
+            # recursive forecast, >1E15) -- possible even at a retuned
+            # theta on a large, high-dimensional network, not a bug to
+            # crash the whole run over.
+            print(f'{network_id}: theta search candidate FAILED: {exc!r}', flush=True)
 
-                if search_mode == 'coarse_retune':
-                    node_theta_for_candidate = retune_theta_for_candidate(
-                        data_after_mde_lib, graph, all_nodes, E, tau, coarse_theta_range, cores,
-                    )
+        theta_search_results[network_id] = {'best_node_theta': best_node_theta}
 
-                for k, theta in enumerate(theta_vals):
-                    if not np.isnan(score_grid[i, j, k]):
-                        continue
-
-                    if search_mode == 'joint':
-                        node_theta = {n: theta for n in all_nodes}
-                    else:
-                        node_theta = node_theta_for_candidate
-
-                    variant = (f'smap_search_{search_mode}_E{E}_tau{tau}'
-                               + (f'_theta{theta}' if search_mode == 'joint' else ''))
-                    try:
-                        score, _, _ = evaluate_smap_candidate(
-                            net_result, prediction_start, prediction_length,
-                            E, tau, node_theta, variant, results_dir, cores,
-                            generate_length=generate_length,
-                        )
-                    except Exception as exc:
-                        # gmn.Generate raises RuntimeError on SMap divergence
-                        # (blown-up recursive forecast, >1E15) -- an expected
-                        # outcome for at least some (E, tau[, theta])
-                        # candidates on a large, high-dimensional network, not
-                        # a bug to crash the whole search over. Record this
-                        # cell as failed (score_grid keeps its NaN default,
-                        # same as target_E_tau_rho_grid's own except: pass
-                        # elsewhere in this file) and move on to the next
-                        # candidate instead of losing the rest of the search.
-                        print(f'[PROGRESS] {network_id} candidate {done + 1}/{n_candidates} '
-                              f'(E={E}, tau={tau}{f", theta={theta}" if search_mode == "joint" else ""}) '
-                              f'FAILED: {exc!r} -- scoring as NaN', flush=True)
-                        score = np.nan
-
-                    score_grid[i, j, k] = score
-
-                    with open(checkpoint_file, 'wb') as f:
-                        pickle.dump({'e_vals': e_vals, 'tau_vals': tau_vals,
-                                     'theta_vals': theta_vals, 'generate_length': generate_length,
-                                     'score_grid': score_grid}, f)
-
-                    done += 1
-                    elapsed = time.monotonic() - start_t
-                    per_candidate = elapsed / max(1, done - done_at_start)
-                    eta = per_candidate * (n_candidates - done)
-                    theta_str = f', theta={theta}' if search_mode == 'joint' else ''
-                    print(f'[PROGRESS] {network_id} candidate {done}/{n_candidates} '
-                          f'(E={E}, tau={tau}{theta_str}) score={score:.4f} '
-                          f'(elapsed {elapsed:6.1f}s, ETA {eta:6.1f}s)', flush=True)
-
-        smap_search_results[network_id] = {
-            'mode': search_mode,
-            'e_vals': e_vals, 'tau_vals': tau_vals, 'theta_vals': theta_vals,
-            'score_grid': score_grid,
-        }
+        if fingerprint is not None:
+            save_best_params(results_dir, network_id, fingerprint, best_node_theta, source='computed')
 
-        flat_grid = score_grid[:, :, 0] if search_mode != 'joint' else np.nanmax(score_grid, axis=2)
-        best_i, best_j = np.unravel_index(np.nanargmax(flat_grid), flat_grid.shape)
-        best_E, best_tau = e_vals[best_i], tau_vals[best_j]
-
-        if search_mode == 'joint':
-            best_k = int(np.nanargmax(score_grid[best_i, best_j]))
-            best_theta_global = theta_vals[best_k]
-            print(f'{network_id}: best E={best_E}, tau={best_tau}, theta={best_theta_global} '
-                  f'(score={score_grid[best_i, best_j, best_k]:.4f})', flush=True)
-            smap_search_results[network_id]['best_node_theta'] = {n: best_theta_global for n in all_nodes}
-        else:
-            best_node_theta = retune_theta_for_candidate(
-                data_after_mde_lib, graph, all_nodes, best_E, best_tau, coarse_theta_range, cores,
-            )
-            print(f'{network_id}: best E={best_E}, tau={best_tau} '
-                  f'(theta=coarse-retuned per node, score={flat_grid[best_i, best_j]:.4f})', flush=True)
-            smap_search_results[network_id]['best_node_theta'] = best_node_theta
+    return theta_search_results
 
-        smap_search_results[network_id]['best_E'] = best_E
-        smap_search_results[network_id]['best_tau'] = best_tau
 
-        fig, ax = plt.subplots(figsize=(8, 6))
-        im = ax.imshow(flat_grid, aspect='auto', cmap='viridis')
-        ax.set_xticks(range(len(tau_vals))); ax.set_xticklabels(tau_vals)
-        ax.set_yticks(range(len(e_vals))); ax.set_yticklabels(e_vals)
-        ax.set_xlabel('tau'); ax.set_ylabel('E')
-        ax.set_title(f'GMN E/tau search ({search_mode}) -- {network_id}\nmean Generate() rho across targets')
-        fig.colorbar(im, ax=ax, label='mean target rho')
-        rect = patches.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
-                                  fill=False, edgecolor='red', linewidth=2.5)
-        ax.add_patch(rect)
+# =============================================================================
+# Generate()/Forecast() start-point sweep
+# =============================================================================
+
+def sweep_generate_start(gmn_runs, theta_search_results, prediction_start, prediction_length,
+                          results_dir, cores, generate_length, start_offsets, exclusion_radius=0):
+    '''Finds, per network, which row Generate() should start free-running
+    from to best match the real held-out data -- at the per-node theta
+    run_theta_search() already chose, not re-searched here (E and each
+    driver's delay are never searched at all -- see that function's
+    docstring). GMN's Generate() is a recursive nonlinear simulation, so
+    (like any such system) how well it tracks reality can depend on
+    which state it launches from, independent of how good the embedding/
+    theta are; this sweep answers "given our best model, which starting
+    point does it explain best" as a separate question from "what is our
+    best model."
+
+    One evaluate_smap_candidate() call per offset in `start_offsets`
+    (each offset a row count added to prediction_start), scored on the
+    exact same mean-rho-across-targets objective the theta search uses
+    -- see that function's docstring. Checkpoints to disk after every
+    offset (same resume pattern as run_theta_search's cache) since each
+    candidate is its own full Generate() call and this shouldn't have to
+    restart from scratch after an interrupted job.
+
+    A single-offset `start_offsets` (the default -- see load_config()'s
+    GENERATE_START_MIN/MAX_OFFSET) means the sweep is disabled: that one
+    offset is returned directly with no Generate() call at all, since
+    rebuild_at_best_params() is about to run Generate() at that exact
+    (theta, start) anyway -- running it again here first would just be a
+    wasted duplicate call.
+
+    Returns {network_id: best_generate_start_row}.'''
+    best_starts = {}
+    for network_id, run in gmn_runs.items():
+        if len(start_offsets) == 1:
+            best_starts[network_id] = prediction_start + start_offsets[0]
+            continue
+
+        net_result = run['net_result']
+        node_theta = theta_search_results[network_id]['best_node_theta']
+
+        checkpoint_file = results_dir / f'generate_start_checkpoint_{network_id}.pkl'
+        scores = {}
+        if checkpoint_file.exists():
+            with open(checkpoint_file, 'rb') as f:
+                ckpt = pickle.load(f)
+            if (ckpt.get('start_offsets') == start_offsets
+                    and ckpt.get('generate_length') == generate_length):
+                scores = ckpt['scores']
+                print(f'{network_id}: resumed start-sweep checkpoint, '
+                      f'{len(scores)}/{len(start_offsets)} offsets already done', flush=True)
+
+        for offset in start_offsets:
+            if offset in scores and not np.isnan(scores[offset]):
+                continue
+
+            generate_start = prediction_start + offset
+            try:
+                score, _, _ = evaluate_smap_candidate(
+                    net_result, prediction_start, prediction_length,
+                    node_theta, f'startsweep_offset{offset}', results_dir, cores,
+                    generate_length=generate_length, generate_start=generate_start,
+                    exclusion_radius=exclusion_radius,
+                )
+            except Exception as exc:
+                # Same rationale as run_theta_search's own try/except: a
+                # divergent SMap Generate() is an expected failure mode
+                # for at least some starting points, not a bug to crash
+                # the whole sweep over.
+                print(f'{network_id}: start offset {offset} FAILED: {exc!r} -- scoring as NaN', flush=True)
+                score = np.nan
+
+            scores[offset] = score
+            with open(checkpoint_file, 'wb') as f:
+                pickle.dump({'start_offsets': start_offsets,
+                             'generate_length': generate_length, 'scores': scores}, f)
+            print(f'{network_id}: start offset {offset:5d} (row {generate_start}) score={score:.4f}', flush=True)
+
+        valid_offsets = [o for o in start_offsets if not np.isnan(scores.get(o, np.nan))]
+        best_offset = max(valid_offsets, key=lambda o: scores[o]) if valid_offsets else start_offsets[0]
+        best_starts[network_id] = prediction_start + best_offset
+        print(f'{network_id}: best generate start = row {best_starts[network_id]} '
+              f'(offset {best_offset}, score={scores.get(best_offset, float("nan")):.4f})', flush=True)
+
+        fig, ax = plt.subplots(figsize=(7, 4))
+        ax.plot(start_offsets, [scores.get(o, np.nan) for o in start_offsets], marker='o')
+        ax.axvline(best_offset, color='gray', ls='--', lw=0.8)
+        ax.set_xlabel('Start offset from PREDICTION_START (rows)')
+        ax.set_ylabel('Generate() rho (mean over targets)')
+        ax.set_title(f'Generate() start-point sweep -- {network_id}')
         fig.tight_layout()
-        fig.savefig(results_dir / f'gmn_E_tau_search_{search_mode}_{network_id}.png', dpi=300)
+        fig.savefig(results_dir / f'generate_start_sweep_{network_id}.png', dpi=300)
         plt.close(fig)
 
-    return smap_search_results
+    return best_starts
 
 
 # =============================================================================
@@ -1427,75 +2384,178 @@
 # (mirrors notebook cells 11-13)
 # =============================================================================
 
-def rebuild_at_best_params(gmn_runs, smap_search_results, prediction_start, prediction_length,
-                            results_dir, cores, generate_length=None):
+def rebuild_at_best_params(gmn_runs, theta_search_results, prediction_start, prediction_length,
+                            results_dir, cores, generate_length=None, generate_starts=None,
+                            exclusion_radius=0):
+    '''Reruns Generate() at the winning per-node theta found by
+    run_theta_search(), and the winning start row found by
+    sweep_generate_start() (generate_starts; defaults to prediction_start
+    per network when None). Plotting is deliberately NOT done here -- it
+    used to draw its own actual-vs-Generate() figure, but that was a
+    strict subset of run_forecast_diagnostic()'s own actual-vs-
+    Generate()-vs-Forecast() figure (same two of its three lines, same
+    nodes), so the two have been merged into the one figure
+    run_forecast_diagnostic() draws, and this function now only reruns
+    Generate() (needed downstream by run_forecast_diagnostic,
+    gmn_fidelity_rho, and the raw output CSV) and prints the per-node
+    rho.
+
+    Unlike run_theta_search()'s/sweep_generate_start()'s own probe
+    Generate() calls, this one used to be completely unguarded --
+    gmn.Generate() can still raise RuntimeError on SMap divergence
+    (>1E15) here even at an already-retuned theta and swept start row
+    (both only ever *tested* a subset of what gets rebuilt: a theta that
+    diverged during search can still be handed here if run_theta_search()
+    swallowed that failure and kept it anyway -- see that function's own
+    try/except), and this is the one Generate() call everything
+    downstream (run_forecast_diagnostic, gmn_fidelity_rho, the raw output
+    CSV) depends on. Now try/except-guarded per network: a failure here
+    is logged and that network is skipped (run['G'] stays unset) rather
+    than crashing the whole job and losing every other, unrelated
+    network's already-finished or still-pending work -- callers must
+    check for run.get('G') before using a network's result (main() does).'''
     for network_id, run in gmn_runs.items():
         net_result = run['net_result']
-        search = smap_search_results[network_id]
-        best_E, best_tau = search['best_E'], search['best_tau']
-        node_theta = search['best_node_theta']
-
-        config_file = build_gmn_config(
-            net_result, prediction_start, prediction_length, results_dir,
-            mode='Generate', E=best_E, tau=best_tau,
-            node_theta_overrides=node_theta,
-            variant='best', generate_length=generate_length,
-        )
+        node_theta = theta_search_results[network_id]['best_node_theta']
+        generate_start = generate_starts[network_id] if generate_starts else prediction_start
 
-        G = gmn.GMN( configFile = str(config_file), cores = cores )
-        G.Generate()
+        try:
+            config_file = build_gmn_config(
+                net_result, prediction_start, prediction_length, results_dir,
+                mode='Generate', node_theta_overrides=node_theta,
+                variant='best', generate_length=generate_length, generate_start=generate_start,
+                exclusion_radius=exclusion_radius,
+            )
+            G = gmn.GMN( configFile = str(config_file), cores = cores )
+            G.Generate()
+        except Exception as exc:
+            print(f'{network_id}: FINAL Generate() rebuild FAILED: {exc!r} -- '
+                  f'skipping this network (no plots/fidelity/output for it)', flush=True)
+            continue
 
         run['G'] = G
-        run['best_E'] = best_E
-        run['best_tau'] = best_tau
         run['best_node_theta'] = node_theta
+        run['generate_start'] = generate_start
 
         actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
         generated = G.DataOut.reset_index(drop=True)
-        generate_time = np.arange(prediction_start, prediction_start + len(generated))
-        n_rows = prediction_start + prediction_length
         nodes = [c for c in generated.columns if c != 'Time']
 
-        fig, axes = plt.subplots(len(nodes), 1, figsize=(11, 2.2 * len(nodes)), sharex=False)
-        if len(nodes) == 1:
-            axes = [axes]
-
-        for ax, node in zip(axes, nodes):
-            ax.plot(np.arange(n_rows), actual_df[node].to_numpy(),
-                    color='tab:blue', lw=1, label='Lib/actual')
-            ax.plot(generate_time, generated[node].to_numpy(),
-                    color='tab:red', lw=1.2, label=f'Generate() SMap E={best_E}, tau={best_tau}')
-            ax.axvline(prediction_start, color='gray', ls='--', lw=0.8)
-            ax.set_title(node, fontsize=10)
-            ax.legend(fontsize=7, loc='upper right')
-
-        fig.suptitle(f"GMN Generate() (SMap, {search['mode']} search) -- {network_id}", fontsize=11)
-        fig.tight_layout()
-        fig.savefig(results_dir / f'gmn_generate_{network_id}.png', dpi=300)
-        plt.close(fig)
-
-        print(f"{network_id} Generate() rho, held-out window:", flush=True)
+        print(f"{network_id} Generate() rho, held-out window (start row {generate_start}):", flush=True)
         for node in nodes:
-            # Sliced to len(generated), not n_rows/prediction_length: once
+            # Sliced to len(generated), not prediction_length: once
             # generate_length can be shorter than the full held-out window
             # (the whole point of decoupling it), actual_df's slice must
             # match Generate()'s actual output length or ComputeError()
             # compares mismatched-length arrays.
-            a = actual_df[node].to_numpy()[prediction_start:prediction_start + len(generated)]
+            a = actual_df[node].to_numpy()[generate_start:generate_start + len(generated)]
             r = ComputeError(a, generated[node].to_numpy())['rho']
             tag = 'TARGET' if node in net_result['targets'] else 'driver'
             print(f"  {node:20s} {tag:7s} rho={r:7.3f}", flush=True)
 
 
 def run_forecast_diagnostic(net_result, G, prediction_start, prediction_length,
-                             E, tau, node_theta, results_dir, cores, save_path):
-    config_file = build_gmn_config(
-        net_result, prediction_start, prediction_length, results_dir, mode='Forecast',
-        E=E, tau=tau, node_theta_overrides=node_theta,
-    )
+                             node_theta, results_dir, cores, save_path,
+                             generate_length=None, generate_start=None, nodes_per_figure=10,
+                             exclusion_radius=0, plot_range_mode='full'):
+    '''Runs the Forecast() diagnostic and draws the one actual-vs-
+    Generate()-vs-Forecast() figure for this network (see
+    rebuild_at_best_params()'s docstring for why there's only one figure
+    here rather than a separate Generate()-only one).
+
+    generate_start (defaults to prediction_start when None) anchors both
+    Forecast()'s own pred window and the plotted window to the same row
+    sweep_generate_start() picked for G's own Generate() run -- so all
+    three lines on the figure (actual, Generate(), Forecast()) describe
+    the same starting point, "to be consistent with the generative mode."
+    See GENERATE_START_MIN/MAX/STEP_OFFSET in load_config() for how a
+    swept starting point ends up here: sweep_generate_start() only ever
+    changes this one row number (per network) that both Generate() (G,
+    already run by rebuild_at_best_params() by the time this is called)
+    and this function's own Forecast() call anchor to -- it does not
+    otherwise reshape either trace or this plot.
+
+    generate_length is the length Generate() (G, passed in already built)
+    free-ran for -- the length actually used for Forecast()'s own pred
+    window and for how far both traces get *plotted* depends on
+    plot_range_mode, not directly on this value alone; see that
+    parameter.
+
+    plot_range_mode -- two options, per explicit direction, both anchored
+    at generate_start:
+      'full' (default, previous unconditional behavior): actual/lib is
+        plotted from row 0 through generate_start+generate_length;
+        Forecast() itself computes exactly generate_length steps, same as
+        Generate(). All three traces effectively share one right edge.
+      'windowed': plotted window is asymmetric around generate_start --
+        2*generate_length rows before it (actual/lib only -- Generate()
+        never ran that far back) and 2*generate_length rows after it,
+        clamped to the data actually available (can't plot before row 0,
+        and can't plot past this network's held-out
+        prediction_start+prediction_length rows -- see actual_df's own
+        construction in build_network()). Forecast() (this function's own
+        call, independent of Generate()'s own generate_length -- Forecast
+        is a direct per-row SMap prediction against real held-out
+        predictor data, not a recursive simulation like Generate(), so
+        asking it for more steps here doesn't compound error the way
+        widening Generate() would) is deliberately run out to that same
+        2*generate_length-clamped length instead of generate_length, so
+        its trace visibly spans twice Generate()'s own length rather than
+        stopping at the window's midpoint. Net effect: actual/lib always
+        spans wider than Generate()'s own window (it's real data, cheap
+        to show further out); Forecast() always spans exactly twice
+        Generate()'s window (real computation, capped by available
+        data); Generate() itself is unchanged, still only generate_length
+        long -- the three traces' differing plotted extents are the
+        point, not a bug: it isolates "how far can this network's own
+        embedding usefully predict" (Forecast, cheap to extend) from "how
+        far did we actually ask the recursive simulation to run"
+        (Generate, deliberately unchanged here).
+
+    Split across ceil(len(nodes) / nodes_per_figure) separate files
+    (save_path with a _partNN suffix inserted before its extension)
+    rather than one figure with a subplot per node -- this network can
+    have 100+ nodes (see build_nested_mde_network()'s docstring on
+    network size), and a single figure that tall was slow to render and
+    load for no benefit over several page-sized ones.
+
+    Returns (None, {}) instead of raising if the Forecast() build/call
+    itself fails -- either gmn.Forecast() raising RuntimeError on SMap
+    divergence (>1E15, the same failure mode Generate() risks, per that
+    function's own docstring -- Forecast() isn't recursive so it can't
+    compound error the way Generate() does, but its own local SMap fits
+    can still occasionally blow up) or build_gmn_config()'s own explicit
+    ValueError (an infeasible offset/forecast_length combination, more
+    reachable than before now that plot_range_mode='windowed' can request
+    a longer forecast_compute_length than the 'full' default ever did).
+    Previously unguarded -- see rebuild_at_best_params()'s docstring for
+    the same rationale, applied here: a bad Forecast() must not crash
+    every other, unrelated network in the same job. Callers must check
+    for a None first return value before using it.'''
+    generate_length = generate_length if generate_length is not None else prediction_length
+    generate_start = generate_start if generate_start is not None else prediction_start
+    total_rows = prediction_start + prediction_length  # == len(actual_df) -- see build_network()
+
+    if plot_range_mode == 'windowed':
+        forecast_compute_length = min(2 * generate_length, max(total_rows - generate_start, 0))
+    else:
+        forecast_compute_length = generate_length
 
-    Gf = gmn.GMN( configFile = str(config_file), cores = cores )
-    Gf.Forecast()
+    try:
+        max_offset = network_max_embedding_offset(net_result['network']['Graph'])
+        config_file = build_gmn_config(
+            net_result, prediction_start, prediction_length, results_dir, mode='Forecast',
+            node_theta_overrides=node_theta, generate_length=forecast_compute_length,
+            generate_start=generate_start, max_offset=max_offset,
+            exclusion_radius=exclusion_radius,
+        )
+        Gf = gmn.GMN( configFile = str(config_file), cores = cores )
+        Gf.Forecast()
+    except Exception as exc:
+        print(f"{net_result['network_id']}: Forecast() diagnostic FAILED: {exc!r} -- "
+              f"skipping its plot/rho for this network (Generate()-based outputs are unaffected)",
+              flush=True)
+        return None, {}
 
     actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
     forecast_out = Gf.DataOut.dropna().reset_index(drop=True)
@@ -1503,37 +2563,61 @@
     nodes = [c for c in forecast_out.columns if c != 'Time']
 
     generate_out = G.DataOut.reset_index(drop=True)
-    generate_time = np.arange(prediction_start, prediction_start + len(generate_out))
-    n_rows = prediction_start + prediction_length
+    generate_time = np.arange(generate_start, generate_start + len(generate_out))
+
+    if plot_range_mode == 'windowed':
+        plot_start = generate_start - min(2 * generate_length, generate_start)
+        plot_end = generate_start + forecast_compute_length
+    else:
+        plot_start = 0
+        plot_end = generate_start + generate_length
 
     rhos = {}
-    fig, axes = plt.subplots(len(nodes), 1, figsize=(11, 2.2 * len(nodes)), sharex=False)
-    if len(nodes) == 1:
-        axes = [axes]
-
-    for ax, node in zip(axes, nodes):
-        actual = actual_df[node].to_numpy()
-        forecast_pred = forecast_out[node].to_numpy()
-        rhos[node] = ComputeError(actual[time_idx], forecast_pred)['rho']
-
-        ax.plot(np.arange(n_rows), actual, color='tab:blue', lw=1, label='Lib/actual')
-        if node in generate_out.columns:
-            ax.plot(generate_time, generate_out[node].to_numpy(),
-                    color='tab:red', lw=1, label='Generate()')
-        ax.plot(time_idx, forecast_pred, color='tab:purple', lw=1.2, ls=':', label='Forecast()')
-        ax.axvline(prediction_start, color='gray', ls='--', lw=0.8)
-        ax.set_title(f"{node}   Forecast rho={rhos[node]:.3f}", fontsize=10)
-        ax.legend(fontsize=7, loc='upper right')
+    n_parts = (len(nodes) + nodes_per_figure - 1) // nodes_per_figure
+    for part in range(n_parts):
+        part_nodes = nodes[part * nodes_per_figure:(part + 1) * nodes_per_figure]
+        fig, axes = plt.subplots(len(part_nodes), 1, figsize=(11, 2.2 * len(part_nodes)), sharex=False)
+        if len(part_nodes) == 1:
+            axes = [axes]
 
-    fig.suptitle(f"Forecast() vs Generate() vs actual -- {net_result['network_id']}", fontsize=11)
-    fig.tight_layout()
-    fig.savefig(save_path, dpi=300)
-    plt.close(fig)
+        for ax, node in zip(axes, part_nodes):
+            actual = actual_df[node].to_numpy()
+            forecast_pred = forecast_out[node].to_numpy()
+            rhos[node] = ComputeError(actual[time_idx], forecast_pred)['rho']
+
+            # alpha=0.6 (not fully opaque) so the overlaid Generate()/
+            # Forecast() lines stay easy to pick out against the real
+            # trace -- per explicit direction.
+            ax.plot(np.arange(plot_start, plot_end), actual[plot_start:plot_end],
+                     color='tab:blue', lw=1, alpha=0.6, label='Lib/actual')
+            if node in generate_out.columns:
+                ax.plot(generate_time, generate_out[node].to_numpy(),
+                        color='tab:red', lw=1, label='Generate()')
+            ax.plot(time_idx, forecast_pred, color='tab:purple', lw=1.2, ls=':', label='Forecast()')
+            ax.axvline(generate_start, color='gray', ls='--', lw=0.8)
+            ax.set_title(f"{node}   Forecast rho={rhos[node]:.3f}", fontsize=10)
+            ax.legend(fontsize=7, loc='upper right')
+
+        fig.suptitle(f"Generate() vs Forecast() vs actual -- {net_result['network_id']} "
+                     f"(nodes {part * nodes_per_figure + 1}-"
+                     f"{part * nodes_per_figure + len(part_nodes)} of {len(nodes)})", fontsize=11)
+        fig.tight_layout()
+        part_path = save_path.with_name(f'{save_path.stem}_part{part + 1:02d}{save_path.suffix}')
+        fig.savefig(part_path, dpi=300)
+        plt.close(fig)
 
     return Gf, rhos
 
 
-def gmn_fidelity_rho(net_result, G, prediction_start, default_E=3, tau=-1):
+def gmn_fidelity_rho(net_result, G, generate_start, default_E=3, tau=-1):
+    '''Fig-4e-style generative fidelity: per node, a fresh Simplex predicts
+    the real actual series *from* the generated series (not the other way
+    around, and not a direct rho(actual, generated) comparison) on a 50/50
+    split of the generated window -- this tests whether the generated
+    trajectory's own dynamics stay on the same manifold as reality, a
+    stricter/different question than pointwise closeness. E defaults to
+    node_dimensions (an EmbedDimension-based estimate from build_network(),
+    when computed -- see that function's docstring on when it isn't).'''
     actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
     generated = G.DataOut.reset_index(drop=True)
     nodes = [c for c in generated.columns if c != 'Time']
@@ -1544,7 +2628,7 @@
     fidelity_rho = {}
     for node in nodes:
         gen_vals = generated[node].to_numpy()
-        actual_vals = actual_df[node].to_numpy()[prediction_start: prediction_start + n_gen]
+        actual_vals = actual_df[node].to_numpy()[generate_start: generate_start + n_gen]
         pair_df = pd.DataFrame({
             'Time': np.arange(n_gen),
             f'{node}_gen': gen_vals,
@@ -1711,18 +2795,43 @@
               f"({cfg['PREDICTION_LENGTH']} rows); clamping to it.", flush=True)
         cfg['GENERATE_LENGTH'] = cfg['PREDICTION_LENGTH']
 
+    # Resolve the generate-start sweep's candidate offsets now that
+    # PREDICTION_LENGTH/GENERATE_LENGTH are known: drop any offset that
+    # would leave too little real data to score GENERATE_LENGTH steps
+    # against (see load_config()'s docstring on GENERATE_START_MAX_OFFSET).
+    max_feasible_offset = cfg['PREDICTION_LENGTH'] - cfg['GENERATE_LENGTH']
+    start_offsets = list(range(cfg['GENERATE_START_MIN_OFFSET'], cfg['GENERATE_START_MAX_OFFSET'],
+                                cfg['GENERATE_START_STEP']))
+    feasible_offsets = [o for o in start_offsets if 0 <= o <= max_feasible_offset]
+    if feasible_offsets != start_offsets:
+        print(f'GENERATE_START offsets {start_offsets} include values outside the feasible '
+              f'range [0, {max_feasible_offset}]; using {feasible_offsets} instead.', flush=True)
+    cfg['GENERATE_START_OFFSETS'] = feasible_offsets or [0]
+
     print('Config:', {k: v for k, v in cfg.items() if k != 'TARGET_COLORS'}, flush=True)
 
     obs_cols = observable_columns(df, cfg['REGION_CHANNEL_CAP'])
 
     # ---- MDE (or its full-channel-pool stand-in for MODE='GMN') -------
     mde_results = {}
+    expanded_df = expanded_obs_cols = None  # MDE-GMN only -- see below
     if cfg['MODE'] == 'MDE-GMN':
+        # Built once, shared by every depth-0 target below and by
+        # build_nested_mde_network()'s own nested calls (via build_network()
+        # -- see expand_delay_columns()'s and build_nested_mde_network()'s
+        # docstrings) -- both need to search the exact same delay-expanded
+        # candidate pool for the depth-0 reuse (build_nested_mde_network's
+        # own docstring) to be valid.
+        expanded_df, expanded_obs_cols = expand_delay_columns(df, obs_cols, cfg['DELAY_SET'])
+
         for target in cfg['TARGETS']:
+            candidate_pool = [c for c in expanded_obs_cols
+                              if observable_and_delay(c, obs_cols)[0] != target]
+            effective_d = effective_mde_dimension(cfg['MDE_D'], candidate_pool, len(cfg['DELAY_SET']))
             mde_results[target] = run_mde_for_target(
-                target, df, obs_cols, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
-                cfg['MDE_D'], cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
-                cfg['CORES'], cfg['RESULTS_DIR'],
+                target, expanded_df, candidate_pool, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
+                effective_d, cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
+                cfg['EXCLUSION_RADIUS_VALUE'], cfg['CORES'], cfg['RESULTS_DIR'],
             )
 
         if cfg['RUN_MDE_DIAGNOSTICS']:
@@ -1731,7 +2840,7 @@
                 cfg['RESULTS_DIR'] / 'mde_rho_vs_dimension_multi_target.png',
             )
             plot_3d_mde_projection(
-                cfg['TARGETS'][0], cfg['SECOND_TARGET'], mde_results, df,
+                cfg['TARGETS'][0], cfg['SECOND_TARGET'], mde_results, df, obs_cols,
                 cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                 cfg['RESULTS_DIR'] / f"mde_3d_projection_{cfg['TARGETS'][0]}.png",
             )
@@ -1750,6 +2859,9 @@
             cfg['INTERACTION_MATRIX_OUTER_CORES'],
             observable_columns=obs_cols, nested_n=cfg['MDE_D'], nested_m=cfg['NESTED_M'],
             embed_dim_rho_min=cfg['EMBED_DIM_RHO_MIN'], cross_map_rho_min=cfg['CROSS_MAP_RHO_MIN'],
+            exclusion_radius=cfg['EXCLUSION_RADIUS_VALUE'],
+            expanded_df=expanded_df, expanded_observable_columns=expanded_obs_cols,
+            delay_set=cfg['DELAY_SET'],
         )
         draw_colored_network(
             net_result, cfg['TARGET_COLORS'],
@@ -1757,33 +2869,66 @@
         )
         gmn_runs[net_result['network_id']] = {'group': group, 'net_result': net_result}
 
-    smap_search_results = run_smap_search(
-        gmn_runs, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
-        cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['THETA_RANGE'], cfg['COARSE_THETA_RANGE'],
-        cfg['SMAP_SEARCH_MODE'], cfg['RESULTS_DIR'], cfg['CORES'],
+    fingerprints = None
+    if cfg['REUSE_BEST_PARAMS']:
+        data_file_sha256 = _hash_file(cfg['DATA_DIR'] / cfg['DATA_FILE'])
+        fingerprints = {
+            network_id: build_search_fingerprint(cfg, run['net_result'], data_file_sha256)
+            for network_id, run in gmn_runs.items()
+        }
+
+    theta_search_results = run_theta_search(
+        gmn_runs, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'], cfg['THETA_RANGE'],
+        cfg['RESULTS_DIR'], cfg['CORES'],
         generate_length=cfg['GENERATE_LENGTH'],
+        reuse_best_params=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
+        fingerprints=fingerprints, exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
+    )
+
+    generate_starts = sweep_generate_start(
+        gmn_runs, theta_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
+        cfg['RESULTS_DIR'], cfg['CORES'], cfg['GENERATE_LENGTH'], cfg['GENERATE_START_OFFSETS'],
+        exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
     )
 
     rebuild_at_best_params(
-        gmn_runs, smap_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
+        gmn_runs, theta_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
         cfg['RESULTS_DIR'], cfg['CORES'],
-        generate_length=cfg['GENERATE_LENGTH'],
+        generate_length=cfg['GENERATE_LENGTH'], generate_starts=generate_starts,
+        exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
     )
 
     for network_id, run in gmn_runs.items():
+        if 'G' not in run:
+            # rebuild_at_best_params() already logged why and skipped
+            # this network -- nothing downstream (Forecast, fidelity, the
+            # output CSV) has anything to work with without G.
+            print(f'{network_id}: no successful Generate() rebuild -- skipping '
+                  f'Forecast()/fidelity/output for this network', flush=True)
+            continue
+
         Gf, forecast_rho = run_forecast_diagnostic(
             run['net_result'], run['G'], cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
-            run['best_E'], run['best_tau'], run['best_node_theta'],
+            run['best_node_theta'],
             cfg['RESULTS_DIR'], cfg['CORES'],
-            cfg['RESULTS_DIR'] / f'gmn_forecast_vs_generate_{network_id}.png',
+            cfg['RESULTS_DIR'] / f'gmn_generate_forecast_{network_id}.png',
+            generate_length=cfg['GENERATE_LENGTH'], generate_start=run['generate_start'],
+            exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
+            plot_range_mode=cfg['FORECAST_PLOT_RANGE'],
         )
         run['Gf'] = Gf
         run['forecast_rho'] = forecast_rho
-        print(f"{network_id} Forecast() rho per node (weakest first):", flush=True)
-        for node, rho in sorted(forecast_rho.items(), key=lambda kv: kv[1]):
-            print(f"  {node:20s} {rho:.4f}", flush=True)
+        if Gf is None:
+            # run_forecast_diagnostic() already logged why -- Generate()
+            # (run['G']) still succeeded, so fidelity/the output CSV
+            # below are unaffected and still worth producing.
+            print(f'{network_id}: no Forecast() rho to report for this network', flush=True)
+        else:
+            print(f"{network_id} Forecast() rho per node (weakest first):", flush=True)
+            for node, rho in sorted(forecast_rho.items(), key=lambda kv: kv[1]):
+                print(f"  {node:20s} {rho:.4f}", flush=True)
 
-        rho = gmn_fidelity_rho(run['net_result'], run['G'], cfg['PREDICTION_START'])
+        rho = gmn_fidelity_rho(run['net_result'], run['G'], run['generate_start'])
         plot_fidelity(rho, network_id, cfg['RESULTS_DIR'] / f'gmn_fidelity_{network_id}.png')
         print(f'{network_id} fidelity rho (weakest first):', flush=True)
         for node in sorted(rho, key=lambda n: (np.isnan(rho[n]), -(rho[n] if not np.isnan(rho[n]) else 0))):
```

---


## Job 35998003

Generated: 2026-08-02T21:34:38+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_35928213**

### Outcome (manually appended note, not from generate_job_description.sh)

Failed during the generative (Generate()) step, most likely due to an
insufficient number of observables -- this run used `REGION_CHANNEL_CAP=15`,
a deliberately tiny smoke-test pool (see summary below), which apparently
wasn't enough for the nested-MDE network / Generate() step to run cleanly.

### Plain-language summary

Lightweight smoke test submitted alongside job_35928213 (left running
unaffected) purely to check the pipeline still runs end-to-end without
error -- specifically to exercise the divergence try/except fix added in
job_35928213 on a fast turnaround, rather than waiting out a multi-day
full-scale run to find out. Every knob here trades scientific fidelity
for speed; none of it is meant to produce usable results:

- `REGION_CHANNEL_CAP=15`, `MDE_D=4`, `NESTED_M=1` -- shrinks the nested-MDE
  network from the real run's 140+ nodes down to a handful, per this same
  sbatch file's own pre-existing comment recommending exactly this
  combination for a first test run.
- `E_MAX=7`/`TAU_MIN=-5` (vs. 16/-15) -- shrinks the E/tau search grid from
  56 candidates to 6.
- `GENERATE_LENGTH=50` (vs. 300) -- shortens every Generate() call inside
  the search loop.
- `--time=00:30:00` (vs. 2 days) -- fails fast if something hangs instead
  of sitting queued/running for a full allocation.

The canonical `scripts/batch/simple-mde-gmn.sbatch` was restored to the
real job_35928213 values (999/5/3/300/full grid/2-day time limit)
immediately after this job was submitted -- Slurm snapshots a script's
full contents at submission time, so this round-trip edit doesn't affect
either the already-submitted job_35928213 or this job_35998003 itself,
and leaves the canonical file back in its real-experiment state for any
future default `submit.sh` call. `scripts/src/simple-mde-gmn.py` was not
touched -- same divergence-handling fix as job_35928213.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_35928213 | job_35998003 |
|---|---|---|
| E_MAX | 16 | 7 |
| GENERATE_LENGTH | 300 | 50 |
| MDE_D | 5 | 4 |
| NESTED_M | 3 | 1 |
| REGION_CHANNEL_CAP | 999 | 15 |
| TAU_MIN | -15 | -5 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

```diff
--- job_35928213/simple-mde-gmn.sbatch
+++ job_35998003/simple-mde-gmn.sbatch
@@ -1,7 +1,7 @@
 #!/bin/bash
 #SBATCH --job-name=run_simple_mde_gmn
 #SBATCH --partition=compute
-#SBATCH --time=2-00:00:00
+#SBATCH --time=00:30:00
 #SBATCH --mem=48G
 #SBATCH --cpus-per-task=32
 # ADJUST: must be a literal absolute path, not $HOME/$PROJ_DIR -- Slurm
```

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---


## Job 35928213

Generated: 2026-08-01T15:12:34+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_35886791**

### Plain-language summary

job_35886791 crashed after ~11.5 hours (candidate 2 of 56 in the E/tau
search) with `RuntimeError: Generate:SMap() Node: CA1_80 Divergence detected
(>1E15)`. Investigation ([results/job_35886791](results/job_35886791)'s
`.err`) found this is a real, expected numerical failure mode of gmn's
recursive S-map `Generate()` step -- not a bug the nested-MDE
parallelization work introduced, and not evidence that any node was
assigned more drivers than `MDE_D` allows (a separate per-node
`EmbedDimension`-derived diagnostic value, `N_D`, happened to be larger
than `MDE_D` for `CA1_80`, which looks suspicious at a glance but turned
out to be unrelated -- `N_D` can only ever shrink a node's already
`MDE_D`-capped driver count via a `min()`, never grow it, and it isn't
even used as the `E` passed into the `Generate()` call that diverged).
The real reason it surfaced now: `REGION_CHANNEL_CAP=999` (restored in
job_35886791, vs. a deliberately tiny 15 in job_35885157's parallelization
smoke test) produces a much larger, higher-dimensional network
(143-166 nodes), giving the 56-candidate automated (E, tau) search far
more chances to land on at least one numerically unstable combination for
at least one node.

This job makes exactly one change: `evaluate_smap_candidate`'s call inside
`run_smap_search` (`simple-mde-gmn.py`) is now wrapped in `try/except`,
matching the `except Exception: pass` pattern `target_E_tau_rho_grid`
already uses elsewhere in this file. A divergent (or otherwise-failing)
candidate is now logged as `FAILED` and scored `NaN` instead of crashing
the whole job; the search then continues to the remaining candidates, and
best-candidate selection downstream already uses `nanargmax`, so NaN'd
cells are correctly skipped. No science/config parameters changed --
same `REGION_CHANNEL_CAP=999`, `MDE_D=5`, `NESTED_M=3`,
`GMN_MODE=separate`, `GENERATE_LENGTH=300` as job_35886791.

Known caveat: the search's checkpoint/resume logic treats a `NaN` cell as
"not yet computed," so if this job is ever checkpoint-resumed, a
candidate that fails deterministically will be retried (and fail again)
rather than being remembered as permanently bad. Not fixed here since it
was out of scope for unblocking this crash -- worth a follow-up if
checkpoint-resume of a partially-failed search becomes common.

### Parameter changes (simple-mde-gmn.sbatch)

_No parameter changes._

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

```diff
--- job_35886791/simple-mde-gmn.py
+++ job_35928213/simple-mde-gmn.py
@@ -1341,11 +1341,27 @@
 
                     variant = (f'smap_search_{search_mode}_E{E}_tau{tau}'
                                + (f'_theta{theta}' if search_mode == 'joint' else ''))
-                    score, _, _ = evaluate_smap_candidate(
-                        net_result, prediction_start, prediction_length,
-                        E, tau, node_theta, variant, results_dir, cores,
-                        generate_length=generate_length,
-                    )
+                    try:
+                        score, _, _ = evaluate_smap_candidate(
+                            net_result, prediction_start, prediction_length,
+                            E, tau, node_theta, variant, results_dir, cores,
+                            generate_length=generate_length,
+                        )
+                    except Exception as exc:
+                        # gmn.Generate raises RuntimeError on SMap divergence
+                        # (blown-up recursive forecast, >1E15) -- an expected
+                        # outcome for at least some (E, tau[, theta])
+                        # candidates on a large, high-dimensional network, not
+                        # a bug to crash the whole search over. Record this
+                        # cell as failed (score_grid keeps its NaN default,
+                        # same as target_E_tau_rho_grid's own except: pass
+                        # elsewhere in this file) and move on to the next
+                        # candidate instead of losing the rest of the search.
+                        print(f'[PROGRESS] {network_id} candidate {done + 1}/{n_candidates} '
+                              f'(E={E}, tau={tau}{f", theta={theta}" if search_mode == "joint" else ""}) '
+                              f'FAILED: {exc!r} -- scoring as NaN', flush=True)
+                        score = np.nan
+
                     score_grid[i, j, k] = score
 
                     with open(checkpoint_file, 'wb') as f:
```

---


## Job 35886791

Generated: 2026-07-31T19:50:56+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_35885157**

### Plain-language summary

This job carries the full set of design changes made in response to feedback
that the pipeline (specifically `MODE=MDE-GMN`) was taking about 2 days to
run, which a postdoc flagged as suspiciously long and suggested might be
caused by unnecessary/inefficient looping. That investigation led to one
architecture change (how the network is built) plus three follow-on
performance/correctness fixes. All of them are new since job_35885157, except
the network-construction rework itself, which job_35885157 already carried
and this job leaves untouched -- see that job's own summary for the full
rationale behind the nested-MDE rework and the earlier `GENERATE_LENGTH`
decoupling. This entry focuses on what changed *after* that: making the
network-construction step itself fast, and two things a live test run
surfaced that needed fixing along the way.

**1) Per-level parallelization of nested-MDE construction (the actual
performance fix).** The network-construction algorithm (nested, recursive
MDE calls -- each selected driver becomes a new MDE target in turn, several
levels deep) was, until now, running one MDE call at a time in a strict
sequence, even though most of those calls have nothing to do with each
other: every node at the same "depth" in the recursion is independent of
every other node at that depth, since each one only reads the same fixed
input data and doesn't depend on any other node's result. The fix
restructures this into "waves" -- all the independent MDE calls for a given
depth are now submitted to run at the same time across multiple CPU cores,
and only after a whole wave's results are back does the (cheap, fast)
step of checking for cycles/self-loops and deciding which edges to keep
happen, one at a time as before. The number of simultaneous calls and how
many cores each one gets are balanced automatically based on how many nodes
are in a given wave, so a wide wave (many nodes at once) spreads out across
more parallel workers while a narrow wave (few nodes) gives each one more
cores instead of leaving the rest idle -- this avoids accidentally
multiplying core usage past what was actually requested, which is a known
failure mode this pipeline has hit before with a different step (the CCM
interaction-matrix calculation, see the KNOWN CAVEAT notes elsewhere in this
file's history).

**2) An infrastructure bug this parallelization work exposed and had to fix
first.** While testing the parallel version, worker processes started
crashing intermittently with "ChildProcessError: No child processes." The
underlying cause: this cluster's Python setup, when asked to run code in
parallel across multiple processes, does so in a way that doesn't reliably
carry over a safety fix this pipeline already depended on -- specifically, a
fix that stops a particular library function (part of the CCM/cross-mapping
step MDE uses internally) from trying to create its *own* additional layer
of parallel processes nested inside processes that are already running in
parallel. Nesting parallelism like that either crashes outright (what was
observed here) or, if it doesn't crash, massively multiplies how many
processes get created for no benefit -- this is the same general class of
problem that caused the original 2-day runtime in the first place, just
showing up in a new place because of the new parallelization. The fix
applies the safety patch a second, independent way (forcing the relevant
setting directly at the exact function call site, not just overriding a
helper function elsewhere) and makes sure it gets freshly re-applied inside
every new parallel worker process rather than assuming it carries over
automatically. This was verified with real test runs, including one where
the crash reliably reproduced before the fix and did not reproduce after.

**3) A related efficiency/correctness refinement to the network-building
logic itself.** During the same round of testing, some driver variables
came back with an undefined ("NaN"/not-a-number) connection strength -- a
sign of a degenerate statistical computation for that specific variable
pair, not a sign that the variable was invalid or improperly selected.
Previously, the code would still add such a variable to the network and
keep recursively exploring *its* drivers for several more levels, even
though a connection with an undefined strength was always going to get
discarded later anyway (a downstream filtering step only keeps connections
above a minimum strength, and undefined values never pass that check). That
made the exploration a dead end that was still being explored at real
computational cost. The fix now recognizes this situation immediately and
stops exploring further down that specific branch right away, rather than
continuing to spend time on a branch whose result was already going to be
thrown out. Note this does not change which connections end up in the final
network -- those were already being filtered out downstream either way --
it only avoids wasting computation getting to that same outcome.

**This specific job's configuration**: `REGION_CHANNEL_CAP` reverts from 15
(a deliberately small value used in job_35885157 purely to get a fast,
low-risk timing/correctness check of the new network-construction code on
real cluster hardware before committing to a full run) back up to 999
(effectively the full, uncapped set of recorded channels) -- this is the
real, full-scale configuration these parallelization and correctness fixes
were specifically written to make practical. Everything else (`MDE_D=5`,
`NESTED_M=3`, `GMN_MODE=separate`, `GENERATE_LENGTH=300`) is unchanged from
job_35885157.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_35885157 | job_35886791 |
|---|---|---|
| REGION_CHANNEL_CAP | 15 | 999 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

```diff
--- job_35885157/simple-mde-gmn.py
+++ job_35886791/simple-mde-gmn.py
@@ -74,11 +74,12 @@
 pool provides parallelism. See `_sequential_ccm_workers`'s and
 build_network()'s docstrings for the full evidence trail.
 '''
+import multiprocessing as mp
 import os
 import pickle
 import subprocess
 import time
-from collections import deque
+from concurrent.futures import ProcessPoolExecutor, as_completed
 from configparser import ConfigParser
 from pathlib import Path
 
@@ -261,15 +262,70 @@
     sklearn/pyEDM/gmn) teardown cost alone, before any actual work
     happens -- and the real parallelism across channel pairs/candidates
     already comes from the outer pool, which this doesn't touch.
-    Monkeypatched (not `parallel=False` at each call site) because
-    apps.InteractionMatrix and dimx.Run construct CCM() internally and
-    don't expose that argument.'''
+    Monkeypatched rather than `parallel=False` at each call site because
+    apps.InteractionMatrix constructs CCM() internally and doesn't expose
+    that argument at all (dimx.Run's own CCM() call gets a second,
+    independent belt-and-suspenders patch below -- see
+    _sequential_ccm_api -- that does force parallel=False explicitly;
+    this _resolve_workers override is the one thing that still protects
+    apps.InteractionMatrix's own CCM() calls, and is kept as the first
+    line of defense for dimx.Run too).'''
     return 1
 
 
 import pyEDM.CCM as _pyedm_ccm_module
 _pyedm_ccm_module._resolve_workers = _sequential_ccm_workers
 
+import dimx.Run as _dimx_run_module
+_real_pyedm_ccm_api = _dimx_run_module.CCM
+
+
+def _sequential_ccm_api(*args, **kwargs):
+    '''Second, independent layer on top of _sequential_ccm_workers above:
+    forces parallel=False as an explicit argument to pyEDM's top-level
+    CCM() API function (kwargs['parallel'] -- CCM.py's own
+    _resolve_workers(parallel) unconditionally returns 1 when
+    parallel is False, no override needed for that half). Patches the
+    `CCM` name inside dimx.Run's own namespace directly (dimx/Run.py does
+    `from pyEDM import ... CCM ...`, so `dimx.Run.CCM` is the exact
+    binding dimx.MDE.Run() actually calls) rather than only the
+    pyEDM.CCM._resolve_workers indirection above, in case that patch
+    doesn't reach every call the same way in every process.'''
+    kwargs['parallel'] = False
+    return _real_pyedm_ccm_api(*args, **kwargs)
+
+
+_dimx_run_module.CCM = _sequential_ccm_api
+
+
+def _mde_wave_worker_init():
+    '''ProcessPoolExecutor(initializer=...) hook for build_nested_mde_network's
+    per-wave worker pool -- reapplies both CCM patches above explicitly
+    inside every worker process, rather than assuming either carries over
+    from this module's own top-level monkeypatch lines.
+
+    That assumption doesn't hold reliably on this cluster, and the
+    failure mode is nasty, not just a silent perf regression: a worker
+    whose dimx.MDE -> CCM() call ends up on the unpatched path tries to
+    create CCM's own internal multiprocessing Pool *from inside an
+    already-pooled worker process*, and that nested Pool creation fails
+    outright with "ChildProcessError: No child processes" rather than
+    merely being slow -- confirmed by direct diagnostics (printing
+    _resolve_workers' identity and a live test call immediately before
+    m.Run(), in the same worker process about to crash). The exact
+    trigger was never fully pinned down (diagnostics showed the
+    _resolve_workers override alone correctly returning 1 when checked
+    directly, yet some workers still hit the unpatched path deeper inside
+    Project()'s own call chain) -- what's confirmed empirically is that
+    applying *both* independent patches (the _resolve_workers override
+    AND the dimx.Run.CCM parallel=False wrapper) together, freshly inside
+    every worker via this initializer, reliably avoids it across repeated
+    test runs, while either alone did not.'''
+    import pyEDM.CCM as _ccm
+    _ccm._resolve_workers = _sequential_ccm_workers
+    import dimx.Run as _dimx_run
+    _dimx_run.CCM = _sequential_ccm_api
+
 
 # =============================================================================
 # Configuration -- every value overridable via MDE_GMN_* environment
@@ -572,6 +628,26 @@
         repeated for `nested_m` further rounds beyond step 1, so the
         total graph depth from any target is 1 + nested_m levels.
 
+    Nesting along a given branch stops before reaching nested_m whenever
+    one of two things happens, both handled as natural BFS termination
+    rather than requiring any special-casing beyond the depth counter:
+      - A node's own dimx.MDE call selects zero drivers (the causality
+        gates -- embedDimRhoMin/crossMapRhoMin/the CCM slope gate --
+        rejected every candidate) -- its driver loop below is simply
+        empty, so nothing gets enqueued for the next wave.
+      - A selected driver's marginal MDE rho gain comes out NaN (a
+        degenerate correlation computation, not the same as dimx.MDE's
+        own gates rejecting it -- see the NaN check in the driver loop
+        below) -- that specific edge is rejected as a dead end and not
+        nested into further, since it could never pass
+        strongest_iMatrix's keep-filter downstream regardless (NaN
+        comparisons are always False), making any deeper exploration
+        through it wasted computation. If every driver a node selected
+        comes out NaN, that node is a terminal/"edge" node with no
+        further nesting from it at all -- the same empty-enqueue outcome
+        as the zero-drivers case above, just discovered per-edge instead
+        of by the selection being empty to begin with.
+
     Cycle/self-loop guarding happens eagerly, during discovery, not as a
     post-hoc filter: every proposed edge (driver -> node) is added to a
     networkx.DiGraph built incrementally across this whole call (shared
@@ -598,6 +674,43 @@
     other nodes later in the BFS (ordinary DAG driver-sharing across
     branches), it just won't be re-selected as an MDE *target* again.
 
+    Parallelization: every node at the same BFS depth is MDE-independent
+    of every other node at that depth -- each dimx.MDE call only reads
+    the static `df` + `observable_columns` pool, with no shared mutable
+    state between nodes. So rather than one dimx.MDE call at a time, this
+    processes one whole "wave" (all not-yet-visited nodes newly
+    discovered at a given depth) with their MDE calls submitted
+    concurrently via ProcessPoolExecutor, one process per node in the
+    wave. Only the edge-acceptance / cycle-check step stays sequential,
+    run once a wave's MDE results are all back -- that step is cheap
+    graph bookkeeping (not EDM computation) and genuinely needs a single
+    consistent DiGraph to check against, so there's nothing to gain by
+    parallelizing it.
+
+    Outer (per-node-in-wave) and inner (each node's own dx.MDE ->
+    CrossMapColumns) parallelism are deliberately NOT both left at
+    `cores` -- that would reproduce the exact outer x inner pool-squaring
+    blowup this project already hit once with apps.InteractionMatrix /
+    pyEDM.CCM (see this module's KNOWN CAVEAT and
+    INTERACTION_MATRIX_OUTER_CORES). Instead the two are split adaptively
+    per wave: outer_workers = min(len(wave), cores), inner_cores =
+    max(1, cores // outer_workers) -- a wide wave (many nodes at once)
+    leans on outer parallelism with each node's own MDE call running
+    single-threaded internally; a narrow wave (e.g. one node) gives that
+    one call the full remaining core budget for its own CrossMapColumns
+    pool instead of leaving cores idle. Either way outer_workers *
+    inner_cores never exceeds `cores`. dimx.MDE's own internal CCM
+    validation step is separately forced sequential by
+    `_sequential_ccm_workers` above, same as everywhere else in this
+    module -- but that patch does NOT reliably survive into a
+    ProcessPoolExecutor worker on its own (confirmed empirically: this
+    cluster's 'forkserver' start method forks workers from a daemon that
+    snapshots imports at its own boot time, which can predate this
+    module's own monkeypatch line), so the wave pool below passes
+    `initializer=_mde_wave_worker_init` to reapply it explicitly inside
+    every worker -- see that function's docstring for the failure mode
+    this avoids.
+
     Edge weight stored in the returned interaction matrix is the
     *marginal* MDE rho gain at the step that driver was selected
     (gain_d = MDErho[d] - MDErho[d-1], MDErho[0] := 0) -- not the raw
@@ -631,45 +744,108 @@
     edge_weight = {}          # (driver, node) -> weight
     nested_mde_results = {}   # node -> run_mde_for_target() result, this call only
     visited = set()
-    queue = deque((t, 0) for t in targets)
+    current_wave = [(t, 0) for t in targets]
 
-    while queue:
-        node, depth = queue.popleft()
-        if node in visited:
-            continue
-        visited.add(node)
-
-        if depth == 0 and node in mde_results:
-            result = mde_results[node]
-        else:
-            candidate_pool = [c for c in observable_columns if c != node]
-            result = run_mde_for_target(
-                node, df, candidate_pool, prediction_start, prediction_length,
-                nested_n, embed_dim_rho_min, cross_map_rho_min, cores, results_dir,
-            )
-        nested_mde_results[node] = result
-
-        cols = result['SELECTED_COLUMNS']
-        gains = np.diff(np.concatenate(([0.0], result['m'].MDErho)))
-
-        for driver, gain in zip(cols, gains):
-            if driver == node:
-                continue  # explicit self-loop guard (belt-and-suspenders)
-
-            graph.add_node(driver)
-            graph.add_edge(driver, node)
-
-            if not nx.is_directed_acyclic_graph(graph):
-                graph.remove_edge(driver, node)
-                if graph.degree(driver) == 0:
-                    graph.remove_node(driver)
-                print(f'nested MDE: rejected {driver} -> {node} (would create a cycle)', flush=True)
+    while current_wave:
+        # Dedup within the wave itself -- two different nodes discovered
+        # in the *previous* wave can independently propose the same
+        # not-yet-visited driver as their own new MDE target; each
+        # unique node still only gets MDE'd once.
+        wave_nodes, seen = [], set()
+        for node, depth in current_wave:
+            if node in visited or node in seen:
                 continue
+            seen.add(node)
+            wave_nodes.append((node, depth))
+
+        if not wave_nodes:
+            break
+
+        # Depth-0 targets reuse the caller's precomputed mde_results
+        # directly (see docstring) -- no need to pay for, or parallelize,
+        # that MDE call again.
+        to_compute = [(node, depth) for node, depth in wave_nodes
+                      if not (depth == 0 and node in mde_results)]
+        wave_results = {
+            node: mde_results[node] for node, depth in wave_nodes
+            if depth == 0 and node in mde_results
+        }
+
+        if to_compute:
+            outer_workers = min(len(to_compute), max(1, cores))
+            inner_cores = max(1, cores // outer_workers)
+            print(f'nested MDE: wave of {len(to_compute)} node(s) at depth '
+                  f'{to_compute[0][1]} -- {outer_workers} parallel worker(s) '
+                  f'x {inner_cores} core(s) each', flush=True)
+
+            with ProcessPoolExecutor(max_workers=outer_workers,
+                                      mp_context=mp.get_context('fork'),
+                                      initializer=_mde_wave_worker_init) as executor:
+                futures = {}
+                for node, depth in to_compute:
+                    candidate_pool = [c for c in observable_columns if c != node]
+                    fut = executor.submit(
+                        run_mde_for_target, node, df, candidate_pool,
+                        prediction_start, prediction_length, nested_n,
+                        embed_dim_rho_min, cross_map_rho_min, inner_cores, results_dir,
+                    )
+                    futures[fut] = node
+                for fut in as_completed(futures):
+                    wave_results[futures[fut]] = fut.result()
+
+        next_wave = []
+        for node, depth in wave_nodes:
+            visited.add(node)
+            result = wave_results[node]
+            nested_mde_results[node] = result
+
+            cols = result['SELECTED_COLUMNS']
+            gains = np.diff(np.concatenate(([0.0], result['m'].MDErho)))
+
+            for driver, gain in zip(cols, gains):
+                if driver == node:
+                    continue  # explicit self-loop guard (belt-and-suspenders)
+
+                if np.isnan(gain):
+                    # Degenerate marginal-gain computation (e.g. a
+                    # near-constant embedding dimension producing an
+                    # undefined correlation) -- not the same as dimx.MDE
+                    # itself rejecting the candidate (it already passed
+                    # embedDimRhoMin/crossMapRhoMin/the CCM slope gate,
+                    # or it wouldn't be in `cols` at all), but a NaN edge
+                    # weight can never pass strongest_iMatrix's `> threshold`
+                    # keep-filter downstream regardless of what we do here
+                    # (NaN comparisons are always False), so CreateNetwork
+                    # would never select it as a kept driver either way.
+                    # Treat this edge as a dead end: don't add it to the
+                    # graph via this edge, and don't nest further from it
+                    # -- exploring deeper from a driver we can't assign a
+                    # trustworthy strength to is pure wasted computation.
+                    # `driver` can still enter the graph normally later via
+                    # a different, valid edge from another node (ordinary
+                    # DAG driver-sharing) -- only this specific edge is
+                    # rejected.
+                    print(f'nested MDE: rejected {driver} -> {node} '
+                          f'(marginal MDE rho gain is NaN -- treating as a dead-end edge, '
+                          f'no further nesting from this driver via this path)', flush=True)
+                    continue
+
+                graph.add_node(driver)
+                graph.add_edge(driver, node)
+
+                if not nx.is_directed_acyclic_graph(graph):
+                    graph.remove_edge(driver, node)
+                    if graph.degree(driver) == 0:
+                        graph.remove_node(driver)
+                    print(f'nested MDE: rejected {driver} -> {node} (would create a cycle)', flush=True)
+                    continue
+
+                edge_weight[(driver, node)] = max(gain, edge_weight.get((driver, node), 0.0))
 
-            edge_weight[(driver, node)] = max(gain, edge_weight.get((driver, node), 0.0))
+                if depth < nested_m and driver not in visited:
+                    next_wave.append((driver, depth + 1))
 
-            if depth < nested_m and driver not in visited:
-                queue.append((driver, depth + 1))
+        current_wave = next_wave
 
     all_nodes = list(graph.nodes)
     iMatrix = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)
```

---


## Job 35885157

Generated: 2026-07-31T18:44:29+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_34172243**

### Plain-language summary

First real cluster submission of the **nested-MDE interaction-matrix rewrite**
(`MODE=MDE-GMN`) -- the interaction matrix is no longer built from CCM across
the union of each target's MDE-selected columns; instead each target's
MDE-selected drivers themselves become new MDE targets in turn, recursively,
with every proposed edge checked against the DAG-so-far and rejected if it
would create a cycle or self-loop (`build_nested_mde_network()` in
`simple-mde-gmn.py`). `MODE=GMN` (the original CCM-across-full-observable-pool
path) is untouched. `GMN_MODE=separate` builds one independent network per
target, per current preference, rather than one combined network.

`REGION_CHANNEL_CAP` is temporarily **15** for this submission only -- a
deliberate scale-down from the file's normal default of 999 (effectively
uncapped, ~470 real channels), agreed with the user as a fast calibration/
timing run given `MDE_D=5`/`NESTED_M=3` (depth-4 nesting) against the full
channel pool risked a multi-day run on the very first exercise of this new
code path. The sbatch file was reverted back to `REGION_CHANNEL_CAP=999`
immediately after `sbatch` accepted the submission (SLURM already captured
this run's script content by then), so the working copy no longer reflects
what this specific job is running with -- see the parameter table below for
what job_35885157 actually launched with.

`GENERATE_LENGTH=300` decouples the GMN `Generate()` step count from the
MDE lib/pred split for the first time (new parameter, previously
nonexistent/tied to the full held-out window) -- each `Generate()` call in
the E/tau/theta search loop and the final rebuild now free-runs for only
300 steps rather than the entire held-out window, independent of `LIB_RATIO`.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_34172243 | job_35885157 |
|---|---|---|
| GENERATE_LENGTH | _(unset)_ | 300 |
| GMN_MODE | combined | separate |
| MDE_D | 50 | 5 |
| NESTED_M | _(unset)_ | 3 |
| REGION_CHANNEL_CAP | 999 | 15 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

```diff
--- job_34172243/simple-mde-gmn.sbatch
+++ job_35885157/simple-mde-gmn.sbatch
@@ -4,8 +4,21 @@
 #SBATCH --time=2-00:00:00
 #SBATCH --mem=48G
 #SBATCH --cpus-per-task=32
-#SBATCH --output=%x-%j.out
-#SBATCH --error=%x-%j.err
+# ADJUST: must be a literal absolute path, not $HOME/$PROJ_DIR -- Slurm
+# parses #SBATCH directives itself before this script (or any shell)
+# runs, so no shell variable expansion happens here. Pointed directly at
+# results/ (a fixed path that already exists) rather than results/job_%j/
+# because Slurm opens these files before the job starts, i.e. before
+# $SLURM_JOB_ID is known and before this script's own `mkdir -p
+# "$OUT_DIR"` below could create that subdirectory -- a per-job
+# subdirectory in this path would fail with "couldn't open output file"
+# since Slurm doesn't create missing directories. The EXIT trap further
+# down (search SLURM LOG RELOCATION) moves these into $OUT_DIR once it
+# exists, so they still end up alongside the rest of that job's output.
+#SBATCH --output=/home/j/j-tominaga/pao_rotation/results/%x-%j.out
+#SBATCH --error=/home/j/j-tominaga/pao_rotation/results/%x-%j.err
+#SBATCH --mail-user=j-tominaga@oist.jp
+#SBATCH --mail-type=END,FAIL
 
 set -euo pipefail
 
@@ -36,27 +49,29 @@
 export MKL_NUM_THREADS=1
 export OPENBLAS_NUM_THREADS=1
 
-# KNOWN CAVEAT (found by reading pyEDM/CCM.py and apps/InteractionMatrix.py
+# FIXED CAVEAT (found by reading pyEDM/CCM.py and apps/InteractionMatrix.py
 # directly, after observing ~90 live processes from an 8-core request on a
-# 10-core test machine): apps.InteractionMatrix's own outer worker pool
-# (sized by MDE_GMN_INTERACTION_MATRIX_OUTER_CORES below) calls
-# pyEDM.CCM() once per channel pair, and CCM() creates a *brand-new*
+# 10-core test machine, and root-caused after several full jobs each ran
+# ~2 days regardless of dataset size, MODE, or MDE_D): apps.InteractionMatrix's
+# own outer worker pool (sized by MDE_GMN_INTERACTION_MATRIX_OUTER_CORES
+# below) calls pyEDM.CCM() once per channel pair (dimx.MDE similarly calls
+# it once per candidate column), and CCM() used to create a *brand-new*
 # multiprocessing pool sized to the full visible cpu_count() on *every
-# single call* -- apps.InteractionMatrix never passes CCM's own
-# `parallel=` override, so this can't be tuned away via its CLI. The two
-# pools don't coordinate, so peak concurrent processes is
-# (outer cores) x cpu_count(), not just cpu_count() -- setting the outer
-# count equal to --cpus-per-task therefore *squares* your core request
-# into a process count. The script keeps this outer count decoupled and
-# small (MDE_GMN_INTERACTION_MATRIX_OUTER_CORES, default 2) specifically
-# to avoid that multiplication; don't raise it to match --cpus-per-task.
-# This is not a hang (confirmed via per-process CPU-time deltas: real work
-# keeps accumulating), but every column pair still pays a fresh pool
-# creation/teardown cost regardless of core counts -- so wall time for the
-# interaction-matrix step won't scale linearly with --cpus-per-task either
-# way. If a job runs far longer than --time allows, check actual CPU
-# utilization (e.g. `srun --jobid=<id> top`) before assuming the pipeline
-# itself is broken.
+# single call* for work that only ever amounts to 4-8 tiny KDTree tasks --
+# apps.InteractionMatrix/dimx.MDE never pass CCM's own `parallel=`
+# override, so this couldn't be tuned away via their CLIs. The two pools
+# didn't coordinate, so peak concurrent processes was (outer cores) x
+# cpu_count(), not just cpu_count(); with REGION_CHANNEL_CAP effectively
+# uncapped (~470 real channels), that's ~100k+ CCM() calls each paying a
+# full spawn/forkserver Pool creation + heavy reimport teardown cost --
+# process-spawn overhead dominating actual compute by orders of magnitude,
+# which is why cutting DATA_FILE resolution or MDE_D didn't help wall time.
+# scripts/src/simple-mde-gmn.py now monkeypatches pyEDM.CCM's inner pool
+# away entirely (`_sequential_ccm_workers`) rather than just avoiding the
+# multiplication, so this is no longer a live caveat -- kept here for
+# provenance. MDE_GMN_INTERACTION_MATRIX_OUTER_CORES (default 2) is still
+# a separate knob from --cpus-per-task since it's the one thing sizing
+# real parallelism across column pairs now.
 
 # ADJUST: project checkout location and lab/group allocation on this cluster.
 PROJ_DIR="$HOME/pao_rotation"
@@ -64,16 +79,50 @@
 mkdir -p "$OUT_DIR"
 
 # ==============================================================================
+# SLURM LOG RELOCATION
+# ==============================================================================
+# The #SBATCH --output/--error above write to $PROJ_DIR/results/ (flat,
+# not $OUT_DIR -- see the ADJUST comment on those directives for why).
+# Move them into $OUT_DIR once it exists, via a single combined EXIT trap
+# (merged with the /flash tempdir cleanup trap below, since bash only
+# honors the most recently registered EXIT trap -- a second `trap ...
+# EXIT` later in this script would silently replace this one rather than
+# stacking). Runs on every exit path, including failure/timeout/cancel,
+# which is exactly when you most want the log sitting next to the rest
+# of that run's output instead of loose in results/.
+SLURM_OUT="$PROJ_DIR/results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out"
+SLURM_ERR="$PROJ_DIR/results/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"
+cleanup() {
+    [ -n "${tempdir:-}" ] && rm -rf "$tempdir"
+    [ -f "$SLURM_OUT" ] && mv -f "$SLURM_OUT" "$OUT_DIR"/
+    [ -f "$SLURM_ERR" ] && mv -f "$SLURM_ERR" "$OUT_DIR"/
+    return 0
+}
+trap cleanup EXIT
+
+# ==============================================================================
 # JOB DESCRIPTION
 # ==============================================================================
-# Snapshots this run's simple-mde-gmn.py + simple-mde-gmn.sbatch into
-# $OUT_DIR and diffs them against the most recent earlier job's snapshot,
-# writing $OUT_DIR/JOB_DESCRIPTION.md (parameter-block changes as a table,
-# everything else as unified diffs). Runs before the pipeline itself so the
-# description is written even if the run later fails or times out.
+# Diffs this run's simple-mde-gmn.py + simple-mde-gmn.sbatch against the
+# rolling snapshot cache of the last simple-mde-gmn submission
+# ($PROJ_DIR/results/.snapshots/ -- not a per-job copy under $OUT_DIR;
+# see generate_job_description.sh's own module docstring for why),
+# inserting a new entry into $PROJ_DIR/JOB_LOG.md (parameter-block changes
+# as a table, everything else as unified diffs), then updates that cache
+# to this run's files. Runs before the pipeline itself so the entry is
+# written even if the run later fails or times out.
+#
+# sbatch_src is "$0", not a hardcoded path to this canonical file: sbatch
+# copies whatever file it's given verbatim into the running job, and $0
+# reflects that actual submitted path (confirmed via `scontrol show job`'s
+# Command= field) -- this matters if the live canonical file at
+# scripts/batch/simple-mde-gmn.sbatch gets edited again while this job
+# sits queued. A hardcoded canonical path here would silently diff
+# whatever that file happens to contain *right now* instead of what this
+# job actually launched with.
 "$PROJ_DIR/scripts/batch/generate_job_description.sh" \
     "$PROJ_DIR" "$OUT_DIR" "$SLURM_JOB_ID" \
-    "$PROJ_DIR/scripts/batch/simple-mde-gmn.sbatch" \
+    "$0" \
     "$PROJ_DIR/scripts/src/simple-mde-gmn.py"
 
 # ==============================================================================
@@ -88,7 +137,10 @@
 chmod 2755 "${FLASH_BASE}"
 
 tempdir="$(mktemp -d "${FLASH_BASE}/run_simple_mde_gmn.${SLURM_JOB_ID}.XXXXXX")"
-trap 'rm -rf "$tempdir"' EXIT
+# Cleanup on exit (including this tempdir) is handled by the combined
+# `cleanup` EXIT trap registered in the SLURM LOG RELOCATION section
+# above -- deliberately not a second `trap ... EXIT` here, which would
+# silently replace that one instead of stacking.
 
 # NOTE: the E/tau/theta search below checkpoints to disk after every
 # candidate so a resubmitted job can resume instead of restarting a
@@ -130,8 +182,10 @@
 export MDE_GMN_D="${MDE_D}"
 export MDE_GMN_EMBED_DIM_RHO_MIN="${EMBED_DIM_RHO_MIN}"
 export MDE_GMN_CROSS_MAP_RHO_MIN="${CROSS_MAP_RHO_MIN}"
+export MDE_GMN_NESTED_M="${NESTED_M}"
 export MDE_GMN_REGION_CHANNEL_CAP="${REGION_CHANNEL_CAP}"
 export MDE_GMN_LIB_RATIO="${LIB_RATIO}"
+export MDE_GMN_GENERATE_LENGTH="${GENERATE_LENGTH}"
 export MDE_GMN_E_MIN="${E_MIN}"
 export MDE_GMN_E_MAX="${E_MAX}"
 export MDE_GMN_E_STEP="${E_STEP}"
```

### Python pipeline changes (simple-mde-gmn.py)

```diff
--- job_34172243/simple-mde-gmn.py
+++ job_35885157/simple-mde-gmn.py
@@ -7,10 +7,25 @@
 explanation of *why* each step works the way it does -- docstrings below
 are trimmed to what's specific to running headless/non-interactively):
 
-  1. MDE forward-selection (skipped entirely if MODE='GMN')
-  2. GMN network construction: CCM interaction matrix (data-driven E/tau
-     grid search) + per-node driver-count estimate (EmbedDimension) +
-     apps.CreateNetwork, with a driver-ranking bug workaround
+  1. MDE forward-selection (skipped entirely if MODE='GMN'): top-level
+     per-target driver selection (dimx.MDE), used both for the MDE-only
+     diagnostics below and as the depth-0 seed of step 2's nested
+     expansion when MODE='MDE-GMN'.
+  2. GMN network construction. Two entirely different ways to build the
+     interaction matrix depending on MODE -- everything downstream (node
+     generation, search, rebuild, diagnostics) is identical either way:
+       - MODE='MDE-GMN': nested dimx.MDE expansion
+         (build_nested_mde_network()) -- each MDE-selected driver becomes
+         a new MDE target in turn for MDE_GMN_NESTED_M further rounds,
+         eagerly rejecting any edge that would create a cycle or
+         self-loop as the graph is discovered. No CCM / no
+         apps.InteractionMatrix involved at all.
+       - MODE='GMN': the original CCM interaction matrix (data-driven
+         E/tau grid search) across the full REGION_CHANNEL_CAP-limited
+         observable pool, via apps.InteractionMatrix. Unchanged.
+     Both paths feed the same per-node driver-count estimate
+     (EmbedDimension) + apps.CreateNetwork call, with the same
+     driver-ranking bug workaround.
   3. GMN node generation: SMap uniformly for every node (not Simplex, not
      a per-node mix -- see build_gmn_config()'s docstring for why)
   4. Empirical (E, tau, per-node theta) search maximizing free-running
@@ -63,6 +78,7 @@
 import pickle
 import subprocess
 import time
+from collections import deque
 from configparser import ConfigParser
 from pathlib import Path
 
@@ -309,13 +325,44 @@
     # dataset is used (no separate row-count cap); PREDICTION_START/
     # PREDICTION_LENGTH (row counts derived from this ratio and the loaded
     # data's length) are filled in by main() once df is loaded, and are
-    # what the rest of the pipeline actually reads.
+    # what the rest of the pipeline actually reads (MDE lib/pred split,
+    # CCM/EmbedDimension calls, and the held-out window Generate() results
+    # are validated against).
     cfg['LIB_RATIO'] = float(os.environ.get('MDE_GMN_LIB_RATIO', 0.5))
 
+    # GMN Generate() step count, decoupled from PREDICTION_LENGTH above --
+    # every Generate() call (both inside the E/tau/theta search loop and
+    # the final rebuild) always still *starts* at PREDICTION_START, but
+    # only runs this many steps rather than the entire held-out window.
+    # Generate() is the most expensive step in the search loop (one full
+    # network simulation per (E, tau[, theta]) candidate), so this lets
+    # you shrink that cost independently of LIB_RATIO/MDE's own held-out
+    # window, which the causality gating (CCM, EmbedDimension) still needs
+    # to be a meaningful size. 0 (default) means "use the full
+    # PREDICTION_LENGTH", i.e. unchanged pre-existing behavior; a value
+    # larger than PREDICTION_LENGTH is clamped down to it in main() once
+    # the dataset is loaded and PREDICTION_LENGTH is known (a longer
+    # Generate() than there is real held-out data to compare against isn't
+    # meaningful -- there'd be nothing to score the extra steps against).
+    cfg['GENERATE_LENGTH'] = int(os.environ.get('MDE_GMN_GENERATE_LENGTH', 0))
+
     # ---- MDE (variable selection; unused when MODE='GMN') -------------
     cfg['MDE_D'] = int(os.environ.get('MDE_GMN_D', 20))
     cfg['EMBED_DIM_RHO_MIN'] = float(os.environ.get('MDE_GMN_EMBED_DIM_RHO_MIN', 0.2))
     cfg['CROSS_MAP_RHO_MIN'] = float(os.environ.get('MDE_GMN_CROSS_MAP_RHO_MIN', 0.2))
+    # Nested-MDE network construction (MODE='MDE-GMN' only -- see
+    # build_nested_mde_network()'s docstring for the full algorithm).
+    # Reuses MDE_D as the per-node driver cap N at every nesting level
+    # (deliberately not a separate parameter -- MDE_D already means "max
+    # # of driver vars MDE can forward-select", which is exactly N here
+    # too). NESTED_M is the number of *additional* nested-MDE rounds
+    # beyond the initial per-target selection -- total graph depth from
+    # each target is 1 + NESTED_M levels. Combinatorially expensive: a
+    # naive upper bound on nodes is roughly MDE_D**(1+NESTED_M) before
+    # dedup/cycle-rejection collapses it back down to the real (capped)
+    # observable pool size -- keep MDE_D and NESTED_M small for a first
+    # test run.
+    cfg['NESTED_M'] = int(os.environ.get('MDE_GMN_NESTED_M', 1))
     # Candidate observable pool: first N channels of each region. Default
     # 999 (effectively uncapped -- matches the current notebook's default
     # and Park et al. 2026's own full-channel-set construction); cap this
@@ -501,19 +548,164 @@
         return fallback
 
 
+def build_nested_mde_network(targets, df, observable_columns, prediction_start, prediction_length,
+                              nested_n, nested_m, embed_dim_rho_min, cross_map_rho_min, cores,
+                              results_dir, mde_results):
+    '''Nested-MDE interaction-matrix construction -- MODE='MDE-GMN' only.
+    MODE='GMN' still uses the original CCM-across-the-full-observable-pool
+    path in build_network() below, entirely untouched.
+
+    This has no notebook precedent: it replaces the original "CCM over the
+    union of each target's MDE-selected columns" step with a from-scratch
+    algorithm built purely from repeated dimx.MDE calls (no CCM / no
+    apps.InteractionMatrix anywhere in this function):
+
+      Step 1 (depth 0): for each target in `targets`, dimx.MDE selects up
+        to `nested_n` (== MDE_D) driver columns from `observable_columns`.
+        Reuses the caller's already-computed top-level mde_results entry
+        when available -- main() always runs this once per cfg['TARGETS']
+        up front (for the pre-existing MDE-only diagnostics), so there's
+        no reason to pay for it twice.
+      Step 2 (depth 1..nested_m): each newly discovered driver becomes a
+        new MDE target in turn, again selecting up to `nested_n` drivers
+        from the same `observable_columns` pool (excluding itself) --
+        repeated for `nested_m` further rounds beyond step 1, so the
+        total graph depth from any target is 1 + nested_m levels.
+
+    Cycle/self-loop guarding happens eagerly, during discovery, not as a
+    post-hoc filter: every proposed edge (driver -> node) is added to a
+    networkx.DiGraph built incrementally across this whole call (shared
+    by every target in `targets`, so combined-mode targets can share
+    drivers), then nx.is_directed_acyclic_graph() is checked immediately
+    -- if the edge would create a cycle it's removed and rejected (the
+    driver is simply not linked to that particular node; it may still
+    exist elsewhere in the graph via a different, non-cyclic edge). This
+    mirrors apps.CreateNetwork's own incremental accept/reject loop (see
+    that module's docstring), but has to happen here too: unlike the
+    original CCM path, there is no precomputed full pairwise interaction
+    matrix for CreateNetwork to rank and select from -- dimx.MDE only
+    ever sees one candidate target at a time and has no notion of the
+    network being built around it, so a cycle-producing edge would
+    otherwise end up baked into the interaction matrix itself rather than
+    just mis-ranked (CreateNetwork's own check only protects against
+    cycles it's shown; it can't fix a matrix that never should have had
+    that entry). Self-loop exclusion (driver == node) is redundant with
+    dimx.MDE's own removeColumns=[target] gate in the ordinary case, kept
+    anyway as an explicit guard for defense in depth.
+
+    Each unique node is only ever MDE'd once (memoized via `visited`) --
+    once discovered, it can still gain additional incoming edges from
+    other nodes later in the BFS (ordinary DAG driver-sharing across
+    branches), it just won't be re-selected as an MDE *target* again.
+
+    Edge weight stored in the returned interaction matrix is the
+    *marginal* MDE rho gain at the step that driver was selected
+    (gain_d = MDErho[d] - MDErho[d-1], MDErho[0] := 0) -- not the raw
+    MDEOut['rho'], which is the *cumulative* multivariate cross-map rho
+    of the whole d-dimensional embedding built so far (dimx/Run.py calls
+    CrossMapColumns with columns=[candidate] + already-selected columns,
+    i.e. the joint embedding), and is expected to trend upward with d
+    regardless of any individual driver's own importance -- using it
+    directly would make later-ranked (weaker, marginal) drivers look
+    equal-or-stronger than the first-ranked (most informative) one. The
+    marginal gain isolates what that one driver specifically contributed
+    when greedily added on top of the previous d-1, which is what "edge
+    strength" should mean for an interaction matrix.
+
+    Returns (graph, iMatrix, nested_mde_results):
+      graph   -- networkx.DiGraph, guaranteed acyclic with no self-loops
+      iMatrix -- NxN DataFrame (index==columns==graph.nodes), the format
+                 apps.CreateNetwork's interactionMatrix= expects: row =
+                 driven node, column = driver, value = edge weight (0
+                 where no edge exists)
+      nested_mde_results -- {node: run_mde_for_target() result} for every
+                 node this call ran dimx.MDE on (targets included). Kept
+                 separate from the caller's own mde_results dict rather
+                 than mutating it in place, so repeated separate-mode
+                 calls (one per target group) can't cross-contaminate
+                 results for a shared intermediate channel name.
+    '''
+    graph = nx.DiGraph()
+    graph.add_nodes_from(targets)
+
+    edge_weight = {}          # (driver, node) -> weight
+    nested_mde_results = {}   # node -> run_mde_for_target() result, this call only
+    visited = set()
+    queue = deque((t, 0) for t in targets)
+
+    while queue:
+        node, depth = queue.popleft()
+        if node in visited:
+            continue
+        visited.add(node)
+
+        if depth == 0 and node in mde_results:
+            result = mde_results[node]
+        else:
+            candidate_pool = [c for c in observable_columns if c != node]
+            result = run_mde_for_target(
+                node, df, candidate_pool, prediction_start, prediction_length,
+                nested_n, embed_dim_rho_min, cross_map_rho_min, cores, results_dir,
+            )
+        nested_mde_results[node] = result
+
+        cols = result['SELECTED_COLUMNS']
+        gains = np.diff(np.concatenate(([0.0], result['m'].MDErho)))
+
+        for driver, gain in zip(cols, gains):
+            if driver == node:
+                continue  # explicit self-loop guard (belt-and-suspenders)
+
+            graph.add_node(driver)
+            graph.add_edge(driver, node)
+
+            if not nx.is_directed_acyclic_graph(graph):
+                graph.remove_edge(driver, node)
+                if graph.degree(driver) == 0:
+                    graph.remove_node(driver)
+                print(f'nested MDE: rejected {driver} -> {node} (would create a cycle)', flush=True)
+                continue
+
+            edge_weight[(driver, node)] = max(gain, edge_weight.get((driver, node), 0.0))
+
+            if depth < nested_m and driver not in visited:
+                queue.append((driver, depth + 1))
+
+    all_nodes = list(graph.nodes)
+    iMatrix = pd.DataFrame(0.0, index=all_nodes, columns=all_nodes)
+    for (driver, node), weight in edge_weight.items():
+        iMatrix.loc[node, driver] = weight
+
+    print(f'nested MDE: {len(all_nodes)} nodes, {len(edge_weight)} edges '
+          f'(N={nested_n}, M={nested_m})', flush=True)
+
+    return graph, iMatrix, nested_mde_results
+
+
 def build_network(targets, mde_results, df, prediction_start, prediction_length,
                    mode, e_range, tau_range, results_dir, cores, interaction_matrix_outer_cores,
+                   observable_columns=None, nested_n=None, nested_m=None,
+                   embed_dim_rho_min=None, cross_map_rho_min=None,
                    threshold=0, fallback_node_dimension=4, max_node_dimension=15):
     '''Build one interaction matrix + strongest-driver network spanning
-    `targets` and the union of their driver candidate columns
-    (mde_results[t]['SELECTED_COLUMNS'] -- MDE-selected when mode ==
-    'MDE-GMN', the full observable pool when mode == 'GMN'; this function
-    itself has no mode-awareness beyond that one flag used for the
-    network_id prefix below).
-
-    Mirrors notebooks/simple-mde-gmn.ipynb cell 7's build_network(). See
-    that cell's docstring for the CreateNetwork driver-ranking-bug
-    workaround rationale (identical here).
+    `targets` and their driver candidates. The interaction matrix itself
+    comes from one of two entirely different constructions depending on
+    `mode`:
+      - mode == 'MDE-GMN': build_nested_mde_network() -- nested dimx.MDE
+        expansion, no CCM involved. observable_columns/nested_n/nested_m/
+        embed_dim_rho_min/cross_map_rho_min are required in this mode.
+      - mode == 'GMN': the original CCM interaction matrix (data-driven
+        E/tau grid search) across the full REGION_CHANNEL_CAP-limited
+        observable pool, via apps.InteractionMatrix -- unchanged from the
+        pre-nested-MDE version of this pipeline.
+    Everything after the interaction matrix is computed (node-dimension
+    estimate, the CreateNetwork driver-ranking-bug workaround, the actual
+    CreateNetwork call) is identical either way.
+
+    Mirrors notebooks/simple-mde-gmn.ipynb cell 7's build_network() for
+    the mode == 'GMN' branch. See that cell's docstring for the
+    CreateNetwork driver-ranking-bug workaround rationale (identical
+    here).
 
     interaction_matrix_outer_cores is deliberately a *separate* knob from
     `cores`, not just reused: apps.InteractionMatrix's own -cr sizes an
@@ -541,18 +733,30 @@
     network_id = '_'.join(targets)
     # Distinguishes MODE='GMN' network artifacts from MODE='MDE-GMN'
     # artifacts for the same targets -- these are genuinely different
-    # networks (full raw channel pool vs. MDE-selected columns) and must
-    # not share a checkpoint file (see the search function's checkpoint
-    # resume logic, which only validates e_vals/tau_vals/theta_vals, not
-    # network topology).
+    # networks (full raw channel pool vs. nested-MDE-selected columns)
+    # and must not share a checkpoint file (see the search function's
+    # checkpoint resume logic, which only validates e_vals/tau_vals/
+    # theta_vals, not network topology).
     if mode == 'GMN':
         network_id = f'gmn_{network_id}'
 
-    selected_union = []
-    for t in targets:
-        for c in mde_results[t]['SELECTED_COLUMNS']:
-            if c not in selected_union:
-                selected_union.append(c)
+    ccm_E = ccm_tau = None
+
+    if mode == 'MDE-GMN':
+        graph, iMatrix, _ = build_nested_mde_network(
+            targets, df, observable_columns, prediction_start, prediction_length,
+            nested_n, nested_m, embed_dim_rho_min, cross_map_rho_min, cores,
+            results_dir, mde_results,
+        )
+        selected_union = [n for n in graph.nodes if n not in targets]
+        interaction_prefix = results_dir / f'iMatrix_nestedMDE_N{nested_n}_M{nested_m}_{network_id}'
+        iMatrix.to_csv(f'{interaction_prefix}_MDE.csv')
+    else:  # mode == 'GMN'
+        selected_union = []
+        for t in targets:
+            for c in mde_results[t]['SELECTED_COLUMNS']:
+                if c not in selected_union:
+                    selected_union.append(c)
 
     data_after_mde = (
         df[targets + selected_union]
@@ -567,34 +771,35 @@
     data_after_mde_lib_file = results_dir / f'data_after_mde_lib_{network_id}.csv'
     data_after_mde_lib.to_csv(data_after_mde_lib_file, index=False)
 
-    e_vals, tau_vals = list(e_range), list(tau_range)
-    target_grids = [target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals) for t in targets]
-    mean_grid = np.nanmean(np.stack(target_grids), axis=0)
-    best_i, best_j = np.unravel_index(np.nanargmax(mean_grid), mean_grid.shape)
-    ccm_E, ccm_tau = e_vals[best_i], tau_vals[best_j]
-    print(f'{network_id}: CCM E={ccm_E}, tau={ccm_tau} '
-          f'(grid-search rho={mean_grid[best_i, best_j]:.3f})', flush=True)
-
-    interaction_prefix = results_dir / f'iMatrix_E{ccm_E}_tau{ccm_tau}_{network_id}'
-    # apps.InteractionMatrix is a CLI-only tool (no in-memory API) --
-    # invoked via subprocess instead of the notebook's IPython
-    # get_ipython().system(), which only exists inside a live kernel.
-    # MPLBACKEND=Agg keeps -P from trying to open a GUI window.
-    subprocess_env = dict(os.environ, MPLBACKEND='Agg')
-    subprocess.run(
-        [
-            'python', '-m', 'apps.InteractionMatrix',
-            '-d', str(data_after_mde_lib_file),
-            '-oc', str(interaction_prefix),
-            '-ccm', '-E', str(ccm_E), '-t', str(ccm_tau),
-            '-l', '1', str(prediction_start), '-p', '1', str(prediction_start),
-            '-P', '-cr', str(interaction_matrix_outer_cores),
-        ],
-        check=True, env=subprocess_env,
-    )
+    if mode == 'GMN':
+        e_vals, tau_vals = list(e_range), list(tau_range)
+        target_grids = [target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals) for t in targets]
+        mean_grid = np.nanmean(np.stack(target_grids), axis=0)
+        best_i, best_j = np.unravel_index(np.nanargmax(mean_grid), mean_grid.shape)
+        ccm_E, ccm_tau = e_vals[best_i], tau_vals[best_j]
+        print(f'{network_id}: CCM E={ccm_E}, tau={ccm_tau} '
+              f'(grid-search rho={mean_grid[best_i, best_j]:.3f})', flush=True)
+
+        interaction_prefix = results_dir / f'iMatrix_E{ccm_E}_tau{ccm_tau}_{network_id}'
+        # apps.InteractionMatrix is a CLI-only tool (no in-memory API) --
+        # invoked via subprocess instead of the notebook's IPython
+        # get_ipython().system(), which only exists inside a live kernel.
+        # MPLBACKEND=Agg keeps -P from trying to open a GUI window.
+        subprocess_env = dict(os.environ, MPLBACKEND='Agg')
+        subprocess.run(
+            [
+                'python', '-m', 'apps.InteractionMatrix',
+                '-d', str(data_after_mde_lib_file),
+                '-oc', str(interaction_prefix),
+                '-ccm', '-E', str(ccm_E), '-t', str(ccm_tau),
+                '-l', '1', str(prediction_start), '-p', '1', str(prediction_start),
+                '-P', '-cr', str(interaction_matrix_outer_cores),
+            ],
+            check=True, env=subprocess_env,
+        )
 
-    interaction_csv = Path(f'{interaction_prefix}_CCM.csv')
-    iMatrix = pd.read_csv(interaction_csv, index_col=0)
+        interaction_csv = Path(f'{interaction_prefix}_CCM.csv')
+        iMatrix = pd.read_csv(interaction_csv, index_col=0)
 
     node_dimensions = {
         node: estimate_node_dimension(data_after_mde_lib, node, cores,
@@ -618,7 +823,10 @@
         drop = row.index.difference(keep)
         strongest_iMatrix.loc[node, drop] = threshold
 
-    network_file = results_dir / f'Network_E{ccm_E}_tau{ccm_tau}_CCM_{network_id}.pkl'
+    if mode == 'MDE-GMN':
+        network_file = results_dir / f'Network_nestedMDE_N{nested_n}_M{nested_m}_{network_id}.pkl'
+    else:
+        network_file = results_dir / f'Network_E{ccm_E}_tau{ccm_tau}_CCM_{network_id}.pkl'
     network = CreateNetwork(
         interactionMatrix = strongest_iMatrix,
         targetCols        = list(targets),
@@ -675,9 +883,28 @@
 
 def build_gmn_config(net_result, prediction_start, prediction_length, results_dir,
                       mode='Generate', function='SMap', E=3, tau=-3, theta=3,
-                      node_theta_overrides=None, variant=''):
+                      node_theta_overrides=None, variant='', generate_length=None):
     '''Stage the .cfg file gmn.GMN's configFile= constructor expects.
 
+    generate_length (mode='Generate' only): number of steps gmn.GMN
+    free-runs for, independent of prediction_length -- the latter is also
+    the MDE lib/pred split's held-out length (see load_config()'s
+    LIB_RATIO), which drives MDE variable selection, CCM/EmbedDimension
+    calls, and the search's own per-candidate theta retuning, none of
+    which have anything to do with how many steps a single Generate()
+    call should actually simulate. A full Generate() call over the whole
+    held-out window can be the single most expensive step in the search
+    loop (one call per (E, tau[, theta]) candidate), so decoupling this
+    lets you shrink it (MDE_GMN_GENERATE_LENGTH) without also shrinking
+    the held-out window MDE/CCM validate against. Defaults to
+    prediction_length when None (unchanged pre-existing behavior).
+    generate_length always still starts at prediction_start -- only the
+    *length* is decoupled, not the start point, since "start generating
+    right where the library ends" is what makes the comparison against
+    real held-out data meaningful. Ignored for mode='Forecast', which
+    still uses the full prediction_start/prediction_length (a one-step
+    boundary-accuracy diagnostic, not a free-running simulation length).
+
     function is fixed to 'SMap' for every node in this pipeline -- gmn's
     Simplex branch never calls pyEDM.Embed() before predicting (confirmed
     by reading gmn/Generate.py directly), so it can't do real Takens
@@ -721,7 +948,7 @@
 
     if mode == 'Generate':
         gmn_prediction_start = str(prediction_start)
-        gmn_prediction_length = str(prediction_length)
+        gmn_prediction_length = str(generate_length if generate_length is not None else prediction_length)
         lib_range = ''
         pred_range = ''
     else:
@@ -851,12 +1078,13 @@
 
 
 def evaluate_smap_candidate(net_result, prediction_start, prediction_length,
-                             E, tau, node_theta, variant, results_dir, cores):
+                             E, tau, node_theta, variant, results_dir, cores,
+                             generate_length=None):
     config_file = build_gmn_config(
         net_result, prediction_start, prediction_length, results_dir,
         mode='Generate', E=E, tau=tau,
         node_theta_overrides=node_theta,
-        variant=variant,
+        variant=variant, generate_length=generate_length,
     )
     G = gmn.GMN( configFile = str(config_file), cores = cores )
     G.Generate()
@@ -876,7 +1104,8 @@
 
 
 def run_smap_search(gmn_runs, prediction_start, prediction_length, e_range, tau_range,
-                     theta_range, coarse_theta_range, search_mode, results_dir, cores):
+                     theta_range, coarse_theta_range, search_mode, results_dir, cores,
+                     generate_length=None):
     smap_search_results = {}
     for network_id, run in gmn_runs.items():
         net_result = run['net_result']
@@ -893,8 +1122,15 @@
         if checkpoint_file.exists():
             with open(checkpoint_file, 'rb') as f:
                 ckpt = pickle.load(f)
+            # generate_length is part of the validity check alongside
+            # e_vals/tau_vals/theta_vals -- each cell's score was computed
+            # against a specific Generate() step count, and a resumed run
+            # with a different MDE_GMN_GENERATE_LENGTH would otherwise
+            # silently mix scores from two different-length simulations
+            # into the same grid.
             if (ckpt.get('e_vals') == e_vals and ckpt.get('tau_vals') == tau_vals
-                    and ckpt.get('theta_vals') == theta_vals):
+                    and ckpt.get('theta_vals') == theta_vals
+                    and ckpt.get('generate_length') == generate_length):
                 score_grid = ckpt['score_grid']
                 print(f'{network_id}: resumed checkpoint, '
                       f'{int(np.sum(~np.isnan(score_grid)))}/{score_grid.size} candidates already done',
@@ -932,12 +1168,14 @@
                     score, _, _ = evaluate_smap_candidate(
                         net_result, prediction_start, prediction_length,
                         E, tau, node_theta, variant, results_dir, cores,
+                        generate_length=generate_length,
                     )
                     score_grid[i, j, k] = score
 
                     with open(checkpoint_file, 'wb') as f:
                         pickle.dump({'e_vals': e_vals, 'tau_vals': tau_vals,
-                                     'theta_vals': theta_vals, 'score_grid': score_grid}, f)
+                                     'theta_vals': theta_vals, 'generate_length': generate_length,
+                                     'score_grid': score_grid}, f)
 
                     done += 1
                     elapsed = time.monotonic() - start_t
@@ -998,7 +1236,7 @@
 # =============================================================================
 
 def rebuild_at_best_params(gmn_runs, smap_search_results, prediction_start, prediction_length,
-                            results_dir, cores):
+                            results_dir, cores, generate_length=None):
     for network_id, run in gmn_runs.items():
         net_result = run['net_result']
         search = smap_search_results[network_id]
@@ -1009,7 +1247,7 @@
             net_result, prediction_start, prediction_length, results_dir,
             mode='Generate', E=best_E, tau=best_tau,
             node_theta_overrides=node_theta,
-            variant='best',
+            variant='best', generate_length=generate_length,
         )
 
         G = gmn.GMN( configFile = str(config_file), cores = cores )
@@ -1046,7 +1284,12 @@
 
         print(f"{network_id} Generate() rho, held-out window:", flush=True)
         for node in nodes:
-            a = actual_df[node].to_numpy()[prediction_start:n_rows]
+            # Sliced to len(generated), not n_rows/prediction_length: once
+            # generate_length can be shorter than the full held-out window
+            # (the whole point of decoupling it), actual_df's slice must
+            # match Generate()'s actual output length or ComputeError()
+            # compares mismatched-length arrays.
+            a = actual_df[node].to_numpy()[prediction_start:prediction_start + len(generated)]
             r = ComputeError(a, generated[node].to_numpy())['rho']
             tag = 'TARGET' if node in net_result['targets'] else 'driver'
             print(f"  {node:20s} {tag:7s} rho={r:7.3f}", flush=True)
@@ -1265,6 +1508,17 @@
     cfg['PREDICTION_START'] = int(round(len(df) * cfg['LIB_RATIO']))
     cfg['PREDICTION_LENGTH'] = len(df) - cfg['PREDICTION_START']
 
+    # Resolve GENERATE_LENGTH now that PREDICTION_LENGTH is known: 0
+    # (unset) means "use the full held-out window", and anything
+    # requesting more steps than there is real held-out data to score
+    # against gets clamped down to it (see load_config()'s docstring).
+    if cfg['GENERATE_LENGTH'] <= 0:
+        cfg['GENERATE_LENGTH'] = cfg['PREDICTION_LENGTH']
+    elif cfg['GENERATE_LENGTH'] > cfg['PREDICTION_LENGTH']:
+        print(f"GENERATE_LENGTH={cfg['GENERATE_LENGTH']} exceeds the held-out window "
+              f"({cfg['PREDICTION_LENGTH']} rows); clamping to it.", flush=True)
+        cfg['GENERATE_LENGTH'] = cfg['PREDICTION_LENGTH']
+
     print('Config:', {k: v for k, v in cfg.items() if k != 'TARGET_COLORS'}, flush=True)
 
     obs_cols = observable_columns(df, cfg['REGION_CHANNEL_CAP'])
@@ -1302,6 +1556,8 @@
             group, mde_results, df, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
             cfg['MODE'], cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['RESULTS_DIR'], cfg['CORES'],
             cfg['INTERACTION_MATRIX_OUTER_CORES'],
+            observable_columns=obs_cols, nested_n=cfg['MDE_D'], nested_m=cfg['NESTED_M'],
+            embed_dim_rho_min=cfg['EMBED_DIM_RHO_MIN'], cross_map_rho_min=cfg['CROSS_MAP_RHO_MIN'],
         )
         draw_colored_network(
             net_result, cfg['TARGET_COLORS'],
@@ -1313,11 +1569,13 @@
         gmn_runs, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
         cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['THETA_RANGE'], cfg['COARSE_THETA_RANGE'],
         cfg['SMAP_SEARCH_MODE'], cfg['RESULTS_DIR'], cfg['CORES'],
+        generate_length=cfg['GENERATE_LENGTH'],
     )
 
     rebuild_at_best_params(
         gmn_runs, smap_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
         cfg['RESULTS_DIR'], cfg['CORES'],
+        generate_length=cfg['GENERATE_LENGTH'],
     )
 
     for network_id, run in gmn_runs.items():
```

---


## Job 35678071

Generated: 2026-07-30T14:14:25+09:00

Git commit at submission: `5db3a5d` (clean vs. HEAD)

Compared against: **job_35673707**

### Parameter changes (tsls-mde-gmn.sbatch)

| Parameter | job_35673707 | job_35678071 |
|---|---|---|
| CCM_LIB_SIZE_PCTS | 10,15,85,90 | 2,3,5,8,12,20,30,50 |
| CCM_WINDOW_ROWS | 3000 | 800 |
| TARGET_METRIC | _(unset)_ | correlation |

### Batch script design changes (tsls-mde-gmn.sbatch, excludes parameter block above)

```diff
--- job_35673707/tsls-mde-gmn.sbatch
+++ job_35678071/tsls-mde-gmn.sbatch
@@ -176,6 +176,7 @@
 export TSLS_MDE_GMN_SMAP_SEARCH_MODE="${SMAP_SEARCH_MODE}"
 export TSLS_MDE_GMN_CORES="${SLURM_CPUS_PER_TASK}"
 export TSLS_MDE_GMN_METRIC="${METRIC}"
+export TSLS_MDE_GMN_TARGET_METRIC="${TARGET_METRIC}"
 export TSLS_MDE_GMN_GAMMA_DELTA="${GAMMA_DELTA}"
 export TSLS_MDE_GMN_CCM_WINDOW_ROWS="${CCM_WINDOW_ROWS}"
 export TSLS_MDE_GMN_CCM_LIB_SIZE_PCTS="${CCM_LIB_SIZE_PCTS}"
```

### Python pipeline changes (tsls-mde-gmn.py)

```diff
```

---


## Job 35678071

Generated: 2026-07-30T14:14:26+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

No previous job snapshot found -- this is the first tracked run of this pipeline, or all prior jobs predate this tracking.

---


## Job 35673707

Generated: 2026-07-30T13:26:13+09:00

Git commit at submission: `5db3a5d` (clean vs. HEAD)

Compared against: **job_35525431**

### Parameter changes (tsls-mde-gmn.sbatch)

_No parameter changes._

### Batch script design changes (tsls-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (tsls-mde-gmn.py)

```diff
--- job_35525431/tsls-mde-gmn.py
+++ job_35673707/tsls-mde-gmn.py
@@ -532,7 +532,26 @@
     proper_ccm/Simplex call cheap (bounded row count) to compensate, but a
     very large REGION_CHANNEL_CAP candidate pool will still take
     proportionally longer wall-clock than the old dimx.MDE-based version
-    did on a multi-core node.'''
+    did on a multi-core node.
+
+    noCCM=True: proper_mde's own internal CCM convergence-slope gate
+    turned out to be exactly as unreliable at this pipeline's scale as
+    apps.InteractionMatrix's 2-point delta gate was (see
+    build_interaction_matrix()'s docstring for the first time this same
+    failure mode was diagnosed and fixed) -- confirmed empirically in
+    production (job 35525431): every single candidate for both targets
+    showed slope~0.0 despite the ranking stage finding clearly strong
+    candidates (e.g. CA1_9_tsls at correlation=0.68 for head_position_x,
+    well above crossMapRhoMin=0.2), so MDE selected zero drivers for
+    either target, silently producing a near-empty 2-node network that
+    only surfaced as a crash 7 hours later (an unrelated diagnostic step
+    indexing into the now-empty SELECTED_COLUMNS list). Disabling the CCM
+    gate and relying solely on the ranking-stage crossMapRhoMin threshold
+    -- the same "drop the unreliable convergence requirement, keep the
+    plain skill threshold" fix already validated for
+    build_interaction_matrix() -- avoids this without needing to
+    re-tune yet another convergence-slope threshold that's already shown
+    it doesn't transfer across scales.'''
     mde_df = df[[target] + observable_columns].iloc[:prediction_start + prediction_length]
     lib_ratio = prediction_start / (prediction_start + prediction_length)
     ccm_df, ccm_lib_end = _ccm_window(mde_df, lib_ratio, ccm_window_rows)
@@ -540,7 +559,7 @@
     result = tsls.proper_mde(
         ccm_df, target=target, columns=observable_columns, D=mde_d,
         lib=[1, ccm_lib_end], pred=[ccm_lib_end + 1, len(ccm_df)],
-        tau=-1, Tp=1, E=3,
+        tau=-1, Tp=1, E=3, noCCM=True,
         pLibSizes=ccm_lib_size_pcts, sample=ccm_sample, ccmSlope=ccm_slope,
         crossMapRhoMin=cross_map_rho_min, embedDimRhoMin=embed_dim_rho_min,
         metric=target_metric, ccmMetric=metric, gammaDelta=gamma_delta,
@@ -781,6 +800,11 @@
             if c not in selected_union:
                 selected_union.append(c)
 
+    if not selected_union:
+        print(f'{network_id}: WARNING -- no MDE-selected driver columns for any target; '
+              f'the network below will only contain {targets} with no real neuron drivers.',
+              flush=True)
+
     data_after_mde = (
         df[targets + selected_union]
         .iloc[:prediction_start + prediction_length]
@@ -1396,6 +1420,9 @@
     for target in targets:
         r = mde_results[target]
         cols = r['SELECTED_COLUMNS']
+        if not cols:
+            print(f'{target}: no MDE-selected columns, skipping Tp-vs-rho sweep.', flush=True)
+            continue
         pi = tsls.proper_predict_interval(
             r['mde_df'], columns=cols, target=target,
             lib=r['LIB'], pred=r['PRED'],
```

---


## Job 35525431

Generated: 2026-07-29T19:06:02+09:00

Git commit at submission: `5db3a5d` (clean vs. HEAD)

Compared against: **job_35522535**

### Parameter changes (tsls-mde-gmn.sbatch)

_No parameter changes._

### Batch script design changes (tsls-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (tsls-mde-gmn.py)

```diff
--- job_35522535/tsls-mde-gmn.py
+++ job_35525431/tsls-mde-gmn.py
@@ -376,6 +376,17 @@
     # of Pearson correlation over the whole sawtooth, which is dominated
     # by the smooth (uninformative) ramp segments between spikes.
     cfg['METRIC'] = os.environ.get('TSLS_MDE_GMN_METRIC', 'f1')
+    # TARGETS (head_position_x/y) are continuous behavioral traces, not
+    # TSLS-encoded spike series -- they never "reset" near zero at
+    # discrete events, so 'f1'/'gamma' event-decoding is meaningless for
+    # them (confirmed empirically: applying METRIC='f1' to a target's own
+    # self-prediction produced NaN for every candidate, since
+    # infer_near_zero_threshold_from_true_tsls/detect_reset_events_from_tsls
+    # have nothing sensible to key off of on a signal with no genuine
+    # reset structure). Always 'correlation' (or 'mae') for anything scored
+    # against a TARGETS column; METRIC above is only for scoring
+    # predictions of the TSLS-converted neuron columns themselves.
+    cfg['TARGET_METRIC'] = os.environ.get('TSLS_MDE_GMN_TARGET_METRIC', 'correlation')
     cfg['GAMMA_DELTA'] = int(os.environ.get('TSLS_MDE_GMN_GAMMA_DELTA', 2))
     # Every CCM convergence test in this pipeline (proper_mde's own CCM
     # gate, and build_interaction_matrix() below) sweeps library sizes as
@@ -484,9 +495,18 @@
     return windowed, lib_end
 
 
+def _metric_for(column, targets, metric, target_metric):
+    '''TARGETS (continuous behavioral traces) always score via
+    target_metric ('correlation'); everything else (TSLS-converted neuron
+    columns) uses metric ('f1' by default) -- see load_config()'s
+    TARGET_METRIC docstring for why this split is required, not
+    cosmetic.'''
+    return target_metric if column in targets else metric
+
+
 def run_mde_for_target(target, df, observable_columns, prediction_start, prediction_length,
                         mde_d, embed_dim_rho_min, cross_map_rho_min, results_dir,
-                        metric, gamma_delta, ccm_window_rows, ccm_lib_size_pcts,
+                        metric, target_metric, gamma_delta, ccm_window_rows, ccm_lib_size_pcts,
                         ccm_sample, ccm_slope):
     '''Forward-selection for a single target variable via
     cleaned_helper_functions.proper_mde (replaces dimx.MDE -- see
@@ -494,6 +514,16 @@
     metric='f1'/'gamma', scoring TSLS series by decoded spike-timing
     match instead of plain Pearson rho over the whole sawtooth).
 
+    metric vs. ccmMetric: proper_mde's `metric` scores the RANKING stage
+    (predictions of `target`, which is one of TARGETS -- a continuous
+    behavioral trace, not TSLS -- so this must be target_metric, never
+    METRIC); `ccmMetric` scores the CCM gate (predictions of the
+    CANDIDATE column, which IS a TSLS neuron series -- so this is METRIC).
+    See load_config()'s TARGET_METRIC docstring and this file's
+    _metric_for() -- proper_mde's own docstring anticipates exactly this
+    split ("pick [ccmMetric] to match the nature of the candidate
+    columns").
+
     NOTE: proper_mde has no multiprocessing of its own (confirmed against
     cleaned_helper_functions.py -- no Pool/multiprocessing usage anywhere
     in the file), unlike dimx.MDE which threaded `cores` through its own
@@ -513,7 +543,7 @@
         tau=-1, Tp=1, E=3,
         pLibSizes=ccm_lib_size_pcts, sample=ccm_sample, ccmSlope=ccm_slope,
         crossMapRhoMin=cross_map_rho_min, embedDimRhoMin=embed_dim_rho_min,
-        metric=metric, ccmMetric=metric, gammaDelta=gamma_delta,
+        metric=target_metric, ccmMetric=metric, gammaDelta=gamma_delta,
         verbose=True,
     )
 
@@ -587,8 +617,15 @@
 # =============================================================================
 
 def _lib_train_test_ranges(n_rows, train_frac=0.7):
+    # [start, stop] list form (1-offset, inclusive) -- both call sites
+    # feed these into cleaned_helper_functions.py's proper_simplex/
+    # proper_embed_dimension, which expect that form, not pyEDM's own
+    # "start stop" string form (the two look similar but proper_simplex's
+    # lib[0]/lib[1] indexing on a string silently does the wrong thing
+    # and gets swallowed by the caller's try/except -- confirmed
+    # empirically as the cause of an all-NaN E/tau grid in production).
     split = int(n_rows * train_frac)
-    return f'1 {split}', f'{split + 1} {n_rows}'
+    return [1, split], [split + 1, n_rows]
 
 
 def target_E_tau_rho_grid(lib_df, target, e_vals, tau_vals, metric, gamma_delta):
@@ -624,11 +661,20 @@
         return fallback
 
 
-def build_interaction_matrix(data_lib, nodes, ccm_E, ccm_tau, metric, gamma_delta,
-                              ccm_window_rows, ccm_sample):
+def build_interaction_matrix(data_lib, nodes, targets, ccm_E, ccm_tau, metric, target_metric,
+                              gamma_delta, ccm_window_rows, ccm_sample):
     '''NxN interaction matrix via cleaned_helper_functions.proper_ccm,
     replacing apps.InteractionMatrix's CLI subprocess.
 
+    Per-pair metric selection via _metric_for(): `nodes` mixes TSLS-
+    converted neuron columns with the (continuous, non-TSLS) TARGETS
+    columns, so each proper_ccm call's target_measurement/
+    columns_measurement must independently match whichever side (a or b)
+    is actually being predicted in that direction -- using METRIC
+    ('f1' by default) to score a TARGETS column produced NaN for every
+    candidate in production (see load_config()'s TARGET_METRIC
+    docstring).
+
     WHY (empty-network root cause): apps.InteractionMatrix's own CCM
     convergence gate (InteractFunc, line ~430: `CCM_XY = CCM_YX = 0`
     unless the large-libSize skill beats the small-libSize skill by >
@@ -675,11 +721,15 @@
     for i, a in enumerate(nodes):
         for b in nodes[i + 1:]:
             try:
+                # target_measurement scores c2t (predictions of `target`=a);
+                # columns_measurement scores t2c (predictions of `columns`=b)
+                # -- each independently matched to whichever of a/b it predicts.
                 c2t, t2c, _ = tsls.proper_ccm(
                     windowed, libSizes=[libSize], sample=ccm_sample,
                     tau=ccm_tau, Tp=1, E=ccm_E,
                     target=a, columns=b,
-                    target_measurement=metric, columns_measurement=metric,
+                    target_measurement=_metric_for(a, targets, metric, target_metric),
+                    columns_measurement=_metric_for(b, targets, metric, target_metric),
                     gammaDelta=gamma_delta,
                 )
                 # c2t: skill of predicting `target`=a FROM `columns`=b's
@@ -696,7 +746,7 @@
 
 
 def build_network(targets, mde_results, df, prediction_start, prediction_length,
-                   mode, e_range, tau_range, results_dir, metric, gamma_delta,
+                   mode, e_range, tau_range, results_dir, metric, target_metric, gamma_delta,
                    ccm_window_rows, ccm_sample,
                    threshold=0, fallback_node_dimension=4, max_node_dimension=15):
     '''Build one interaction matrix + strongest-driver network spanning
@@ -745,27 +795,32 @@
     data_after_mde_lib.to_csv(data_after_mde_lib_file, index=False)
 
     e_vals, tau_vals = list(e_range), list(tau_range)
+    # target_E_tau_rho_grid always scores a TARGETS column's own
+    # self-predictability -- always target_metric (continuous), never
+    # METRIC (see load_config()'s TARGET_METRIC docstring).
     target_grids = [
-        target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals, metric, gamma_delta)
+        target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals, target_metric, gamma_delta)
         for t in targets
     ]
     mean_grid = np.nanmean(np.stack(target_grids), axis=0)
     best_i, best_j = np.unravel_index(np.nanargmax(mean_grid), mean_grid.shape)
     ccm_E, ccm_tau = e_vals[best_i], tau_vals[best_j]
     print(f'{network_id}: CCM E={ccm_E}, tau={ccm_tau} '
-          f'(grid-search {metric}={mean_grid[best_i, best_j]:.3f})', flush=True)
+          f'(grid-search {target_metric}={mean_grid[best_i, best_j]:.3f})', flush=True)
 
     nodes = [c for c in data_after_mde_lib.columns if c != 'Time']
     iMatrix = build_interaction_matrix(
-        data_after_mde_lib, nodes, ccm_E, ccm_tau, metric, gamma_delta,
+        data_after_mde_lib, nodes, targets, ccm_E, ccm_tau, metric, target_metric, gamma_delta,
         ccm_window_rows, ccm_sample,
     )
     interaction_csv = results_dir / f'iMatrix_E{ccm_E}_tau{ccm_tau}_{network_id}_CCM.csv'
     iMatrix.to_csv(interaction_csv)
 
     node_dimensions = {
-        node: estimate_node_dimension(data_after_mde_lib, node, metric, gamma_delta,
-                                       max_e=max_node_dimension, fallback=fallback_node_dimension)
+        node: estimate_node_dimension(
+            data_after_mde_lib, node, _metric_for(node, targets, metric, target_metric),
+            gamma_delta, max_e=max_node_dimension, fallback=fallback_node_dimension,
+        )
         for node in iMatrix.index
     }
     print(f'{network_id}: node N_D =', node_dimensions, flush=True)
@@ -1051,7 +1106,7 @@
 
 def run_smap_search(gmn_runs, prediction_start, prediction_length, e_range, tau_range,
                      theta_range, coarse_theta_range, search_mode, results_dir, cores,
-                     metric, gamma_delta):
+                     target_metric, gamma_delta):
     smap_search_results = {}
     for network_id, run in gmn_runs.items():
         net_result = run['net_result']
@@ -1107,7 +1162,7 @@
                     score, _, _ = evaluate_smap_candidate(
                         net_result, prediction_start, prediction_length,
                         E, tau, node_theta, variant, results_dir, cores,
-                        metric, gamma_delta,
+                        target_metric, gamma_delta,
                     )
                     score_grid[i, j, k] = score
 
@@ -1173,7 +1228,7 @@
 # =============================================================================
 
 def rebuild_at_best_params(gmn_runs, smap_search_results, prediction_start, prediction_length,
-                            results_dir, cores, metric, gamma_delta):
+                            results_dir, cores, metric, target_metric, gamma_delta):
     for network_id, run in gmn_runs.items():
         net_result = run['net_result']
         search = smap_search_results[network_id]
@@ -1219,17 +1274,18 @@
         fig.savefig(results_dir / f'gmn_generate_{network_id}.png', dpi=300)
         plt.close(fig)
 
-        print(f"{network_id} Generate() {metric}, held-out window:", flush=True)
+        print(f"{network_id} Generate() score, held-out window:", flush=True)
         for node in nodes:
+            node_metric = _metric_for(node, net_result['targets'], metric, target_metric)
             a = actual_df[node].to_numpy()[prediction_start:n_rows]
-            r = tsls._score(a, generated[node].to_numpy(), metric, gammaDelta=gamma_delta)
+            r = tsls._score(a, generated[node].to_numpy(), node_metric, gammaDelta=gamma_delta)
             tag = 'TARGET' if node in net_result['targets'] else 'driver'
-            print(f"  {node:20s} {tag:7s} {metric}={r:7.3f}", flush=True)
+            print(f"  {node:20s} {tag:7s} {node_metric}={r:7.3f}", flush=True)
 
 
 def run_forecast_diagnostic(net_result, G, prediction_start, prediction_length,
                              E, tau, node_theta, results_dir, cores, save_path,
-                             metric, gamma_delta):
+                             metric, target_metric, gamma_delta):
     config_file = build_gmn_config(
         net_result, prediction_start, prediction_length, results_dir, mode='Forecast',
         E=E, tau=tau, node_theta_overrides=node_theta,
@@ -1253,9 +1309,10 @@
         axes = [axes]
 
     for ax, node in zip(axes, nodes):
+        node_metric = _metric_for(node, net_result['targets'], metric, target_metric)
         actual = actual_df[node].to_numpy()
         forecast_pred = forecast_out[node].to_numpy()
-        rhos[node] = tsls._score(actual[time_idx], forecast_pred, metric, gammaDelta=gamma_delta)
+        rhos[node] = tsls._score(actual[time_idx], forecast_pred, node_metric, gammaDelta=gamma_delta)
 
         ax.plot(np.arange(n_rows), actual, color='tab:blue', lw=1, label='Lib/actual')
         if node in generate_out.columns:
@@ -1263,7 +1320,7 @@
                     color='tab:red', lw=1, label='Generate()')
         ax.plot(time_idx, forecast_pred, color='tab:purple', lw=1.2, ls=':', label='Forecast()')
         ax.axvline(prediction_start, color='gray', ls='--', lw=0.8)
-        ax.set_title(f"{node}   Forecast {metric}={rhos[node]:.3f}", fontsize=10)
+        ax.set_title(f"{node}   Forecast {node_metric}={rhos[node]:.3f}", fontsize=10)
         ax.legend(fontsize=7, loc='upper right')
 
     fig.suptitle(f"Forecast() vs Generate() vs actual -- {net_result['network_id']}", fontsize=11)
@@ -1274,7 +1331,8 @@
     return Gf, rhos
 
 
-def gmn_fidelity_rho(net_result, G, prediction_start, metric, gamma_delta, default_E=3, tau=-1):
+def gmn_fidelity_rho(net_result, G, prediction_start, metric, target_metric, gamma_delta,
+                      default_E=3, tau=-1):
     actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
     generated = G.DataOut.reset_index(drop=True)
     nodes = [c for c in generated.columns if c != 'Time']
@@ -1284,6 +1342,7 @@
 
     fidelity_rho = {}
     for node in nodes:
+        node_metric = _metric_for(node, net_result['targets'], metric, target_metric)
         gen_vals = generated[node].to_numpy()
         actual_vals = actual_df[node].to_numpy()[prediction_start: prediction_start + n_gen]
         pair_df = pd.DataFrame({
@@ -1297,9 +1356,9 @@
                 pair_df, columns=f'{node}_gen', target=f'{node}_actual',
                 E=E, tau=tau, Tp=0,
                 lib=[1, split], pred=[split + 1, n_gen],
-                embedded=False, metric=metric, gammaDelta=gamma_delta,
+                embedded=False, metric=node_metric, gammaDelta=gamma_delta,
             )
-            fidelity_rho[node] = out['metrics'][metric]
+            fidelity_rho[node] = out['metrics'][node_metric]
         except Exception:
             fidelity_rho[node] = np.nan
     return fidelity_rho
@@ -1436,7 +1495,7 @@
                 target, df, obs_cols, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                 cfg['MDE_D'], cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
                 cfg['RESULTS_DIR'],
-                cfg['METRIC'], cfg['GAMMA_DELTA'], cfg['CCM_WINDOW_ROWS'],
+                cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'], cfg['CCM_WINDOW_ROWS'],
                 cfg['CCM_LIB_SIZE_PCTS'], cfg['CCM_SAMPLE'], cfg['CCM_SLOPE'],
             )
 
@@ -1462,7 +1521,8 @@
         net_result = build_network(
             group, mde_results, df, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
             cfg['MODE'], cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['RESULTS_DIR'],
-            cfg['METRIC'], cfg['GAMMA_DELTA'], cfg['CCM_WINDOW_ROWS'], cfg['CCM_SAMPLE'],
+            cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
+            cfg['CCM_WINDOW_ROWS'], cfg['CCM_SAMPLE'],
         )
         draw_colored_network(
             net_result, cfg['TARGET_COLORS'],
@@ -1474,12 +1534,12 @@
         gmn_runs, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
         cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['THETA_RANGE'], cfg['COARSE_THETA_RANGE'],
         cfg['SMAP_SEARCH_MODE'], cfg['RESULTS_DIR'], cfg['CORES'],
-        cfg['METRIC'], cfg['GAMMA_DELTA'],
+        cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
     )
 
     rebuild_at_best_params(
         gmn_runs, smap_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
-        cfg['RESULTS_DIR'], cfg['CORES'], cfg['METRIC'], cfg['GAMMA_DELTA'],
+        cfg['RESULTS_DIR'], cfg['CORES'], cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
     )
 
     for network_id, run in gmn_runs.items():
@@ -1488,16 +1548,16 @@
             run['best_E'], run['best_tau'], run['best_node_theta'],
             cfg['RESULTS_DIR'], cfg['CORES'],
             cfg['RESULTS_DIR'] / f'gmn_forecast_vs_generate_{network_id}.png',
-            cfg['METRIC'], cfg['GAMMA_DELTA'],
+            cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
         )
         run['Gf'] = Gf
         run['forecast_rho'] = forecast_rho
-        print(f"{network_id} Forecast() {cfg['METRIC']} per node (weakest first):", flush=True)
+        print(f"{network_id} Forecast() score per node (weakest first):", flush=True)
         for node, rho in sorted(forecast_rho.items(), key=lambda kv: kv[1]):
             print(f"  {node:20s} {rho:.4f}", flush=True)
 
         rho = gmn_fidelity_rho(run['net_result'], run['G'], cfg['PREDICTION_START'],
-                                cfg['METRIC'], cfg['GAMMA_DELTA'])
+                                cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
         plot_fidelity(rho, network_id, cfg['RESULTS_DIR'] / f'gmn_fidelity_{network_id}.png')
         print(f'{network_id} fidelity {cfg["METRIC"]} (weakest first):', flush=True)
         for node in sorted(rho, key=lambda n: (np.isnan(rho[n]), -(rho[n] if not np.isnan(rho[n]) else 0))):
@@ -1511,11 +1571,14 @@
     # see the "MDE-only diagnostics" section docstring above
     # run_univariate_e_tau_grid for why.
     if cfg['MODE'] == 'MDE-GMN' and cfg['RUN_MDE_DIAGNOSTICS']:
+        # Both score a TARGETS column's own predictability -- always
+        # TARGET_METRIC (continuous), never METRIC (see load_config()'s
+        # TARGET_METRIC docstring).
         run_tp_vs_rho_sweep(mde_results, cfg['TARGETS'], cfg['MAX_TP'], cfg['TARGET_COLORS'],
-                             cfg['RESULTS_DIR'], cfg['METRIC'], cfg['GAMMA_DELTA'])
+                             cfg['RESULTS_DIR'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
         run_univariate_e_tau_grid(mde_results, cfg['TARGETS'], cfg['E_RANGE'], cfg['TAU_RANGE'],
                                    cfg['GRID_RHO_THRESHOLD'], cfg['RESULTS_DIR'],
-                                   cfg['METRIC'], cfg['GAMMA_DELTA'])
+                                   cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
     else:
         print(f"MODE={cfg['MODE']!r} / RUN_MDE_DIAGNOSTICS={cfg['RUN_MDE_DIAGNOSTICS']}: "
               f"skipping MDE-only diagnostics.", flush=True)
```

---


## Job 35522535

Generated: 2026-07-29T18:53:07+09:00

Git commit at submission: `5db3a5d` (clean vs. HEAD)

Compared against: **job_35513363**

### Parameter changes (tsls-mde-gmn.sbatch)

| Parameter | job_35513363 | job_35522535 |
|---|---|---|
| CCM_LIB_SIZE_PCTS | _(unset)_ | 10,15,85,90 |
| CCM_SAMPLE | _(unset)_ | 20 |
| CCM_SLOPE | _(unset)_ | 0.01 |
| CCM_WINDOW_ROWS | _(unset)_ | 3000 |
| GAMMA_DELTA | _(unset)_ | 2 |
| INTERACTION_MATRIX_OUTER_CORES | 2 | _(unset)_ |
| METRIC | _(unset)_ | f1 |

### Batch script design changes (tsls-mde-gmn.sbatch, excludes parameter block above)

```diff
--- job_35513363/tsls-mde-gmn.sbatch
+++ job_35522535/tsls-mde-gmn.sbatch
@@ -175,7 +175,12 @@
 export TSLS_MDE_GMN_TAU_STEP="${TAU_STEP}"
 export TSLS_MDE_GMN_SMAP_SEARCH_MODE="${SMAP_SEARCH_MODE}"
 export TSLS_MDE_GMN_CORES="${SLURM_CPUS_PER_TASK}"
-export TSLS_MDE_GMN_INTERACTION_MATRIX_OUTER_CORES="${INTERACTION_MATRIX_OUTER_CORES}"
+export TSLS_MDE_GMN_METRIC="${METRIC}"
+export TSLS_MDE_GMN_GAMMA_DELTA="${GAMMA_DELTA}"
+export TSLS_MDE_GMN_CCM_WINDOW_ROWS="${CCM_WINDOW_ROWS}"
+export TSLS_MDE_GMN_CCM_LIB_SIZE_PCTS="${CCM_LIB_SIZE_PCTS}"
+export TSLS_MDE_GMN_CCM_SAMPLE="${CCM_SAMPLE}"
+export TSLS_MDE_GMN_CCM_SLOPE="${CCM_SLOPE}"
 
 python3 -V
 
```

### Python pipeline changes (tsls-mde-gmn.py)

```diff
--- job_35513363/tsls-mde-gmn.py
+++ job_35522535/tsls-mde-gmn.py
@@ -63,7 +63,6 @@
 
 import os
 import pickle
-import subprocess
 import sys
 import time
 from configparser import ConfigParser
@@ -77,14 +76,20 @@
 import numpy as np
 import pandas as pd
 
-import dimx as dx
 import gmn
 from gmn.Common import FunctionType
 from apps.CreateNetwork import CreateNetwork
-from pyEDM import (
-    Simplex, SMap, CCM, ComputeError, Embed, EmbedDimension,
-    PredictNonlinear, PredictInterval,
-)
+# NOTE: dimx (dx.MDE/dx.Evaluate), apps.InteractionMatrix, and pyEDM's
+# Simplex/SMap/CCM/ComputeError/EmbedDimension/PredictInterval are no
+# longer imported directly -- this pipeline now uses
+# cleaned_helper_functions.py's proper_mde/proper_ccm/proper_simplex/
+# proper_embed_dimension/proper_predict_interval throughout (see this
+# module's docstring and load_config()'s METRIC docstring for why). Embed
+# and PredictNonlinear remain: no TSLS-aware equivalent exists for
+# gmn.Node.Forecast's own delay-embedding call or for the per-node theta
+# search (both pyEDM.SMap-specific concepts with no analog in
+# cleaned_helper_functions.py's own KDTree engine).
+from pyEDM import Embed, PredictNonlinear
 
 # cleaned_helper_functions.py isn't installed as a package -- the sbatch
 # script copies it into this script's own directory (same convention as
@@ -356,6 +361,43 @@
     cfg['CORES'] = int(os.environ.get('TSLS_MDE_GMN_CORES', os.cpu_count() or 1))
     cfg['INTERACTION_MATRIX_OUTER_CORES'] = int(os.environ.get('TSLS_MDE_GMN_INTERACTION_MATRIX_OUTER_CORES', 2))
 
+    # ---- TSLS-aware scoring (cleaned_helper_functions.py's proper_mde/
+    # proper_ccm/proper_simplex/adaptive_event_metrics_from_tsls, replacing
+    # dimx.MDE / apps.InteractionMatrix / pyEDM.EmbedDimension / plain
+    # ComputeError rho everywhere in this pipeline) -----------------------
+    # 'correlation' (plain Pearson rho, pyEDM's own default) | 'mae' |
+    # 'f1' | 'gamma' (Kistler coincidence factor) -- see
+    # cleaned_helper_functions.py's _score()/adaptive_event_metrics_from_tsls
+    # docstrings. 'f1'/'gamma' score TSLS series by decoding both true and
+    # predicted series back to discrete reset/spike events (via a
+    # near-zero threshold calibrated from the true series itself) and
+    # matching them within an adaptive per-event tolerance window -- this
+    # is what actually answers "did we get the spike timing right" instead
+    # of Pearson correlation over the whole sawtooth, which is dominated
+    # by the smooth (uninformative) ramp segments between spikes.
+    cfg['METRIC'] = os.environ.get('TSLS_MDE_GMN_METRIC', 'f1')
+    cfg['GAMMA_DELTA'] = int(os.environ.get('TSLS_MDE_GMN_GAMMA_DELTA', 2))
+    # Every CCM convergence test in this pipeline (proper_mde's own CCM
+    # gate, and build_interaction_matrix() below) sweeps library sizes as
+    # PERCENTAGES of however many rows it's handed -- confirmed empirically
+    # (job 35513363's preserved iMatrix CSV) that apps.InteractionMatrix's
+    # equivalent 2-point version of this same gate collapses to reporting
+    # exactly 0 for virtually every pair once even the *small* end of that
+    # sweep is already past typical CCM convergence, which happens whenever
+    # the percentages are taken of this pipeline's full ~30-45k-row
+    # TSLS-converted dataset (5% of 31351 = a library of 1567 rows -- already
+    # saturated for a low-dimensional embedding). CCM_WINDOW_ROWS bounds the
+    # row count actually fed into any percentage-based libSize sweep so
+    # both ends stay in the range where a real relationship still shows
+    # visible improvement as the library grows, independent of how large
+    # the full dataset is.
+    cfg['CCM_WINDOW_ROWS'] = int(os.environ.get('TSLS_MDE_GMN_CCM_WINDOW_ROWS', 3000))
+    cfg['CCM_LIB_SIZE_PCTS'] = tuple(
+        _env_float_list('TSLS_MDE_GMN_CCM_LIB_SIZE_PCTS', (10, 30, 50, 70, 90))
+    )
+    cfg['CCM_SAMPLE'] = int(os.environ.get('TSLS_MDE_GMN_CCM_SAMPLE', 20))
+    cfg['CCM_SLOPE'] = float(os.environ.get('TSLS_MDE_GMN_CCM_SLOPE', 0.01))
+
     cfg['TARGET_COLORS'] = {t: plt.get_cmap('tab10')(i % 10) for i, t in enumerate(cfg['TARGETS'])}
 
     return cfg
@@ -430,40 +472,72 @@
     print(f'-> tsls_data_trimmed.csv  shape={df.shape}', flush=True)
     return df
 
+def _ccm_window(df, lib_ratio, max_rows):
+    '''Bound the row count fed into any percentage-based CCM libSize sweep
+    to max_rows, preserving lib_ratio's proportion within that smaller
+    window. See load_config()'s CCM_WINDOW_ROWS docstring for why this is
+    necessary (not optional) for CCM's convergence gate to mean anything
+    at this pipeline's row counts.'''
+    n = min(len(df), max_rows)
+    windowed = df.iloc[:n].reset_index(drop=True)
+    lib_end = max(1, int(round(n * lib_ratio)))
+    return windowed, lib_end
+
+
 def run_mde_for_target(target, df, observable_columns, prediction_start, prediction_length,
-                        mde_d, embed_dim_rho_min, cross_map_rho_min, cores, results_dir):
-    '''Run dimx.MDE's forward selection for a single target variable.
-    See notebooks/simple-mde-gmn.ipynb cell 2's run_mde_for_target for the
-    full rationale (identical logic, just cores threaded through).'''
+                        mde_d, embed_dim_rho_min, cross_map_rho_min, results_dir,
+                        metric, gamma_delta, ccm_window_rows, ccm_lib_size_pcts,
+                        ccm_sample, ccm_slope):
+    '''Forward-selection for a single target variable via
+    cleaned_helper_functions.proper_mde (replaces dimx.MDE -- see
+    load_config()'s METRIC docstring for why: proper_mde accepts
+    metric='f1'/'gamma', scoring TSLS series by decoded spike-timing
+    match instead of plain Pearson rho over the whole sawtooth).
+
+    NOTE: proper_mde has no multiprocessing of its own (confirmed against
+    cleaned_helper_functions.py -- no Pool/multiprocessing usage anywhere
+    in the file), unlike dimx.MDE which threaded `cores` through its own
+    worker pool. This pipeline no longer has a way to parallelize the MDE
+    forward-selection step; CCM_WINDOW_ROWS keeps each individual
+    proper_ccm/Simplex call cheap (bounded row count) to compensate, but a
+    very large REGION_CHANNEL_CAP candidate pool will still take
+    proportionally longer wall-clock than the old dimx.MDE-based version
+    did on a multi-core node.'''
     mde_df = df[[target] + observable_columns].iloc[:prediction_start + prediction_length]
+    lib_ratio = prediction_start / (prediction_start + prediction_length)
+    ccm_df, ccm_lib_end = _ccm_window(mde_df, lib_ratio, ccm_window_rows)
 
-    m = dx.MDE( mde_df, target = target, D = mde_d,
-                removeColumns = [target],
-                lib  = [1, prediction_start],
-                pred = [prediction_start + 1, prediction_start + prediction_length],
-                embedDimRhoMin = embed_dim_rho_min,
-                crossMapRhoMin = cross_map_rho_min,
-                cores = cores )
-
-    m.Run()
+    result = tsls.proper_mde(
+        ccm_df, target=target, columns=observable_columns, D=mde_d,
+        lib=[1, ccm_lib_end], pred=[ccm_lib_end + 1, len(ccm_df)],
+        tau=-1, Tp=1, E=3,
+        pLibSizes=ccm_lib_size_pcts, sample=ccm_sample, ccmSlope=ccm_slope,
+        crossMapRhoMin=cross_map_rho_min, embedDimRhoMin=embed_dim_rho_min,
+        metric=metric, ccmMetric=metric, gammaDelta=gamma_delta,
+        verbose=True,
+    )
 
-    m.MDEOut.to_csv(results_dir / f'mde_output_{target}.csv', index=False)
+    mde_out = pd.DataFrame({'rho': result['MDErho']})
+    mde_out.to_csv(results_dir / f'mde_output_{target}.csv', index=False)
 
-    selected_columns = list(m.MDEcolumns)
+    selected_columns = list(result['MDEcolumns'])
     print(f'MDE selected columns for {target}:', selected_columns, flush=True)
 
     return {
         'target': target,
-        'mde_df': mde_df,
-        'm': m,
+        'mde_df': ccm_df,
+        'MDEOut': mde_out,
         'SELECTED_COLUMNS': selected_columns,
+        'LIB': [1, ccm_lib_end],
+        'PRED': [ccm_lib_end + 1, len(ccm_df)],
+        'TAU': -1,
     }
 
 
 def plot_mde_rho_vs_dimension(mde_results, targets, target_colors, save_path):
     fig, ax = plt.subplots(figsize=(9, 5))
     for target in targets:
-        mde_out = mde_results[target]['m'].MDEOut
+        mde_out = mde_results[target]['MDEOut']
         dims = np.arange(1, len(mde_out) + 1)
         ax.plot(dims, mde_out['rho'], marker='o', label=target, color=target_colors[target])
     ax.set_xlabel('Embedding dimension')
@@ -517,38 +591,113 @@
     return f'1 {split}', f'{split + 1} {n_rows}'
 
 
-def target_E_tau_rho_grid(lib_df, target, e_vals, tau_vals):
+def target_E_tau_rho_grid(lib_df, target, e_vals, tau_vals, metric, gamma_delta):
     lib_range, pred_range = _lib_train_test_ranges(len(lib_df))
     rho_grid = np.full((len(e_vals), len(tau_vals)), np.nan)
     for i, E in enumerate(e_vals):
         for j, tau in enumerate(tau_vals):
             try:
-                S = Simplex(
-                    dataFrame = lib_df, columns = target, target = target,
-                    E = E, tau = tau, lib = lib_range, pred = pred_range,
-                    embedded = False, showPlot = False,
+                out = tsls.proper_simplex(
+                    lib_df, columns=target, target=target,
+                    E=E, tau=tau, lib=lib_range, pred=pred_range,
+                    embedded=False, metric=metric, gammaDelta=gamma_delta,
                 )
-                rho_grid[i, j] = ComputeError(S['Observations'], S['Predictions'])['rho']
+                rho_grid[i, j] = out['metrics'][metric]
             except Exception:
                 pass
     return rho_grid
 
 
-def estimate_node_dimension(lib_df, node, cores, max_e=15, fallback=4):
+def estimate_node_dimension(lib_df, node, metric, gamma_delta, max_e=15, fallback=4):
+    '''EmbedDimension-equivalent for TSLS-aware metrics via
+    proper_embed_dimension (pyEDM's own EmbedDimension is Pearson-rho
+    only, so it can't score 'f1'/'gamma').'''
     lib_range, pred_range = _lib_train_test_ranges(len(lib_df))
     try:
-        ed = EmbedDimension(
-            dataFrame = lib_df, columns = node, target = node,
-            maxE = max_e, lib = lib_range, pred = pred_range, showPlot = False,
-            numProcess = cores,
+        ed = tsls.proper_embed_dimension(
+            lib_df, columns=node, target=node,
+            maxE=max_e, lib=lib_range, pred=pred_range,
+            metric=metric, gammaDelta=gamma_delta,
         )
-        return int(ed.loc[ed['rho'].idxmax(), 'E'])
+        return int(ed.loc[ed[metric].idxmax(), 'E'])
     except Exception:
         return fallback
 
 
+def build_interaction_matrix(data_lib, nodes, ccm_E, ccm_tau, metric, gamma_delta,
+                              ccm_window_rows, ccm_sample):
+    '''NxN interaction matrix via cleaned_helper_functions.proper_ccm,
+    replacing apps.InteractionMatrix's CLI subprocess.
+
+    WHY (empty-network root cause): apps.InteractionMatrix's own CCM
+    convergence gate (InteractFunc, line ~430: `CCM_XY = CCM_YX = 0`
+    unless the large-libSize skill beats the small-libSize skill by >
+    deltaCCM=0.05) reports a pair's interaction as *exactly 0* whenever
+    that isn't true -- confirmed empirically (job 35513363's preserved
+    iMatrix CSV: 88/90 off-diagonal cells exactly 0.0) that at this
+    pipeline's TSLS row counts (tens of thousands), even the *small* end
+    of its percentage-of-N libSize sweep (5% default) is already well
+    past typical CCM convergence for a low-dimensional embedding, so the
+    gate almost never fires. This directly caused the empty (0-node) GMN
+    network in every prior test of this pipeline.
+
+    A slope/delta-based convergence gate was tried here first and
+    rejected: tested directly against a clean synthetic pair with a known
+    real lagged relationship (see this pipeline's own dev notes), it
+    turned out just as unreliable as apps.InteractionMatrix's own 2-point
+    version -- a strong relationship can saturate skill almost
+    immediately (near-zero slope despite being real), while a purely
+    independent noise pair can show a spuriously *positive* delta/slope
+    from finite-sample variance alone. Fixed instead with the simpler,
+    empirically-validated approach: report the multi-sample-averaged
+    skill at ONE appropriately-sized library (CCM_WINDOW_ROWS bounds the
+    row count so this library is neither absurdly small nor already
+    saturated) with no convergence requirement at all, and let the
+    existing downstream threshold + per-node top-N-drivers selection in
+    build_network() (unchanged) do the actual pruning -- exactly the
+    real-valued-strengths-in, threshold-filtered-out design
+    apps.CreateNetwork() already expects. Verified on the same synthetic
+    pair: a real lagged relationship scored ~0.97, an independent noise
+    column scored ~0.07 -- a clean separation with no fragile gate logic.
+
+    Matches apps.CreateNetwork()'s documented convention (confirmed
+    empirically against a working simple-mde-gmn network's own iMatrix.csv,
+    job 34172242): `iMatrix.loc[driven_node, driver_node]` = skill of
+    predicting driven_node FROM driver_node's history. proper_ccm's c2t
+    direction (`target` predicted FROM `columns`) maps directly onto that;
+    calling it once per *unordered* pair yields both c2t and t2c, matching
+    apps.InteractionMatrix's own upper-triangular optimization (its module
+    docstring: "CCM returns X:Y and Y:X, also compute pairs").'''
+    windowed, _ = _ccm_window(data_lib, 0.5, ccm_window_rows)
+    libSize = max(10, len(windowed) - abs(ccm_tau) * ccm_E - 1)
+
+    iMatrix = pd.DataFrame(0.0, index=nodes, columns=nodes)
+    for i, a in enumerate(nodes):
+        for b in nodes[i + 1:]:
+            try:
+                c2t, t2c, _ = tsls.proper_ccm(
+                    windowed, libSizes=[libSize], sample=ccm_sample,
+                    tau=ccm_tau, Tp=1, E=ccm_E,
+                    target=a, columns=b,
+                    target_measurement=metric, columns_measurement=metric,
+                    gammaDelta=gamma_delta,
+                )
+                # c2t: skill of predicting `target`=a FROM `columns`=b's
+                # embedding -> iMatrix.loc[a, b] (a driven by b). t2c
+                # mirrors this the other way -> iMatrix.loc[b, a].
+                iMatrix.loc[a, b] = float(np.nanmean(c2t))
+                iMatrix.loc[b, a] = float(np.nanmean(t2c))
+            except Exception:
+                continue
+
+    for n in nodes:
+        iMatrix.loc[n, n] = np.nan
+    return iMatrix
+
+
 def build_network(targets, mde_results, df, prediction_start, prediction_length,
-                   mode, e_range, tau_range, results_dir, cores, interaction_matrix_outer_cores,
+                   mode, e_range, tau_range, results_dir, metric, gamma_delta,
+                   ccm_window_rows, ccm_sample,
                    threshold=0, fallback_node_dimension=4, max_node_dimension=15):
     '''Build one interaction matrix + strongest-driver network spanning
     `targets` and the union of their driver candidate columns
@@ -559,22 +708,12 @@
 
     Mirrors notebooks/simple-mde-gmn.ipynb cell 7's build_network(). See
     that cell's docstring for the CreateNetwork driver-ranking-bug
-    workaround rationale (identical here).
-
-    interaction_matrix_outer_cores is deliberately a *separate* knob from
-    `cores`, not just reused: apps.InteractionMatrix's own -cr sizes an
-    outer ProcessPoolExecutor across column *pairs*, but its internal
-    pyEDM.CCM() call (one per pair) creates its *own* fresh multiprocessing
-    pool every time, sized to the full visible cpu_count() by default --
-    confirmed by reading pyEDM/CCM.py (`_resolve_workers`) and
-    apps/InteractionMatrix.py's InteractFunc (which calls CCM() without
-    ever passing `parallel=`, so it can't be overridden via the CLI).
-    Neither tool coordinates with the other, so peak concurrent processes
-    is -cr * cpu_count(), not -cr -- setting -cr = cores (the seemingly
-    natural choice) squares your core request into a process count
-    (confirmed empirically: -cr=cores on a 10-core machine produced ~90
-    live processes, not ~10). Keeping this outer count low and letting
-    `cores` size CCM's own inner pool avoids that multiplication.
+    workaround rationale (identical here). The interaction matrix itself
+    is now built by build_interaction_matrix() (see its docstring) instead
+    of apps.InteractionMatrix's subprocess -- fixes the empty-network bug
+    documented there, and (as a side effect) removes the
+    apps.InteractionMatrix/pyEDM.CCM nested-pool concern this docstring
+    used to describe, since proper_ccm has no multiprocessing pool at all.
     '''
     network_id = '_'.join(targets)
     # Distinguishes MODE='GMN' network artifacts from MODE='MDE-GMN'
@@ -606,36 +745,26 @@
     data_after_mde_lib.to_csv(data_after_mde_lib_file, index=False)
 
     e_vals, tau_vals = list(e_range), list(tau_range)
-    target_grids = [target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals) for t in targets]
+    target_grids = [
+        target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals, metric, gamma_delta)
+        for t in targets
+    ]
     mean_grid = np.nanmean(np.stack(target_grids), axis=0)
     best_i, best_j = np.unravel_index(np.nanargmax(mean_grid), mean_grid.shape)
     ccm_E, ccm_tau = e_vals[best_i], tau_vals[best_j]
     print(f'{network_id}: CCM E={ccm_E}, tau={ccm_tau} '
-          f'(grid-search rho={mean_grid[best_i, best_j]:.3f})', flush=True)
+          f'(grid-search {metric}={mean_grid[best_i, best_j]:.3f})', flush=True)
 
-    interaction_prefix = results_dir / f'iMatrix_E{ccm_E}_tau{ccm_tau}_{network_id}'
-    # apps.InteractionMatrix is a CLI-only tool (no in-memory API) --
-    # invoked via subprocess instead of the notebook's IPython
-    # get_ipython().system(), which only exists inside a live kernel.
-    # MPLBACKEND=Agg keeps -P from trying to open a GUI window.
-    subprocess_env = dict(os.environ, MPLBACKEND='Agg')
-    subprocess.run(
-        [
-            'python', '-m', 'apps.InteractionMatrix',
-            '-d', str(data_after_mde_lib_file),
-            '-oc', str(interaction_prefix),
-            '-ccm', '-E', str(ccm_E), '-t', str(ccm_tau),
-            '-l', '1', str(prediction_start), '-p', '1', str(prediction_start),
-            '-P', '-cr', str(interaction_matrix_outer_cores),
-        ],
-        check=True, env=subprocess_env,
+    nodes = [c for c in data_after_mde_lib.columns if c != 'Time']
+    iMatrix = build_interaction_matrix(
+        data_after_mde_lib, nodes, ccm_E, ccm_tau, metric, gamma_delta,
+        ccm_window_rows, ccm_sample,
     )
-
-    interaction_csv = Path(f'{interaction_prefix}_CCM.csv')
-    iMatrix = pd.read_csv(interaction_csv, index_col=0)
+    interaction_csv = results_dir / f'iMatrix_E{ccm_E}_tau{ccm_tau}_{network_id}_CCM.csv'
+    iMatrix.to_csv(interaction_csv)
 
     node_dimensions = {
-        node: estimate_node_dimension(data_after_mde_lib, node, cores,
+        node: estimate_node_dimension(data_after_mde_lib, node, metric, gamma_delta,
                                        max_e=max_node_dimension, fallback=fallback_node_dimension)
         for node in iMatrix.index
     }
@@ -887,7 +1016,8 @@
 
 
 def evaluate_smap_candidate(net_result, prediction_start, prediction_length,
-                             E, tau, node_theta, variant, results_dir, cores):
+                             E, tau, node_theta, variant, results_dir, cores,
+                             metric, gamma_delta):
     config_file = build_gmn_config(
         net_result, prediction_start, prediction_length, results_dir,
         mode='Generate', E=E, tau=tau,
@@ -901,18 +1031,27 @@
     generated = G.DataOut.reset_index(drop=True)
     n_gen = len(generated)
 
+    # gmn.GMN().Generate() itself still runs pyEDM.SMap internally (no
+    # TSLS-aware equivalent exists for that -- out of scope, see this
+    # module's docstring); what changes here is only how we SCORE its
+    # already-generated output against the real held-out targets.
+    # tsls._score is cleaned_helper_functions.py's own metric dispatcher
+    # (same one proper_simplex/proper_ccm/proper_mde call internally),
+    # reused directly instead of re-implementing correlation/f1/gamma
+    # switching here.
     target_rhos = {}
     for target in net_result['targets']:
         actual_vals = actual_df[target].to_numpy()[prediction_start: prediction_start + n_gen]
         gen_vals = generated[target].to_numpy()
-        target_rhos[target] = ComputeError(actual_vals, gen_vals)['rho']
+        target_rhos[target] = tsls._score(actual_vals, gen_vals, metric, gammaDelta=gamma_delta)
 
     score = float(np.nanmean(list(target_rhos.values())))
     return score, target_rhos, G
 
 
 def run_smap_search(gmn_runs, prediction_start, prediction_length, e_range, tau_range,
-                     theta_range, coarse_theta_range, search_mode, results_dir, cores):
+                     theta_range, coarse_theta_range, search_mode, results_dir, cores,
+                     metric, gamma_delta):
     smap_search_results = {}
     for network_id, run in gmn_runs.items():
         net_result = run['net_result']
@@ -968,6 +1107,7 @@
                     score, _, _ = evaluate_smap_candidate(
                         net_result, prediction_start, prediction_length,
                         E, tau, node_theta, variant, results_dir, cores,
+                        metric, gamma_delta,
                     )
                     score_grid[i, j, k] = score
 
@@ -1033,7 +1173,7 @@
 # =============================================================================
 
 def rebuild_at_best_params(gmn_runs, smap_search_results, prediction_start, prediction_length,
-                            results_dir, cores):
+                            results_dir, cores, metric, gamma_delta):
     for network_id, run in gmn_runs.items():
         net_result = run['net_result']
         search = smap_search_results[network_id]
@@ -1079,16 +1219,17 @@
         fig.savefig(results_dir / f'gmn_generate_{network_id}.png', dpi=300)
         plt.close(fig)
 
-        print(f"{network_id} Generate() rho, held-out window:", flush=True)
+        print(f"{network_id} Generate() {metric}, held-out window:", flush=True)
         for node in nodes:
             a = actual_df[node].to_numpy()[prediction_start:n_rows]
-            r = ComputeError(a, generated[node].to_numpy())['rho']
+            r = tsls._score(a, generated[node].to_numpy(), metric, gammaDelta=gamma_delta)
             tag = 'TARGET' if node in net_result['targets'] else 'driver'
-            print(f"  {node:20s} {tag:7s} rho={r:7.3f}", flush=True)
+            print(f"  {node:20s} {tag:7s} {metric}={r:7.3f}", flush=True)
 
 
 def run_forecast_diagnostic(net_result, G, prediction_start, prediction_length,
-                             E, tau, node_theta, results_dir, cores, save_path):
+                             E, tau, node_theta, results_dir, cores, save_path,
+                             metric, gamma_delta):
     config_file = build_gmn_config(
         net_result, prediction_start, prediction_length, results_dir, mode='Forecast',
         E=E, tau=tau, node_theta_overrides=node_theta,
@@ -1114,7 +1255,7 @@
     for ax, node in zip(axes, nodes):
         actual = actual_df[node].to_numpy()
         forecast_pred = forecast_out[node].to_numpy()
-        rhos[node] = ComputeError(actual[time_idx], forecast_pred)['rho']
+        rhos[node] = tsls._score(actual[time_idx], forecast_pred, metric, gammaDelta=gamma_delta)
 
         ax.plot(np.arange(n_rows), actual, color='tab:blue', lw=1, label='Lib/actual')
         if node in generate_out.columns:
@@ -1122,7 +1263,7 @@
                     color='tab:red', lw=1, label='Generate()')
         ax.plot(time_idx, forecast_pred, color='tab:purple', lw=1.2, ls=':', label='Forecast()')
         ax.axvline(prediction_start, color='gray', ls='--', lw=0.8)
-        ax.set_title(f"{node}   Forecast rho={rhos[node]:.3f}", fontsize=10)
+        ax.set_title(f"{node}   Forecast {metric}={rhos[node]:.3f}", fontsize=10)
         ax.legend(fontsize=7, loc='upper right')
 
     fig.suptitle(f"Forecast() vs Generate() vs actual -- {net_result['network_id']}", fontsize=11)
@@ -1133,7 +1274,7 @@
     return Gf, rhos
 
 
-def gmn_fidelity_rho(net_result, G, prediction_start, default_E=3, tau=-1):
+def gmn_fidelity_rho(net_result, G, prediction_start, metric, gamma_delta, default_E=3, tau=-1):
     actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
     generated = G.DataOut.reset_index(drop=True)
     nodes = [c for c in generated.columns if c != 'Time']
@@ -1152,13 +1293,13 @@
         })
         E = max(1, min(node_dimensions.get(node, default_E), split - 1))
         try:
-            S = Simplex(
-                dataFrame = pair_df, columns = f'{node}_gen', target = f'{node}_actual',
-                E = E, tau = tau, Tp = 0,
-                lib  = f'1 {split}', pred = f'{split + 1} {n_gen}',
-                embedded = False, showPlot = False,
+            out = tsls.proper_simplex(
+                pair_df, columns=f'{node}_gen', target=f'{node}_actual',
+                E=E, tau=tau, Tp=0,
+                lib=[1, split], pred=[split + 1, n_gen],
+                embedded=False, metric=metric, gammaDelta=gamma_delta,
             )
-            fidelity_rho[node] = ComputeError(S['Observations'], S['Predictions'])['rho']
+            fidelity_rho[node] = out['metrics'][metric]
         except Exception:
             fidelity_rho[node] = np.nan
     return fidelity_rho
@@ -1182,54 +1323,32 @@
 # MDE-only diagnostics (mirrors notebook cells 15, 17, 19 -- skipped when
 # MODE == 'GMN', same guard the notebook uses)
 # =============================================================================
+# NOTE: the notebook cell 15 diagnostic (dx.Evaluate: MDE selection vs.
+# PCA/diffusion-map baselines) has no cleaned_helper_functions.py
+# equivalent -- proper_mde replaced dimx.MDE for selection itself (see
+# run_mde_for_target), but dx.Evaluate is a standalone dimx tool with no
+# TSLS-aware port, and it depended on the dx.MDE result object (r['m'])
+# this pipeline no longer constructs. Dropped rather than kept partially
+# working against a dimx.MDE object this pipeline doesn't build anymore.
 
-def run_mde_evaluate(mde_results, targets, results_dir):
-    for target in targets:
-        r = mde_results[target]
-        Eval = dx.Evaluate(
-            dataFrame     = r['mde_df'],
-            columns_range = [1, r['mde_df'].shape[1]],
-            mde_columns   = r['SELECTED_COLUMNS'],
-            predictVar    = target,
-            library       = r['m'].args.lib,
-            prediction    = r['m'].args.pred,
-            Tp            = r['m'].args.Tp,
-            tau           = r['m'].args.tau,
-            components    = 5,
-            dmap_k        = 15,
-            dmap_epsilon  = 'bgh_generous',
-            figsize       = (7, 6),
-        )
-        Eval.Run()
-        Eval.Plot()
-        plt.gcf().savefig(results_dir / f'mde_evaluate_{target}.png', dpi=300)
-        plt.close('all')
-
-
-def run_tp_vs_rho_sweep(mde_results, targets, max_tp, target_colors, results_dir):
+def run_tp_vs_rho_sweep(mde_results, targets, max_tp, target_colors, results_dir, metric, gamma_delta):
     fig, ax = plt.subplots(figsize=(8, 5))
     annotation_lines = []
     for target in targets:
         r = mde_results[target]
         cols = r['SELECTED_COLUMNS']
-        pi = PredictInterval(
-            dataFrame = r['mde_df'],
-            columns   = cols,
-            target    = target,
-            lib       = r['m'].args.lib,
-            pred      = r['m'].args.pred,
-            E         = len(cols),
-            tau       = r['m'].args.tau,
-            embedded  = True,
-            maxTp     = max_tp,
-            showPlot  = False,
+        pi = tsls.proper_predict_interval(
+            r['mde_df'], columns=cols, target=target,
+            lib=r['LIB'], pred=r['PRED'],
+            E=len(cols), tau=r['TAU'],
+            maxTp=max_tp, metric=metric, gammaDelta=gamma_delta,
         )
-        ax.plot(pi['Tp'], pi['rho'], marker='o', label=target, color=target_colors[target])
+        ax.plot(pi['Tp'], pi[metric], marker='o', label=target, color=target_colors[target])
         annotation_lines.append(f'{target}: E={len(cols)}, vars={", ".join(cols)}')
 
     ax.axvline(1, color='gray', ls=':', lw=1)
     ax.set_xlabel('Forecast interval Tp (steps)')
-    ax.set_ylabel('Prediction skill (rho)')
+    ax.set_ylabel(f'Prediction skill ({metric})')
     ax.set_title('MDE forecast skill vs. horizon')
     ax.legend(loc='best')
     fig.tight_layout()
@@ -1238,7 +1357,8 @@
     plt.close(fig)
 
 
-def run_univariate_e_tau_grid(mde_results, targets, e_range, tau_range, rho_threshold, results_dir):
+def run_univariate_e_tau_grid(mde_results, targets, e_range, tau_range, rho_threshold, results_dir,
+                               metric, gamma_delta):
     e_vals, tau_vals = list(e_range), list(tau_range)
 
     for target in targets:
@@ -1247,13 +1367,12 @@
 
         for i, E in enumerate(e_vals):
             for j, tau in enumerate(tau_vals):
-                S = Simplex(
-                    dataFrame = r['mde_df'], columns = target, target = target,
-                    E = E, tau = tau, lib = r['m'].args.lib, pred = r['m'].args.pred,
-                    embedded = False, showPlot = False,
+                out = tsls.proper_simplex(
+                    r['mde_df'], columns=target, target=target,
+                    E=E, tau=tau, lib=r['LIB'], pred=r['PRED'],
+                    embedded=False, metric=metric, gammaDelta=gamma_delta,
                 )
-                err = ComputeError(S['Observations'], S['Predictions'])
-                rho_grid[i, j] = err['rho']
+                rho_grid[i, j] = out['metrics'][metric]
 
         fig, ax = plt.subplots(figsize=(8, 7))
         im = ax.imshow(
@@ -1264,7 +1383,7 @@
         ax.set_yticks(range(len(e_vals))); ax.set_yticklabels(e_vals)
         ax.set_xlabel('tau'); ax.set_ylabel('E')
         ax.set_title(f'E, tau grid search: {target}')
-        fig.colorbar(im, ax=ax, label='rho')
+        fig.colorbar(im, ax=ax, label=metric)
 
         for i in range(len(e_vals)):
             for j in range(len(tau_vals)):
@@ -1316,7 +1435,9 @@
             mde_results[target] = run_mde_for_target(
                 target, df, obs_cols, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                 cfg['MDE_D'], cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
-                cfg['CORES'], cfg['RESULTS_DIR'],
+                cfg['RESULTS_DIR'],
+                cfg['METRIC'], cfg['GAMMA_DELTA'], cfg['CCM_WINDOW_ROWS'],
+                cfg['CCM_LIB_SIZE_PCTS'], cfg['CCM_SAMPLE'], cfg['CCM_SLOPE'],
             )
 
         if cfg['RUN_MDE_DIAGNOSTICS']:
@@ -1340,8 +1461,8 @@
     for group in target_groups_for_mode(cfg['TARGETS'], cfg['GMN_MODE']):
         net_result = build_network(
             group, mde_results, df, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
-            cfg['MODE'], cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['RESULTS_DIR'], cfg['CORES'],
-            cfg['INTERACTION_MATRIX_OUTER_CORES'],
+            cfg['MODE'], cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['RESULTS_DIR'],
+            cfg['METRIC'], cfg['GAMMA_DELTA'], cfg['CCM_WINDOW_ROWS'], cfg['CCM_SAMPLE'],
         )
         draw_colored_network(
             net_result, cfg['TARGET_COLORS'],
@@ -1353,11 +1474,12 @@
         gmn_runs, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
         cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['THETA_RANGE'], cfg['COARSE_THETA_RANGE'],
         cfg['SMAP_SEARCH_MODE'], cfg['RESULTS_DIR'], cfg['CORES'],
+        cfg['METRIC'], cfg['GAMMA_DELTA'],
     )
 
     rebuild_at_best_params(
         gmn_runs, smap_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
-        cfg['RESULTS_DIR'], cfg['CORES'],
+        cfg['RESULTS_DIR'], cfg['CORES'], cfg['METRIC'], cfg['GAMMA_DELTA'],
     )
 
     for network_id, run in gmn_runs.items():
@@ -1366,28 +1488,34 @@
             run['best_E'], run['best_tau'], run['best_node_theta'],
             cfg['RESULTS_DIR'], cfg['CORES'],
             cfg['RESULTS_DIR'] / f'gmn_forecast_vs_generate_{network_id}.png',
+            cfg['METRIC'], cfg['GAMMA_DELTA'],
         )
         run['Gf'] = Gf
         run['forecast_rho'] = forecast_rho
-        print(f"{network_id} Forecast() rho per node (weakest first):", flush=True)
+        print(f"{network_id} Forecast() {cfg['METRIC']} per node (weakest first):", flush=True)
         for node, rho in sorted(forecast_rho.items(), key=lambda kv: kv[1]):
             print(f"  {node:20s} {rho:.4f}", flush=True)
 
-        rho = gmn_fidelity_rho(run['net_result'], run['G'], cfg['PREDICTION_START'])
+        rho = gmn_fidelity_rho(run['net_result'], run['G'], cfg['PREDICTION_START'],
+                                cfg['METRIC'], cfg['GAMMA_DELTA'])
         plot_fidelity(rho, network_id, cfg['RESULTS_DIR'] / f'gmn_fidelity_{network_id}.png')
-        print(f'{network_id} fidelity rho (weakest first):', flush=True)
+        print(f'{network_id} fidelity {cfg["METRIC"]} (weakest first):', flush=True)
         for node in sorted(rho, key=lambda n: (np.isnan(rho[n]), -(rho[n] if not np.isnan(rho[n]) else 0))):
             print(f'  {node:20s} {rho[node]:.4f}', flush=True)
 
         generated = run['G'].DataOut
         generated.to_csv(cfg['RESULTS_DIR'] / f'gmn_generate_output_{network_id}.csv', index=False)
 
-    # ---- MDE-only diagnostics ------------------------------------------
+    # ---- MDE-only diagnostics -------------------------------------------
+    # run_mde_evaluate (dx.Evaluate: MDE vs PCA/diffusion-map) dropped --
+    # see the "MDE-only diagnostics" section docstring above
+    # run_univariate_e_tau_grid for why.
     if cfg['MODE'] == 'MDE-GMN' and cfg['RUN_MDE_DIAGNOSTICS']:
-        run_mde_evaluate(mde_results, cfg['TARGETS'], cfg['RESULTS_DIR'])
-        run_tp_vs_rho_sweep(mde_results, cfg['TARGETS'], cfg['MAX_TP'], cfg['TARGET_COLORS'], cfg['RESULTS_DIR'])
+        run_tp_vs_rho_sweep(mde_results, cfg['TARGETS'], cfg['MAX_TP'], cfg['TARGET_COLORS'],
+                             cfg['RESULTS_DIR'], cfg['METRIC'], cfg['GAMMA_DELTA'])
         run_univariate_e_tau_grid(mde_results, cfg['TARGETS'], cfg['E_RANGE'], cfg['TAU_RANGE'],
-                                   cfg['GRID_RHO_THRESHOLD'], cfg['RESULTS_DIR'])
+                                   cfg['GRID_RHO_THRESHOLD'], cfg['RESULTS_DIR'],
+                                   cfg['METRIC'], cfg['GAMMA_DELTA'])
     else:
         print(f"MODE={cfg['MODE']!r} / RUN_MDE_DIAGNOSTICS={cfg['RUN_MDE_DIAGNOSTICS']}: "
               f"skipping MDE-only diagnostics.", flush=True)
```

---


## Job 35513363

Generated: 2026-07-29T18:05:42+09:00

Git commit at submission: `5db3a5d` (clean vs. HEAD)

Compared against: **job_35347307**

### Parameter changes (tsls-mde-gmn.sbatch)

_No parameter changes._

### Batch script design changes (tsls-mde-gmn.sbatch, excludes parameter block above)

```diff
--- job_35347307/tsls-mde-gmn.sbatch
+++ job_35513363/tsls-mde-gmn.sbatch
@@ -21,6 +21,8 @@
 # scripts/batch/ instead of under results/.
 #SBATCH --output=/home/j/j-tominaga/pao_rotation/results/%x-%j.out
 #SBATCH --error=/home/j/j-tominaga/pao_rotation/results/%x-%j.err
+#SBATCH --mail-user=j-tominaga@oist.jp
+#SBATCH --mail-type=END,FAIL
 
 set -euo pipefail
 
@@ -78,21 +80,22 @@
 # ==============================================================================
 # JOB DESCRIPTION
 # ==============================================================================
-# Snapshots this run's tsls-mde-gmn.py + tsls-mde-gmn.sbatch into
-# $OUT_DIR and inserts a new entry into $PROJ_DIR/JOB_LOG.md (parameter-
-# block changes as a table, everything else as unified diffs), diffed
-# against the most recent earlier job with its own tsls-mde-gmn.sbatch/.py
-# snapshot pair (generate_job_description.sh keys the "previous job"
-# search off the source filenames, so this never gets compared against a
-# simple-mde-gmn job by mistake). Runs before the pipeline itself so the
-# entry is written even if the run later fails or times out.
+# Diffs this run's tsls-mde-gmn.py + tsls-mde-gmn.sbatch against the
+# rolling snapshot cache of the last tsls-mde-gmn submission
+# ($PROJ_DIR/results/.snapshots/ -- keyed by filename, so this never gets
+# compared against a simple-mde-gmn job by mistake; see
+# generate_job_description.sh's own module docstring), inserting a new
+# entry into $PROJ_DIR/JOB_LOG.md (parameter-block changes as a table,
+# everything else as unified diffs), then updates that cache to this
+# run's files. Runs before the pipeline itself so the entry is written
+# even if the run later fails or times out.
 #
 # sbatch_src is "$0", not a hardcoded path to this canonical file -- same
 # reasoning as simple-mde-gmn.sbatch: sbatch copies whatever file it's
 # given verbatim into the running job, and $0 reflects that actual
-# submitted path, which matters whenever a job is submitted from
-# something other than this canonical location (e.g. resubmitting a
-# byte-faithful rerun via `sbatch results/job_<id>/tsls-mde-gmn.sbatch`).
+# submitted path, which matters if the live canonical file at
+# scripts/batch/tsls-mde-gmn.sbatch gets edited again while this job sits
+# queued.
 "$PROJ_DIR/scripts/batch/generate_job_description.sh" \
     "$PROJ_DIR" "$OUT_DIR" "$SLURM_JOB_ID" \
     "$0" \
@@ -146,7 +149,16 @@
 
 export TSLS_MDE_GMN_DATA_DIR="$tempdir/data"
 export TSLS_MDE_GMN_DATA_FILE="${DATA_FILE}"
-export TSLS_MDE_GMN_RESULTS_DIR="$tempdir/tsls_mde_gmn_outputs"
+# TEMP DIAGNOSTIC: pointed directly at $OUT_DIR (not $tempdir, which the
+# `cleanup` trap deletes on every exit including a crash) so intermediate
+# outputs -- especially the CCM interaction-matrix CSV -- survive if this
+# run fails the same way jobs 35332887/35347307 did, instead of being lost
+# with the rest of $tempdir before anyone can inspect them. Revert to
+# "$tempdir/tsls_mde_gmn_outputs" once this is resolved -- writing directly
+# under $OUT_DIR (not local /flash scratch) is slower I/O for every
+# intermediate file this pipeline writes, acceptable for one diagnostic run
+# but not a permanent tradeoff.
+export TSLS_MDE_GMN_RESULTS_DIR="$OUT_DIR/live_debug"
 export TSLS_MDE_GMN_MODE="${MODE}"
 export TSLS_MDE_GMN_GMN_MODE="${GMN_MODE}"
 export TSLS_MDE_GMN_TARGETS="${TARGETS}"
@@ -173,7 +185,13 @@
 # SAVE OUTPUTS
 # ==============================================================================
 
-cp -av tsls_mde_gmn_outputs/. "$OUT_DIR"/
+# TEMP DIAGNOSTIC: TSLS_MDE_GMN_RESULTS_DIR above already points at
+# $OUT_DIR/live_debug (not a relative $tempdir path), so there's nothing
+# to copy from $tempdir on success -- flatten live_debug/ into $OUT_DIR
+# directly instead, matching this pipeline's normal final layout (files
+# directly under $OUT_DIR, not nested). Revert alongside the
+# TSLS_MDE_GMN_RESULTS_DIR change above once resolved.
+cp -av "$OUT_DIR/live_debug/." "$OUT_DIR"/
 
 echo "========================================================"
 echo "Job ${SLURM_JOB_ID} complete. Outputs saved to:"
```

### Python pipeline changes (tsls-mde-gmn.py)

_No changes._

---


## Job 35347307

Generated: 2026-07-28T19:18:03+09:00

Git commit at submission: `5db3a5d` (clean vs. HEAD)

Compared against: **job_35332887**

### Parameter changes (tsls-mde-gmn.sbatch)

| Parameter | job_35332887 | job_35347307 |
|---|---|---|
| E_MAX | 5 | 13 |
| E_STEP | 2 | 4 |
| TAU_MAX | -1 | 0 |
| TAU_MIN | -3 | -9 |
| TAU_STEP | 2 | 4 |

### Batch script design changes (tsls-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (tsls-mde-gmn.py)

_No changes._

---


## Job 35332887

Generated: 2026-07-28T17:21:09+09:00

Git commit at submission: `5db3a5d` (clean vs. HEAD)

No previous job snapshot found -- this is the first tracked run, or all prior jobs predate this tracking.

---


## Job 34172243

Generated: 2026-07-19T00:52:17+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_34172242**

### Plain-language summary

This is a **byte-identical rerun of job_34145281** -- same `simple-mde-gmn.sbatch`
parameters (submitted directly from `results/job_34145281/simple-mde-gmn.sbatch`),
now with the full performance fix applied. job_34145281 was killed at the 2-day SLURM time limit without finishing.

This supersedes an earlier same-day resubmission attempt (job_341714xx,
cancelled and deleted) that only had a partial fix: that round patched
`pyEDM.CCM`'s inner pool from inside simple-mde-gmn.py itself, but
`apps.InteractionMatrix` runs in a **separate subprocess**
(`subprocess.run(['python', '-m', 'apps.InteractionMatrix', ...])`), which
re-imports pyEDM fresh from disk -- an in-process monkeypatch in the
parent can't reach it. Two more pyEDM functions this pipeline calls
in-process (`EmbedDimension`, once per node; `PredictNonlinear`, once per
node per (E,tau) candidate in the E/tau/theta search) turned out to have
the identical fresh-Pool-per-call bug with no patchable hook at all, so
they needed replacing outright, not just tuning.

Full fix, now in `.venv/lib/python3.11/site-packages/sitecustomize.py`
(auto-loaded by every Python process using this venv, including the
InteractionMatrix subprocess -- the only patch location that reaches
both): forces `pyEDM.CCM`, `EmbedDimension`, `PredictInterval`, and
`PredictNonlinear` all onto their sequential in-process path instead of
spawning a fresh multiprocessing Pool per call. Verified directly: all
four patched functions confirmed active in a plain subprocess, dimx.Run,
and this pipeline's own process; EmbedDimension/PredictNonlinear
sequential replacements produce numerically sane rho values in well
under a second per call (vs. spawning up to `cores`=32 processes each
call previously). Root cause and full evidence trail in
`scripts/src/simple-mde-gmn.py`'s module docstring, `_sequential_ccm_workers`,
and `sitecustomize.py`'s own docstring.

The auto-generated parameter table below compares this job against
whichever job most recently got its own snapshot written (a side effect
of resubmitting all 6 originals back-to-back) -- **not** against
job_34145281 specifically. Ignore it for understanding this rerun.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_34172242 | job_34172243 |
|---|---|---|
| MDE_D | 20 | 50 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---

## Job 34172242

Generated: 2026-07-19T00:52:17+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_34172241**

### Plain-language summary

This is a **byte-identical rerun of job_34145278** -- same `simple-mde-gmn.sbatch`
parameters (submitted directly from `results/job_34145278/simple-mde-gmn.sbatch`),
now with the full performance fix applied. job_34145278 completed, but only barely (1 day 21 hours -- within ~3 hours of the 2-day cap).

This supersedes an earlier same-day resubmission attempt (job_341714xx,
cancelled and deleted) that only had a partial fix: that round patched
`pyEDM.CCM`'s inner pool from inside simple-mde-gmn.py itself, but
`apps.InteractionMatrix` runs in a **separate subprocess**
(`subprocess.run(['python', '-m', 'apps.InteractionMatrix', ...])`), which
re-imports pyEDM fresh from disk -- an in-process monkeypatch in the
parent can't reach it. Two more pyEDM functions this pipeline calls
in-process (`EmbedDimension`, once per node; `PredictNonlinear`, once per
node per (E,tau) candidate in the E/tau/theta search) turned out to have
the identical fresh-Pool-per-call bug with no patchable hook at all, so
they needed replacing outright, not just tuning.

Full fix, now in `.venv/lib/python3.11/site-packages/sitecustomize.py`
(auto-loaded by every Python process using this venv, including the
InteractionMatrix subprocess -- the only patch location that reaches
both): forces `pyEDM.CCM`, `EmbedDimension`, `PredictInterval`, and
`PredictNonlinear` all onto their sequential in-process path instead of
spawning a fresh multiprocessing Pool per call. Verified directly: all
four patched functions confirmed active in a plain subprocess, dimx.Run,
and this pipeline's own process; EmbedDimension/PredictNonlinear
sequential replacements produce numerically sane rho values in well
under a second per call (vs. spawning up to `cores`=32 processes each
call previously). Root cause and full evidence trail in
`scripts/src/simple-mde-gmn.py`'s module docstring, `_sequential_ccm_workers`,
and `sitecustomize.py`'s own docstring.

The auto-generated parameter table below compares this job against
whichever job most recently got its own snapshot written (a side effect
of resubmitting all 6 originals back-to-back) -- **not** against
job_34145278 specifically. Ignore it for understanding this rerun.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_34172241 | job_34172242 |
|---|---|---|
| DATA_FILE | J1620210605_epoch_8_freq_4_Hz.csv | J1620210605_epoch_8_freq_1_Hz.csv |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---

## Job 34172241

Generated: 2026-07-19T00:52:16+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_34172240**

### Plain-language summary

This is a **byte-identical rerun of job_34145275** -- same `simple-mde-gmn.sbatch`
parameters (submitted directly from `results/job_34145275/simple-mde-gmn.sbatch`),
now with the full performance fix applied. job_34145275 was killed at the 2-day SLURM time limit without finishing.

This supersedes an earlier same-day resubmission attempt (job_341714xx,
cancelled and deleted) that only had a partial fix: that round patched
`pyEDM.CCM`'s inner pool from inside simple-mde-gmn.py itself, but
`apps.InteractionMatrix` runs in a **separate subprocess**
(`subprocess.run(['python', '-m', 'apps.InteractionMatrix', ...])`), which
re-imports pyEDM fresh from disk -- an in-process monkeypatch in the
parent can't reach it. Two more pyEDM functions this pipeline calls
in-process (`EmbedDimension`, once per node; `PredictNonlinear`, once per
node per (E,tau) candidate in the E/tau/theta search) turned out to have
the identical fresh-Pool-per-call bug with no patchable hook at all, so
they needed replacing outright, not just tuning.

Full fix, now in `.venv/lib/python3.11/site-packages/sitecustomize.py`
(auto-loaded by every Python process using this venv, including the
InteractionMatrix subprocess -- the only patch location that reaches
both): forces `pyEDM.CCM`, `EmbedDimension`, `PredictInterval`, and
`PredictNonlinear` all onto their sequential in-process path instead of
spawning a fresh multiprocessing Pool per call. Verified directly: all
four patched functions confirmed active in a plain subprocess, dimx.Run,
and this pipeline's own process; EmbedDimension/PredictNonlinear
sequential replacements produce numerically sane rho values in well
under a second per call (vs. spawning up to `cores`=32 processes each
call previously). Root cause and full evidence trail in
`scripts/src/simple-mde-gmn.py`'s module docstring, `_sequential_ccm_workers`,
and `sitecustomize.py`'s own docstring.

The auto-generated parameter table below compares this job against
whichever job most recently got its own snapshot written (a side effect
of resubmitting all 6 originals back-to-back) -- **not** against
job_34145275 specifically. Ignore it for understanding this rerun.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_34172240 | job_34172241 |
|---|---|---|
| MODE | GMN | MDE-GMN |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---

## Job 34172240

Generated: 2026-07-19T00:52:16+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_34172239**

### Plain-language summary

This is a **byte-identical rerun of job_34143223** -- same `simple-mde-gmn.sbatch`
parameters (submitted directly from `results/job_34143223/simple-mde-gmn.sbatch`),
now with the full performance fix applied. job_34143223 was killed at the 2-day SLURM time limit without finishing.

This supersedes an earlier same-day resubmission attempt (job_341714xx,
cancelled and deleted) that only had a partial fix: that round patched
`pyEDM.CCM`'s inner pool from inside simple-mde-gmn.py itself, but
`apps.InteractionMatrix` runs in a **separate subprocess**
(`subprocess.run(['python', '-m', 'apps.InteractionMatrix', ...])`), which
re-imports pyEDM fresh from disk -- an in-process monkeypatch in the
parent can't reach it. Two more pyEDM functions this pipeline calls
in-process (`EmbedDimension`, once per node; `PredictNonlinear`, once per
node per (E,tau) candidate in the E/tau/theta search) turned out to have
the identical fresh-Pool-per-call bug with no patchable hook at all, so
they needed replacing outright, not just tuning.

Full fix, now in `.venv/lib/python3.11/site-packages/sitecustomize.py`
(auto-loaded by every Python process using this venv, including the
InteractionMatrix subprocess -- the only patch location that reaches
both): forces `pyEDM.CCM`, `EmbedDimension`, `PredictInterval`, and
`PredictNonlinear` all onto their sequential in-process path instead of
spawning a fresh multiprocessing Pool per call. Verified directly: all
four patched functions confirmed active in a plain subprocess, dimx.Run,
and this pipeline's own process; EmbedDimension/PredictNonlinear
sequential replacements produce numerically sane rho values in well
under a second per call (vs. spawning up to `cores`=32 processes each
call previously). Root cause and full evidence trail in
`scripts/src/simple-mde-gmn.py`'s module docstring, `_sequential_ccm_workers`,
and `sitecustomize.py`'s own docstring.

The auto-generated parameter table below compares this job against
whichever job most recently got its own snapshot written (a side effect
of resubmitting all 6 originals back-to-back) -- **not** against
job_34143223 specifically. Ignore it for understanding this rerun.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_34172239 | job_34172240 |
|---|---|---|
| MODE | MDE-GMN | GMN |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---

## Job 34172239

Generated: 2026-07-19T00:52:16+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_34172238**

### Plain-language summary

This is a **byte-identical rerun of job_34143219** -- same `simple-mde-gmn.sbatch`
parameters (submitted directly from `results/job_34143219/simple-mde-gmn.sbatch`),
now with the full performance fix applied. job_34143219 was killed at the 2-day SLURM time limit without finishing.

This supersedes an earlier same-day resubmission attempt (job_341714xx,
cancelled and deleted) that only had a partial fix: that round patched
`pyEDM.CCM`'s inner pool from inside simple-mde-gmn.py itself, but
`apps.InteractionMatrix` runs in a **separate subprocess**
(`subprocess.run(['python', '-m', 'apps.InteractionMatrix', ...])`), which
re-imports pyEDM fresh from disk -- an in-process monkeypatch in the
parent can't reach it. Two more pyEDM functions this pipeline calls
in-process (`EmbedDimension`, once per node; `PredictNonlinear`, once per
node per (E,tau) candidate in the E/tau/theta search) turned out to have
the identical fresh-Pool-per-call bug with no patchable hook at all, so
they needed replacing outright, not just tuning.

Full fix, now in `.venv/lib/python3.11/site-packages/sitecustomize.py`
(auto-loaded by every Python process using this venv, including the
InteractionMatrix subprocess -- the only patch location that reaches
both): forces `pyEDM.CCM`, `EmbedDimension`, `PredictInterval`, and
`PredictNonlinear` all onto their sequential in-process path instead of
spawning a fresh multiprocessing Pool per call. Verified directly: all
four patched functions confirmed active in a plain subprocess, dimx.Run,
and this pipeline's own process; EmbedDimension/PredictNonlinear
sequential replacements produce numerically sane rho values in well
under a second per call (vs. spawning up to `cores`=32 processes each
call previously). Root cause and full evidence trail in
`scripts/src/simple-mde-gmn.py`'s module docstring, `_sequential_ccm_workers`,
and `sitecustomize.py`'s own docstring.

The auto-generated parameter table below compares this job against
whichever job most recently got its own snapshot written (a side effect
of resubmitting all 6 originals back-to-back) -- **not** against
job_34143219 specifically. Ignore it for understanding this rerun.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_34172238 | job_34172239 |
|---|---|---|
| DATA_FILE | J1620210605_epoch_8_freq_40_Hz.csv | J1620210605_epoch_8_freq_4_Hz.csv |
| MODE | GMN | MDE-GMN |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---

## Job 34172238

Generated: 2026-07-19T00:52:16+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_34145281**

### Plain-language summary

This is a **byte-identical rerun of job_34143131** -- same `simple-mde-gmn.sbatch`
parameters (submitted directly from `results/job_34143131/simple-mde-gmn.sbatch`),
now with the full performance fix applied. job_34143131 was killed at the 2-day SLURM time limit without finishing.

This supersedes an earlier same-day resubmission attempt (job_341714xx,
cancelled and deleted) that only had a partial fix: that round patched
`pyEDM.CCM`'s inner pool from inside simple-mde-gmn.py itself, but
`apps.InteractionMatrix` runs in a **separate subprocess**
(`subprocess.run(['python', '-m', 'apps.InteractionMatrix', ...])`), which
re-imports pyEDM fresh from disk -- an in-process monkeypatch in the
parent can't reach it. Two more pyEDM functions this pipeline calls
in-process (`EmbedDimension`, once per node; `PredictNonlinear`, once per
node per (E,tau) candidate in the E/tau/theta search) turned out to have
the identical fresh-Pool-per-call bug with no patchable hook at all, so
they needed replacing outright, not just tuning.

Full fix, now in `.venv/lib/python3.11/site-packages/sitecustomize.py`
(auto-loaded by every Python process using this venv, including the
InteractionMatrix subprocess -- the only patch location that reaches
both): forces `pyEDM.CCM`, `EmbedDimension`, `PredictInterval`, and
`PredictNonlinear` all onto their sequential in-process path instead of
spawning a fresh multiprocessing Pool per call. Verified directly: all
four patched functions confirmed active in a plain subprocess, dimx.Run,
and this pipeline's own process; EmbedDimension/PredictNonlinear
sequential replacements produce numerically sane rho values in well
under a second per call (vs. spawning up to `cores`=32 processes each
call previously). Root cause and full evidence trail in
`scripts/src/simple-mde-gmn.py`'s module docstring, `_sequential_ccm_workers`,
and `sitecustomize.py`'s own docstring.

The auto-generated parameter table below compares this job against
whichever job most recently got its own snapshot written (a side effect
of resubmitting all 6 originals back-to-back) -- **not** against
job_34143131 specifically. Ignore it for understanding this rerun.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_34145281 | job_34172238 |
|---|---|---|
| DATA_FILE | J1620210605_epoch_8_freq_1_Hz.csv | J1620210605_epoch_8_freq_40_Hz.csv |
| MDE_D | 50 | 20 |
| MODE | MDE-GMN | GMN |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

```diff
--- job_34145281/simple-mde-gmn.py
+++ job_34172238/simple-mde-gmn.py
@@ -42,17 +42,22 @@
     hardcoded default, which doesn't know how many cores SLURM actually
     allocated.
 
-KNOWN CAVEAT (found empirically running the MODE='GMN' interaction-matrix
-step against the real dataset): apps.InteractionMatrix's own
+FIXED CAVEAT (found empirically running the MODE='GMN' interaction-matrix
+step against the real dataset, root-caused after several jobs each ran
+~2 days regardless of dataset size or MODE): apps.InteractionMatrix's own
 ProcessPoolExecutor (sized by -cr/CORES) and pyEDM.CCM()'s own *internal*
-multiprocessing nest inside each other -- with CORES=8 this produced
-~90 concurrent processes contending for the cores actually available,
-not 8. This isn't a hang (confirmed via per-process CPU-time deltas: work
-keeps accumulating, just slower than a naive CORES-fold speedup would
-predict) but it does mean interaction-matrix wall time won't scale
-linearly with -cr, and can badly oversubscribe a shared/cgroup-limited
-compute node. If a job runs far longer than expected, check actual CPU
-utilization before assuming something is broken.
+multiprocessing nested inside each other -- with CORES=8 this produced
+~90 concurrent processes, not 8, and every one of pyEDM.CCM()'s ~100k+
+per-channel-pair (or per-MDE-candidate) calls paid a full `spawn`/
+`forkserver` Pool creation + heavy reimport (numpy/scipy/pandas/sklearn/
+pyEDM/gmn) teardown cost for work that's only 4-8 tiny KDTree tasks --
+process-spawn overhead dominating actual compute by orders of magnitude,
+which is why cutting dataset resolution or capping MDE_D didn't help.
+Now patched: `_sequential_ccm_workers` below forces pyEDM.CCM() onto its
+in-process sequential path everywhere in this pipeline, eliminating the
+inner pool entirely so only the (already-present, actually useful) outer
+pool provides parallelism. See `_sequential_ccm_workers`'s and
+build_network()'s docstrings for the full evidence trail.
 '''
 import os
 import pickle
@@ -210,6 +215,46 @@
 gmn.Node.Forecast = _fixed_node_forecast
 
 
+def _sequential_ccm_workers(parallel):
+    '''Replaces pyEDM.CCM's private _resolve_workers(), forcing every
+    CCM().Project() call in this process onto its sequential in-process
+    path instead of spawning a fresh multiprocessing Pool.
+
+    This is the actual fix for the nested-pool blowup documented in this
+    module's KNOWN CAVEAT and in build_network()'s docstring below: every
+    single pyEDM.CCM() call -- one per channel pair in
+    apps.InteractionMatrix, one per MDE candidate column in dimx.Run --
+    creates its *own* fresh `spawn`/`forkserver` Pool sized to
+    cpu_count() (CCM.py's `_resolve_workers`, called from `Project()`),
+    on top of whichever outer pool (apps.InteractionMatrix's
+    ProcessPoolExecutor, dimx's CrossMapColumns Pool) is already
+    dispatching that call. Neither apps.InteractionMatrix nor dimx.Run
+    ever pass `parallel=` through to CCM(), so this can't be tuned via
+    their CLIs -- confirmed empirically (see the "No module named
+    'kedm'" line in a job's .out, printed once per spawned worker
+    process re-importing gmn from scratch: hundreds of thousands of
+    occurrences on a single run) and by reading pyEDM/CCM.py directly.
+
+    Forcing sequential execution here is strictly a win, not a
+    parallelism/wall-time tradeoff: every CCM() call in this pipeline
+    resolves to only 4-8 tiny tasks (2 directions x 2-4 libSizes -- see
+    apps/InteractionMatrix.py's libSizes=[libMin,libMax] and dimx's
+    default pLibSizes=[10,15,85,90]), each a KDTree build+query over a
+    handful of embedding rows. Running those 4-8 tasks in-process is
+    faster than the Pool spawn + full heavy-import (numpy/scipy/pandas/
+    sklearn/pyEDM/gmn) teardown cost alone, before any actual work
+    happens -- and the real parallelism across channel pairs/candidates
+    already comes from the outer pool, which this doesn't touch.
+    Monkeypatched (not `parallel=False` at each call site) because
+    apps.InteractionMatrix and dimx.Run construct CCM() internally and
+    don't expose that argument.'''
+    return 1
+
+
+import pyEDM.CCM as _pyedm_ccm_module
+_pyedm_ccm_module._resolve_workers = _sequential_ccm_workers
+
+
 # =============================================================================
 # Configuration -- every value overridable via MDE_GMN_* environment
 # variables; see scripts/batch/simple-mde-gmn.sbatch's MODIFIABLE
@@ -472,18 +517,26 @@
 
     interaction_matrix_outer_cores is deliberately a *separate* knob from
     `cores`, not just reused: apps.InteractionMatrix's own -cr sizes an
-    outer ProcessPoolExecutor across column *pairs*, but its internal
-    pyEDM.CCM() call (one per pair) creates its *own* fresh multiprocessing
-    pool every time, sized to the full visible cpu_count() by default --
-    confirmed by reading pyEDM/CCM.py (`_resolve_workers`) and
-    apps/InteractionMatrix.py's InteractFunc (which calls CCM() without
-    ever passing `parallel=`, so it can't be overridden via the CLI).
-    Neither tool coordinates with the other, so peak concurrent processes
-    is -cr * cpu_count(), not -cr -- setting -cr = cores (the seemingly
-    natural choice) squares your core request into a process count
-    (confirmed empirically: -cr=cores on a 10-core machine produced ~90
-    live processes, not ~10). Keeping this outer count low and letting
-    `cores` size CCM's own inner pool avoids that multiplication.
+    outer ProcessPoolExecutor across column *pairs*. Its internal
+    pyEDM.CCM() call (one per pair) used to create its *own* fresh
+    multiprocessing pool every time, sized to the full visible
+    cpu_count() by default -- confirmed by reading pyEDM/CCM.py
+    (`_resolve_workers`) and apps/InteractionMatrix.py's InteractFunc
+    (which calls CCM() without ever passing `parallel=`, so it couldn't
+    be overridden via the CLI). Neither tool coordinated with the other,
+    so peak concurrent processes was -cr * cpu_count(), not -cr --
+    setting -cr = cores (the seemingly natural choice) squared your core
+    request into a process count (confirmed empirically: -cr=cores on a
+    10-core machine produced ~90 live processes, not ~10).
+
+    This module now monkeypatches pyEDM.CCM's inner pool away entirely
+    (see `_sequential_ccm_workers` near the top of this file) rather than
+    just avoiding the multiplication by keeping this knob low -- every
+    CCM() call here only ever resolves to 4-8 tiny tasks, so the inner
+    pool was pure spawn/reimport overhead with no parallelism benefit to
+    trade away. interaction_matrix_outer_cores stays a separate knob
+    regardless, since it's still the one thing sizing real parallelism
+    across column pairs.
     '''
     network_id = '_'.join(targets)
     # Distinguishes MODE='GMN' network artifacts from MODE='MDE-GMN'
```

---

## Job 34145281

Generated: 2026-07-16T23:55:11+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_34145278**

### Plain-language summary

Single change from job_34145278: **MDE_D** `20` -> `50` -- raises the
forward-selection dimension ceiling, so MDE is now allowed to select up
to 50 drivers per target instead of 20 before handing off to GMN. Since
MODE is still `MDE-GMN` and DATA_FILE is still the 1 Hz recording
(both unchanged from job_34145278), this widens the resulting GMN
network (more potential driver columns per target) without touching
which dataset or overall pipeline shape is used. MDE's own per-step cost
scales with this ceiling, so expect a somewhat longer MDE
forward-selection phase than job_34145278, though still on the cheap
1 Hz data.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_34145278 | job_34145281 |
|---|---|---|
| MDE_D | 20 | 50 |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---

## Job 34145278

Generated: 2026-07-16T23:54:26+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_34145275**

### Plain-language summary

Single change from job_34145275: **DATA_FILE** `..._freq_4_Hz.csv` ->
`..._freq_1_Hz.csv` -- a further ~4x row-count drop (1 Hz is the
coarsest of the three recordings on disk: 1/4/40 Hz). `MODE` stays at
`MDE-GMN` (unchanged from job_34145275), so this still does MDE
pre-filtering before building the GMN network, just on much sparser
data. Confirmed the 1 Hz file exists in `data/` before submitting.

This should be the cheapest/fastest of the 6 jobs submitted today by a
wide margin, given the combination of MDE pre-filtering + the sparsest
dataset.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_34145275 | job_34145278 |
|---|---|---|
| DATA_FILE | J1620210605_epoch_8_freq_4_Hz.csv | J1620210605_epoch_8_freq_1_Hz.csv |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---

## Job 34145275

Generated: 2026-07-16T23:53:34+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_34143223**

### Plain-language summary

Single change from job_34143223: **MODE** reverted `GMN` -> `MDE-GMN`,
re-enabling MDE forward-selection so the GMN network is built from
`TARGETS` + each target's MDE-selected drivers (capped by `MDE_D=20`)
instead of the full REGION_CHANNEL_CAP=999 channel set. `DATA_FILE`
stays at the 4 Hz recording (unchanged from job_34143223).

Note: across the last 4 submissions MODE has alternated
GMN -> MDE-GMN -> GMN -> MDE-GMN, each on the 4 Hz data. All 4 are still
running concurrently (5+ hours in, well within the 2-day budget) -- if
this alternation was meant as an A/B comparison rather than exploratory
toggling, it's worth deciding which of the 4 concurrent runs' results
you actually need before submitting further variants, since they're
consuming 32 cores x 48G each on the same partition.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_34143223 | job_34145275 |
|---|---|---|
| MODE | GMN | MDE-GMN |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---

## Job 34143223

Generated: 2026-07-16T18:51:15+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_34143219**

### Plain-language summary

Single change from job_34143219: **MODE** reverted `MDE-GMN` -> `GMN`,
undoing that job's switch back to MDE pre-filtering. `DATA_FILE` stays
at the 4 Hz recording (unchanged from job_34143219).

With MODE=GMN, MDE forward-selection is skipped again and the GMN
interaction matrix is built directly from the full REGION_CHANNEL_CAP=999
channel set (all CA1_/mPFC_/OFC_ channels), same shape as job_34143131 --
but now on the smaller 4 Hz dataset (~10x fewer rows) instead of the 40 Hz
one job_34143131 used, so this should still run meaningfully faster than
34143131 despite using the same uncapped-channel-set mode.

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_34143219 | job_34143223 |
|---|---|---|
| MODE | MDE-GMN | GMN |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---

## Job 34143219

Generated: 2026-07-16T18:49:10+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

Compared against: **job_34143131**

### Plain-language summary

Two parameter changes from job_34143131, and they compound:

- **DATA_FILE**: 40 Hz recording -> 4 Hz recording. ~10x fewer rows, so
  every downstream step (interaction matrix, E/tau search, Generate())
  runs on a much smaller dataset.
- **MODE**: `GMN` -> `MDE-GMN`. job_34143131 skipped MDE entirely and
  built the interaction matrix from every channel (the expensive,
  uncapped path). This job re-enables MDE forward-selection first, so
  the GMN network is built from just `TARGETS` plus each target's
  MDE-selected drivers (bounded by `MDE_D=20`) instead of the full
  channel set -- a much smaller network, and it also re-enables the
  MDE-only diagnostic plots (E/tau grid, rho-vs-dimension, PCA/Diffusion
  Map comparison) that MODE=GMN skips.

Net effect: this run trades the previous job's "large network, no
pre-filtering, high-resolution data" shape for a much cheaper "MDE-
filtered network, low-resolution data" shape -- expect a substantially
shorter wall time than job_34143131, which is still running concurrently
(unrelated to this change, not superseded by it).

### Parameter changes (simple-mde-gmn.sbatch)

| Parameter | job_34143131 | job_34143219 |
|---|---|---|
| DATA_FILE | J1620210605_epoch_8_freq_40_Hz.csv | J1620210605_epoch_8_freq_4_Hz.csv |
| MODE | GMN | MDE-GMN |

### Batch script design changes (simple-mde-gmn.sbatch, excludes parameter block above)

_No changes._

### Python pipeline changes (simple-mde-gmn.py)

_No changes._

---

## Job 34143131

Generated: 2026-07-16T18:32:31+09:00

Git commit at submission: `5db3a5d` (uncommitted local changes vs. HEAD)

### Plain-language summary

No earlier job has a saved snapshot to diff against (job_34119660 and
job_34123136 predate this tracking), so this is effectively the baseline
run for the new job-description system. What it's configured to do:

- **MODE=GMN** (not the MDE-GMN default): skips the MDE forward-selection
  step entirely and builds the CCM interaction matrix directly from the
  full channel set. This is the more expensive path -- an O(channels^2)
  interaction matrix -- rather than first narrowing to each target's
  MDE-selected drivers.
- **GMN_MODE=combined**, **TARGETS=head_position_x,head_position_y**: one
  network spans both position targets together, rather than building two
  independent networks.
- **REGION_CHANNEL_CAP=999**: effectively uncapped -- every CA1_/mPFC_/OFC_
  channel in the recording is used as a node, not a test-sized subset.
- **DATA_FILE=J1620210605_epoch_8_freq_40_Hz.csv**: the 40 Hz recording,
  10x the row count of the 4 Hz file the script's comment lists as
  default.
- **LIB_RATIO=0.5**: a 50/50 library vs. held-out split (down from the
  0.7 the header comment above this line still describes -- that
  comment is now stale/out of sync with the value; the standalone
  `# default: 0.5` comment on the same line is the current one).
- **E/tau search**: E swept 3..16 (step 2), tau swept -15..0 (step 2) --
  a wide grid of GMN node-generator candidates, each evaluated with a
  full free-running `Generate()` call.
- **SMAP_SEARCH_MODE=coarse_retune**: the cheaper of the two search
  modes (per-node theta re-tuned with a 5-value grid at each E/tau
  candidate, rather than a full 15-value joint theta sweep, ~15x cheaper).

Net effect: this is a large, expensive run (full channel set, no MDE
pre-filtering, wide E/tau grid) sized for the 2-day/48G/32-core budget
requested at the top of the sbatch script, not a quick test.

No previous job snapshot found -- this is the first tracked run, or all prior jobs predate this tracking.

---

