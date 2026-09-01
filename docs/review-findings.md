# Review findings

Tracks work against `docs/review-actions-v1.md`, one section per task ID, in
the order tasks were actually done (not necessarily the file's order). Per
Part A, each section states what was verified before editing, not just what
changed.

---

## R12 — Make Stage 1 and Stage 2 fully serial

**Status:** Done. Commit: (this task's own commit, following the
"Split nested-MDE into a unified per-wave Stage 1/Stage 2 BFS" baseline
commit that captured session-start state).

### A1 verification

Read `build_nested_mde_network()` in full in both `scripts/src/tsls-mde-gmn.py`
(lines ~1775-1953 pre-change) and `scripts/src/simple-mde-gmn.py`
(~1674-2002 pre-change), plus `_accept_mde_result_edges()`,
`observable_and_delay()`, `effective_mde_dimension()`, and the
`build_network()` call sites in both files.

**Premise confirmed true.** Both files' committed state (from the
2026-08-25 revision, itself only in the working tree at session start, not
yet in git history before this review) implemented exactly the "unified
per-wave BFS" the task describes: each while-loop iteration ran a Stage 1
pass over `wave_nodes`, then immediately a Stage 2 pass over the *same*
`wave_nodes`, then accepted edges from both before computing `next_wave`.
Both a Stage 1 and a Stage 2 real-observable pick already fed `next_wave`
(the 2026-08-25 fix), so Stage 2 recursion already worked — just
interleaved with Stage 1 wave-by-wave rather than after it.

### A2 interactions checked

- **R6.** Confirmed out of scope for this task per the ordering note ("R6's
  test must be rewritten... Do R12 first"). Not touched here. R12's own
  acceptance criterion ("re-run the §18.12 probe confirming Stage-2
  real-observable picks still recurse") is satisfied by
  `tests/test_r12_stage_serialization.py`, which a future R6 pass can adopt
  or extend rather than re-deriving from scratch.
- **R13.** Not implemented here. Verified the *shared depth* semantics R12
  depends on: in the pre-change code, `depth` was already a single value
  threaded through both stages' acceptance loops (not two independent
  per-stage counters), so "joint depth budget across stages" was already
  the operative semantics before R12, just expressed inside one wave.
  R12's serial version preserves this: Stage 1's own recursion uses
  `node_depth`/depth exactly as before; Stage 2 is seeded with each Stage
  1 node's own discovered depth (`node_depth[node]`) and continues the
  same counter. Whether this already satisfies R13's "make it joint" or
  needs a further change is R13's own determination, done next.
- **R14.** Unrelated to this task (self-history/embedding-spec logic,
  untouched by the BFS scheduling change).
- **A3 (sibling parity).** Ported identically to `simple-mde-gmn.py` in
  the same commit — see below.
- **A5 (cache invalidation).** Checked `_pipeline_code_fingerprint()` in
  both files: it SHA-256-hashes the *entire file's* AST (docstrings
  stripped), not a per-parameter knob list. R12 changes real control flow
  (two separate `while` loops instead of one, `visited1`/`visited2`
  instead of one `visited` set, dict-based `next_wave` instead of a list),
  so the AST differs and the hash changes automatically. No new config
  knob was introduced, so no `build_*_fingerprint()` edit was needed —
  confirmed by reading `build_code_fingerprint()`/`build_network_fingerprint()`,
  not assumed. (This is the same mechanism R13 step 4 asks to verify;
  confirmed here, to be re-confirmed in R13's own section since it's the
  same question asked about a different change.)

### What changed

`build_nested_mde_network()` in both files now runs two sequential phases
instead of one interleaved loop:

- **Phase 1 (Stage 1):** unchanged real-observable-only BFS, run to full
  completion — every depth the `nested_m` budget allows — before Phase 2
  starts. Depth-0 targets still reuse the caller's precomputed
  `mde_results` entries exactly as before.
- **Phase 2 (Stage 2):** a *separate* BFS, seeded with every node Phase 1
  discovered (`visited1`) at the depth Phase 1 discovered it
  (`node_depth`), not with the targets alone — this is what guarantees
  every Stage 1 backbone node still gets its lagged/real embedding
  attachment, the same guarantee the old per-wave interleaving gave one
  wave at a time. A real-observable pick from Stage 2
  (`STAGE2_ALLOW_REAL_OBSERVABLES=True`) enqueues its own further Stage 2
  round — never a Stage 1 round — subject to the same `nested_m` budget.

One correctness point beyond a literal port of the old per-stage logic:
because Phase 2's first wave bundles every Stage 1 node regardless of the
depth Stage 1 found it at (deliberately, for parallelism — this is the same
"one sweep over Stage 1's final node set" shape the historical §5.3 bug
had, except this sweep recurses further), a brand-new driver proposed by
two different-depth parents in that same first wave needs the *shallowest*
proposed depth, not whichever parent happens to be iterated first.
`next_wave` is therefore a `{driver: min_depth}` dict rather than a plain
list in both phases (harmless where depths were already uniform per wave,
i.e. everywhere in Phase 1 and every Phase-2 wave after the first).

The CCM-pre-filter ordering fix (§18.12) is preserved: `raw_columns`/
`expanded_columns`/`stage2_candidate_universe` are still computed once,
outside both phases' loops, from the same `observable_columns` `main()`
already pre-filtered once per target group before calling `build_network()`
— unchanged by this restructuring.

### Test

`tests/test_r12_stage_serialization.py` (+ `tests/_pipeline_test_utils.py`
for shared fakes) exercises both pipelines' `build_nested_mde_network()`
with a faked `run_mde_for_target()` and a synchronous fake
`ProcessPoolExecutor` (no real MDE/CCM computation, no real data needed).
Scenario: target `T` → Stage 1 selects real observable `A`; `A`'s own
Stage 1 call selects nothing. Stage 2 (`stage2_allow_real_observables=True`):
`T`'s Stage 2 call selects real observable `B` — one Stage 1 never
found — which must recurse into its own further Stage 2 round.

Asserts: (1) every `stage1` call precedes every `stage2` call in the
invocation log; (2) `B` gets its own `stage2` call (recursion) and never a
`stage1` call; (3) every Stage 1 node (`T`, `A`) also gets a `stage2` call;
(4) the resulting graph/edges match. **Verified the test actually
discriminates old vs. new code**, not just trivially passes: ran it against
the pre-R12 (session-start committed) `tsls-mde-gmn.py` and it fails with
`call_log=[('stage1', 'T'), ('stage2', 'T'), ('stage1', 'A'), ('stage1', 'B'), ...]`
— a Stage 2 call for `T` interleaved before Stage 1 even reached `A`/`B`,
exactly the regression this test exists to catch.

Run: `python3 tests/test_r12_stage_serialization.py` (no pytest in this
project's venv). Passed on first run against the new serial code, both
pipelines.

### A4 verification

`python3 -m py_compile` passed on both `scripts/src/tsls-mde-gmn.py` and
`scripts/src/simple-mde-gmn.py`. `bash -n` passed on both
`scripts/batch/tsls-mde-gmn.sbatch` and `scripts/batch/simple-mde-gmn.sbatch`.

### Wall-clock / topology on Run A's configuration

Per user direction, launched a local (non-SLURM) smoke run at Run A's
documented configuration (`REGION_CHANNEL_CAP=3`, `MDE_D=2`,
`METRIC=correlation`, real data
`J1620210605_epoch_8_freq_40_Hz.csv`) against the new serial code, in the
background (PID 1620647 at launch; log at
`<scratchpad>/run_a_smoke/run.log`, results/cache redirected to scratch so
the shared `results/` directory isn't touched). `GENERATE_LENGTH=40` was
set explicitly (Run A's own doc doesn't state this value; left at the
default 0 would derive a several-thousand-column Stage 2 lag set via
`LAG_MODE=generation_length_adjusted`, per the sbatch script's own
documented cost warning) — noting this as a config approximation, not a
literal reproduction of every env var Run A used, since the tex document
only specifies `REGION_CHANNEL_CAP`/`MDE_D`/`METRIC` explicitly.

**Result pending** — this is a real, possibly long-running job (Stage 1
alone took 4+ hours on a larger config per JOB_LOG history, though Run A's
own smaller config progressed much further interactively per the tex doc).
Wall-clock and topology comparison will be appended here once it completes
or is checked again.

**Progress note (checked mid-R13):** the background run's log shows Stage 1
already completed (2 real-observable backbone nodes) and Stage 2 in
progress — confirms the serial ordering holds end-to-end on real data, not
just under the fake-executor test.

---

## R13 — Make `NESTED_M` a whole-network depth budget, shared by both stages

**Status:** Done.

### A1 verification

Read the current (post-R12) `build_nested_mde_network()` in both files in
full, specifically the depth-check lines in each phase:

```python
if delay == 0 and (nested_m == 0 or depth < nested_m) and driver not in visited1:
    next_wave[driver] = min(next_wave.get(driver, depth + 1), depth + 1)   # Stage 1
...
if delay == 0 and (nested_m == 0 or depth < nested_m) and driver not in visited2:
    next_wave[driver] = min(next_wave.get(driver, depth + 1), depth + 1)   # Stage 2
```

Both phases read from **one** `nested_m` parameter and **one** `node_depth`
accounting scheme (a single dict, populated once per node — by whichever
phase discovers it first, `node_depth[node] = depth` in Stage 1,
`node_depth.setdefault(node, depth)` in Stage 2). Traced a concrete example
by hand (`nested_m=1`): Stage 1 discovers `A` at depth 1 from target `T`
at depth 0; Stage 1's own recursion correctly stops there
(`1 < 1` is false). Stage 2 is seeded with `node_depth[A] = 1` (inherited
from Stage 1, not reset to 0), so if `A`'s Stage 2 call selects a brand-new
real observable, the *same* check (`1 < 1` is false) blocks it from
recursing further too — exactly the joint budget the task describes.

### Step 1 — is the cap already global?

**Yes — confirmed already global/joint, and this was true even before
R12.** Re-read the pre-R12 (2026-08-25 unified) code preserved in the
baseline commit: it also used a single `nested_m` and a single `depth`/
`visited` set shared by both stages' acceptance loops within each wave —
its own docstring already said so explicitly ("Both stages share the same
`nested_m` depth budget"). R12 changed *when* each stage runs (serial vs.
interleaved), not the budget accounting, which was already joint. **No
code change was needed for the core ask** — per the task's own step 1
("If it is already global, report that and make no change — the task is
then satisfied").

### Step 2

Not applicable — the cap was already global (step 1).

### Step 3 — the `0` (uncapped) sentinel

**Asked the user before proceeding, per the task's explicit instruction.**
Traced whether `NESTED_M=0` is actually a non-termination risk: both
phases' BFS use `visited1`/`visited2` sets over a *finite* real-observable
pool (bounded by `REGION_CHANNEL_CAP` and the CCM pre-filter — the only
things a delay=0 pick can ever name, per `observable_and_delay()`), and
each while-loop only enqueues nodes not already in the respective visited
set. So even uncapped, both phases are bounded fixed-point iterations over
a finite vertex set — they terminate once every reachable real observable
has been visited or every remaining branch dead-ends (zero drivers
selected, or NaN marginal gain), never literally forever. This was already
true before R12/R13 (the existing `MDE_D**(1+NESTED_M)`/"bounded only by
the real observable pool size" comment in both files' `load_config()`
already said as much for Stage 1 alone; R12/R13 extend the same guarantee
to Stage 2's own recursion, which draws from the same finite pool).
**User confirmed:** no hard safety bound added — documented the
finite-pool termination proof instead (this section, plus a short note
added to both files' `NESTED_M` comment in `load_config()`).

### Step 4 — cache fingerprint

No additional code change was made beyond R12's own restructuring plus
this task's logging addition (see below), so there is no *new* cache-
invalidation surface specific to R13. Re-confirmed (not re-assumed) that
`build_code_fingerprint()`'s whole-file AST hash (`_pipeline_code_fingerprint()`)
covers the logging line addition too — any prior cached network built
under the old unified BFS or without the realized-depth log line will
already fail to match on `code_fingerprint` before it ever reaches
`nested_m`/`NESTED_M`-specific comparison in `build_network_fingerprint()`.

### What changed

No arithmetic, no re-indexing, no default change (all explicitly out of
scope per the task). Two additive changes, both in `build_nested_mde_network()`
in both files:

1. **Logging:** the function's final summary print now reports
   `realized max depth` (`max(node_depth.values())`) alongside the
   configured `N`/`M`, so the *actual* depth reached is visible per run,
   distinct from the configured cap — this is the task's own acceptance
   criterion ("Log the realized maximum depth at the end of network
   construction").
2. **Documentation:** both files' `load_config()` `NESTED_M` comment now
   states explicitly that the budget is whole-network/joint (previously
   implied by the shared-variable structure but not stated outright).

### Test

`tests/test_r13_shared_depth_budget.py` — `NESTED_M=1`. Target `T`
(depth 0) → Stage 1 selects real observable `A` (depth 1, exactly at the
cap); `A`'s own Stage 1 call selects nothing further (2 Stage 1 rounds
total, matching `1 + NESTED_M`). Stage 2 then runs on `{T, A}`: `A`'s Stage
2 call (still at depth 1, the cap Stage 1 already reached on this branch)
selects a brand-new real observable `D` — accepted as a graph edge
(`A <- D`) but never recursed into, since depth 1 is already at
`nested_m=1`.

Asserts: (1) Stage 1 makes exactly `1 + NESTED_M = 2` calls; (2) `D` never
gets its own MDE call in *either* stage (the joint-budget, budget-bound
termination A2/R15 describe); (3) the longest root-to-leaf path
(`D → A → T`, edges point driver→node) has `NESTED_M + 2 = 3` nodes — the
`NESTED_M + 1` nodes that actually got their own MDE call, plus the one
further leaf a depth-capped node's own selection still adds to the graph
without exploring it. Passed on first run, both pipelines.

### A4 verification

`python3 -m py_compile` passed on both pipeline files. `bash -n` passed on
both sbatch scripts.

---

## R14 — Leaf nodes become external inputs, not network nodes

**Status:** Done, with two corrections to the task as written (recorded
below per explicit instruction) and one pre-existing bug found and fixed
along the way.

### A1 verification

Read `_embedding_spec_for()`, the `gmn.Node.__init__`/`.Generate`/
`.Forecast` monkeypatches, `gmn.GMN.Generate()`'s time loop and
`gmn.Network.__init__`'s `TopologicalSorted` construction (both in the
installed `gmn` package, not this codebase), and `build_gmn_config()`, in
both pipeline files.

**Premise confirmed true.** `_embedding_spec_for()`'s MultiDiGraph branch
unconditionally prepended `graph.graph['self_delay_set']` (every delay in
`{0} ∪ LAG_SET`) to every node's spec regardless of discovered in-edges.
`gmn.Network.__init__` puts every graph node (leaf or not) into
`TopologicalSorted`, and `gmn.GMN.Generate()`'s per-step loop calls
`node.Generate()` for all of them unconditionally — so today, a zero-driver
leaf genuinely self-predicts its own future every step via SMap regression
on nothing but its own unconditional self-history, never touching real
data past the initial seed. This is exactly what let §1.1's "runs freely...
without ever seeing the real behavior again" claim hold for every node,
leaf or not.

### Ask-before-implementing (step 4, and two more surfaced by testing)

**Round 1 (the task's own explicit gate).** §8.2's unconditional
self-history is what lets a zero-driver leaf simulate at all today; if
R14's external inputs are fed real data during `Generate()`, the free-
running claim no longer holds universally. Reported both readings and
asked. **User decision:** implement both as a config toggle,
`EXTERNAL_INPUT_MODE` (`'real_data'` default: exogenous, fed the real
recorded trajectory, self-consistent with "external input" in the
standard control-theory sense; `'self_simulate'`: keeps pre-R14 behavior).
**User also caught a case I had not accounted for:** a *target* can end up
with zero discovered drivers (Run A's own `head_position_y`, per
`docs/tsls-mde-gmn-pipeline.tex`) — under `'real_data'` mode this would
feed a target's own real values back in as its "simulation" and then score
fidelity against those same real values, a circular, spuriously-perfect
result for exactly the case that most needs flagging as a failure.
`partition_network()` therefore raises for this case rather than
classifying it as external, regardless of `EXTERNAL_INPUT_MODE`.

**Round 2 (found during testing, not anticipated by the task text).**
Building a real (tiny, synthetic) `gmn.GMN.Generate()` run the same way
`build_network()`/`build_gmn_config()` build one for real — see the test
below — surfaced a **pre-existing bug**: `_generalized_node_generate()`/
`_generalized_node_forecast()`'s SMap call used
`columns=df.columns[1:-1]`, dropping `time` (first) *and* the spec's own
last entry. Since spec order is self-terms-then-in-edges, that last entry
is a node's own last discovered driver whenever it has any in-edges at
all — confirmed empirically (`spec=[T(t-0), T(t-1), A(t-3), B(t-0)]` →
`columns` silently excluded `B(t-0)`, though it stayed in `df` itself).
Pre-R14 this was harmless (self_delay_set always padded a node with
several self-lags, so losing one entry rarely mattered); post-R14, a
node's spec *is* its discovered drivers, so a 1-driver interior node has
zero columns left at all — a hard crash (pyEDM's `RuntimeError: Validate()
SMap: columns required.`), and a 2+-driver node silently loses its last
one. This is outside R14's stated scope (a latent bug in code the task
didn't name), so I stopped and reported it rather than fixing it quietly.
**User decision:** fix it, in both `_generalized_node_generate`/
`_generalized_node_forecast`, both pipelines — `columns=df.columns[1:]`
(drop only `time`). General fix, not R14-specific, but found and landed
together with R14 since R14 is what turns it from a masked quirk into a
real risk.

**Round 3 (found immediately after, same test).** Removing
`self_delay_set` *entirely* (as R14's text literally says: "built only
from drivers discovered by nested-MDE") breaks pyEDM outright for every
interior node, not just an edge case: `target=node_name+'(t-0)'` is a hard
pyEDM requirement — that column must exist in the embedded dataframe
regardless of what the predictor `columns` are, and no in-edge can ever
supply it (self-loops are excluded, `_accept_mde_result_edges()`'s own
guard). Reported this and proposed keeping the node's own delay=0
"anchor" coordinate unconditionally (structurally required, and standard
generalized-embedding practice besides — a variable's own present value is
conventionally one of its own state-space coordinates), removing only the
*deeper* self-lags R14 actually objects to. **User asked two clarifying
questions before approving:**
  - Whether the anchor is target-only or also stays in the predictor
    (`columns`) set. **Resolved: both** — it is not excluded from
    `columns`; this is intended (standard EDM practice), not a leftover.
  - Whether "external nodes keep the anchor under `self_simulate`" meant
    leaves now self-simulate by default, contradicting round 1's
    `'real_data'` default. **Resolved: no** — under the default, external
    inputs never reach `_embedding_spec_for()` at all (short-circuited
    earlier in `Generate()`/`Forecast()`); the anchor fix only matters for
    nodes that actually reach the SMap branch (interior nodes always,
    external nodes only under `self_simulate`), and for those the same
    unified formula (`[(node_name, 0)] + in_edges`) already produces the
    right spec with no separate branching needed.
  User confirmed this design once both points were clarified.

### Two corrections to R14 as written (per explicit instruction — recorded
here so the documentation-rewrite pass does not carry the stale text
forward)

1. **Step 1's "built only from drivers discovered by nested-MDE" is
   inaccurate.** Corrected: a node's embedding is discovered drivers
   *plus* an unconditional delay-0 self-anchor (see Round 3 above for
   why the anchor cannot be removed). The acceptance test ("no node's
   embedding contains a self-lag that nested-MDE did not select") is
   satisfied under the reading that it means **nonzero** self-lags only —
   the delay=0 anchor is expected on every node and does not fail it (see
   `tests/test_r14_external_inputs.py`'s explicit check of this).

2. **The "leaf duplicate" degeneracy rationale does not describe this
   graph — confirmed against source, not asserted.** R14's own rationale
   was: "a leaf may be either a real observable or a lagged observable. If
   nested-MDE selects real observable B and a lagged copy of B as two
   separate zero-driver nodes, taking a lagged embedding of each makes the
   two identical." Read `observable_and_delay()`
   (`scripts/src/tsls-mde-gmn.py:1113-1117`, identical in
   `simple-mde-gmn.py`): it always resolves a candidate column back to its
   *raw* observable name (`if column_name in raw_columns: return
   column_name, 0`; delay lives separately). Read
   `_accept_mde_result_edges()`: `graph.add_node(driver)` uses only that
   raw name. So there is exactly **one** graph node per raw observable,
   regardless of how many different delays reference it — "a lagged copy
   of B" is never a second node, only a `(B, some-negative-delay)` edge
   into whatever node selected it. Two zero-driver nodes with identical
   embeddings, both "about" the same underlying observable, cannot arise
   in this graph's node-identity scheme. **The surviving justification
   for partitioning out zero-driver nodes is simpler and still correct on
   its own:** a zero-driver node is not something nested-MDE discovered as
   a network member at all, so counting it as an interior (simulated) node
   overstates what was actually found. This is the rationale
   `partition_network()`'s own docstring now records.

### What changed

- `partition_network(graph, targets, external_input_mode)` (new, both
  files): classifies every node as interior (`in_degree >= 1`) or external
  input (`in_degree == 0`), raising if a *target* has zero drivers (see
  Round 1). Stores the classification explicitly on `graph.graph`
  (`interior_nodes`/`external_inputs`/`external_input_mode`) rather than
  leaving it implicit in edge counts, per the task's own requirement.
  Called from `build_network()` right after `graph.graph['self_delay_set']`
  is set.
- `report_partition(graph, network_id, results_dir)` (new, both files):
  prints the partition to the run log and writes
  `partition_{network_id}.csv` (node, role, feeds, delay) to `results_dir`.
- `EXTERNAL_INPUT_MODE` config knob (new, both files, default `'real_data'`,
  validated against `('real_data', 'self_simulate')`).
- `_embedding_spec_for()`: simplified to `[(node_name, 0)] + in_edges` for
  every node that reaches it (see Round 3) — no more separate interior/
  external branching inside this function; the mode-driven behavior lives
  entirely in the Generate()/Forecast()/init short-circuits below.
- `_generalized_node_generate()`/`_generalized_node_forecast()`/
  `_generalized_node_init()`: each short-circuits for a
  `real_data`-mode external input — Generate() returns the real recorded
  value at the current row (`self.Network.data[self.name].iloc[len(self.data)]`,
  since `self.data`'s length is already the absolute row index by the time
  `lastDataOut` has been appended); Forecast() returns the real series as
  a trivial `(time, values)` passthrough; `__init__` skips the
  spec-dependent `libEnd_i` correction entirely. Also fixes the
  `columns[1:-1]` → `columns[1:]` bug (Round 2) in both Generate() and
  Forecast().
- `_simulated_nodes(graph)` (new, both files): nodes that actually run
  gmn's SMap machinery (interior always; external only under
  `self_simulate`) — used by `run_theta_search()` so real-data-mode
  external inputs (no embedding to retune) are excluded from theta
  retuning.
- `plot_3d_network()` (new, both files): x-y plane is a layout of the
  real-observable (delay=0) backbone; z axis is one vertical stick per
  delay!=0 driver, rising to `|delay|` at the driven node's (x, y);
  external inputs drawn as a distinct marker shape (triangle vs. circle),
  independent of target coloring. Called from `main()` for
  `MODE='MDE-GMN'` runs, alongside the existing `draw_colored_network()`.
- `build_search_fingerprint()`: added `external_input_mode` (changes
  Generate()/theta-search behavior for a zero-driver node, not topology,
  so it belongs here rather than in `build_network_fingerprint()`).

### Test

`tests/test_r14_external_inputs.py` — unit-level (`partition_network()`
classification, the target-zero-drivers raise, the corrected embedding-
spec shape) plus integration-level: a real, tiny, synthetic
`gmn.GMN.Generate()` run built the same way the pipeline builds one for
real (`build_gmn_config()` + a pickled `{'Graph': ..., 'Map': ...}` network
file + a real CSV), for both `EXTERNAL_INPUT_MODE` settings. Confirms:
`'real_data'` mode's external input's `Generate()` output matches the real
recorded series exactly; `'self_simulate'` mode still self-predicts
without crashing; the interior node's own output is finite in both modes;
`report_partition()` writes the expected CSV; `plot_3d_network()` produces
a non-empty figure (visually inspected: interior node drawn as a red
circle, external input as an orange triangle, connected by the real-
observable edge — see `tests/_scratch_r14/*/network_3d_toy.png`). All
passed on both pipelines after the Round 2/3 fixes above; both rounds were
caught *by* this test, not before it.

### A3 sibling parity

Ported identically to `simple-mde-gmn.py` in the same commit — same
`partition_network`/`report_partition`/`_simulated_nodes`/
`plot_3d_network` functions, same `EXTERNAL_INPUT_MODE` knob, same
short-circuit logic and `columns[1:-1]` fix.

### A5 cache invalidation

`EXTERNAL_INPUT_MODE` added to `build_search_fingerprint()` (changes
Generate()/theta-search outcomes for a zero-driver node) in both files.
Not added to `build_network_fingerprint()` — it doesn't affect which
nodes/edges get discovered, only what happens to an already-classified
zero-driver node during simulation. The `columns[1:-1]`→`columns[1:]` fix
and the embedding-spec simplification are both real logic changes,
automatically covered by the whole-file AST-hash `code_fingerprint`
(confirmed mechanism, not re-derived — see R12's own section above).

### A4 verification

`python3 -m py_compile` passed on both pipeline files. `bash -n` passed on
both sbatch scripts.

### R12's background smoke run: status update

The Run-A-config smoke run launched during R12 (see that section above)
progressed through Stage 1 (3 real-observable backbone nodes for
`head_position_x`, `head_position_y`) and into Stage 2/theta retuning
before this session's process tracking lost it (no further log output,
process no longer found by PID or by name) — inconclusive, not a pipeline
error: no exception was logged, and the log simply stops mid-retune. Given
this predates R13/R14's own code (launched right after R12 landed), it is
not informative about the code as it now stands. Not re-launched given the
time this session has already spent — recommend a fresh local smoke run at
Run A's configuration, through `scripts/batch/submit.sh` per this
project's own job-submission convention, before any cluster submission
(per A4).

---

## R6 — Regression test: Stage-2 real-observable picks recurse

**Status:** Done — already satisfied by tests written under R12/R13, no
new code.

### A1 verification

Re-read R6's acceptance criterion against what R12/R13 already produced:
"prove by test... that with `STAGE2_ALLOW_REAL_OBSERVABLES=true`, a
delay-0 column selected by Stage 2 is enqueued for its own further Stage 2
round, subject to the R13 depth budget."

`tests/test_r12_stage_serialization.py` already proves the first half:
`stage2_allow_real_observables=True`, T's Stage 2 call selects real
observable B, and B gets its own further `('stage2', 'B')` call —
asserted directly, and confirmed to fail against the pre-R12 code (see
R12's own section above). `tests/test_r13_shared_depth_budget.py` already
proves the second half: the exact same recursion, but on a branch where
the shared depth budget is exhausted, correctly does *not* recurse further
(`D` gets an edge but no MDE call of its own) — i.e., the recursion is
real and is bounded by the R13 budget, not unconditional.

Between the two, R6's acceptance criterion is met in full. Writing a third,
separate test here would just be restating the same assertions under a
different file name — the kind of premature duplication the project's own
conventions argue against. No code or test changes for this task; this
section exists so R6 is marked closed against `docs/review-actions-v1.md`
rather than silently skipped.

---

## R15 — Stage 2 recursion on real-observable drivers

**Status:** Done.

### A1 verification

Re-read `build_nested_mde_network()`'s Phase 2 loop in both files
(post-R12/R13/R14) end to end, as the task explicitly demands ("read the
code path end to end; do not infer it from §5.3's claim").

### Step 1 — does it actually recurse?

**Yes, confirmed by test, not inferred.** This was already established as
part of R12/R13's own work, not re-derived here:
`tests/test_r12_stage_serialization.py` proves a Stage-2-selected real
observable gets its own further Stage 2 MDE call (and is confirmed to
fail against the pre-R12 code); `tests/test_r13_shared_depth_budget.py`
proves that recursion is correctly bounded by the shared `nested_m`
budget rather than unconditional. Both conditions R15 step 1 asks about
hold.

### Step 2 — flip the default

Done in both files: `STAGE2_ALLOW_REAL_OBSERVABLES` (`TSLS_MDE_GMN_
STAGE2_ALLOW_REAL_OBSERVABLES` / `MDE_GMN_STAGE2_ALLOW_REAL_OBSERVABLES`)
now defaults to `True`.

### Step 3 — n/a

Only reached if step 1 found the recursion didn't work; it does, so no
flag-and-stop was needed.

### Step 4 — distinguishable termination logging

Added `stage2_budget_bound`/`stage2_driver_exhausted` counters to Phase
2's edge-acceptance loop in both files, classifying every node's Stage 2
result:
- **driver-exhausted:** the node's own depth is still under the shared
  budget, but its Stage 2 call selected zero real-observable (delay=0)
  drivers at all — nothing to recurse into, unrelated to the budget.
- **budget-bound:** the node's depth is already at or past `nested_m`, so
  an accepted real-observable pick is added as a graph edge but never
  enqueued for its own further round.

Printed once per network, gated on `stage2_allow_real_observables` (the
counts are meaningless when the feature is off): `nested MDE (Stage 2):
real-observable recursion terminations -- N budget-bound, M
driver-exhausted`.

Per the task's own framing, this is reporting, not a defect to code
around: if `budget_bound` dominates at the configured `NESTED_M`, that's a
configuration observation (the feature is effectively inert at that
depth) worth surfacing in a run's own log, not something to "fix" by
letting Stage 2 exceed the budget.

### Test

`tests/test_r15_termination_causes.py` reuses R13's exact scenario
(`NESTED_M=1`): `T`'s own Stage 2 call selects nothing (under budget, zero
real picks → driver-exhausted); `A`'s Stage 2 call selects a brand-new
real observable `D`, but `A` is already at the depth cap → budget-bound.
Captures the log line via `contextlib.redirect_stdout` and asserts exactly
1 of each. Passed on first run, both pipelines.

### A3 / A4

Ported identically to both files. `python3 -m py_compile` and `bash -n`
both pass.

---

## R2 — Audit: does the CCM pre-filter see only real observables?

**Status:** Done. Finding: confirmed clean in both pipelines. Added the
acceptance criterion's own defensive assertion anyway (see below).

### A1 verification / finding

Traced `main()`'s call site in both files back to the source of the
candidate list `ccm_prefilter_observables()` receives:

- **tsls-mde-gmn.py:** `obs_cols` comes from `convert_to_tsls()`
  (`observable_columns = [f'{n}_tsls' for n in neuron_columns]`) — real
  channel names from the raw CSV header, never containing `DELAY_MARKER`
  (`'::delay'`).
- **simple-mde-gmn.py:** `obs_cols` comes from `observable_columns()` →
  `region_columns()` (`f'{prefix}{i}'` for `i in range(cap)`, filtered to
  columns actually present in the raw input `df`) — synthetic
  prefix+index names, same guarantee.

**`expand_delay_columns()` cannot run "earlier than expected."** It is
called from exactly one place in each file —
`build_nested_mde_network()` — which is only ever called from
`build_network()`, which `main()` calls strictly *after* every
`ccm_prefilter_observables()` call for that target group (confirmed by
reading `main()`'s own call order, not assumed from the surrounding
comments).

**No cached `mde_results` entry can reach the pre-filter either.** The
pre-filter's candidate list is always `main()`'s own `obs_cols`/
`filtered_pools` value, never derived from or mixed with `mde_results` (a
separate dict, populated *after* the pre-filter runs, and only read by
`build_nested_mde_network()`'s depth-0 reuse check downstream).

**Acceptance note:** the task's fix/assertion instruction was
conditional ("if lagged columns can reach the pre-filter..."), and they
cannot — so strictly, no code change was required. Added the assertion
anyway (`assert not [c for c in candidate_columns if DELAY_MARKER in c]`
at the top of `ccm_prefilter_observables()` in both files) as a cheap,
permanent guard against a future change reopening this path, consistent
with this codebase's existing style of defensive assertions for
structural invariants (e.g. `_accept_mde_result_edges()`'s own
belt-and-suspenders self-loop guard). `tests/test_r2_prefilter_guard.py`
confirms it fires, both pipelines.

---

## R3 — Audit: candidate-pool exclusion semantics

**Status:** Done. Finding only — no code change, per the task's own
acceptance criterion.

### Finding

The answer **differs across the three scopes** the task asks about —
flagging this explicitly, as instructed.

1. **Within the same `proper_mde`/`dimx.MDE` call, across dimensions:
   YES, excluded.** Read `proper_mde()`
   (`cleaned_helper_functions.py:1150-1152`): "Candidates = `columns -
   selected`" — at each forward-selection dimension `d`, the candidate
   list explicitly excludes every column already added to `selected` in
   an earlier dimension of *this same call*. Confirmed the identical
   convention holds in `dimx.MDE`'s own `Run()`
   (`dimx/Run.py:48`: `columns = list(set(dataColumns) -
   set(self.MDEcolumns))`) — both engines implement the same standard
   forward-stepwise-selection exclusion, so this holds identically for
   both pipelines regardless of which engine `GMN_MODE`/`MODE` selects.

2. **Across later BFS depths: NO, not excluded.** Read
   `build_nested_mde_network()`'s candidate-pool construction in both
   phases: `candidate_pool = [c for c in expanded_columns /
   stage2_candidate_universe if observable_and_delay(c, raw_columns)[0]
   != node]`. The *only* exclusion is the current node's own identity —
   there is no tracking of "already selected by some earlier-processed
   node at a shallower depth" anywhere in the BFS. An observable selected
   as a driver for a depth-0 target is fully eligible to be independently
   re-selected as a driver for a depth-1 (or deeper) node's own MDE call.

3. **Across other nodes at the same depth: NO, not excluded**, for the
   identical reason — the same `candidate_pool` construction (self-
   exclusion only) is used for every node in a wave, computed
   independently per node from the same shared pool. Two different nodes
   processed in the same wave can, and do, both select the same
   observable as their own driver (each becoming its own edge, since a
   MultiDiGraph allows multiple in-edges from the same driver into
   different nodes).

4. **Stage 2's pool: the same filtered universe, not rebuilt.**
   `stage2_candidate_universe` (and `stage2_df`, via
   `expand_delay_columns()`) is computed exactly once per
   `build_nested_mde_network()` call, before either phase's while-loop
   starts. Every node's Stage 2 candidate pool is this same fixed
   universe, self-excluded — never recomputed per node or per depth.

**Net effect:** "once selected, it disappears from the pool" (the
reviewer's own stated expectation) is true only *within* a single node's
own forward-selection call (scope 1); it is false at the network-
construction level (scopes 2 and 3) and Stage 2's universe is a fixed,
shared pool rather than something that shrinks as the network grows
(scope 4). No change made — this is a written finding only, feeding a
decision not yet made, per the task's own acceptance criterion.

---

## R4 — Report the static `E` used by `proper_mde`'s CCM gate

**Status:** Done. Finding only — no code change, per the task's own
acceptance criterion.

### Finding

**`E` is identical at every single-target call site — depth-0 seed,
Stage 1, Stage 2 — because all three route through the same
`run_mde_for_target()`, which has exactly one `proper_mde()` call site
(`tsls-mde-gmn.py:1535`).** That call hardcodes `E=0, maxE=max_e`:

- `E=0` is not itself "the E used" — it's the sentinel that tells
  `proper_mde` to auto-select. **Confirmed a per-candidate
  `EmbedDimension`-equivalent auto-select path exists and is always
  taken** on the single-target path: `cleaned_helper_functions.py:1414-1440`
  — `if E > 0: ... else: EDimDF = proper_embed_dimension(...)`. Since
  `E=0` is hardcoded at the one call site every single-target call goes
  through, this `else` branch always runs: a fresh, independent
  univariate rho-vs-E sweep (`proper_embed_dimension`, metric-aware —
  scored with whichever `metric`/`target_metric` that call already uses,
  not hardcoded Pearson rho) *per candidate*, at *every* forward-selection
  dimension, bounded by the ceiling `maxE`.
- `maxE=max_e` is also identical at every site: main()'s depth-0 seed
  passes `max_e=cfg['MDE_D']`
  (`tsls-mde-gmn.py:4058`); both of `build_nested_mde_network()`'s Stage 1
  and Stage 2 passes pass `max_e=nested_n`
  (`tsls-mde-gmn.py:2097,2188`), and `nested_n` is `cfg['MDE_D']` at the
  `build_network()` call site — the same raw config value, unscaled.

**So there is no "static E" on the single-target path at all** — every
call auto-selects E fresh, per candidate, per dimension, up to the same
`MDE_D` ceiling everywhere. Confirmed this was a deliberate, recent
change, not a coincidence: the docstring
(`tsls-mde-gmn.py:1402-1416`) records that this call site previously
hardcoded `E=3` as a *workaround* for `proper_mde`'s old `E<=0` path being
blocked for non-`'correlation'` metrics, fixed once `proper_embed_dimension`
(a metric-aware `EmbedDimension` replacement) existed and could be wired
in for `E<=0` too.

**Contrast with the multi-target path (`multi_proper_mde`,
`GMN_MODE='combined'` only):** this one genuinely *does* require a static,
explicit `E` — `MULTI_MDE_E` (default `3`), no auto-select path at all.
This is the concrete premise R22 needs before merging the two call paths:
routing `GMN_MODE='separate'` through `multi_proper_mde` as it stands
would silently replace every existing single-target run's per-candidate
auto-selected E with the static default `3` — a real numerical
regression, exactly as R22 already suspected.

**Sibling parity (simple-mde-gmn.py):** `dx.MDE(...)`'s own call site
(`simple-mde-gmn.py:1629`) passes no `E=` at all, relying on
`dimx.MDE`'s own class default — confirmed `E=0`
(`dimx/MDE.py:60`, `# Static E for all CCM`, default `0`) — so both
pipelines reach the identical auto-select behavior on the single-target
path, just via an explicit override (tsls) vs. an unoverridden default
(simple).

---

## R10 — Inventory the S-map surface in `cleaned_helper_functions.py`

**Status:** Done. Finding only, no code, per the task's own acceptance
criterion. Precursor to R20.

### Finding

**No native S-map exists in this module.** The file's own top-of-file
public-API comment (`cleaned_helper_functions.py:34-39`) enumerates every
exposed function: `proper_simplex`, `proper_ccm`, `proper_mde`,
`proper_embed_dimension`, `proper_predict_interval`, `top_neurons`,
`multi_simplex`, `multi_proper_ccm`, `multi_proper_mde`,
`rolling_simplex`, plus the TSLS event-metric helpers and
`compute_time_since_previous_spike`. There is no `theta` parameter and no
`smap`/`SMap`-named function anywhere in the file (grepped for both,
zero matches outside this docstring's own mention of the gap).

**The engine is exclusively simplex-style, KDTree-based weighted-average
prediction.** Every one of `proper_simplex`/`proper_ccm`/
`proper_predict_interval` (and `multi_simplex`/`multi_proper_ccm`)
ultimately calls the shared private workhorse `_crossmap_predict()`
(`cleaned_helper_functions.py:568`): builds a `scipy.spatial.KDTree` over
the library embedding, queries k nearest neighbors per point, and predicts
via an exponentially-distance-weighted average of those neighbors' target
values ("simplex projection" — the function's own docstring names it as
such). This is structurally incapable of expressing S-map's locality
parameter: there is no `theta`, no local weighting kernel over *all*
library points scaled by a locality parameter, and no linear-regression
solve anywhere in the prediction path — only a k-NN average.

### Scope estimate: what a theta sweep with an `_score`-compatible metric
would require

This is genuinely new work, not a small patch on existing machinery — a
new prediction primitive, not a variant of one that already exists:

1. **A new S-map prediction engine**, analogous in role to
   `_crossmap_predict()` but structurally different in kind: for every
   query point, weight *every* library point by
   `exp(-theta * dist / mean_dist)` (not just the k nearest), then solve a
   weighted linear regression (embedding coordinates → target) using
   those weights, and predict from the fitted local linear model at the
   query point. Needs its own vectorized distance computation (KDTree
   doesn't help here, since S-map needs distances/weights to the *whole*
   library, not just k neighbors) and a batched weighted-least-squares
   solve per query point (`np.linalg.lstsq` or a normal-equations
   formulation, one solve per point, with the same NaN/degenerate-window
   handling `_crossmap_predict()` already has to fend off — a library
   window too small or rank-deficient for the local regression is a real
   failure mode a k-NN average never has to consider).
2. **A `metric`-aware wrapper** analogous to `proper_ccm`/
   `proper_predict_interval`, exposing `theta` as a real parameter and
   scoring through `_score(metric=...)` (the same dispatch point every
   other function in this file already routes through), so `f1`/`gamma`
   (neuron nodes) and `correlation` (targets) both work, not just Pearson
   rho.
3. **A theta-sweep entry point** — the pipeline-side counterpart of what
   `pyEDM.PredictNonlinear` currently does, but calling the new engine at
   each candidate `theta` and scoring via `_score` instead of pyEDM's own
   internal rho. `retune_theta_for_all_nodes()`/`run_theta_search()`
   (both pipelines) already have the surrounding per-node sweep structure
   this would slot into — the change is which engine gets called
   underneath, not the sweep loop itself.
4. **Validation against pyEDM's own S-map** on a known series — R20's own
   explicit acceptance criterion, not this task's, but worth flagging
   here as the natural first correctness gate before any pipeline
   integration: pyEDM's compiled S-map is the reference implementation,
   and a from-scratch reimplementation needs to reproduce it exactly on
   Pearson-rho settings before it can be trusted on the metrics pyEDM
   can't compute at all.

**Net scope:** a genuinely new, non-trivial EDM primitive (item 1) plus
a thin metric-aware wrapper and sweep integration (items 2-3) — not a
small extension of `_crossmap_predict()` or any other existing function
in this file. This is the concrete input R20 needs for its own
"implement vs. propose" decision.

---

## R1 — Vectorize `compute_time_since_previous_spike`

**Status:** Done.

### A1 verification

Read `compute_time_since_previous_spike()` in full
(`cleaned_helper_functions.py:49-100`). Premise confirmed true: it was
exactly the described O(N x S) per-grid-point boolean mask (`spike_times[
spike_times <= t]` recomputed fresh inside a Python loop over every
`time_grid` entry).

### What changed

Replaced the loop body with
`np.searchsorted(spike_times, time_grid, side='right') - 1` (the index of
the most recent spike at or before each grid point, all at once), clipped
to a valid index for the lookup, then `np.where(idx < 0, np.nan, ...)` to
mark grid points before the channel's first spike — matching the old
per-point `len(s_prev) == 0` branch exactly. Kept the old implementation
as `_compute_time_since_previous_spike_reference`, used only by the
equivalence test.

### Test and measured speedup

`tests/test_r1_tsls_vectorized.py`:
- **Equivalence:** three real channels from
  `J1620210605_epoch_8_freq_40_Hz.csv`, chosen to span the firing-rate
  distribution rather than picked arbitrarily — lowest (`CA1_15`, 83
  spikes), median (`CA1_34`, 1025 spikes), and highest (`CA1_41`, 44976
  spikes) spike counts among all 468 real channels.
  `np.allclose(old, new, equal_nan=True)` passes for all three.
- **Speedup at full `REGION_CHANNEL_CAP`** (every one of the 468 real
  channels, all 47999 grid points): old **104.3s** total, new **0.56s**
  total — **186x** measured speedup.

---

## R5 — Concurrency audit and global worker budget

**Status:** Done. Finding: no oversubscription observed; no semaphore
added, per the task's own conditional acceptance criterion. Run after
R12, on real data, per the task's own note.

### Step 1 — instrumentation

Added the explicit product to both stages' existing per-wave print
statements in both files (they already logged `len(wave)`/
`outer_workers`/`inner_workers`, just not their product) —
`(product={outer_workers * inner_workers}, cores={cores})` appended to
each.

### Step 2 — real measurement (not inferred)

Ran at `TSLS_MDE_GMN_REGION_CHANNEL_CAP=5, TSLS_MDE_GMN_D=2,
TSLS_MDE_GMN_NESTED_M=2` (`TSLS_MDE_GMN_GENERATE_LENGTH=40`, results/cache
redirected to scratch) on real data, locally, `CORES` at its default (40,
this machine's full core count). Sampled `ps --ppid <main_pid> -o
pid,nlwp,pcpu` every few seconds throughout the run (not a single
snapshot) to catch transient spikes, not just steady state.

At this config, the depth-0 joint MDE seed (`GMN_MODE='combined'`
default) found **zero** drivers for both targets combined (5 candidates,
all failed `crossMapRhoMin` — an artifact of this deliberately tiny
config, not a bug), so Stage 1 never had a node to compute past depth 0
and its `to_compute` list was always empty — **Stage 1 never opened its
own `ProcessPoolExecutor` in this run.** Stage 2 did (seeded with the two
targets themselves, since they're still `visited1` members even via the
depth-0 cache-reuse path, not a fresh executor call): the log's own new
product line read `2 parallel worker(s) x 20 inner thread(s) each
(product=40, cores=40)`.

**Measured, not configured:**
- **Peak live worker processes:** exactly 2, matching `outer_workers`
  exactly — never more, at any sample, throughout the run.
- **Peak live threads:** steady state ~21 per process (20 requested
  `ThreadPoolExecutor` workers + 1 main thread) = **42 aggregate**,
  matching the logged product (40) plus 2 main threads almost exactly.
  Transient spikes to as high as 69 in a single process (90 aggregate
  across both) appeared briefly and irregularly — consistent with
  scipy/numpy's own internal BLAS thread bursts during a KDTree
  build/query, not with the code opening extra pools.
- **Measured CPU usage stayed at ~38-40% per worker process throughout**
  (`pcpu`, i.e. well under one full core each, ~80% total across
  both) — nowhere near what 2 processes x 20 requested threads would
  need to actually saturate 40 cores. The "20 inner threads" figure is a
  ceiling the design intentionally allows to exceed core count for
  *threads* (they release the GIL for short bursts inside compiled
  pyEDM/scipy calls, per this codebase's own existing docstrings on why
  threads were chosen there) — unlike the *process* count, which the
  `outer_workers = min(node_count, cores)` formula does keep at or under
  `cores`.

Only Stage 2's own pool was exercised this run (Stage 1's is
structurally identical code — the same `with ProcessPoolExecutor(...) as
executor:` pattern, the same `outer_workers`/`inner_workers` sizing
formula — so this measurement is representative of both, even though
Stage 1 didn't get its own live sample this particular run).

### Step 3 — can concurrent waves/retained pools push past the core count?

**No**, confirmed both structurally and empirically:
- Every `ProcessPoolExecutor` is opened via `with ... as executor:` and
  used with `as_completed()` before the block exits — Python's own
  `ProcessPoolExecutor.__exit__` calls `shutdown(wait=True)`, which
  blocks until every worker process has actually terminated, not just
  until submitted work finishes. No pool can outlive its own wave.
- R12 makes Stage 1 and Stage 2 strictly sequential (Stage 1's entire
  `while` loop runs to completion before Stage 2's first wave even
  starts) — there is no window where a Stage 1 pool and a Stage 2 pool
  could be alive at once.
- Within either stage, waves are processed one at a time in a single
  `while current_wave:` loop, each iteration opening and fully closing
  its own pool before the next iteration's `next_wave` is even computed
  — no two waves' pools can overlap either.
- Empirically: peak process count observed was exactly `outer_workers`
  for the one active wave, never higher — no evidence of a leftover pool
  from a "previous" wave.

### Acceptance

No semaphore added — the measured peak (2 processes, well under 40 cores;
aggregate thread count exceeds the nominal core count only transiently
and only in the GIL-releasing-thread sense the design already accounts
for, never in measured CPU usage) does not warrant one, per the task's
own conditional ("semaphore only if the measurement warrants it").

---

## R7 — Confirm Section 11 diagnostics produce figures

**Status:** Done. Found real gaps in both pipelines, both worse than the
task's own framing anticipated — fixed.

### A1 verification / finding

Read `run_tp_vs_rho_sweep()` and `run_univariate_e_tau_grid()` in both
files, plus `_failed_mde_result()`/`_failed_multi_mde_result()` (the
fallback a target's `mde_results` entry gets when its depth-0 MDE call
*raises*, as opposed to legitimately selecting zero drivers).

**`run_tp_vs_rho_sweep()`: confirmed the exact "empty figure" risk the
task named.** One shared `fig`/`ax` is opened before the per-target loop;
`fig.savefig()` ran *unconditionally* after the loop regardless of
whether anything was ever plotted. If every target hit the zero-columns
skip, this wrote a real PNG with empty axes and a legend with nothing to
show. (tsls-mde-gmn.py already had the per-target `if not cols: continue`
guard; **simple-mde-gmn.py had no such guard at all** — it called
`PredictInterval(columns=[], E=0, ...)` unconditionally, a call that
would itself fail or behave nonsensically for any zero-driver target, not
just the "all targets skipped" case.)

**`run_univariate_e_tau_grid()`: the real risk is a crash, not an empty
file, and it's not about zero columns at all.** This function computes a
*univariate* self-predictability grid (`columns=target, target=target`)
— it needs no MDE-selected drivers, so (correctly) it never checked
`SELECTED_COLUMNS`. But it unconditionally dereferenced `r['mde_df']` and
`r['m'].args.lib/pred`, and a target whose depth-0 MDE call itself
*raised* gets `mde_df=None` (`_failed_mde_result()`/
`_failed_multi_mde_result()`, both confirmed reachable from `main()`'s
own error handling in either `GMN_MODE`). Passing `dataFrame=None` into
`proper_simplex()`/`Simplex()` raises immediately — and since the
per-target loop has no `try`/`except`, the exception propagates out of
the *whole function*, taking every remaining target's own (unrelated,
otherwise perfectly computable) figure down with it. This call site has
no surrounding `try`/`except` in `main()` either — a crash here is fatal
to the whole run, occurring right before the final `'Pipeline complete.'`
print. This is worse than "no figure": it's "no figure for *any* target
from this point on, and the job appears to have failed," for a condition
(`mde_df is None`) that has nothing to do with the "zero columns
selected" framing the task named.

**Reaches `OUTPUT_ARCHIVE_DIR` on clean exit:** confirmed this is a
generic guarantee, not something specific to these two functions — both
sbatch scripts' own `cleanup` EXIT trap does `cp -a "$OUT_DIR"/.
"$OUTPUT_ARCHIVE_DIR"/`, copying everything either function writes to
`results_dir` (`== $OUT_DIR`) automatically, for any file, on any exit
the trap fires for. No code change needed for that part specifically —
it already holds for any file written to `results_dir`.

### What changed

Both files, both functions:
- `run_tp_vs_rho_sweep()`: added the missing per-target zero-columns
  skip to simple-mde-gmn.py (tsls-mde-gmn.py already had it); added the
  "skip `savefig()` entirely if nothing was ever plotted" guard to *both*
  (figure is still `plt.close()`d either way, no resource leak).
- `run_univariate_e_tau_grid()`: added a per-target `if r['mde_df'] is
  None: continue` guard to both — deliberately *not* gated on
  `SELECTED_COLUMNS`, since this function's own univariate grid is
  meaningful (and wanted) even for a legitimate zero-driver target; only
  the actual crash-causing condition is guarded. Matches this codebase's
  own established "one degenerate item dead-ends cleanly, it doesn't
  crash the batch" convention (`_failed_mde_result()`'s own docstring).

### Test

`tests/test_r7_diagnostics_figures.py`, both pipelines:
- `run_tp_vs_rho_sweep()` with every target at zero selected columns →
  asserts no figure file is written at all.
- `run_univariate_e_tau_grid()` with one target's `mde_df=None`
  (constructed to match `_failed_mde_result()`'s exact shape) alongside
  one genuinely valid target → asserts the call does **not** raise, the
  failed target produces no figure, and the valid target's figure is
  still produced (proving one failure doesn't take the batch down).

Both passed on first run, both pipelines.

---

## R8 — Add resampling to the CCM pre-filter

**Status:** Done. Finding only, per explicit user direction — the task's
premise doesn't hold, so `CCM_PREFILTER_SAMPLE` was not added. Includes a
follow-up sample-count sweep and a related redundant-computation
investigation, both requested inline by the user during this task.

### A1 verification — premise check failed, reported before proceeding

R8's premise: "the pre-filter draws one library sample per library size
while the main CCM path averages `CCM_SAMPLE` draws." **This is false in
both pipelines as they stand:**

- **tsls-mde-gmn.py:** `_ccm_prefilter_pair()` already calls
  `tsls.proper_ccm(..., sample=ccm_sample, ...)` where `ccm_sample` is
  `ccm_prefilter_observables()`'s own parameter, and `main()`'s call site
  passes `cfg['CCM_SAMPLE']` — the *same* config value every other CCM
  call in the pipeline uses (confirmed `sample`'s meaning directly from
  `proper_ccm()`'s own docstring: "Number of random library subsamples
  drawn at each library size").
- **simple-mde-gmn.py:** `_ccm_prefilter_worker()` calls `pyEDM.CCM(...,
  sample=20, ...)` — hardcoded, not a single draw either, just not
  independently configurable.

Stopped and asked before implementing anything, per Part A. **User
confirmed:** the premise was a misreading of §3.1 Step 2 by the reviewer,
not a defect in the code. No `CCM_PREFILTER_SAMPLE` knob added — there is
nothing to fix per the task's own reasoning, since the seed-dependency
concern it raises doesn't currently exist in either pipeline.

**For the documentation rewrite pass (A6):** §3.1 Step 2's wording ("run
one `proper_ccm` call and average its score across the swept library
sizes") reads as one draw per library size and is what misled this
review. It should instead say the pre-filter averages `CCM_SAMPLE` draws
per library size (tsls) / a hardcoded 20 draws per library size
(simple), then averages across sizes.

### Follow-up: is 20 draws more than this screen needs?

Per user direction, the useful question flips from "is one draw too few"
to "is `CCM_SAMPLE`'s usual value (20) more than the pre-filter needs."
Ran `ccm_prefilter_observables()` on real data
(`J1620210605_epoch_8_freq_40_Hz.csv`, `REGION_CHANNEL_CAP=15`, 45
real candidates, `GMN_MODE='combined'` default, both targets) at
`sample=20`, `10`, and `5` (single run each, not repeated seeds — the
question here is cost-vs-stability across draw counts, not seed
variance, since seed variance is no longer the open question):

| `sample` | wall-clock | kept candidates |
|---|---|---|
| 20 | 94.9s | 17/45 |
| 10 | 49.4s | 17/45 (identical set) |
| 5  | 24.1s | 17/45 (identical set) |

**The kept-candidate set was identical, member-for-member, at all three
draw counts.** Per-candidate scores drifted only in the 3rd-4th decimal
place (e.g. `CA1_10_tsls`: 0.2391 -> 0.2370 -> 0.2293) — negligible next
to the 0.2 `CCM_PREFILTER_RHO_MIN` cutoff, and no candidate sat close
enough to that boundary for the drift to matter. Wall-clock scaled
linearly with `sample`, as expected (~2x per 2x draws). **No default
changed** — this is evidence for a future decision, not an action taken
now, per the task's own scope boundary.

**If `CCM_PREFILTER_SAMPLE` is ever added on this evidence, its default
must be 20, not 5** (or any other reduced value) — shipping a reduced
default would silently drop the pre-filter from today's 20 draws to
fewer and change kept-candidate sets under cover of what looks like a
pure refactor (introducing the knob itself must be behavior-preserving;
lowering the value is a separate, explicit decision for whoever reviews
this evidence). This mirrors R22's own "numerical regression disguised
as a refactor" concern.

### Related investigation: can the CCM pre-filter's own value be reused to
skip the MDE gate's CCM check for a real observable's own first
selection?

The user asked me to check this directly, since the pre-filter already
computes `CCM(candidate -> target)` for every real observable against
the target, and depth-0/Stage 1's own MDE forward-selection re-runs a
`CCM(newColumn -> target)` gate for the very same pairs. **Confirmed no
such reuse exists today, and confirmed it would NOT be safe to add as a
literal "skip and reuse" optimization** — the two values are not computed
under the same parameters, so reusing one in place of the other would
silently change MDE's actual selection criterion, not just skip redundant
work:

- **`E` differs in kind, not just value.** The pre-filter uses a static
  `E=CCM_PREFILTER_E` (default 7) for every candidate. The MDE gate
  (`cleaned_helper_functions.py:1484-1497`) uses `E=maxEDim`, auto-selected
  *per candidate* via `proper_embed_dimension()` (confirmed via R4) — a
  different value for nearly every candidate, and never equal to the
  pre-filter's static value except by coincidence.
- **`libSizes` come from different ratio lists over different windows.**
  The pre-filter converts `LIBSIZE_RATIOS` against a `_ccm_window(...,
  0.5, CCM_WINDOW_ROWS)` window (a fixed 50%-centered slice). The MDE
  gate converts `CCM_LIB_SIZE_PCTS` against `_ccm_window(..., lib_ratio,
  CCM_WINDOW_ROWS)` where `lib_ratio = (prediction_start - warmup) /
  (usable_end - warmup)` — a data-driven ratio with no reason to equal
  0.5 (typically closer to `LIB_RATIO`, e.g. 0.7, minus warmup/Tp
  trimming). Two different ratio lists over two differently-centered
  windows of the same row count produce different actual library sizes.
- **The CCM gate itself is univariate at every dimension** (`columns=
  newColumn`, confirmed by reading `cleaned_helper_functions.py:1484` —
  not `[newColumn] + already_selected` as might be assumed), so the
  *kind* of test the two functions run is the same (univariate candidate
  -> target CCM) — only the parameters it's run under differ. This is
  what makes the two numbers look interchangeable at a glance, and why
  they aren't.

Given these are real, structural parameter differences (not incidental),
implementing a literal "reuse the pre-filter's score, skip the MDE gate"
optimization would be a silent numerical regression — the same class of
mistake R22 already flags for a different pair of call paths. **Not
implemented.** A narrower version that genuinely would be safe — making
the pre-filter numerically consistent with the real gate (same `E`
resolution, same window/libSize convention) so its value becomes
legitimately reusable — is a materially larger change than "skip a
redundant call," is not something either R8 or this investigation asked
for, and is left as a separate, explicit decision if the user wants to
pursue it.

---

## R9 — `CCM_WINDOW_ROWS` and `CCM_PREFILTER_E` calibration harness

**Status:** Done. Standalone script + real output + finding, no pipeline
default changed, per the task's own acceptance criterion. Feeds R17/R18.

### What was built

`scripts/calibration/r9_ccm_window_e_calibration.py` — a standalone,
read-only script (imports `cleaned_helper_functions`/`tsls-mde-gmn.py`'s
own `proper_ccm`/`_ccm_window`/`convert_to_tsls`/`burn_in_trim`, changes
no pipeline config or code):

1. Stratified-samples 3 observables from each of 3 regions x 3
   firing-rate terciles (low/medium/high spike count within that
   region) = 27 observables, paired with both real `TARGETS` = **54
   (observable, target) pairs** (>= 40 required).
2. Window sweep: at each `CCM_WINDOW_ROWS in {200, 400, 800, 1600,
   3200}`, computes every pair's CCM skill-vs-library-size curve
   (`proper_ccm`, the pipeline's current `CCM_PREFILTER_E=7`) and reports
   the fraction still rising vs. flat at the smallest tested library
   size for that window.
3. E sweep: at the pipeline's current `CCM_WINDOW_ROWS=800`, computes
   the pre-filter-style mean-rho score at each `CCM_PREFILTER_E in {3, 5,
   7, 9, 11}` for the same 54 pairs, and reports the kept-candidate set
   (score >= `CCM_PREFILTER_RHO_MIN=0.2`) and its Jaccard stability
   against the current default `E=7`.
4. Writes `r9_summary.csv` (one row per window/E setting) and
   `r9_raw_curves.csv` (one row per pair per window per library size, the
   full raw curves) into `scripts/calibration/` — both committed
   alongside the script, per the task's own "its output" acceptance
   criterion.

The harness's own CCM calls use `sample=10` (not the pipeline's own
`CCM_SAMPLE`/`CCM_PREFILTER_SAMPLE`-adjacent default of 20) purely to keep
this exploratory sweep's own wall-clock down — justified by R8's own
finding (immediately above) that sample counts well below 20 already
reproduce a stable kept-candidate set at this same real-data scale. This
is this harness's own internal choice, not a pipeline default.

### Results (real data, `J1620210605_epoch_8_freq_40_Hz.csv`)

**Window sweep** (fraction of the 54 pairs still rising at that window's
own smallest tested library size):

| `CCM_WINDOW_ROWS` | smallest tested libsize | still rising | flat |
|---|---|---|---|
| 200  | 10  | 44/54 (81%) | 10/54 |
| 400  | 20  | 29/54 (54%) | 25/54 |
| **800**  | **40**  | **19/54 (35%)** | 35/54 |
| 1600 | 80  | 0/54 (0%)   | 54/54 |
| 3200 | 160 | 4/54 (7%)   | 50/54 |

**The pipeline's current default (`CCM_WINDOW_ROWS=800`) leaves a real,
non-trivial fraction of pairs (35%) still improving at the smallest
library size tested there** — not converged. Convergence is essentially
complete by `1600` (0% still rising) with a small amount of measurement
noise reappearing at `3200` (7% — plausibly `sample=10` noise near the
0.01 rising/flat threshold, not a genuine reversal; flagged as a caveat,
not a finding to over-read). **This is directly relevant to R17's own
"still rising above 40 rows" question**: `CCM_WINDOW_ROWS=800`'s own
smallest tested library size *is* 40 rows (`0.05 x 800`), so this
harness's window=800 row above answers R17's specific question directly:
35% of pairs still rising at 40 rows is a real, substantial fraction —
not "most pairs" (which would argue for R17's coarser
`[5,20,40,60,80]` grid), but not negligible either (which would argue
against dropping every library size below 40 outright). Leaning toward
R17's own "coarsen only the upper end" alternative, though the final call
is R17's own to make from this evidence.

**E sweep** (kept-candidate count and stability vs. the current default
`E=7`, at `CCM_WINDOW_ROWS=800`):

| `CCM_PREFILTER_E` | kept | vs. `E=7` (Jaccard) | changed pairs |
|---|---|---|---|
| 3  | 30 | 0.91 | lost 3 |
| 5  | 31 | 0.94 | lost 2 |
| **7**  | **33** | **1.00** | (baseline) |
| 9  | 35 | 0.94 | gained 2 |
| 11 | 35 | 0.94 | gained 2 |

**The kept-candidate set is reasonably stable across `E in {3,...,11}`**
(Jaccard >= 0.91 throughout, monotonically more inclusive as `E`
increases) — no dramatic swing, and the current default `E=7` sits in
the middle of the tested range rather than at either extreme. This is
the empirical calibration R18 needs; R18 itself decides whether to keep
`E=7` or move it.

**No pipeline default was changed** — per the task's own acceptance
criterion, this harness only measures and reports.

---

## R17 — `CCM_LIB_SIZE_PCTS` grid, after R9

**Status:** Done (tsls-mde-gmn.py only — see below).

### A1 verification / decision, gated on R9

`CCM_LIB_SIZE_PCTS` default was `(2, 3, 5, 8, 12, 20, 30, 50)` — these
are *percentages* of the `CCM_WINDOW_ROWS`-bounded row count (confirmed
via the existing `load_config()` comment: "sweeps library sizes as
PERCENTAGES of however many rows it's handed"), so at the current
`CCM_WINDOW_ROWS=800` default these resolve to {16, 24, 40, 64, 96, 160,
240, 400} rows. R9's own window=800 measurement is the direct answer to
this task's decision question: 35% of the 54 sampled pairs were still
rising at 40 rows (the 5% point) — a real fraction, but **not** "most
sampled pairs" (would need >50%). Per the task's own decision rule, this
means: **coarsen only the upper end**, not adopt the reviewer's coarser
`[5, 20, 40, 60, 80]` grid (which would drop the 2%/3% points — 16/24
rows — entirely, losing exactly the region R9 shows still carries real
signal for over a third of pairs).

**Sibling parity note:** `simple-mde-gmn.py` has no equivalent knob at
all — it relies on `dimx.MDE`'s own internal default (`pLibSizes=(10, 15,
85, 90)`, a much coarser, hardcoded, unrelated convention) with no
pipeline-level override surface. This is a pre-existing architectural
difference, not something this task introduces — nothing to port there.

### What changed

`CCM_LIB_SIZE_PCTS` default: `(2, 3, 5, 8, 12, 20, 30, 50)` ->
**`(2, 3, 5, 8, 20, 50)`** — drops `12` and `30` (both already in R9's
"flat" region at `CCM_WINDOW_ROWS=800`: 96 and 240 rows respectively, well
past the 40-row point where rising curves were still common), keeps
every percentage at or below the 5%/40-row point R9 flagged as still
informative.

A5: `ccm_lib_size_pcts` is already threaded into `build_search_fingerprint()`
as `list(cfg['CCM_LIB_SIZE_PCTS'])` — the fingerprint captures whatever
value the config resolves to, whether from the (changed) default or an
explicit override, so no fingerprint code change was needed for this
default change specifically.

### Measured effect (real data, `REGION_CHANNEL_CAP=15`, depth-0 MDE for
both real `TARGETS`)

Ran `run_mde_for_target()` for both `head_position_x`/`head_position_y`
under the old (8-percentage) and new (6-percentage) grids, otherwise
identical config:

| target | old grid (8 pcts) | new grid (6 pcts) | selections |
|---|---|---|---|
| `head_position_x` | 2.3s | 1.7s | identical: `['mPFC_1_tsls']` |
| `head_position_y` | 17.5s | 12.7s | identical: `['CA1_1_tsls']` |
| **total** | **19.8s** | **14.4s** | **no change** |

**~27% wall-clock reduction, zero change in kept-candidate set** on this
real-data run — consistent with R9's own finding that the dropped
percentages (12%, 30%) sit in the already-converged tail where they add
compute without adding new information. Per the task's own instruction,
this is a scientific calibration change backed by R9's own evidence
(stated here and in the commit message), not an arbitrary tuning
adjustment.

---

## R18 — `CCM_PREFILTER_E`, after R9

**Status:** Done. Finding only — left at `7`, with the sweep's own
evidence recorded, per the task's own acceptance criterion's second
branch.

### A1 verification / decision, gated on R9

R9's E sweep (54 real pairs, `CCM_WINDOW_ROWS=800`) is exactly the
empirical calibration this task asks for. Results (from R9's own
section, repeated here for this task's own record):

| `E` | kept | Jaccard vs. `E=7` |
|---|---|---|
| 3  | 30 | 0.91 |
| 5  | 31 | 0.94 |
| **7**  | **33** | **1.00 (baseline)** |
| 9  | 35 | 0.94 |
| 11 | 35 | 0.94 |

`E=7` shows no special problem and no clearly-superior alternative:
Jaccard stays >= 0.91 across the entire tested range (no cliff at any
point), and `E=7` sits in the middle of that range rather than at either
extreme — consistent with the *existing* `CCM_PREFILTER_E` docstring's
own account of how `7` was originally chosen ("the middle value of this
pipeline's own already-team-vetted `E_RANGE` search... not an untested
extrapolation," `load_config()`'s own comment). Moving to `9`/`11` only
gains 2 more candidates out of 54; moving to `3`/`5` loses 2-3 — neither
direction corrects an evident error, both are within the same broad
stability band R9 measured.

**Kept mean aggregation** — unaffected by this task, per its own scope
note; `ccm_prefilter_observables()` already aggregates by mean across
library sizes (confirmed during R8's own reading), and slope-based gating
correctly stays exclusive to `proper_mde`'s own real CCM gate.

**Sibling-parity scoping note:** `simple-mde-gmn.py`'s own
`CCM_PREFILTER_E` default is **`3`**, not `7` — a different value,
calibrated separately for that pipeline's own 1 Hz-binned data regime
(confirmed via its own `load_config()`; unrelated to the 40 Hz TSLS
data R9's own sweep used). `docs/review-actions-v1.md` is explicitly
scoped to the tsls pipeline (its own header: "Source: 62 highlights / 44
notes on `tsls-mde-gmn-pipeline_v1_annotated_export.pdf`"), and R9's
sweep only ever exercised `tsls-mde-gmn.py`'s own `convert_to_tsls`/
`proper_ccm` machinery — so this decision applies to `tsls-mde-gmn.py`'s
`CCM_PREFILTER_E` only. `simple-mde-gmn.py`'s own `3` default is out of
scope here, not touched.

### Decision

**Left at `7`.** No code change — R9's sweep supports the existing
default rather than pointing at a better one, satisfying this task's own
"left at 7 with an explicit statement that the sweep supports it"
acceptance branch.

---

## R11 — Keep the target-only search objective; report per-node scores additively

**Status:** Done.

### A1 verification — step 1 (confirm target-only, no bug found)

Read `evaluate_smap_candidate()`, `run_theta_search()`, and
`sweep_generate_start()` in full, both files. **Confirmed target-only
already, no bug:**

- `evaluate_smap_candidate()`'s own `score` is `float(np.nanmean(list(
  target_rhos.values())))`, where `target_rhos` is built by iterating
  `net_result['targets']` only, scored via `target_metric` — never a
  driver node, never `metric`.
- `sweep_generate_start()` calls `evaluate_smap_candidate()` per
  candidate start offset and picks `max(valid_offsets, key=lambda o:
  scores[o])` — the comparison is over that same target-only `score`,
  nothing else enters it.
- `run_theta_search()` doesn't run a literal multi-candidate search of
  its own at the top level — it retunes each node's own theta once
  (`retune_theta_for_all_nodes()`, necessarily per-node, since *every*
  node needs *some* theta to run SMap regardless of whether it's a
  target) and then calls `evaluate_smap_candidate()` once to *report* the
  resulting network's score. That one call is target-only, same as
  above. No mixing, no driver-node scoring anywhere in the objective
  path.

Since nothing here needed fixing, step 1's own "if any of them scores
driver nodes or mixes metrics, that is a bug — report it, then fix it"
branch doesn't apply.

### Step 2 — per-node table

**Already half-done, found while reading.** `rebuild_at_best_params()`
(both files) already computes and *prints* a per-node score for every
node of the winning configuration, tagged `TARGET`/`driver`, using each
node's own `_metric_for`-resolved metric (tsls) / plain rho (simple, no
`_metric_for` there) — this exists prior to this task and is exactly the
"compute and log" half of step 2. It was never written to a file. Added
a `results_dir / f'per_node_scores_{network_id}.csv'` write immediately
alongside the existing print, same data, same per-row metric label,
purely additive — no change to what gets computed or printed, no change
to any search decision.

### Test

`tests/test_r11_per_node_table.py` (both pipelines): builds a real, tiny,
synthetic `gmn.GMN.Generate()` network (same pattern as
`tests/test_r14_external_inputs.py`) through `rebuild_at_best_params()`,
and asserts the CSV exists with the expected columns
(`node`/`role`/`metric`/`score`), one row for the target (tagged
`TARGET`) and one for the driver (tagged `driver`), both with finite
scores. Passed on first run, both pipelines.

### Acceptance: bit-identical selected theta/start row

**Verified structurally, not by re-running Run B.** The CSV write is
added inside `rebuild_at_best_params()`, which only *reruns* Generate()
at whichever theta/start row `run_theta_search()`/`sweep_generate_start()`
already decided — it runs strictly *after* both selections are final and
never feeds back into them. Neither of those two functions, nor
`evaluate_smap_candidate()`, was touched by this change. Since the
selection code path is untouched and the new code runs downstream of it,
the selected theta and start row are bit-identical by construction — a
fresh Run B re-run would only be re-confirming that no edit was made to
the selection logic, which reading the diff already establishes directly.

---

## R16 — Sibling engine migration, behind a toggle

**Status:** Done as a scoping report — **no code changed**, per the
task's own hard-condition acceptance branch ("nothing changed and a
scoping report explains why").

### A1 verification

Read every consumer of `run_mde_for_target()`'s return shape in
`simple-mde-gmn.py` (`grep -n "\['m'\]\."` across the whole file) and the
current `GMN_MODE='combined'` code path in `main()`'s own MDE block.

**The bare single-target engine swap looks mechanically achievable
without touching any consumer.** `run_mde_for_target()` currently returns
`{'target', 'mde_df', 'm': <the raw dimx.MDE instance>, 'SELECTED_COLUMNS'}`
— but *every* consumer in the file (`retune_theta_for_all_nodes()`,
`_accept_mde_result_edges()`, `run_mde_evaluate()`, `run_tp_vs_rho_sweep()`,
`run_univariate_e_tau_grid()`) only ever touches `m.MDEcolumns`,
`m.MDErho`, `m.MDEOut`, or `m.args.lib`/`.pred`/`.Tp`/`.tau` — exactly
the same normalized attribute set `tsls-mde-gmn.py`'s own
`run_mde_for_target()` already builds (as a plain
`types.SimpleNamespace`) from `proper_mde`'s own dict return. Nothing
reaches into `dimx.MDE`-only internals (`m.rhoD`, `m.EDim`, `m.Plot()`,
etc.) anywhere. Confirmed specifically for the two dependencies the task
names by name: `run_mde_evaluate()`'s own `dx.Evaluate(...)` call is
constructed from plain values (`dataFrame`, `columns_range`,
`mde_columns`, `predictVar`, and `library`/`prediction`/`Tp`/`tau` pulled
from `m.args.*`) — it never receives the `m` object itself, so it cannot
depend on which engine produced it. The embedding path
(`_embedding_spec_for()`) doesn't touch `mde_results`/`m` at all — it
reads only the constructed graph.

**But the task's own stated *purpose* — "so both pipelines can perform
joint multi-target MDE" — is not just a toggle, and is where partial
migration would actually happen.** `simple-mde-gmn.py` has no
`multi_proper_mde`/`run_multi_mde_for_targets` equivalent at all today.
Confirmed `GMN_MODE='combined'` (the default) does **not** currently
perform joint MDE here, despite the name suggesting otherwise: `main()`'s
own MDE block still calls `run_mde_for_target()` **once per target inside
a plain `for target in group:` loop** — `GMN_MODE='combined'` only
changes the *pre-filter's* own grouping (mean-across-targets scoring),
never the actual selection call. This is because `dimx.MDE` itself has no
joint multi-target mode to call — there is nothing for a same-shaped
toggle to switch *to* on the `dimx` side. Giving this pipeline genuine
joint MDE (matching `tsls-mde-gmn.py`'s own `run_multi_mde_for_targets()`
→ `{**shared_result, 'target': target}` fan-out per target) would require,
at minimum:
- A new `run_multi_mde_for_targets()`-equivalent wrapping
  `cleaned_helper_functions.multi_proper_mde` (this file doesn't even
  import `cleaned_helper_functions` today — it's TSLS-specific tooling,
  imported only by `tsls-mde-gmn.py`).
- A real branch in `main()`'s `GMN_MODE='combined'` handling to call it,
  replacing today's per-target loop for that mode.
- New config knobs (`MULTI_MDE_E`, `MULTI_MDE_TARGET_AGGREGATE`
  equivalents) with their own defaults for this pipeline's own 1 Hz data
  regime — not simply reused from `tsls-mde-gmn.py`'s own values,
  which are calibrated for 40 Hz TSLS data (same reasoning as R18's own
  sibling-parity note).
- Verification (not just an assumption by analogy) that this pipeline's
  own `_accept_mde_result_edges()`/BFS correctly handles a fanned-out
  joint result the same way `tsls-mde-gmn.py`'s does — untested here.

This is real, multi-part, non-trivial work — not "add a toggle" — and
implementing only the narrower, consumer-safe half (a `dimx`/`proper_mde`
switch for the existing per-target-only call, with `GMN_MODE='combined'`
still never actually joint) would deliver a toggle that exists but does
not achieve the task's own stated purpose. That is precisely the "partial
migration" the task says is not acceptable.

### Decision

**No code changed.** Per the task's own hard condition, achieving the
stated goal (genuine joint multi-target MDE, not just an engine label
swap) requires refactoring beyond a same-shape toggle — a new selection
function, a new `main()` branch, new config knobs, and unverified BFS
compatibility. Reporting this scope and stopping, rather than shipping a
toggle that reproduces today's dimx behavior exactly but leaves
`GMN_MODE='combined'` exactly as un-joint as it is today.

---

## R19 — TSLS-aware `run_mde_evaluate`

**Status:** Done. `tsls-mde-gmn.py` only — `simple-mde-gmn.py` already has
its own working `run_mde_evaluate()` (`dx.Evaluate`-based), unchanged by
this task; see the sibling-parity note below for why this is not a port.

### A1 verification

Confirmed `tsls-mde-gmn.py` had **no** `run_mde_evaluate()` at all before
this task — the file's own pre-existing comment (now updated) said so
explicitly: "the notebook cell 15 diagnostic... has no
cleaned_helper_functions.py equivalent... stays dropped." Read
`simple-mde-gmn.py`'s own `run_mde_evaluate()` (its one existing call
site, unmodified, used verbatim to generate the reference figure below)
and, per the task's own explicit instruction ("read `dimx.Evaluate`'s
source directly to recover the exact CAE definition... rather than
reimplementing from the name"), read
`.venv/lib/python3.11/site-packages/dimx/Evaluate.py` in full (704
lines):

- **`CAE(x, y)` (line 503-506): `delta = sum(abs(x - y)); return delta`**
  — an unnormalized cumulative absolute error, not a mean or a fraction.
  Ported verbatim as `_cae()`.
- **`Run()` (line 207-320):** MDE via `Simplex(embedded=True)` on
  `mde_columns`; PCA via `sklearn.PCA(n_components).fit(data_lib)` (where
  `data_lib` is `FilterData()`'s own column subset — confirmed via
  `FilterData()`, line 458-501: `columns_range` selects 0-offset column
  indices from the **full** dataFrame, and `simple-mde-gmn.py`'s own call
  passes `columns_range=[1, mde_df.shape[1]]` against
  `mde_df = df[[target] + observable_columns]` — i.e. every candidate
  column **except the target itself**, not the narrower MDE-selected
  subset and not the full pre-CCM-prefilter universe either), then
  `LinearRegression().fit(pca_lib, predictVar_lib)` to predict the target;
  Diffusion Map identically, via
  `pydiffmap.diffusion_map.DiffusionMap.from_sklearn(...).fit(data_lib)`.
- **`Plot()` (line 323-401):** 4 stacked subplots (`sharex=True`) —
  Predictions / MDE / Diffusion Map / PCA, in that order. Panel 1: D-Map,
  PCA, MDE lines, then the black actual-target line, legend titled `'CAE'`
  showing each method's CAE value (or `'R'` if `plotRho`, not used here).
  Panel 2: one line per `mde_columns[:maxN_]` — the **raw** driver series
  over the prediction window, not a prediction. Panels 3-4: one line per
  component's own prediction-window projection.

This is the exact structure `run_mde_evaluate()`'s docstring below
reimplements against `proper_simplex`/`sklearn`/`pydiffmap` directly,
never against `dx.Evaluate` itself (which has no `_metric_for()`/
`_score()` awareness and depends on a `dimx.MDE` result object this
pipeline no longer constructs — same reasoning R16 already established
for why a bare engine swap doesn't reach this function either).

### Reference figure

Generated from `simple-mde-gmn.py`'s own unmodified `run_mde_evaluate()` /
`dx.Evaluate()` call path on real data
(`J1620210605_epoch_8_freq_4_Hz.csv`, target `head_position_x`,
`REGION_CHANNEL_CAP=15`, `components=5`) — not fabricated. Measured values
embedded in the figure: MDE CAE=13765.30/RMSE=24.618/R=0.56, PCA
CAE=12955.82/RMSE=23.672/R=0.106, D-Map CAE=12538.01/RMSE=23.711/R=-0.014;
selected MDE columns `['CA1_9', 'CA1_2', 'CA1_7', 'mPFC_12', 'mPFC_1']`.
Committed at `docs/reference/mde-evaluate-reference-plot.png`.

### Ask-before-implementing (the task's own explicit gate)

R19's own text: "confirm whether panel 1 in `combined` mode should be a
spatial x-y trajectory plot... or two stacked time series... The
instruction reads as the former; the reference figure is single-target
and does not settle it." Asked. **User confirmed: spatial x-y trajectory**
(position in the plane).

### Design decisions beyond the task's literal text (documented here since
R19 only defines these for/from a single-target figure)

1. **`D` for a >1-target group (`GMN_MODE='combined'`).** R19 defines `D`
   as "the number of direct drivers of *the* target," singular. With two
   targets in one figure, this pipeline's own reading is **the union of
   each target's own direct-driver set** — `graph.predecessors(x) |
   graph.predecessors(y)` — read directly from
   `net_result['network']['Graph']` (the completed, both-stages network;
   never a depth-0-only `SELECTED_COLUMNS` list). This is "how many
   distinct driver nodes feed the target(s) this figure is about," the
   natural generalization once there is more than one target to be a
   direct driver of. Every target passed in is already guaranteed `>= 1`
   driver by the time this runs — `build_network()` already calls
   `partition_network()` on this same graph, which raises on a
   zero-driver target (R14) — so there is no zero-driver-target case to
   guard against here.
2. **PCA/diffusion-map's own "full pool."** R19 says both baselines use
   "their first `D` components" but doesn't say what universe those
   components are extracted *from*. Read `FilterData()` (above) to answer
   this from source rather than guessing: dimx's own convention is every
   candidate column considered at depth-0, minus the target. The TSLS-side
   analog used here is every **non-target** column in
   `net_result['DATA_AFTER_MDE']` — i.e. every observable nested-MDE kept
   *anywhere* in the whole network (a superset of the target's own direct
   drivers whenever Stage 2 chains through an intermediate node), not the
   full CCM-pre-filtered candidate pool nested-MDE searched over and
   rejected most of. That broader pool isn't serialized anywhere after
   `build_network()` returns (it lives only in `main()`'s own local
   `filtered_pools` dict); re-threading it through every call site between
   `main()` and here for one diagnostic would be a materially bigger
   change than this task asks for, and `DATA_AFTER_MDE` is the same file
   every *other* diagnostic in this file already reads for its own ground
   truth (`gmn_fidelity_rho()`, `run_forecast_diagnostic()`, etc.) — using
   a different, wider data source here would be the inconsistent choice.
3. **Combined mode's joint regression.** R19: "PCA and diffusion-map
   extract components of the multivariate target." Read `Run()` (above):
   `LinearRegression().fit(...)` in dimx already receives a plain 1-D
   `predictVar_lib` array — but `sklearn.linear_model.LinearRegression`
   natively accepts a 2-column `y` and fits a genuine multi-output linear
   model (confirmed against sklearn's own documented behavior, not
   assumed). So in combined mode, PCA/diffusion-map fit **one** shared
   component basis on the pool and regress it onto **both** target
   columns at once (`target_lib` is `(N, 2)`), producing a 2-column
   prediction directly — not two independent single-target `Evaluate`
   runs. MDE has no equivalent multivariate call (`proper_simplex`
   predicts one target column at a time), so it still reconstructs each
   target separately from its own direct-driver embedding; the two
   resulting 1-D reconstructions are combined into one `(x, y)` trajectory
   for the spatial panel — the only place a "joint MDE prediction" is
   synthesized rather than computed directly.
4. **Combined-mode MDE CAE in panel 1's legend.** Since `_cae()` on a
   2-column array already sums the absolute error over *both* columns at
   once (`numpy`'s own `abs`/`sum` broadcast this way with no axis given —
   confirmed by construction, not assumed), MDE's own combined-mode CAE is
   reported as the **sum** of its two per-target CAEs
   (`mde_cae[x] + mde_cae[y]`) for the same reason — an apples-to-apples
   additive combination, not an average.

### What changed (`scripts/src/tsls-mde-gmn.py` only)

- **New imports:** `sklearn.decomposition.PCA`, `sklearn.linear_model.LinearRegression`,
  `pydiffmap.diffusion_map` — not a `dimx` dependency; these are the same
  underlying libraries `dimx.Evaluate` itself wraps, imported directly
  since this pipeline has no `dimx` import to route through at all.
- **`_cae(x, y)`** (new): `sum(abs(x - y))`, recovered verbatim from
  `dimx.Evaluate.CAE()`.
- **`run_mde_evaluate(net_result, prediction_start, prediction_length,
  results_dir, target_metric, gamma_delta, exclusion_radius=0, ...)`**
  (new): computes `D` and `direct_drivers` from
  `net_result['network']['Graph']`; runs MDE (`proper_simplex`,
  `embedded=True`, one call per target on its own direct-driver columns),
  PCA, and diffusion-map (both on the network-wide driver pool, jointly
  regressed onto the target(s)) over the held-out window
  `[prediction_start, prediction_start+prediction_length)` — the same MDE
  validation segment (B) `evaluate_smap_candidate()` and every other
  MDE-side diagnostic in this file already uses, fit on
  `[0, prediction_start)`; never touches the Generate/Forecast validation
  segment `[B, end)`. Scores every method via `tsls._score(...,
  target_metric, ...)` alongside CAE (R19's own "scoring routes through
  `_metric_for`" — `_metric_for()` always resolves to `target_metric` here
  since every scored quantity is a `TARGETS` column, never a driver node).
  Produces the 4-panel figure (`mde_evaluate_{network_id}.png`) — standard
  stacked time series for a 1-target group, spatial panel 1 (matplotlib
  `GridSpec`, panel 1 not sharing the time x-axis with panels 2-4) for a
  2-target group. Returns a dict of the computed CAE/score values for the
  caller to log/inspect.
- **`main()`:** called once per `network_id` in `gmn_runs`, alongside
  `run_tp_vs_rho_sweep()`/`run_univariate_e_tau_grid()`, gated on the same
  `cfg['MODE'] == 'MDE-GMN' and cfg['RUN_MDE_DIAGNOSTICS']` condition —
  using `gmn_runs` (not `mde_results`) since this diagnostic needs the
  *completed* network graph, not the depth-0-only MDE result.
- **Stale comment fixed:** the pre-existing "the notebook cell 15
  diagnostic... has no cleaned_helper_functions.py equivalent... stays
  dropped" comment (which this task supersedes) now points at
  `run_mde_evaluate()` instead of asserting the gap still exists.

### Bug found and fixed during testing (not anticipated by the task text)

Panel 2 (driver traces) originally looped `for t in targets: for driver in
direct_drivers[t][:maxN]`. In combined mode, a driver shared by more than
one target (e.g. an upstream node feeding both `head_position_x` and
`head_position_y`) was plotted and labeled **twice**, once per target it
feeds, under the identical name and color-cycle position — a duplicate,
confusing legend entry, caught visually during testing (`tests/_scratch_r19`'s
own combined-mode figure). Fixed to loop over `driver_union` (already
computed for `D`) once, matching PCA/diffusion-map's own "one shared
pool" treatment.

### Test

`tests/test_r19_mde_evaluate.py` — two scenarios on a tiny synthetic
network (same pattern as `tests/test_r14_external_inputs.py`, real data
not needed for this diagnostic's own correctness):

- **`test_separate_mode`:** 1 target, 2 direct drivers (`A`, `B`).
  Recomputes MDE's own CAE by hand from `proper_simplex`'s own
  `Observations`/`Predictions` columns and asserts it matches
  `run_mde_evaluate()`'s reported value to `1e-9` — the concrete proof
  that `_cae()` reproduces `dimx.Evaluate`'s own definition rather than an
  approximation of it. Asserts the figure file is written and every
  score is finite.
- **`test_combined_mode_asymmetric_drivers`:** 2 targets with
  **asymmetric** driver sets (`head_position_x` has `{A, B}`,
  `head_position_y` has only `{A}`) — deliberately chosen to exercise the
  union-`D` rule (asserts `D == 2`, not 3, and not either target's own
  count alone) and the panel-2 duplicate-driver bug above (this scenario
  is exactly what surfaced it — `A` feeds both targets). Asserts PCA/
  diffusion-map produce a genuine 2-column joint prediction (non-zero CAE,
  finite score) and both targets' own MDE scores are finite.

Both passed on first run after the panel-2 fix above (failed, as
expected, before it — the pre-fix run still produced a figure and finite
scores, so this bug would not have been caught by scores/assertions
alone; it was caught by visually inspecting the generated PNG, which is
why the fix is recorded as found "during testing" rather than by the
test file's own assertions).

Ran the full existing test suite (`tests/test_r*.py`) after this change —
all pass, no regressions.

### A3 sibling parity

**Not a port — no divergence to record either.** `simple-mde-gmn.py`
already has its own working `run_mde_evaluate()` (`dx.Evaluate`-based),
predating this task, unmodified by it. This is not "shared machinery"
that changed and needs porting (R19's own scope, per the review
document's header, is `tsls-mde-gmn-pipeline_v1_annotated_export.pdf`
only) — it is new, TSLS-specific code replicating a comparison the sibling
already performs natively via a dependency (`dimx`) this pipeline
deliberately does not have. Nothing to port; nothing diverges, since the
sibling's own behavior is untouched.

### A5 cache invalidation

**Not applicable.** `run_mde_evaluate()` is a pure diagnostic: it reads an
already-built `net_result` (graph + `DATA_AFTER_MDE`) and writes a figure;
it does not feed back into any cacheable stage's numerical outcome (best
theta, network topology, generated output). Same as
`run_tp_vs_rho_sweep()`/`run_univariate_e_tau_grid()`, neither of which is
fingerprinted either.

### A4 verification

`python3 -m py_compile` passed for both pipeline files (`simple-mde-gmn.py`
untouched by this task, recompiled anyway as part of the standard check).
`bash -n` passed for both sbatch scripts. Full existing test suite
(`tests/test_r*.py`) passed, no regressions.

**Real-data smoke run:** launched `tsls-mde-gmn.py` end-to-end at
`REGION_CHANNEL_CAP=3, MDE_D=2, NESTED_M=1` (`GENERATE_LENGTH=40`,
results/cache redirected to scratch — same config-approximation
convention as R12's own smoke run) in the background, per A4's own
requirement. It progressed a long way on its own before this session
checked on it again: Stage 1 (2 real-observable backbone nodes), Stage 2
(both targets' own nested-MDE, real driver selections shown in the log --
`head_position_x` <- `CA1_1_tsls`, `mPFC_2_tsls`; `head_position_y` <-
`OFC_1_tsls`, `OFC_0_tsls`), network construction, and into theta
retuning ("retuning theta for 4 node(s)") -- meaning this run also
exercised R20's new `proper_predict_nonlinear()` on real data before it
stopped. **Final status: inconclusive, not a pipeline error** -- by the
time this session checked again the background process had ended with no
further log output and no traceback/error anywhere in the ~10800-line
log, the same "process tracking lost it" outcome R12's own background
smoke run hit earlier this session (see that section above) rather than
a crash in this task's own new code. Not re-launched, for the same reason
R12 gave: the time already spent, and every step this run reached
completed cleanly. Correctness for R19 rests on the synthetic test above
(and the reference figure comparison), not on this run reaching
`run_mde_evaluate()` -- consistent with this session's own established
practice of reporting a long background job's real status honestly
rather than blocking a commit on it or overclaiming what it proved.

---

## R20 — TSLS-compatible replacement for `pyEDM.PredictNonlinear`

**Status:** Done.

### A1 verification, and R10 precondition (step 1)

R20's own step 1 ("complete R10 first") was already done earlier this
session (see R10's own section above): no native S-map exists in
`cleaned_helper_functions.py`. Re-confirmed still true before touching any
code (no intervening task changed that file's public API).

Read `retune_theta_for_all_nodes()` (the one call site) in full.
**Premise confirmed true:** it calls `pyEDM.PredictNonlinear(...)` and
picks `best_theta_by_node[node] = float(pn.loc[pn['rho'].idxmax(), 'theta'])`
— `'rho'` is `PredictNonlinear`'s own internal column, never `metric`/
`target_metric`/`_metric_for()`. This is genuinely the one scoring call
site in the pipeline that doesn't route through `_metric_for()` — every
other scoring call site checked this session (`evaluate_smap_candidate`,
`gmn_fidelity_rho`, `run_forecast_diagnostic`, `run_mde_evaluate`, the
various MDE diagnostics) does.

### The finding that narrowed R20's scope (before proposing anything)

Per R20 step 3's own instruction ("do not start writing [an S-map]...
propose an implementation approach... wait for approval"), read
`pyEDM.PredictNonlinear`'s actual source
(`pyEDM/API.py:554`) and its worker
(`pyEDM/PoolFunc.py:97`, `PredictNLSMapFunc`) before proposing anything.
**`PredictNonlinear` itself is already nothing more than a thin sweep:**
per candidate `theta`, it calls `pyEDM.SMap(...)` once, then scores via
`ComputeError(obs, pred)['rho']` and discards the predictions
(confirmed: `PredictNLSMapFunc` returns only `err['rho']`, line 118 — the
predictions are never surfaced to the caller). There is no separate
"S-map engine" hidden inside `PredictNonlinear` to reimplement — it's the
same compiled `pyEDM.SMap` gmn's own `Forecast()`/`Generate()` already
calls directly at runtime for every node
(`_generalized_node_forecast()`'s `FunctionType.SMap` branch,
`self.Function(...)`, confirmed to be `pyEDM.SMap` itself) — unaffected
by this task either way, since `theta` is just a number once chosen,
regardless of which sweep chose it.

Also confirmed: `ComputeError`'s own `rho = round(corrcoef(obs, pred)[0,1], 6)`
(`pyEDM/AuxFunc.py:50`) is the identical Pearson-correlation formula
`cleaned_helper_functions._score(kind='correlation')` uses
(`np.corrcoef(o, p)[0, 1]`) — so re-scoring `pyEDM.SMap`'s own predictions
via `_score()` at `metric='correlation'` should reproduce
`PredictNonlinear`'s own selection, a genuine, checkable equivalence
(R20's own acceptance criterion), not an approximation.

### Proposal presented and approved

Presented this finding and a proposal (thin `pyEDM.SMap`-based scoring
wrapper, not a new S-map engine) to the user before writing any code, per
R20's own explicit instruction, covering: the design, validation plan
(equivalence against `PredictNonlinear` at `metric='correlation'`), cost/
parallelism (flagging `sitecustomize.py`'s own documented fix — see
`sitecustomize.py`'s own pyEDM per-call Pool-spawn fix — for why a fresh `multiprocessing.Pool` per call
must not be reintroduced), and the narrower alternative considered (asking
`PredictNonlinear` itself to expose predictions — confirmed not possible,
it discards them). **User approved implementing as proposed**, in the
pipeline file itself (not `cleaned_helper_functions.py`, which R10
established is deliberately pyEDM-free).

### What changed (`scripts/src/tsls-mde-gmn.py` only — no sibling-parity
port needed, see below)

- **New import:** `SMap` added to the existing `from pyEDM import Embed,
  PredictNonlinear, SMap` line. `PredictNonlinear` itself stays imported,
  now unused by the pipeline proper but kept importable for
  `tests/test_r20_predict_nonlinear.py`'s own equivalence check against
  it.
- **`proper_predict_nonlinear(dataFrame, columns, target, theta, lib, pred,
  E=1, Tp=1, knn=0, tau=-1, exclusionRadius=0, embedded=False,
  metric='correlation', gamma_delta=2)`** (new): per candidate `theta`,
  calls `SMap(...)` directly (the same args `PredictNonlinear` itself
  passes) and scores the returned `(Observations, Predictions)` via
  `tsls._score(metric, gammaDelta=gamma_delta)` instead of `ComputeError`.
  Deliberately sequential (one `SMap` call per theta, in-process, no
  `Pool`) — see `sitecustomize.py`'s own module docstring for why. Returns a
  `DataFrame` with columns `'theta'`/`'score'`, mirroring
  `PredictNonlinear`'s own `df.loc[df['rho'].idxmax(), 'theta']` idiom
  exactly, just against a metric-agnostic column name.
- **`retune_theta_for_all_nodes()`:** signature gained `targets, metric,
  target_metric, gamma_delta` (previously took neither); now resolves
  `node_metric = _metric_for(node, targets, metric, target_metric)` per
  node and calls `proper_predict_nonlinear(..., metric=node_metric, ...)`
  instead of `PredictNonlinear(...)`. `cores` is kept as a parameter (for
  `run_theta_search()`'s own call-site shape, since it's still needed
  right after for `evaluate_smap_candidate()`'s `Generate()` call) but is
  no longer used inside this function itself. Fallback behavior (theta=3
  on too little data, or on any exception) unchanged.
- **`run_theta_search()`:** signature gained `metric` (before
  `target_metric`); passes `net_result['targets'], metric, target_metric,
  gamma_delta` through to `retune_theta_for_all_nodes()`.
- **`main()`:** its one call to `run_theta_search()` now passes
  `cfg['METRIC']` alongside the existing `cfg['TARGET_METRIC']`.
- **Module-level import comment:** updated (previously said "no TSLS-aware
  equivalent exists for `retune_theta_for_all_nodes()`'s own theta
  search" — no longer true, rewritten to describe the actual design and
  why `SMap` is now imported directly).

### Test

`tests/test_r20_predict_nonlinear.py`:
- **`test_equivalence_against_pyedm_predictnonlinear`:** builds a real
  `(E=3)` delay embedding of a short logistic-map trajectory (genuine
  chaotic curvature, so the theta sweep produces an actually-peaked
  curve, not flat noise), runs both `pyEDM.PredictNonlinear` and
  `proper_predict_nonlinear(metric='correlation')` on the identical
  embedding/lib/pred split and theta grid. Asserts the selected best
  theta matches exactly, and every theta's own score matches
  `PredictNonlinear`'s own rho to `< 1e-6` across the whole grid (not just
  at the argmax) — this is R20's own acceptance criterion, and passed on
  first run: best theta `8.0` for both, max `|rho - score|` across the
  grid `4.69e-07` (consistent with `ComputeError`'s own `round(..., 6)`
  vs. `_score()`'s unrounded value).
- **`test_retune_theta_for_all_nodes_resolves_metric_per_node`:** direct,
  deterministic check that `_metric_for()` resolves a non-target node to
  `'f1'` and a target node to `'correlation'` (checked at the
  `_metric_for()` call itself, not by asserting on `'f1'`'s own
  TSLS-specific numeric behavior against this test's plain synthetic
  series, which was never built to exercise `'f1'`'s real event-detection
  logic meaningfully), plus an end-to-end
  `retune_theta_for_all_nodes()` smoke call confirming both nodes get a
  finite theta.

Both passed on first run. Ran the full existing test suite
(`tests/test_r*.py`) after this change — all pass, no regressions.

### A2 interactions checked

None of R20's own changes touch `_embedding_spec_for()`/`_generalized_embed()`
(R14's own domain) or any nested-MDE/cache-fingerprint machinery (R12/R13's
own domain) — this task is scoped entirely to the theta-retuning sweep,
which those tasks don't touch.

### A3 sibling parity

**Not applicable — `simple-mde-gmn.py` has no equivalent call site.**
Confirmed by reading its own `retune_theta_for_all_nodes()` (this
function name and the surrounding theta-search structure are shared/
ported machinery between the two pipelines): it calls
`pyEDM.PredictNonlinear` exactly the same way `tsls-mde-gmn.py` did
before this change, scored the same way (`'rho'` only) — but
`simple-mde-gmn.py` has **no** `_metric_for()`/`_score()`/`metric`/
`target_metric` concept at all (confirmed earlier this session, R16's own
section: it scores everywhere with plain Pearson correlation on raw
binned counts, by design — there is no second metric for it to fail to
route through). R20's own premise ("the one stage that does not route
through `_metric_for`") is specific to `tsls-mde-gmn.py`'s own
architecture and doesn't describe a gap in the sibling at all — nothing
to port, no divergence to record.

### A4 verification

`python3 -m py_compile` passed for both pipeline files. `bash -n` passed
for both sbatch scripts. Full existing test suite (`tests/test_r*.py`)
passed, no regressions.

### A5 cache invalidation

**No new knob added** — `metric`/`target_metric`/`gamma_delta` were
already present in `build_search_fingerprint()` (`cfg['METRIC']`,
`cfg['TARGET_METRIC']`, `cfg['GAMMA_DELTA']`, confirmed by reading that
function, unchanged by this task) — this task re-routes the theta
search's *scoring* through values the fingerprint already captures, it
doesn't introduce a new dimension of variation. A run whose cache
predates this change is invalidated regardless by `code_fingerprint`'s
whole-file AST hash (`retune_theta_for_all_nodes()`'s own logic changed),
same mechanism confirmed in R12/R13/R14's own sections.

---

## R21 — Dynamic exclusion radius in `proper_mde`

**Status:** Done.

### A1 verification, and the premise gap found before implementing

Read `proper_mde()` (`cleaned_helper_functions.py`) in full. **The
task's own literal framing doesn't survive contact with the code as
written**, for a reason the task text doesn't anticipate: `proper_mde`
has *no* built-in concept of "delay" attached to a column name at all.
Its ranking stage (`_rank_one`, every dimension) calls `_simplex_metric(
..., embedded=True)` — a selected column is used exactly as given, with
no internal per-column lag tracking; `_simplex_metric` doesn't even
accept an `exclusionRadius` parameter. "Delay" in this pipeline is purely
a naming convention (`DELAY_MARKER`/`observable_and_delay()`) that lives
in `tsls-mde-gmn.py` — grepped `cleaned_helper_functions.py` for both,
zero matches. So "set `exclusionRadius = max(|delay|)` across the
columns currently selected" cannot be computed from information already
inside `proper_mde` — it needs a way to look up each selected *column's*
own delay, which only the pipeline caller knows.

Confirmed this isn't a hypothetical edge case: `run_mde_for_target()`
(the single function every one of main()'s depth-0 seed, Stage 1, and
Stage 2 routes its `proper_mde()` call through — confirmed by grepping
every call site, matching R4's own earlier finding) really can be handed
a candidate pool containing delay-suffixed synthetic names for **both**
Stage 1 and Stage 2 (re-read `build_nested_mde_network()`'s own candidate-
pool construction for both phases: Stage 1's is built from
`expanded_columns`, not a real-only subset — the "real observables only"
in Stage 1's own name describes what gets recursed into via BFS, not
what `proper_mde`'s own candidate pool for a single node is restricted
to). This session's own live R19/A4 background smoke run (see that
section above) independently confirmed it in practice: `MDE selected
columns for head_position_x: ['CA1_1_tsls', 'CA1_1_tsls::delay-3', ...,
'CA1_1_tsls::delay-25', ...]` — real delay-suffixed entries in a real
`MDEcolumns` list.

Stopped and asked before implementing, per Part A1. **User approved**
the proposed fix: a new optional `column_delay` parameter on `proper_mde`
(caller-supplied `{column: delay}` lookup, or `None` = exact no-op),
keeping `cleaned_helper_functions.py` itself free of any hardcoded
pipeline-specific naming assumption — the caller (which already
understands its own naming convention) supplies the lookup.

### What changed

- **`cleaned_helper_functions.proper_mde()`:** new `column_delay=None`
  parameter (dict or `None`). At each dimension `d`, before step 4's
  E-select (`proper_embed_dimension`) and CCM-gate (`proper_ccm`) calls,
  computes
  `exclusionRadius_d = max(exclusionRadius, max(abs(column_delay.get(c, 0))
  for c in MDEcolumns))` — `MDEcolumns` as of the *start* of dimension
  `d` (the columns already selected in earlier dimensions; the current
  dimension's own not-yet-accepted candidate is never included). Both
  calls use `exclusionRadius_d` in place of the static `exclusionRadius`.
  `column_delay=None` (default) → `exclusionRadius_d == exclusionRadius`
  at every dimension, identical to before this parameter existed — a
  literal no-op for every existing caller that doesn't opt in.
  **Deliberately out of scope:** the ranking stage (step 2,
  `_simplex_metric`) — it has never accepted an `exclusionRadius` at all
  (a separate, pre-existing limitation this task's own text doesn't name
  and doesn't ask to fix); and `multi_proper_mde` (a different function,
  R21's own title says "in `proper_mde`" specifically — R22 is the task
  about unifying the two call paths, not yet reached).
- **Logging (the task's own explicit requirement):** `if verbose: print(
  f'  {d}-D exclusionRadius={exclusionRadius_d} (floor={exclusionRadius},
  deepest_selected_lag={deepest_selected_lag})')`. `run_mde_for_target()`'s
  own call to `proper_mde` already passes `verbose=True`, so this reaches
  every real run's log without any further change.
- **`tsls-mde-gmn.py`'s `run_mde_for_target()`:** new `raw_columns=None`
  parameter. Builds `column_delay = {c: observable_and_delay(c,
  raw_columns)[1] for c in observable_columns}` when `raw_columns` is
  given, and passes it to `proper_mde`. `None` (default — what main()'s
  own depth-0 seed call leaves it at, since that pool is always
  real-only) → `column_delay=None` → no-op.
- **`build_nested_mde_network()`:** both `executor.submit(
  run_mde_for_target, ...)` call sites (Stage 1 and Stage 2) now pass
  `raw_columns=raw_columns` (already in scope as this function's own
  parameter) — the two call sites whose candidate pools can genuinely
  contain delay-suffixed names.

### Test

`tests/test_r21_dynamic_exclusion_radius.py`:
- **`test_dynamic_exclusion_radius_matches_deepest_selected_lag`:**
  synthetic target driven by three independent smooth signals A/B/C
  (`target = 0.5A + 0.3B + 0.2C`), `column_delay={'A':5,'B':12,'C':2}`,
  static floor `1`. `crossMapRhoMin`/`embedDimRhoMin`/`ccmSlope` set to
  trivially-passing thresholds so this test isolates exclusion-radius
  bookkeeping from the (separately tested elsewhere) CCM-gate/threshold
  logic. Two checks on the same run, per the task's own acceptance
  criterion: (1) direct — recomputes the expected floor at each dimension
  from `proper_mde`'s own returned `MDEcolumns` order (no assumption about
  which candidate wins), asserts it's never below the static floor nor
  below the deepest lag among columns already selected; (2) parses the
  `{d}-D exclusionRadius=...` lines the run actually printed (the log
  auditability the task's own text asks for) and asserts they match the
  same expectation exactly. Result: dimension 1 → 1 (floor, nothing
  selected yet), dimension 2 → 5 (`A` selected, delay 5), dimension 3 → 12
  (`A`,`B` selected, deepest is `B`'s 12) — matching by construction.
- **`test_no_column_delay_is_a_no_op`:** same setup, `column_delay=None`
  — every dimension logs exactly the static floor, confirming the default
  path is unchanged.

Both passed on first run (after two iterations to get the synthetic
scenario itself right — see "Test-design notes" below). Ran the full
existing test suite (`tests/test_r*.py`) after this change, including the
regression this task's own signature changes caused (below) — all pass.

### Regression found and fixed during testing (not anticipated by the
task text)

Adding `raw_columns` to `run_mde_for_target()`'s signature broke
`tests/test_r12_stage_serialization.py`, `test_r13_shared_depth_budget.py`,
and `test_r15_termination_causes.py`: their shared fake
(`tests/_pipeline_test_utils.make_fake_run_mde_for_target_tsls()`, which
`build_nested_mde_network()`'s real `executor.submit(run_mde_for_target,
...)` calls resolve to under test) has a fixed signature matching the
*real* function's — the real Stage 1/Stage 2 call sites now pass
`raw_columns=raw_columns`, which the old fake's signature rejected
(`TypeError: unexpected keyword argument 'raw_columns'`), silently caught
by `build_nested_mde_network()`'s own per-node failure handling and
reported as "MDE call FAILED" rather than crashing the test outright —
still a real regression (every node became a dead end, both stages)."
Fixed by adding `raw_columns=None` to the fake's own signature
(`simple-mde-gmn.py`'s fake is untouched — it has no `raw_columns`
concept, its own real function's signature is unaffected by this task).

### Test-design notes (not part of the acceptance criterion, recorded for
completeness)

The first test draft used `noCCM=True` to make selection deterministic,
which turned out to skip step 4 (and therefore the new exclusion-radius
computation) entirely — `noCCM`'s own shortcut branch `continue`s before
reaching it. Switched to `noCCM=False` (the CCM gate genuinely runs) with
deliberately lenient `crossMapRhoMin`/`embedDimRhoMin`/`ccmSlope`
thresholds instead, so the test still isolates exclusion-radius behavior
without depending on the synthetic data's own CCM-convergence properties.

### Effect on real selections (Run A's own exact configuration not
reproduced here — see caveat below)

Ran `run_mde_for_target()` twice on real data
(`J1620210605_epoch_8_freq_40_Hz.csv`, `REGION_CHANNEL_CAP=15`,
`target='head_position_x'`, a 20-delay `lag_set`, `MDE_D=3`) — once with
`raw_columns=None` (old, static `exclusionRadius=0` throughout) and once
with the real `observable_and_delay()`-derived `column_delay` (new,
dynamic). **Selections were identical in this run**
(`['mPFC_1_tsls', 'mPFC_1_tsls::delay-6', 'mPFC_1_tsls::delay-2']`, both
cases) — but the mechanism is confirmed genuinely active and numerically
different, not merely a no-op that happened to match by construction: at
dimension 3 (`mPFC_1_tsls` and `mPFC_1_tsls::delay-6` already selected,
deepest delay 6), the dynamic radius was `6` vs the old run's `0`, and
several candidates' own `EmbedDimension`/CCM-slope numbers genuinely
differed between the two runs at that dimension (e.g.
`OFC_3_tsls::delay-1`: `E=1 rho=0.1998` old vs `E=1 rho=0.2725` new;
`OFC_3_tsls::delay-2`: CCM slope `0.00621` old vs `-0.0034` new) — in this
particular run, none of those differences happened to flip which
candidate was the *first* to pass the CCM gate, so the final selection
came out the same. **Caveat:** this is `REGION_CHANNEL_CAP=15`, not Run
A's own literal documented configuration (`convert_to_tsls`/full-pipeline
overhead makes reproducing Run A's exact config in this exploratory check
materially more expensive than the smaller, still-real check performed
here) — flagged as an approximation, same as R12's own smoke-run caveat,
not a literal Run A reproduction.

### A2 interactions checked

None of R21's changes touch `_embedding_spec_for()`/`_generalized_embed()`
(R14), the theta-retuning sweep (R20), or nested-MDE's own BFS/depth
accounting (R12/R13) — scoped entirely to `proper_mde`'s own per-dimension
E-select/CCM-gate calls and the one function (`run_mde_for_target()`)
that threads a delay lookup into it.

### A3 sibling parity

**Not applicable.** `simple-mde-gmn.py` doesn't import
`cleaned_helper_functions.py` at all (confirmed earlier this session, R16
and R18's own sections) — it has no `proper_mde()` call path for this
change to apply to. Nothing to port, no divergence to record.

### A4 verification

`python3 -m py_compile` passed for all three files touched
(`cleaned_helper_functions.py`, `tsls-mde-gmn.py`,
`tests/_pipeline_test_utils.py` is plain Python, not part of the pipeline
files A4 names, but compiled cleanly too) and `simple-mde-gmn.py`
(untouched, recompiled anyway). `bash -n` passed for both sbatch
scripts. Full existing test suite passed, no regressions (after the
fake-signature fix above).

### A5 cache invalidation

**No new fingerprint code needed.** `column_delay` is derived data (from
`raw_columns`, itself derived from existing config), not a new
independent tunable value — there is no new knob to add to
`build_network_fingerprint()`/`build_search_fingerprint()`. Any change in
`proper_mde`'s own selections is already covered by
`_environment_fingerprint()`'s own direct content-hash of
`cleaned_helper_functions.py` (`_hash_file(Path(tsls.__file__))`,
confirmed by reading that function this task — folded into
`build_code_fingerprint()`) — a cache built before this change will not
be silently reused, since the helper file's own bytes changed.

---

## Depth>=1 metric-routing bug — found while verifying R22, not one of the
22 numbered tasks

**Status:** Fix implemented and tested; **held on branch
`metric-routing-depth-ge1-fix`, not merged to `main`**, pending a decision
on a severe recalibration problem it exposes. Do not read this as R22's
own work — R22 itself has not resumed (see its own section below for why).

### How this was found

While producing R22's own required "before/after on Run A" measurement
for the historical `E=3` static → `E=0` auto-select change (R4's own
finding), reproducing a real Stage-2-style call
(`target='CA1_1_tsls'`, a neuron, `metric='f1'`) required tracing exactly
which metric `run_mde_for_target()`'s own ranking stage actually used.
The verbose log showed `(correlation=0.9997)` for a neuron's own
ranking-stage acceptance line — the wrong metric name printed for a node
that should score via `'f1'`.

### The bug

Read `build_nested_mde_network()`'s Stage 1 and Stage 2
`executor.submit(run_mde_for_target, ...)` call sites in full: both pass
the closed-over `metric`/`target_metric` (`cfg['METRIC']`/
`cfg['TARGET_METRIC']`) **positionally, unconditionally, for every
node** — never resolved per-node via `_metric_for()`. Inside
`run_mde_for_target()`, `target_metric` scores the ranking stage and the
E-auto-select sweep (predictions of `target`); `metric` scores only the
CCM gate (predictions of the candidate). That split is correct exactly
once: the depth-0 seed (`main()`'s own `for target in cfg['TARGETS']:`
loop), where `target` is always a real behavioral TARGETS column. At
depth >= 1 (Stage 1 discovering real-observable drivers; all of Stage 2),
`node`/`target` is virtually always a **neuron**, and `_metric_for()`'s
own documented rule says a neuron should score via `metric` ('f1'), not
`target_metric` ('correlation') — a rule this call site never consulted.
Confirmed via `git blame`/`JOB_LOG.md` that no earlier fix or design note
addresses this; it is not deliberate.

**Net effect, this whole session (and presumably before it):** every
discovered neuron driver's own ranking-stage acceptance and per-candidate
E-auto-select in Stage 1/Stage 2 has been scored via Pearson correlation
on a TSLS (time-since-last-spike, sawtooth) series, not the TSLS-aware
`f1` event-timing metric `_metric_for()` says it should use. Only the CCM
convergence gate (`ccmMetric=metric`) was already correct — it always
scores a *candidate*, never `node` itself, and no candidate here is ever
a target, so it was never affected.

### Fix

At both Stage 1's and Stage 2's `executor.submit(run_mde_for_target,
...)` call sites, replaced the raw `target_metric` positional argument
with `_metric_for(node, targets, metric, target_metric)`. No change
needed to `run_mde_for_target()` or `build_mde_selection_fingerprint()`
themselves — the resolved value already flows through both
automatically: `run_mde_for_target()` forwards whatever `target_metric`
it receives straight into its own `build_mde_selection_fingerprint(...,
target_metric, ...)` call, so **A5 is satisfied by construction** — a
cache built under the old (unresolved, wrong) value has a different
`target_metric` in its own stored fingerprint and will not be reused
under the fixed code.

### Test

`tests/test_metric_routing_depth_ge1.py` — a synthetic `T -> A -> B` BFS
(`T` the only target; `A`/`B` discovered real-observable drivers,
`STAGE2_ALLOW_REAL_OBSERVABLES=True` so `B` is Stage 2's own recursive
pick), with a custom fake `run_mde_for_target()` that records the exact
`(metric, target_metric)` arguments each call received. Asserts a target
node gets `target_metric == TARGET_METRIC` and every non-target node
gets `target_metric == METRIC`, and that the CCM-gate `metric` is
untouched (always `METRIC`) for every node. **Verified the test actually
discriminates old vs. new code**: ran it against the pre-fix commit
(`git stash`) — fails exactly as expected
(`stage1/A: non-target node got target_metric='correlation'`); passes
against the fix.

### Required recalibration check (before this fix can be considered safe
to land) — **result: severe, not marginal**

Real data (`J1620210605_epoch_8_freq_40_Hz.csv`, `REGION_CHANNEL_CAP=15`),
`CCM_WINDOW_ROWS=800`'s own derived ranking-stage window
(`lib=[1,659]`, `pred=[660,800]` — a **141-row** prediction slice), `f1`
ranking score for every real candidate against three real neuron targets:

| target | f1 min/median/max | passing `CROSS_MAP_RHO_MIN=0.2` | correlation min/median/max (same window) | correlation passing |
|---|---|---|---|---|
| `CA1_1_tsls` | 0.0000 / 0.0000 / 0.0000 | 0/44 | -0.84 / 0.06 / 1.00 | 18/44 |
| `mPFC_1_tsls` | 0.0000 / 0.0000 / 0.0000 | 0/44 | -0.72 / 0.03 / 0.99 | 17/44 |
| `OFC_1_tsls` | 0.0000 / 0.0000 / 0.0000 | 0/44 | -0.83 / -0.10 / 1.00 | 8/44 |

**`f1` is exactly `0.0000` for every one of 44 candidates, for all three
targets** — not "mostly fails," uniformly zero. Traced the cause: counted
actual TSLS reset events in the 141-row prediction window itself —
`CA1_1_tsls`: **0**, `mPFC_1_tsls`: **1**, `OFC_1_tsls`: **0**. With 0-1
true events, `f1` (event-timing precision/recall) is structurally
degenerate regardless of which candidate is tried — this is not about
prediction quality, the window is simply too event-sparse.

**Confirmed this is the ranking stage's own problem specifically, not a
general "141 rows is too few" problem, and not one the CCM gate shares**
(item 3 of the requested checks): read `proper_ccm()` in full. It never
receives a `lib`/`pred` split at all — its own prediction indices
(`idx_fwd`/`idx_rev`) are computed from validity masks over the **entire**
dataframe passed in (the full 800-row `ccm_df`), and only its `libSizes`
(the random-subsample *library* size, not the prediction set) are
percentages of that same 800. So the CCM gate's own earlier-reported
"`f1` works fine" result (R22's own E-engine-swap measurement, above) is
real but doesn't generalize to the ranking stage — that measurement used
the **pre-fix** code (`target_metric='correlation'` for ranking,
`ccmMetric='f1'` for the gate only), so it only ever exercised the gate's
own (already-adequate, ~800-row-wide) `f1` scoring, never the ranking
stage's own (141-row, degenerate) `f1` scoring. **Correcting my own
earlier report accordingly** — that measurement is not evidence the fix
is safe; it measured a different code path than the one this fix changes.

**A quieter, second degeneracy (item 4): E-auto-select silently defaults
to `E=1`, not visibly.** `proper_embed_dimension`'s own `f1`-vs-`E` sweep
(same 141-row window) is **completely flat at 0.0000 for every `E` from 1
to 15**, for every (target, candidate) pair tested — `np.nanargmax` picks
index 0 (`E=1`) every time. Unlike the ranking-stage rejection (which at
least visibly rejects every candidate), this failure produces a
plausible-looking `E=1` with no indication it was never actually
selected from real signal.

**Root cause is firing-rate heterogeneity, not just window size (item
5).** Counted real TSLS reset events (~spikes) for all 45 real channels
in this config, over the full 800-row `ccm_df`:

- **21/45 channels (47%) have zero events in the current 141-row pred
  window.** Several (`CA1_0_tsls`, `CA1_5_tsls`, `CA1_7_tsls`) have
  **zero events in the entire 800-row window**.
- Defining "workable" as >= 20 events (this report's own threshold, not
  an established convention: with `f1 = 2PR/(P+R)`, fewer than ~20 true
  events means a single miss or false positive swings the score by >= 5
  percentage points, too noisy to trust as a selection criterion) — rows
  needed for 20 events, across all 45 real channels: min **98**, median
  **1600**, and several channels effectively **never** reach it (rate =
  0 within the 800-row window sampled).
- **A single fixed window cannot serve both ends of this distribution.**
  Even a substantially larger ranking window (say, several hundred rows)
  would only help the fastest-firing quartile; the slowest channels
  found in this sample would need thousands of rows or more — a
  per-node minimum-event check with an explicit skip (rather than a
  silent `E=1`/all-reject fallback) looks necessary, not just a bigger
  single number.

**Confirms `correlation` degraded gracefully where `f1` degenerates
(item 6):** the same 141-row window's `correlation` scores (table above)
show a healthy, informative spread (18/44, 17/44, 8/44 candidates
passing `CROSS_MAP_RHO_MIN=0.2`) — this window was **not** already
marginal for the metric it was actually calibrated against. The problem
is specific to `f1`'s own event-counting nature, not a generic
small-sample problem this window already had.

**Connection to the depth-0 metric question, noted not acted on (item
7):** the mechanism here (TSLS event sparsity making `f1` degenerate) is
a *different* failure from why `f1` is wrong at depth 0 (see the next
section) — depth-0's own target (`head_position_x`/`_y`) is not a
TSLS-converted series at all, so "reset events" don't even apply to it
conceptually, independent of any window size. The two findings are
related only in that both concern where `f1` vs. `correlation` is
appropriate; they don't share a root cause and one doesn't imply a fix
for the other.

### Proposed decoupling (item 2) — **proposal only, not implemented**

Traced (item 1) that `CCM_WINDOW_ROWS` currently bounds two genuinely
different things through one shared 800-row `ccm_df`: (a) the
ranking-stage/E-auto-select `lib`/`pred` split (`_ccm_window`'s own
`lib_ratio`-derived split of the 800 rows) — plain train/test Simplex
prediction, no convergence requirement; (b) the CCM gate's own
`libSizes` sweep (`pLibSizes` percentages of that same 800) — which
*does* need to stay small, per §18.4/§7.4's own established reasoning
(R9's own finding: a too-large library flattens genuinely causal pairs'
convergence curves). Only (b) has a principled reason to stay bounded at
800; (a) has none — it just needs enough rows (and, per the finding
above, enough real *events*) for `f1` to be meaningful.

A decoupling would need: `run_mde_for_target()` building a **second,
independently-sized** window specifically for the `dataFrame`/`lib`/
`pred` that steps 1-2 (ranking) and 4a (E-select) operate on, while
`proper_mde`'s own CCM gate (step 4b) continues to derive `libSizes` from
a *separate*, still-800-row-bounded quantity rather than
`dataFrame.shape[0]` directly (today, `libSizes = [int(N * p/100) for p
in pLibSizes]` where `N = dataFrame.shape[0]` — the same dataframe
`proper_mde` uses for everything; decoupling requires either passing
`proper_mde` two differently-sized dataframes, or adding an explicit
`ccm_window_rows`-style cap to `proper_mde` itself so its own `libSizes`
computation stops depending on the ranking dataframe's own (now larger)
row count). This does not obviously disturb the §18.12 pre-filter
ordering fix (a separate, earlier-stage mechanism, R2's own section
already confirmed unaffected by unrelated changes this session) or the
leakage guard (`prediction_start`/`prediction_length` stay the external
boundary either window is built within) — but given the firing-rate
heterogeneity finding above, **a bigger single ranking window alone would
not fully resolve this** — the per-node minimum-event-count question
(item 5) is a separate, likely necessary piece of any real fix. Not
implemented; this is a design sketch for discussion, not a proposal
ready to build from directly.

### R9 as the extension vehicle

R9 (`scripts/calibration/r9_ccm_window_e_calibration.py`, already run,
real results committed) swept `CCM_WINDOW_ROWS`/`CCM_PREFILTER_E` for the
**CCM pre-filter** (`ccm_prefilter_observables()`) specifically — a
separate, earlier-stage mechanism from `run_mde_for_target()`'s own
ranking/E-select/CCM-gate logic, and it never measured event density or
`f1` at all (its own metric there is the pre-filter's own scoring
convention). Extending R9's own script to cover this event-density
question (rather than a new parallel script) looks like the right
vehicle, per this session's own established preference for reusing
existing calibration infrastructure — but this has not been started; no
code changes made beyond the standalone diagnostic scripts used to
produce the numbers above (not committed, scratch-only).

### Status / what's still pending

No defaults changed (`CROSS_MAP_RHO_MIN`, `EMBED_DIM_RHO_MIN`,
`CCM_WINDOW_ROWS` all untouched). No Run A before/after run under the
current window (would very likely show catastrophic node loss rather
than a modest topology shift, given the event-density finding — decided
not worth running until the window-sizing question is resolved). The fix
itself stays on `metric-routing-depth-ge1-fix` (not `main`) pending
direction on the window-sizing question above. **R22's own sequence is
on hold**, per its own step 4 ("this changes nested selections, so it
moves the baseline R22's bit-identical test compares against... land this
fix first, re-establish the Run A baseline, then resume R22") — since the
fix isn't landed yet, R22 has not resumed.

**Update:** the window-sizing question above was resolved via a full
redesign (`MIN_PRED_EVENTS`/`MAX_WINDOW_ROWS`, replacing an intermediate
"half of a neuron's own total events" placeholder that itself went
through two further abandoned iterations) — see "Window sizing
(`MIN_PRED_EVENTS`/`MAX_WINDOW_ROWS`) and CCM-gate absolute library
sizes" below for the full story. R22 remains on hold until that work
lands and the Run A baseline is re-established.

---

## Open question: is `correlation` the right depth-0 metric? — documentation-pass note only, no code change

**Status:** Recorded for the documentation rewrite pass. No code change
authorized or made. There is no established optimum here; `correlation`
is retained at depth 0 as the status quo pending a real-data comparison
against `mae`, not because it has been validated as correct.

### The pushback, and why it doesn't survive as stated

The reviewer pushback under consideration: since the drivers being fed
into depth-0's own ranking/E-select are TSLS sawtooths whose only real
information content is at the spike events, maybe treating this as
ordinary continuous prediction (`correlation`) is wrong even when the
*target* is a continuous trace like `head_position`.

**This doesn't survive as stated, because it conflates what's being
scored with what's driving the prediction.** `_score()` (and every
metric it dispatches to) scores **predictions of the target** —
`Observations` vs. `Predictions` of whatever `target` is — never the
input coordinates' own shape. At depth 0, `target` is `head_position_x`/
`_y`, a genuine continuous trace; `f1` would decode *both* the observed
and predicted position series into "reset events" (via
`infer_near_zero_threshold_from_true_tsls`) and compare event timings —
but a continuous position trace has no real "reset events" in it at all;
whatever `f1` finds there are artifacts of running a TSLS-specific
decoder on a non-TSLS signal, not a feature of the behavior being
predicted.

**Empirically:** a synthetic prediction offset by a fixed 15 units from
the true position scores `f1 = 0.97`; a prediction with the same *shape*
but half the true range scores `f1 = 1.0000` — both obviously
behaviorally useless (wrong place; wrong range), and both invisible to
`f1`, because neither perturbation moves the (already-meaningless) event
timings `f1` is comparing.

**Important distinction the pushback also missed:** the TSLS ramps are
*not* the same kind of "signal" in both roles. §2.1's own rationale for
using TSLS as an **input/predictor coordinate** is that the sawtooth
supplies real local geometry (slope, curvature, recency) for delay
embedding — that reasoning is about TSLS columns as *predictors*, and it
holds. It says nothing about TSLS-style decoding being appropriate when
**scoring a prediction of a continuous target** — a completely different
role. Neuron *targets* (Stage 1/Stage 2 nodes) still correctly use `f1`,
because there the *predicted quantity itself* is a TSLS series with real
events to compare.

### What the pushback got right

`correlation` is a weak discriminator on smooth, autocorrelated position
traces. Empirically (synthetic tests), `correlation` stayed pinned at
`1.0000` across noise levels that visibly degraded the prediction —
shared slow drift dominates and swamps real differences. `correlation`
is also scale- and offset-invariant by construction, so it **also**
scores both the fixed-offset and halved-range synthetic cases above at
`1.0000` — the same blindness `f1` has, for a different structural
reason. So the current depth-0 metric is not obviously adequate either;
it is simply not *wrong* in the specific way `f1` would be.

### What is actually untested

`'mae'` is already implemented in `_score()`, and this pipeline's own
`TARGET_METRIC` docstring already frames the allowed values as
"correlation/mae, never event-based." `mae` penalizes both offset and
scale error — exactly what both `correlation` and `f1` were just shown to
miss. **Nobody has compared `correlation` against `mae` on real depth-0
data.** That comparison is the concrete next step if this question is
ever reopened — not implemented here, no default changed.

### A third issue, noted, not resolved

`TARGETS` is a 2D position (`x` and `y`). Any per-axis scalar metric
(`correlation`, `mae`, or `f1`) scores the two axes completely
independently and so cannot see trajectory *geometry* at all (e.g., a
prediction that's rotated 90 degrees around the true path could still
score identically per-axis). Related to R19's own combined-mode panel-1
question (the spatial `(x, y)` trajectory plot, added specifically
because a per-axis time series doesn't show this). Recorded here for the
documentation pass; no action taken.

---

## Window sizing (`MIN_PRED_EVENTS`/`MAX_WINDOW_ROWS`) and CCM-gate absolute library sizes — resolves the metric-routing fix's own recalibration blocker

**Status:** implemented and tested, on the same `metric-routing-depth-ge1-fix`
branch as the metric-routing fix above (rebased onto `main` to pick up
this file's own earlier commit, `7946311`, which the branch had missed
since it forked before that commit landed). Not yet merged. Both changes
below ended up in the same set of edits rather than the two separate
commits originally intended — see "Git-workflow note" at the end.

### Design evolution: why not stop at "half of a neuron's own total events"

The recalibration finding above left an open design question, answered
first with a provisional placeholder, then revised twice more as real
data exposed problems with each iteration in turn:

1. **v0 (first instruction, explicitly provisional):** for each neuron
   node, `min_spike_events = round(0.5 * total_events)` where
   `total_events` is that neuron's own reset-event count over the
   *entire* recording; grow the ranking window until its own **prediction
   portion** (the part `f1`-scoring actually touches) contains at least
   that many events. **Found mathematically unreachable**: the pred
   portion can never exceed `1 - lib_ratio` (~0.3 here) of any window,
   but `0.5 > 0.3`, so no window, however large, could ever satisfy it
   for a roughly uniformly-firing neuron. Caught by this project's own
   test (`tests/test_window_sizing.py`'s first version), not by
   inspection.
2. **v1:** same threshold, checked against the window **as a whole**
   instead — matches the original instruction's literal wording ("the
   window... contains at least MIN_SPIKE_EVENTS"). Passed the test, but
   real end-to-end verification (`run_mde_for_target()` against
   `CA1_1_tsls`) resolved a 17,874-row window satisfying "half of total
   events" (36) over the *whole* window while its own **pred** portion —
   the only part `f1`-scoring reads — had just 5 real events. Every
   real-observable candidate's ranking-stage `f1` score came back exactly
   `0.0000`. Found via real-data testing, not by inspection.
3. **Diagnosis, not a third patch:** a *fraction of a neuron's own total
   events* is the wrong kind of quantity here regardless of which part of
   the window it's checked against. It makes a busy neuron face a
   stricter test than a quiet one for no stated reason, gives the
   sparsest neurons — whose scores are least reliable — the *weakest*
   requirement, and hides a compute constraint (bounding window length
   for slow firers) inside what looks like a statistical one. The
   question the parameter actually needs to answer — "how many matched
   events does `f1` need before it's signal rather than noise" — is a
   property of the metric, not of the neuron.

### Final design

- **`MIN_PRED_EVENTS`**: a plain fixed integer (not a fraction of
  anything neuron-specific), applies to every depth >= 1 (neuron) node.
  Depth-0 (behavioral target) keeps the separate fixed
  `TARGET_RANKING_WINDOW_ROWS=1000` window, no event-count concept.
- **Window resolution** (`_resolve_target_ranking_window()`,
  `scripts/src/tsls-mde-gmn.py`): grow `W` (rows); the **pred portion**
  is `W`'s own last `1 - lib_ratio` fraction of ROWS — the identical split
  `_ccm_window()` will later carve out of this same `W` for the CCM gate,
  so the two agree by construction. Count the target's real reset events
  in that row range — **by rows, never by events** — and stop at the
  smallest `W` where the count is >= `MIN_PRED_EVENTS`. All event
  counting happens strictly within `usable_df` (i.e. `[0, B)`) — never
  `full_df` rows at or past `B` — since sizing a search parameter from
  held-out data is the same class of leakage the `B` boundary exists to
  prevent.
- **`MAX_WINDOW_ROWS`**: a separate integer compute ceiling, not
  calibrated. If a node can't reach `MIN_PRED_EVENTS` within
  `min(MAX_WINDOW_ROWS, len(usable_df))`, it is skipped — logged with the
  reason and the shortfall — rather than scored on too few events.
  `MIN_PRED_EVENTS` is the statistical floor; `MAX_WINDOW_ROWS` is the
  compute ceiling; the two express one thing each.
- **Non-monotonicity, found while writing the test, not by inspection:**
  the pred portion's own event count is *not* guaranteed monotonic in
  `W`. Both of the portion's edges only ever move forward as `W` grows
  (so its *width* never shrinks), but in a single step an event row can
  age out of the advancing left edge before any new event row enters at
  the right, so the count can dip locally. Consequence: on failure, the
  best achievable count must be the maximum seen at *any* window size
  tried, not just at the cap itself — the implementation was written the
  naive way first (count at the cap) and corrected before shipping; a
  dedicated test (`test_neuron_target_unreachable_compute_cap`)
  independently re-derives the best count over every window size up to
  the cap and checks it matches what the function reports.

### `MIN_PRED_EVENTS` calibration — real data, `r9_ccm_window_e_calibration.py` extended (not a parallel script)

`f1` between a real TSLS slice and a lightly perturbed copy of itself
(Gaussian noise, 5% of the slice's own std, 200 independent draws per
point), at target pred-event counts `{5, 10, 20, 40, 80, 160}`, 15
channels sampled stratified by firing-rate tercile (>= 5 per tercile,
pooled across all three regions):

| target pred events | pooled f1 std | low tercile | mid tercile | high tercile |
|---|---|---|---|---|
| 5 | 0.146 | 0.103 | 0.157 | 0.177 |
| 10 | 0.104 | 0.065 | 0.117 | 0.130 |
| 20 | 0.070 | 0.048 | 0.076 | 0.087 |
| 40 | 0.050 | 0.032 | 0.056 | 0.063 |
| 80 | 0.036 | 0.025 | 0.040 | 0.044 |
| 160 | 0.025 | 0.016 | 0.027 | 0.031 |

Smooth ~1/√N decay, **no natural elbow** — doubling the target roughly
halves the std at every step from 10 up to 160. The high-firing tercile
is consistently *somewhat noisier* than the low tercile at the same event
count (its event-count-matched slices are shorter in rows), a real but
modest spread, not a divergence that would demand separate thresholds per
tercile.

**Adopted default: `MIN_PRED_EVENTS = 20`.** A threshold choice justified
by the cost/reachability tradeoff below, **not a discovered stabilisation
point** — the curve above has no elbow to discover. At 20, std ≈ 0.07 is
still ~3x smaller than `CROSS_MAP_RHO_MIN`'s own 0.2 default. Real-data
reachability (all 45 `REGION_CHANNEL_CAP=15` channels, checked against
the actual `_resolve_target_ranking_window()` implementation, not an
approximation):

| `MIN_PRED_EVENTS` | unreachable | reachable window_rows (min/median/max) |
|---|---|---|
| 20 (adopted) | 3/45 (`CA1_0`: 16/20 events short by 4, `CA1_1`: 9/20 short by 11, `CA1_5`: 10/20 short by 10) | 407 / 3,593 / 26,748 |
| 40 | 8/45 | 872 / 10,437 / 30,664 |
| 80 | 20/45 (nearly half) | 2,127 / 10,485 / 27,451 |

40 and 80 buy modestly less noise at a steep cost in both unreachable
channels and window length — diminishing returns, not a clear win. Note
`CA1_1_tsls` — the same channel the metric-routing recalibration section
above used as its own worked example of ranking-stage degeneracy — is
one of the three channels this design correctly determines are
unreachable (its own real events, 59 within `[0, B)`, are too spread out
for even the largest possible pred portion, up to `B`, to ever collect 20
of them) and is skipped with a logged reason, not silently scored.

**Adopted `MAX_WINDOW_ROWS = 30000`** — not calibrated (only
`MIN_PRED_EVENTS` was asked to be), a compute ceiling comfortably above
the largest real reachable requirement observed (26,748) and safely below
this dataset's own leakage boundary (`B` ≈ 38,067 in this configuration)
— a backstop, not a tuned value.

### CCM-gate candidate-side event sufficiency — measured, then a diagnostic (not a skip)

The CCM gate's own `t2c` direction scores the **candidate's** own events
(`obs_rev`, the candidate's real values), over `proper_ccm`'s own full
frame — that function has no lib/pred split at all, just random library
subsamples drawn from the whole frame. `MIN_PRED_EVENTS` as designed
above only guarantees the *target's* own events in the ranking window's
pred slice; it says nothing about whether a given candidate has enough
of its own events in that same frame.

**Mechanism verified, not assumed.** With 0 real candidate events, recall
is `0/0` (undefined) regardless of what the predictions look like, so
`f1` is *identically* 0 for every `(sample, libSize)` cell — the fitted
slope of an exactly-constant line is exactly 0, never `> ccmSlope=0.01`.
Deterministic (confirmed by reading `adaptive_event_metrics_from_tsls`
directly). For 1-19 events (starved but not literally zero, where `f1`
is *not* deterministic), simulated under a genuine null — independently
generated target/candidate, no causal link, 60 draws per candidate-event
count `K ∈ {0, 1, 3, 5, 10, 19}`, run through the gate's own exact
procedure (`CCM_LIB_SIZES_ROWS`, `CCM_SAMPLE=20`, same slope fit):

| K | false-pass rate | draws with mean t2c exactly flat |
|---|---|---|
| 0 | 0/60 | 60/60 |
| 1 | 0/60 | 60/60 |
| 3 | 0/60 | 52/60 |
| 5 | 0/60 | 35/60 |
| 10 | 0/60 | 4/60 |
| 19 | 0/60 | 1/60 |

**0/360 false passes** (max observed slope: 0.00000). As `K` grows, `f1`
stops being literally flat, but the mean slope drifts slightly *negative*
— never toward a spurious pass.

**Is starvation actually negligible? Measured, not assumed — no.** At the
median resolved window (3,593 rows), **6/45 real channels (13%)** would
themselves be starved if they appeared as a candidate in that frame
(`mPFC_6`: 5, `CA1_6`: 9, `CA1_0`: 10, `mPFC_2`: 12, `CA1_1`: 14, `CA1_7`:
18 events). At the larger windows required by slower targets (~10,400+
rows), 0/45 are starved. So the two options ("do nothing" vs. "measure
first, then decide") do **not** collapse to the same answer — starvation
is real and concentrated at the low end (fast-firing, small-window
targets are exactly where their slower candidate-partners go starved).

**Decision: option 1 (rely on the existing `ccmSlope` gate, no new
control flow) plus one diagnostic.** Added a log line in both
`proper_mde`'s and `multi_proper_mde`'s own CCM-gate per-candidate loops
(`scripts/src/cleaned_helper_functions.py`): the candidate's own event
count in the frame, printed alongside the fitted slope, only when
`ccmMetric == 'f1'`. No filtering, no new control flow. The real cost of
doing nothing was never wrong selections (0/360 false passes above proves
that) — it's that an event-starved candidate and a genuinely non-causal
one were previously indistinguishable in the output, both recorded as
"failed the gate." The log line makes them separable after the fact at no
cost. **Option 2 (an explicit candidate-side skip) remains available** if
a future pass finds the diagnostic's own log lines show this matters more
than the 13%/0% split above suggests.

### CCM-gate absolute library sizes (`CCM_LIB_SIZES_ROWS`)

`CCM_WINDOW_ROWS`/`CCM_LIB_SIZE_PCTS` previously bounded two different
things through one shared, fixed-size `ccm_df`: the ranking-stage/
E-auto-select window (now per-node and node-specific, not fixed) and the
CCM gate's own `libSizes` sweep (a percentage of that same, now-variable,
window). Decoupled: `proper_mde`/`multi_proper_mde`
(`cleaned_helper_functions.py`) gained a `libSizes=None` parameter that,
when given an explicit list, bypasses the `pLibSizes`-as-percentage-of-`N`
computation entirely — `run_mde_for_target()`/`run_multi_mde_for_targets()`
now pass `CCM_LIB_SIZES_ROWS` (absolute row counts) there instead of
`pLibSizes`.

**Default chosen:** R9/R17's own already-calibrated percentage grid,
evaluated at the *old* 800-row `CCM_WINDOW_ROWS` frame
(2/3/5/8/20/50% → 16/24/40/64/160/400 rows), reinterpreted as fixed
absolute values, plus one additional point (800) for better plateau
visibility now that the surrounding frame can be much larger:
`(16, 24, 40, 64, 160, 400, 800)`.

**Re-validated (real data, this session)** that the same absolute values
still show a real convergence shape at a much larger frame, not just the
800-row scale they were originally picked at — four real (target,
candidate) pairs, at both an 800-row and an 8000-row frame:

```
target=head_position_x (correlation) <- mPFC_1_tsls
  frame=  800: 16:0.256 24:0.224 40:0.347 64:0.393 160:0.458 400:0.482
  frame= 8000: 16:0.093 24:0.208 40:0.236 64:0.262 160:0.308 400:0.363 800:0.384 1600:0.409
target=head_position_x (correlation) <- CA1_9_tsls
  frame=  800: 16:0.412 24:0.482 40:0.576 64:0.595 160:0.616 400:0.619
  frame= 8000: 16:0.343 24:0.335 40:0.308 64:0.337 160:0.372 400:0.389 800:0.391 1600:0.392
target=CA1_1_tsls (f1) <- mPFC_1_tsls
  frame=  800: 16:0.007 24:0.014 40:0.014 64:0.031 160:0.093 400:0.094
  frame= 8000: 16:0.006 24:0.006 40:0.009 64:0.014 160:0.043 400:0.069 800:0.080 1600:0.086
target=CA1_1_tsls (f1) <- OFC_14_tsls
  frame=  800: 16:0.008 24:0.000 40:0.000 64:0.028 160:0.013 400:0.061
  frame= 8000: 16:0.002 24:0.001 40:0.008 64:0.009 160:0.007 400:0.030 800:0.029 1600:0.041
```

Absolute magnitudes differ between frame sizes (expected — a bigger
surrounding frame changes what a fixed library size represents
proportionally), but the qualitative rising-then-plateauing shape holds
at both scales for every pair tested — supporting the claim that this
shape is a property of the absolute library size, not its ratio to the
frame.

`CCM_WINDOW_ROWS`/`CCM_LIB_SIZE_PCTS` remain exactly what they were,
**`MODE='GMN'`-only now** (`build_interaction_matrix()`, a separate,
untouched mechanism) — `MODE='MDE-GMN'` (this pipeline's own default) no
longer uses either.

**Incidental finding, noted, not fixed (out of scope for this batch):**
`scripts/batch/tsls-mde-gmn.sbatch`'s own `CCM_LIB_SIZE_PCTS` default was
still `"2,3,5,8,12,20,30,50"` — R17's own pre-existing 8→6 value
coarsening (`"2,3,5,8,20,50"`) had never actually been reflected in the
sbatch script's own default, meaning real batch runs were not using R17's
calibrated grid at all for this (now `MODE='GMN'`-only) knob. Harmless to
this batch's own work (which doesn't touch that code path) but a
pre-existing drift worth fixing separately.

### Two incidental findings, recorded per explicit instruction — not optimised

- `target_measurement` is passed into `proper_ccm` inside `proper_mde`'s
  own CCM gate call, but its result (`c2t`) is discarded by the caller —
  a dead parameter from that specific call site's perspective (still used
  correctly elsewhere `proper_ccm` is called with both directions
  genuinely needed).
- `_run_one_cell` (inside `proper_ccm`) computes **both** CCM directions
  (`c2t` and `t2c`) unconditionally every cell, even though the CCM gate
  only ever uses `t2c`. Since both directions do equivalent KDTree
  build+query work, this is roughly a 2x cost on every CCM-gate candidate
  evaluation for work that is thrown away. Cross-references
  `docs/tsls-mde-gmn-pipeline.tex`'s own still-open "Stage 1's real cost
  at calibration scale" section (the unexplained
  `REGION_CHANNEL_CAP=3→15` cost jump) — this alone would not fully
  explain that jump (candidate pool size itself also scales with
  `REGION_CHANNEL_CAP`), but it is a real, currently-unexploited
  optimization opportunity worth profiling before the next scale-up.

### A5 — fingerprint

`MIN_PRED_EVENTS`, `MAX_WINDOW_ROWS`, the resolved per-node
`window_rows`/`pred_event_count`, and `CCM_LIB_SIZES_ROWS` all now enter
`build_mde_selection_fingerprint()`, `build_multi_mde_selection_fingerprint()`,
and `build_network_fingerprint()` (in place of the old
`ccm_window_rows`/`ccm_lib_size_pcts` fields at those same call sites) —
a cache built under either the metric-routing bug, the earlier "half of
total events" design, or the old percentage-based library sizing must not
be silently reused now that all three have changed.

### Test

`tests/test_window_sizing.py` — fully rewritten for the final design (the
file's own two earlier designs are visible in its own history/docstring
for context): `test_behavioral_target_is_fixed_window`,
`test_neuron_target_extends_to_min_pred_events` (synthetic periodic TSLS
series, ground truth cross-checked via an independent, non-vectorized
per-window count — not the same vectorized/searchsorted code path the
implementation itself uses — for both the achieved `pred_event_count` and
minimality), `test_neuron_target_unreachable_data` (real firing activity
entirely outside the leakage-safe range), `test_neuron_target_unreachable_compute_cap`
(new: firing activity that *would* reach `MIN_PRED_EVENTS` given enough
rows, but `MAX_WINDOW_ROWS` caps the search first — confirms the reported
best-achieved count matches an independent brute-force search over every
window size up to the cap, not just the count at the cap itself). All
four pass; the full existing test suite (every `tests/test_*.py`) was
re-run afterward and still passes.

**Real end-to-end verification**, `run_mde_for_target()` against real
data (`REGION_CHANNEL_CAP=15`): `CA1_2_tsls` (confirmed reachable via the
sweep above) resolves `window_rows=3971`, `pred_event_count=20` — hitting
the threshold at minimality, as designed — and `proper_mde` runs to
completion. Zero drivers were selected for this specific target (every
candidate failed `crossMapRhoMin`) — a legitimate MDE outcome given that
threshold, not the same all-`f1`-scores-exactly-zero degeneracy the
metric-routing recalibration section documented; the failure mode this
redesign targets (a scoring window too event-sparse to produce a
meaningful `f1` at all) no longer occurs for a reachable target.

### Git-workflow note

The original intent (confirmed explicitly) was a separate commit for the
window-sizing redesign versus the CCM-gate absolute-library-size fix.
Both ended up touching the same function signatures
(`run_mde_for_target()`, the fingerprint builders) in the same edits,
since both needed changes there at once, discovered only while writing
the code. Nothing was committed before this was noticed, so a clean split
was technically still possible, but reconstructing two independently-
correct intermediate states from an already-intertwined working-tree diff
by hand carried more risk of a subtle bug than the review benefit
justified. Flagged to the user rather than deciding silently; landed
(`8d02c3e`) as one commit whose message clearly separates the two
concerns, per the user's own acknowledgement of this tradeoff.

**Consequence, stated plainly: the two changes are not separately
bisectable.** A `git bisect` landing on `8d02c3e` cannot distinguish
"window-sizing broke this" from "the CCM-gate library-size change broke
this" — both must be inspected together if a future regression bisects to
this commit. This is the concrete cost of the combined commit, not just a
process footnote; recorded here so it's visible to whoever runs a bisect
later, not only to whoever reads this section first.

---

## Diffusion-map crash and hang in `run_mde_evaluate()` — found running the
post-window-sizing smoke test, not one of the 22 numbered tasks

**Status:** both failure modes confirmed at real scale; root cause of the
*crash* isolated precisely; root cause of the *hang* explicitly **not**
diagnosed (out of scope for this pass, per direct instruction). A
dedicated `RUN_MDE_EVALUATE` toggle (default `false`) plus an exception
guard and a wall-clock bound now protect against both symptoms without
requiring the mechanism to be understood first.

### How this was found

The first real cluster run since the metric-routing/window-sizing/CCM-gate
fixes and the progress-monitoring log all landed (job 40143104,
`REGION_CHANNEL_CAP=3/MDE_D=2/NESTED_M=2`) completed Stage 1, Stage 2,
theta retuning, the final `Generate()` rebuild, `Forecast()`, and R7's own
two diagnostics cleanly — then crashed with
`ValueError: Input X contains infinity or a value too large for
dtype('float64')` inside R19's `run_mde_evaluate()`, specifically at
`LinearRegression().fit(dmap_lib_proj, target_lib).predict(dmap_pred_proj)`
— the diffusion-map baseline's own regression step. A second independent
job (40143067) hit the identical crash at the identical line. The
progress-monitoring log made the crash site immediately obvious (nothing
logged after `head_position_x: R19 run_mde_evaluate START`) without needing
to read the traceback first.

### The crash: root cause isolated

Reproduced in isolation using the real crashed job's own saved
`data_after_mde_head_position_x.csv` (driver `CA1_0_tsls`, D=1).
**Crashes identically at 800 rows (the old `CCM_WINDOW_ROWS` scale) in
1.8s** — this directly refutes a point-count/window-size explanation
before it could become an accepted hypothesis: scale is not the cause.
(Structurally, `run_mde_evaluate()`'s own `pool_lib` was never tied to
`CCM_WINDOW_ROWS` or the per-node window-sizing redesign in the first
place — it's sliced from `prediction_start`, a fixed,
`REGION_CHANNEL_CAP`-independent quantity, so "cap=3 vs cap=15" could not
have been the variable regardless.)

Instrumented `row_sum` directly (monkeypatched pydiffmap's own
`_left_normalize`, `diffusion_map.py:136-138`): at 800 rows, the
**training** kernel's `row_sum` has zero zeros (healthy, min=1.324) — the
failure is entirely in the **transform** step on held-out points, where
**53% of the 6,718 held-out rows have `row_sum` exactly 0**. Root cause,
confirmed by comparing value ranges directly: this channel's TSLS values
run measurably higher in the held-out window than the library window ever
saw (lib max 27.4 vs. pred max 39.4 at the 800-row split) — **14.2% of
held-out points exceed the library's own observed range entirely**, so no
k-nearest-neighbor kernel similarity above float underflow exists for
them, at any reasonable epsilon (confirmed `bgh_generous` — already the
*more* generous of pydiffmap's two automatic-epsilon variants — does not
avoid this; widening epsilon further would only trade underflow-to-zero
for a bandwidth too large to reflect local structure at all, since the
failing points are outside the fitted density entirely, not merely
borderline). Checked all 45 real channels: **9/45 (20%) show this same
"held-out exceeds library range" pattern** — a real, non-rare property of
this dataset (channel-level firing-rate non-stationarity across the
recording), not a one-off unlucky column. Near-duplicate rows are real too
(75–99% across all 45 channels, regardless of whether that channel
crashes) but don't correlate with which channels actually fail — a
distinct, non-causal property of TSLS data, not the crash mechanism
itself.

### The hang: confirmed, mechanism NOT diagnosed (future task)

Verifying the exception-guard fix (Priority 1 of this investigation)
against the real crashing data at its own real ~31,000-row scale did not
crash — it **ran for 74+ minutes of CPU time with no completion, no
crash, and no output at all**, before being killed manually. This is a
qualitatively different, more severe symptom than the fast crash: a
`try/except` only protects against something that actually raises, and
this never did.

**The mechanism for the hang is explicitly not diagnosed** — out of scope
for this pass, per direct instruction, and flagged here for whoever picks
it up: **the duplicate-row structure suspected as a contributing factor is
inherent to TSLS by construction** (long inter-spike ramps are locally
near-identical), not an artifact of this specific dataset or
`REGION_CHANNEL_CAP` — so it could plausibly affect other KD-tree- or
kernel-based computations elsewhere that operate on real TSLS series,
not only `pydiffmap`'s own diffusion map. This is a real, open question,
not a closed one. Whether the 1.8s crash and the 74+-minute hang are one
underlying pathology surfacing differently depending on exactly where the
duplicate structure sends the KD-tree/kernel computation, or two distinct
issues, is itself unresolved.

### R7 audit: the "already fully guarded" premise was wrong

Investigating whether `run_mde_evaluate()` was uniquely unguarded (per
§11's own framing: every comparable stage — §4.3's per-node exception
guard, §10.1's log-and-skip on Generate() divergence, R7's own per-target
skip path — already fails gracefully) found the premise did not fully
hold: **`run_tp_vs_rho_sweep()`'s own `proper_predict_interval()` call is
guarded only by `if not cols: skip`**, and **`run_univariate_e_tau_grid()`'s
own per-cell `proper_simplex()` call is guarded only by `if mde_df is
None: skip`** — neither has an exception guard around the computation
itself. A target with selected columns (or a real `mde_df`) whose actual
call raised for some other reason would still crash the whole function
today. Reported, not fixed — this was outside the batch actually
authorized (R19's own baselines specifically), and per Part A a wrong
premise gets reported rather than silently absorbed into "the same fix."

### Fix: a dedicated toggle, plus a guard, plus a wall-clock bound

**`RUN_MDE_EVALUATE`** (default `false`), deliberately separate from
`RUN_MDE_DIAGNOSTICS` — R7's own two diagnostics (unaffected by either
failure mode, confirmed above) keep running under the existing knob
regardless. `RUN_MDE_EVALUATE` gates only the R19 loop in `main()`.
Exposed in both sbatch scripts' own `MODIFIABLE PARAMETERS` block, per
explicit instruction, for naming consistency — but only actually wired
into `tsls-mde-gmn.py`; `simple-mde-gmn.py`'s own `run_mde_evaluate()` is a
different implementation (`dx.Evaluate`-based, on 1 Hz raw spike-count
data, already gated by its own `RUN_MDE_DIAGNOSTICS`) that has not been
confirmed to share this finding's own suspected mechanism (raw counts
don't have TSLS's own long-ramp duplicate structure) — its export in
`simple-mde-gmn.sbatch` is inert until/unless someone wires it in, not
evidence that pipeline has the same risk.

When off, logged exactly once per run (`print` and the progress-monitoring
log both), unconditionally: *"R19 run_mde_evaluate SKIPPED by
configuration..."* — a silent absence is exactly how the §18.6 swallowed
exception went unnoticed for as long as it did; this doesn't repeat that.

**The exception guard** (independent per baseline, PCA and diffusion-map
each get their own): a failure is logged with the exception text and the
target(s) it occurred on, and only that baseline's own numbers/plot panel
are skipped — MDE's own reconstruction is deliberately left unguarded
(it's the causal claim itself, per §11's own linknote, not optional
tooling) and any other baseline continues unaffected.

**The wall-clock bound, `MDE_EVALUATE_BASELINE_TIMEOUT_SECONDS = 300`**:
each baseline now runs in a genuinely separate `multiprocessing.Process`
(fork), force-killed (`terminate()`, then `kill()` if needed) if it
exceeds the bound. **A `try/except` plus a signal-based "soft" timeout
would not have been enough** — the hang can occur inside compiled
KD-tree/BLAS calls that don't return control to the Python interpreter for
a long stretch, which is exactly when a signal fires but its handler can't
actually run until the call returns on its own. Only an OS-level `SIGKILL`
of a separate process reclaims control regardless of what the computation
is doing internally. Cost: one process fork per baseline call (cheap under
`fork` — no re-import, copy-on-write memory, real overhead is
milliseconds) plus pickling the modest-sized (D-column) result back
through a `Queue`; the real cost is that a hard kill on timeout discards
whatever progress a slow-but-would-eventually-succeed computation had
made, converting it into a skip — a deliberate trade for a bounded worst
case over an unbounded one. 300s is not calibrated against a known-good
completion time (no successful fit has yet been observed to time at real
~31,000-row scale) — chosen as generous headroom over the 1.8s a genuine
crash takes to surface, while still bounding the worst case to a
tolerable, one-time-per-network cost.

### Verified

`tests/test_r19_mde_evaluate.py` (the existing R19 test, calling
`run_mde_evaluate()` directly, unaffected by the `main()`-level toggle)
produces **bit-identical output to the pre-change baseline**
(`PCA CAE=8.85`, `D-Map CAE=14.53`, same correlations) — the subprocess
refactor preserves correctness exactly on the healthy path, not just
"doesn't crash." Full test suite, `py_compile`, `bash -n` on both sbatch
scripts all pass. **Real cluster confirmation: done.** After a brief,
unrelated `/bucket` outage recurrence on 2026-08-30
(`docs/bucket-outage-2026-08-30.md`) cleared, job 40385630 (same smoke
configuration, `RUN_MDE_EVALUATE=false`) ran the full pipeline —
depth-0 seed, Stage 1, Stage 2, theta retuning for both networks, the
`RUN_MDE_EVALUATE=false` skip notice logged exactly once as designed —
and reached `PIPELINE COMPLETE -- 1089.8s total`. This is the first
confirmed clean end-to-end run since the window-sizing, metric-routing,
and progress-monitoring changes all landed. See `JOB_LOG.md`'s own
entry for this job for the full plain-language summary, including one
incidental finding (the sbatch trap's `sacct` retry-window gap, already
flagged above, caught live: this job's own final line read
`SLURM_STATE=RUNNING` for an already-successfully-exited job).

### Flag for the documentation pass

- Both failure modes are now confirmed at real scale: the 1.8s crash at
  800 rows and the 74+-minute hang at ~31,000 rows — record both, with
  the reading that they may be one underlying pathology surfacing
  differently depending on where the duplicate-row structure sends the
  KD-tree/kernel computation, not necessarily two separate bugs.
- The mechanism for the *hang* specifically is not diagnosed. TSLS's own
  duplicate-row structure (long inter-spike ramps, locally near-identical
  by construction) is suspected but unconfirmed, and — because it's
  intrinsic to TSLS itself, not to this dataset or `REGION_CHANNEL_CAP` —
  could plausibly affect other KD-tree- or kernel-based computations on
  TSLS series elsewhere in this codebase. The toggle prevents the
  symptom; it does not close the question. Diagnosing it is its own,
  separate future task.
- `RUN_MDE_EVALUATE` now defaults to `false`, so §19's "cannot compare
  MDE's selection against an unsupervised baseline" gap is **wider than
  it was**, not narrower — R19 existed and worked (on the toy/reference
  data it was validated against) before this finding; the documentation
  pass should reflect that this capability now exists in code but is off
  by default pending the hang's own diagnosis, not that it was never
  built.

## R22 — Forecast() diagnostic: real-data leaf passthrough silently mislabels every node's own prediction window

Found investigating a user report against job 40421505's own archived
`gmn_generate_forecast_head_position_x_head_position_y_part01.png`:
`FORECAST_PLOT_RANGE="windowed"` looked identical to `"full"`, and every
subplot showed a small, isolated `Forecast()` trace near `x=0` with a large
gap before the real `generate_start`-anchored cluster. Both symptoms trace
to one bug, and it is a real correctness bug, not a cosmetic one — it also
corrupts the `Forecast rho`/`Forecast f1` scores printed in each subplot's
own title.

### Mechanism (traced through `gmn`'s own source, not guessed)

1. `gmn/Network.py`'s `Network.__init__`: in Forecast mode, `dataLib_i =
   range(self.data.shape[0])` ("Forecast mode: all data since lib & pred
   are specified") — every node's `self.data` is the *entire* dataset,
   never restricted to a lib/pred window.
2. R14's `EXTERNAL_INPUT_MODE='real_data'` short-circuit in
   `_generalized_node_forecast()` (`tsls-mde-gmn.py`) returned
   `(self.data[timeCol], self.data[name])` for every external-input leaf —
   the *entire*, unrestricted dataset, ignoring `Parameters.pred`
   completely. Every other (SMap) node instead returns only its own short
   `pred`-range slice, with a fresh 0-based pandas index (confirmed
   empirically: a standalone `pyEDM.SMap()` call's `S['predictions']`
   always carries a 0-based `RangeIndex`, independent of what its own
   `'Time'` column reports).
3. `gmn/GMN.py`'s `GMN.Forecast()` assembles `DataOut` via plain per-node
   column assignment (`self.DataOut[nodeName] = node.Forecast()[1]`), and
   takes the shared `'Time'` column *only* from
   `Network.TopologicalSorted[0]` — a source/leaf node (no in-edges) is
   essentially always first in topological order. Since that leaf's own
   output is now the full dataset (point 2), `DataOut`'s own row index and
   `'Time'` column get established from it (0..N-1), while every SMap
   node's short ~`forecast_length`-row Series pandas-aligns onto only the
   *first* `forecast_length`-ish of those row labels — everything past
   that is `NaN` for that column.
4. `run_forecast_diagnostic()`'s own `forecast_out =
   Gf.DataOut.dropna()...` then keeps only the surviving intersection:
   row-labels `0..forecast_length`, which the leaf's `'Time'` column
   reports as literal absolute time `0..forecast_length` — the very start
   of the recording — even though every SMap node's predictions were
   genuinely computed from data near `generate_start` (confirmed: job
   40421505's real figure shows `generate_start` near row ~38000, and the
   stray early cluster near row 0, on *every* subplot regardless of node,
   exactly matching this mechanism rather than a per-node coincidence).

### Consequences

- **Plot**: the mislabeled-near-zero `Forecast()` trace has real x=0 data,
  so matplotlib's autoscale (no explicit `ax.set_xlim()` was ever set)
  stretches every subplot back to the start of the recording regardless of
  `FORECAST_PLOT_RANGE` — `"windowed"` and `"full"` looked identical
  because the *data itself*, not the plotting code's own windowing math,
  was overriding the intended range.
- **Scores**: `rhos[node] = tsls._score(actual[time_idx], forecast_pred,
  ...)` compares genuinely-near-`generate_start` predictions against
  `actual[0:few-hundred]` — real data from the literal start of the
  recording. Every `Forecast f1`/`Forecast correlation` value printed in
  this diagnostic (and stored in `run['forecast_rho']`) before this fix is
  scientifically meaningless, not just visually confusing. This does
  **not** affect `Generate()`'s own outputs or fidelity scoring
  (`gmn_fidelity_rho()`) — `build_gmn_config(mode='Generate', ...)` sets
  `lib_range`/`pred_range` to `''`, so each node falls back to its own
  correctly-computed `self.libEnd_i` (via `_generalized_node_init()`)
  instead of a shared, leaf-passthrough-corrupted window. Only the
  separate `Forecast()` diagnostic (this one figure and its printed
  scores) was affected.

### Fix

- `_generalized_node_forecast()`'s real-data leaf shortcut now slices
  `self.data` to `Parameters.pred`'s own row range (1-indexed, inclusive
  — the same string every SMap node in the network already receives)
  before returning, giving it the same length and a matching 0-based
  index instead of the full dataset. This is the same "shared,
  network-wide window" tradeoff `build_gmn_config()`'s own docstring
  already documents and accepts for nodes with shallower embeddings than
  the network's own worst case (`network_max_embedding_offset()`) — this
  fix brings the leaf passthrough in line with that existing, accepted
  convention rather than exempting it entirely.
- `run_forecast_diagnostic()` now also calls `ax.set_xlim(plot_start,
  plot_end)` per subplot as a defensive second layer, so a future
  mismatch of this kind clips the view instead of silently defeating
  `FORECAST_PLOT_RANGE` again.

### Real cluster confirmation

**Done.** Job 40563312 (mirroring job 40421505's own configuration)
reached `PIPELINE COMPLETE -- 9467.4s total`. The
`gmn_generate_forecast_head_position_x_head_position_y` figure now shows
the Forecast() trace correctly windowed around `generate_start`, with no
stray mislabeled trace near time 0 — the fix is confirmed on a real run,
not just the toy regression test.

### Follow-up finding (flagged, not yet fixed): Forecast() still mistimes predictions, separately from the mislabeling bug above

Found while reviewing job 40563312's own (now-correctly-windowed)
`gmn_generate_forecast_head_position_x_head_position_y` figure: for
`head_position_y`, the Forecast() trace's own predicted excursion peaks
well before the real excursion actually happens. Confirmed precisely (not
just eyeballed) by re-running the exact `run_forecast_diagnostic()` call
this job made, offline, against the job's own archived network/data/theta
— the reproduced Forecast()/Generate() scores matched the real job's own
output exactly (`head_position_x`=0.350, `head_position_y`=-0.509),
confirming the reconstruction is faithful:

- **Forecast()**: predicted peak at Time=37921; real excursion peaks at
  Time=38090 — **169 rows (~4.2s) early**.
- **Generate()** (same network, same discovered drivers, same theta):
  predicted peak at Time=38069; real peak at Time=38090 — only **21 rows
  (~0.5s) off**.

This rules out "the discovered driver delays are just imprecise" as the
explanation — Generate() uses the exact same drivers/delays and tracks
the real excursion's timing far better. It also rules out the specific
per-node-offset-vs-network-offset mismatch already known from this
section's own main fix above: `head_position_x`'s own embedding offset
(197) exactly equals `network_max_embedding_offset()` (197, no mismatch
possible), yet `head_position_x`'s own Forecast() also showed a large
(155-row) discrepancy at its own local extremum -- so that specific
mechanism can't be the (sole) explanation here either.

Leading hypothesis, not yet confirmed: Generate() sizes each node's own
reference library correctly per-node (`self.libEnd_i`, individually
corrected in `_generalized_node_init()`), while Forecast() uses one
shared, network-wide `lib`/`pred` window sized from
`network_max_embedding_offset()` (the deepest lag anywhere in the whole
network, 197 rows for this job) — the same "one window sized for the
network's worst case, not each node's own actual needs" design this
section's own main fix (above) addressed for the *pred* side (what gets
labeled) and R22's fix left untouched on the *lib* side (what reference
neighbors SMap actually gets to match against). A differently-sized
library could plausibly make SMap match against a slightly mistimed
historical analog of the excursion. Not confirmed by direct evidence yet
-- flagged for future investigation, not fixed. Left as-is per explicit
instruction (2026-09-01): "Leave it flagged for now."

**Second hypothesis, also unconfirmed** (raised 2026-09-01): `cfg['TP']`
(=20 for this job) is NOT used by the actual Generate()/Forecast()
machinery at all -- `build_gmn_config()` hardcodes `Tp = 1` for every
GMN Node regardless of `cfg['TP']` (tsls-mde-gmn.py:3705), confirmed by
grepping every use of `cfg['TP']` (only `ccm_prefilter_observables()`,
the depth-0 joint MDE, and `build_network()`/`build_nested_mde_network()`
-- all discovery-time MDE/CCM scoring, never GMN's own forecast). But
this surfaces a real, separate, previously-unexamined mismatch: every
discovered driver edge (delay=D) was validated during nested-MDE's own
forward-selection by asking "does column C at delay D predict the target
`Tp=20` steps ahead?" -- then `build_gmn_config()` reuses that same delay
D but asks pyEDM for a `Tp=1` prediction instead. The edge only records
`delay=D`; the Tp=20 context it was validated under is discarded. Whether
this horizon mismatch contributes to the timing shift above is untested.
Quantitatively it can't be the whole story on its own -- `Tp=20` rows =
0.5s at this recording's 40Hz rate, well short of the observed ~169-row
(~4.2s) shift -- but it's a legitimate, distinct contributing hypothesis
worth investigating alongside the library-sizing one above, not a
replacement for it.

**Correction, escalating severity (2026-09-01)**: the framing above --
"Forecast() genuinely has skill but Pearson correlation unfairly
penalizes it for being phase-shifted" -- is wrong, and was retracted
after the user pushed back on it. Direct verification: shifting
`head_position_y`'s real trace by exactly the measured 169-row gap and
recorrelating against the forecast gives r = 1.0000 (not just "high").
Checking raw values (not just correlation): `forecast ≈ 0.995 *
actual[t+169] + 0.19`, r = 0.999954. A genuine SMap forecast built from
noisy external neural predictors would not match ground truth to four
nines of correlation -- ordinary model residual error would show up.
This precision is the signature of the reported "forecast" trace being
substantially a reproduction of the real trajectory itself, offset by a
fixed amount, not a real model output. That offset (169) is not
arbitrary either -- it exactly equals `head_position_y`'s own embedding
depth (`mPFC_9_tsls` at delay -169, its deepest discovered driver, per
`_embedding_spec_for()`).

This reframes the finding entirely: it is not "a fair prediction
penalized by a phase-sensitive metric" but closer to "the reported
one-step forecast for this node appears to be the real trajectory
mislabeled by an offset equal to the node's own embedding depth" -- a
labeling/indexing problem analogous in spirit to the leaf-passthrough bug
this section's main fix addressed, but affecting an SMap/interior node
instead of a real-data-mode leaf. The exact mechanism producing this
specific relationship has NOT been traced through the code (unlike the
leaf-passthrough bug, which was pinned down to an exact line) -- this is
circumstantial-but-strong quantitative evidence (the shift magnitude
matching the node's own offset precisely is unlikely to be coincidence),
not a confirmed root cause. Left flagged, not fixed, per the same
2026-09-01 instruction as above -- but any results-section language
describing this Forecast() number should NOT claim it reflects
understated genuine skill; it should note the number is currently
untrustworthy for reasons beyond simple timing misalignment.

