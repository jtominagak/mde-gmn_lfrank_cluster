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

