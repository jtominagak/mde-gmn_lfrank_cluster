# tsls-mde-gmn: review actions from annotated documentation v1

Source: 62 highlights / 44 notes on `tsls-mde-gmn-pipeline_v1_annotated_export.pdf`.
Covers the 22 notes that request or imply a change to the pipeline. Documentation-only
notes (21) are tracked separately and are out of scope here.

All open design decisions in the previous revision of this file have been resolved by the
reviewer. There are no blocked items. Every task below is authorized **subject to the
verification protocol in Part A**, which is not optional.

---

## Part A — Ground rules

### A1. Verify before you edit. This overrides every instruction below.

The tasks in Part B were written from the pipeline's *documentation*, not from its source.
The documentation is known to be imprecise in places — that is why it was reviewed. Before
making any edit for a task, you must:

1. **Read the actual implementation** of every function the task names, plus its call sites.
2. **Confirm the task's premise is true of the code.** Several tasks assert current behavior
   ("X currently does Y"). If the code does not do Y, the task's reasoning may not survive.
3. **Confirm the change still makes sense** given what you found. A task that presupposes a
   structure that does not exist is not a task to be approximated.
4. **Check for contradictions with other tasks in this file.** Known interactions are listed
   in A2, but that list is not exhaustive — you are expected to find others.

If any of steps 2–4 fails, or you are unsure: **stop, report, and ask for permission before
proceeding.** Ask each time. Do not batch questions to the end of a long autonomous run, and
do not proceed on a "most likely intended" reading. A wrong guess here costs more than a
question. State plainly what you found, why it conflicts with the instruction as written, and
what you would do instead — then wait.

Report findings in `docs/review-findings.md`, one section per task ID.

### A2. Known contradictions and interactions — check these explicitly

- **R12 (serial stages) vs R6 (Stage-2 recursion test).** R12 reverts to a serial
  architecture; R6 was written against the unified BFS. R6's test must be rewritten to match
  the architecture R12 produces, not deleted. Do R12 first.
- **R13 (`NESTED_M` scope) is not a renumbering.** The counting convention and default stay
  exactly as they are. Only the *scope* changes — per-stage to whole-network. Do not propose
  a new arithmetic or a new default; a change with no practical effect is not wanted.
- **R14 (leaf nodes become external inputs) vs the pipeline's free-running claim.** §1.1 claims
  the network runs freely "without ever seeing the real behavior again". If external-input
  leaves are fed from real data during `Generate()`, the simulation is no longer autonomous.
  See R14 step 4 — this must be resolved before implementing, not after.
- **R14 vs §8.2's simulation guarantee.** §8.2 states that unconditional self-history is what
  lets a node reached only by a `delay != 0` edge simulate at all. Removing it without a
  replacement may make such nodes unsimulable.
- **R16 (sibling toggle) vs A3 (sibling parity).** R16 deliberately introduces divergence in
  the sibling; A3's port-in-the-same-commit rule does not apply to it.
- **R13 and R15 share one budget, and Stage 1 spends it first.** These do not conflict, but
  they interact in a way that is easy to miss. Under R12, Stage 1 runs to full depth before
  Stage 2 begins, so on any branch where Stage 1 already reached the cap, a real observable
  Stage 2 selects there sits beyond the budget and is terminated immediately. Stage 2's
  recursion is therefore live only on branches Stage 1 left shallower than the cap. This is
  correct behavior, not a bug — but it means the recursion can look broken when it is merely
  budget-bound, so the two termination causes must be logged distinguishably (R15 step 4) and
  the realized depth reported per branch.

  **Serialization and recursion are orthogonal — do not conflate them.** The historical
  dead-end-leaf bug (§5.3) was caused by Stage 2 being a *single non-recursive sweep*
  (`attach_lag_embeddings()` running once over Stage 1's final node set), not by the stages
  being serial. Unifying the stages was one way to fix it, not the only way. Serial stages
  with an internally recursive Stage 2 reintroduce nothing.

### A3. Sibling parity
`simple-mde-gmn.py` is a deliberate near-verbatim sibling. Changes to shared machinery are
ported to it in the same commit, or the divergence is recorded in §1.2's asymmetry list with a
one-line reason. R16 is the explicit exception.

### A4. Verification before submission
`python3 -m py_compile` on both pipelines and `bash -n` on both sbatch scripts must pass. No
cluster submission from this work without a local smoke test on real data at
`REGION_CHANNEL_CAP=3, MDE_D=2`.

### A5. Cache invalidation
Any change that alters a cacheable stage's numerical outcome must add its new knobs to the
relevant `build_*_fingerprint()` in the same commit. Silent reuse of a stale cache across this
change set is a correctness bug.

### A6. Documentation is downstream
Do not edit the pipeline document's LaTeX source as part of these tasks. The documentation
rewrite is a separate pass that runs after these changes land, so it describes the code as it
then is. Do record, per task, what changed in a form the rewrite can consume.

---

## Part B — Actions

Ordering note: R12, R13, R14 are architectural and change what the network *is*. Do them
before the tasks that measure or tune the network (R11, R15, R19, R20). R9 gates R17 and R18.

---

### R1 — Vectorize `compute_time_since_previous_spike`
**Anchor:** p.7, §2.2, "Cost note:"

Replace the O(N x S) per-grid-point boolean mask in
`cleaned_helper_functions.compute_time_since_previous_spike` with
`np.searchsorted(spike_times, time_grid, side='right') - 1`, giving the index of the most
recent spike at or before each grid point. Index `-1` marks grid points preceding the
channel's first spike and must yield `NaN`. Value is `time_grid - spike_times[idx]`.

**Acceptance:** elementwise identical to the old implementation on at least three real
channels from `J1620210605_epoch_8_freq_40_Hz.csv` under `np.allclose(..., equal_nan=True)`.
Keep the old version as `_compute_time_since_previous_spike_reference`, used only by the
equivalence test. Report the speedup at full `REGION_CHANNEL_CAP`.

---

### R2 — Audit: does the CCM pre-filter see only real observables?
**Anchor:** p.10, §3.1 heading

Confirm `ccm_prefilter_observables()`'s candidate list contains only real (delay-0) columns,
and that no `DELAY_MARKER`-suffixed column can reach it — including via `expand_delay_columns`
running earlier than expected, or via a cached `mde_results` entry. Trace the actual list
construction at the `main()` call site.

**Acceptance:** written finding. If lagged columns can reach the pre-filter, fix it and add an
assertion at the top of `ccm_prefilter_observables()` that raises on any column containing
`DELAY_MARKER`.

---

### R3 — Audit: candidate-pool exclusion semantics
**Anchor:** p.11, §3.2 Step 2

Report precisely what happens to an observable once selected — whether it is removed from the
pool for later dimensions within the same `proper_mde` call; for later BFS depths; for other
nodes at the same depth; and whether Stage 2's pool is the same filtered universe or rebuilt.

**Acceptance:** written finding only — **make no change.** The reviewer's expectation ("once
selected, it disappears from the pool") may or may not match intent, and the answer feeds a
decision not yet made. Flag explicitly if the answer differs across the three scopes.

---

### R4 — Report the static `E` used by `proper_mde`'s CCM gate
**Anchor:** p.20, §6.2 Step 2

Report the concrete value `E` takes at every call site — depth-0 seed, Stage 1, Stage 2 —
where it comes from, whether it differs between sites, and whether any per-candidate
`EmbedDimension` auto-select path exists on the single-target path.

**Acceptance:** written finding only — **make no change.** Feeds R18 and R22.

---

### R5 — Concurrency audit and global worker budget
**Anchor:** p.15, §4.3 heading

1. Instrument the BFS to log, per wave per pass, `len(wave)`, `outer_workers`,
   `inner_workers`, and their product.
2. Run at `REGION_CHANNEL_CAP=5, MDE_D=2, NESTED_M=2` on real data and **measure** the peak
   count of simultaneously live worker processes and threads — do not infer it from configured
   values.
3. Determine whether concurrent waves, retained pools, or `ProcessPoolExecutor`s outliving
   their wave can push the aggregate past the core count.

If the measured peak exceeds the requested core count, add a module-level semaphore capping
total concurrent workers at `cores`, acquired by every pool before submission, and document
the degradation behavior (tasks queue; they do not fail).

**Acceptance:** finding with measured peaks; semaphore only if the measurement warrants it.
Note that R12 changes the wave structure — run this after R12 lands, or re-run it then.

---

### R6 — Regression test: Stage-2 real-observable picks recurse
**Anchor:** p.17, §5.2 Step 4

**Rewritten under R12.** Once the serial architecture is in place, prove by test — not by
reading code — that with `STAGE2_ALLOW_REAL_OBSERVABLES=true`, a delay-0 column selected by
Stage 2 is enqueued for its own further Stage 2 round, subject to the R13 depth budget.

**Acceptance:** a test that fails if the recursion is removed. Report whether it passed on
first run. If Stage 2's recursion turns out not to work this way, that is R15's flag condition
— report rather than fixing silently.

---

### R7 — Confirm Section 11 diagnostics produce figures
**Anchor:** p.30, §11.1

Verify `run_tp_vs_rho_sweep` and `run_univariate_e_tau_grid` each write a figure to `OUT_DIR`
on a normal run with `RUN_MDE_DIAGNOSTICS=true`, and that those files reach
`OUTPUT_ARCHIVE_DIR` on clean exit. Check the per-target skip path (zero columns selected)
leaves no half-written or empty figure.

**Acceptance:** if either produces no figure, add it — Tp-vs-skill line plot per target;
`(E, tau)` heatmap per target with cells above `GRID_RHO_THRESHOLD` boxed. Follow the existing
`plot_*` filename convention.

---

### R8 — Add resampling to the CCM pre-filter
**Anchor:** p.10, §3.1 Step 2

The pre-filter draws one library sample per library size while the main CCM path averages
`CCM_SAMPLE` draws, making the `CCM_PREFILTER_RHO_MIN` cutoff partly seed-dependent. Add
`CCM_PREFILTER_SAMPLE` (default `5`), averaging that many independent draws per library size
before averaging across sizes. Add it to every fingerprint that already includes
`CCM_PREFILTER_RHO_MIN`.

**Acceptance:** run the pre-filter three times under different seeds at
`CCM_PREFILTER_SAMPLE=1` and `=5`; report variability of the kept-candidate set in each case,
and the added wall-clock cost at `REGION_CHANNEL_CAP=15`.

**Scope boundary:** changes the number of draws only. `CCM_PREFILTER_E` is R18; mean-vs-slope
aggregation is settled in R18 as "keep the mean".

---

### R9 — `CCM_WINDOW_ROWS` and `CCM_PREFILTER_E` calibration harness
**Anchor:** p.10, §3.1 Step 1

Section 18.7 calibrated against a small hand-picked set of pairs. Build a harness testing
whether the current settings generalize:

1. Sample at least 40 (observable, target) pairs spanning all three regions, stratified by
   firing rate (low/medium/high tercile).
2. Compute the CCM skill-vs-library-size curve at
   `CCM_WINDOW_ROWS in {200, 400, 800, 1600, 3200}`.
3. Sweep `CCM_PREFILTER_E in {3, 5, 7, 9, 11}` across the same pairs (this is the empirical
   calibration R18 depends on).
4. Per window, report the fraction of pairs whose curve is still rising at the smallest tested
   library size versus already flat (Figure 14b), and per `E`, the stability of the resulting
   kept-candidate set.
5. Output a summary table plus the raw curves.

**Acceptance:** a standalone script under `scripts/calibration/`, its output, and a finding.
**Do not change any default from this task** — it feeds R17 and R18.

---

### R10 — Inventory the S-map surface in `cleaned_helper_functions.py`
**Anchor:** p.26, §9.2 box

Precursor to R20. Report whether the module contains a native S-map (local weighted linear
regression with a theta locality parameter) or whether its KD-tree engine only does
simplex-style weighted-average prediction. State exactly what would have to be written to
sweep theta with an `_score`-compatible metric.

**Acceptance:** finding with function names and a scope estimate. **No code.**

---

### R11 — Keep the target-only search objective; report per-node scores additively
**Anchor:** p.26, §9.3 and p.27, §9.4
**Decision:** the objective stays target-only. Theta search and the start-row sweep optimize
performance at target nodes.

1. Confirm `evaluate_smap_candidate()`, `run_theta_search`, and `sweep_generate_start` all in
   fact optimize the mean score across target nodes only. If any of them scores driver nodes
   or mixes metrics, that is a bug — report it, then fix it to target-only.
2. Additionally compute and log, for the winning configuration only, the per-node score for
   every node using each node's `_metric_for`-resolved metric, written to a per-node table in
   `OUT_DIR`. This is reporting, not optimization.

**Acceptance:** selected theta and start row are bit-identical to current behavior on Run B's
configuration. The per-node table is purely additive. Note the metrics in that table are not
mutually comparable (correlation for targets, f1 for neurons) — label the metric per row and
do not total the column.

---

### R12 — Make Stage 1 and Stage 2 fully serial
**Anchor:** p.14, §4.2
**Decision:** Stage 1 must complete its entire search — the full depth allowed by the budget —
before Stage 2 begins. Stage 1 and Stage 2 are not to run within the same BFS wave.

Current behavior is a unified per-wave BFS where every node gets a Stage-1-style pass and a
Stage-2-style pass in the same wave, with Stage 2 submitted after that wave's Stage 1 pool
returns. That per-wave interleaving is what is being removed: the guarantee must be at the
level of the whole stage, not the wave.

**Required order:**
1. Stage 1 BFS runs to completion over real observables only, across every depth the budget
   allows, producing the full real-observable backbone.
2. Only then does Stage 2 run, attaching lagged (and, per R15, real) drivers.

**Preserve, do not regress.** Diagnose the historical bug correctly before restructuring. The
dead-end-leaf problem (§5.3) existed because Stage 2 was a *single non-recursive sweep* —
`attach_lag_embeddings()` ran once over Stage 1's final node set — not because the stages ran
in sequence. Serialization and recursion are independent properties. What must be restored is
the sequencing; what must be kept is Stage 2's own recursion. A real observable selected in
Stage 2 enqueues its own further Stage 2 round (R15), within the R13 budget. Serializing the
stages does not flatten Stage 2 into a single sweep, and any implementation in which it does
is wrong.

**Also required:** preserve the CCM-pre-filter ordering fix from §18.12 — depth-0 and every
later round must still search the identical filtered pool. Do not let this restructuring
resurrect the `CA1_12_tsls` mismatch.

**Acceptance:** a test asserting no Stage 2 MDE call is issued before every Stage 1 call has
returned. Re-run the §18.12 probe confirming Stage-2 real-observable picks still recurse.
Report the change in wall-clock and in final network topology on Run A's configuration —
topology is expected to change; state how.

---

### R13 — Make `NESTED_M` a whole-network depth budget, shared by both stages
**Anchor:** p.18, §5.3 box
**Decision:** `NESTED_M` bounds the depth of the *entire* constructed network, not each stage
independently. Stage 1 and Stage 2 draw from one shared budget jointly: a node's depth is its
distance from the target regardless of which stage discovered it, and Stage 2 may not extend a
path beyond what Stage 1 has already consumed on that branch.

**Keep the existing indexing convention.** This task changes scope only, not arithmetic. Do
**not** renumber the parameter, redefine how depth is counted, or change the default. The
reviewer's `NESTED_M - 2` phrasing was a description of intent, not a request to re-index —
adopt whatever convention the current code already uses (`NESTED_M` as additional nested rounds
beyond depth 0, default `1`) and express the joint budget in those same terms. A change with no
practical effect is explicitly not wanted.

**Steps:**
1. Determine from the source whether the depth cap is currently applied per stage or globally.
   If it is already global, report that and make no change — the task is then satisfied.
2. If per-stage, make it joint, leaving the parameter's name, default, and counting convention
   untouched.
3. Report the `0`-means-uncapped sentinel's behavior. An uncapped whole-network depth combined
   with R15's Stage 2 recursion is a non-termination risk; if the sentinel is kept, propose a
   hard safety bound and **ask before adding one.**
4. If step 2 changes which networks get built, cached networks from before the change are no
   longer equivalent. Confirm whether `build_network_fingerprint()` already distinguishes them
   via the code fingerprint (§14.3 hashes the AST, so a logic change should invalidate them
   automatically). Verify this rather than assuming it, and report.

**Acceptance:** a test constructing a network at a known small `NESTED_M` and asserting the
longest root-to-leaf path matches the definition above, counted in nodes. Log the realized
maximum depth at the end of network construction.

---

### R14 — Leaf nodes become external inputs, not network nodes
**Anchor:** p.24, §8.2 Step 1 and the self-history box
**Decision:** remove the unconditional self-delay embedding. A node's state space is built
only from drivers discovered by nested-MDE. Zero-driver nodes (leaves) are not genuine members
of the network; they are external inputs feeding it.

**Rationale to preserve in the implementation:** a leaf may be either a real observable or a
lagged observable. If nested-MDE selects real observable `B` and a lagged copy of `B` as two
separate zero-driver nodes, taking a lagged embedding of each makes the two identical.
Treating leaves as external inputs avoids constructing that degeneracy.

**Steps:**
1. Remove the unconditional union of `graph.graph['self_delay_set']` into every node's
   embedding spec in `_embedding_spec_for`. Lagged copies of a node's own observable remain
   admissible **when nested-MDE genuinely selects them** — the change removes the unconditional
   default, not the possibility.
2. Partition the graph explicitly into interior nodes (>= 1 discovered driver, simulated) and
   external inputs (zero drivers, not simulated). Expose this partition on the graph object;
   do not leave it implicit in edge counts.
3. Report the partition explicitly in the run log and in the results directory: how many
   external inputs, which observables and delays, and which interior nodes each feeds.
4. **Resolve before implementing — ask.** §8.2 currently relies on unconditional self-history
   to let a node reached only by a `delay != 0` edge simulate at all, and §1.1 claims the
   network free-runs "without ever seeing the real behavior again". If external inputs are fed
   from real data during `Generate()`, the simulation is no longer autonomous and that claim
   needs revising. Determine what `Generate()` actually does with a zero-driver node today,
   state what it would do under this change, and report both before editing.

**Visualization (the explicit representation this decision requires):** add a 3D rendering of
the constructed network — the x-y plane carrying the real-observable network from Stages 1
and 2, the z axis carrying the delayed observables that further drive each node. External
inputs must be visually distinct from interior nodes. Write the figure to `OUT_DIR` alongside
the existing network outputs.

**Acceptance:** a test asserting no node's embedding contains a self-lag that nested-MDE did
not select. The partition report and the 3D figure are both produced on a normal run.

---

### R15 — Stage 2 recursion on real-observable drivers
**Anchor:** p.17, §5.1 and §5.2 Step 4

1. **Determine first, then act.** Establish whether setting
   `STAGE2_ALLOW_REAL_OBSERVABLES=true` actually causes Stage 2 to run further nested-MDE on
   real-observable drivers it selects — not merely to admit delay-0 columns into its candidate
   pool. Read the code path end to end; do not infer it from §5.3's claim.
2. **If it does behave that way:** flip the default to `true`.
3. **If it does not:** do not flip it. Flag it — report exactly where the recursion stops and
   what would be required to make the switch behave as intended, and ask before implementing
   that.
4. Depth is bounded by `NESTED_M` under R13's shared budget. Stage 2 does **not** recurse as
   far as real observables passing the driver criteria allow — the budget stops it first.

   **Watch for the case this creates.** Because R12 runs Stage 1 to full depth before Stage 2
   begins, Stage 1 may already have consumed the budget on a given branch. A real observable
   Stage 2 then selects on that branch is beyond the cap and terminates immediately — the
   recursion is implemented and working, but has nothing to do there. It is live only on
   branches Stage 1 left shallower than the cap. This is correct behavior, and it must not be
   mistaken for the recursion being broken, nor "fixed" by letting Stage 2 exceed the budget.

   Log the two termination causes distinguishably — budget-bound versus driver-exhausted — and
   report, per run, how many Stage 2 real-observable picks terminated for each reason. If the
   budget-bound count is the overwhelming majority, say so plainly in the finding: it means the
   feature is effectively inert at the current `NESTED_M`, which is a configuration observation
   worth surfacing, not a defect to code around.

**Acceptance:** the finding from step 1 stated plainly; the default flipped only if step 2's
condition holds; R6's test passing against whichever outcome obtains.

---

### R16 — Sibling engine migration, behind a toggle
**Anchor:** p.5, §1.2

Migrate `simple-mde-gmn.py`'s selection engine from `dimx.MDE` to `proper_mde` /
`multi_proper_mde`, so both pipelines can perform joint multi-target MDE.

**Hard condition — read before starting.** Add a toggle in the sibling's sbatch script
selecting between the two engines (`dimx.MDE` as today, or `proper_mde`). **If implementing
that toggle requires refactoring the processes that depend on the selection engine — the
embedding path, the diagnostics that consume `dimx` result objects (`dimx.Evaluate`,
`run_mde_evaluate`), the fingerprinting, or anything else — then make no changes for this task
at all.** Report what the refactor would touch and stop. Partial migration is not acceptable.

Note the sibling scores with Pearson correlation on raw binned counts, while `proper_mde`'s
scoring is pluggable and defaults differently. The toggle must not silently change the
sibling's metric — under the `dimx` setting, results must be bit-identical to today.

**Acceptance:** either the toggle exists and the `dimx` path reproduces current results
exactly, or nothing changed and a scoping report explains why.

---

### R17 — `CCM_LIB_SIZE_PCTS` grid, after R9
**Anchor:** p.23, §7.4
**Decision:** gated on R9's findings. The reviewer's proposed `[5, 20, 40, 60, 80]` drops every
library size below 40 rows, and §18.7 concluded the rising part of a real convergence curve is
only visible in the tens of rows.

- If R9 shows curves still rising above 40 rows for most sampled pairs, adopt the coarser grid.
- If not, coarsen only the upper end — e.g. `2, 3, 5, 8, 20, 50` — cutting compute without
  losing the rising region.

Report the compute saved and the change in kept-candidate sets either way. Per §7.4, treat
this as a scientific calibration change: state the evidence in the commit message.

---

### R18 — `CCM_PREFILTER_E`, after R9
**Anchor:** p.10, §3.1 Step 2
**Decision:** keep a static `E`, but set it from R9's empirical sweep rather than by
assumption. Keep mean aggregation across library sizes in the pre-filter — slope-based gating
stays in `proper_mde`'s real CCM gate, which already performs it. Adding a slope test here
would turn a cheap screen into a second full convergence test and defeat its purpose.

**Acceptance:** the default updated to R9's calibrated value with the evidence recorded, or
left at `7` with an explicit statement that the sweep supports it.

---

### R19 — TSLS-aware `run_mde_evaluate`
**Anchor:** p.6, §1.2 and p.30, §11.2
**Reference figure:** `mde-evaluate-reference-plot.png` — an example generated from a
`dimx.Evaluate()` object. Commit it under `docs/reference/` and match its layout.

**Number of components.** `D` is the number of *direct* drivers of the target — nodes selected
as immediate drivers of the target after the complete nested-MDE algorithm (both stages), from
the CCM-pre-filtered pool. Same rule in `separate` and `combined` modes. PCA and diffusion-map
baselines each use their first `D` components, so all three methods are compared at equal
dimensionality.

**Combined mode.** PCA and diffusion-map extract components of the multivariate target.

**Layout — four stacked subplots, matching the reference:**
1. *Predictions.* In `combined` mode this panel is 2D, since the target has both x and y
   position: actual behavior in black, the trajectory reconstructed from the network's direct
   node drivers in green, first-`D` PCA in orange, first-`D` diffusion-map in blue. CAE for
   each method in the legend, as in the reference.
2. *MDE.* Trajectory of each direct node driver.
3. *Diffusion Map.* Trajectory of each of the `D` components.
4. *PCA.* Trajectory of each of the `D` components.

Panels 2–4 follow the reference's styling — one colored trace per driver/component, legend
naming each.

**CAE.** Evaluate as in the original implementation. `dimx` is installed for the sibling, so
read `dimx.Evaluate`'s source directly to recover the exact CAE definition and replicate it
rather than reimplementing from the name. If the definition cannot be recovered from source,
stop and report — do not substitute a plausible error metric.

**Ask before implementing:** confirm whether panel 1 in `combined` mode should be a spatial
x-y trajectory plot (position in the plane) or two stacked time series, one per axis. The
instruction reads as the former; the reference figure is single-target and does not settle it.

**Acceptance:** the figure is produced on a normal run with `RUN_MDE_DIAGNOSTICS=true` and
archived with the other diagnostics. Scoring routes through `_metric_for`, all three methods
on the same held-out window.

---

### R20 — TSLS-compatible replacement for `pyEDM.PredictNonlinear`
**Anchor:** p.26, §9.2 box

Theta is currently retuned by `pyEDM.PredictNonlinear`, which scores by internal Pearson rho
regardless of `METRIC` — the one stage in the pipeline that does not route through
`_metric_for`.

1. Complete R10 first.
2. **If a native S-map exists** in `cleaned_helper_functions.py`: implement the theta sweep on
   top of it, scoring through `_metric_for` per node (`f1` for neuron nodes, `correlation` for
   targets).
3. **If no native S-map exists:** do not start writing one. Propose an implementation
   approach, set out its pros and cons explicitly, and **wait for approval before writing any
   code.** Cover at minimum: how the S-map would be validated against pyEDM's on a known
   series, the cost relative to the current call, and whether a narrower change (a scoring
   wrapper) could close the gap instead.

**Acceptance:** either the theta sweep routes through `_metric_for` with an equivalence test
against pyEDM on Pearson-rho settings, or a written proposal awaiting approval.

---

### R21 — Dynamic exclusion radius in `proper_mde`
**Anchor:** p.21, §6.3
**Decision:** at each forward-selection dimension `d`, set
`exclusionRadius = max(|delay|)` across the columns currently selected, floored at the static
`EXCLUSION_RADIUS`, so the Theiler window always covers the deepest lag actually present in
the embedding.

§6.3 says `proper_mde` is a single opaque call per dimension with no staging to intercept —
but it is in-house code, so add the per-dimension staging. `EXCLUSION_RADIUS` remains the
floor and stays configurable. Report the realized radius per dimension in the run log so the
behavior is auditable.

**Acceptance:** a test asserting the radius at dimension `d` is never below the deepest
selected lag nor below the static floor. Report the effect on Run A's selections — if the
selections change, say how, since this alters which neighbors are admissible.

---

### R22 — Unify `multi_proper_mde` across both `GMN_MODE`s, conditionally
**Anchor:** p.13, §3.5 heading
**Decision:** unify only after `multi_proper_mde` gains a per-candidate auto-select path for
the single-target case, so the merge is behavior-preserving.

Routing `GMN_MODE='separate'` through `multi_proper_mde` as it stands would replace
per-candidate auto-selected `E` with the static `MULTI_MDE_E` default of `3` for every existing
single-target run — a numerical regression disguised as a refactor, invalidating cached results
without any visible knob changing.

**Order:** (1) confirm via R4 that single-target `proper_mde` really does auto-select per
candidate; (2) add that path to `multi_proper_mde` for the single-target case; (3) only then
merge the two call paths, skipping the `MULTI_MDE_TARGET_AGGREGATE` step when one target is
passed.

**If step 2 proves impractical:** stop, keep the two paths separate, and record the reason in
§3.5's rationale for the documentation pass.

**Acceptance:** with the unified path and one target, selections are bit-identical to
today's `proper_mde` output on Run A's configuration. If they are not, the merge is not
behavior-preserving — report rather than accepting the diff.
