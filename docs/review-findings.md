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
