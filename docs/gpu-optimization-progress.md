# GPU Optimization Progress Log

Companion to `docs/gpu-optimization-plan.md` (read that first — it has the actual analysis and
the stage definitions referenced below). This file is a running, dated log of what's been tried,
what worked, what didn't, and what to do next. Update it at the end of every work session on
this effort; don't rewrite history in it, append.

Branch: `gpu-implementation` (created off `main` at commit `fd184d1`, 2026-09-01).

---

## 2026-09-01 — Branch setup

- Committed all pending work on `main` (R22 Forecast-window fix, `multi_proper_embed_dimension`,
  plotly 3D network redesign, new tests/docs) as `fd184d1`.
- Created `gpu-implementation` as a new local git branch off that commit (user's explicit choice
  over a real GitHub fork, via AskUserQuestion — reason: simpler, same repo, easy to diff back
  against `main`).
- No code changes yet on this branch as of this entry.

## 2026-09-02 — Stage 0: environment facts + kEDM API analysis + first benchmarks

**Status: Stage 0 (plan §3.0) complete. Stage 1 (batching feasibility) not started.**

### What was done
1. Checked this cluster (Deigo, OIST, host `deigo-login3.oist.jp`) for GPU hardware:
   `sinfo`/`scontrol` across every partition, `module spider cuda`/`kokkos`. **Result: no GPU
   anywhere on this cluster.** See plan §1 for full detail. This was not previously verified in
   this project — worth remembering for any future session that assumes GPU access exists here.
2. Installed kEDM: `uv add kedm` → resolved to the pre-built CPU/OpenMP wheel `kedm==0.10.1`
   (`manylinux_2_28`, matches this cluster's glibc 2.28 and this venv's Python 3.11/x86_64
   exactly — no source build was needed). `pyproject.toml`/`uv.lock` updated by `uv add`.
3. Verified `kedm.get_kokkos_config()` — confirms `Default Device: OpenMP`, thread pool
   `[1 x 40 x 1]` on this login node, no GPU arch (expected, given step 1).
4. Ran a numpy-2.4.6 compatibility smoke test: `edim`, `simplex` (1D and 2D/multivariate),
   `eval_simplex`, `smap`, `eval_smap`, `ccm`, `xmap` all imported and ran correctly, numerically
   sane outputs. One test-script bug surfaced a real API nuance, not a kEDM defect: `smap()`
   requires `lib.shape[0] == pred.shape[0]` when `target` is omitted (target defaults to `pred`,
   and the binding checks `lib` against whatever `target` resolved to) — fixed by passing
   equal-length arrays or an explicit `target`. This is now documented in plan §3.
5. Pulled kEDM's actual C++ source (`src/bindings.cpp`, `src/simplex.cpp`, `src/knn.hpp`,
   `python/kedm/__init__.py`) from GitHub directly, rather than relying on the doc site alone —
   the doc site's Python-API page doesn't show full signatures/return-shape detail; the source
   does. This is where the `ccm()`-returns-rho-only-no-raw-predictions finding, the
   `smap`-is-1D-only finding, the `Kokkos::initialize()`-at-import finding, and the dense
   `O(n_lib × n_pred)` distance-matrix finding all came from. See plan §3 for the full table.
6. Ran three benchmarks comparing `pyEDM.Simplex` (current engine) against `kedm.simplex`
   under the ranking stage's actual call shape (many small independent calls) and under
   increasing single-call data size. **Result: kEDM was ~25x slower in the realistic
   ranking-stage shape, showed irregular ~2-regime per-call latency on tiny problems (not simple
   warmup), got progressively worse than pyEDM as `n` grew, and crashed with a 281.6 GiB
   allocation attempt at n=600,000** (a deliberately oversized synthetic test to find the
   ceiling — real production data tops out around ~44,800 rows/node, well short of the crash
   point, but Benchmark C's degradation curve shows compute time getting bad well before memory
   would be the limiting factor). Full numbers in plan §2. **This overturned the working
   assumption from the pre-benchmark verbal analysis** that the ranking stage was the best,
   most-obviously-parallel kEDM target — it is not, at least not via a naive per-candidate
   call-for-call swap.
7. Re-read `multi_proper_mde`'s docstring (`cleaned_helper_functions.py:1169-1178`,
   1255-1261) to check whether the CCM gate's convergence-slope criterion could route around the
   `f1`/`gamma`-needs-raw-predictions problem by using a different metric than the ranking stage.
   **Confirmed it can't, by default**: `ccmMetric` falls back to `metric` (production default
   `'f1'`) unless explicitly overridden, so the gate needs raw predictions in production
   configuration too, not just the ranking stage. `kedm.ccm()`'s rho-only return is therefore
   blocked here as well, unless the user deliberately chooses to run the gate under
   `ccmMetric='correlation'` — a real behavior change, flagged as an open question for the user
   in plan §4/§6, not decided here.
8. Wrote `docs/gpu-optimization-plan.md` (the reference plan, revised roadmap in its §5) and
   this progress log.

### What worked
- kEDM installs and imports cleanly via the plain PyPI wheel — no build toolchain, no version
  conflicts with this project's numpy/pyEDM/scipy/networkx/plotly versions.
- `simplex()`'s 2D/multivariate mode works and is a genuine structural match for this pipeline's
  `embedded=True` generalized-embedding convention (confirmed by direct call, not just reading
  the docstring).

### What didn't work / was wrong
- The pre-benchmark plan (this session's own earlier verbal GPU-optimization analysis, before
  any code was run) assumed the forward-selection ranking stage was the best, safest kEDM target
  because it's "embarrassingly parallel across candidates." **That framing missed that "many
  small independent calls issued one at a time from Python" is a bad shape for kEDM's API
  regardless of parallelism**, because each call pays fixed Kokkos/OpenMP overhead (thread-team
  wake, host-side copy-in/copy-out, fresh distance-matrix allocation) that dominates at small
  problem sizes. Recorded here explicitly so a future session doesn't re-propose the same
  "just swap `_simplex_metric`'s `pyEDM.Simplex` call for `kedm.simplex`" plan without rediscovering
  this.
- No code in the actual pipeline (`tsls-mde-gmn.py`, `cleaned_helper_functions.py`) has been
  modified yet on this branch — everything so far is environment setup, library analysis, and
  standalone benchmarking scripts (not committed; see below).

### Scratch/benchmark scripts
Ad hoc, not committed — run directly via `.venv/bin/python3 - <<'EOF' ... EOF` during this
session, not saved as files. If these need to be rerun or extended, recreate from plan §2's
described shapes rather than searching for a saved script.

### Next steps (Stage 1, per plan §5)
1. Prototype whether kEDM's per-call overhead can be amortized by batching many *unrelated*
   ranking candidates into fewer, larger calls — check whether `simplex()`'s 2D mode could be
   abused for this (likely no — it treats columns as one joint embedding, not independent
   parallel jobs; confirm by reading `src/simplex.cpp`'s multivariate `calc_distances` before
   ruling it out for certain) versus requiring new C++ in a kEDM fork (a real multi-day
   contribution, not a config change — scope it as such if it comes to this).
2. Audit for any *single-shot* (non-looped) `EmbedDimension`/`proper_simplex` call site that
   might benefit from `kedm.edim()`/`kedm.simplex()` without needing the batching problem solved
   first (Stage 2 in the plan) — this hasn't been done yet, only the high-volume ranking-loop
   and CCM-gate call sites have been traced so far.
3. Ask the user where GPU hardware would actually be reachable, before Stage 3 becomes
   actionable — not yet asked.
4. Before any integration touches the outer `ProcessPoolExecutor` worker path: switch
   `mp_context` from `fork` to `spawn` at both call sites (`tsls-mde-gmn.py` ~2603-2604,
   ~2781-2782) and re-verify picklability (Stage 4 in the plan). Not started — no reason to do
   this yet since no kEDM call has been wired into pipeline code.
