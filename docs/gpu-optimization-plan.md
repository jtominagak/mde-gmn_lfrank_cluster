# GPU Optimization Plan for `tsls-mde-gmn`

Branch: `gpu-implementation`. This document is the reference plan for porting parts of the
`tsls-mde-gmn` pipeline to GPU-accelerated EDM via [kEDM](https://github.com/keichi/kEDM)
([docs](https://kedm.readthedocs.io/en/latest/)). It exists so that work can be picked back up
across sessions/agents without re-deriving the analysis below. Companion document:
`docs/gpu-optimization-progress.md` tracks what stage the work is actually at, what's been tried,
and what's next — update that one as work happens; update this one only when the plan itself
changes.

**Read this whole document before writing any integration code.** Section 2 in particular
overturns the naive "kEDM is a GPU EDM library, so swap it in and it'll be faster" assumption —
that assumption was tested empirically (2026-09-02) and found false for the most obvious target.

---

## 1. Environment facts (verified 2026-09-02)

- **This cluster (Deigo, OIST) has no GPU hardware anywhere.** `sinfo -N -o "%N %P %G %f"`
  across every partition (`short`, `largejob`, `compute`, `largemem`, `bigmem`, `datacp`, `dcv`)
  shows `GRES=(null)` and no `gpu` feature tag on any node. The `dcv` partition (`deigo050101`,
  `deigo050201`) is NICE DCV remote-desktop, not GPU compute passthrough — confirmed via
  `scontrol show node`, also `Gres=(null)`. `module spider cuda` tops out at `cuda/8.0.27`
  (~2016 vintage); `module spider kokkos` finds nothing.
- **Practical consequence**: nothing in this plan can be run on an actual GPU from this cluster
  today. Kokkos (kEDM's backend) also supports pure-CPU backends (`Serial`, `OpenMP`), so
  correctness/integration work can and should proceed now — but any *speedup-from-GPU* claim is
  unverifiable until GPU-bearing hardware is identified (a different OIST resource/allocation, a
  cloud instance, or a future cluster addition). **Before Stage 3 (real GPU run) starts, someone
  needs to find where that will actually execute — this is a decision for the user, not something
  to guess at.**
- **kEDM 0.10.1 is installed** in this project's `.venv` via `uv add kedm` (pre-built PyPI wheel
  `kedm-0.10.1-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl` — matches this venv's
  Python 3.11/x86_64, and this cluster's glibc 2.28 exactly meets the `manylinux_2_28` floor).
  `pyproject.toml`/`uv.lock` updated accordingly.
- **This wheel is CPU/OpenMP-only** — confirmed via `kedm.get_kokkos_config()`: `Default Device:
  OpenMP`, `GPU architecture: none`, thread pool auto-sized to `40` on this login node. GPU
  support would require the separate `kedm-cuda12x`/`kedm-cuda13x` wheels or a source build with
  `-DKEDM_ENABLE_GPU=ON`, neither of which is useful until GPU hardware exists to run on.
- **numpy 2.4.6 compatibility confirmed empirically**, not just by version-range reading:
  `edim`, `simplex` (both 1D and 2D/multivariate), `eval_simplex`, `smap`, `eval_smap`, `ccm`, and
  `xmap` were all exercised against real float32 arrays in this venv and returned numerically
  sane results (see §2 and `docs/gpu-optimization-progress.md` for the exact smoke-test output).
  No ABI issues. kEDM's own `pyproject.toml` only pins `numpy>=1.24.0` (no upper bound) — this
  matters less than the empirical check, since a source-built or wheel-built numpy 2.x binding
  can still be ABI-broken in principle; it isn't, here.
- Other installed versions relevant to compatibility: `pyEDM 2.5.0`, `networkx 3.6.1`,
  `scipy 1.17.1`, `plotly 7.0.0` (from the prior session's 3D-plot work).

## 2. Critical finding: call-granularity mismatch, verified by benchmark

The original (pre-benchmark) hypothesis was that kEDM's `simplex()` is a near-drop-in
replacement for `_simplex_metric()`'s `pyEDM.Simplex` call in the forward-selection ranking loop
(`proper_mde`/`multi_proper_mde`, `cleaned_helper_functions.py:1440-1449` — one call per
remaining candidate, per dimension, tens of thousands of calls per node in production), and that
Kokkos/GPU batching would accelerate exactly this workload.

**This was tested directly and is false for the current API, as currently used.** Two
benchmarks (`.venv/bin/python3`, single-threaded, this cluster's login node, 2026-09-02):

**Benchmark A — realistic ranking-stage call shape** (E=2, embedded 2-column state space,
lib=400 rows, pred=200 rows, 149 sequential per-candidate calls, mirroring the actual loop):

| Engine | Total (149 calls) | Per call |
|---|---|---|
| `pyEDM.Simplex` (current) | 1.070 s | 7.18 ms |
| `kedm.simplex` (2D multivariate) | 26.930 s | 180.74 ms |

**kEDM was ~25x slower**, not faster, in this exact usage pattern.

**Benchmark B — per-call latency has two regimes, and it's not simple warmup.** Ten consecutive
isolated `kedm.simplex` calls, same tiny problem size, gave:
`0.2009, 0.0003, 0.0003, 0.0984, 0.2020, 0.0003, 0.0003, 0.0979, 0.0007, 0.0003` (seconds) — an
irregular alternation between sub-millisecond and ~100-200ms, not a one-time slow first call. The
likely mechanism: Kokkos's OpenMP backend spins up its thread team once at `import kedm`
(`Kokkos::initialize()` is called at Python **module import time**, confirmed by reading
`src/bindings.cpp`'s `PYBIND11_MODULE` block — `Kokkos::finalize()` is registered as an
interpreter-exit capsule). Idle OpenMP threads go through a spin-wait-then-sleep cycle; waking a
sleeping thread team costs orders of magnitude more than reusing an already-spinning one. A tight
loop of tiny, independent calls with Python-level work in between (exactly this pipeline's
ranking-stage shape) lands on both sides of that cycle unpredictably.

**Benchmark C — scaling with problem size is worse than pyEDM's, and eventually catastrophic.**
Single isolated calls, `E=1`, varying `n` (70/30 lib/pred split):

| n | pyEDM.Simplex | kedm.simplex | ratio (pyEDM/kedm) |
|---|---|---|---|
| 600 | 106.81 ms | 0.48 ms | 222x (kedm faster — favorable regime) |
| 6,000 | 101.26 ms | 589.03 ms | 0.17x (kedm 5.8x slower) |
| 60,000 | 154.97 ms | 8,639.80 ms | 0.018x (kedm 56x slower) |
| 600,000 | — | **crash** | `Kokkos ERROR: Host memory space failed to allocate 281.6 GiB (label="tmp_distances")` |

**The crash is structural, not a fluke** — confirmed by reading `src/simplex.cpp`: every
`simplex()`/`smap()` call unconditionally allocates
`TmpDistances tmp("tmp_distances", pred.extent(0), lib.extent(0))`, a **dense
`n_pred × n_lib` float32 matrix**, with no chunking/blocking and no approximate-neighbor
fallback (that exists only for `ccm()`/`ccm_naive()`, which expose an `accuracy` parameter —
`simplex`/`smap` do not). At n=600,000 (lib≈420k, pred≈180k) that's `420000 × 180000 × 4 bytes ≈
282 GB` — matches the observed crash size almost exactly. Production data in this pipeline runs
to **~44,800 rows** per node (confirmed: `data_after_mde_head_position_x_head_position_y.csv`
from job 40563312's archive, `wc -l` = 44,788) — a full-length lib/pred split at that size is
`~44,800² × 4 bytes ≈ 8 GB` for one call, survivable on the 64GB nodes this pipeline already
requests, but Benchmark C shows the *compute time* degrades severely well before memory becomes
the limiting factor.

**Conclusion — this is the single most important thing to internalize before writing any
integration code**: kEDM's Python-exposed API issues one Kokkos-parallel-region (and one dense
distance matrix) **per call**, and its OpenMP backend has real, unpredictable per-call
overhead. This is not a CPU-vs-GPU issue — a CUDA backend would very likely make per-call kernel
launch overhead *worse*, not better, for this exact call shape (kernel launch latency dominates
GPU-side for small problems even more than OpenMP thread-wake latency does on CPU). **The
current pipeline architecture — many independent, small, per-candidate EDM calls issued
one-at-a-time from Python — is structurally mismatched with every Kokkos backend kEDM offers,
not just the CPU one available here.** Fixing that mismatch (batching many candidates into far
fewer, larger calls) is a prerequisite for *any* backend to help, GPU included. This reframes
everything below.

## 3. kEDM API reference (from `src/bindings.cpp`, v0.10.1 — verified by reading source, not
just docs)

| Function | Signature (Python-facing) | Returns | Verified notes |
|---|---|---|---|
| `edim(timeseries, E_max=20, tau=1, Tp=1)` | 1D only | `int` — best E | Cheap, single time series. Matches `proper_embed_dimension`'s *shape* but not its metric-awareness (see caveats). |
| `simplex(lib, pred, *, target=None, E=1, tau=1, Tp=1)` | **1D or 2D** `lib`/`pred` | `np.ndarray` — raw predictions, length `n_pred-(E-1)*tau` | 2D = "mixed multivariate embedding," confirmed working via smoke test (`kedm.simplex(lib2d, pred2d, target=x1d, E=2, ...)`). This is the one function that matches this pipeline's generalized (`embedded=True`, arbitrary explicit column list) convention. `target` defaults to `lib` if omitted; if `lib.shape[0] > target.shape[0]` it raises. |
| `eval_simplex(lib, pred, *, target=None, E=1, tau=1, Tp=1)` | **1D only** — raises on 2D | `float` — Pearson r vs target (or vs pred if no target) | No 2D/multivariate path at all (unlike plain `simplex`). |
| `smap(lib, pred, *, target=None, E=1, tau=1, Tp=1, theta=1.0)` | **1D only** — raises on 2D | `np.ndarray` — raw predictions | **Confirmed empirically it does NOT accept 2D `lib`/`pred`** (throws `"lib and pred must be 1D arrays"`), unlike `simplex`. Also: if `target` omitted, it defaults to `pred`, and the binding then requires `lib.shape[0] == target.shape[0]` — i.e. **when `target` is omitted, `lib` and `pred` must be equal length** (verified: passing differently-sized lib/pred with no target raises `"lib size and target size must be equal"`; fixed by either passing equal-length lib/pred or an explicit `target`). |
| `eval_smap(...)` | same shape as `smap` | `float` — Pearson r | Same 1D-only, same target-default gotcha. |
| `ccm(lib, target, *, lib_sizes=[], sample=1, E=1, tau=1, Tp=0, seed=0, accuracy=1.0)` | 1D only | `list[float]` — **one Pearson r per lib_size, nothing else** | Confirmed by reading `bindings.cpp`: the C++ core (`edm::ccm`) never surfaces raw per-row predictions to Python — only the aggregated correlation per library size. The whole resampling/library-size sweep (this pipeline's `CCM_LIB_SIZES_ROWS × CCM_SAMPLE` loop) happens *inside* one C++ call. `accuracy<1.0` enables approximate nearest-neighbor search for speed (the only kEDM function exposing this). |
| `ccm_naive(...)` | same as `ccm`, no `accuracy` | `list[float]` | Present in source, undocumented on the doc site. Reference/comparison implementation, no reason to use it here. |
| `xmap(dataset_2d, edims, tau=1, Tp=0)` | 2D dataset (columns = series) | `np.ndarray` — full pairwise interaction-strength matrix | **One-shot, full-library cross map — NOT a convergent (library-size-sweep) computation.** Docstring is explicit: *"kedm.xmap performs a single cross mapping using the full time series as library. To perform Convergent Cross Mapping, use kedm.ccm."* One internal loop already computes all pairs in the dataset in one call — the closest thing kEDM has to a "batched" primitive, but it answers a different, weaker question (single-library association, not CCM convergence) than this pipeline's causality gate needs. |
| `get_kokkos_config()` | none | `str` | Confirmed: `Default Device: OpenMP` on this install; also reports GPU arch when built with `KEDM_ENABLE_GPU`. |

Additional structural facts (from reading `src/bindings.cpp`, `src/simplex.cpp`,
`python/kedm/__init__.py`):
- `Kokkos::initialize()` runs at `import kedm` (module scope); `Kokkos::finalize()` at
  interpreter exit via an atexit-style capsule. This is the actual mechanism behind kEDM's
  documented fork-hang warning: forking a process **after** `kedm` has been imported (and its
  OpenMP thread team started) duplicates only the calling thread in the child, leaving Kokkos's
  thread-pool bookkeeping in an inconsistent state — a classic post-fork-with-live-threads
  deadlock, not a vague/unexplained bug.
- `python/kedm/__init__.py` sets `OMP_PROC_BIND=close` by default if unset — an environment
  detail worth being aware of if this pipeline's own thread/process pinning assumptions
  (`inner_workers`, `outer_workers` sizing) ever interact with it.
- Every `simplex`/`smap` call does a **host-side threaded copy-in/copy-out** of every input/output
  array (`bindings.cpp`'s `copy()` helper: allocate a Kokkos View, `Kokkos::parallel_for` element
  copy, `Kokkos::deep_copy`) — additional fixed per-call overhead on top of the thread-wake cost,
  most visible exactly at the small-array end of Benchmark C.

## 4. Reconciling with this pipeline's actual metric requirements

`multi_proper_mde`'s docstring (`cleaned_helper_functions.py:1169-1178`) confirms the `metric`
argument (default in production: `'f1'`, per this session's prior confirmation that
`METRIC="f1"` is the real default) is applied identically at **three** places: the ranking
step, the `crossMapRhoMin`/`embedDimRhoMin` thresholds, and — critically — **the CCM gate's own
convergence-slope criterion** (`ccmMetric`, which defaults to `metric` when unset, per
`cleaned_helper_functions.py:1255-1261`). This means the CCM gate needs raw per-row predictions
(to compute `f1`/`gamma` via `adaptive_event_metrics_from_tsls`/`_coincidence_factor`,
`cleaned_helper_functions.py:789-845`) in the **production-default** configuration too, not only
in some non-default configuration. `kedm.ccm()`'s rho-only return value is therefore a genuine
blocker for the CCM gate under the actual default settings — it would only work if `ccmMetric`
were deliberately set to `'correlation'`/`'mae'`, which is a real behavior change to the
pipeline's science, not a transparent engine swap. Flag this to the user explicitly before ever
switching `ccmMetric` to make kEDM's `ccm()` usable — that decision is theirs, not an
implementation detail.

## 5. Revised roadmap (supersedes the informal ranking given in this session's earlier verbal
analysis, which was written before the benchmarks above)

Ordered by realistic payoff given everything in §2-§4. Each stage names its own go/no-go
condition — do not proceed to the next stage until the current one's condition is actually met.

### Stage 0 — Environment & library analysis — **done 2026-09-02**
Everything in §1-§4. See `docs/gpu-optimization-progress.md` for the session log.

### Stage 1 — Decide whether batching is worth building at all
Before writing any pipeline integration: prototype, in isolation (a throwaway script, not
pipeline code), whether kEDM's per-call overhead can be amortized by batching many candidates
into **fewer, larger** calls rather than by using kEDM's existing per-call API as-is. Two
concrete options to test, cheaply, before committing:
1. **Multivariate stacking**: instead of one `kedm.simplex()` call per candidate, can several
   *unrelated* candidates be evaluated by concatenating them along a shared axis and splitting
   the result? (Likely no — `simplex`'s 2D mode treats all columns as one joint embedding, not
   independent parallel jobs — but confirm by reading `src/simplex.cpp`'s multivariate distance
   calc before assuming.)
2. **Manual Kokkos-level batching**: would require writing new C++ (a new pybind11-exposed
   function in a kEDM fork) that loops over many `(lib, pred, target)` triples *inside* a single
   `Kokkos::initialize()`-amortized call, sharing one thread-team wake per batch instead of one
   per candidate. This is a real contribution to kEDM itself, not a pipeline-side change — scope
   it as such (a multi-day C++ task, not a config change) before committing to it.
**Go/no-go**: if neither option shows a measured speedup over pyEDM at production-realistic
candidate-pool sizes (1000-2000+) in a standalone benchmark, do not proceed to Stage 2 for the
ranking stage. Revisit only if GPU hardware becomes available (kernel-launch overhead may change
the calculus, but per §2 it's likely to make things worse, not better, for this call shape).

### Stage 2 — Lower-risk, narrower integration points (worth doing regardless of Stage 1's outcome)
These don't depend on solving the batching problem, because each replaces a *single* Python-level
call site with a *single* kEDM call, not a loop:
- **`edim()` for single-series embedding-dimension estimates** where this pipeline currently uses
  `pyEDM.EmbedDimension` or the custom KDTree `proper_simplex`-based sweep for a *single*
  candidate at a time outside the hot ranking loop (if any such single-shot use exists — audit
  before assuming). Low volume, low risk, easy to benchmark in isolation first.
- **`gmn`'s own pre-existing `kedmSimplex`/`kedmSMap` hooks** (`FunctionType` enum in the
  installed `gmn` package, `.venv/lib/python3.11/site-packages/gmn/{Common,Node}.py`) — these
  exist upstream already; this pipeline's own monkeypatches (`_generalized_node_forecast()`,
  `_generalized_node_generate()` in `tsls-mde-gmn.py`) currently `raise RuntimeError(...)` for
  those branches instead of implementing them. Implementing the `kedmSimplex` branch (not
  `kedmSMap` — see below) is low-effort scaffolding-wise, since the config plumbing already
  exists — but confirm real value first: GMN's `Generate()` calls SMap **sequentially, once per
  simulated timestep** (via `lastDataOut` feedback), which is a single small call per step, not a
  batch — likely to hit the same per-call-overhead problem as the ranking stage. Benchmark this
  specific call shape (one call, small n, high call frequency) before investing effort here.
  **`smap`'s 1D-only constraint (§3) is a hard blocker for any node with more than one driver** —
  this pipeline's whole reason for the `_generalized_node_*` monkeypatches is *multi-driver,
  non-uniform-delay* embedding, so `kedmSMap` cannot be wired in for the general case without
  first extending kEDM's own C++ `smap` binding to accept 2D input the way `simplex` already
  does. Scope this as a kEDM contribution, not a pipeline change, same as Stage 1 option 2.

### Stage 3 — Actual GPU execution (blocked on hardware)
Not started, and not startable from this cluster. Before this stage can begin:
1. Identify where GPU hardware is actually reachable (a different OIST cluster/allocation,
   cloud, or a future Deigo addition) — **ask the user**, do not assume.
2. Once identified, install `kedm-cuda12x`/`kedm-cuda13x` (or build from source with
   `-DKEDM_ENABLE_GPU=ON` and the right `-DKokkos_ARCH_*` flag for the target GPU) there, and
   re-run the Stage 0/1 benchmarks on that hardware — the CPU/OpenMP numbers in §2 do not
   transfer to CUDA numbers; kernel-launch overhead has a different (and plausibly worse, per
   §2's conclusion) profile than OpenMP thread-wake overhead for small per-call workloads.
3. Do not attempt this stage until Stage 1 has produced a batching strategy that shows a genuine
   win on CPU first — if the call-granularity mismatch isn't fixed, moving to GPU hardware won't
   fix it either.

### Stage 4 — Fork/spawn safety (must be resolved before kEDM is called from *any* worker process)
Both outer `ProcessPoolExecutor` pools (Stage 1 and Stage 2 BFS waves in `tsls-mde-gmn.py`,
lines ~2603-2604 and ~2781-2782) currently use `mp_context=mp.get_context('fork')`. Per §3, this
is a real deadlock risk the moment `kedm` is imported anywhere before those pools fork — not a
hypothetical, given this pipeline's own prior history with a structurally similar `pyEDM`
per-call-Pool-spawn bug (see memory `perf_pyedm_pool_bug`). Required before any integration
lands in the outer worker path: switch those two pools to
`mp_context=mp.get_context('spawn')`, and re-verify pickle-ability of every argument passed into
`ProcessPoolExecutor.submit`/`.map` (spawn re-imports fresh, nothing is fork-inherited). The
*inner* `ThreadPoolExecutor` layer (`inner_workers`) is unaffected — threads don't fork.

## 6. Open questions to verify before writing pipeline-integration code

- Does `simplex()`'s 2D/multivariate mode use the *same* `E` for every column, or can it accept
  a heterogeneous per-column embedding matching this pipeline's own `_generalized_embed()`
  convention (arbitrary per-driver delay, not a uniform `tau`/`E`)? The pybind11 signature takes
  one scalar `E`/`tau` for the whole call — read `src/simplex.cpp`'s multivariate distance
  calculation (`calc_distances(Dataset lib, Dataset pred, ...)`) before assuming compatibility.
- Whether any single-shot (non-looped) `EmbedDimension`/`proper_simplex` call sites exist that
  would benefit from `edim()`/`simplex()` without needing the batching work in Stage 1 at all —
  worth a proper audit pass rather than assuming none exist.
- Whether `ccmMetric='correlation'` (as opposed to the production default, which resolves to
  `'f1'`) would still be scientifically acceptable for the CCM gate specifically — this is a
  question for the user, not an implementation choice to make unilaterally, given §4's finding
  that the default routes `f1`/`gamma` through the gate too.
