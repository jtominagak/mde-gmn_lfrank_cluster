# No-kEDM GPU Optimization Plan (pyEDM + torch workaround)

Branch: `no-kEDM-gpu-optimization` (created off `main` at `fd184d1` — deliberately NOT off
`gpu-implementation`, so this direction stays isolated from the kEDM-specific work there and
from `main`'s original CPU-only pipeline). Companion progress log:
`docs/no-kEDM-gpu-optimization-progress.md`. Sibling document for the other direction:
`docs/gpu-optimization-plan.md` (the kEDM approach) — read that one too; several findings here
echo or contrast with it directly.

---

## START HERE (updated 2026-09-03)

**If you are picking this up to implement it, read in this order and skip the rest for now:**

1. **§0** — the current verdict. Two targets were evaluated; only ONE is proceeding (the CCM
   gate, 12.30x at production scale). The ranking stage is explicitly abandoned for reasons that
   are not fixable by tuning — do not re-attempt it.
2. **§7c** — what hardware measurement settled, including two hypotheses of mine it falsified.
   This is what makes the spec simpler than it would otherwise be; §7b above it is superseded
   pre-measurement reasoning, kept only for context.
3. **§8** — the implementation spec. §8.1 (fork-safety) is already DONE and committed. §8.6 marks
   the next step with "← next".

**Current state in one line**: research and validation are complete and positive; the actual
implementation has NOT started — no torch code exists anywhere in `scripts/src/`, and setting
`GPU_ENABLED=1` today buys nothing but spawn-startup cost.

**§2-§5 are historical reasoning**, not current guidance. §2 in particular reports the approach as
a loss at every scale — that was true on CPU-only evidence and was overturned by §0's GPU results.
Read it to understand *why* things were built the way they were, not to decide what to do next.

---

## 0. UPDATE (2026-09-02): the GPU hypothesis is confirmed — real hardware, real win

§2-§5 below were written from CPU-only benchmarks on Deigo, where this approach measured 3-9x
*slower* than the existing engines. §3 proposed a specific reason to not give up on it anyway:
that slowdown looked compute-volume-bound (more raw FLOPs, brute-force vs. tree search), not
overhead-bound like kEDM's failure — and a compute-volume deficit is exactly what GPU throughput
can close, unlike kEDM's structural per-call overhead. §5 item 3 named the exact decisive
experiment: re-run the same two validated scripts on an actual CUDA device and see which way it
goes.

**That experiment has now been run**, on Saion (`test-gpu` partition, Tesla P100-SXM2-16GB,
compute capability (6,0) — see `docs/no-kEDM-gpu-optimization-progress.md` for the full
procedure and raw terminal output). Both benchmarks reversed cleanly:

| Target | CPU (Deigo, §2c) | GPU (Saion, P100) | 
|---|---|---|
| CCM gate (`_crossmap_predict` sweep) | **3.2x slower** than reference | **8.88x faster** than reference |
| Ranking stage (multivariate Simplex sweep) | **~9x slower** than reference | **3.37x faster** than reference |

Correctness held on GPU too, at the same precision level as the CPU runs (CCM max diff 8.27e-05,
essentially identical to CPU's 8.28e-05; ranking-stage median/mean diff 2.89e-05/3.22e-04,
matching CPU's 2.89e-05/3.24e-04 — the same benign near-tied-neighbor pattern, not a new issue).
This is not a marginal result — it's a near order-of-magnitude reversal on both targets, on a
nine-year-old (Pascal-generation) GPU on the cluster's *open, unrestricted* partition, before any
attempt at further tuning.

**This changes §5's recommendation from "do not proceed" to "proceed, with the specific next
steps in the updated §5 below."** §2-§4's CPU-only findings are NOT invalidated — they remain the
correct explanation of *why* naive CPU tuning wouldn't have helped, and the bug/threading fixes
documented there are still exactly what any GPU port needs too (both scripts already carry them
forward). Read them as the reasoning that got this branch to the point of testing on real
hardware, not as the final verdict.

**UPDATE 2 (2026-09-02, same day): production-scale re-test done — the two targets diverged.**
The paragraph above called for re-testing at real production scale before trusting these numbers.
That re-test is done (`docs/no-kEDM-gpu-optimization-progress.md` has the full raw output). The
two targets did NOT move the same direction:

| Target | Prototype scale | Production scale | Verdict |
|---|---|---|---|
| CCM gate (M=2000 → M=800, `CCM_WINDOW_ROWS`) | 8.88x faster | **12.30x faster** | **Win held, improved** |
| Ranking stage (100 → 1500 candidates, pred 2,000 → 22,394 rows) | 3.37x faster | **0.60x (1.67x SLOWER)** | **Win reversed to a loss** |

The ranking-stage reversal is not noise or a tuning artifact — it's explained precisely (not just
plausibly) by the same compute-volume-vs-tree-search mechanism from §3, now showing its other
edge: **torch's brute-force cost scales as `O(candidates × n_pred × n_lib)` with no shortcut,
while the reference's cost is dominated by a largely pred-size-independent fixed cost (its
index-build over the library) plus a cheap per-pred-row query.** Fitting the reference's two
prototype-vs-production data points to `cost = fixed + k·n_pred` gives `fixed≈14.5ms`,
`k≈0.93 microseconds/row` — meaning growing `n_pred` from 2,000 to 22,394 barely moved the
reference's per-call cost (16.4ms→35.4ms), while it multiplied torch's total compute by the full
`O(n_pred × n_lib)` product. Predicting torch's production time from pure compute-volume scaling
(`0.486s × (1500/100) × (22394/2000) × (22394/20000) ≈ 91.4s`) matches the observed **88.6s**
within 3% — this is quantitatively confirmed, not hand-waved. **The lesson: tree search's
`O(log n_lib)`-per-query advantage over brute force's `O(n_lib)`-per-query cost becomes
overwhelming once `n_pred` itself is large — no amount of GPU throughput or bigger batching fixes
an asymptotically worse algorithm once the constant-factor GPU advantage is outrun by data size.**
Note this could NOT have been predicted from the CCM gate's result alone: CCM's production
`n_pred` (=`M`=800) shrank at production scale, while the ranking stage's production `n_pred`
(≈22,394) grew by 11x — the two targets' production regimes moved in opposite directions on the
one dimension (`n_pred`) that determines which way this mechanism cuts.

**Revised bottom line**: proceed with the CCM gate (confirmed win, both scales, by a wide and
improving margin). **Do not proceed with the ranking-stage torch primitive as currently
designed** — increasing `chunk_b`/batch size will not fix this (the slowdown is compute-volume-
bound, not overhead-bound; the observed time already matches pure FLOP-count scaling with no
excess overhead to recover), and neither will more GPU throughput in isolation, since the
reference's asymptotic advantage grows without bound as `n_pred` grows. A real fix would need a
genuinely sub-linear GPU-based nearest-neighbor search (e.g. a GPU KD-tree, or an approximate-NN
library — FAISS-GPU is the obvious candidate to investigate) rather than the dense brute-force
`cdist`+`topk` approach prototyped here — a materially bigger undertaking than what's been built
so far, and not yet started. See the updated §5 for the concrete, per-target next steps.

---

## 1. The proposal, and which part of it is actually confirmed

The idea (as given): MDE has no kEDM implementation, so instead reimplement the candidate sweep's
underlying EDM computation directly in PyTorch (batched across candidates, GPU-capable), using
`pyEDM` as the correctness reference rather than kEDM as the runtime engine. The stated rationale:
this could route around kEDM's confirmed blocker (`docs/gpu-optimization-plan.md` §4 — `ccm()`
returns only aggregate rho per library size, never raw predictions, so this pipeline's
`ccmMetric`/`metric`='f1'/'gamma' event-based scoring — the actual production default — can't be
computed from it).

**This specific premise is confirmed true, directly, by prototyping** (§3 below): a from-scratch
torch reimplementation is *our own code*, so it can return raw per-row predictions from every
primitive it implements (a Simplex-equivalent, a CCM-cross-map-equivalent), and those feed
straight into this pipeline's existing, unchanged `_score()` (`cleaned_helper_functions.py:789-847`)
— `f1`, `gamma`, `correlation`, `mae` all work, exactly as they do today. This is a genuine,
structural advantage over kEDM that doesn't require touching kEDM's own C++ source at all.

**The separate premise — that this would also make the candidate sweep faster — is NOT
confirmed, and the CPU-only evidence gathered so far points the other way.** See §2. Critically,
*why* it points that way differs in kind from kEDM's own failure mode, which matters for what to
do next (§5).

## 2. Empirical findings (2026-09-02, this cluster, CPU-only — no GPU hardware exists here,
same constraint as `docs/gpu-optimization-plan.md` §1)

Environment: `torch==2.14.0+cpu` (deliberately installed from PyTorch's dedicated CPU wheel
index, not the default PyPI index — the default `pip install torch`/`uv add torch` pulls a
CUDA-bundled build with several GB of `nvidia-*` dependency packages even on a GPU-less machine;
killed and redone via `uv add torch --index-url https://download.pytorch.org/whl/cpu`, then
pinned properly for future syncs via `[tool.uv.sources]`/`[[tool.uv.index]]` in `pyproject.toml`
so it doesn't silently regress on the next `uv sync`). Confirmed `torch.cuda.is_available() ==
False` as expected.

**A real, unrelated environment bug surfaced and was fixed along the way**: `uv add torch`
initially failed resolution entirely (`dimx==1.1.3` not found for the `python_full_version >=
3.14 and sys_platform == 'win32'` marker branch) — a pre-existing latent issue in this project's
`pyproject.toml` (unbounded `requires-python`, no platform scoping) that kEDM's narrower wheel
coverage happened not to trigger, but torch's broader (including Windows) wheel coverage did.
Fixed by adding `[tool.uv] environments = ["sys_platform == 'linux' and platform_machine ==
'x86_64'"]`, scoping resolution to what this project actually runs on. Worth knowing about
independent of this GPU effort — it would have bitten the next `uv add` of any
broadly-multi-platform package.

### 2a. First implementation attempt: a real correctness bug, not just a speed question

The first prototype (batched torch reimplementation of `_crossmap_predict`,
`cleaned_helper_functions.py:599-712` — the CCM gate's per-subsample cross-map engine) produced
predictions that diverged from the numpy/KDTree reference by up to **1.16 absolute** — far
beyond float32-vs-float64 noise. Root cause, found by direct comparison against the reference
line-by-line: the reference computes `d_min` (the weighting formula's normalizing nearest-neighbor
distance, Sugihara & May 1990's `exp(-d/d_min)`) **after** compacting to exactly the first `k`
*valid* (non-self, non-excluded) neighbors — `cleaned_helper_functions.py:686-694`'s
`np.where`+reshape step. The first torch attempt computed `d_min` from the raw nearest-of-`k_query`
over-fetched neighbors *before* removing invalid ones, so a self-match or an excluded
near-in-time neighbor could silently become the weighting reference point. **This is exactly the
kind of subtle, non-obvious parity bug this document's §1 risk warning exists to flag** — get
this exact kind of detail wrong in a real integration and network-structure decisions (which
columns MDE selects) could silently diverge from the validated reference algorithm's behavior
without any exception being raised. Fixed in the second prototype via a proper batched
compaction (`torch.argsort` on the invalid mask, stable sort, then `torch.gather` the first `k`
columns — mirrors the reference's reshape exactly). After the fix: **max abs diff 8.28e-05**
across 280,000 compared prediction points, fully consistent with float32 vs. float64 precision
noise, 0 NaN-pattern mismatches. **Numerical parity is achievable, but was not free — it required
finding and fixing a real bug, not a first-try success.**

### 2b. Threading: the same class of finding as kEDM's, independently confirmed

Before finding the bug above, the first prototype was also timed at **~25x slower** than the
reference (54.8s vs 1.47s for an equivalent 140-call sweep). Instrumenting every internal step
found the actual cause: torch's default CPU thread pool (`torch.get_num_threads() == 40` on this
node, auto-matching core count) imposes large fixed dispatch overhead **per elementwise tensor
op** on tensors this small (tens of thousands of elements) — a single chained sequence of
`torch.where`/`torch.exp`/`torch.cumsum` calls, each individually dispatching across 40 threads,
cost **3.2 seconds** for what should be sub-millisecond arithmetic. Setting
`torch.set_num_threads(1)` dropped the *same* computation from 7.09s to 0.33s — a 21.7x speedup
from turning threading **off**. **This is independent confirmation of the exact mechanism found
in the kEDM investigation** (`docs/gpu-optimization-plan.md` §2's Benchmark B): this pipeline's
actual per-call/per-op problem sizes are small enough that naive multi-threaded dispatch (whether
Kokkos/OpenMP or torch's own ATen thread pool) is a net loss, not a win, and must be explicitly
disabled or the batch granularity must be large enough to amortize it. Any torch code path this
pipeline ends up using must call `torch.set_num_threads(1)` (or a small, deliberately-chosen
count) rather than relying on the default.

### 2c. Even after fixing both the bug and the threading problem: still slower, at every scale tested

Three benchmarks, all single-threaded, all numerically validated against the actual reference
engine (not just internally self-consistent):

| Target | Scale | Reference | Torch (best version) | Result |
|---|---|---|---|---|
| CCM gate (`_crossmap_predict`), one candidate's full `libSizes × sample` sweep | M=2000, L up to 1900, 140 calls, cached (M,M) `cdist` once | 1.19s | 3.77s | **torch 3.2x slower** |
| Ranking stage (`_simplex_metric`/pyEDM.Simplex), one dimension's full candidate sweep | lib=20,000, pred=2,000, E=2, 100 candidates, chunked batches of 5 (memory-forced) | 4.01s | 35.78s | **torch ~9x slower** |

**Production-scale sanity check on the CCM gate specifically**: the actual sbatch configuration
(`scripts/batch/tsls-mde-gmn.sbatch:419-441`) uses `CCM_WINDOW_ROWS=800`,
`CCM_LIB_SIZES_ROWS="16,24,40,64,160,400,800"` — i.e. the CCM gate's real embedding pool is
**~800 rows**, smaller than the M=2000 tested above, not larger. Since a tree-based search's
advantage over brute force grows as the library gets smaller relative to the full pool, the real
production regime is very likely *worse* for the torch approach than the already-unfavorable
M=2000 result, not better.

**A memory ceiling was hit too, structurally identical to kEDM's own crash**
(`docs/gpu-optimization-plan.md` §2, the 281.6 GiB `tmp_distances` crash): the naive
"stack all 100 candidates into one `torch.cdist` call" version of the ranking-stage benchmark was
SIGKILLed (OOM) — `100 candidates × 2,000 pred rows × 20,000 lib rows × 4 bytes ≈ 16 GB` for a
single dense tensor. The number in the table above is from a **chunked** version (5 candidates
per `cdist` call) required just to make the benchmark run at all — meaning any real
implementation must tile the batch dimension, which itself reduces the batching benefit the
whole approach was supposed to provide.

## 3. Why this differs from kEDM's failure mode — the one reason this branch still exists

kEDM's slowdown (`docs/gpu-optimization-plan.md` §2) is **overhead-bound**: each call pays a
mostly-fixed Kokkos/OpenMP thread-team-wake tax and host-copy tax, largely independent of how
much actual arithmetic the call does. That kind of overhead is a property of the *backend*
(Kokkos), not the *algorithm*, and — per that document's own conclusion — is likely to persist on
an actual GPU backend too (kernel-launch latency is at least as bad as OpenMP thread-wake
latency for small workloads).

This benchmark's slowdown is **compute-volume-bound**: the reference engines (scipy's `KDTree` in
`_crossmap_predict`, and whatever `pyEDM.Simplex` uses internally — likely a similarly
tree/index-based nearest-neighbor search) do genuinely less arithmetic per query than a brute-force
dense-distance-matrix-plus-topk approach, because a tree search is roughly `O(log L)` per query
against an `O(L)` library, while the batched-tensor approach here is `O(L)` per query (a full scan
over the library, batched across many queries at once, but still fundamentally a linear scan, not
a logarithmic one). **This is a difference in FLOP count, not dispatch overhead — and FLOP-bound
slowdowns are exactly what GPU hardware is built to close or reverse**, because GPU throughput for
regular, dense, parallel arithmetic (a big batched `cdist`+`topk`) is typically 10-50x a CPU core's
throughput for the same operation, whereas GPU kernel-launch overhead doesn't get better than
CPU thread-wake overhead in the same way. **In short: kEDM's problem is structural and
backend-independent; this approach's CPU-measured problem is a raw-throughput deficit that GPU
hardware could plausibly offset.**

**Confirmed 2026-09-02 (see §0): this is no longer a hypothesis.** Both benchmarks reversed on an
actual GPU (Saion, P100) — CCM gate went from 3.2x slower to 8.88x faster; ranking stage went
from 9x slower to 3.37x faster. The reasoning above was the correct explanation, verified by
testing it rather than trusting the analogy.

## 4. What does NOT need re-litigating for this approach (shared with the kEDM plan)

- No GPU hardware exists on this cluster (Deigo) — see `docs/gpu-optimization-plan.md` §1 for the
  full verification (`sinfo`/`scontrol` across every partition, `module spider cuda/kokkos`).
  Nothing here changes that; it's the same blocking fact for both directions.
- `multi_proper_mde`'s docstring-confirmed metric routing (`cleaned_helper_functions.py:1169-1178,
  1255-1261`) — `ccmMetric` defaults to `metric` (production default `'f1'`), so the CCM gate
  needs raw predictions in the default configuration, not just the ranking stage. This is *why*
  the torch approach's raw-prediction access matters (§1) — it is not a nice-to-have, it's the
  actual production requirement.

## 5. Recommendation (updated 2026-09-02, twice — first after prototype-scale GPU results, then
after production-scale re-test; this is the current version)

**Proceed with the CCM gate. Do not proceed with the ranking-stage torch primitive as currently
designed — it needs a different algorithm, not more tuning, before it's worth revisiting.**

The original (pre-GPU) recommendation was "do not proceed at all" (CPU-only evidence). The first
update, after prototype-scale GPU results, was "proceed with both, pending production-scale
validation." That validation is now done (see the "UPDATE 2" block above) and the two targets
split: CCM gate's win held and improved (8.88x→12.30x); the ranking stage's win reversed into a
1.67x loss, for reasons that are understood precisely (compute-volume scaling, not a fixable
inefficiency — see above). Both earlier recommendations are kept below for the record.

**Concrete next steps, in order:**

1. **CCM gate: proceed to integration design.** This target is validated at real production scale
   on real hardware, twice, with an improving margin. It's the one to actually build.
2. **Ranking stage: do not integrate the current design.** Re-confirm this is truly a dead end
   (not a chunk_b/batching artifact) is not needed — the compute-volume-scaling match (91.4s
   predicted vs. 88.6s observed) already rules that out quantitatively. If this target is worth
   revisiting later, it needs a genuinely sub-linear GPU nearest-neighbor search (e.g. GPU KD-tree,
   or an approximate-NN library like FAISS-GPU), which is new engineering, not a parameter tweak —
   scope it as such if picked up again, and don't re-attempt the brute-force approach at a "maybe
   a bigger batch would help" premise; the mechanism rules that out.
3. **The metric-fidelity result stands regardless of the above, for the CCM gate specifically** —
   a from-scratch torch engine exposes raw predictions, so `f1`/`gamma` work on it exactly as on
   the CPU reference, unlike kEDM's `ccm()`. This was already true before the GPU confirmation.
4. **~~Confirm whether `test-gpu` is suitable for production runs~~ — deliberately deferred
   (2026-09-02, user decision).** The user has judged `test-gpu` is likely not appropriate for
   production-length runs and restricted-partition access (`gpu-v100`/`gpu-a100`) will probably be
   needed — but is explicitly choosing to defer requesting it until a working GPU-optimized
   pipeline exists to justify the request. Do not block on this; `test-gpu` remains available and
   sufficient for all the validation/development work in steps 1, 5, and 6.
5. **Resolved — see §7**: torch/CUDA fork-safety (confirmed hazard, same class as kEDM's, concrete
   fix identified) and how a single shared GPU interacts with this pipeline's existing
   `ProcessPoolExecutor`/`ThreadPoolExecutor` parallelism model. This reasoning was done with both
   targets in mind; re-read it now scoped to the CCM gate only, since that's the one proceeding.
6. **Revisit the memory-tiling problem from §2c/§7 with real GPU memory numbers now in hand**: the
   P100 tested has 16GB VRAM. The CCM gate's own memory footprint at production scale (M=800) is
   trivial compared to what the ranking-stage test needed — no tiling pressure expected for this
   target specifically; confirm this assumption when building the real integration rather than
   carrying over the ranking-stage-motivated tiling design unnecessarily.

<details>
<summary>Original (CPU-only) recommendation — superseded by §0, kept for the record</summary>

Do not proceed to a full pipeline integration of the brute-force/dense-batched approach as
prototyped. It is confirmed slower on the only hardware available to test it on, at every scale
tested, even after fixing both the correctness bug and the threading misconfiguration that made
the first attempt look worse than it really was. Building this out further on CPU would be
speculative engineering against a metric (GPU throughput) this cluster cannot measure. What was
worth keeping regardless: the metric-fidelity result (§1's premise); not chasing CPU-side tuning
further since the gap was compute-volume-bound, not overhead-bound; and getting access to actual
GPU hardware to run the decisive experiment before deciding anything else — which is exactly what
happened next (§0).

</details>

<details>
<summary>First GPU-results recommendation (prototype scale only) — superseded by the production-
scale re-test above, kept for the record</summary>

Proceed with both targets, but through a production-scale validation step before any pipeline
integration — do not generalize from the two prototype-scale numbers (CCM 8.88x, ranking 3.37x)
directly to "ship it." This was the right caution to have written down: the ranking-stage number
it warned against trusting is exactly the one that reversed once actually re-tested.

</details>

## 6. Open questions to verify before any further work

- Whether `pyEDM.Simplex`'s own internal nearest-neighbor implementation is genuinely tree-based
  (as assumed in §3) or something else — this was inferred from its CPU speed profile relative to
  a known-brute-force alternative, not confirmed by reading pyEDM's own source. Worth confirming
  if the compute-volume theory in §3 needs to be relied on for a real GPU go/no-go decision.
  reference: `pyedm` is installed at `.venv/lib/python3.11/site-packages/pyEDM` (or its compiled
  extension) in this project's venv — not yet inspected in this session.
- Whether GPU memory capacity (once hardware is identified) would even fit this pipeline's actual
  production embedding sizes (CCM gate ~800 rows — trivially small; ranking stage's lib/pred could
  be tens of thousands of rows per node) without the chunking problem in §2c/§5 becoming a hard
  blocker rather than a solvable engineering detail.
- Whether theta retuning (`retune_theta_for_all_nodes`, direct `pyEDM.SMap` calls,
  `tsls-mde-gmn.py:4190-4242`) and GMN's own `Generate()`/`Forecast()` SMap calls are worth a
  similar torch reimplementation — not prototyped in this session (only Simplex-family primitives
  were tested); S-Map's locally-weighted regression is a different (batched weighted least
  squares, not weighted k-NN averaging) computation and would need its own from-scratch
  correctness validation before any speed claims are made about it.

## 7. Fork-safety and GPU-sharing design (2026-09-02) — closes plan item 4 above

These two questions were flagged as open in the 2026-09-02 progress log entry. Both are now
reasoned through from documented CUDA/PyTorch behavior — not yet empirically re-verified on Saion,
but no longer blank unknowns. Verify empirically (a small standalone script, same spirit as
§2's prototypes) before relying on this section for a real integration decision.

### 7a. Fork-safety: confirmed hazard, same fix as kEDM's Stage 4

**CUDA does not support `fork()` at all** — this is not a performance footgun like kEDM's
Kokkos-at-import issue, it is an explicit, documented incompatibility. PyTorch's own
multiprocessing guidance states plainly that `spawn` or `forkserver` are *required* to use CUDA in
subprocesses; forking a process that already holds a CUDA context produces undefined behavior in
the child, commonly surfacing as `RuntimeError: Cannot re-initialize CUDA in forked subprocess`
(this is the loud, well-known form of the failure — a real crash, not a silent hang like kEDM's
OpenMP issue, which makes it easier to *detect* but no less necessary to *design around*).
Sources: [PyTorch issue #15734](https://github.com/pytorch/pytorch/issues/15734),
[PyTorch issue #77159](https://github.com/pytorch/pytorch/issues/77159),
[PyTorch forums thread](https://discuss.pytorch.org/t/unable-to-fix-runtimeerror-cannot-re-initialize-cuda-in-forked-subprocess-to-use-cuda-with-multiprocessing-you-must-use-the-spawn-start-method/208718).

The subtlety: PyTorch's CUDA context creation is *lazy* — plain `import torch` does not itself
create a context; one gets created on first CUDA-touching call (`.cuda()`, a CUDA tensor
allocation, etc. — `torch.cuda.is_available()` queries the driver but is generally safe/does not
itself force full context init, though this specific boundary is worth re-verifying rather than
relying on memory of it). In principle, if the *parent* process orchestrating
`ProcessPoolExecutor` never touches CUDA before forking, and each forked child creates its own
CUDA context fresh, entirely after the fork, this could work even under `mp_context='fork'`. This
is fragile in practice — one accidental CUDA-touching import or debug check anywhere in the main
process's code path before the pool is created breaks every worker with the same hard crash — and
is exactly the shape of latent bug this codebase has already been bitten by once (the pyEDM
per-call-Pool-spawn issue in memory `perf_pyedm_pool_bug`). **Recommendation, matching kEDM's
Stage 4 exactly**: switch both outer `ProcessPoolExecutor` pools in `tsls-mde-gmn.py`
(`~2603-2604`, `~2781-2782`) from `mp_context=mp.get_context('fork')` to
`mp_context=mp.get_context('spawn')` before any torch/CUDA code is introduced into a code path
those pools can reach — not conditional on whether the fragile fork-after-no-CUDA-touch approach
would happen to work today, since that's one careless import away from breaking. Same downstream
cost as the kEDM plan noted: spawn re-imports fresh per worker (no fork-inherited state, slower
startup, every submitted argument must be picklable) — accept this cost once, deliberately, rather
than debug an intermittent CUDA crash later.

### 7b. GPU-sharing under the existing thread/process parallelism model

This pipeline's current parallelism is two-layer: outer `ProcessPoolExecutor` (Stage 1/2 BFS-wave
workers, CPU-core-budgeted via `outer_workers`) and, inside each worker, an inner
`ThreadPoolExecutor` (`inner_workers`, sized because "pyEDM.Simplex releases the GIL" so CPU
threads get real parallelism). **This sizing model is calibrated for CPU core budgets and does
not translate to a GPU unchanged**:

- **Across outer processes**: each forked-then-spawned worker process that touches CUDA gets its
  *own* CUDA context on the *same physical device* (a P100 has one GPU; multiple processes can
  each hold a context on it, but they then share one 16GB VRAM budget and the device's single
  execution engine between them — this is supported, not an error, but it's *contention*, not
  *added throughput*, unlike CPU cores where more processes running independently really do add
  parallel capacity). Running `outer_workers` processes each independently launching large batched
  GPU calls is likely to fight over both memory (each process's own batch competing for the same
  16GB) and the GPU's scheduler, not scale the way it does on CPU cores today.
- **Across inner threads within one process**: multiple Python threads *can* issue CUDA operations
  concurrently on the same device from a single process — the CUDA driver serializes/queues kernel
  launches issued to the same (default) stream safely, no explicit locking required for
  correctness. But without deliberately using separate CUDA streams per thread (an optimization
  this pipeline does not currently do and would need to add), concurrent GPU-bound work from
  multiple threads does not get true concurrent execution the way CPU-bound threaded work does
  today — the GPU processes queued kernels from all threads, interleaved, on largely the same
  timeline it would if called from one thread. Thread count is not the lever for GPU throughput
  the way it is for CPU throughput.

**Practical implication for a future integration, not yet empirically verified**: the actual lever
for GPU throughput in this pipeline is *batch size per call* (as demonstrated in §2c/§0 — one
call handling many candidates at once), not *thread or process count*. A reasonable starting
design: route GPU-eligible work (ranking-stage candidate sweeps, CCM-gate sweeps) through a single
GPU-owning worker (or a small, deliberately-capped number, sized to fit VRAM concurrently, likely
1-2 rather than today's CPU-core-sized `outer_workers`) that internally batches as much of its
pending work as memory allows per call, rather than assuming the existing `outer_workers ×
inner_workers` product still describes available parallelism once GPU calls are introduced. This
needs an actual experiment (queue depth vs. throughput, concurrently-submitted-batch vs. VRAM
ceiling) before being trusted as a final design — treat it as the working hypothesis to test, not
a settled conclusion.

**Status update (2026-09-03): §7b has now been MEASURED, and two of its claims are FALSIFIED.**
§7a is implemented (§8.1) and independently confirmed on hardware. The measurement results and
what they change are in §7c below — read that before using anything in §7b above, which is
retained as the pre-measurement reasoning.

### 7c. Measured results (2026-09-03, Saion `test-gpu`, P100) — §7b partly wrong

`scripts/gpu_prototypes/measure_gpu_sharing.py`, run on `saion-gpu05`. Full raw output in the
progress log. **The headline: measurement contradicted the reasoning in §7b on two of four
points, and the corrections all SIMPLIFY the integration rather than complicate it.**

| Hypothesis (from §7b) | Result | Evidence |
|---|---|---|
| H1 "batch size is the real throughput lever" | **FALSIFIED** | 16x more batching bought **11%** (14.80 → 13.19 ms/sweep) |
| H3 "more inner threads don't add GPU parallelism" | **CONFIRMED** | dead flat: 67.7 / 68.3 / 68.3 / 67.9 sweeps/s at 1/2/4/8 threads |
| H2 "more processes = contention, not throughput" | **INCONCLUSIVE** — test design flaw | see below |
| H4 "CUDA cannot fork" (§7a) | **CONFIRMED on hardware** | `RuntimeError: Cannot re-initialize CUDA in forked subprocess` |

**H1 — why the "batching is the lever" claim was wrong.** Total FLOPs are constant no matter how
the work is chunked; batching only pays when *per-call overhead* is significant relative to
per-call work. At the CCM gate's production size (M=800) it isn't — the workload is already
compute-bound, so consolidating 448 kernel launches into 28 changed almost nothing. **Design
consequence: do NOT build a candidate-level batching layer into the CCM-gate integration.** The
already-validated primitive (which batches across the `sample` dimension per libSize) is
sufficient; adding another batching tier would be complexity bought for ~11%.

**H2 — the apparent 3x "win" is not a real scaling result.** H3 establishes one process does
~68 sweeps/s, so H2's 32 sweeps is **~0.47s of actual compute** — against wall times of
9.8-28.3s. H2 therefore measured spawn startup (torch import + CUDA context creation), not GPU
contention. The monotonic improvement across `procs=1→2→4` is also consistent with page-cache
warming, since `procs=1` ran first and cold. **The contention question remains genuinely open**;
answering it would need a redesigned test where per-worker GPU work dominates startup.

**But the question H2 was meant to settle is now substantially de-risked anyway**, by three facts
that don't depend on it:
1. **Spawn startup is negligible in production.** 10-28s per worker per wave, against real
   per-node `run_mde_for_target` runtimes of minutes to many hours (job 40582445's
   `head_position_x`: 49,238s for one node) — 0.02-0.3% overhead. The benchmark's 0.47s workload
   made this look alarming; at production scale it isn't.
2. **VRAM pressure is nil.** Peak GPU memory for the whole run: **1.27 GB of 17.1 GB.** Several
   concurrent workers fit comfortably; this also retires plan §5 item 6 (no tiling design needed
   for this target — that concern was inherited from the ranking-stage experiments, which are now
   out of scope).
3. **Only the CCM gate is being ported.** The ranking stage stays on CPU (§0 UPDATE 2) and
   dominates per-node runtime, so each worker spends only a fraction of its life touching the GPU
   — intermittent multi-worker GPU access, not sustained contention.

**Revised recommendation, replacing §7b's "shrink `outer_workers` drastically"**: do NOT change
`outer_workers`/`inner_workers` sizing preemptively. Keep today's CPU-core-based sizing, and
measure contention in the real end-to-end A/B (§8.6 step 5) where the workload is realistic. §7b's
reasoning that a GPU "is a single shared resource in a way CPU cores aren't" is still true in
principle — it simply doesn't bind here, because GPU work is a small, intermittent slice of each
worker's total runtime and the memory footprint is tiny.

**H3's confirmation is also a simplification, not a constraint**: extra threads neither help nor
*hurt* (flat, not degrading), so the existing inner `ThreadPoolExecutor` structure can stay
exactly as it is. No threading redesign is needed for the integration.

## 8. Code-level integration spec (CCM gate only)

Written 2026-09-03, after the production-scale results narrowed the scope to one target. This is
the section an implementing agent should follow. **Scope: the CCM gate only.** The ranking stage
is explicitly OUT of scope (§0's UPDATE 2 — it loses at production scale for algorithmic reasons
that batching cannot fix); do not route `_simplex_metric` through torch on the strength of this
spec.

### 8.1 Fork-safety — DONE (2026-09-03)

Implemented and tested; nothing further needed here unless the design changes.

- `load_config()` gained `GPU_ENABLED` (`TSLS_MDE_GMN_GPU_ENABLED`, default **False**) and
  `MP_START_METHOD` (derived: `'spawn'` when GPU is on, `'fork'` otherwise).
- `build_network()` and `build_nested_mde_network()` both gained `mp_start_method='fork'`
  (defaulted so every existing caller is unaffected), threaded from `main()`'s
  `cfg['MP_START_METHOD']`; both outer `ProcessPoolExecutor` pools now use it.
- **Deliberately conditional, not a blanket switch**: a spawn worker re-imports this module from
  scratch — measured **18.3s** on Deigo — and the pools are recreated per BFS wave, so CPU-only
  runs would pay that repeatedly for a path they never use. With `GPU_ENABLED=False` the pipeline
  is byte-for-byte unchanged.
- Preconditions verified before making the change: gmn monkeypatching is at module scope
  (re-executes correctly on re-import), there is a proper `if __name__ == "__main__"` guard, and
  there are no `global` statements (no runtime-mutated module state a spawn worker would miss).
- Regression test: `tests/test_gpu_fork_safety.py` — checks the default stays fork, the flag flips
  it, `mp_start_method` actually reaches both pools (not silently dropped), and that a real
  spawn-context `ProcessPoolExecutor` runs and round-trips the argument *types* the real
  `submit()` passes (DataFrame/Path/dict/list/ndarray). Existing suite
  (`test_r12_stage_serialization`, `test_r13_shared_depth_budget`,
  `test_metric_routing_depth_ge1`) re-run green after the change.

### 8.2 The exact integration point

`proper_ccm()` in `cleaned_helper_functions.py` (~lines 1030-1060). Its current structure is
already ideal for this — do not restructure it:

```python
for t in range(sample):                      # (1) ALL random draws happen first, in order,
    for i, L in enumerate(libSizes):         #     on this function's own rng
        lib_idx_fwd = rng.choice(M_fwd, size=min(L, M_fwd), replace=False); lib_idx_fwd.sort()
        lib_idx_rev = rng.choice(M_rev, size=min(L, M_rev), replace=False); lib_idx_rev.sort()
        tasks.append((t, i, lib_idx_fwd, lib_idx_rev))

def _run_one_cell(task):                     # (2) one (sample, libSize) cell = TWO
    ...                                      #     _crossmap_predict calls (fwd + rev),
    preds_fwd = _crossmap_predict(...)       #     each scored by _score()
    c2t = _score(obs_fwd, preds_fwd, target_measurement, gammaDelta=gammaDelta)
    preds_rev = _crossmap_predict(...)
    t2c = _score(obs_rev, preds_rev, columns_measurement, gammaDelta=gammaDelta)

mapper = pool.map if pool is not None else map   # (3) mapped over the sample x libSize grid
```

Two properties of this make the integration clean and low-risk:
- **Reproducibility is already decoupled from execution order.** Every `rng.choice` draw happens
  up-front, sequentially, before any compute — the function's own docstring calls this out as why
  `pool` doesn't change results. The torch path reuses those exact same `lib_idx` arrays, so a
  given `seed` must still reproduce bit-identical library draws. **Do not regenerate the draws
  inside the torch path.**
- **Scoring stays on CPU, unchanged.** `_score()` (with `f1`/`gamma`) keeps operating on raw
  prediction arrays exactly as today — this is the whole reason this approach beats kEDM's `ccm()`
  (§1/§4). The torch path replaces *only* `_crossmap_predict`, never `_score`.

### 8.3 What to build

**New module `scripts/src/gpu_edm.py`** (new file, not additions to
`cleaned_helper_functions.py`, so the torch import stays optional and isolated):

- `is_available()` → bool. True only if torch imports AND `torch.cuda.is_available()`. Must never
  raise when torch is absent — a CPU-only environment (Deigo) must import this module fine.
- `batched_crossmap_predict(embed, pred_vals, lib_idx_batch, k, exclusionRadius, time_indices,
  full_dist=None, device=...)` → `(B, M)` array. Port from the validated
  `scripts/gpu_prototypes/validate_batched_ccm.py::batched_crossmap_predict_torch` **unchanged in
  algorithm** — it is already numerically validated against `_crossmap_predict` (max abs diff
  1.89e-05 on GPU at production scale, 0/140 NaN-pattern mismatches). Carry over its two
  hard-won details: the first-`k`-valid **compaction** before computing `d_min` (getting this
  wrong silently produced 1.16-magnitude errors — see §2a), and `torch.set_num_threads(1)`.
- Returns a **numpy** array (caller stays numpy-only; keep torch types from leaking into
  `cleaned_helper_functions.py`).

**Modification to `proper_ccm()`**: add a `use_gpu=False` keyword. When True *and*
`gpu_edm.is_available()`, group the already-drawn tasks by `(direction, len(lib_idx))` — same `L`
means same tensor shape, which is exactly what the batch dimension requires — issue one batched
call per group, then score each resulting row with the same `_score()` calls as today. When False
or unavailable, run the existing `mapper(_run_one_cell, tasks)` path untouched.

**Fallback behaviour is mandatory, not optional**: `use_gpu=True` on a machine with no GPU must
transparently fall back to the CPU path, not raise. The same code must run on Deigo (no GPU) and
Saion (GPU) with only the flag differing.

### 8.4 Wiring the flag through

`cfg['GPU_ENABLED']` already exists (§8.1). It needs threading to `proper_ccm`'s callers the same
way `mp_start_method` was threaded to the pools: `main()` → `build_network()` →
`build_nested_mde_network()` → `run_mde_for_target()` → `proper_mde`/`multi_proper_mde` → the
CCM-gate `proper_ccm`/`multi_proper_ccm` call. Add it as a defaulted keyword (`use_gpu=False`) at
each hop so no existing caller changes behaviour. Note `run_mde_for_target` runs **inside** the
worker processes — with `GPU_ENABLED=1` those are spawn workers, so each initialises its own CUDA
context. **Per §7c, do NOT change `outer_workers`/`inner_workers` sizing as part of this work**
(the earlier §7b hypothesis that they'd need to shrink drastically did not survive measurement);
keep today's sizing and evaluate contention in the §8.6 step-5 A/B instead.

### 8.4b What §7c's measurements removed from this spec

Three things this spec would otherwise have called for are now explicitly NOT needed — recorded
so they don't get reintroduced by someone reasoning from first principles later:
- **No candidate-level batching tier** (H1 falsified: ~11% for 16x batching). The validated
  primitive's existing per-libSize batching across the `sample` dimension is enough.
- **No threading redesign** (H3: extra threads are neutral, neither helping nor hurting). The
  existing inner `ThreadPoolExecutor` stays as-is.
- **No VRAM tiling/chunking design** (peak 1.27 GB of 17.1 GB for the whole measurement run). The
  `chunk_b`-style tiling the ranking-stage prototype needed is irrelevant to this target.

### 8.5 Test strategy

1. **Equivalence test (the important one)** — `tests/test_gpu_ccm_equivalence.py`: run
   `proper_ccm` twice on the same synthetic data with the same `seed`, once `use_gpu=False` and
   once `use_gpu=True`, and assert the returned `c2t`/`t2c` arrays match within float32 tolerance.
   Must **skip cleanly** (not fail) when no GPU is present, so it's runnable on Deigo. This is the
   test that would catch a silent scientific regression — it compares the pipeline-level output,
   not just the primitive.
2. **Fallback test**: `use_gpu=True` with no GPU available must produce results identical to
   `use_gpu=False`, no exception.
3. **Reproducibility test**: same `seed`, `use_gpu` True vs False → identical library draws
   (assert on the drawn indices, not just final scores).
4. Re-run the existing suite; `GPU_ENABLED=False` must leave every current test green.

### 8.6 Order of work

1. ~~Fork-safety~~ — done (§8.1), and independently confirmed on hardware by H4 (§7c).
2. ~~Run `measure_gpu_sharing.py`~~ — done 2026-09-03 (§7c). Outcome: no sizing changes needed;
   three planned pieces of complexity dropped (§8.4b).
3. **Build `gpu_edm.py` + `proper_ccm(use_gpu=...)` + the equivalence test (§8.3, §8.5).** ← next
4. Thread the flag through the call chain (§8.4).
5. End-to-end run on Saion with `GPU_ENABLED=1` at small scale (`REGION_CHANNEL_CAP` low), diffed
   against the same config with `GPU_ENABLED=0`, before any long production run. This is also
   where multi-worker GPU contention gets evaluated for real (§7c), since the H2 measurement
   couldn't answer it.
