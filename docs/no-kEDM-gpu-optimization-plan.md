# No-kEDM GPU Optimization Plan (pyEDM + torch workaround)

Branch: `no-kEDM-gpu-optimization` (created off `main` at `fd184d1` — deliberately NOT off
`gpu-implementation`, so this direction stays isolated from the kEDM-specific work there and
from `main`'s original CPU-only pipeline). Companion progress log:
`docs/no-kEDM-gpu-optimization-progress.md`. Sibling document for the other direction:
`docs/gpu-optimization-plan.md` (the kEDM approach) — read that one too; several findings here
echo or contrast with it directly.

**Read §2 before writing any integration code.** The empirical benchmarks there are less
favorable than the proposal's premise suggested, and the reasoning for *why* matters more than
the numbers themselves for deciding what (if anything) to build next.

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
hardware could plausibly offset — but this is a hypothesis, not yet a finding, because no GPU
exists on this cluster to test it against.** Do not treat this distinction as license to assume a
GPU win; it is the reason to keep this line of investigation open pending hardware, not a
substitute for testing it.

## 4. What does NOT need re-litigating for this approach (shared with the kEDM plan)

- No GPU hardware exists on this cluster (Deigo) — see `docs/gpu-optimization-plan.md` §1 for the
  full verification (`sinfo`/`scontrol` across every partition, `module spider cuda/kokkos`).
  Nothing here changes that; it's the same blocking fact for both directions.
- `multi_proper_mde`'s docstring-confirmed metric routing (`cleaned_helper_functions.py:1169-1178,
  1255-1261`) — `ccmMetric` defaults to `metric` (production default `'f1'`), so the CCM gate
  needs raw predictions in the default configuration, not just the ranking stage. This is *why*
  the torch approach's raw-prediction access matters (§1) — it is not a nice-to-have, it's the
  actual production requirement.

## 5. Revised, evidence-based recommendation

**Do not proceed to a full pipeline integration of the brute-force/dense-batched approach as
prototyped.** It is confirmed slower on the only hardware available to test it on, at every scale
tested, even after fixing both the correctness bug and the threading misconfiguration that made
the first attempt look worse than it really was. Building this out further on CPU would be
speculative engineering against a metric (GPU throughput) this cluster cannot measure.

**What is worth keeping and building on, in order:**

1. **The metric-fidelity result stands on its own merit, independent of speed.** If there is ever
   a reason to want a from-scratch, fully custom-metric-capable EDM engine — for portability, for
   avoiding a C++ dependency, or simply because kEDM's `ccm()` is a hard blocker under
   `ccmMetric='f1'` — the validated torch primitives from §2a/§2c's second prototype (properly
   batched-compacted CCM cross-map; batched multivariate Simplex) are a correct starting point,
   not a wrong one. Keep the code (see progress log for its current location); don't discard it as
   worthless just because it's not fast yet.
2. **Do not invest further CPU-side tuning effort chasing a speed win here.** The compute-volume
   gap (tree search vs. brute force) is large (3-9x in the measurements above) and unlikely to be
   closed by micro-optimizing the torch code further on CPU — the entire premise of this approach
   (regular, dense, parallel arithmetic) is what GPU hardware, not CPU vectorization, is suited to
   accelerate.
3. **Before any further investment, get access to actual GPU hardware and re-run the exact same
   three validated benchmarks (§2c) there.** This is a small, cheap, decisive experiment —
   re-running `torch.cdist`/`topk`-based code on a CUDA device requires no algorithm change, only
   a `.to('cuda')` on the tensors and re-measuring. If the GPU numbers still lose to
   `pyEDM.Simplex`/scipy's `KDTree` (both CPU-only, unchanged), the compute-volume theory in §3 is
   wrong and this whole direction should be closed out. If GPU numbers win, that's the actual
   green light to build the full integration — not before.
4. **This is the same hardware-access blocker as the kEDM plan's Stage 3** — ask the user where
   GPU hardware is actually reachable (a different OIST cluster/allocation, cloud, or a future
   Deigo addition) before treating this as actionable. Not yet asked in this branch's own work.
5. **If/when GPU access exists, also revisit the memory-tiling problem found in §2c** (the 16 GB
   OOM from an unchunked candidate batch) — a real GPU implementation will need deliberate,
   correctly-sized chunking of the batch dimension to fit in GPU memory (which is typically
   *smaller* than a compute node's system RAM, making this more urgent on GPU than it was here),
   not just "port the CPU code as-is."

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
