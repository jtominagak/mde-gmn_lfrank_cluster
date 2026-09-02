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

**What is NOT yet validated, and matters before treating this as production-ready** (see updated
§5): the CCM benchmark used M=2000, but the actual production `CCM_WINDOW_ROWS=800` (§2c) is
smaller — smaller libraries favor tree search even more, so the real regime hasn't been tested and
the 8.88x figure may not hold there unchanged. The ranking-stage benchmark used 100 candidates;
production candidate pools run 1000-2000+. Both need a production-scale re-test before any claim
stronger than "the mechanism is real" is made.

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

## 5. Recommendation (updated 2026-09-02 after real GPU results — see §0)

**Proceed, but through a production-scale validation step before any pipeline integration —
do not generalize from the two prototype-scale numbers in §0 directly to "ship it."**

The original (pre-GPU) recommendation here was "do not proceed" — that was the correct call
*given only CPU evidence*, and is kept below for the record since the reasoning (don't chase a
compute-volume gap with CPU tuning) is still correct on its own terms. It is superseded now that
the GPU experiment it called for has actually been run and won.

**Concrete next steps, in order:**

1. **Re-run both benchmarks at production scale on the same Saion GPU**, not the prototype
   scale used to get the go signal. Specifically: the CCM gate benchmark used M=2000 while real
   production uses `CCM_WINDOW_ROWS=800` (§2c) — smaller libraries favor tree search more, so
   confirm the speedup survives at the real size, not just assume it does because M=2000 won. The
   ranking-stage benchmark used 100 candidates while production pools run 1000-2000+ — confirm
   the chunked-batching approach (chunk_b=5, chosen for a 16GB P100) still fits and still wins at
   that pool size, adjusting chunk_b as needed for whatever GPU memory is actually available.
2. **The metric-fidelity result stands regardless of the above** — a from-scratch torch engine
   exposes raw predictions, so this pipeline's `f1`/`gamma` metrics work on it exactly as they do
   on the CPU reference, unlike kEDM's `ccm()`. This was already true before the GPU confirmation
   and doesn't need re-validating.
3. **~~Confirm whether `test-gpu` is suitable for production runs~~ — deliberately deferred
   (2026-09-02, user decision).** The user has judged `test-gpu` is likely not appropriate for
   production-length runs and restricted-partition access (`gpu-v100`/`gpu-a100`) will probably be
   needed — but is explicitly choosing to defer requesting it until a working GPU-optimized
   pipeline exists to justify the request. Do not block on this; `test-gpu` remains available and
   sufficient for all the validation/development work in steps 1, 4, and 5 below.
4. **Now resolved — see §7**: torch/CUDA fork-safety (confirmed hazard, same class as kEDM's,
   concrete fix identified) and how a single shared GPU interacts with this pipeline's existing
   `ProcessPoolExecutor`/`ThreadPoolExecutor` parallelism model (a real architectural question,
   reasoned through below — not empirically tested on Saion yet, but not a blank unknown either).
5. **Revisit the memory-tiling problem from §2c** with real GPU memory numbers now in hand: the
   P100 tested has 16GB VRAM (`nvidia-smi`, confirmed 2026-09-02) — smaller than assumed when
   `chunk_b=5` was chosen defensively against CPU RAM. Now that an actual device exists to profile
   against, tune `chunk_b` (or an equivalent tiling parameter) against real VRAM headroom rather
   than a conservative guess.

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
