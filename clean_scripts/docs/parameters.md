# Parameter reference

Where the numbers in `batch/*.sbatch` come from. The sbatch files carry a
one-line comment per parameter; this file holds the evidence behind the values
that have any, and says plainly which ones are guesses.

**Read the "guess or measurement?" column first.** Several of these look like
tuned constants and are not. Treating an arbitrary backstop as calibrated is
how people end up afraid to change it.

---

## 1. Scale: how big a run is this?

"Production" means the pipeline's documented defaults: `REGION_CHANNEL_CAP=999`
(all 468 real channels — CA1 147, OFC 234, mPFC 87), `MDE_D=20`, `NESTED_M=1`,
`GENERATE_LENGTH=0` (the full held-out window).

The shipped sbatch is **half** that. `REGION_CHANNEL_CAP` and `NESTED_M` are
overridable from the submitting shell -- `REGION_CHANNEL_CAP=999
saion/submit_saion.sh saion_parallel_mde.sbatch` -- so a production-scale run
needs no edit to the file:

| Parameter | Production | Shipped | Why |
|---|---|---|---|
| `REGION_CHANNEL_CAP` | 999 | 78 | 78×3 = 234 channels, exactly half of 468 |
| `MDE_D` | 20 | 10 | Half |
| `NESTED_M` | 1 | 0 = **uncapped** | Raised, not halved: at 1 the BFS stopped one step past the targets' own drivers, so anything further out became an external input regardless of the evidence (job 4774136 discarded five nodes' drivers that way). Nearly free — both scopes precompute the whole universe anyway |
| `GENERATE_LENGTH` | 0 | 400 | **Deliberately not halved** — see below |

### Why `GENERATE_LENGTH` is not halved

Under `LAG_MODE=generation_length_adjusted`, `GENERATE_LENGTH` is also the
depth of Stage 2's lag set. `0` resolves to the full held-out window (~6,700
rows here); half of that at `LAG_STEP=8` would be ~410 lags per channel — an
~8× wider Stage 2 candidate pool than any run so far, and runs at the current
width have already produced 36-hour single nodes. 400 (→ 50 lags at
`LAG_STEP=8`) is what every recent run used.

That coupling is the reason `DELAY_SOURCE=tau_count` exists: it states the lag
count directly (`LAG_STEP × LAG_COUNT`), which frees `GENERATE_LENGTH` to mean
only how long `Generate()` free-runs. The same shape stated directly is
`DELAY_SOURCE=tau_count`, `LAG_STEP=8`, `LAG_COUNT=50`. Switch deliberately:
changing the resolved lag set invalidates the MDE cache.

The coupling has cost real jobs. `GENERATE_LENGTH=400` with `LAG_STEP=4`
resolves to 100 lags per observable; leaving `GENERATE_LENGTH` at its `0`
default resolved to roughly 830.

---

## 2. Memory and time

**Measurement, not estimate** — job 40684253 (Deigo, COMPLETED,
`REGION_CHANNEL_CAP=30` → 48 of 90 observables kept by the CCM pre-filter,
`MDE_D=6`, a lag set of 101, 44,449 usable rows, 64 cores, fork workers):

- Stage 1 nodes: **under 1 s each.**
- Stage 2 nodes: **11 s to 132,219 s (36.7 h)**, heavy-tailed. Stage 2 is where
  all the time goes.
- Batch-step MaxRSS **20.7 GB** with 7 concurrent workers on a ~1.7 GB Stage 2
  DataFrame, i.e. ~2.4 GB per worker.

**Extrapolation** to `REGION_CHANNEL_CAP=78` (~125 observables after the same
~53% pre-filter keep rate): the Stage 2 DataFrame is ~125 × 101 cols × 44k rows
× 8 B ≈ 4.5 GB, and under `GPU_ENABLED=1` every worker is a *spawn* worker with
its own pickled copy (no fork copy-on-write). ~6.5 GB/worker + ~1 GB for the
torch/CUDA import, × 9 workers, + ~15 GB in the main process ≈ **83 GB**.

This is an extrapolation that has never been measured at this scale, hence the
headroom in `--mem`.

> **If the job is OOM-killed, lower `--cpus-per-task` (= the worker count), not
> `MDE_D`.** Memory here scales per worker, not with the model.

### What the core count does and does not buy

Per-node Stage 2 cost is unchanged by the core count — one node's work is one
node's work. What the core count changes is how many nodes run at once.

For the GPU-side CCM gate it buys **nothing**: preflight 4724497 on a P100
found total throughput flat at ~58 CCM sweeps/s for 1, 9, 18 and 36 workers
alike (per-sweep time scaled exactly with the worker count). One card is
already saturated by a single worker. That preflight also measured 9 workers at
3,785 MiB of the card's 16,384 (23%) against 92% for 36 — so the 9-worker shape
has large VRAM margin where the 36-worker shape had almost none.

What does speed up with more cores is the CPU-side work: the ranking stage and
everything outside the GPU kernel.

---

## 3. GPU memory: the mistake that made `gpu_budget.py` necessary

**Job 4741044.** The claim that "9 workers on one 16 GB card is comfortably
inside the measured capacity" was wrong, and wrong in a way the preflight could
not have caught. The preflight measured the ~288 MiB CUDA **context** each
worker costs just to attach to the device, and divided 16 GiB by it. It never
measured the **working set** of a real cross-map: the (M, M) distance matrix
plus the gathered (B, M, L) tensor run to gigabytes — one process in that job
was logged holding **8.51 GiB, ~30× its context**. Nine of those do not fit.
The card peaked at 16203/16384 MiB and **51 CCM calls across 85 nodes failed,
each one silently recorded as a dead end.**

A dropped candidate is a missing edge, and downstream a missing edge is
indistinguishable from one that was measured and found absent. So capacity is
no longer asserted anywhere in configuration: `src/gpu_budget.py` sizes a
per-device budget **at runtime** from what the device reports, and workers
queue for it. A memory failure now stops the run instead of shrinking the
network quietly.

Job 4744295 then recorded 20 more as dead ends even after that machinery
existed, because an `except Exception` clause upstream caught the re-raise. That
is what `mde.is_gpu_memory_failure()` and its callers now exist to prevent.

---

## 4. The CCM gate

### `CCM_WINDOW_ROWS = 800` (MODE=GMN only) — **calibrated**

Bounds the row count fed into any percentage-based CCM library sweep. This is
**critical for calibration, not a speed knob.** The gate requires skill to
visibly RISE as library size grows; too large a window means even the smallest
tested library is already past where real relationships in this data saturate,
so the gate finds no rise whether or not a real link exists.

That caused two separate production failures: job 35513363
(`apps.InteractionMatrix` reporting ~all-zero interactions) and job 35525431
(`proper_mde` selecting zero drivers for either target). Recalibrated by testing
genuinely small (tens of rows) library sizes against real data: a real, smooth,
directionally-asymmetric convergence curve appears in the **~15–400 row range**.

Do not widen this back toward "a percentage of the full ~30–45k-row dataset"
without re-validating against real small-libSize data first, and keep the
smallest resulting libSize ≥ (max E in `E_RANGE`) + 2.

### `CCM_LIB_SIZE_PCTS` (MODE=GMN only) — **calibrated**

In a 54-pair stratified sample at the 800-row window, **35% of pairs were still
rising at 40 rows** (the 5% point). A real but minority fraction — enough to
keep the small percentages (2/3/5/8%, i.e. 16/24/40/64 rows), which still carry
rising-curve signal for those pairs, while coarsening the already-converged
upper end.

### `CCM_LIB_SIZES_ROWS = 16,40,160,400,800` — **calibrated, then trimmed**

Absolute row counts, never a percentage of the (node-specific, potentially much
larger) ranking window. The convergence curve's "rising, then plateauing" shape
turns out to be a property of the **absolute** library size, not its ratio to
the surrounding frame: calibrating at an 800-row frame and at an 8000-row frame
gave the same qualitative shape for the same absolute values.

The full calibrated set was 16, 24, 40, 64, 160, 400, 800 (the percentages above
converted against the 800-row window, plus one more point so the plateau stays
visible in a larger frame). Trimmed to five on 2026-10-22 to bring `union` scope
inside a 2-day wall: the sweep is sequential within one GPU reservation, so this
is ~5/7 of the time and does **not** change peak memory. 24 and 64 were dropped
and the **endpoints kept** — `proper_ccm`'s verdict needs both the
endpoint-to-endpoint change and the fitted slope to agree, so 16 and 800 are the
two that must stay. What is left is still a near-geometric spread; five points
is ample to fit a line through.

### `CCM_SAMPLE = 10` — **a deliberate precision trade**

Random library resamples averaged per libSize. Reduced from 20 on 2026-10-22,
same 2-day motivation.

This is also the **batch dimension of the GPU cross-map**, so it buys time *and*
memory: by `gpu_budget.crossmap_bytes()`'s own terms the B-dependent terms are
~80% of a call, so halving B takes the worst observed `union` call (M=28251,
~14.9 GiB live in job 4744295) to roughly 9 GiB.

It costs **estimator precision, not model space.** No observable, driver, depth,
lag, generative window or data row is given up; the CCM rho at each library size
is averaged over 10 random draws instead of 20, so the convergence slope is
noisier.

Both this and `CCM_LIB_SIZES_ROWS` are part of the per-node cache fingerprint,
so changing either correctly **misses** every node cached at the old value
rather than silently reusing it.

### `CCM_PREFILTER_E = 7` — **a reasoned compromise, not a measurement**

The pre-filter's own static embedding dimension. Deliberately **not** a reuse of
`MULTI_MDE_E`: this pipeline's data is TSLS-converted from a 40 Hz recording,
versus 1 Hz for the sibling pipeline — 40× more rows per unit of real time. Both
fix tau = −1 (one sample step), so one embedding step here covers only 1/40th
the real time one step covers at 1 Hz. Matching that in raw sample-count terms
would need an impractical E (~40× higher, both computationally and against the
smallest-libSize-vs-E feasibility floor).

7 is the middle value of this pipeline's own already-vetted `E_RANGE` search
(`E_MIN=3`/`E_MAX=13`/`E_STEP=4` → 3, 7, 11) rather than an untested
extrapolation, and still leaves large margin against the ~38 feasibility ceiling
(`CCM_WINDOW_ROWS=800` × the smallest `LIBSIZE_RATIOS` entry 0.05, minus the +2
floor).

---

## 5. Per-node window sizing (MODE=MDE-GMN)

These four replaced `CCM_WINDOW_ROWS`/`CCM_LIB_SIZE_PCTS` for the default mode.
They separate two things one global window used to conflate: **how much data a
node is scored over**, and **which library sizes the CCM gate sweeps**.

A single fixed window with a percentage-based library grid made f1-scored
ranking degenerate for real neurons: their reset events are too sparse within
any window small enough to keep the convergence curve meaningful.

### `MIN_PRED_EVENTS = 20` — **calibrated, but a threshold choice**

How many of a neuron's own real reset events must fall in its scoring window's
PRED portion before f1 is signal rather than noise. The window grows until it
does — never past the leakage boundary B.

A plain fixed integer, not a fraction of anything neuron-specific: it answers a
question about the **metric**, not about the neuron. A "half of this neuron's own
total events" design was tried first and rejected — it made a busy neuron face a
stricter test than a quiet one for no stated reason, and, checked against the
whole window rather than its pred portion, passed on real data while the pred
portion still had too few events to score anything on.

Measured as f1 between a real TSLS slice and a lightly perturbed copy of itself,
≥200 draws per point, stratified by firing-rate tercile:

| Events | f1 std (pooled) |
|---|---|
| 5 | 0.146 |
| 10 | 0.104 |
| **20** | **0.070** |
| 40 | 0.050 |
| 80 | 0.036 |
| 160 | 0.025 |

A smooth ~1/√N decay with **no natural elbow** — so 20 is a threshold justified
by a cost/reachability trade-off, not a discovered stabilisation point. At 20,
std ≈ 0.07 is still ~3× smaller than `CROSS_MAP_RHO_MIN`'s own 0.2 default, and
on real data (`REGION_CHANNEL_CAP=15`) only **3 of 45 channels** were
unreachable within the leakage-safe range (median 3,593 rows needed). 40 and 80
buy modestly less noise at a steep cost in both unreachable channels (8/45 and
20/45) and window length (median ~10,400 rows either way).

### `MAX_WINDOW_ROWS = 30000` — **NOT calibrated**

A plain compute ceiling on that search, and a backstop rather than a tuned
value. If a node cannot reach `MIN_PRED_EVENTS` within
`min(MAX_WINDOW_ROWS, B)` it is skipped — logged, with the shortfall — rather
than scored on too few events. `MIN_PRED_EVENTS` is the statistical floor; this
is the compute ceiling. One thing each.

30000 is comfortably above the largest real row count `MIN_PRED_EVENTS=20` was
observed to need (26,748) and safely below this dataset's own leakage boundary
(~38,067).

### `TARGET_RANKING_WINDOW_ROWS = 1000` — **arbitrary**

The depth-0 seed's own fixed ranking window. Behavioural targets have no
reset-event concept, so there is nothing to extend against. Set to 1000 because
it is larger than the old 800-row default. **No other rationale.**

---

## 6. Scoring

### `METRIC = f1`, `TARGET_METRIC = correlation` — **not interchangeable**

`METRIC` scores predictions of TSLS-converted **neuron** columns. `f1` decodes
both the true and predicted series back to discrete reset events (via a
near-zero threshold calibrated from the true series itself) and matches them
within an adaptive per-event tolerance window. It therefore credits getting
spike **timing** right — which plain Pearson correlation over the whole TSLS
sawtooth does not, being dominated by the smooth, uninformative ramp segments
between spikes.

`TARGET_METRIC` scores predictions of the **targets**, which are continuous
behavioural traces with no genuine reset structure. `f1`/`gamma` event-decoding
on them produced NaN for every candidate in production. This is not the knob to
change for TSLS-aware scoring; use `METRIC`.

The split is enforced per node by `tsls_data._metric_for()`, not left to call
sites.

---

## 7. Diagnostics

### `RUN_MDE_EVALUATE = false` — **off because it can hang**

Gates only the PCA/diffusion-map baseline comparison, deliberately separate from
`RUN_MDE_DIAGNOSTICS`, which still gates the two sweeps and the 3D projection.

pydiffmap's diffusion-map fit, on real TSLS data, both **raises**
(`ValueError: Input X contains infinity...`, in 1.8 s at an 800-row scale) and
**hangs** (past 74 CPU-minutes with no completion, at the real ~31,000-row
scale). The mechanism of the hang is **not diagnosed.**

`run_mde_evaluate()` does guard each baseline with an exception handler and a
wall-clock bound in a separate process — a plain try/except is not enough,
because pydiffmap's KD-tree/BLAS calls can run a long time without ever
returning control to the interpreter, so a signal-based timeout cannot fire
until the call returns on its own. But a guard only helps where someone
remembered to put one, so this stays off until the hang is understood. Nothing
in the causal claim depends on it; it is optional tooling, and its absence is
logged once per run so a skipped diagnostic is never silent.

---

## 8. `GPU_ENABLED` and the multiprocessing start method

`GPU_ENABLED` gates the whole torch/CUDA path, and also decides the
multiprocessing start method, because **CUDA does not support `fork()`** —
forking a process that holds a CUDA context is undefined behaviour in the child,
usually surfacing as `RuntimeError: Cannot re-initialize CUDA in forked
subprocess`.

Spawn is not free, which is why it is conditional rather than always on: a spawn
worker re-imports the pipeline from scratch (pandas/numpy/matplotlib/pyEDM/gmn
all re-imported), **measured at ~18 s on Deigo**. CPU-only runs get no benefit,
so they keep fork.

The parallel builder reduces the cost of that further by collapsing every BFS
wave into one pool: the spawn startup is paid once per network build instead of
once per wave.
