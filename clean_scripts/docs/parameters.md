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

## 7. Forward selection and the free run

### `MDE_D` — **a cap on embedding coordinates, lags included**

`MDE_D` is the number of forward-selection rounds, so it caps how many columns
a node selects, and every selected column becomes a coordinate of that node's
S-map embedding. A lagged column counts like any other.

It used to be multiplied by the number of delays per observable (×4 with three
lags under `union`), on the reasoning that it should count distinct real
observables. Nothing enforced that; the multiplier just let a node take four
times as many columns of any kind. In job 4799407 the median node took 40
columns — the inflated cap exactly — covering 22 distinct observables, mostly
short lags of columns already chosen. A TSLS series at lag 1, 2 and 4 rows
correlates 0.995 / 0.991 / 0.981 with itself and differs from `X(t) − k·dt`
only when a spike fell inside the lag (1–4 % of rows), so those coordinates add
little but a 40-dimensional S-map. That network's theta search took 290,824 s,
against 54,551 s without lags.

### `MDE_MIN_GAIN = 0` — **a stopping rule the reference algorithm lacks**

The reference forward selection (dimx `Run.py`) keeps a candidate whenever its
*cumulative* score clears `crossMapRhoMin`. Once a node's embedding scores
well, a candidate that makes it worse still clears that bar, and selection runs
to `MDE_D` whenever anything passes the CCM gate. In job 4799404 that accepted
41.7 % of all edges at a **negative** marginal gain: one node peaked at
f1 = 0.359 with 5 drivers and was handed 5 more, ending at 0.263, and the
10-driver embedding is what GMN used.

With `MDE_MIN_GAIN` set, from the second round on a candidate is only
considered if it beats the last accepted score by more than this; selection
stops when none does. `0` is the minimal version: never add a driver that
leaves prediction no better. It does not account for noise: an f1 computed
from ~20 reset events moves in steps of roughly 0.05, so gains far below that
are not evidence of anything. A threshold calibrated against that granularity
(or against surrogate drivers) is the obvious next step; `0` is not it.
`none` restores the reference behaviour.

Checked out of sample on job 4799404's data: selection on rows 0–16k at
`Tp = 20`, then every prefix of the reference selection scored on rows
16k–27.4k, which selection never saw. For the four neuron nodes that selected
anything, out-of-sample f1 peaked at 1–2 drivers and fell to a half to a third
of that peak at the 9–10 drivers the reference rule keeps (e.g. 0.279 → 0.141,
0.230 → 0.074); `MDE_MIN_GAIN = 0` stopped at the peak in three of the four and
one driver short of it in the fourth. For `head_position_x` out-of-sample
correlation kept rising to 0.94 at 9 drivers, and `0` kept all 10. It also
made the MDE call 7–10× faster for those neurons (≈15 s against 109–165 s).

`MDE_MIN_GAIN = surrogate` replaces the fixed 0 with a per-node, per-round
threshold: 19 surrogates (remaining candidates circularly shifted by a random
¼–¾ of the window) give the null distribution of the round's score, and a
candidate must beat its mean by `z·sd`, with `z` the normal `1 − 0.05/N`
quantile for the `N` candidates the round compares (forward selection keeps
the best of N, so the null has to be a maximum). It is cheap — a few percent
of a ranking pass — but on the same check it **matched `0` on every neuron node
and stopped `head_position_x` at 1 driver (out-of-sample 0.754) where the real
optimum was 9 (0.941)**. Circular shifts keep any periodic structure, and
behaviour on a track repeats lap after lap, so shifted behaviour-locked
neurons stay partly informative and the null comes out too high. It is
therefore an option, not the default.

### `DELAY_SET = -16,-63,-173` — **derived from the recording's spike timing**

A lagged TSLS copy `X(t−k)` says anything `X(t)` does not only when a spike
fell in `(t−k, t]`; otherwise it is exactly `X(t) − k·dt`. That happens when
`TSLS(t) < k` rows, so the fraction of rows in which lag `k` is informative is
the CDF of the TSLS values, and the lag informative in a fraction `q` of rows
is the `q`-quantile of TSLS (in rows). On the library rows (before
`PREDICTION_START`) of `J1620210605_epoch_8_freq_40_Hz.csv`, over the 213
neurons of job 4799404's network, median over neurons:

| q | lag | IQR over neurons |
|---|---|---|
| 0.10 | 16 rows (400 ms) | 3–52 |
| 0.25 | 63 rows (1.6 s) | 15–159 |
| 0.50 | 173 rows (4.3 s) | 54–372 |

Firing is bursty — median rate 0.35 Hz, median inter-spike interval 5 rows
(125 ms) — so short lags are almost always redundant: the previous `-1,-2,-4`
were informative in 1.2 %, 2.1 % and 3.5 % of rows. One shared set is a
compromise across neurons whose rates differ by an order of magnitude;
re-derive it for another recording the same way.

### `GMN_TP = match` — **the free run uses the horizon the network was built at**

`TP` is the horizon MDE and CCM score candidates at. `GMN_TP` is the horizon of
the S-map each node uses in Generate/Forecast; `match` (the default) sets it to
`TP`, and an integer sets it independently. With `GMN_TP > 1`, Generate runs as
`GMN_TP` interleaved chains, so each predicted row lands at its own time.

The idea behind a large `TP` was that it lets slow, long-range drivers into the
network. That part is reasonable. Generating at Tp=1 from such a network is not:
a driver that was picked because it predicts the target 20 rows ahead is
not thereby a good one-step predictor, and one-step S-map then fits coefficients
on a relationship the selection never tested. Matching the horizons keeps the
selection criterion and the use of the selected drivers the same. A Tp=20 replay
of job 4799404 against the Tp=1 replay did not finish in its time limit (theta
retune at Tp=20), so this default rests on that argument, not a measurement.
Set `GMN_TP=1` to reproduce the earlier mismatched behaviour.

### `GENERATE_NOISE = local_residual` — **why the free run goes quiet, and the fix**

A deterministic free run iterates the model's *conditional mean*. For a TSLS
node the next value is bimodal — `x + dt` (no spike) or ≈ 0 (spike) — and the
mean, `(x + dt)(1 − p)`, is neither: it ramps too slowly and never resets. In
job 4799404's free run, all 116 simulated nodes produced **zero** resets in 400
steps, ramped at a fraction of the true 0.025/row, some ramped *downward*
(which TSLS cannot do), and had a standard deviation of ~0.3 against ~2.9 in
the recording.

Small isotropic noise on every node — the trick spiking networks with
inhibition use to keep activity alive — does not fix this. A spiking model has
a threshold that turns noise into spikes; S-map has none, so the noise only
jitters the mean path, and every downward wobble breaks TSLS's
"step down only to reset" rule and scores as a spurious event.

`local_residual` instead samples the conditional distribution: at each step it
picks one of the node's E+1 nearest library analogues (Simplex weights) and
adds that analogue's deviation from the neighbourhood mean to the S-map
prediction (`gmn_embedding.local_residual_draw`). Resets then occur with the
frequency the library shows at that state. It runs as one extra free run after
the final one, writes `gmn_generate_output_<id>_stochastic.csv` and
`generate_noise_comparison_<id>.csv`, and changes nothing else. One
realization's pointwise score is expected to be *lower* than the deterministic
run's, since the mean minimises squared error; judge it on reset counts and
variability against the recording, averaged over seeds.

### Start-offset sweep scores — **the winner is biased upwards**

The start sweep picks the offset whose free run best matches the held-out data,
and every score reported afterwards is at that offset, on that same data. In
job 4799404 the 20 offsets, one row apart, scored 0.177–0.697, and the
reported 0.697 was the maximum. `start_sweep_summary_<id>.csv` now records the
mean and spread over offsets; quote the mean.

---

## 8. Diagnostics

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

## 9. `GPU_ENABLED` and the multiprocessing start method

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
