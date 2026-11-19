# Replacing the sawtooth: causal LTI operators for sparse event channels

Branch `exp/spike-conversion-general` (from `no-kEDM-gpu-optimization` @ 353fca0).
Data: `J1620210605_epoch_8_freq_40_Hz.csv` — 47,999 rows at 40 Hz (20 min),
468 single-unit channels (CA1, mPFC, OFC; 0.1–71 Hz), targets `head_position_x/y`.

<!-- STAGE2_SUMMARY -->

## 1. What was changed, and what was not

One selector, `TSLS_MDE_GMN_SPIKE_TRANSFORM ∈ {sawtooth, gamma, lmu, logsig}`,
chooses how each event channel becomes one continuous column. `sawtooth` is the
parent's time-since-last-spike (TSLS) converter, bit-for-bit
(`tests/test_spike_general_pipeline.py::test_transform_sawtooth_and_gamma_exact`).
Every method obeys the same contract:

* **one column per channel** — no PCA, no mixing; the vector-valued operators
  (LMU, log-signature) are reduced to one scalar per channel (§2.4), because a
  graph node in this pipeline is one column (audit, §A);
* **strictly causal** — the value at row *n* depends on events at rows ≤ *n* only;
* **no trainable parameters** — every operator is a fixed linear filter
  (plus, for the signature, a fixed polynomial map), set by one timescale.

Around the operators, the evaluation was changed so that a network can run on
its own and be judged on it (§3):

| Change | Setting | Why |
|---|---|---|
| Common trim | `WARMUP_ROWS=400`, + slowest first event | every method gets the same rows, so the train/validation/test split is identical |
| Observability filter | `N_MIN_EVENTS=100` | a channel needs events in the **training** rows to be modelled at all |
| Takens lag from data | `EMBED_TAU=auto`, `EMBED_TAU_METHOD=ami` | parent hard-coded τ = −1 row (25 ms) |
| Stage 2 lag step = Takens lag | `DELAY_SOURCE=tau_count` | lagged leaves at −τ, −2τ |
| Continuous metric for neurons | `METRIC=correlation` | the parent's `f1` decodes sawtooth resets; meaningless for smooth operators |
| Input-free free runs | `EXTERNAL_INPUT_MODE=self_simulate` | the parent default replays the recording into every zero-driver node |
| Autonomous evaluation | `ROLLOUT_N_STARTS=24` | rollouts, one-step, stability, topology, graph (§3) |

## 2. The operators

All are exact discretisations of a continuous linear system driven by the
event train *c[n]* (counts per 25 ms bin): `z[n] = Φ z[n−1] + B c[n]`,
`Φ = expm(A·dt)`. Exact (impulse-invariant) discretisation keeps the impulse
response of the continuous kernel at every sample, so nothing depends on the
bin width beyond sampling.

### 2.1 M1 — Gamma (Erlang) cascade
Kernel `g_m(t) = t^m e^{−t/τ} / (τ^{m+1} m!)`: a chain of *m*+1 leaky
integrators with a common time constant τ. Peak at *mτ*, mean (m+1)τ.
Options: order *m* ∈ {2, 3}, τ ∈ {100, 250, 500} ms, a multi-scale sum
(`GAMMA_TAUS_MS=100,500`), and log compression `log(1 + κ x / x̄)` with x̄ the
channel's training mean (κ = 100).

**Smoothness, corrected:** g_m and its first m−1 derivatives vanish at t = 0;
the m-th jumps. The response to one event is therefore C^{m−1}, not C^m. C²
needs m ≥ 3, which is why m = 3 is the default.

### 2.2 M2 — Legendre Memory Unit (LMU) state
`θ ż = A z + B u`, with Voelker's (A, B): the state is the projection of the
last θ seconds of input onto D shifted Legendre polynomials. θ ∈ {500, 1000, 2000} ms,
D ∈ {2, 3, 4}.

**Corrected premise:** the LMU impulse response `e^{As/θ}B/θ` is non-zero at
s = 0, so on a raw event train **the state jumps at every event** — it is as
discontinuous as the sawtooth. It becomes C^p only after a Gamma pre-filter of
order p (`LMU_PRESMOOTH_ORDER=p`, τ_p = θ/(4D) by default). Settings with
`_pre2` use p = 2.

### 2.3 M3 — Fading-memory log-signature
The path is X(t) = (t, N(t)), N = cumulative count. The fading signature
`S_n = δ_λ(S_{n−1}) ⊗ exp(ΔX_n)`, λ = e^{−dt/W}, is the signature of the
exponentially reweighted path, so its log is well defined (computed in the
truncated tensor algebra). Depth K ∈ {2, 3}; Lyndon coordinates:

| K | coordinates | meaning |
|---|---|---|
| 1 | N | faded count (an exponential filter) |
| 2 | [t, N] | Lévy area: > 0 when the faded events are recent, < 0 when old |
| 3 | [t,[t,N]], [[t,N],N] | curvature in time; burstiness |

Like the LMU, the raw state jumps at events; `LOGSIG_PRESMOOTH_ORDER=3`
pre-filters the event train with a Gamma cascade (τ_p = W/8).

### 2.4 One scalar per channel
`lmu_norm`: `sqrt(Σ_d m_d² / (2d+1))`, proportional to the L² norm, over the
window, of the input the D Legendre coefficients reconstruct. `rms_norm`: each coordinate divided by its training
RMS, then the Euclidean norm (uncentred, so a quiet window maps to ≈ 0). In Stage I the scalar reduction costs ≈ 0.1 in
per-channel cross-map r against the full vector (`vector_xmap_top5` vs the
scalar column of the same row) — the price of the one-column-per-node graph.

## 3. Evaluation

* **Takens lag** — first minimum of the average mutual information (16 bins;
  a minimum counts only once AMI has risen ≥ 2 % of its range after it, so
  noise wiggles do not count); fallback ACF 1/e; capped at 40 rows (1 s).
  Median over channels, training rows only.
* **One-step** — gmn `Forecast()` over the whole held-out segment (6,667 rows,
  167 s); r and NRMSE per node; skill = 1 − NRMSE / NRMSE(persistence).
* **Free runs** — gmn `Generate()` from 24 evenly spaced held-out starts,
  400 steps (10 s), scored at 2, 5, 10 s: r and speed-weighted r of (x, y)
  against the recording, NRMSE, mean (x, y) path error against **holding the
  seed position**.
* **Attractor stability** — `manifold_ratio`: nearest-neighbour distance of the
  generated population state to the training cloud, over the same for the
  recording at those rows (1 = as on-manifold as the data); `motion_ratio`: sd
  over the last min(h, 2 s), generated / recorded, median over nodes that move
  in the recording (≪ 1 = collapse to a fixed point); `collapsed` = motion_ratio
  < 0.05; `blew_up` = non-finite or > 10× the library's largest |z|.
* **Topology** — Vietoris–Rips persistence (H0 from the MST; H1 by Z/2
  boundary-matrix reduction to the enclosing radius; 80 max-min landmarks) of
  the (x, y) delay embedding and of the population state, all clouds scaled by
  the training data. Wasserstein-1 generated-vs-real. **Noise floor:** the same
  number of equal-length windows drawn from the training rows, vs real — a
  generator is as faithful as the data allows when its distance is at the floor.
* **Graph** — nodes, lagged leaves, edges, density, zero-driver nodes,
  within-region edge fraction, target drivers.

<!-- STAGE1 -->

<!-- STAGE2 -->

<!-- CALIBRATION -->

<!-- LIMITATIONS -->

## A. Audit of the parent pipeline (Step 1)

* **Event preprocessing:** `tsls_data.py::convert_to_tsls()` →
  `cleaned_helper_functions.compute_time_since_previous_spike()`;
  `burn_in_trim()` drops rows until every channel has fired once.
* **Network construction:** `mde.py` (forward selection with a CCM gate),
  `network.py` / `network_parallel.py` (two-stage nested graph; Stage 2 adds
  lagged copies as leaf nodes).
* **Free run:** `generate.py` (gmn `Generate()` / `Forecast()`), `theta.py`
  (per-node S-map θ), `gmn_embedding.py` (each node embeds on its own in-edges
  plus its own present value).
* **Nodes are scalar.** A node is a DataFrame column; there is no multi-column
  node. Hence §2.4.
* **The parent's "free run" is not input-free.** With
  `EXTERNAL_INPUT_MODE=real_data` every zero-driver node, including every
  lagged leaf, replays the recording. `self_simulate` is input-free, but it
  simulates a lagged leaf `X::delay-k` as its own one-dimensional self-map, not
  as the delayed copy of the generated X — a limitation kept here, applied
  equally to every method.
* **Sawtooth-specific evaluation:** neuron nodes were scored by `f1` on reset
  events decoded from the sawtooth, and each neuron's MDE window was grown until
  it held `MIN_PRED_EVENTS` resets. Both now apply only under an event metric
  (`f1`, `gamma`).
* **Fixed Takens lag** τ = −1 row everywhere.

### Corrections to the experiment prompt's premises
1. *"The sawtooth needs t_{k+1}"* — not in this code: TSLS(t) = t − (last
   event ≤ t) is causal. Its defect is the jump at each event, not look-ahead.
2. *"Gamma order m is C^m"* — the response to an event is C^{m−1}.
3. *"The LMU gives smooth coordinates"* — on raw events the LMU state jumps at
   every event; it needs a pre-filter.
4. *"Silent intervals collapse to (0, …, 0)"* — true of every decaying operator
   (Gamma, LMU, fading signature) and **not** of the sawtooth, which ramps
   through a silence. Measured: §4.

### A bug found in the parent's Forecast() (fixed on this branch)
`_generalized_node_forecast` applied the configured lib/pred row range
unchanged to each node's embedding frame, which starts `offset` rows into the
data (offset = the node's deepest edge delay); gmn then labels every node's
predictions with the Time column of an offset-0 source node. A node with a
lagged in-edge therefore predicted rows *k* later than its label. In the
parent's runs (lags of a few rows) this mildly shifted the Forecast()
diagnostic; at this branch's lags (40/80 rows) it sent every driven node's
one-step r to ≈ 0. The predictions themselves were causal — only the labels
were wrong — and `Generate()`, θ tuning, and the network were never affected.
Fixed in 773e99d (`tests/test_forecast_offset_alignment.py`: with the fix
r = 0.9999 at the label, before it r = −0.72 at the label and 0.9999 six rows
away). Jobs that ran the old code had their one-step re-run with
`scripts/experiments/spike_general/rescore_onestep.py`.
