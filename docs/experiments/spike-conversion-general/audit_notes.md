# Step 1 audit (working notes, folded into REPORT.md)

Base: `no-kEDM-gpu-optimization` @ 353fca0.

## Where things are
- Event preprocessing: `scripts/src/mde_gmn/tsls_data.py::convert_to_tsls()` calls
  `cleaned_helper_functions.compute_time_since_previous_spike()`; `burn_in_trim()`
  drops rows until every channel has fired once (TSLS is NaN before a channel's
  first event).
- MDE causal-network construction: `mde.py` (`run_mde_for_target` /
  `run_multi_mde_for_targets` -> `proper_mde` / `multi_proper_mde` forward
  selection with a CCM convergence gate), `network.py` / `network_parallel.py`
  (two-stage nested graph; Stage 2 adds lagged copies as leaf nodes).
- Free run / evaluation: `generate.py` (gmn `Generate()` and `Forecast()`),
  `theta.py` (per-node S-map theta), `gmn_embedding.py` (patches gmn so each
  node embeds on its own in-edges).

## Scalar or vector nodes?
Strictly scalar. A graph node is one DataFrame column. A node's S-map embedding
is built from `(column, delay)` pairs read off its in-edges plus its own
present value (`gmn_embedding._embedding_spec_for`). There is no notion of a
node owning several coordinates; giving a channel D columns would give it D
nodes, which breaks the 1-to-1 channel/node rule. So the vector-valued
operators (LMU, log-signature) are reduced to one scalar per channel.

## How a free run is seeded
`Generate()` starting at row s seeds every node's delay vector from the
recorded rows before s, and its S-map library is every row before s. With
the default `EXTERNAL_INPUT_MODE=real_data`, every zero-driver node (including
every Stage 2 lagged leaf) REPLAYS the recording during the run, so the
parent's "free run" is not input-free. `self_simulate` makes it input-free:
zero-driver nodes are simulated from their own present value. A lagged leaf
`X::delay-k` is then simulated as its own 1-D self-map, not as the delayed copy
of the generated `X` -- a limitation of the parent design that this
experiment keeps (it applies to every method equally).

## The parent's evaluation
- Neuron nodes are scored by `f1` on reset events decoded from the TSLS
  sawtooth, and each neuron node's MDE window is grown until its pred portion
  holds MIN_PRED_EVENTS reset events (`_resolve_target_ranking_window`): both
  only make sense for a sawtooth.
- Every MDE / CCM embedding uses tau = -1 row (25 ms), hard-coded.

## Corrections to the experiment prompt's premises
1. "The sawtooth requires knowing t_{k+1}": not in this code. TSLS(t) =
   t - (last event <= t) is causal. Its defect is the jump at each event
   (C^-1), not lookahead.
2. "Gamma order m is C^m": g_m and its first m-1 derivatives vanish at 0, the
   m-th jumps. The response to one event is C^(m-1); C^2 needs m >= 3.
3. "The LMU gives smooth coordinates": its impulse response e^{As/theta}B/theta
   is nonzero at s=0, so on a raw event train the state jumps at every event.
   It is C^p only after a Gamma pre-filter of order p.
4. "Silent intervals collapse to (0,...,0)": true of every decaying operator
   (Gamma, LMU, fading signature), NOT of the sawtooth, which ramps
   linearly through a silence. Measured in Stage I.
