# ====================================================================
# cleaned_helper_functions.py
# ====================================================================
# Toolkit for empirical dynamic modeling (EDM: Simplex / CCM / MDE) on
# spike-timing data represented in "TSLS" form -- Time Since Last Spike.
#
# WHY TSLS: raw spike trains are sparse binary/count series (mostly 0,
# occasionally 1+ per bin), which is a poor variable for Takens delay
# embedding -- most neighbors in the embedding space collapse onto the
# "no spike" state and Simplex/CCM have almost no structure to exploit.
# TSLS recodes the same information as a continuous, sawtooth-shaped
# signal: it counts *up* linearly between spikes (time elapsed since the
# last one) and *drops* back down close to 0 the instant a new spike
# occurs. That sawtooth has genuine local geometry a delay embedding can
# use, while still being a lossless encoding of the original spike
# times (the drop times ARE the spike times).
#
# The functions in this file fall into four groups (see the section
# banners below):
#   1. Small standalone utilities (ISI computation, binning, plotting).
#   2. TSLS-specific event metrics: turn a TSLS signal back into
#      discrete spike/reset events and score how well two such event
#      sets line up (precision/recall/F1, Kistler's coincidence factor).
#   3. A from-scratch reimplementation of pyEDM's Simplex/CCM engine
#      (KDTree-based nearest-neighbor prediction) that supports the
#      same TSLS-aware metrics as scoring functions, plus
#      "proper_mde"/"multi_proper_mde": ports of the MDE/dimx forward-
#      selection algorithm parameterized on those metrics.
#   4. Multi-target variants of the above (joint embeddings /
#      predictions over several target columns at once).
#
# Functions prefixed with "_" are private implementation details (not
# meant to be called directly from a notebook); the rest form the
# public API: proper_simplex, proper_ccm, proper_mde, proper_embed_dimension,
# proper_predict_interval, top_neurons, multi_simplex, multi_proper_ccm,
# multi_proper_mde, rolling_simplex, plus the TSLS event-metric helpers
# (tsls_drop_signal, infer_near_zero_threshold_from_true_tsls,
# detect_reset_events_from_tsls, adaptive_event_metrics_from_tsls) and
# compute_time_since_previous_spike (the spike-train -> TSLS converter).
# ====================================================================

import numpy as np
import pandas as pd
from pyEDM import Simplex   # only used by the thin _simplex_metric() wrapper below (kept for
                             # backward-compat / cross-checking against the reference pyEDM engine)
import copy


def compute_time_since_previous_spike(spike_times, time_grid):
    """Convert a set of spike times into a TSLS (Time Since Last Spike) series.

    INTUITION
    ---------
    This is *the* spike-train -> TSLS converter. At every point of
    `time_grid` it looks backward to the most recent spike and reports
    how long ago that spike happened. The result is a sawtooth: it
    ramps up linearly while no spike occurs, and resets to (near) 0 the
    instant a new spike happens. Feeding this sawtooth into
    proper_simplex / proper_ccm / proper_mde lets EDM work on a smooth,
    information-preserving encoding of spike timing instead of a sparse
    binary/count spike train.

    INPUT
    -----
    spike_times : 1D array-like of the timestamps at which spikes
        occurred (any units -- seconds, samples, bin index -- as long as
        they're on the same scale as `time_grid`). Does not need to be
        pre-sorted for correctness here (the boolean mask below re-derives
        the comparison at every grid point), but should be sorted for
        this function to run efficiently on long inputs.
    time_grid : 1D array-like of the times at which you want a TSLS
        value reported -- typically the bin centers / sample times of
        whatever other signal you're going to align this against (e.g.
        the `time_bin_center` column of the recording CSV).

    OUTPUT
    ------
    1D numpy array the same length as `time_grid`: `isi_values[i]` is
    `time_grid[i] - (largest spike_time <= time_grid[i])`, i.e. elapsed
    time since the previous spike. Entries are `np.nan` for any grid
    point that falls *before* the very first spike (there's no "previous
    spike" to reference yet).

    HOW TO USE
    ----------
    >>> tsls = compute_time_since_previous_spike(spike_times, df['time_bin_center'].to_numpy())
    >>> df['neuron_0_tsls'] = tsls   # now usable as a proper_simplex/proper_ccm/proper_mde column

    R1 (docs/review-actions-v1.md): vectorized -- the old implementation
    boolean-masked every spike at or before `t`, for every grid point `t`
    (O(N x S), N = len(time_grid), S = len(spike_times)); this version
    uses np.searchsorted (O(N log S) with a sorted `spike_times`, and
    `spike_times` is already assumed sorted per this docstring's own INPUT
    section above) to find each grid point's most recent spike directly.
    Kept as `_compute_time_since_previous_spike_reference`, used only by
    `tests/test_r1_tsls_vectorized.py`'s equivalence check -- see
    docs/review-findings.md's R1 section for the elementwise-identical
    confirmation and the measured speedup.
    """
    spike_times = np.asarray(spike_times)
    time_grid = np.asarray(time_grid)

    # searchsorted(..., side='right') - 1: the index of the last
    # spike_times entry <= t, for every t in time_grid at once. -1 means
    # every spike is strictly after t (before this channel's first spike)
    # -- clip to 0 for the lookup below, then overwrite those positions
    # with NaN, matching the old per-point "no s_prev" branch exactly.
    idx = np.searchsorted(spike_times, time_grid, side='right') - 1
    safe_idx = np.clip(idx, 0, len(spike_times) - 1)
    isi_values = time_grid - spike_times[safe_idx]
    isi_values = np.where(idx < 0, np.nan, isi_values)
    return isi_values


def _compute_time_since_previous_spike_reference(spike_times, time_grid):
    """Pre-R1 reference implementation of compute_time_since_previous_spike()
    -- kept only for tests/test_r1_tsls_vectorized.py's elementwise
    equivalence check against the vectorized version above. Not part of
    the public API; do not call this from pipeline code."""
    isi_values = np.zeros_like(time_grid)
    for i, t in enumerate(time_grid):
        # Boolean-mask every spike at or before the current grid time t;
        # since spike_times is assumed sorted, the *last* entry of that
        # masked slice is the most recent (largest) spike <= t.
        s_prev = spike_times[spike_times <= t]

        if len(s_prev) == 0:
            isi_values[i] = np.nan  # undefined if outside spike range
        else:
            isi_values[i] = t - s_prev[-1]  # ISI between nearest surrounding spikes
    return isi_values


def mean_in_time_grid(timestamps, values, t_start, t_end, bin_width):
    """
    Downsample an irregularly-sampled (timestamp, value) series onto a
    regular time grid by averaging within fixed-width, non-overlapping
    bins -- e.g. collapsing high-rate position tracking down to the
    same bin width as a spike-count / TSLS series so the two can be
    merged into one DataFrame for EDM.

    Parameters:
    - timestamps: 1D array-like, time stamps of measurements.
    - values: 1D array-like, values corresponding to timestamps.
    - t_start: float, start time of the first bin.
    - t_end: float, end time of the last bin (exclusive).
    - bin_width: float, width of each time bin.

    Returns:
    - bin_edges: 1D NumPy array of shape (num_bins + 1,), edges of time bins.
    - bin_means: 1D NumPy array of shape (num_bins,), mean values in each bin (np.nan if bin is empty).
    """
    import numpy as np
    timestamps = np.asarray(timestamps)
    values = np.asarray(values)

    # Define bin edges and initialize output
    bin_edges = np.arange(t_start, t_end + bin_width, bin_width)
    num_bins = len(bin_edges) - 1
    bin_means = np.full(num_bins, np.nan)

    # Assign each timestamp to a bin: (timestamp - t_start) / bin_width tells you
    # how many whole bin-widths past t_start the timestamp is; floor() gives the
    # 0-based bin index it falls into.
    bin_indices = np.floor((timestamps - t_start) / bin_width).astype(int)

    # Process bins
    for i in range(num_bins):
        in_bin = bin_indices == i
        if np.any(in_bin):
            bin_means[i] = np.mean(values[in_bin])
        else:
            # Empty bin (no timestamps fell inside it): forward-fill from the
            # previous bin's mean instead of leaving a NaN gap. Note this reads
            # bin_means[-1] (== NaN, since it's still np.full's default) when i==0
            # and the first bin happens to be empty -- a harmless edge case.
            bin_means[i] = bin_means[i-1]

    return bin_edges, bin_means


def plot_ccm_results(length,c2t,t2c,c2t_std,t2c_std,label_left,label_right,columns,target):
    """Plot a classic two-axis CCM convergence curve: cross-map skill in
    each direction (columns->target and target->columns) vs. library
    size, with the two directions sharing an x-axis but drawn against
    independent y-axes (since the two metrics being plotted -- e.g. one
    correlation, one F1 -- aren't necessarily on the same scale).

    INPUT: `length` = x-axis values (library sizes, despite the axis
    being labeled 'Time (s)' below -- a leftover label from an earlier
    version of this plot). `c2t`/`t2c` = mean score at each library size
    for each direction; `c2t_std`/`t2c_std` = their error bars.
    `label_left`/`label_right` = y-axis labels (typically the metric
    names). `columns`/`target` = variable names, used to build the
    legend text. Called internally by proper_ccm/multi_proper_ccm when
    show_plot=True -- not usually called directly.

    OUTPUT: none (renders the figure with plt.show(); does not return
    or save a Figure object -- if you need to save the plot yourself,
    call plt.savefig() right after invoking the function that triggers
    this, before pyplot's inline backend clears the figure).
    """
    import matplotlib.pyplot as plt
    fig, ax1 = plt.subplots(figsize=(10,5))
    #plt.title(title,fontsize=20);
    # Left axis / black trace: columns -> target cross-map skill vs library size.
    ax1.errorbar(length, c2t, yerr=c2t_std, color='black',alpha=0.7,label=columns+' : '+target);
    ax1.set_ylabel(label_left,fontsize=16,color='black');

    # Right axis (twin of ax1, shares x) / blue trace: target -> columns direction.
    ax2 = ax1.twinx()
    ax2.errorbar(length, t2c, t2c_std, color='blue',alpha=0.6, label=target+' : '+columns);
    ax2.set_ylabel(label_right,fontsize=16,color='blue');

    ax1.set_xlabel('Time (s)',fontsize=16);
    plt.legend(loc='center right',fontsize=12);
    plt.tight_layout()
    plt.show()
    #if plot_name != None:
    #    plt.savefig(figure_dir+plot_name+'.png',dpi=512)


################################################################################################################################################################################
#### Below are functions for new metrics for TSLS data ##################
################################################################################################################################################################################
def tsls_drop_signal(x):
    """
    Positive drop magnitudes in TSLS.

    INTUITION: a TSLS series ramps up between spikes and drops sharply
    back down at each new spike. This returns, at every bin, "how big
    was the *downward* step from the previous bin" -- 0 everywhere the
    series is ramping up (or flat), and a large positive number at each
    reset/spike bin. It's the building block used to both auto-detect
    the neuron's near-zero threshold (infer_near_zero_threshold_from_true_tsls)
    and to locate the reset events themselves (detect_reset_events_from_tsls).

    INPUT: x -- 1D array, a TSLS series (e.g. one neuron's column).
    OUTPUT: 1D array same length as x; dx[i] = max(0, x[i-1] - x[i]).
    """
    dx = np.diff(x, prepend=x[0])
    return np.maximum(-dx, 0.0)

def infer_near_zero_threshold_from_true_tsls(x_true):
    """
    Your current logic:
    Find all bins where true TSLS drops, then take the largest TSLS value
    among those post-drop values. This becomes the neuron-specific near-zero threshold.

    INTUITION: after a spike, TSLS should reset close to 0 -- but "close
    to 0" isn't a fixed number, it depends on the bin width and how
    exactly the TSLS series was constructed (e.g. it might land on 1
    bin-width rather than exactly 0). This function calibrates that
    threshold empirically from the *true* (observed) TSLS series: look
    at every bin where a drop occurred (tsls_drop_signal > 0), and take
    the largest post-drop TSLS value seen among those -- i.e. the
    largest value TSLS was ever observed to reset down to. Any bin at or
    below that value is then treated as "just reset" by
    detect_reset_events_from_tsls.

    INPUT: x_true -- 1D array, the ground-truth/observed TSLS series.
    OUTPUT: float threshold (0.0 if the series never drops at all, e.g.
    a flat/constant input).
    """
    d_true = tsls_drop_signal(x_true)
    mask = d_true > 0
    if not np.any(mask):
        return 0.0
    return float(np.max(x_true[mask]))


def detect_reset_events_from_tsls(x,near_zero_threshold,drop_threshold = 0.0):
    """
    Detect reset events as bins where:
      1) there is a downward step larger than drop_threshold
      2) the post-drop TSLS value is <= near_zero_threshold

    INTUITION: this is the TSLS -> spike-event decoder (the inverse of
    compute_time_since_previous_spike). A bin is flagged as a spike/reset
    event only if the series both (a) just dropped by more than
    drop_threshold from the previous bin, AND (b) landed at or below
    near_zero_threshold -- requiring both avoids false positives from
    small numerical wobbles that happen not to be at a true reset.

    INPUT: x -- 1D TSLS array (observed or predicted). near_zero_threshold
    -- typically from infer_near_zero_threshold_from_true_tsls(x_true).
    drop_threshold -- minimum downward step size to count as a "drop"
    (default 0.0 = any decrease at all counts).

    OUTPUT: 1D int array of bin indices where a reset event was detected
    (usable directly as "predicted/true spike times" in bin-index units
    for adaptive_match_events / _coincidence_factor).
    """
    x = np.asarray(x, dtype=float)
    drops = np.diff(x, prepend=x[0])
    is_drop = drops < -drop_threshold
    near_zero = x <= near_zero_threshold
    events = np.where(is_drop & near_zero)[0]
    return events.astype(int)


def local_event_tolerances(event_times,alpha = 0.25,min_tol = 0,max_tol = 2):
    """
    Adaptive tolerance per true event based on local spacing.

    tau_i = alpha * local spacing, clipped to [min_tol, max_tol]

    local spacing is estimated from neighboring true events:
      - first event: distance to next
      - last event: distance to previous
      - middle events: min(prev gap, next gap)

    INTUITION: when matching predicted spike/reset events to true ones
    (adaptive_match_events, below), a fixed +/-N-bin tolerance window is
    too generous for a densely-firing neuron (tiny inter-spike gaps --
    a fixed window would let a predicted event "match" the wrong true
    spike) and too strict for a sparsely-firing one. This computes a
    tolerance *per true event*, proportional (`alpha`) to how close that
    event's neighbors are, clipped to a sane [min_tol, max_tol] range in
    bins.

    INPUT: event_times -- 1D int array of true event bin indices
    (typically output of detect_reset_events_from_tsls on the ground
    truth). alpha -- fraction of local spacing to allow as tolerance.
    min_tol/max_tol -- hard floor/ceiling on the tolerance, in bins.

    OUTPUT: 1D int array, same length as event_times, one tolerance
    value per event.
    """
    event_times = np.asarray(event_times, dtype=int)
    n = len(event_times)

    if n == 0:
        return np.array([], dtype=int)

    if n == 1:
        return np.array([min_tol], dtype=int)

    tols = np.zeros(n, dtype=int)

    for i in range(n):
        if i == 0:
            local_gap = event_times[i + 1] - event_times[i]
        elif i == n - 1:
            local_gap = event_times[i] - event_times[i - 1]
        else:
            prev_gap = event_times[i] - event_times[i - 1]
            next_gap = event_times[i + 1] - event_times[i]
            local_gap = min(prev_gap, next_gap)

        tol = int(np.floor(alpha * local_gap))
        tol = max(min_tol, tol)
        tol = min(max_tol, tol)
        tols[i] = tol

    return tols


def adaptive_match_events(true_events,pred_events,tolerances):
    """
    One-to-one nearest-neighbor matching with adaptive per-true-event tolerance.

    For each true event, find the nearest unmatched predicted event within:
        [true_event - tolerance, true_event + tolerance]

    Returns a dict with matches and counts.

    INTUITION: standard greedy bipartite matching for spike-timing
    evaluation. Walk the true events in order; for each one, look among
    the *still-unmatched* predicted events for any within its tolerance
    window and grab the nearest one as a match (so no predicted event
    can double-count for two true events). Whatever's left over is
    counted as false negatives (unmatched true events) / false positives
    (unmatched predicted events) -- the classic TP/FP/FN triad used to
    compute precision/recall/F1 downstream.

    INPUT: true_events, pred_events -- 1D int arrays of event bin
    indices (from detect_reset_events_from_tsls on ground truth / on a
    prediction respectively). tolerances -- 1D int array, one tolerance
    per true event (from local_event_tolerances), same length as
    true_events.

    OUTPUT: dict with:
      'matches' -- list of per-match detail dicts (true/pred index & time,
                   tolerance used, absolute and tolerance-normalized error).
      'tp'/'fn'/'fp' -- ints, true positives / false negatives / false positives.
      'used_pred_mask' -- bool array flagging which pred_events got matched.
    """
    true_events = np.asarray(true_events, dtype=int)
    pred_events = np.asarray(pred_events, dtype=int)
    tolerances = np.asarray(tolerances, dtype=int)

    used_pred = np.zeros(len(pred_events), dtype=bool)
    matches = []

    for i, t in enumerate(true_events):
        tol = tolerances[i]

        # Candidate unmatched predicted events within tolerance window
        candidate_idx = np.where(
            (~used_pred) & (np.abs(pred_events - t) <= tol)
        )[0]

        if len(candidate_idx) == 0:
            continue

        # Pick nearest predicted event; tie-break by earliest
        best_j = candidate_idx[np.argmin(np.abs(pred_events[candidate_idx] - t))]
        used_pred[best_j] = True

        matches.append({
            "true_index": i,
            "true_time": int(t),
            "pred_index": int(best_j),
            "pred_time": int(pred_events[best_j]),
            "tol": int(tol),
            "abs_error": int(abs(pred_events[best_j] - t)),
            "norm_error": float(abs(pred_events[best_j] - t) / max(tol, 1)),
        })

    tp = len(matches)
    fn = len(true_events) - tp
    fp = len(pred_events) - tp

    return {
        "matches": matches,
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "used_pred_mask": used_pred,
    }

def adaptive_event_metrics_from_tsls(x_true,x_pred,near_zero_threshold = None,drop_threshold = 0.0,alpha = 0.25,min_tol = 0,max_tol = 2):
    """
    Full pipeline:
      1) infer threshold from x_true if not provided
      2) detect true/pred reset events
      3) compute adaptive per-event tolerances
      4) one-to-one matching
      5) return precision/recall/F1 + timing summaries

    INTUITION: this is the top-level "how good is this predicted TSLS
    series at getting spike *timing* right" scorer -- it's what backs
    the ``'f1'`` option of the ``metric``/``ccmMetric`` arguments
    throughout this file (proper_simplex, proper_ccm, proper_mde, ...).
    Unlike Pearson correlation (which rewards overall shape similarity),
    this decodes both series back to discrete spike/reset events and
    scores event-by-event timing precision/recall -- appropriate when
    what actually matters is "did we get the spikes at (approximately)
    the right times", not just "do the two curves look similar".

    INPUT: x_true -- observed TSLS series (ground truth). x_pred --
    predicted TSLS series (e.g. Simplex/CCM output), same length as
    x_true. near_zero_threshold -- passed through to
    detect_reset_events_from_tsls; if None (default), it's inferred from
    x_true via infer_near_zero_threshold_from_true_tsls. drop_threshold,
    alpha, min_tol, max_tol -- passed through to
    detect_reset_events_from_tsls / local_event_tolerances (see those
    functions).

    OUTPUT: dict with keys 'precision', 'recall', 'f1' (floats, NaN-safe
    -- e.g. precision is NaN if there were zero predicted events). Several
    additional diagnostic keys (raw event indices, matches, timing-error
    stats) are computed internally but commented out of the returned
    dict below to keep the hot path (this runs inside every CCM/MDE
    inner loop) lean -- uncomment as needed for debugging.
    """
    x_true = np.asarray(x_true, dtype=float)
    x_pred = np.asarray(x_pred, dtype=float)

    if near_zero_threshold is None:
        near_zero_threshold = infer_near_zero_threshold_from_true_tsls(x_true)

    true_events = detect_reset_events_from_tsls(
        x_true,
        near_zero_threshold=near_zero_threshold,
        drop_threshold=drop_threshold,
    )

    pred_events = detect_reset_events_from_tsls(
        x_pred,
        near_zero_threshold=near_zero_threshold,
        drop_threshold=drop_threshold,
    )

    tolerances = local_event_tolerances(
        true_events,
        alpha=alpha,
        min_tol=min_tol,
        max_tol=max_tol,
    )

    match_result = adaptive_match_events(true_events, pred_events, tolerances)

    tp = match_result["tp"]
    fn = match_result["fn"]
    fp = match_result["fp"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(precision) and np.isfinite(recall) and (precision + recall) > 0
        else 0
    )

    #abs_errors = np.array([m["abs_error"] for m in match_result["matches"]], dtype=float)
    #norm_errors = np.array([m["norm_error"] for m in match_result["matches"]], dtype=float)

    #mean_abs_error = float(np.mean(abs_errors)) if len(abs_errors) > 0 else np.nan
    #median_abs_error = float(np.median(abs_errors)) if len(abs_errors) > 0 else np.nan
    #mean_norm_error = float(np.mean(norm_errors)) if len(norm_errors) > 0 else np.nan

    # Optional soft timing score: 1 is perfect, 0 is right at tolerance boundary
    #soft_timing_score = (
    #    float(np.mean(1.0 - np.clip(norm_errors, 0.0, 1.0)))
    #    if len(norm_errors) > 0
    #    else np.nan
    #)

    return {
        #"near_zero_threshold": float(near_zero_threshold),
        #"true_events": true_events,
        #"pred_events": pred_events,
        #"tolerances": tolerances,
        #"matches": match_result["matches"],
        #"tp": tp,
        #"fp": fp,
        #"fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        #"mean_abs_error_bins": mean_abs_error,
        #"median_abs_error_bins": median_abs_error,
        #"mean_norm_error": mean_norm_error,
        #"soft_timing_score": soft_timing_score,
    }

################################################################################################################################################################################
#### Below are customized ccm, mde, simplex functions for spike timing (TSLS) data ##################
################################################################################################################################################################################


def _build_delay_embedding(vec, E, tau):
    """Build a Takens time-delay embedding of a single 1D series.

    INTUITION: reconstructs an E-dimensional state-space trajectory from
    one scalar time series by stacking E lagged copies of it,
    `tau` samples apart: row t of the output is
    `[vec[t], vec[t+tau], vec[t+2*tau], ..., vec[t+(E-1)*tau]]`
    (with `tau` conventionally negative, so these are *past* lags).
    This is the standard EDM embedding step underlying every
    non-``embedded=True`` call in this file (proper_simplex, proper_ccm,
    etc. when given raw columns instead of a ready-made embedding).

    INPUT: vec -- 1D array, the raw series to embed. E -- embedding
    dimension (number of lagged coordinates). tau -- lag step in samples
    (negative = look backward).

    OUTPUT: (emb, valid) tuple.
      emb   -- (N, E) array; emb[t, d] = vec[t + d*tau], or NaN where
               that index falls outside [0, N).
      valid -- (N,) bool array, True for rows where every one of the E
               lagged values was in-bounds (no NaN padding) -- i.e. rows
               usable as a genuine E-dimensional embedding point.
    """
    N = len(vec)
    shifts = np.arange(E) * tau
    emb = np.full((N, E), np.nan, dtype=np.float64)
    for d, s in enumerate(shifts):
        # s = d*tau is this coordinate's lag. For tau<0 (the common case),
        # s<=0 so vec[t+s] looks backward: emb[i,d]=vec[i+s], valid for
        # i in [-s, N) -- hence emb[-s:, d] = vec[:N+s]. The s>0 branch is
        # the mirror-image forward-shift case (tau>0).
        if s <= 0:
            emb[-s:, d] = vec[:N + s]
        else:
            emb[:N - s, d] = vec[s:]
    valid = ~np.any(np.isnan(emb), axis=1)
    return emb, valid


def _tp_valid_mask(N, vec, Tp):
    """Bool mask flagging rows t where `vec[t + Tp]` is a valid (in-bounds,
    non-NaN) prediction target -- i.e. rows usable as the "future value to
    predict" side of a Tp-step-ahead forecast.

    INPUT: N -- series length. vec -- 1D array being predicted (the
    target). Tp -- prediction horizon in samples (can be 0 or negative).
    OUTPUT: (N,) bool array; True at row t iff t+Tp is within [0, N) and
    vec[t+Tp] is not NaN.
    """
    shifted = np.arange(N) + Tp
    in_bounds = (shifted >= 0) & (shifted < N)
    clipped = np.clip(shifted, 0, N - 1)  # avoid an out-of-bounds index; masked out by in_bounds anyway
    return in_bounds & ~np.isnan(vec[clipped])


def _crossmap_predict(embed, pred_vals, lib_idx, k, exclusionRadius=0,
                      time_indices=None):
    """One CCM subsample: given L already-drawn random library points from
    the M embedding rows, build a KDTree, predict every M point with k-NN
    simplex weights drawn only from the library. Returns predictions
    (length M) aligned with the actuals `pred_vals`.

    INTUITION: this is the vectorized workhorse behind proper_ccm's
    "cross mapping" step (Sugihara-style CCM). Given an embedding of
    variable A, for every point in it (query point) it finds the k
    nearest neighbors drawn *only* from a random L-point subset of A
    (the "library" -- CCM's convergence test relies on scanning L from
    small to large), then predicts variable B's value at that same time
    index as an exponentially-distance-weighted average of B's values at
    those k neighbors' time indices ("simplex projection" weighting).
    Called twice per (library size x sample) inside proper_ccm: once
    with A=columns embedding / B=target values (the "columns predicts
    target" c2t direction) and once with the roles swapped (t2c).

    INPUT: embed -- (M, E) embedding matrix (all M points, not just the
    library). pred_vals -- (M,) array of the *target*-side values to be
    predicted at each of those M time indices (already Tp-shifted by the
    caller). lib_idx -- sorted 1-D int array of already-drawn library row
    indices into `embed` (proper_ccm draws this itself, sequentially, via
    its own rng -- see that function's own docstring for why the random
    draw and this function's own KDTree work are deliberately split: it
    lets proper_ccm parallelize the latter across a thread pool without
    touching reproducibility, since nothing here consumes any randomness
    of its own). k -- number of nearest library neighbors used per
    prediction (typically E+1, the minimal enclosing simplex).
    exclusionRadius -- neighbors whose *time index* is within this many
    steps of the query point's own time index are excluded (guards
    against autocorrelation letting a point "predict" itself via a
    temporal near-neighbor rather than a true dynamical one).
    time_indices -- (M,) array of the actual row indices each embedding
    row corresponds to (needed to make exclusionRadius meaningful when
    `embed` doesn't itself start at row 0, e.g. after NaN-filtering).

    OUTPUT: (M,) float array of predictions aligned 1:1 with
    `pred_vals`; NaN wherever a query point couldn't get k valid
    (non-excluded) neighbors from the library.
    """
    from scipy.spatial import KDTree
    M = embed.shape[0]
    L = len(lib_idx)
    k = min(k, L - 1)       # can't have more neighbors than library points minus the point itself
    if k < 1:
        return np.full(M, np.nan)

    # Over-fetch neighbors: +1 for the query point possibly being its own
    # nearest neighbor (when it's inside the library), +2*exclusionRadius
    # as headroom for however many additional near-in-time neighbors get
    # excluded below, so there's still enough left to fill k slots.
    k_query = k + 1
    if exclusionRadius > 0:
        k_query += 2 * exclusionRadius
    k_query = min(k_query, L)

    tree = KDTree(embed[lib_idx])
    dist, loc = tree.query(embed, k=k_query)   # query EVERY point (not just the library) against the library tree
    if dist.ndim == 1:                          # k_query==1 collapses the trailing dim; restore it for uniform indexing below
        dist = dist[:, np.newaxis]
        loc = loc[:, np.newaxis]
    glob = lib_idx[loc]   # map KDTree's library-local neighbor indices back to global row indices

    r_M = np.arange(M)[:, np.newaxis]
    is_self = (glob == r_M)   # a query point that's also in the library would otherwise "predict" itself trivially
    if exclusionRadius > 0 and time_indices is not None:
        qt = time_indices[:, np.newaxis]
        nt = time_indices[glob]
        mask_bad = is_self | (np.abs(qt - nt) <= exclusionRadius)
    else:
        mask_bad = is_self

    # For each query row, keep only the first k neighbors (in increasing-distance
    # order, since KDTree.query already returns them sorted) that survive the
    # self/exclusion mask -- this is a vectorized "take first k valid" trick:
    # cumsum over the valid mask gives each valid neighbor's running rank.
    valid = ~mask_bad
    cs = np.cumsum(valid, axis=1)
    first_k = valid & (cs <= k)
    insufficient = cs[:, -1] < k   # rows where even ALL fetched neighbors weren't enough to reach k valid ones
    if insufficient.all():
        return np.full(M, np.nan)
    if insufficient.any():
        first_k[insufficient, :] = False   # drop those rows from the reshape below; handled as NaN at the end

    # Gather exactly k neighbor distances/indices per row (only for "sufficient" rows).
    _, cols = np.where(first_k)
    suf = ~insufficient
    n_suf = int(suf.sum())
    cols = cols.reshape(n_suf, k)
    rows = np.where(suf)[0][:, np.newaxis]
    nn_dist = dist[rows, cols]
    nn_glob = glob[rows, cols]

    # Simplex-projection exponential weighting (Sugihara & May 1990): weight
    # each neighbor by exp(-distance / nearest-neighbor-distance), so the
    # closest neighbor always gets weight 1 and farther ones decay
    # exponentially relative to it; then normalize to sum to 1 per row.
    d_min = nn_dist[:, 0:1]
    d_min_nz = np.where(d_min > 0.0, d_min, 1.0)   # avoid divide-by-zero when the nearest neighbor is an exact duplicate
    w = np.exp(-nn_dist / d_min_nz)
    zero = (d_min == 0.0)
    if zero.any():
        # Degenerate case: nearest neighbor is at distance 0 (exact duplicate
        # embedding point). Put all weight on exact-distance-0 neighbors only,
        # zero out the rest, instead of the (numerically fine but
        # dynamically wrong) exp(-0/1)=1 that d_min_nz's substitution would give them.
        w = np.where(zero, np.where(nn_dist == 0.0, 1.0, 0.0), w)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum > 0.0, w_sum, 1.0)
    w /= w_sum

    # Final prediction: weighted average of pred_vals at the k neighbors' time indices.
    preds = np.full(M, np.nan)
    preds[suf] = np.sum(w * pred_vals[nn_glob], axis=1)
    return preds


def _coincidence_factor(true_events, pred_events, delta, T):
    """Kistler (1997) coincidence factor Γ for two event-time sets.

    Γ = (N_coinc - <N_coinc>) / (0.5 * (N_true + N_pred) * (1 - 2*ν*Δ))

    where ν = N_pred / T is the predicted event rate and <N_coinc>
    = 2*ν*Δ*N_true is the expected coincidence count under a
    homogeneous-Poisson null. Γ = 1 perfect, Γ = 0 chance.

    Uses greedy one-to-one matching (closest pair first) so that each
    true event and each predicted event participates in at most one
    coincidence.  This prevents a single predicted event from satisfying
    multiple true events and keeps Γ ∈ (-∞, 1].

    INTUITION: an alternative to the F1-based event score
    (adaptive_event_metrics_from_tsls) for scoring predicted vs. true
    spike/reset timing -- this is the standard neuroscience "coincidence
    factor" used to compare spike trains, correcting the raw
    coincidence count for how many coincidences you'd expect *by chance*
    from a Poisson process firing at the same rate as the predictions.
    Gamma=1 is a perfect match, Gamma=0 is no better than chance, and it
    can go negative if the prediction actively anti-correlates with the
    true timing. This backs the ``'gamma'`` option of the
    ``metric``/``ccmMetric`` arguments elsewhere in this file.

    INPUT: true_events, pred_events -- 1D arrays of event bin indices
    (e.g. from detect_reset_events_from_tsls). delta -- coincidence
    window: two events within `delta` bins of each other count as
    coincident. T -- total series length (bins), used to estimate the
    predicted event rate nu = N_pred / T.

    OUTPUT: float Gamma (NaN if either event set is empty, T<=0, or the
    chance-correction denominator underflows to ~0, i.e. the predicted
    rate is so high that "chance" would already produce total overlap).
    """
    n_true = len(true_events)
    n_pred = len(pred_events)
    if n_true == 0 or n_pred == 0 or T <= 0:
        return np.nan
    te = np.asarray(true_events, dtype=np.float64)
    pe = np.asarray(pred_events, dtype=np.float64)

    dists = np.abs(te[:, None] - pe[None, :])   # all-pairs |true_time - pred_time| distance matrix
    n_coinc = 0
    used_true = np.zeros(n_true, dtype=bool)
    used_pred = np.zeros(n_pred, dtype=bool)
    flat = np.argsort(dists, axis=None)   # visit every (true, pred) pair in increasing-distance order
    for idx in flat:
        i, j = divmod(int(idx), n_pred)   # unravel the flat sort index back to (true_idx, pred_idx)
        if dists[i, j] > delta:
            break   # remaining pairs are even farther apart -- none can be within delta either
        if used_true[i] or used_pred[j]:
            continue   # greedy: skip pairs where either side is already claimed by a closer match
        n_coinc += 1
        used_true[i] = True
        used_pred[j] = True

    # Kistler et al. 1997 coincidence factor: n_coinc corrected for the
    # coincidences expected from a chance (homogeneous Poisson) process at
    # the same predicted rate nu, normalized so a perfect match gives 1.
    nu = n_pred / float(T)
    two_nu_delta = 2.0 * nu * delta
    norm = 1.0 - two_nu_delta
    if norm <= 1e-9:
        return np.nan
    expected = two_nu_delta * n_true
    gamma = (n_coinc - expected) / (0.5 * (n_true + n_pred) * norm)
    return min(gamma, 1.0)


def _score(obs, preds, kind, gammaDelta=2):
    """Apply the requested metric to aligned obs/preds vectors.

    INTUITION: the single dispatch point every prediction-scoring
    function in this file routes through -- this is what the
    ``metric``/``target_measurement``/``columns_measurement``/``ccmMetric``
    string arguments actually select. Swapping the metric changes what
    "good prediction" means without touching any of the surrounding
    Simplex/CCM/MDE machinery.

    INPUT: obs -- 1D array of observed (true) values. preds -- 1D array
    of predicted values, same length, aligned index-for-index with obs.
    kind -- one of:
      'correlation' -- Pearson correlation coefficient (higher=better;
                       standard EDM rho). Good default for continuous,
                       roughly-linear-in-neighborhood signals.
      'mae'         -- mean absolute error (lower=better). Use when you
                       care about magnitude accuracy, not just co-movement.
      'f1'          -- TSLS event-timing F1 via
                       adaptive_event_metrics_from_tsls (higher=better).
                       Use for spike-timing (TSLS) data where getting the
                       *event times* right matters more than curve shape.
      'gamma'       -- Kistler coincidence factor via _coincidence_factor
                       (higher=better, <=1). Alternative event-timing
                       metric to 'f1', chance-corrected.
    gammaDelta -- coincidence window (bins) forwarded to 'gamma'.

    OUTPUT: float score (NaN if fewer than 3 valid, non-NaN paired
    points exist, or if the metric-specific degenerate case applies,
    e.g. zero-variance obs/preds for 'correlation').
    """
    mask = ~(np.isnan(obs) | np.isnan(preds))
    if mask.sum() < 3:
        return np.nan
    o = obs[mask]
    p = preds[mask]
    if kind == 'correlation':
        if o.std() == 0 or p.std() == 0:
            return np.nan
        return np.corrcoef(o, p)[0, 1]
    if kind == 'mae':
        return float(np.mean(np.abs(o - p)))
    if kind == 'f1':
        return adaptive_event_metrics_from_tsls(
            o, p,
            drop_threshold=0.0, alpha=0.25, min_tol=0, max_tol=2,
        )['f1']
    if kind == 'gamma':
        thr = infer_near_zero_threshold_from_true_tsls(o)
        true_events = detect_reset_events_from_tsls(
            o, near_zero_threshold=thr, drop_threshold=0.0,
        )
        pred_events = detect_reset_events_from_tsls(
            p, near_zero_threshold=thr, drop_threshold=0.0,
        )
        return _coincidence_factor(true_events, pred_events,
                                   delta=gammaDelta, T=len(o))
    raise ValueError(f'Unknown metric: {kind}')


def _parse_libSizes(libSizes, E, N):
    """Mirror pyEDM.CCM libSizes parsing: accept ``'start stop inc'`` /
    ``[start, stop, inc]`` (sequence form) or an explicit list of L's.

    INTUITION: CCM's convergence test needs a *sweep* of library sizes
    (small to large) -- this normalizes the many ways a caller might
    specify that sweep into a plain list of ints, matching pyEDM's own
    flexible ``libSizes`` argument so this file's proper_ccm is a
    drop-in for pyEDM.CCM's argument style.

    INPUT: libSizes -- either a whitespace-separated string "start stop
    inc" (pyEDM string form), a 3-element sequence [start, stop, inc]
    (only expanded into a range when inc < stop; a genuine 3-value
    explicit list would need inc >= stop), or any other list of explicit
    library sizes to use as-is. E -- current embedding dimension (used
    for the E+2 minimum-libSize sanity check). N -- total series length
    (used for the max-libSize sanity check).

    OUTPUT: list[int] of library sizes, validated to be within
    [E+2, N]. Raises ValueError on malformed/out-of-range input.
    """
    if isinstance(libSizes, str):
        libSizes = [int(s) for s in libSizes.split()]
    libSizes = [int(L) for L in libSizes]
    if len(libSizes) == 0:
        raise ValueError('libSizes required.')
    if len(libSizes) == 3:
        start, stop, inc = libSizes
        if inc < stop:
            if inc < 1:
                raise ValueError(f'libSizes increment {inc} invalid.')
            if start > stop:
                raise ValueError(
                    f'libSizes start {start} > stop {stop}.'
                )
            if start < E + 2:
                raise ValueError(
                    f'libSizes start {start} < E+2 ({E + 2}).'
                )
            libSizes = list(range(start, stop + 1, inc))
    if libSizes[-1] > N:
        raise ValueError(
            f'Maximum libSize {libSizes[-1]} exceeds data size {N}.'
        )
    if libSizes[0] < E + 2:
        raise ValueError(
            f'Minimum libSize {libSizes[0]} < E+2 ({E + 2}).'
        )
    return libSizes


def proper_ccm(dataFrame, libSizes, sample, tau, Tp, E,
               target, columns, target_measurement, columns_measurement,
               embedded=False, exclusionRadius=0, seed=None,
               gammaDelta=2,
               show_plot=False, pass_fail=False, pool=None):
    """CCM with random-library subsampling and pluggable metric.

    Parameters
    ----------
    dataFrame : pandas.DataFrame
    libSizes : str | list[int]
        Same semantics as ``pyEDM.CCM``: a whitespace-separated string
        ``"start stop inc"``, a 3-element list ``[start, stop, inc]``
        (interpreted as a range when ``inc < stop``), or an explicit
        list of library sizes.
    sample : int
        Number of random library subsamples drawn at each library size.
    target_measurement, columns_measurement : str
        Metric kind: ``'correlation'``, ``'mae'``, ``'f1'``, or
        ``'gamma'`` (Kistler coincidence factor; see ``gammaDelta``).
    gammaDelta : int
        Coincidence window (bins) used by the ``'gamma'`` metric.
    pool : concurrent.futures.Executor or None
        Optional already-open worker pool (e.g. a ``ThreadPoolExecutor``
        sized by the caller's own outer/inner core budget -- see
        proper_mde's own ``pool`` docstring). When given, the
        ``sample x len(libSizes)`` grid of independent cross-map
        evaluations is dispatched across it via ``pool.map`` instead of
        running sequentially; every random library draw still happens
        first, in-order, on this function's own ``rng`` (see below), so
        a given ``seed`` reproduces bit-identical results either way --
        only wall-clock time changes. ``None`` (default) runs the exact
        same work inline via the builtin ``map``, unchanged from before
        this parameter existed.

    HOW TO USE
    ----------
    >>> c2t, t2c, libSizes = proper_ccm(
    ...     df, libSizes='20 200 20', sample=30, tau=-1, Tp=1, E=3,
    ...     target='neuron_A_tsls', columns='neuron_B_tsls',
    ...     target_measurement='f1', columns_measurement='f1',
    ... )
    `c2t`/`t2c` are (sample, n_libSizes) arrays -- e.g. plot
    `c2t.mean(axis=0)` vs `libSizes` to see the classic CCM convergence
    curve (skill should rise and plateau as library size grows, if a
    genuine dynamical link exists in that direction).

    OUTPUT
    ------
    Returns a 3-tuple `(c2t_measurements, t2c_measurements, libSizes)`
    (or, if `pass_fail=True`, a single bool instead -- see below):
      c2t_measurements : (sample, n_libSizes) array. Row = one random
        subsample; column = one library size. Score of predicting
        `target` FROM `columns`'s embedding (does `columns`'s history
        let you cross-map `target`? -- convergence here is evidence
        `target` causally influences `columns`, the classic CCM
        causality-direction convention).
      t2c_measurements : same shape; the reverse direction (predicting
        `columns` from `target`'s embedding).
      libSizes : the parsed list[int] actually used (echoes back
        `_parse_libSizes`'s output for convenience).
    If `pass_fail=True`, instead returns a single bool: whether the t2c
    direction's mean score shows a convergence trend (rising for
    correlation/f1/gamma, falling for mae) with slope beyond +/-0.01 --
    a quick pass/fail convergence gate used internally by proper_mde's
    CCM step, rather than the full measurement arrays.
    """
    col_name = columns if isinstance(columns, str) else columns[0]
    tgt_name = target  if isinstance(target,  str) else target[0]

    col_vec = np.asarray(dataFrame[col_name], dtype=np.float64)
    tgt_vec = np.asarray(dataFrame[tgt_name], dtype=np.float64)
    N = len(col_vec)

    # embedded=True: `columns`/`target` are themselves interpreted as a
    # ready-made multivariate embedding (e.g. MDE's selected-column set),
    # stacked as-is with no further delay-lagging. embedded=False (the
    # typical single-variable case): build a fresh E-tau delay embedding
    # from the single named column.
    if embedded:
        col_list = [columns] if isinstance(columns, str) else list(columns)
        tgt_list = [target]  if isinstance(target,  str) else list(target)
        emb_col = np.column_stack([
            np.asarray(dataFrame[c], dtype=np.float64) for c in col_list
        ])
        valid_col = ~np.any(np.isnan(emb_col), axis=1)
        emb_tgt = np.column_stack([
            np.asarray(dataFrame[c], dtype=np.float64) for c in tgt_list
        ])
        valid_tgt = ~np.any(np.isnan(emb_tgt), axis=1)
        E_eff = emb_col.shape[1]
    else:
        emb_col, valid_col = _build_delay_embedding(col_vec, E, tau)
        emb_tgt, valid_tgt = _build_delay_embedding(tgt_vec, E, tau)
        E_eff = E

    libSizes = _parse_libSizes(libSizes, E_eff, N)

    # Forward direction (c2t): rows usable as query points are those whose
    # `columns` embedding is complete AND whose target(t+Tp) is a valid
    # prediction target. Reverse (t2c) is the mirror image with roles swapped.
    valid_fwd = (valid_col
                 & _tp_valid_mask(N, tgt_vec, Tp)
                 & ~np.isnan(col_vec))
    valid_rev = (valid_tgt
                 & _tp_valid_mask(N, col_vec, Tp)
                 & ~np.isnan(tgt_vec))

    idx_fwd = np.where(valid_fwd)[0]
    idx_rev = np.where(valid_rev)[0]

    embed_fwd = np.ascontiguousarray(emb_col[idx_fwd])
    embed_rev = np.ascontiguousarray(emb_tgt[idx_rev])
    obs_fwd = tgt_vec[idx_fwd + Tp]   # what `columns` is trying to recover
    obs_rev = col_vec[idx_rev + Tp]   # what `target`  is trying to recover

    k = E_eff + 1   # minimal enclosing simplex in E dimensions needs E+1 vertices
    rng = np.random.default_rng(seed)

    n_L = len(libSizes)
    c2t_measurements = np.full((sample, n_L), np.nan)
    t2c_measurements = np.full((sample, n_L), np.nan)

    M_fwd = embed_fwd.shape[0]
    M_rev = embed_rev.shape[0]

    # Draw every (sample, libSize) cell's random library SEQUENTIALLY,
    # in the same (t, i, fwd-then-rev) order the old inline loop consumed
    # `rng` in -- this is what keeps a given `seed` reproducible
    # bit-for-bit regardless of `pool` below. Only the draw is sequential;
    # the expensive part each cell then does (KDTree build + query in
    # _crossmap_predict, scipy/numpy -- releases the GIL for the bulk of
    # it) is what actually gets split across `pool` when one is given.
    tasks = []
    for t in range(sample):
        for i, L in enumerate(libSizes):
            lib_idx_fwd = rng.choice(M_fwd, size=min(L, M_fwd), replace=False)
            lib_idx_fwd.sort()
            lib_idx_rev = rng.choice(M_rev, size=min(L, M_rev), replace=False)
            lib_idx_rev.sort()
            tasks.append((t, i, lib_idx_fwd, lib_idx_rev))

    def _run_one_cell(task):
        t, i, lib_idx_fwd, lib_idx_rev = task
        preds_fwd = _crossmap_predict(
            embed_fwd, obs_fwd, lib_idx_fwd, k,
            exclusionRadius=exclusionRadius, time_indices=idx_fwd,
        )
        c2t = _score(obs_fwd, preds_fwd, target_measurement, gammaDelta=gammaDelta)

        preds_rev = _crossmap_predict(
            embed_rev, obs_rev, lib_idx_rev, k,
            exclusionRadius=exclusionRadius, time_indices=idx_rev,
        )
        t2c = _score(obs_rev, preds_rev, columns_measurement, gammaDelta=gammaDelta)
        return t, i, c2t, t2c

    mapper = pool.map if pool is not None else map
    for t, i, c2t, t2c in mapper(_run_one_cell, tasks):
        c2t_measurements[t, i] = c2t
        t2c_measurements[t, i] = t2c

    if pass_fail:
        # Quick convergence verdict instead of the full measurement arrays:
        # fit a line through mean t2c score vs. libSize index and check both
        # that the endpoint-to-endpoint change and the fitted slope agree in
        # the "improving" direction (sign depends on whether higher or lower
        # is better for the chosen metric).
        from sklearn.linear_model import LinearRegression
        mean_t2c = np.nanmean(t2c_measurements, axis=0)
        model = LinearRegression().fit(
            np.arange(n_L).reshape(-1, 1), mean_t2c
        )
        slope = model.coef_[0]
        if columns_measurement in ('correlation', 'f1', 'gamma'):
            return bool((mean_t2c[-1] - mean_t2c[0]) > 0 and slope > 0.01)
        if columns_measurement == 'mae':
            return bool((mean_t2c[-1] - mean_t2c[0]) < 0 and slope < -0.01)

    if show_plot:
        mean_c2t = np.nanmean(c2t_measurements, axis=0)
        mean_t2c = np.nanmean(t2c_measurements, axis=0)
        std_c2t = np.nanstd(c2t_measurements, axis=0) / np.sqrt(n_L)
        std_t2c = np.nanstd(t2c_measurements, axis=0) / np.sqrt(n_L)
        plot_ccm_results(libSizes, mean_c2t, mean_t2c, std_c2t, std_t2c,
                         target_measurement, columns_measurement,
                         col_name, tgt_name)

    return c2t_measurements, t2c_measurements, libSizes

def _metric_direction(metric):
    """+1 if higher score = better, -1 if lower = better.

    Small lookup used everywhere a "best candidate" ranking or a
    threshold comparison needs to be sign-aware across metrics (e.g.
    sorting descending for correlation/f1/gamma but ascending for mae).
    Raises ValueError on an unrecognized metric name -- acts as the
    input-validation checkpoint for the `metric` string across this file.
    """
    if metric in ('correlation', 'f1', 'gamma'):
        return +1
    if metric == 'mae':
        return -1
    raise ValueError(f'Unknown metric: {metric}')


def _simplex_metric(data, columns, target, lib, pred, tau, Tp, E,
                    embedded, metric, gammaDelta=2, noTime=True):
    """Run a single Simplex(target | columns) over (lib, pred) and score
    Predictions vs Observations with `_score`. NaN-safe: returns NaN if
    Simplex throws or the prediction series is unusable.

    INTUITION: thin bridge between this file's metrics and the actual
    reference pyEDM.Simplex engine (rather than this file's own
    KDTree-based re-implementation used by proper_simplex/proper_ccm).
    Used internally by proper_mde/multi_proper_mde's *ranking* stage
    (embedded=True cross-map of "[candidate]+selected -> target") to
    stay a faithful line-by-line port of the reference MDE/dimx
    algorithm, which itself calls pyEDM.Simplex directly.

    INPUT: same meaning as pyEDM.Simplex's own arguments (dataFrame,
    columns, target, lib, pred, tau, Tp, E, embedded, noTime), plus
    `metric`/`gammaDelta` selecting how the resulting Observations vs
    Predictions get scored (via `_score`).

    OUTPUT: float score, or NaN if pyEDM.Simplex itself raised (e.g. a
    degenerate/too-short lib or NaN-only column).
    """
    try:
        df = Simplex(
            dataFrame=data, lib=lib, pred=pred, tau=tau, Tp=Tp,
            target=target, columns=columns, E=E,
            embedded=embedded, showPlot=False, noTime=noTime,
        )
    except Exception:
        return np.nan
    obs = df['Observations'].to_numpy(dtype=np.float64)
    prd = df['Predictions'].to_numpy(dtype=np.float64)
    return _score(obs, prd, metric, gammaDelta=gammaDelta)


def proper_mde(
    dataFrame, target, D=3,
    *,
    columns=None,
    removeColumns=(),
    removeTime=False,
    lib=None, pred=None,
    Tp=1, tau=-1, exclusionRadius=0,
    sample=20,
    pLibSizes=(10, 15, 85, 90),
    libSizes=None,
    noCCM=False,
    ccmSlope=0.01,
    ccmSeed=None,
    E=0,
    crossMapRhoMin=0.5,
    embedDimRhoMin=0.5,
    maxE=15,
    firstEMax=False,
    metric='correlation',
    ccmMetric='correlation',
    gammaDelta=2,
    noTime=True,
    verbose=False,
    pool=None,
    column_delay=None,
):
    """Forward-stepwise Manifold Dimensional Expansion.

    A line-by-line port of ``MDE/dimx/Run.py``. The only deliberate
    deviation from the reference algorithm is the ``metric`` argument,
    which lets you swap Pearson rho for a TSPS-appropriate event score
    (``'f1'`` or ``'gamma'``) or for an error metric (``'mae'``). The
    metric is applied identically at every place the reference uses
    ``rho``: the ``CrossMapColumns`` ranking step, the
    ``crossMapRhoMin`` / ``embedDimRhoMin`` thresholds, and the
    convergence-slope criterion in the CCM gate.

    Algorithm (matches ``Run.py`` step by step)
    -------------------------------------------
    * ``libSizes = [int(N * p/100) for p in pLibSizes]`` and the
      normalised ``libSizesVec`` for the slope fit.
    * For each ``d`` in ``1..D``:
        1. Candidates = ``columns - selected``. Each candidate's
           multivariate column list is ``[candidate] + selected``.
        2. Rank: Simplex with ``embedded=True`` (the column list IS
           the embedding, one coord per neuron), score with
           ``metric``, sort best-first, drop entries past
           ``crossMapRhoMin``. If none survive: ``continue`` to next
           ``d`` (matches Run.py).
        3. If ``noCCM``: take the top-ranked candidate and continue.
        4. Otherwise, for each candidate in rank order:
            (a) Resolve ``E``: if the kwarg ``E > 0`` use it; else
                ``EmbedDimension`` on the single new column and take
                global argmax (or first local max if ``firstEMax``).
                Reject if ``maxRhoEDim < embedDimRhoMin``.
            (b) Run ``proper_ccm`` on ``target | newColumn`` with
                ``libSizes`` from step 0, ``sample`` subsamples, the
                resolved ``E``, and the chosen metric. Linear-fit the
                metric vs normalised ``libSizesVec``; accept if the
                slope passes ``ccmSlope`` (sign-aware: positive for
                ``correlation``/``f1``/``gamma``, negative for
                ``mae``).
            (c) First passer wins; ``break``.
           If no candidate passes: ``break`` the outer loop (matches
           Run.py's terminate-on-CCM-failure behaviour).

    Parameters
    ----------
    dataFrame : pandas.DataFrame
    target : str
    D : int
        Maximum dimension (``args.D``).
    columns : list[str] or None
        Candidate pool. ``None`` (default) ⇒ all DataFrame columns
        minus ``removeColumns`` and ``target``.
    removeColumns : iterable[str]
    removeTime : bool
        If True, drop the first column of ``dataFrame`` before
        anything else (matches ``MDE/dimx/MDE.py``'s ``removeTime``).
        Use when the input has a leading time column.
    lib, pred : list[int]
        1-offset ``[start, stop]``. Defaults: ``[1, N/2]`` and
        ``[N/2+1, N]``.
    Tp, tau, exclusionRadius, sample : passed to ``proper_ccm``.
    pLibSizes : iterable[float]
        Percentiles of ``N`` for the CCM library sweep. Ignored when
        ``libSizes`` is given.
    libSizes : iterable[int] or None
        Explicit, absolute row counts for the CCM library sweep --
        bypasses ``pLibSizes``'s percentage-of-``N`` computation entirely
        when given. ``None`` (default): use ``pLibSizes`` as before.
    noCCM : bool
        Skip the CCM convergence gate.
    ccmSlope : float
        Convergence-slope threshold.
    ccmSeed : int or None
        Seed passed to ``proper_ccm``.
    E : int
        Static E for all CCM. ``0`` (default) ⇒ per-candidate, per-dimension
        auto-select via ``proper_embed_dimension`` -- a metric-aware rho-vs-E
        sweep (works with any of this file's metrics, unlike pyEDM's own
        Pearson-only ``EmbedDimension``), scored with `metric` (the same one
        the ranking stage uses, since both predict `target`).
    crossMapRhoMin, embedDimRhoMin : float
        Thresholds; sign is auto-flipped for ``mae``.
    maxE, firstEMax : int, bool
        Passed to ``proper_embed_dimension`` when ``E<=0``.
    metric : {'correlation', 'mae', 'f1', 'gamma'}
        Used for the ranking stage (Simplex of ``[candidate]+selected
        -> target``, scoring predictions of ``target``). Pick this to
        match the nature of ``target`` -- e.g. ``'correlation'`` or
        ``'mae'`` for a continuous behaviour signal.
    ccmMetric : {'correlation', 'mae', 'f1', 'gamma'} or None
        Used for the CCM convergence gate, which scores predictions of
        the candidate neuron (``newColumn``), so pick it to match the
        nature of the candidate columns -- e.g. ``'gamma'`` or
        ``'f1'`` for TSPS neuron series. ``None`` (default) falls back
        to ``metric``, reproducing the reference MDE's single-metric
        behaviour.
    gammaDelta : int
    noTime : bool
        Passed to ``Simplex`` / ``EmbedDimension``.
    pool : concurrent.futures.Executor or None
        Optional already-open worker pool (a ``ThreadPoolExecutor`` is
        the intended case -- every hot loop this reaches into,
        ``_simplex_metric``'s ``pyEDM.Simplex`` call and
        ``_crossmap_predict``'s KDTree build/query, is a compiled
        extension that releases the GIL for the bulk of its own work,
        so threads get real parallelism here without a second
        multiprocessing pool nested inside whatever outer
        ``ProcessPoolExecutor`` is already running this node's whole
        ``proper_mde`` call). When given, this function's own ranking
        loop (one ``_simplex_metric`` call per remaining candidate, every
        dimension) and every ``proper_embed_dimension``/``proper_ccm``
        call in the CCM gate below use it instead of running that
        candidate's own work inline. ``None`` (default): everything runs
        exactly as before this parameter existed. Caller is responsible
        for sizing and the pool's lifetime -- see ``run_mde_for_target``
        in tsls-mde-gmn.py for the intended one-pool-per-node-call
        pattern (opened once for this whole call, not once per candidate).
    column_delay : dict[str, int] or None
        R21 (docs/review-actions-v1.md): dynamic per-dimension exclusion
        radius. This function has no built-in notion of "delay" attached
        to a column name -- a candidate here is just a string, and the
        ranking stage's ``embedded=True`` Simplex call uses each selected
        column exactly as given, with no internal per-column lagging.
        Callers whose own column-naming convention *does* encode a delay
        (e.g. tsls-mde-gmn.py's ``::delay``-suffixed Stage 2 candidates)
        can pass a ``{column_name: delay}`` lookup here (any delay's sign
        doesn't matter -- only its magnitude is used); columns not in the
        mapping are treated as delay 0. At each dimension ``d``, before
        scoring that dimension's own remaining candidates (step 4a/4b
        below), the exclusion radius actually used is
        ``max(exclusionRadius, max(abs(column_delay.get(c, 0)) for c in
        MDEcolumns))`` -- i.e. never below the static ``exclusionRadius``
        floor, and never below the deepest lag genuinely present among
        the columns *already selected* for this node (the columns the
        new candidate's own state-space neighbors will be found among).
        ``None`` (default): every dimension uses the static
        ``exclusionRadius`` unchanged -- an exact no-op, so no existing
        caller is affected unless it opts in. Does not affect the ranking
        stage (step 2 above), which has never taken an ``exclusionRadius``
        argument at all (a separate, pre-existing limitation this task
        does not address -- see docs/review-findings.md's R21 section).

    Returns
    -------
    dict with keys
        ``MDEcolumns``  list[str] of selected variables, in order
        ``MDErho``      ndarray of the ranking-stage scores at each step
        ``EDim``        dict ``'{newColumn}:{target}' -> E``
        ``rhoD``        dict ``d -> list[(score, [columns])]``
        ``libSizes``    list[int] used in the CCM gate

    INTUITION (plain-language)
    ---------------------------
    Forward feature selection for building a multivariate embedding that
    predicts (and, via the CCM gate, causally influences) `target`.  At
    each dimension d = 1..D it tries adding each remaining candidate
    column on top of the columns already picked, keeps only candidates
    whose combined embedding predicts `target` well enough
    (`crossMapRhoMin`), and -- unless `noCCM` -- further requires the
    *best* surviving candidate's own univariate CCM cross-map to `target`
    to show real convergence as library size grows (not just a good
    single-library-size score, which could be a coincidence). The first
    candidate to pass both gates is appended to the selection and the
    next dimension starts. If nothing passes a dimension's CCM gate, the
    whole search stops early (that's `Run.py`'s original
    terminate-on-CCM-failure behavior).

    HOW TO USE
    ----------
    >>> out = proper_mde(df, target='neuron_A_tsls',
    ...                   columns=['neuron_B_tsls', 'neuron_C_tsls', ...],
    ...                   D=3, tau=-1, Tp=1, E=3, metric='f1', ccmMetric='f1')
    >>> out['MDEcolumns']   # e.g. ['neuron_C_tsls', 'neuron_B_tsls'] -- the selected drivers, best-first
    >>> out['MDErho']       # ranking-stage score at each accepted step
    Pass an explicit ``E > 0`` whenever ``metric`` isn't ``'correlation'``
    (pyEDM's ``EmbedDimension``, used to auto-pick E, only understands
    Pearson rho).
    """
    from scipy.signal import argrelextrema
    from sklearn.linear_model import LinearRegression

    # Match MDE/dimx/MDE.py:263-265 -- drop the first (time) column.
    if removeTime:
        dataFrame = dataFrame.drop(columns=dataFrame.columns[0])

    N = dataFrame.shape[0]
    if lib  is None: lib  = [1, N // 2]
    if pred is None: pred = [N // 2 + 1, N]

    if ccmMetric is None:
        ccmMetric = metric

    direction     = _metric_direction(metric)      # ranking + thresholds
    direction_ccm = _metric_direction(ccmMetric)   # CCM slope sign
    # Sign-aware ">= threshold" check: for metrics where higher is better
    # (direction=+1) this is a plain x >= t; for mae (direction=-1) the
    # multiplication flips it into x <= t, so callers below never need to
    # branch on which metric is active.
    sign_better = lambda x, t: (direction * x) >= (direction * t)

    # CCM library-size sweep: `libSizes` (explicit absolute row counts),
    # when given, bypasses the `pLibSizes`-as-percentage-of-N computation
    # entirely -- decouples the CCM gate's own convergence-curve library
    # sizing from `dataFrame`'s own row count (which the ranking stage/
    # E-auto-select's own lib/pred window may need to be much larger than,
    # for reasons that have nothing to do with library-size calibration --
    # see docs/review-findings.md's window-sizing section). `None`
    # (default): exactly the old behavior, percentiles of N.
    # libSizesVec is the same sweep normalized to [0,1] (last value = 1)
    # purely for fitting a convergence slope on a consistent x-scale
    # regardless of N.
    libSizes = list(libSizes) if libSizes is not None else [int(N * p / 100.0) for p in pLibSizes]
    libSizesVec = np.array(libSizes, dtype=float).reshape(-1, 1)
    libSizesVec = libSizesVec / libSizesVec[-1]

    remove_set = set(removeColumns) | {target}
    if columns is None:
        dataColumns = [c for c in dataFrame.columns if c not in remove_set]
    else:
        dataColumns = [c for c in columns if c not in remove_set]

    MDEcolumns = []
    MDErho     = []
    EDim       = {}
    rhoD_log   = {}

    for d in range(1, D + 1):
        # ---- Step 1: candidate column lists [candidate] + MDEcolumns
        # Each candidate is tried as the NEW leading column, prepended to
        # whatever's already been selected -- e.g. at d=3 with
        # MDEcolumns=['B','A'], candidate 'C' is tried as ['C','B','A'].
        candidates = [c for c in dataColumns if c not in MDEcolumns]
        column_lists = [[c] + MDEcolumns for c in candidates]

        if verbose:
            print(f'\n{d}-D Cross Map  target={target!r}  '
                  f'{len(column_lists)} candidates')

        # ---- Step 2: rank with embedded=True Simplex
        # embedded=True treats each column_list AS the embedding directly
        # (one raw coordinate per selected neuron, no further delay-lagging)
        # -- this is what makes the ranking "cumulative": a d-D score
        # reflects the skill of the d columns selected so far together,
        # not any one column's standalone skill.
        def _rank_one(cl):
            s = _simplex_metric(
                dataFrame, cl, target, lib, pred,
                tau=tau, Tp=Tp, E=0,
                embedded=True,
                metric=metric, gammaDelta=gammaDelta,
                noTime=noTime,
            )
            return cl, s

        # Every remaining candidate's ranking score is independent of
        # every other's (same dataFrame/lib/pred, read-only) -- see this
        # function's own `pool` docstring for why a ThreadPoolExecutor
        # here is both safe and effective (pyEDM.Simplex releases the GIL).
        mapper = pool.map if pool is not None else map
        rhoD = {}
        for cl, s in mapper(_rank_one, column_lists):
            key = ','.join(cl) + ':' + target
            rhoD[key] = (s, cl)

        # Sort candidates best-first (NaN scores sorted to the very end
        # regardless of metric direction, via the -direction*inf sentinel),
        # then drop anything that's NaN or fails the crossMapRhoMin gate.
        L_rhoD = sorted(
            rhoD.values(),
            key=lambda x: -direction * (
                x[0] if not (isinstance(x[0], float) and np.isnan(x[0]))
                else -direction * np.inf
            ),
        )
        L_rhoD = [
            (s, cl) for (s, cl) in L_rhoD
            if not (isinstance(s, float) and np.isnan(s))
            and sign_better(s, crossMapRhoMin)
        ]

        if len(L_rhoD) == 0:
            if verbose:
                print(f'  {d}-D failed to pass crossMapRhoMin.')
            continue  # matches Run.py: try next d

        rhoD_log[d] = L_rhoD

        if verbose:
            print(f'  L_rhoD[:3] {[(round(s,4), cl) for s, cl in L_rhoD[:3]]}')

        # ---- Step 3: noCCM shortcut
        if noCCM:
            s_top, cl_top = L_rhoD[0]
            newColumn = cl_top[0]
            MDEcolumns.append(newColumn)
            MDErho.append(s_top)
            if verbose:
                print(f'  {d}-D added {newColumn!r} (noCCM, '
                      f'{metric}={s_top:.4f})')
            continue

        # ---- Step 4: CCM convergence gate
        # Walk candidates best-ranked-first; the FIRST one whose CCM slope
        # passes wins (not necessarily the single best-ranked one overall --
        # a higher-ranked candidate that fails CCM is skipped in favor of
        # the next-best one that passes).

        # R21 (docs/review-actions-v1.md): dynamic exclusion radius -- the
        # Theiler window used below must cover the deepest lag genuinely
        # present among the columns *already selected* (MDEcolumns, as of
        # the start of this dimension), not just the static floor. See
        # this function's own `column_delay` docstring for why this can't
        # be derived from MDEcolumns's own column-name strings without a
        # caller-supplied lookup.
        if column_delay:
            deepest_selected_lag = max(
                (abs(column_delay.get(c, 0)) for c in MDEcolumns), default=0,
            )
        else:
            deepest_selected_lag = 0
        exclusionRadius_d = max(exclusionRadius, deepest_selected_lag)
        if verbose:
            print(f'  {d}-D exclusionRadius={exclusionRadius_d} '
                  f'(floor={exclusionRadius}, deepest_selected_lag={deepest_selected_lag})')

        maxCol_i = None
        for col_i in range(len(L_rhoD)):
            s_rank, cl = L_rhoD[col_i]
            newColumn = cl[0]

            # (a) embedding dimension: either use the caller-supplied static E,
            # or auto-pick it per-candidate via proper_embed_dimension's own
            # univariate rho-vs-E sweep -- a drop-in, metric-aware replacement
            # for pyEDM.EmbedDimension (which is Pearson-rho only and can't
            # score 'f1'/'gamma'/'mae'), scored with the SAME `metric` the
            # ranking stage above uses (this call also predicts `target`, from
            # newColumn's own embedding -- same quantity, same metric).
            if E > 0:
                maxEDim    = E
                maxRhoEDim = s_rank
            else:
                EDimDF = proper_embed_dimension(
                    dataFrame, columns=newColumn, target=target,
                    maxE=maxE, lib=lib, pred=pred,
                    Tp=Tp, tau=tau, exclusionRadius=exclusionRadius_d,
                    metric=metric, gammaDelta=gammaDelta,
                    noTime=noTime, showPlot=False, pool=pool,
                )
                # Column is named after `metric` itself (proper_embed_dimension
                # returns one column per requested metric), not the fixed
                # 'rho' pyEDM.EmbedDimension always used -- and argmax/
                # argrelextrema(np.greater) both assume higher-is-better, so
                # both need the same sign-flip `sign_better`/`direction` above
                # already applies for 'mae' (lower-is-better).
                edim_scores = EDimDF[metric].round(4).to_numpy()
                if firstEMax:
                    extrema_fn = np.greater if direction > 0 else np.less
                    iMaxArr = argrelextrema(edim_scores, extrema_fn)[0]
                    iMax = int(iMaxArr[0]) if len(iMaxArr) \
                        else int(np.nanargmax(direction * edim_scores))
                else:
                    iMax = int(np.nanargmax(direction * edim_scores))
                maxRhoEDim = float(edim_scores[iMax])
                maxEDim    = int(EDimDF['E'].iloc[iMax])

            if verbose:
                print(f'    EDim {newColumn} E={maxEDim} '
                      f'rho={maxRhoEDim:.4f}')

            if not sign_better(maxRhoEDim, embedDimRhoMin):
                continue

            EDim[f'{newColumn}:{target}'] = maxEDim

            # (b) CCM convergence
            try:
                _, t2c, _ = proper_ccm(
                    dataFrame,
                    libSizes=libSizes,
                    sample=sample,
                    tau=tau, Tp=Tp, E=maxEDim,
                    target=target, columns=newColumn,
                    target_measurement=ccmMetric,
                    columns_measurement=ccmMetric,
                    embedded=False,
                    exclusionRadius=exclusionRadius_d,
                    seed=ccmSeed,
                    gammaDelta=gammaDelta,
                    pool=pool,
                )
            except Exception as exc:
                if verbose:
                    print(f'    {newColumn}: proper_ccm raised {exc}')
                continue

            # Average the t2c direction (target's manifold cross-mapping to
            # newColumn) over samples, then fit its slope vs. normalized
            # libSize -- a positive (correlation/f1/gamma) or negative (mae)
            # slope beyond ccmSlope is the CCM convergence signature: skill
            # keeps improving as more library data is added, which is what
            # distinguishes a genuine dynamical link from a coincidental
            # single-library-size correlation.
            ccmVals = np.nan_to_num(np.nanmean(t2c, axis=0))
            slope = float(
                LinearRegression().fit(libSizesVec, ccmVals).coef_[0]
            )
            slope = round(slope, 5)

            if verbose:
                diag = ''
                if ccmMetric == 'f1':
                    # Diagnostic only (no filtering, no new control flow) --
                    # a candidate starved of its own TSLS reset events
                    # within this frame and a genuinely non-causal candidate
                    # both fail this gate identically (see
                    # docs/review-findings.md's CCM-gate candidate-side
                    # event-sufficiency section); this makes the two
                    # separable after the fact from the log alone.
                    cand_vals = np.asarray(dataFrame[newColumn], dtype=np.float64)
                    cand_events = len(detect_reset_events_from_tsls(
                        cand_vals,
                        near_zero_threshold=infer_near_zero_threshold_from_true_tsls(cand_vals),
                        drop_threshold=0.0,
                    ))
                    diag = f' candidate_events_in_frame={cand_events}'
                print(f'    {target}:{newColumn} slope {slope}{diag}')

            if (direction_ccm * slope) > ccmSlope:
                maxCol_i = col_i
                break  # this candidate is good

        if maxCol_i is not None:
            s_chosen, cl_chosen = L_rhoD[maxCol_i]
            newColumn = cl_chosen[0]
            MDEcolumns.append(newColumn)
            MDErho.append(s_chosen)
            if verbose:
                print(f'  {d}-D added {newColumn!r}  '
                      f'({metric}={s_chosen:.4f})')
        else:
            if verbose:
                print(f'  {d}-D CCM failed to find columns.')
            break  # matches Run.py: terminate on CCM failure

    return {
        'MDEcolumns': MDEcolumns,
        'MDErho':     np.array(MDErho),
        'EDim':       EDim,
        'rhoD':       rhoD_log,
        'libSizes':   libSizes,
    }

# ====================================================================
# Multi-target variants
# ====================================================================

def _build_multi_target_pool(dataFrame, targets, pool_depth, tau):
    """Stack delay embeddings of multiple target columns into a single
    pool. Each target contributes ``pool_depth`` delay columns; the
    pool has shape ``(N, len(targets) * pool_depth)`` with column labels
    ``(target_name, lag_index)``.

    INTUITION: building block for the multi-target functions below
    (multi_proper_ccm, multi_simplex, multi_proper_mde) -- when you want
    to jointly cross-map *several* target variables at once (e.g.
    predicting/being-predicted-by a whole population, not one neuron),
    you first need a single combined embedding that represents "the
    joint state of all targets together". This builds that by
    delay-embedding each target separately (via _build_delay_embedding)
    and horizontally stacking the results, lag 0 of every target coming
    before any target's lag 1, etc. is NOT how it's ordered -- see
    OUTPUT below for the actual column order.

    INPUT: dataFrame -- source data. targets -- list[str] of target
    column names. pool_depth -- delay-embedding depth applied to EACH
    target (same E for every target). tau -- shared embedding lag.

    OUTPUT: (pool, labels, valid) tuple.
      pool   -- (N, len(targets)*pool_depth) array. Columns are grouped
                per-target (all of target[0]'s pool_depth lags first,
                then target[1]'s, ...), lag 0 first within each group.
      labels -- list of (target_name, lag_index) tuples, one per pool
                column, in the same order as pool's columns.
      valid  -- (N,) bool array, AND of every individual target's
                embedding validity mask (True only where ALL targets'
                embeddings are simultaneously complete).
    """
    embeddings = []
    labels = []
    valid = None
    for name in targets:
        vec = np.asarray(dataFrame[name], dtype=np.float64)
        emb, v = _build_delay_embedding(vec, pool_depth, tau)
        embeddings.append(emb)
        labels.extend([(name, d) for d in range(pool_depth)])
        valid = v if valid is None else (valid & v)
    pool = np.hstack(embeddings)
    return pool, labels, valid


def multi_proper_ccm(
    dataFrame, libSizes, sample, tau, Tp, E,
    targets, columns,
    target_measurement='correlation',
    columns_measurement='correlation',
    target_aggregate='mean',
    target_pool_depth=None,
    target_select_E=None,
    exclusionRadius=0,
    seed=None,
    gammaDelta=2,
    show_plot=False,
    pass_fail=False,
):
    """CCM with a multi-variable target manifold.

    Source side (``columns``): standard E-delay Takens embedding of one
    variable.

    Target side (``targets``, a list): build a pool of delayed
    embeddings of every target variable
    (``len(targets) * target_pool_depth`` columns total).

    Two ways to use that pool, controlled by ``target_select_E``:

    * ``None`` (default) -- use the **full pool** deterministically.
      The target embedding has dimension
      ``len(targets) * target_pool_depth`` (asymmetric with the
      source). Recommended in most cases: it loses no delay
      information and avoids the variance from random column draws.

    * ``int`` -- at each library/sample iteration assemble the target
      embedding as

          all m lag-0 columns (one per target, always included)
          + (target_select_E - m) columns drawn at random, without
            replacement, from the strictly delayed columns of the pool.

      Requires ``target_select_E >= len(targets)``. Useful when the
      full pool gets too high-dimensional (curse of dimensionality)
      or when matched source/target dimensions are desired.

    Two directions:
      * fwd (``c2t``): source embedding finds k-NN; predicts each
        target_i(t+Tp) separately and aggregates per-target scores via
        ``target_aggregate``.
      * rev (``t2c``): randomly-drawn target embedding finds k-NN;
        predicts columns(t+Tp).

    With ``len(targets) == 1`` and ``target_pool_depth=E,
    target_select_E=E`` this reduces to ``proper_ccm`` (modulo the
    deterministic identity sampling of the single-target pool).

    Parameters
    ----------
    targets : list[str]
        Target variables.
    target_aggregate : {'mean', 'min'}
        How to combine per-target fwd scores into one c2t value.
    target_pool_depth : int or None
        Per-target delay depth used to build the pool. Defaults to E.
    target_select_E : int or None
        Number of columns to randomly draw from the pool to form the
        target embedding. Defaults to E.
    Other parameters: same as ``proper_ccm``.

    HOW TO USE
    ----------
    >>> c2t, t2c, libSizes = multi_proper_ccm(
    ...     df, libSizes='20 200 20', sample=30, tau=-1, Tp=1, E=3,
    ...     targets=['neuron_A_tsls', 'neuron_B_tsls'], columns='neuron_C_tsls',
    ...     target_measurement='f1', columns_measurement='f1',
    ... )
    Use this instead of proper_ccm when you want to test "does
    `columns` causally drive the *joint* dynamics of several target
    neurons together" rather than one target at a time.
    """
    if isinstance(targets, str):
        targets = [targets]
    if target_pool_depth is None:
        target_pool_depth = E
    # target_select_E=None  -> use the FULL joint pool deterministically.
    # target_select_E=int   -> lag-0-of-every-target + random delayed draws.
    use_full_pool = target_select_E is None

    col_name = columns if isinstance(columns, str) else columns[0]
    col_vec = np.asarray(dataFrame[col_name], dtype=np.float64)
    N = len(col_vec)

    emb_col, valid_col = _build_delay_embedding(col_vec, E, tau)

    tgt_vecs = {
        t: np.asarray(dataFrame[t], dtype=np.float64) for t in targets
    }

    pool, pool_labels, valid_pool = _build_multi_target_pool(
        dataFrame, targets, target_pool_depth, tau,
    )
    pool_size = pool.shape[1]
    m = len(targets)

    if use_full_pool:
        target_select_E = pool_size
        full_pool_idx = np.arange(pool_size, dtype=np.intp)
        lag0_idx = delayed_idx = None
        n_extra  = 0
    else:
        # Lag-0-of-every-target + random delayed draws.
        if target_select_E < m:
            raise ValueError(
                f"target_select_E={target_select_E} must be >= "
                f"len(targets)={m} so every target is represented."
            )
        if pool_size < target_select_E:
            raise ValueError(
                f"Target pool has {pool_size} columns but "
                f"target_select_E={target_select_E} required."
            )
        # _build_multi_target_pool stacks per-target embeddings in
        # order, so target_i's columns occupy indices
        # [i*target_pool_depth, (i+1)*target_pool_depth), with lag 0
        # first.
        lag0_idx = np.array(
            [i * target_pool_depth for i in range(m)], dtype=np.intp,
        )
        delayed_idx = np.setdiff1d(
            np.arange(pool_size, dtype=np.intp), lag0_idx,
        )
        n_extra = target_select_E - m
        if n_extra > len(delayed_idx):
            raise ValueError(
                f"Not enough delayed columns ({len(delayed_idx)}) to "
                f"fill {n_extra} extra slots; increase "
                f"target_pool_depth or decrease target_select_E."
            )
        full_pool_idx = None

    # Forward valid: source embedding usable, all target(t+Tp) usable,
    # source(t) non-NaN.
    valid_fwd = valid_col & ~np.isnan(col_vec)
    for tname in targets:
        valid_fwd = valid_fwd & _tp_valid_mask(N, tgt_vecs[tname], Tp)

    # Reverse valid: full pool usable, source(t+Tp) usable,
    # all target(t) non-NaN.
    valid_rev = valid_pool & _tp_valid_mask(N, col_vec, Tp)
    for tname in targets:
        valid_rev = valid_rev & ~np.isnan(tgt_vecs[tname])

    idx_fwd = np.where(valid_fwd)[0]
    idx_rev = np.where(valid_rev)[0]

    embed_fwd  = np.ascontiguousarray(emb_col[idx_fwd])
    pool_valid = np.ascontiguousarray(pool[idx_rev])

    obs_fwd_dict = {
        t: tgt_vecs[t][idx_fwd + Tp] for t in targets
    }
    obs_rev = col_vec[idx_rev + Tp]

    libSizes = _parse_libSizes(libSizes, E, N)
    n_L = len(libSizes)

    k_fwd = E + 1
    k_rev = target_select_E + 1

    root_seq = np.random.SeedSequence(seed)
    child_seqs = root_seq.spawn(2 * n_L)

    c2t_measurements = np.full((sample, n_L), np.nan)
    t2c_measurements = np.full((sample, n_L), np.nan)

    M_fwd = embed_fwd.shape[0]

    for i, L in enumerate(libSizes):
        rng_fwd = np.random.default_rng(child_seqs[2 * i])
        rng_rev = np.random.default_rng(child_seqs[2 * i + 1])
        for t in range(sample):
            # ---- fwd: predict each target with source embedding ----
            per_target = []
            for tname in targets:
                # _crossmap_predict no longer draws its own library --
                # see that function's own docstring -- so the draw that
                # used to happen inside it now happens here, in the same
                # per-target spot in the loop, preserving this function's
                # existing rng_fwd consumption order/count exactly.
                lib_idx_fwd = rng_fwd.choice(M_fwd, size=min(L, M_fwd), replace=False)
                lib_idx_fwd.sort()
                preds = _crossmap_predict(
                    embed_fwd, obs_fwd_dict[tname], lib_idx_fwd, k_fwd,
                    exclusionRadius=exclusionRadius,
                    time_indices=idx_fwd,
                )
                per_target.append(_score(
                    obs_fwd_dict[tname], preds, target_measurement,
                    gammaDelta=gammaDelta,
                ))
            arr = np.asarray(per_target, dtype=float)
            if target_aggregate == 'mean':
                c2t_measurements[t, i] = np.nanmean(arr)
            elif target_aggregate == 'min':
                c2t_measurements[t, i] = np.nanmin(arr)
            else:
                raise ValueError(
                    f'target_aggregate must be mean/min, got '
                    f'{target_aggregate!r}'
                )

            # ---- rev: target embedding ---------------------------------
            # full pool (deterministic) or lag-0 + random delayed draws.
            if use_full_pool:
                col_idx = full_pool_idx
            elif n_extra > 0:
                extra = rng_rev.choice(
                    delayed_idx, size=n_extra, replace=False,
                )
                col_idx = np.concatenate([lag0_idx, extra])
            else:
                col_idx = lag0_idx
            target_embed = np.ascontiguousarray(pool_valid[:, col_idx])
            lib_idx_rev = rng_rev.choice(
                target_embed.shape[0], size=min(L, target_embed.shape[0]), replace=False,
            )
            lib_idx_rev.sort()
            preds_rev = _crossmap_predict(
                target_embed, obs_rev, lib_idx_rev, k_rev,
                exclusionRadius=exclusionRadius,
                time_indices=idx_rev,
            )
            t2c_measurements[t, i] = _score(
                obs_rev, preds_rev, columns_measurement,
                gammaDelta=gammaDelta,
            )

    if pass_fail:
        from sklearn.linear_model import LinearRegression
        mean_t2c = np.nanmean(t2c_measurements, axis=0)
        model = LinearRegression().fit(
            np.arange(n_L).reshape(-1, 1), mean_t2c,
        )
        slope = model.coef_[0]
        if columns_measurement in ('correlation', 'f1', 'gamma'):
            return bool((mean_t2c[-1] - mean_t2c[0]) > 0 and slope > 0.01)
        if columns_measurement == 'mae':
            return bool((mean_t2c[-1] - mean_t2c[0]) < 0 and slope < -0.01)

    if show_plot:
        mean_c2t = np.nanmean(c2t_measurements, axis=0)
        mean_t2c = np.nanmean(t2c_measurements, axis=0)
        std_c2t  = np.nanstd(c2t_measurements,  axis=0) / np.sqrt(n_L)
        std_t2c  = np.nanstd(t2c_measurements,  axis=0) / np.sqrt(n_L)
        plot_ccm_results(
            libSizes, mean_c2t, mean_t2c, std_c2t, std_t2c,
            target_measurement, columns_measurement,
            col_name, ','.join(targets),
        )

    return c2t_measurements, t2c_measurements, libSizes


def _simplex_multi_metric(
    data, columns, targets, lib, pred, tau, Tp, E,
    embedded, metric, gammaDelta=2, noTime=True, aggregate='mean',
):
    """Run Simplex once per target in ``targets`` with the same source
    ``columns``; aggregate per-target scores via ``aggregate``.

    INTUITION: multi_proper_mde's ranking-stage counterpart to
    _simplex_metric -- since a candidate column list now has to predict
    *several* targets at once, this scores each target separately with
    its own Simplex(target | columns) call and folds the per-target
    scores into one number via `aggregate` ('mean' rewards good average
    performance; 'min' enforces "must do reasonably well on EVERY
    target", a stricter requirement).

    INPUT: same as _simplex_metric, but `targets` is a list[str] (or a
    single str, normalized to a 1-element list). `aggregate` -- 'mean'
    or 'min'.
    OUTPUT: float aggregated score (NaN-propagating: np.nanmean/nanmin
    over the per-target scores).
    """
    if isinstance(targets, str):
        targets = [targets]
    scores = []
    for tname in targets:
        s = _simplex_metric(
            data, columns, tname, lib, pred, tau, Tp, E,
            embedded, metric, gammaDelta=gammaDelta, noTime=noTime,
        )
        scores.append(s)
    arr = np.asarray(scores, dtype=float)
    if aggregate == 'mean':
        return float(np.nanmean(arr))
    if aggregate == 'min':
        return float(np.nanmin(arr))
    raise ValueError(f'aggregate must be mean/min, got {aggregate!r}')


def multi_proper_mde(
    dataFrame, targets, D=3,
    *,
    columns=None,
    removeColumns=(),
    removeTime=False,
    lib=None, pred=None,
    Tp=1, tau=-1, exclusionRadius=0,
    sample=20,
    pLibSizes=(10, 15, 85, 90),
    libSizes=None,
    noCCM=False,
    ccmSlope=0.01,
    ccmSeed=None,
    E=3,
    crossMapRhoMin=0.5,
    embedDimRhoMin=0.5,
    metric='correlation',
    ccmMetric=None,
    gammaDelta=2,
    target_aggregate='mean',
    target_pool_depth=None,
    target_select_E=None,
    noTime=True,
    verbose=False,
):
    """Multi-target MDE: forward-stepwise selection of neurons that
    jointly predict (and causally drive) all variables in ``targets``.

    Same structure as ``proper_mde`` but every reference to a single
    ``target`` is replaced by the list ``targets`` and:

    * Ranking stage runs Simplex once per target with the same
      ``[candidate] + selected`` column list and embedded=True, then
      aggregates per-target scores via ``target_aggregate`` (mean or
      min).
    * CCM gate calls ``multi_proper_ccm`` with the multi-target joint
      pool; the slope test uses the ``t2c`` direction (joint target
      manifold cross-mapped to neuron), which is the right test for
      "neuron causally drives the joint behavior".

    ``EmbedDimension`` is skipped (it's a single-target rho-only
    pyEDM routine); pass an explicit ``E > 0``. Defaults to ``E=3``.

    Parameters
    ----------
    targets : list[str]
        Target variables to jointly predict.
    target_aggregate : {'mean', 'min'}
        How to combine per-target scores at both stages.
    target_pool_depth : int or None
        Forwarded to ``multi_proper_ccm``; defaults to ``E``.
    target_select_E : int or None
        Forwarded to ``multi_proper_ccm``. ``None`` (default) ⇒ use
        the full joint target pool deterministically (recommended).
        An int opts into the lag-0 + random delayed draws rule.
    Other parameters: same meaning as in ``proper_mde``.

    INTUITION (plain-language)
    ---------------------------
    Same forward-selection idea as `proper_mde`, but instead of picking
    neurons that predict/are-predicted-by one target column, it picks
    neurons that jointly predict/are-driven-by a whole *group* of target
    columns at once (their `target_aggregate`-combined score). Useful
    when "the behavior" you care about is really several correlated
    variables together (e.g. x and y position) rather than one scalar.

    HOW TO USE
    ----------
    >>> out = multi_proper_mde(df, targets=['head_position_x', 'head_position_y'],
    ...                         D=3, E=3, tau=-1, Tp=1, metric='correlation')
    >>> out['MDEcolumns']
    Requires an explicit ``E > 0`` (no per-candidate EmbedDimension
    auto-search here, unlike single-target `proper_mde` with ``E=0``).
    """
    from sklearn.linear_model import LinearRegression

    if isinstance(targets, str):
        targets = [targets]

    if removeTime:
        dataFrame = dataFrame.drop(columns=dataFrame.columns[0])

    N = dataFrame.shape[0]
    if lib  is None: lib  = [1, N // 2]
    if pred is None: pred = [N // 2 + 1, N]

    if E <= 0:
        raise ValueError(
            "multi_proper_mde requires an explicit E > 0 "
            "(EmbedDimension is single-target rho-only)."
        )
    if ccmMetric is None:
        ccmMetric = metric

    if target_pool_depth is None: target_pool_depth = E
    # target_select_E=None  ->  multi_proper_ccm uses the full joint
    # pool deterministically (recommended). Pass an int to opt into
    # the lag-0 + random delayed draws rule.

    direction     = _metric_direction(metric)
    direction_ccm = _metric_direction(ccmMetric)
    sign_better = lambda x, t: (direction * x) >= (direction * t)

    # See proper_mde()'s own identical `libSizes` docstring/comment for why
    # this bypasses the `pLibSizes`-as-percentage-of-N computation when given.
    libSizes = list(libSizes) if libSizes is not None else [int(N * p / 100.0) for p in pLibSizes]
    libSizesVec = np.array(libSizes, dtype=float).reshape(-1, 1)
    libSizesVec = libSizesVec / libSizesVec[-1]

    remove_set = set(removeColumns) | set(targets)
    if columns is None:
        dataColumns = [c for c in dataFrame.columns if c not in remove_set]
    else:
        dataColumns = [c for c in columns if c not in remove_set]

    MDEcolumns = []
    MDErho     = []
    rhoD_log   = {}

    for d in range(1, D + 1):
        candidates = [c for c in dataColumns if c not in MDEcolumns]
        column_lists = [[c] + MDEcolumns for c in candidates]

        if verbose:
            print(f'\n{d}-D Multi Cross Map  targets={targets}  '
                  f'{len(column_lists)} candidates')

        # ---- Ranking: multi-target Simplex, embedded=True ----
        rhoD = {}
        for cl in column_lists:
            s = _simplex_multi_metric(
                dataFrame, cl, targets, lib, pred,
                tau=tau, Tp=Tp, E=0, embedded=True,
                metric=metric, gammaDelta=gammaDelta,
                noTime=noTime, aggregate=target_aggregate,
            )
            key = ','.join(cl) + ':' + ','.join(targets)
            rhoD[key] = (s, cl)

        L_rhoD = sorted(
            rhoD.values(),
            key=lambda x: -direction * (
                x[0] if not (isinstance(x[0], float) and np.isnan(x[0]))
                else -direction * np.inf
            ),
        )
        L_rhoD = [
            (s, cl) for (s, cl) in L_rhoD
            if not (isinstance(s, float) and np.isnan(s))
            and sign_better(s, crossMapRhoMin)
        ]

        if len(L_rhoD) == 0:
            if verbose:
                print(f'  {d}-D failed to pass crossMapRhoMin.')
            continue

        rhoD_log[d] = L_rhoD
        if verbose:
            print(f'  L_rhoD[:3] {[(round(s,4), cl) for s, cl in L_rhoD[:3]]}')

        if noCCM:
            s_top, cl_top = L_rhoD[0]
            newColumn = cl_top[0]
            MDEcolumns.append(newColumn)
            MDErho.append(s_top)
            if verbose:
                print(f'  {d}-D added {newColumn!r} (noCCM, '
                      f'{metric}={s_top:.4f})')
            continue

        # ---- CCM gate via multi_proper_ccm ----
        maxCol_i = None
        for col_i, (s_rank, cl) in enumerate(L_rhoD):
            newColumn = cl[0]

            if not sign_better(s_rank, embedDimRhoMin):
                continue

            try:
                _, t2c, _ = multi_proper_ccm(
                    dataFrame,
                    libSizes=libSizes, sample=sample,
                    tau=tau, Tp=Tp, E=E,
                    targets=targets, columns=newColumn,
                    target_measurement=ccmMetric,
                    columns_measurement=ccmMetric,
                    target_aggregate=target_aggregate,
                    target_pool_depth=target_pool_depth,
                    target_select_E=target_select_E,
                    exclusionRadius=exclusionRadius,
                    seed=ccmSeed,
                    gammaDelta=gammaDelta,
                )
            except Exception as exc:
                if verbose:
                    print(f'    {newColumn}: multi_proper_ccm raised {exc}')
                continue

            ccmVals = np.nan_to_num(np.nanmean(t2c, axis=0))
            slope = float(
                LinearRegression().fit(libSizesVec, ccmVals).coef_[0]
            )
            slope = round(slope, 5)

            if verbose:
                diag = ''
                if ccmMetric == 'f1':
                    # Same candidate-side event-count diagnostic as
                    # proper_mde's own CCM gate -- see that call site's
                    # comment for why (docs/review-findings.md).
                    cand_vals = np.asarray(dataFrame[newColumn], dtype=np.float64)
                    cand_events = len(detect_reset_events_from_tsls(
                        cand_vals,
                        near_zero_threshold=infer_near_zero_threshold_from_true_tsls(cand_vals),
                        drop_threshold=0.0,
                    ))
                    diag = f' candidate_events_in_frame={cand_events}'
                print(f'    {",".join(targets)}:{newColumn} slope {slope}{diag}')

            if (direction_ccm * slope) > ccmSlope:
                maxCol_i = col_i
                break

        if maxCol_i is not None:
            s_chosen, cl_chosen = L_rhoD[maxCol_i]
            newColumn = cl_chosen[0]
            MDEcolumns.append(newColumn)
            MDErho.append(s_chosen)
            if verbose:
                print(f'  {d}-D added {newColumn!r}  '
                      f'({metric}={s_chosen:.4f})')
        else:
            if verbose:
                print(f'  {d}-D CCM failed to find columns.')
            break

    return {
        'MDEcolumns': MDEcolumns,
        'MDErho':     np.array(MDErho),
        'rhoD':       rhoD_log,
        'libSizes':   libSizes,
        'targets':    list(targets),
    }


def _simplex_kdtree_predict(
    embedding, scalar_target_vec, lib_lo, lib_hi, pred_lo, pred_hi,
    valid_rows, Tp, knn, exclusionRadius,
):
    """Run one Simplex projection from a pre-built embedding to a
    single scalar target vector. Returns ``(obs_full, preds_full,
    pred_idx_full)``: per-row obs and pred arrays the length of the
    pred range, with NaN where invalid.

    INTUITION: this is the single-library-split counterpart to
    _crossmap_predict -- rather than CCM's many random library
    subsamples across a size sweep, it's a classic, one-shot Simplex
    projection: a FIXED library range [lib_lo, lib_hi) is used to build
    the KDTree, and every point in a FIXED prediction range
    [pred_lo, pred_hi) is predicted from its k nearest library
    neighbors. This is what backs proper_simplex, multi_simplex, and
    rolling_simplex (each piece of a rolling window calls this once).

    INPUT: embedding -- (N, D) pre-built embedding matrix (already
    delay-lagged or otherwise assembled by the caller).
    scalar_target_vec -- (N,) array, the raw (unlagged) values being
    predicted; this function applies the Tp shift itself.
    lib_lo/lib_hi, pred_lo/pred_hi -- 0-indexed half-open [lo, hi)
    row ranges for the library and prediction sets respectively (may
    overlap, e.g. rolling_simplex's baseline-piece-predicts-itself case).
    valid_rows -- (N,) bool mask of rows with a complete, non-NaN
    embedding (from _build_delay_embedding / caller-provided). Tp --
    prediction horizon. knn -- number of nearest neighbors to average.
    exclusionRadius -- temporal exclusion window (see _crossmap_predict).

    OUTPUT: (obs_full, preds_full, pred_idx_full) -- all three arrays
    are exactly `pred_hi - pred_lo` long, one entry per row in the
    prediction range (including invalid/skipped rows, which get NaN in
    obs_full/preds_full so the output always aligns positionally with
    `pred_idx_full = arange(pred_lo, pred_hi)`).

    Raises ValueError if the library doesn't have more than `knn` valid
    rows (can't even form one full neighbor set).
    """
    from scipy.spatial import KDTree

    N = embedding.shape[0]
    all_idx = np.arange(N)

    lib_mask  = (all_idx >= lib_lo)  & (all_idx < lib_hi)  & valid_rows
    pred_mask = (all_idx >= pred_lo) & (all_idx < pred_hi) & valid_rows

    lib_idx       = np.where(lib_mask)[0]
    pred_idx_full = np.arange(pred_lo, pred_hi)
    pred_idx      = np.where(pred_mask)[0]

    if len(lib_idx) <= knn:
        raise ValueError(
            f'Library has {len(lib_idx)} valid rows, need > knn={knn}.'
        )
    if len(pred_idx) == 0:
        obs_full   = np.full(len(pred_idx_full), np.nan)
        preds_full = np.full(len(pred_idx_full), np.nan)
        return obs_full, preds_full, pred_idx_full

    tree = KDTree(embedding[lib_idx])
    k_query = knn + 1
    if exclusionRadius > 0:
        k_query += 2 * exclusionRadius
    k_query = min(k_query, len(lib_idx))

    dist, loc = tree.query(embedding[pred_idx], k=k_query)
    if dist.ndim == 1:
        dist = dist[:, None]
        loc  = loc[:,  None]
    nn_global = lib_idx[loc]

    M = len(pred_idx)
    r_M = pred_idx[:, None]
    is_self = (nn_global == r_M)
    if exclusionRadius > 0:
        is_excl = np.abs(r_M - nn_global) <= exclusionRadius
        mask_bad = is_self | is_excl
    else:
        mask_bad = is_self

    valid_neigh = ~mask_bad
    cs          = np.cumsum(valid_neigh, axis=1)
    first_k     = valid_neigh & (cs <= knn)
    insufficient = cs[:, -1] < knn

    nn_dist_k = np.full((M, knn), np.nan)
    nn_glob_k = np.zeros((M, knn), dtype=np.intp)
    suf = ~insufficient
    if suf.any():
        sel = first_k.copy()
        sel[insufficient, :] = False
        _, cols = np.where(sel)
        cols = cols.reshape(int(suf.sum()), knn)
        rows = np.where(suf)[0][:, None]
        nn_dist_k[suf] = dist[rows, cols]
        nn_glob_k[suf] = nn_global[rows, cols]

    d_min    = nn_dist_k[:, 0:1]
    d_min_nz = np.where(d_min > 0.0, d_min, 1.0)
    w        = np.exp(-nn_dist_k / d_min_nz)
    zero     = (d_min == 0.0)
    if zero.any():
        w = np.where(zero, np.where(nn_dist_k == 0.0, 1.0, 0.0), w)
    w_sum = w.sum(axis=1, keepdims=True)
    w_sum = np.where(w_sum > 0.0, w_sum, 1.0)
    w /= w_sum

    nbr_tp = nn_glob_k + Tp
    in_b   = (nbr_tp >= 0) & (nbr_tp < N)
    safe   = np.where(in_b, nbr_tp, 0)
    nbr_v  = scalar_target_vec[safe]
    nbr_v  = np.where(in_b, nbr_v, np.nan)

    preds_valid = np.full(M, np.nan)
    if suf.any():
        preds_valid[suf] = np.sum(w[suf] * nbr_v[suf], axis=1)

    pos_in_full = np.searchsorted(pred_idx_full, pred_idx)
    obs_full   = np.full(len(pred_idx_full), np.nan)
    preds_full = np.full(len(pred_idx_full), np.nan)
    obs_full[pos_in_full]   = scalar_target_vec[pred_idx + Tp]
    preds_full[pos_in_full] = preds_valid

    return obs_full, preds_full, pred_idx_full


def multi_simplex(
    dataFrame, columns, targets,
    lib=None, pred=None,
    E=3, tau=-1, Tp=1,
    knn=None,
    exclusionRadius=0,
    embedded=False,
    target_pool_depth=None,
    target_select_E=None,
    seed=None,
    noTime=True,
):
    """Simplex projection with joint multi-target embedding.

    Two manifolds are built once over the full data:

    * **Source** manifold from ``columns``. ``embedded=False``: build
      an ``E``-delay Takens embedding per source column and stack them
      horizontally (dim = ``E * len(columns)``). ``embedded=True``:
      take ``columns`` as the embedding itself (dim = ``len(columns)``).

    * **Joint-target** manifold from ``targets``. Build an
      ``target_pool_depth``-delay embedding of every target and stack
      them into a pool (``len(targets) * target_pool_depth`` columns).
      Two modes, controlled by ``target_select_E``:

      * ``None`` (default) -- use the **full pool** deterministically;
        target embedding dim = ``len(targets) * target_pool_depth``.
      * ``int`` -- assemble as
        ``[every target's lag-0 column] + (target_select_E - m)``
        randomly drawn columns from the strictly delayed pool
        (seeded by ``seed``). Same rule as ``multi_proper_ccm``.

      Default ``target_pool_depth = E``.

    The cross-map is computed in *both* directions, mirroring
    ``multi_proper_ccm`` but at a single fixed ``(lib, pred)`` split
    with no random library subsampling:

    * ``c2t`` — source neighbors → predict each target's value at
      ``t+Tp``. One DataFrame per target.
    * ``t2c`` — joint-target neighbors → predict the source's value at
      ``t+Tp`` (the *first* source column when ``columns`` is
      multivariate, matching pyEDM CCM's convention).

    Parameters
    ----------
    dataFrame : pandas.DataFrame
    columns : str or list[str]
        Source variable(s).
    targets : str or list[str]
        Target variables; ``len(targets)`` of them.
    lib, pred : list[int] or None
        1-offset ``[start, stop]`` inclusive ranges. Defaults:
        ``[1, N//2]``, ``[N//2+1, N]``.
    E, tau, Tp, knn, exclusionRadius, embedded :
        Standard pyEDM Simplex semantics. ``knn`` defaults to the
        relevant embedding's dim + 1, computed separately for each
        direction.
    target_pool_depth : int or None
        Per-target delay depth used to build the target pool. Default
        ``E``.
    target_select_E : int or None
        Number of columns the target embedding ends up with (must be
        ≥ ``len(targets)`` so every original is represented). Default
        ``E``.
    seed : int or None
        Seed for the random draw of the delayed target columns.
    noTime : bool
        API parity with pyEDM; this function never emits a Time column.

    Returns
    -------
    dict with keys
        ``'c2t'``  dict ``{target_name: DataFrame(Observations,
                   Predictions)}`` indexed by the pred range.
        ``'t2c'``  ``DataFrame(Observations, Predictions)`` indexed by
                   the pred range; predicts ``columns[0]``.
        ``'target_columns'``  list of ``(target_name, lag_index)``
                   tuples describing the columns of the joint target
                   embedding actually used.

    INTUITION (plain-language)
    ---------------------------
    multi-target version of proper_simplex over a single fixed lib/pred
    split (no CCM library-size sweep): builds a source embedding from
    `columns` and a *joint* embedding from all of `targets` stacked
    together (via _build_multi_target_pool), then runs Simplex
    projection in both directions -- source neighbors predict each
    target individually (`c2t`, one DataFrame per target), and joint-
    target neighbors predict the source (`t2c`). Useful as a quick,
    single-split diagnostic for "how well does this one neuron connect
    to this group of targets" without the cost of a full CCM sweep.

    HOW TO USE
    ----------
    >>> out = multi_simplex(df, columns='neuron_C_tsls',
    ...                      targets=['neuron_A_tsls', 'neuron_B_tsls'], E=3, tau=-1, Tp=1)
    >>> out['c2t']['neuron_A_tsls']   # DataFrame(Observations, Predictions)
    >>> out['t2c']                    # DataFrame(Observations, Predictions), predicts neuron_C_tsls
    """
    # ---- Normalise arguments ----
    if isinstance(columns, str):
        col_list = columns.split()
    else:
        col_list = list(columns)
    if isinstance(targets, str):
        tgt_list = targets.split()
    else:
        tgt_list = list(targets)
    if not col_list:
        raise ValueError('columns is empty.')
    if not tgt_list:
        raise ValueError('targets is empty.')

    if target_pool_depth is None:
        target_pool_depth = E
    use_full_pool = target_select_E is None

    m = len(tgt_list)
    if not use_full_pool and target_select_E < m:
        raise ValueError(
            f'target_select_E={target_select_E} must be >= len(targets)={m}.'
        )

    N = dataFrame.shape[0]
    if lib  is None: lib  = [1, N // 2]
    if pred is None: pred = [N // 2 + 1, N]
    lib_lo,  lib_hi  = lib[0]  - 1, lib[1]
    pred_lo, pred_hi = pred[0] - 1, pred[1]

    # ---- Source embedding ----
    if embedded:
        source_emb = np.column_stack([
            np.asarray(dataFrame[c], dtype=np.float64) for c in col_list
        ])
        valid_src = ~np.any(np.isnan(source_emb), axis=1)
    else:
        per_col = []
        valid_src = None
        for c in col_list:
            vec = np.asarray(dataFrame[c], dtype=np.float64)
            emb, v = _build_delay_embedding(vec, E, tau)
            per_col.append(emb)
            valid_src = v if valid_src is None else (valid_src & v)
        source_emb = np.hstack(per_col)
    D_src = source_emb.shape[1]

    # ---- Joint-target embedding pool ----
    pool, pool_labels, valid_pool = _build_multi_target_pool(
        dataFrame, tgt_list, target_pool_depth, tau,
    )
    pool_size = pool.shape[1]

    if use_full_pool:
        target_select_E = pool_size
        sel_idx = np.arange(pool_size, dtype=np.intp)
    else:
        if pool_size < target_select_E:
            raise ValueError(
                f'Target pool has {pool_size} columns but '
                f'target_select_E={target_select_E} required.'
            )
        # lag-0 column indices (one per target), then sample the rest
        # from the strictly delayed columns. Same rule as
        # multi_proper_ccm.
        lag0_idx = np.array(
            [i * target_pool_depth for i in range(m)], dtype=np.intp,
        )
        delayed_idx = np.setdiff1d(
            np.arange(pool_size, dtype=np.intp), lag0_idx,
        )
        n_extra = target_select_E - m
        if n_extra > len(delayed_idx):
            raise ValueError(
                f'Not enough delayed columns ({len(delayed_idx)}) for '
                f'{n_extra} extra slots; increase target_pool_depth or '
                f'decrease target_select_E.'
            )
        rng = np.random.default_rng(seed)
        if n_extra > 0:
            extra = rng.choice(delayed_idx, size=n_extra, replace=False)
            sel_idx = np.concatenate([lag0_idx, extra])
        else:
            sel_idx = lag0_idx
    target_emb = np.ascontiguousarray(pool[:, sel_idx])
    target_columns_used = [pool_labels[int(i)] for i in sel_idx]

    # ---- Scalar prediction targets ----
    src_first_vec = np.asarray(dataFrame[col_list[0]], dtype=np.float64)
    tgt_vecs = {
        t: np.asarray(dataFrame[t], dtype=np.float64) for t in tgt_list
    }

    # ---- Joint validity (must satisfy both directions) ----
    #   * source embedding rows usable
    #   * target embedding rows usable
    #   * source(t+Tp) usable
    #   * every target(t+Tp) usable
    tp_mask_targets = np.ones(N, dtype=bool)
    for t in tgt_list:
        tp_mask_targets = tp_mask_targets & _tp_valid_mask(
            N, tgt_vecs[t], Tp,
        )
    tp_mask_src = _tp_valid_mask(N, src_first_vec, Tp)

    valid_rows = valid_src & valid_pool & tp_mask_targets & tp_mask_src

    # ---- knn per direction ----
    knn_fwd = (D_src + 1) if (knn is None or knn <= 0) else knn
    knn_rev = (target_select_E + 1) if (knn is None or knn <= 0) else knn

    # ---- c2t: source embedding -> each target ----
    c2t = {}
    for tname in tgt_list:
        obs, preds, pred_idx_full = _simplex_kdtree_predict(
            source_emb, tgt_vecs[tname],
            lib_lo, lib_hi, pred_lo, pred_hi,
            valid_rows, Tp, knn_fwd, exclusionRadius,
        )
        c2t[tname] = pd.DataFrame(
            {'Observations': obs, 'Predictions': preds},
            index=pred_idx_full,
        )

    # ---- t2c: joint target embedding -> source[0] ----
    obs_r, preds_r, pred_idx_full = _simplex_kdtree_predict(
        target_emb, src_first_vec,
        lib_lo, lib_hi, pred_lo, pred_hi,
        valid_rows, Tp, knn_rev, exclusionRadius,
    )
    t2c = pd.DataFrame(
        {'Observations': obs_r, 'Predictions': preds_r},
        index=pred_idx_full,
    )

    return {
        'c2t':            c2t,
        't2c':            t2c,
        'target_columns': target_columns_used,
    }


def rolling_simplex(
    dataFrame, columns, target,
    window_size, step=None, baseline=0,
    E=3, tau=-1, Tp=1, knn=None,
    exclusionRadius=0, embedded=False,
    metric='correlation', gammaDelta=2,
    noTime=True,
    start_piece=0, skip_pieces=None,
    min_valid_frac=0.0,
):
    """Simplex prediction on rolling windows with a fixed baseline library.

    Divides the time series into overlapping or non-overlapping pieces of
    ``window_size`` rows, separated by ``step`` rows.  One piece (selected
    by ``baseline``) is used as the library; every piece (including the
    baseline itself) is predicted from that library, and the chosen
    ``metric`` is reported per piece.

    Pieces that contain too many NaNs to run Simplex (fewer valid
    embedding rows than ``knn + 1``) are automatically skipped and
    reported as NaN.

    Parameters
    ----------
    dataFrame : pd.DataFrame
    columns : str
        Source column name (the variable whose embedding is used).
    target : str
        Target column name (the variable being predicted).
    window_size : int
        Number of rows in each piece.
    step : int or None
        Stride between consecutive pieces.  Default (None) = window_size
        (non-overlapping).
    baseline : int or 'best'
        0-based index of the piece to use as the library.
        If ``'best'``, the piece whose in-sample metric is best is chosen
        automatically (higher is better for correlation / f1 / gamma;
        lower for mae).  Pieces in ``skip_pieces`` and pieces before
        ``start_piece`` are excluded from the search.
    E : int
        Embedding dimension.
    tau : int
        Embedding delay (negative = past).
    Tp : int
        Prediction horizon.
    knn : int or None
        Number of nearest neighbours (default E + 1).
    exclusionRadius : int
        Temporal exclusion radius around each prediction point.
    embedded : bool
        If True, ``columns`` is treated as a pre-embedded set of columns
        (no delay embedding is built).
    metric : str
        One of 'correlation', 'mae', 'f1', 'gamma'.
    gammaDelta : int
        Timing tolerance for coincidence factor Γ (only used when
        metric='gamma').
    noTime : bool
        If True, first column of dataFrame is data, not a time index.
    start_piece : int
        0-based index of the first piece to evaluate.  Pieces before
        this index are still listed in the output but with NaN metric.
    skip_pieces : set/list of int, or None
        Piece indices to skip entirely (reported as NaN).
    min_valid_frac : float
        Minimum fraction of valid (non-NaN) embedding rows within a
        piece for it to be evaluated.  Default 0.0 (rely only on the
        hard ``knn + 1`` floor).  Set e.g. to 0.5 to require at least
        half the piece to be usable.

    Returns
    -------
    dict with keys:
        'results'  : pd.DataFrame — columns [piece, start, end, metric,
                     valid_rows, skipped].  One row per piece.
        'baseline' : int — index of the piece used as library.
        'metric'   : str — name of the metric used.
        'pieces'   : list of (start, end) tuples per piece.
        'per_piece': dict mapping piece index → pd.DataFrame with columns
                     [Observations, Predictions] for that piece (only for
                     non-skipped pieces).

    INTUITION (plain-language)
    ---------------------------
    Tests whether a Simplex model trained on ONE chunk of the recording
    ("baseline") still predicts well in every OTHER chunk -- a
    non-stationarity / drift diagnostic. Chop the series into
    fixed-width, back-to-back pieces; pick one as the fixed library
    (`baseline`, or auto-select via `baseline='best'` the piece that
    predicts itself best in-sample); predict every piece (including
    itself) from that one library; report the chosen metric per piece.
    A metric that stays flat across pieces means the neuron's dynamics
    (or its relationship to `target`) are stable over the recording; a
    metric that degrades in pieces far from the baseline suggests drift.

    HOW TO USE
    ----------
    >>> out = rolling_simplex(df, columns='neuron_B_tsls', target='neuron_A_tsls',
    ...                        window_size=500, baseline='best', E=3, tau=-1, Tp=1, metric='f1')
    >>> out['results']   # DataFrame: one row per piece, with the metric + whether it was skipped
    """
    df = dataFrame.copy()

    col_vec = df[columns].values.astype(np.float64)
    tgt_vec = df[target].values.astype(np.float64)
    N = len(col_vec)

    if step is None:
        step = window_size

    if knn is None:
        knn = E + 1

    if skip_pieces is None:
        skip_pieces = set()
    else:
        skip_pieces = set(skip_pieces)

    # --- build embedding ---
    if embedded:
        emb = col_vec.reshape(N, 1) if col_vec.ndim == 1 else col_vec
        valid = ~np.any(np.isnan(emb), axis=1)
    else:
        emb, valid = _build_delay_embedding(col_vec, E, tau)

    tp_ok = _tp_valid_mask(N, tgt_vec, Tp)
    valid = valid & tp_ok

    # --- determine pieces ---
    pieces = []
    s = 0
    while s + window_size <= N:
        pieces.append((s, s + window_size))
        s += step
    if len(pieces) == 0:
        raise ValueError(
            f'window_size={window_size} exceeds data length {N}.'
        )

    n_pieces = len(pieces)

    # --- count valid rows per piece ---
    piece_valid_count = np.array(
        [valid[lo:hi].sum() for lo, hi in pieces], dtype=int
    )
    min_valid_abs = max(knn + 1, int(np.ceil(min_valid_frac * window_size)))

    def _piece_usable(idx):
        if idx < start_piece:
            return False
        if idx in skip_pieces:
            return False
        if piece_valid_count[idx] < min_valid_abs:
            return False
        return True

    # --- helper to predict one piece from the baseline library ---
    def _predict_piece(lib_piece_idx, pred_piece_idx):
        lib_lo, lib_hi = pieces[lib_piece_idx]
        pred_lo, pred_hi = pieces[pred_piece_idx]
        obs, preds, _ = _simplex_kdtree_predict(
            emb, tgt_vec, lib_lo, lib_hi, pred_lo, pred_hi,
            valid, Tp, knn, exclusionRadius,
        )
        return obs, preds

    # --- auto-select baseline if 'best' ---
    if baseline == 'best':
        best_i = None
        best_val = None
        higher_better = metric in ('correlation', 'f1', 'gamma')
        for i in range(n_pieces):
            if not _piece_usable(i):
                continue
            try:
                obs, preds = _predict_piece(i, i)
            except ValueError:
                continue
            val = _score(obs, preds, metric, gammaDelta=gammaDelta)
            if np.isnan(val):
                continue
            if best_val is None:
                best_val = val
                best_i = i
            elif higher_better and val > best_val:
                best_val = val
                best_i = i
            elif not higher_better and val < best_val:
                best_val = val
                best_i = i
        if best_i is None:
            raise ValueError('No usable piece found for baseline.')
        baseline = best_i

    if baseline < 0 or baseline >= n_pieces:
        raise ValueError(
            f'baseline={baseline} out of range [0, {n_pieces - 1}].'
        )
    if not _piece_usable(baseline):
        raise ValueError(
            f'baseline piece {baseline} is not usable '
            f'(valid_rows={piece_valid_count[baseline]}, '
            f'need >= {min_valid_abs}).'
        )

    # --- predict every piece from the baseline library ---
    rows = []
    per_piece = {}
    for j in range(n_pieces):
        lo, hi = pieces[j]
        n_valid = int(piece_valid_count[j])

        if not _piece_usable(j):
            rows.append({
                'piece': j, 'start': lo, 'end': hi,
                metric: np.nan, 'valid_rows': n_valid, 'skipped': True,
            })
            continue

        try:
            obs, preds = _predict_piece(baseline, j)
        except ValueError:
            rows.append({
                'piece': j, 'start': lo, 'end': hi,
                metric: np.nan, 'valid_rows': n_valid, 'skipped': True,
            })
            continue

        val = _score(obs, preds, metric, gammaDelta=gammaDelta)
        rows.append({
            'piece': j, 'start': lo, 'end': hi,
            metric: val, 'valid_rows': n_valid, 'skipped': False,
        })
        per_piece[j] = pd.DataFrame({
            'Observations': obs,
            'Predictions':  preds,
        })

    results = pd.DataFrame(rows)

    return {
        'results':   results,
        'baseline':  baseline,
        'metric':    metric,
        'pieces':    pieces,
        'per_piece': per_piece,
    }


def proper_simplex(
    dataFrame, columns, target,
    lib=None, pred=None,
    E=3, tau=-1, Tp=1, knn=None,
    exclusionRadius=0, embedded=False,
    metric='correlation', gammaDelta=2,
    noTime=True,
):
    """Simplex projection that directly returns the chosen metric.

    A thin wrapper around the KDTree-based simplex engine that mirrors
    pyEDM's ``Simplex`` interface but lets you score predictions with
    any of the custom metrics (``'correlation'``, ``'mae'``,
    ``'f1'``, ``'gamma'``).

    Parameters
    ----------
    dataFrame : pd.DataFrame
    columns : str or list[str]
        Source variable(s) whose delay embedding drives the prediction.
        When ``embedded=True`` these columns ARE the embedding (no
        further delay expansion).
    target : str
        Variable to predict.
    lib, pred : list[int] or None
        1-offset ``[start, stop]`` inclusive ranges.
        Default: ``lib = [1, N//2]``, ``pred = [N//2+1, N]``.
    E : int
        Embedding dimension (ignored when ``embedded=True``; the number
        of ``columns`` is used instead).
    tau : int
        Embedding delay (negative → past lags).
    Tp : int
        Prediction horizon.
    knn : int or None
        Number of nearest neighbours.  Default ``E + 1`` (or
        ``len(columns) + 1`` when ``embedded=True``).
    exclusionRadius : int
        Temporal exclusion radius around each prediction point.
    embedded : bool
        If True, treat the supplied ``columns`` as a ready-made
        embedding matrix.
    metric : str or list[str]
        One of ``'correlation'``, ``'mae'``, ``'f1'``, ``'gamma'``,
        or a list of several.  When a list is given every metric is
        computed and the returned ``'metrics'`` dict has one entry per
        name.
    gammaDelta : int
        Timing tolerance for Γ (used only when metric includes
        ``'gamma'``).
    noTime : bool
        If True, first column is data, not a time index.

    Returns
    -------
    dict with keys
        ``'predictions'`` : pd.DataFrame with ``Observations`` and
            ``Predictions`` columns, indexed by the pred range.
        ``'metrics'``     : dict ``{metric_name: float}`` — one entry per
            requested metric.
        ``'lib'``         : ``[start, stop]`` (1-offset) actually used.
        ``'pred'``        : ``[start, stop]`` (1-offset) actually used.

    INTUITION (plain-language)
    ---------------------------
    This is the file's basic building block: "given `columns`'s recent
    history (an E-dimensional delay embedding, or the columns themselves
    if `embedded=True`), predict `target`, `Tp` steps ahead, using
    k-nearest-neighbor simplex projection" -- the classic univariate EDM
    Simplex algorithm, but reimplemented with a KDTree instead of calling
    out to pyEDM, so it can be scored with the TSLS-aware metrics
    ('f1'/'gamma') pyEDM itself doesn't know about. Half the data
    (`lib`) is used to build the neighbor-search structure; the other
    half (`pred`) is where prediction quality is actually measured --
    this train/test split is what makes the resulting score meaningful
    (not just curve-fitting).

    HOW TO USE
    ----------
    >>> out = proper_simplex(df, columns='neuron_B_tsls', target='neuron_A_tsls',
    ...                       E=3, tau=-1, Tp=1, metric=['correlation', 'f1'])
    >>> out['metrics']        # {'correlation': 0.62, 'f1': 0.71}
    >>> out['predictions']    # DataFrame(Observations, Predictions) you can plot directly
    Use ``columns=[c1, c2, ...], embedded=True`` to predict `target` from
    a ready-made multivariate state (e.g. an MDE-selected column set)
    instead of a single variable's own delay embedding.
    """
    df = dataFrame
    if isinstance(columns, str):
        columns = [columns]

    N = df.shape[0]

    if lib is None:
        lib = [1, N // 2]
    if pred is None:
        pred = [N // 2 + 1, N]

    lib_lo  = lib[0]  - 1
    lib_hi  = lib[1]
    pred_lo = pred[0] - 1
    pred_hi = pred[1]

    tgt_vec = df[target].values.astype(np.float64)

    # --- build embedding ---
    if embedded:
        emb_mat = df[columns].values.astype(np.float64)
        if emb_mat.ndim == 1:
            emb_mat = emb_mat[:, np.newaxis]
        valid = ~np.any(np.isnan(emb_mat), axis=1)
        eff_dim = emb_mat.shape[1]
    else:
        parts = []
        valid = np.ones(N, dtype=bool)
        for c in columns:
            e, v = _build_delay_embedding(df[c].values.astype(np.float64),
                                          E, tau)
            parts.append(e)
            valid &= v
        emb_mat = np.hstack(parts)
        eff_dim = emb_mat.shape[1]

    tp_ok = _tp_valid_mask(N, tgt_vec, Tp)
    valid = valid & tp_ok

    if knn is None:
        knn = eff_dim + 1

    # --- run simplex ---
    obs, preds, pred_idx = _simplex_kdtree_predict(
        emb_mat, tgt_vec, lib_lo, lib_hi, pred_lo, pred_hi,
        valid, Tp, knn, exclusionRadius,
    )

    pred_df = pd.DataFrame(
        {'Observations': obs, 'Predictions': preds},
        index=pred_idx,
    )

    # --- score ---
    if isinstance(metric, str):
        metrics_list = [metric]
    else:
        metrics_list = list(metric)

    metrics = {}
    for m in metrics_list:
        metrics[m] = _score(obs, preds, m, gammaDelta=gammaDelta)

    return {
        'predictions': pred_df,
        'metrics':     metrics,
        'lib':         lib,
        'pred':        pred,
    }


def proper_embed_dimension(
    dataFrame, columns, target,
    maxE=10,
    lib=None, pred=None,
    Tp=1, tau=-1,
    exclusionRadius=0,
    metric='correlation', gammaDelta=2,
    noTime=True,
    showPlot=False,
    pool=None,
):
    """Estimate optimal embedding dimension E ∈ [1, maxE] under any metric.

    Drop-in replacement for pyEDM's ``EmbedDimension`` but scores with
    ``'correlation'``, ``'mae'``, ``'f1'``, or ``'gamma'`` (or several
    at once).

    Parameters
    ----------
    dataFrame : pd.DataFrame
    columns : str or list[str]
        Source variable(s).
    target : str
        Variable to predict.
    maxE : int
        Maximum embedding dimension to test (default 10).
    lib, pred : list[int] or None
        1-offset ``[start, stop]`` inclusive.  Defaults: ``[1, N//2]``,
        ``[N//2+1, N]``.
    Tp : int
        Prediction horizon.
    tau : int
        Embedding delay.
    exclusionRadius : int
        Temporal exclusion radius.
    metric : str or list[str]
        Metric(s) to compute at each E.  Single string or list.
    gammaDelta : int
        Timing tolerance for Γ.
    noTime : bool
        First column is data, not time.
    showPlot : bool
        If True, plot each metric vs E.
    pool : concurrent.futures.Executor or None
        Optional already-open worker pool -- see proper_mde's own `pool`
        docstring. Each E in ``range(1, maxE+1)`` is fully independent
        (its own ``proper_simplex`` call), so when given, the sweep below
        is dispatched across it instead of running sequentially. ``None``
        (default): unchanged sequential behavior.

    Returns
    -------
    pd.DataFrame with column ``'E'`` and one column per metric.

    INTUITION (plain-language)
    ---------------------------
    Simplest possible embedding-dimension search: just call
    proper_simplex once for every E from 1 to maxE and tabulate the
    score. The E with peak score is generally the "true" dimensionality
    of the underlying dynamics (Takens' theorem territory) -- pick that
    E for downstream Simplex/CCM calls. Unlike pyEDM's own
    EmbedDimension (rho-only), this accepts any of this file's metrics,
    so you can pick E by 'f1'/'gamma' event-timing skill instead of
    correlation when working with TSLS/spike-timing data.

    HOW TO USE
    ----------
    >>> edf = proper_embed_dimension(df, columns='neuron_B_tsls', target='neuron_A_tsls',
    ...                               maxE=15, tau=-1, Tp=1, metric='f1')
    >>> best_E = int(edf.loc[edf['f1'].idxmax(), 'E'])
    """
    if isinstance(metric, str):
        metrics_list = [metric]
    else:
        metrics_list = list(metric)

    N = dataFrame.shape[0]
    if lib is None:
        lib = [1, N // 2]
    if pred is None:
        pred = [N // 2 + 1, N]

    def _run_one_E(E):   # brute-force sweep E=1..maxE, one proper_simplex call each -- independent
        result = proper_simplex(
            dataFrame, columns, target,
            lib=lib, pred=pred,
            E=E, tau=tau, Tp=Tp,
            exclusionRadius=exclusionRadius,
            metric=metrics_list, gammaDelta=gammaDelta,
            noTime=noTime,
        )
        row = {'E': E}
        row.update(result['metrics'])
        return row

    mapper = pool.map if pool is not None else map
    rows = list(mapper(_run_one_E, range(1, maxE + 1)))

    df = pd.DataFrame(rows)

    if showPlot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(
            len(metrics_list), 1,
            figsize=(6, 3 * len(metrics_list)),
            squeeze=False,
        )
        for i, m in enumerate(metrics_list):
            ax = axes[i, 0]
            ax.plot(df['E'], df[m], 'o-', linewidth=2)
            ax.set_xlabel('Embedding Dimension (E)')
            ax.set_ylabel(m)
            ax.set_title(f'Tp={Tp}')
        plt.tight_layout()
        plt.show()

    return df


def proper_predict_interval(
    dataFrame, columns, target,
    maxTp=10,
    lib=None, pred=None,
    E=3, tau=-1,
    exclusionRadius=0,
    metric='correlation', gammaDelta=2,
    noTime=True,
    showPlot=False,
):
    """Estimate optimal prediction interval Tp ∈ [1, maxTp] under any metric.

    Drop-in replacement for pyEDM's ``PredictInterval`` but scores with
    ``'correlation'``, ``'mae'``, ``'f1'``, or ``'gamma'`` (or several
    at once).

    Parameters
    ----------
    dataFrame : pd.DataFrame
    columns : str or list[str]
        Source variable(s).
    target : str
        Variable to predict.
    maxTp : int
        Maximum prediction interval to test (default 10).
    lib, pred : list[int] or None
        1-offset ``[start, stop]`` inclusive.
    E : int
        Embedding dimension.
    tau : int
        Embedding delay.
    exclusionRadius : int
        Temporal exclusion radius.
    metric : str or list[str]
        Metric(s) to compute at each Tp.
    gammaDelta : int
        Timing tolerance for Γ.
    noTime : bool
        First column is data, not time.
    showPlot : bool
        If True, plot each metric vs Tp.

    Returns
    -------
    pd.DataFrame with column ``'Tp'`` and one column per metric.

    INTUITION (plain-language)
    ---------------------------
    Mirror image of proper_embed_dimension: fixes E and sweeps the
    forecast horizon Tp=1..maxTp instead, showing how quickly prediction
    skill decays the further ahead you try to forecast. A rapid initial
    drop indicates a short predictability horizon (e.g. fast, chaotic-ish
    dynamics); a slowly-decaying curve indicates the system is
    predictable further into the future.

    HOW TO USE
    ----------
    >>> tdf = proper_predict_interval(df, columns='neuron_B_tsls', target='neuron_A_tsls',
    ...                                maxTp=10, E=3, tau=-1, metric='f1')
    >>> tdf.plot(x='Tp', y='f1')
    """
    if isinstance(metric, str):
        metrics_list = [metric]
    else:
        metrics_list = list(metric)

    N = dataFrame.shape[0]
    if lib is None:
        lib = [1, N // 2]
    if pred is None:
        pred = [N // 2 + 1, N]

    rows = []
    for Tp_i in range(1, maxTp + 1):
        result = proper_simplex(
            dataFrame, columns, target,
            lib=lib, pred=pred,
            E=E, tau=tau, Tp=Tp_i,
            exclusionRadius=exclusionRadius,
            metric=metrics_list, gammaDelta=gammaDelta,
            noTime=noTime,
        )
        row = {'Tp': Tp_i}
        row.update(result['metrics'])
        rows.append(row)

    df = pd.DataFrame(rows)

    if showPlot:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(
            len(metrics_list), 1,
            figsize=(6, 3 * len(metrics_list)),
            squeeze=False,
        )
        for i, m in enumerate(metrics_list):
            ax = axes[i, 0]
            ax.plot(df['Tp'], df[m], 'o-', linewidth=2)
            ax.set_xlabel('Prediction Interval (Tp)')
            ax.set_ylabel(m)
            ax.set_title(f'E={E}')
        plt.tight_layout()
        plt.show()

    return df

def top_neurons(
    dataFrame, target, N=10,
    *,
    columns=None,
    removeColumns=(),
    removeTime=False,
    lib=None, pred=None,
    Tp=1, tau=-1, exclusionRadius=0,
    sample=20,
    pLibSizes=(10, 15, 85, 90),
    noCCM=False,
    ccmSlope=0.01,
    ccmSeed=None,
    E=0,
    predRhoMin=0.1,
    embedDimRhoMin=0.5,
    maxE=15,
    firstEMax=False,
    metric='correlation',
    ccmMetric=None,
    gammaDelta=2,
    noTime=True,
    verbose=False,
):
    """Find the top *N* independent neurons that predict *target* and
    pass a CCM convergence test.

    Unlike ``proper_mde`` (which builds a multi-dimensional embedding
    one neuron at a time), this function evaluates every candidate
    neuron **independently** — each is scored on its own Simplex
    prediction of ``target``, filtered by ``predRhoMin``, then gated
    with CCM.  The result is a ranked list of up to *N* individually
    validated predictors.

    Algorithm
    ---------
    1. **Rank**: run Simplex(``target`` | ``neuron``) for every
       candidate neuron (each on its own, ``embedded=False``), score
       with ``metric``, sort best-first.
    2. **Threshold**: discard neurons whose prediction score is worse
       than ``predRhoMin`` (sign-aware: ``<`` for correlation / f1 /
       gamma, ``>`` for mae).  These neurons skip the CCM test
       entirely.
    3. **CCM gate** (unless ``noCCM``): for each surviving neuron in
       rank order, resolve ``E`` (via ``EmbedDimension`` or the
       supplied static ``E``), then run ``proper_ccm``.  Accept if the
       convergence slope passes ``ccmSlope``.
    4. Collect up to *N* passers.

    Parameters
    ----------
    dataFrame : pd.DataFrame
    target : str
        Behaviour variable to predict.
    N : int
        Maximum number of neurons to return.
    columns : list[str] or None
        Candidate neuron pool.  ``None`` → all columns minus
        ``removeColumns`` and ``target``.
    removeColumns : iterable[str]
    removeTime : bool
        Drop the first column (time) before anything else.
    lib, pred : list[int] or None
        1-offset ``[start, stop]``.
    Tp, tau, exclusionRadius, sample :
        Passed to Simplex / ``proper_ccm``.
    pLibSizes : iterable[float]
        Percentiles of data length for the CCM library sweep.
    noCCM : bool
        Skip the CCM convergence gate (rank-only mode).
    ccmSlope : float
        CCM convergence-slope threshold.
    ccmSeed : int or None
    E : int
        Static embedding dimension for CCM.  ``0`` → per-neuron
        ``EmbedDimension`` (requires ``metric='correlation'``).
    predRhoMin : float
        Minimum Simplex prediction score to enter the CCM gate.
        Sign is handled automatically (lower-is-better for mae).
    embedDimRhoMin : float
        ``EmbedDimension`` rho threshold (only when ``E=0``).
    maxE, firstEMax : int, bool
        Passed to ``EmbedDimension``.
    metric : str
        Metric for the ranking stage (prediction of ``target``).
    ccmMetric : str or None
        Metric for the CCM gate (prediction of the neuron).
        ``None`` → falls back to ``metric``.
    gammaDelta : int
    noTime : bool
    verbose : bool

    Returns
    -------
    dict with keys
        ``'neurons'``   list[str] — accepted neurons, best first.
        ``'scores'``    np.ndarray — ranking-stage score of each
                        accepted neuron (aligned with ``neurons``).
        ``'EDim'``      dict ``'{neuron}:{target}' -> E``.
        ``'all_ranked'`` list[(score, neuron)] — full ranking before
                        thresholding and CCM, for inspection.
        ``'ccm_slopes'`` dict ``'{neuron}' -> slope`` — CCM slope for
                        every neuron that reached the CCM gate.
        ``'libSizes'``  list[int] used in the CCM gate.

    INTUITION (plain-language)
    ---------------------------
    Different question from `proper_mde`: instead of building ONE
    multivariate embedding by greedily adding neurons one at a time
    (where later picks are scored on top of earlier ones), this scores
    every candidate neuron completely independently against `target`,
    keeps the ones above a skill threshold, validates each with its own
    CCM convergence test, and returns up to N validated neurons ranked
    by their own standalone skill. Use `proper_mde` when you want a
    compact joint embedding (dimensions build on each other); use
    `top_neurons` when you want a flat, independent list of "which
    individual neurons are validated predictors of/predicted by
    `target`" (e.g. to build a candidate pool for a GMN network, or to
    survey a population without imposing a selection order).

    HOW TO USE
    ----------
    >>> out = top_neurons(df, target='head_speed', N=10,
    ...                    columns=candidate_neuron_columns,
    ...                    tau=-1, Tp=1, E=3, metric='correlation')
    >>> out['neurons']   # up to 10 validated neuron column names, best first
    >>> out['scores']    # their ranking-stage scores
    """
    from pyEDM import EmbedDimension as _EmbedDimension
    from scipy.signal import argrelextrema
    from sklearn.linear_model import LinearRegression

    if removeTime:
        dataFrame = dataFrame.drop(columns=dataFrame.columns[0])

    n_rows = dataFrame.shape[0]
    if lib  is None: lib  = [1, n_rows // 2]
    if pred is None: pred = [n_rows // 2 + 1, n_rows]

    if ccmMetric is None:
        ccmMetric = metric

    if E <= 0 and metric != 'correlation':
        raise ValueError(
            "EmbedDimension uses Pearson rho only.  Pass an explicit "
            "E > 0 when metric != 'correlation'."
        )

    direction     = _metric_direction(metric)
    direction_ccm = _metric_direction(ccmMetric)
    sign_better   = lambda x, t: (direction * x) >= (direction * t)

    libSizes    = [int(n_rows * p / 100.0) for p in pLibSizes]
    libSizesVec = np.array(libSizes, dtype=float).reshape(-1, 1)
    libSizesVec = libSizesVec / libSizesVec[-1]

    remove_set = set(removeColumns) | {target}
    if columns is None:
        candidates = [c for c in dataFrame.columns if c not in remove_set]
    else:
        candidates = [c for c in columns if c not in remove_set]

    # ---- Step 1: rank every neuron independently ----
    if verbose:
        print(f'Ranking {len(candidates)} candidates  '
              f'target={target!r}  metric={metric!r}')

    scored = []
    for neuron in candidates:
        s = _simplex_metric(
            dataFrame, neuron, target, lib, pred,
            tau=tau, Tp=Tp, E=(E if E > 0 else 1),
            embedded=False,
            metric=metric, gammaDelta=gammaDelta,
            noTime=noTime,
        )
        scored.append((s, neuron))

    scored.sort(key=lambda x: -direction * (
        x[0] if not (isinstance(x[0], float) and np.isnan(x[0]))
        else -direction * np.inf
    ))

    all_ranked = [(s, n) for s, n in scored]

    # ---- Step 2: threshold ----
    passed_threshold = [
        (s, n) for s, n in scored
        if not (isinstance(s, float) and np.isnan(s))
        and sign_better(s, predRhoMin)
    ]

    if verbose:
        print(f'  {len(passed_threshold)}/{len(candidates)} pass '
              f'predRhoMin={predRhoMin}')
        if passed_threshold:
            print(f'  top 5: {[(round(s,4), n) for s, n in passed_threshold[:5]]}')

    # ---- Step 3: CCM gate ----
    accepted   = []
    acc_scores = []
    EDim       = {}
    ccm_slopes = {}

    for s_rank, neuron in passed_threshold:
        if len(accepted) >= N:
            break

        if noCCM:
            accepted.append(neuron)
            acc_scores.append(s_rank)
            if verbose:
                print(f'  accepted {neuron!r} (noCCM, '
                      f'{metric}={s_rank:.4f})')
            continue

        # (a) resolve E
        if E > 0:
            maxEDim = E
        else:
            EDimDF = _EmbedDimension(
                dataFrame=dataFrame, columns=neuron, target=target,
                maxE=maxE, lib=lib, pred=pred,
                Tp=Tp, tau=tau, exclusionRadius=exclusionRadius,
                noTime=noTime, showPlot=False,
            )
            rho_arr = EDimDF['rho'].to_numpy()
            if firstEMax:
                iMaxArr = argrelextrema(rho_arr, np.greater)[0]
                iMax = int(iMaxArr[0]) if len(iMaxArr) \
                    else len(rho_arr) - 1
            else:
                iMax = int(EDimDF['rho'].round(4).argmax())
            maxRhoEDim = float(EDimDF['rho'].iloc[iMax].round(4))
            maxEDim    = int(EDimDF['E'].iloc[iMax])

            if verbose:
                print(f'    EDim {neuron} E={maxEDim} rho={maxRhoEDim:.4f}')

            if not sign_better(maxRhoEDim, embedDimRhoMin):
                if verbose:
                    print(f'    {neuron} failed embedDimRhoMin')
                continue

        EDim[f'{neuron}:{target}'] = maxEDim

        # (b) CCM convergence
        try:
            _, t2c, _ = proper_ccm(
                dataFrame,
                libSizes=libSizes,
                sample=sample,
                tau=tau, Tp=Tp, E=maxEDim,
                target=target, columns=neuron,
                target_measurement=ccmMetric,
                columns_measurement=ccmMetric,
                embedded=False,
                exclusionRadius=exclusionRadius,
                seed=ccmSeed,
                gammaDelta=gammaDelta,
            )
        except Exception as exc:
            print(f'  {neuron}: proper_ccm failed — {exc}')
            continue

        ccmVals = np.nan_to_num(np.nanmean(t2c, axis=0))
        slope = float(
            LinearRegression().fit(libSizesVec, ccmVals).coef_[0]
        )
        slope = round(slope, 5)
        ccm_slopes[neuron] = slope

        if verbose:
            print(f'    {target}:{neuron} slope={slope}')

        if (direction_ccm * slope) >= ccmSlope:
            accepted.append(neuron)
            acc_scores.append(s_rank)
            if verbose:
                print(f'  accepted {neuron!r}  '
                      f'({metric}={s_rank:.4f}, slope={slope})')
        else:
            if verbose:
                print(f'    {neuron} failed ccmSlope')

    if verbose:
        print(f'\n{len(accepted)}/{N} neurons accepted.')

    return {
        'neurons':    accepted,
        'scores':     np.array(acc_scores),
        'EDim':       EDim,
        'all_ranked': all_ranked,
        'ccm_slopes': ccm_slopes,
        'libSizes':   libSizes,
    }