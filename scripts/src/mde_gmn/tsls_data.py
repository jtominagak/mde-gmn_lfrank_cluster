"""Input data preparation: TSLS conversion, burn-in trim, window sizing.

Turns the raw recording into the frame every later stage works on:
  * convert_to_tsls()  spike trains -> time-since-last-spike signals
  * burn_in_trim()     drops the leading rows where some channel has not
                       fired yet, so no TSLS column is still undefined
  * _resolve_target_ranking_window()  picks each node's own scoring window,
                       large enough to contain MIN_PRED_EVENTS real events
  * _metric_for()      which metric scores a given node (behavioural targets
                       are continuous, neurons are event trains)

Depends on: cleaned_helper_functions (scoring), progress-free (no logging).
Imported by: mde, network, network_parallel, pipeline, diagnostics.
"""

import numpy as np
import pandas as pd

import cleaned_helper_functions as tsls


def region_neuron_columns(df, prefix, n):
    candidates = [f'{prefix}{i}' for i in range(n)]
    return [c for c in candidates if c in df.columns]

def raw_neuron_columns(df, cap):
    return (
        region_neuron_columns(df, 'CA1_', cap)
        + region_neuron_columns(df, 'mPFC_', cap)
        + region_neuron_columns(df, 'OFC_', cap)
    )

def convert_to_tsls(df_raw, neuron_columns, targets, results_dir):
    '''Spike counts -> TSLS for every candidate channel; behavioral TARGETS
    columns pass through untouched (they are already continuous, not spike
    counts). Returns the full (not yet burn-in-trimmed) TSLS DataFrame and
    the list of its _tsls-suffixed observable columns.'''
    time_grid = df_raw['time_bin_center'].to_numpy()

    data = {'time_bin_center': time_grid}
    for t in targets:
        data[t] = df_raw[t].to_numpy()

    for i, neuron in enumerate(neuron_columns):
        spike_times = df_raw.loc[df_raw[neuron] > 0, 'time_bin_center'].to_numpy()
        data[f'{neuron}_tsls'] = tsls.compute_time_since_previous_spike(spike_times, time_grid)
        if (i + 1) % 100 == 0 or i + 1 == len(neuron_columns):
            print(f'  TSLS-converted {i + 1}/{len(neuron_columns)} channels', flush=True)

    tsls_df_full = pd.DataFrame(data)
    tsls_df_full.to_csv(results_dir / 'tsls_data_full.csv', index=False)
    observable_columns = [f'{n}_tsls' for n in neuron_columns]
    print(f'-> tsls_data_full.csv  shape={tsls_df_full.shape}', flush=True)
    return tsls_df_full, observable_columns

def burn_in_trim(tsls_df_full, observable_columns, results_dir):
    '''Drop leading rows until every observable_columns entry is non-NaN
    (see module docstring for why this is a correctness requirement for the
    gmn.Node Generate()/Forecast() handoff, not just cleanliness). Raises
    if any candidate channel never fires at all in this window.'''
    first_valid_idx = tsls_df_full[observable_columns].notna().idxmax()
    never_valid = [c for c in observable_columns if not tsls_df_full[c].notna().any()]
    if never_valid:
        raise ValueError(
            f'{len(never_valid)} channel(s) never fire in this window, TSLS is '
            f'all-NaN: {never_valid} -- widen the window or drop these channels.'
        )

    burn_in_rows = int(first_valid_idx.max())
    slowest_channel = first_valid_idx.idxmax()
    df = tsls_df_full.iloc[burn_in_rows:].reset_index(drop=True)

    df.to_csv(results_dir / 'tsls_data_trimmed.csv', index=False)
    print(f'Burn-in: dropped first {burn_in_rows}/{len(tsls_df_full)} rows '
          f'({burn_in_rows / len(tsls_df_full):.1%}) -- slowest channel to fire: {slowest_channel}',
          flush=True)
    print(f'-> tsls_data_trimmed.csv  shape={df.shape}', flush=True)
    return df

def _ccm_window(df, lib_ratio, max_rows):
    '''Bound the row count fed into any percentage-based CCM libSize sweep
    to max_rows, preserving lib_ratio's proportion within that smaller
    window. See load_config()'s CCM_WINDOW_ROWS docstring for why this is
    necessary (not optional) for CCM's convergence gate to mean anything at
    this pipeline's row counts.'''
    n = min(len(df), max_rows)
    windowed = df.iloc[:n].reset_index(drop=True)
    lib_end = max(1, int(round(n * lib_ratio)))
    return windowed, lib_end

def _resolve_target_ranking_window(target, full_df, usable_df, lib_ratio,
                                    is_behavioral_target, target_ranking_window_rows,
                                    min_pred_events, max_window_rows):
    '''How many rows of data one node's MDE call should score over.

    Resolves the ranking-stage/E-auto-select window for a single
    run_mde_for_target() call, decoupled from CCM_WINDOW_ROWS
    (which stays governing the CCM gate's own convergence-curve library
    sizing, now via absolute row counts -- see `ccm_lib_sizes_rows` in
    run_mde_for_target()'s own docstring -- not this function's own frame
    size at all).

    BEHAVIORAL TARGETS (`is_behavioral_target=True`, always true at the
    depth-0 seed): no TSLS reset-event concept applies (the target is a
    genuine continuous trace, correlation-scored) -- window is simply
    `min(target_ranking_window_rows, len(usable_df))`, a fixed constant
    (default 1000), unrelated to firing rate. `min_pred_events` is not
    used, `max_window_rows` is not used, and the returned event count is
    always `None`.

    NEURON TARGETS (`is_behavioral_target=False`, virtually every Stage 1/
    Stage 2 node): grow the window W (in rows); within W, the PRED portion
    is its own last `1 - lib_ratio` fraction of ROWS (the same split
    `_ccm_window()` will later carve out of this exact W for the CCM
    gate -- see that function). Count `target`'s real reset events that
    fall in that row range -- never by events, only ever by rows -- and
    stop at the smallest W where that count is >= `min_pred_events`.
    `min_pred_events` is a plain, fixed integer (not a fraction of
    anything neuron-specific): it answers "how many matched events does
    f1 need before it is signal rather than noise", which is a property
    of the metric, not of the neuron. A fraction of each neuron's own total
    event count was tried first and rejected: it made a busy neuron face a
    stricter test than a quiet one for no stated reason, and -- checked against
    the WHOLE window rather than its pred
    portion -- was later found on real data to pass while the pred
    portion itself still had too few events to score anything on).

    The search never grows W past `min(max_window_rows, len(usable_df))`
    -- `max_window_rows` is a plain compute ceiling (not calibrated; see
    load_config()'s own docstring), `len(usable_df)` is the leakage-safe
    boundary this function's own caller has already trimmed to (see
    run_mde_for_target()'s own LEAKAGE docstring). ALL event counting here
    happens strictly within `usable_df` (i.e. `[0, B)`) -- never using
    `full_df` rows at or past B, since sizing a search parameter from
    held-out data is the same class of leakage the B boundary exists to
    prevent (`full_df` is accepted only for signature symmetry with the
    behavioral branch and is not read in this branch).

    Returns (window_rows, pred_event_count). On success, `pred_event_count`
    is the achieved count in the resolved window's own pred portion
    (>= min_pred_events). On failure (`min_pred_events` unreachable
    anywhere within the cap), returns (None, pred_event_count) where
    pred_event_count is the best (largest) count achieved at ANY window
    size tried, not necessarily at the cap itself -- the pred portion's
    own event count is not guaranteed monotonic in W (both its edges only
    ever move forward as W grows, but an event row can age out of the
    portion's left edge in the same step before any new event row enters
    at the right, so the count can dip locally) -- the caller can and
    should report the shortfall as `min_pred_events - pred_event_count`
    rather than silently scoring the node on too few events.'''
    n_usable = len(usable_df)
    if is_behavioral_target:
        return min(target_ranking_window_rows, n_usable), None

    cap = min(max_window_rows, n_usable)
    if cap < 1:
        return None, 0

    usable_vals = usable_df[target].to_numpy()[:cap]
    near_zero_threshold = tsls.infer_near_zero_threshold_from_true_tsls(
        full_df[target].to_numpy())
    # detect_reset_events_from_tsls classifies each row from only that row
    # and the one before it (np.diff(..., prepend=x[0])), so truncating the
    # input to `cap` rows first cannot change any detection within [0, cap).
    event_indices = tsls.detect_reset_events_from_tsls(
        usable_vals, near_zero_threshold=near_zero_threshold, drop_threshold=0.0)

    # event_indices is sorted, so searchsorted gives the exact cumulative
    # count up to any row bound with no explicit loop. pred_start[i] is the
    # first row of the pred portion for window size w_candidates[i] -- the
    # same `round(w * lib_ratio)` formula _ccm_window() uses for its own
    # lib_end, so the final resolved window's pred portion here is exactly
    # what the CCM gate will later carve out of it too.
    w_candidates = np.arange(1, cap + 1)
    pred_start = np.round(w_candidates * lib_ratio).astype(int)
    lo = np.searchsorted(event_indices, pred_start, side='left')
    hi = np.searchsorted(event_indices, w_candidates, side='left')
    pred_events_up_to_w = hi - lo

    passing = pred_events_up_to_w >= min_pred_events
    if not passing.any():
        # Best count seen at ANY window size tried, not just at the cap --
        # see this function's own docstring for why the two can differ.
        return None, int(pred_events_up_to_w.max())
    first = int(np.argmax(passing))
    return int(w_candidates[first]), int(pred_events_up_to_w[first])


def _rho_min_for_metric(rho_min, node_metric):
    '''One skill gate resolved for the node actually being explained.

    `rho_min` is either a plain float -- the historical behaviour, one gate
    for every node -- or a {metric: float} mapping, in which case the entry
    for `node_metric` wins and 'default' covers anything unlisted.

    WHY A MAPPING AT ALL. EMBED_DIM_RHO_MIN/CROSS_MAP_RHO_MIN were a single
    number applied to two metrics that do not share a scale. A behavioural
    target is scored by `correlation`, whose ceiling is 1.0. A TSLS neuron
    node is scored by `f1`, and the r9 calibration
    (scripts/calibration/r9_min_pred_events_curve.csv) measured what a
    NEAR-PERFECT predictor of a TSLS channel actually scores: f1 ~ 0.49-0.54
    on average, never above 0.69. So one 0.2 bar asked a neuron for ~38% of
    achievable skill and a behavioural target for 20% of it -- a far harsher
    test, which is part of why 101 of the nodes in job 4774136 had no
    candidate clear the gate at all.'''
    if isinstance(rho_min, dict):
        return float(rho_min.get(node_metric, rho_min.get('default', 0.2)))
    return float(rho_min)

def _preview_window_rows(node, full_df, delay_set, prediction_start, prediction_length, tp,
                          is_behavioral_target, target_ranking_window_rows,
                          min_pred_events, max_window_rows):
    '''Speculative, read-only resolve of one node's own expected ranking-
    window size, called from the progress-monitoring log's own ETA
    estimate (see _format_eta()'s per-window-row mode) BEFORE that node's
    real run_mde_for_target() call -- independent of, and without any
    effect on, the real resolve that call performs later. Node cost scales
    roughly with resolved window size (see _resolve_target_ranking_window()
    itself), so this gives the monitoring log a real cost signal per
    pending node instead of treating every node as equally expensive.

    Mirrors run_mde_for_target()'s own usable_df/lib_ratio derivation
    exactly (see that function's own LEAKAGE docstring), narrowed to just
    `node`'s own column -- cheap relative to the real MDE call it is
    estimating (a single-column slice plus one vectorized event-count
    search, not the full candidate-pool ranking sweep). Returns 0 (not a
    real window size) if the node turns out unreachable -- an unreachable
    node fails fast in the real call regardless, so weighting it as
    negligible cost for the ETA is reasonable, not misleading.

    Behavioral targets short-circuit before touching `full_df` at all:
    `_resolve_target_ranking_window()`'s own behavioral branch never reads
    the target's column either (the window is the fixed
    `target_ranking_window_rows`, capped by available rows) -- mirroring
    that here avoids requiring `full_df` to even contain this node's own
    column for a case where the real resolve never looks at it.'''
    warmup = max((abs(d) for d in delay_set), default=0)
    usable_end = prediction_start + prediction_length - tp
    n_usable = usable_end - warmup
    if is_behavioral_target:
        return min(target_ranking_window_rows, n_usable)
    usable_df = full_df[[node]].iloc[warmup:usable_end].reset_index(drop=True)
    lib_ratio = (prediction_start - warmup) / max(1, n_usable)
    window_rows, _ = _resolve_target_ranking_window(
        node, full_df, usable_df, lib_ratio,
        is_behavioral_target, target_ranking_window_rows, min_pred_events, max_window_rows,
    )
    return window_rows or 0

def _metric_for(column, targets, metric, target_metric):
    '''TARGETS (continuous behavioral traces) always score via
    target_metric ('correlation'); everything else (TSLS-converted neuron
    columns) uses metric ('f1' by default) -- see load_config()'s
    TARGET_METRIC docstring for why this split is required, not cosmetic.'''
    return target_metric if column in targets else metric
