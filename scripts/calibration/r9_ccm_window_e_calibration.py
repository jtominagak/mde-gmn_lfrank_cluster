'''R9 (docs/review-actions-v1.md): calibration harness testing whether
CCM_WINDOW_ROWS and CCM_PREFILTER_E generalize beyond the small hand-picked
set of pairs Section 18.7 calibrated against.

Samples >=40 (observable, target) pairs spanning all three regions
(CA1/mPFC/OFC), stratified by firing-rate tercile (low/medium/high) within
each region, then:
  1. Computes the CCM skill-vs-library-size curve at each
     CCM_WINDOW_ROWS in {200, 400, 800, 1600, 3200}, at the pipeline's
     current CCM_PREFILTER_E default -- reports, per window, the fraction
     of pairs whose curve is still rising at the smallest tested library
     size vs. already flat.
  2. Sweeps CCM_PREFILTER_E in {3, 5, 7, 9, 11} across the same pairs, at
     the pipeline's current CCM_WINDOW_ROWS default -- reports, per E, the
     resulting kept-candidate set and its stability against the pipeline's
     current default (E=7).
  3. MIN_PRED_EVENTS calibration (extension, found while verifying R22 --
     see docs/review-findings.md's own window-sizing section): estimates
     the noise floor of `f1` (this pipeline's own TSLS event-timing
     metric) as a function of how many real reset events fall in the
     scored window, by comparing a real TSLS slice against a lightly
     perturbed copy of itself (a stand-in for a near-perfect predictor)
     across many independent draws, at each of several target event
     counts, stratified by firing-rate tercile. Reports the variance
     curve and proposes a MIN_PRED_EVENTS default from where it
     stabilizes -- this is what run_mde_for_target()'s own per-node
     ranking window (`_resolve_target_ranking_window()`) is now grown
     until it contains, replacing an earlier, since-abandoned
     "fraction of a neuron's own total event count" placeholder (found,
     during real-data testing, to not actually guarantee the *scored*
     portion of the window has enough events -- see the findings for the
     full story).

Changes no pipeline default -- this is a standalone, read-only
measurement script; run against real data (data/J1620210605_epoch_8_freq_
40_Hz.csv) via:

    python3 scripts/calibration/r9_ccm_window_e_calibration.py

Writes r9_summary.csv (one row per window/E setting), r9_raw_curves.csv
(one row per pair per window per library size), and
r9_min_pred_events_curve.csv (one row per channel per event-count target,
mean/std of f1 across independent noise draws) into this same
scripts/calibration/ directory. See docs/review-findings.md's R9 section
(window-sizing subsection) for the narrative results.
'''
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SRC_DIR = REPO_ROOT / 'scripts' / 'src'
OUT_DIR = Path(__file__).resolve().parent
DATA_FILE = REPO_ROOT / 'data' / 'J1620210605_epoch_8_freq_40_Hz.csv'

sys.path.insert(0, str(SRC_DIR))
import cleaned_helper_functions as tsls  # noqa: E402

_spec = importlib.util.spec_from_file_location('tsls_pipeline_r9', SRC_DIR / 'tsls-mde-gmn.py')
pipeline = importlib.util.module_from_spec(_spec)
sys.modules['tsls_pipeline_r9'] = pipeline
_spec.loader.exec_module(pipeline)

# ---- Fixed calibration settings (this harness's own, not pipeline config) --
TARGETS = ['head_position_x', 'head_position_y']
REGIONS = ['CA1_', 'mPFC_', 'OFC_']
N_PER_REGION_TERCILE = 3          # 3 regions x 3 terciles x 3 = 27 observables
RNG_SEED = 0
SAMPLE_DRAWS = 10                 # this harness's own CCM sample count -- see
                                   # docs/review-findings.md's R9 section for why
                                   # (R8's own finding: sample counts well below
                                   # the pipeline's CCM_SAMPLE default already
                                   # reproduce stable results at real-data scale)
LIBSIZE_RATIOS = (0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9)  # matches load_config()'s own default
WINDOW_SIZES = (200, 400, 800, 1600, 3200)
E_VALUES = (3, 5, 7, 9, 11)
CURRENT_CCM_PREFILTER_E = 7        # pipeline's current default, used for the window sweep
CURRENT_CCM_WINDOW_ROWS = 800      # pipeline's current default, used for the E sweep
CCM_PREFILTER_RHO_MIN = 0.2        # pipeline's current default
METRIC = 'correlation'

# ---- MIN_PRED_EVENTS calibration settings (item 3, see module docstring) --
MIN_PRED_EVENTS_TERCILE_N = 5       # >= 5 channels per (pooled, not per-region)
                                     # firing-rate tercile, or all if fewer
PRED_EVENT_COUNTS = (5, 10, 20, 40, 80, 160)
N_NOISE_DRAWS = 200                  # independent random draws per (channel, event-count)
NOISE_STD_FRACTION = 0.05            # "lightly perturbed": Gaussian noise at 5% of the
                                      # real slice's own std -- a stand-in for a
                                      # near-perfect predictor with small timing error
GAMMA_DELTA_FOR_F1 = 2                # matches the pipeline's own GAMMA_DELTA default


def stratified_sample(df_raw, rng):
    '''Returns a list of raw channel names: N_PER_REGION_TERCILE channels
    from each (region, firing-rate tercile) bucket, i.e. up to
    3 regions x 3 terciles x N_PER_REGION_TERCILE observables.'''
    chosen = []
    for prefix in REGIONS:
        region_cols = [c for c in df_raw.columns if c.startswith(prefix)
                       and c[len(prefix):].isdigit()]
        counts = (df_raw[region_cols] > 0).sum().sort_values()
        n = len(counts)
        low_cut, high_cut = counts.iloc[n // 3], counts.iloc[(2 * n) // 3]
        low = counts[counts <= low_cut].index.tolist()
        high = counts[counts >= high_cut].index.tolist()
        mid = counts[(counts > low_cut) & (counts < high_cut)].index.tolist()
        for bucket in (low, mid, high):
            pick = list(rng.choice(bucket, size=min(N_PER_REGION_TERCILE, len(bucket)), replace=False))
            chosen.extend(pick)
    return chosen


def curve_still_rising(rho_by_libsize, libsizes_sorted):
    '''"Still rising at the smallest tested library size" -- the skill at
    the two smallest library sizes increased by more than a small noise
    epsilon (0.01), i.e. the curve had not yet plateaued even at the
    smallest size tested. NaN-safe: a pair with an all-NaN curve is
    excluded from both the rising and flat counts (reported separately).'''
    vals = [rho_by_libsize[ls] for ls in libsizes_sorted]
    if len(vals) < 2 or any(np.isnan(v) for v in vals[:2]):
        return None
    return (vals[1] - vals[0]) > 0.01


def compute_curve(windowed, observable, target, e, sample):
    lib_sizes = sorted({max(10, round(r * len(windowed))) for r in LIBSIZE_RATIOS})
    try:
        c2t, _, _ = tsls.proper_ccm(
            windowed, libSizes=lib_sizes, sample=sample, tau=-1, Tp=1, E=e,
            target=target, columns=observable,
            target_measurement=METRIC, columns_measurement=METRIC,
            exclusionRadius=0, gammaDelta=2,
        )
        rho_curve = np.nanmean(c2t, axis=0)
    except Exception:
        rho_curve = np.full(len(lib_sizes), np.nan)
    return dict(zip(lib_sizes, rho_curve)), lib_sizes


def min_pred_events_calibration(df, obs_cols, rng):
    '''Item 3 (see module docstring): estimates f1's own noise floor as a
    function of how many real reset events fall in the scored window.

    For each of several target event counts, takes the smallest leading
    slice of a channel's own real TSLS series containing that many events,
    perturbs a copy of it with light Gaussian noise (a stand-in for a
    near-perfect predictor -- NOT a random guess; the point is to isolate
    f1's own sampling noise from genuine prediction failure), and scores
    `_score(real, noisy, 'f1')` -- the exact same call
    run_mde_for_target()'s own ranking stage makes -- across
    N_NOISE_DRAWS independent draws. Reports the resulting f1 mean/std per
    channel per event count, pooled and per firing-rate tercile (channels
    pooled across ALL three regions into one overall tercile split here,
    not the per-region terciles `stratified_sample()` above uses -- this
    question is about a channel's own absolute firing rate, not region
    identity).

    `obs_cols` is the already-TSLS-converted channel pool `main()` itself
    built (via convert_to_tsls/burn_in_trim) for the window/E sweeps above
    -- reused here rather than converting a separate channel set, since it
    already spans a wide firing-rate range (27 channels, 3 per
    region-tercile bucket).'''
    total_events = {}
    for col in obs_cols:
        vals = df[col].to_numpy(dtype=float)
        thr = tsls.infer_near_zero_threshold_from_true_tsls(vals)
        events = tsls.detect_reset_events_from_tsls(vals, near_zero_threshold=thr, drop_threshold=0.0)
        total_events[col] = len(events)

    counts_series = pd.Series(total_events).sort_values()
    n = len(counts_series)
    low_cut, high_cut = counts_series.iloc[n // 3], counts_series.iloc[(2 * n) // 3]
    tercile_of = {
        col: ('low' if cnt <= low_cut else 'high' if cnt >= high_cut else 'mid')
        for col, cnt in total_events.items()
    }
    tercile_channels = {'low': [], 'mid': [], 'high': []}
    for col in obs_cols:
        tercile_channels[tercile_of[col]].append(col)
    for tercile, pool in list(tercile_channels.items()):
        pick_n = min(MIN_PRED_EVENTS_TERCILE_N, len(pool))
        tercile_channels[tercile] = (
            sorted(rng.choice(pool, size=pick_n, replace=False)) if pool else []
        )
    print(f'\nMIN_PRED_EVENTS calibration: tercile channels = {tercile_channels}', flush=True)

    rows = []
    for tercile, channels in tercile_channels.items():
        for col in channels:
            vals = df[col].to_numpy(dtype=float)
            thr = tsls.infer_near_zero_threshold_from_true_tsls(vals)
            event_idx = tsls.detect_reset_events_from_tsls(
                vals, near_zero_threshold=thr, drop_threshold=0.0)
            for target_events in PRED_EVENT_COUNTS:
                if len(event_idx) < target_events:
                    continue  # this channel never reaches this many events at all
                end = int(event_idx[target_events - 1]) + 1
                real_slice = vals[:end]
                std = real_slice.std()
                noise_scale = NOISE_STD_FRACTION * (std if std > 0 else 1.0)
                f1_draws = []
                for _ in range(N_NOISE_DRAWS):
                    noisy = np.clip(real_slice + rng.normal(scale=noise_scale, size=real_slice.shape), 0, None)
                    f1 = tsls._score(real_slice, noisy, 'f1', gammaDelta=GAMMA_DELTA_FOR_F1)
                    if not (isinstance(f1, float) and np.isnan(f1)):
                        f1_draws.append(f1)
                if not f1_draws:
                    continue
                rows.append({
                    'tercile': tercile, 'channel': col, 'target_events': target_events,
                    'slice_rows': end, 'n_draws': len(f1_draws),
                    'f1_mean': float(np.mean(f1_draws)), 'f1_std': float(np.std(f1_draws)),
                })
                print(f'  {tercile}/{col}: target_events={target_events} slice_rows={end} '
                      f'f1 mean={np.mean(f1_draws):.3f} std={np.std(f1_draws):.3f} (n={len(f1_draws)})',
                      flush=True)

    curve_df = pd.DataFrame(rows)
    curve_df.to_csv(OUT_DIR / 'r9_min_pred_events_curve.csv', index=False)

    print('\nMIN_PRED_EVENTS: pooled f1_std by target event count (lower = f1 more reliable):',
          flush=True)
    for target_events in PRED_EVENT_COUNTS:
        subset = curve_df[curve_df['target_events'] == target_events]
        if len(subset):
            print(f'  target_events={target_events:4d}: pooled f1_std mean={subset["f1_std"].mean():.4f} '
                  f'(over {len(subset)} channels)', flush=True)

    print('\nMIN_PRED_EVENTS: per-tercile f1_std by target event count:', flush=True)
    for tercile in ('low', 'mid', 'high'):
        for target_events in PRED_EVENT_COUNTS:
            subset = curve_df[(curve_df['tercile'] == tercile) & (curve_df['target_events'] == target_events)]
            if len(subset):
                print(f'  {tercile}/target_events={target_events:4d}: '
                      f'f1_std mean={subset["f1_std"].mean():.4f} (over {len(subset)} channels)', flush=True)

    return curve_df


def main():
    print(f'Loading {DATA_FILE.name}...', flush=True)
    df_raw = pd.read_csv(DATA_FILE)
    rng = np.random.default_rng(RNG_SEED)

    neuron_columns = stratified_sample(df_raw, rng)
    print(f'Sampled {len(neuron_columns)} observables across {len(REGIONS)} regions x 3 terciles: '
          f'{neuron_columns}', flush=True)

    tsls_df_full, obs_cols = pipeline.convert_to_tsls(df_raw, neuron_columns, TARGETS, OUT_DIR)
    df = pipeline.burn_in_trim(tsls_df_full, obs_cols, OUT_DIR)
    (OUT_DIR / 'tsls_data_full.csv').unlink(missing_ok=True)
    (OUT_DIR / 'tsls_data_trimmed.csv').unlink(missing_ok=True)

    pairs = [(obs, t) for obs in obs_cols for t in TARGETS]
    print(f'{len(pairs)} (observable, target) pairs (>= 40 required by the task).', flush=True)
    assert len(pairs) >= 40, f'Only {len(pairs)} pairs sampled -- increase N_PER_REGION_TERCILE.'

    usable_df = df.reset_index(drop=True)

    # ---- Item 3: MIN_PRED_EVENTS calibration (window-sizing fix) ----------
    min_pred_events_calibration(usable_df, obs_cols, rng)

    raw_rows = []
    summary_rows = []

    # ---- 1/2/4a: window sweep, at the pipeline's current CCM_PREFILTER_E ----
    for window in WINDOW_SIZES:
        t0 = time.perf_counter()
        windowed, _ = pipeline._ccm_window(usable_df, 0.5, window)
        n_rising, n_flat, n_excluded = 0, 0, 0
        for observable, target in pairs:
            rho_by_libsize, lib_sizes = compute_curve(
                windowed, observable, target, CURRENT_CCM_PREFILTER_E, SAMPLE_DRAWS)
            for ls in lib_sizes:
                raw_rows.append({'sweep': 'window', 'setting': window, 'observable': observable,
                                  'target': target, 'libsize': ls, 'rho': rho_by_libsize[ls]})
            rising = curve_still_rising(rho_by_libsize, lib_sizes)
            if rising is None:
                n_excluded += 1
            elif rising:
                n_rising += 1
            else:
                n_flat += 1
        elapsed = time.perf_counter() - t0
        n_valid = n_rising + n_flat
        frac_rising = n_rising / n_valid if n_valid else float('nan')
        print(f'window={window}: {elapsed:.1f}s -- still-rising {n_rising}/{n_valid} '
              f'({frac_rising:.0%}), flat {n_flat}/{n_valid}, excluded(all-NaN) {n_excluded}',
              flush=True)
        summary_rows.append({'sweep': 'window', 'setting': window, 'n_pairs': len(pairs),
                              'n_rising': n_rising, 'n_flat': n_flat, 'n_excluded': n_excluded,
                              'frac_rising': frac_rising, 'elapsed_s': elapsed})

    # ---- 3/4b: E sweep, at the pipeline's current CCM_WINDOW_ROWS ----
    windowed, _ = pipeline._ccm_window(usable_df, 0.5, CURRENT_CCM_WINDOW_ROWS)
    kept_by_e = {}
    for e in E_VALUES:
        t0 = time.perf_counter()
        kept = set()
        for observable, target in pairs:
            rho_by_libsize, lib_sizes = compute_curve(windowed, observable, target, e, SAMPLE_DRAWS)
            mean_rho = float(np.nanmean(list(rho_by_libsize.values())))
            if mean_rho >= CCM_PREFILTER_RHO_MIN:
                kept.add((observable, target))
        kept_by_e[e] = kept
        elapsed = time.perf_counter() - t0
        print(f'E={e}: {elapsed:.1f}s -- kept {len(kept)}/{len(pairs)} pairs', flush=True)
        summary_rows.append({'sweep': 'E', 'setting': e, 'n_pairs': len(pairs),
                              'n_kept': len(kept), 'elapsed_s': elapsed})

    baseline = kept_by_e[CURRENT_CCM_PREFILTER_E]
    for e in E_VALUES:
        overlap = len(kept_by_e[e] & baseline)
        union = len(kept_by_e[e] | baseline)
        jaccard = overlap / union if union else float('nan')
        only_e = kept_by_e[e] - baseline
        only_baseline = baseline - kept_by_e[e]
        print(f'  E={e} vs baseline E={CURRENT_CCM_PREFILTER_E}: jaccard={jaccard:.2f}, '
              f'gained={sorted(only_e)}, lost={sorted(only_baseline)}', flush=True)
        for row in summary_rows:
            if row['sweep'] == 'E' and row['setting'] == e:
                row['jaccard_vs_e7'] = jaccard
                row['gained_vs_e7'] = len(only_e)
                row['lost_vs_e7'] = len(only_baseline)

    pd.DataFrame(summary_rows).to_csv(OUT_DIR / 'r9_summary.csv', index=False)
    pd.DataFrame(raw_rows).to_csv(OUT_DIR / 'r9_raw_curves.csv', index=False)
    print(f'\nWrote {OUT_DIR / "r9_summary.csv"} and {OUT_DIR / "r9_raw_curves.csv"}', flush=True)


if __name__ == '__main__':
    main()
