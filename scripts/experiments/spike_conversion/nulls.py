"""Null baselines for the rollout ensemble (used by aggregate_report.py).

Two questions a raw rollout score cannot answer by itself:

1. NEURAL -- "does the network predict anything beyond the representation's
   own spike-free evolution?" Every representation has a deterministic
   no-new-spikes trajectory from its seed: the sawtooth keeps ramping at
   slope 1, the filters decay towards rest. A real unit that does not fire
   during the run follows exactly that trajectory, so a generated trace that
   merely reproduces it scores well without the network knowing anything.
   `free_evolution_null` recomputes it exactly: the same transform, same
   parameters and same library-fitted sigma, on the real spike train with
   every spike from the rollout's first generated row onwards removed. (Row
   s of a rollout is the first PREDICTED row; the run knows the recording up
   to s-1.)

2. BEHAVIOURAL -- "does the run track THIS trajectory, or just look like a
   typical one?" `mismatched_xy_corr` correlates each generated (x, y) run
   with the real trajectory from every OTHER start. Its mean is what the
   x/y correlation would be if the generated run had nothing to do with its
   own seed; the matched score is only informative above it.

Both are computed on the same rows and with the same formulas as
rollout_eval.score_rollout (Pearson r, per-unit NRMSE normalised by the
held-out sd).
"""

import numpy as np
import pandas as pd

from mde_gmn.spike_filters import transform_spikes

DT = 0.025


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.ptp(a[ok]) < 1e-9 or np.ptp(b[ok]) < 1e-9:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def free_evolution_null(raw, unit_cols, method, params, trim, lib_ratio, start, length):
    """No-new-spikes trajectory of `unit_cols` for trimmed rows
    [start, start + length). raw: the raw recording (spike counts)."""
    t = raw['time_bin_center'].to_numpy()
    s_raw = trim + start
    end = min(s_raw + length, len(t))
    S = raw[unit_cols].to_numpy(float)[:end].copy()
    S[s_raw:] = 0.0
    n_rows = len(t) - trim
    lib = slice(trim, trim + int(round(n_rows * lib_ratio)))
    # The library window lies wholly before the held-out starts, so zeroing
    # spikes after s_raw cannot move Model B's library-fitted sigma.
    vals, _ = transform_spikes(method, S, t[:end], params, lib)
    return vals[s_raw:end]


def neural_scores(real, gen, scale):
    """Per-run unit scores, generated (or null) vs real, both (T, U)."""
    corrs = [_corr(real[:, j], gen[:, j]) for j in range(real.shape[1])]
    rmse = np.sqrt(np.nanmean((gen - real) ** 2, axis=0))
    return float(np.nanmean(corrs)), float(np.nanmedian(rmse / scale))


def mismatched_xy_corr(gens, reals, horizon):
    """gens/reals: lists of (T, 2) arrays, index-aligned by start. Mean over
    every ordered pair i != j of the x/y-averaged correlation."""
    vals = []
    for i, g in enumerate(gens):
        for j, r in enumerate(reals):
            if i == j:
                continue
            h = min(horizon, len(g), len(r))
            vals.append(np.nanmean([_corr(r[:h, k], g[:h, k]) for k in (0, 1)]))
    return float(np.nanmean(vals)) if vals else np.nan


def label_nulls(label, method, params, job_dir, raw, rollout_files, horizons_rows,
                lib_ratio=0.7, targets=('head_position_x', 'head_position_y')):
    """One row per (start, horizon) with generated and null neural scores,
    plus per-horizon mismatched x/y correlation rows."""
    real_df = pd.read_csv(job_dir / 'tsls_data_trimmed.csv')
    trim = len(raw) - len(real_df)
    # The rollout starts are spread over [B, end - L] starting AT B, the
    # held-out segment's first row -- the segment score_rollout's NRMSE
    # scale is computed over.
    held_start = min(int(pd.read_csv(f, usecols=['row'], nrows=1)['row'].iloc[0]) for f in rollout_files)
    rows, gens_xy, reals_xy = [], [], []
    for f in rollout_files:
        gen = pd.read_csv(f)
        start = int(gen['row'].iloc[0])
        L = len(gen)
        units = [c for c in gen.columns
                 if c not in ('row', *targets) and '::' not in c and c in real_df.columns]
        raw_cols = [c[:-len('_tsls')] if c.endswith('_tsls') else c for c in units]
        real_u = real_df[units].to_numpy(float)[start:start + L]
        gen_u = gen[units].to_numpy(float)[:len(real_u)]
        scale = np.array(real_df[units].iloc[held_start:].std(), dtype=float)
        scale[scale == 0] = np.nan
        null_u = free_evolution_null(raw, raw_cols, method, params, trim, lib_ratio, start, L)
        for h in horizons_rows:
            hh = min(h, len(real_u))
            gc, gn = neural_scores(real_u[:hh], gen_u[:hh], scale)
            nc, nn = neural_scores(real_u[:hh], null_u[:hh], scale)
            beat = np.mean(np.sqrt(np.nanmean((gen_u[:hh] - real_u[:hh]) ** 2, 0))
                           < np.sqrt(np.nanmean((null_u[:hh] - real_u[:hh]) ** 2, 0)))
            rows.append({'label': label, 'start': start, 'horizon_s': round(h * DT, 3),
                         'n_units': len(units), 'unit_corr_mean': gc, 'unit_nrmse_median': gn,
                         'unit_corr_mean_null': nc, 'unit_nrmse_median_null': nn,
                         'frac_units_beating_null': float(beat)})
        if all(t in gen.columns for t in targets):
            gens_xy.append(gen[list(targets)].to_numpy(float))
            reals_xy.append(real_df[list(targets)].to_numpy(float)[start:start + L])
    mism = [{'label': label, 'horizon_s': round(h * DT, 3),
             'corr_xy_mismatched': mismatched_xy_corr(gens_xy, reals_xy, h)} for h in horizons_rows]
    return rows, mism
