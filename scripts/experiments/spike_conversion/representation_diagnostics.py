"""Representation diagnostics for exp/spike-conversion-neuro (no MDE, no GMN).

For every spike-transform configuration in the grid, on the real recording:

  * Takens lag sanity check: first minimum of the average mutual information
    I(x_t; x_{t+tau}) and the false-nearest-neighbour fraction at E=3 vs E=4,
    for tau in 1..12 rows, averaged over a sample of units. Answers "which of
    tau in {2,4,6,8} does each representation's own autocorrelation favour?"
  * Jumps: the largest single-bin change, as a fraction of the unit's range
    (the sawtooth resets in one bin; a continuous trace never does).
  * Silent-unit collapse: the fraction of library rows where a unit's value
    is within 1% of its range of the no-input rest state 0 (the state a
    filter decays to). Undefined for the sawtooth, which ramps instead.
  * Degenerate neighbours: on a 3-lag population delay vector of the first
    K units, the fraction of library rows whose nearest neighbour is at
    distance 0 (exact duplicates -- the "(0,0,...,0)" singularity).

Writes representation_diagnostics.csv and lag_diagnostics.csv, and figures
(traces_examples.png, ami_vs_tau.png) into --out.

Run on a compute node:
  srun -p intel -t 1:00:00 -c 4 --mem=24G .venv/bin/python3 \
      scripts/experiments/spike_conversion/representation_diagnostics.py \
      --data /bucket/FukaiU/tominaga/pao_rotation/data/J1620210605_epoch_8_freq_40_Hz.csv \
      --out docs/experiments/spike-conversion-neuro/figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / 'src'))
sys.path.insert(0, str(HERE.parent))

from grid import GRID, spike_params  # noqa: E402
from mde_gmn.spike_filters import transform_spikes  # noqa: E402

TAUS = list(range(1, 13))
N_UNITS_LAG = 40
N_UNITS_NN = 30
NN_ROWS = 15000


def ami(x, tau, bins=16):
    a, b = x[:-tau], x[tau:]
    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return np.nan
    ia = np.clip(np.searchsorted(edges, a, side='right') - 1, 0, len(edges) - 2)
    ib = np.clip(np.searchsorted(edges, b, side='right') - 1, 0, len(edges) - 2)
    k = len(edges) - 1
    joint = np.bincount(ia * k + ib, minlength=k * k).reshape(k, k).astype(float)
    joint /= joint.sum()
    pa, pb = joint.sum(1), joint.sum(0)
    nz = joint > 0
    return float((joint[nz] * np.log(joint[nz] / np.outer(pa, pb)[nz])).sum())


def first_minimum(vals):
    for i in range(1, len(vals) - 1):
        if vals[i] < vals[i - 1] and vals[i] <= vals[i + 1]:
            return i
    return int(np.nanargmin(vals))


def fnn_fraction(x, tau, E=3, rtol=10.0, rows=8000):
    n = len(x) - E * tau
    if n < 100:
        return np.nan
    idx = np.arange(n)[:rows]
    emb = np.column_stack([x[idx + k * tau] for k in range(E)])
    nxt = x[idx + E * tau]
    d, j = cKDTree(emb).query(emb, k=2)
    d, j = d[:, 1], j[:, 1]
    ok = d > 0
    if not ok.any():
        return np.nan
    return float(np.mean(np.abs(nxt[ok] - nxt[j[ok]]) / d[ok] > rtol))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--lib-ratio', type=float, default=0.7)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.data)
    units = [c for c in raw.columns if c.startswith(('CA1_', 'mPFC_', 'OFC_'))]
    t = raw['time_bin_center'].to_numpy()
    S = raw[units].to_numpy().astype(float)
    first = np.where((S > 0).any(0), (S > 0).argmax(0), len(t))
    trim = int(first.max())
    lib = slice(trim, trim + int(round((len(t) - trim) * args.lib_ratio)))
    rates = S[lib].sum(0) / (t[lib.stop - 1] - t[lib.start])
    rng = np.random.default_rng(0)
    order = np.argsort(rates)
    # stratified by rate so slow and fast units are both represented
    lag_units = order[np.linspace(0, len(order) - 1, N_UNITS_LAG).astype(int)]
    nn_units = np.sort(rng.choice(len(units), N_UNITS_NN, replace=False))

    configs = [g for g in GRID if g['stage'] == 'I']
    rep_rows, lag_rows, ami_curves, examples = [], [], {}, {}
    for g in configs:
        X, _ = transform_spikes(g['env']['SPIKE_TRANSFORM'], S, t, spike_params(g), lib)
        X = X[trim:]
        L = X[: lib.stop - trim]
        rngs = np.nanmax(L, 0) - np.nanmin(L, 0)
        rngs[rngs == 0] = np.nan
        jump = np.nanmax(np.abs(np.diff(L, axis=0)), 0) / rngs
        # Collapse = sitting at the no-input REST state, which is 0 for every
        # filter. The sawtooth has no rest state (silence makes it ramp), so
        # the measure is undefined for it.
        if g['env']['SPIKE_TRANSFORM'] == 'tsls':
            collapsed = np.full(L.shape[1], np.nan)
        else:
            collapsed = np.nanmean(np.abs(L) <= 0.01 * rngs, axis=0)
        Z = (L[:, nn_units] - np.nanmean(L[:, nn_units], 0)) / np.where(
            np.nanstd(L[:, nn_units], 0) > 0, np.nanstd(L[:, nn_units], 0), 1)
        emb = np.column_stack([Z[8:], Z[4:-4], Z[:-8]])[-NN_ROWS:]
        d = cKDTree(emb).query(emb, k=2)[0][:, 1]
        rep_rows.append({'label': g['label'], 'family': g['family'],
                         'max_jump_frac_median': float(np.nanmedian(jump)),
                         'collapsed_frac_median': float(np.nanmedian(collapsed)),
                         'collapsed_frac_slowest_quartile': float(np.nanmedian(
                             collapsed[rates <= np.quantile(rates, 0.25)])),
                         'nn_duplicate_frac': float(np.mean(d == 0)),
                         'nn_dist_median': float(np.median(d))})
        curves = []
        for j in lag_units:
            x = L[:, j]
            curves.append([ami(x, k) for k in TAUS])
            fnn = {k: fnn_fraction(x, k) for k in (2, 4, 6, 8)}
            lag_rows.append({'label': g['label'], 'unit': units[j], 'rate_hz': rates[j],
                             'ami_first_min_tau': TAUS[first_minimum(curves[-1])],
                             **{f'fnn_E3_tau{k}': v for k, v in fnn.items()}})
        ami_curves[g['label']] = np.nanmean(np.array(curves), 0)
        examples[g['label']] = X[2000:2400, order[len(order) // 2]]
        print(f"{g['label']}: done", flush=True)

    rep = pd.DataFrame(rep_rows)
    lag = pd.DataFrame(lag_rows)
    rep.to_csv(out / 'representation_diagnostics.csv', index=False)
    lag.to_csv(out / 'lag_diagnostics.csv', index=False)
    summ = lag.groupby('label').agg(
        ami_first_min_tau_median=('ami_first_min_tau', 'median'),
        **{f'fnn_tau{k}_median': (f'fnn_E3_tau{k}', 'median') for k in (2, 4, 6, 8)}).reset_index()
    summ.to_csv(out / 'lag_diagnostics_summary.csv', index=False)
    print(rep.to_string())
    print(summ.to_string())

    fig, ax = plt.subplots(figsize=(7, 4))
    for lab, c in ami_curves.items():
        ax.plot(TAUS, c / c[0], lw=1, label=lab)
    for k in (2, 4, 6, 8):
        ax.axvline(k, color='gray', lw=0.5, ls=':')
    ax.set_xlabel('lag tau (rows, 25 ms)')
    ax.set_ylabel('AMI(tau) / AMI(1), mean over 40 units')
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'ami_vs_tau.png', dpi=200)
    plt.close(fig)

    fams = {}
    for g in configs:
        fams.setdefault(g['family'], g['label'])
    fig, axes = plt.subplots(len(fams), 1, figsize=(9, 1.8 * len(fams)), sharex=True)
    tt = np.arange(400) * 0.025
    for ax, (fam, lab) in zip(axes, fams.items()):
        ax.plot(tt, examples[lab], lw=1)
        ax.set_title(f'{lab}  (unit {units[order[len(order) // 2]]}, median rate)', fontsize=8)
    axes[-1].set_xlabel('time (s)')
    fig.tight_layout()
    fig.savefig(out / 'traces_examples.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
