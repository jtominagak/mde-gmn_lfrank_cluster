#!/usr/bin/env python3
'''Derive a DELAY_SET from the recording's own spike timing.

A lagged TSLS copy X(t-k) adds information to X(t) only when a spike fell in
(t-k, t]; otherwise X(t-k) = X(t) - k*dt exactly. That happens when
TSLS(t) < k rows, so the fraction of rows in which lag k is informative is the
CDF of the neuron's TSLS values, and the lag that is informative in a fraction
q of rows is the q-quantile of TSLS (in rows). This script reports those
quantiles, median over neurons, plus the ISI statistics behind them.

Uses library rows only (before PREDICTION_START), so the lag choice never
looks at data the pipeline later validates on.

Run on a compute node, never a login node:
    srun -p gpu-p100 -t 0:15:00 -c 2 --mem=16G .venv-gpu/bin/python3 \\
        scripts/dev/derive_isi_lags.py <tsls_data_trimmed.csv> <PREDICTION_START> \\
        [--columns-from <data_after_mde.csv>] [--q 0.1,0.25,0.5] [--dt 0.025]

tsls_data_trimmed.csv is written by every run into its results directory;
--columns-from restricts the neurons to a network's own (its data_after_mde
CSV), which is the population the lags will actually be applied to.
'''
import argparse

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('tsls_csv')
    ap.add_argument('prediction_start', type=int)
    ap.add_argument('--columns-from')
    ap.add_argument('--q', default='0.1,0.25,0.5')
    ap.add_argument('--dt', type=float, default=0.025)
    args = ap.parse_args()

    data = pd.read_csv(args.tsls_csv).iloc[:args.prediction_start]
    cols = [c for c in data.columns if c.endswith('_tsls')]
    if args.columns_from:
        keep = set(pd.read_csv(args.columns_from, nrows=1).columns)
        cols = [c for c in cols if c in keep]
    qs = [float(x) for x in args.q.split(',')]

    lags = {q: [] for q in qs}
    isi_median, rates = [], []
    for c in cols:
        x = data[c].to_numpy(float)
        resets = np.where(np.diff(x) < -1e-9)[0]
        if len(resets) < 5:
            continue
        isi_median.append(np.median(np.diff(resets)))
        rates.append(len(resets) / (len(x) * args.dt))
        rows = np.round(x / args.dt)
        for q in qs:
            lags[q].append(np.quantile(rows, q))

    print(f'{len(rates)} neurons with >= 5 spikes in rows [0, {args.prediction_start})')
    print(f'firing rate (Hz): median {np.median(rates):.2f}, '
          f'IQR {np.quantile(rates, .25):.2f}-{np.quantile(rates, .75):.2f}')
    print(f'per-neuron median ISI (rows): median {np.median(isi_median):.0f}')
    derived = []
    for q in qs:
        v = np.array(lags[q])
        k = int(round(np.median(v)))
        derived.append(-k)
        print(f'q={q:.2f}: lag {k} rows ({k * args.dt * 1000:.0f} ms), '
              f'IQR over neurons {np.quantile(v, .25):.0f}-{np.quantile(v, .75):.0f}')
    print('DELAY_SET=' + ','.join(str(d) for d in derived))


if __name__ == '__main__':
    main()
