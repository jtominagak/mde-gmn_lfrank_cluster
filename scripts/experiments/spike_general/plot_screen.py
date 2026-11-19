"""Figures and Markdown tables for Stage I (screen_representations.py output).

Usage: python plot_screen.py <screen_dir> <report_dir>
Writes <report_dir>/figures/stage1_*.png and <report_dir>/stage1_table.md.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Fixed method -> colour (reference categorical order; never re-assigned by rank).
METHOD_COLOR = {'sawtooth': '#2a78d6', 'gamma': '#eb6834', 'lmu': '#1baf7a', 'logsig': '#eda100'}
METHOD_LABEL = {'sawtooth': 'sawtooth (TSLS)', 'gamma': 'M1 Gamma', 'lmu': 'M2 Legendre', 'logsig': 'M3 log-signature'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def _style(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def scatter(d, x, y, xl, yl, path, title, logx=False):
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for m, g in d.groupby('method'):
        ax.scatter(g[x], g[y], s=46, c=METHOD_COLOR[m], label=METHOD_LABEL[m],
                   edgecolors='#fcfcfb', linewidths=1.5, zorder=3)
    best = d.loc[d.groupby('method')[y].idxmax()]
    for _, r in best.iterrows():
        ax.annotate(r['name'], (r[x], r[y]), fontsize=7, color=INK2,
                    xytext=(4, 4), textcoords='offset points')
    if logx:
        ax.set_xscale('log')
    ax.set_xlabel(xl, color=INK, fontsize=9)
    ax.set_ylabel(yl, color=INK, fontsize=9)
    ax.set_title(title, color=INK, fontsize=10, loc='left')
    _style(ax)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def traces(screen_dir, names, path, dt=0.025, seconds=20):
    raw = np.load(screen_dir / 'traces' / '_raw_counts.npy')
    n = int(seconds / dt)
    t = np.arange(n) * dt
    ch = int(np.argmax((raw[:n] > 0).sum(0) >= 4)) if ((raw[:n] > 0).sum(0) >= 4).any() else 0
    fig, axes = plt.subplots(len(names) + 1, 1, figsize=(10, 1.25 * (len(names) + 1)), sharex=True)
    ev = np.flatnonzero(raw[:n, ch] > 0) * dt
    axes[0].vlines(ev, 0, 1, color=INK, lw=0.8)
    axes[0].set_ylabel('events', fontsize=7, rotation=0, ha='right', color=INK2)
    axes[0].set_yticks([])
    for ax, name in zip(axes[1:], names):
        x = np.load(screen_dir / 'traces' / f'{name}.npy')[:n, ch]
        m = name.split('_')[0]
        ax.plot(t, x, color=METHOD_COLOR.get(m, INK), lw=1.2)
        for e in ev:
            ax.axvline(e, color=GRID, lw=0.6, zorder=0)
        ax.set_ylabel(name, fontsize=7, rotation=0, ha='right', color=INK2)
        ax.set_yticks([])
    for ax in axes:
        for sp in ('top', 'right', 'left'):
            ax.spines[sp].set_visible(False)
    axes[-1].set_xlabel('seconds (start of the MDE validation segment)', fontsize=8)
    fig.suptitle('One channel through each operator (grey lines = its events)', fontsize=10,
                 x=0.01, ha='left')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    screen_dir, report = Path(sys.argv[1]), Path(sys.argv[2])
    figs = report / 'figures'
    figs.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(screen_dir / 'screen.csv')
    d = d[d['method'].notna()]
    scatter(d, 'roughness', 'joint6_r', 'roughness  sd(2nd diff) / sd(1st diff)   (log; lower = smoother)',
            'joint (x, y) readout r, 6 channels', figs / 'stage1_smooth_vs_decode.png',
            'Smoothness does not buy decodability', logx=True)
    scatter(d, 'collapse_frac', 'joint6_r', 'fraction of time on the channel floor (median channel)',
            'joint (x, y) readout r, 6 channels', figs / 'stage1_collapse_vs_decode.png',
            'Silent-interval collapse vs decodability')
    scatter(d, 'xmap_top5', 'self_gain_vs_persist_1s', 'per-channel cross-map r to (x, y), top-5 mean',
            'self-prediction 1 s ahead: r(simplex) - r(persistence)', figs / 'stage1_xmap_vs_autonomy.png',
            'Decodability vs autonomous predictability')
    picks = ['sawtooth', 'gamma_m3_t100', 'gamma_m3_t500', 'gamma_m3_t250_log100',
             'lmu_th1000_D3', 'lmu_th1000_D3_pre2', 'logsig_W1000_K2', 'logsig_W1000_K3_pre3']
    picks = [p for p in picks if (screen_dir / 'traces' / f'{p}.npy').exists()]
    traces(screen_dir, picks, figs / 'stage1_example_traces.png')

    cols = ['name', 'embed_tau_rows', 'roughness', 'max_step', 'collapse_frac', 'degenerate_nn_frac',
            'xmap_top5', 'joint6_r', 'self_r_1s', 'self_gain_vs_persist_1s',
            'vector_xmap_top5', 'scalar_nodelay_xmap_top5']
    t = d[[c for c in cols if c in d.columns]].copy()
    t = t.sort_values(['joint6_r'], ascending=False)
    with open(report / 'stage1_table.md', 'w') as f:
        f.write('| ' + ' | '.join(t.columns) + ' |\n|' + '---|' * len(t.columns) + '\n')
        for _, r in t.iterrows():
            f.write('| ' + ' | '.join(
                (f'{v:.3f}' if isinstance(v, float) and np.isfinite(v) else
                 ('' if isinstance(v, float) else str(v))) for v in r.values) + ' |\n')
    print(f'wrote {report / "stage1_table.md"} and figures')


if __name__ == '__main__':
    main()
