'''CCM_SLOPE calibration: the slope distribution, split by target type.

WHY. CCM_SLOPE (default 0.01) is the convergence gate: a candidate is only
accepted as a driver if its cross-map skill keeps RISING as the library grows,
which is the signature of a causal link rather than a coincidence. One
threshold currently serves every node. But the pipeline already scores the two
kinds of node with two different metrics -- `f1` (event timing) for a TSLS
neuron node, `correlation` for a behavioural target -- so the slope those
scores produce is not on a comparable scale either, and one threshold cannot
be right for both.

In job 4774136 the gate rejected 53 nodes whose best candidate had RANKED but
failed to converge, out of 154 nodes that selected nothing at all. The ones
sampled by hand were marginal (mPFC_13_tsls's best candidate: slope 0.00761
against the 0.01 bar), which is what prompted this.

WHAT IT DOES. Parses the per-candidate CCM lines the pipeline already prints
into every run log -- no recompute, no cluster job, no data file:

    1-D Cross Map  target='CA1_100_tsls'  244 candidates
      EDim mPFC_9_tsls E=5 rho=0.4444
      CA1_100_tsls:mPFC_9_tsls slope 0.02155 candidate_events_in_frame=188
      1-D added 'mPFC_9_tsls'  (f1=0.4444)

and reports, separately for neuronal and behavioural targets:
  - the slope distribution, and its quantiles;
  - how many candidates a given threshold would admit (the sweep), so the bar
    can be placed against a known cost in admitted links;
  - the slope distribution of candidates that were ACTUALLY SELECTED, which is
    the empirical answer to "what does a real link look like here".

WHAT IT CANNOT DO, AND WHY. It cannot identify an optimum, and the reason is
worth stating plainly because the obvious analysis is circular: the slopes of
the candidates that were SELECTED are censored at the current threshold by
construction -- nothing below CCM_SLOPE can appear in that set, so its 5th
percentile recovers the existing bar (measured: 0.0113 neuronal, 0.0106
behavioural, against a 0.01 gate) no matter where the bar actually belongs.

A run log also cannot say how far BELOW the gate the rejected candidates fell:
the pipeline prints the ranked list only when something clears the ranking
gate, so the sub-threshold tail is absent from the logs entirely.

What this script therefore reports is the honest part -- the full distribution
of evaluated slopes per target type, and the admitted-fraction sweep, which
prices any proposed threshold in candidates admitted. Identifying an optimum
needs a NULL: the slope distribution for pairs known to be unrelated
(surrogate/shuffled candidates), which has to be computed, not parsed.

Usage (login node is fine -- it only reads text):
    python3 scripts/calibration/ccm_slope_distribution.py
    python3 scripts/calibration/ccm_slope_distribution.py --logs '<glob>' --out <dir>
'''
import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
DEFAULT_LOGS = '/home/j/j-tominaga/saion_slurm_logs/*.out'
CURRENT_SLOPE = 0.01

# Categorical slots 1 and 2 of the validated default palette, in fixed order
# (dataviz skill, references/palette.md). Light chart surface from the same
# file. Not hand-picked, and not cycled.
SERIES = {'neuronal': '#2a78d6', 'behavioural': '#eb6834'}
SURFACE = '#fcfcfb'
INK = '#1a1a19'
INK_MUTED = '#6b6b68'

RE_BLOCK = re.compile(r"^\s*(\d+)-D (?:Multi )?Cross Map\s+targets?=(.+?)\s+(\d+) candidates")
RE_SLOPE = re.compile(
    r"^\s+(?P<target>[A-Za-z0-9_,]+):(?P<candidate>\S+) slope (?P<slope>-?\d+(?:\.\d+)?(?:e-?\d+)?)"
    r" candidate_events_in_frame=(?P<events>\d+)")
RE_ADDED = re.compile(r"^\s*(\d+)-D added '(?P<candidate>[^']+)'\s+\((?P<metric>\w+)=")


def target_kind(target_field):
    '''Behavioural if the node being explained is a head-position channel.

    The slope line names its own target, comma-joined for a joint
    (multi-target) call, so this needs no block tracking.'''
    return 'behavioural' if 'head_position' in target_field else 'neuronal'


def parse_log(path):
    '''One row per (candidate, dimension) CCM evaluation in this log.'''
    rows = []
    pending = []          # slope rows of the current N-D block
    for line in path.open(errors='replace'):
        if RE_BLOCK.match(line):
            rows.extend(pending)
            pending = []
            continue
        m = RE_SLOPE.match(line)
        if m:
            pending.append({
                'log': path.name,
                'target': m['target'],
                'kind': target_kind(m['target']),
                'candidate': m['candidate'],
                'slope': float(m['slope']),
                'events_in_frame': int(m['events']),
                'selected': False,
            })
            continue
        m = RE_ADDED.match(line)
        if m:
            # The candidate added at this dimension passed every gate,
            # including the slope test. Anything else evaluated in the same
            # block did not win this round.
            for row in pending:
                if row['candidate'] == m['candidate']:
                    row['selected'] = True
            rows.extend(pending)
            pending = []
    rows.extend(pending)
    return rows


def sweep(slopes, thresholds):
    '''Fraction of candidates a threshold would admit, per threshold.'''
    slopes = np.asarray(slopes, dtype=float)
    return np.array([float((slopes >= t).mean()) if slopes.size else np.nan
                     for t in thresholds])


def plot(df, thresholds, out_path):
    fig, (ax_dist, ax_sweep) = plt.subplots(1, 2, figsize=(11, 4.2))
    fig.patch.set_facecolor(SURFACE)

    # NOT a percentile range: the 99th percentile of all evaluated slopes is
    # ~0.01, so clipping there would hide the entire admitted region -- every
    # candidate the gate accepts lies above it. Span the believed links too
    # (selected p95 ~0.157), and use a log density axis, because the spike of
    # exactly-zero slopes is ~10x taller than everything else and would
    # otherwise flatten both tails into the axis.
    lo, hi = -0.05, 0.08
    bins = np.linspace(lo, hi, 70)
    for kind, color in SERIES.items():
        slopes = df.loc[df['kind'] == kind, 'slope']
        if slopes.empty:
            continue
        # Density, not counts: the two groups differ by an order of magnitude
        # in n, and the question here is the SHAPE of each.
        ax_dist.hist(slopes, bins=bins, density=True, histtype='step',
                     linewidth=2, color=color,
                     label=f'{kind} (n={len(slopes):,})')
        ax_sweep.plot(thresholds, sweep(slopes, thresholds) * 100,
                      linewidth=2, color=color, label=f'{kind} (n={len(slopes):,})')

    for ax in (ax_dist, ax_sweep):
        ax.set_facecolor(SURFACE)
        ax.axvline(CURRENT_SLOPE, color=INK_MUTED, linestyle='--', linewidth=1)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['left', 'bottom']].set_color(INK_MUTED)
        ax.tick_params(colors=INK_MUTED, labelsize=9)
        ax.grid(axis='y', color='#e8e8e5', linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK)

    ax_dist.set_yscale('log')
    ax_dist.set_xlim(lo, hi)
    ax_dist.set_title('CCM slope distribution', fontsize=11, color=INK, loc='left')
    ax_dist.set_xlabel('slope (cross-map skill vs library size)', fontsize=9, color=INK_MUTED)
    ax_dist.set_ylabel('density (log scale)', fontsize=9, color=INK_MUTED)
    # The gate admits everything to the RIGHT of the line; shade it so the
    # panel reads as "what is kept" rather than just "where the line is".
    ax_dist.axvspan(CURRENT_SLOPE, hi, color='#000000', alpha=0.04, linewidth=0)
    ax_dist.annotate(f'CCM_SLOPE={CURRENT_SLOPE} -> admitted',
                     xy=(CURRENT_SLOPE, ax_dist.get_ylim()[1]),
                     xytext=(4, -12), textcoords='offset points',
                     fontsize=8, color=INK_MUTED)

    ax_sweep.set_title('Candidates admitted at each threshold', fontsize=11, color=INK, loc='left')
    ax_sweep.set_xlabel('candidate CCM_SLOPE threshold', fontsize=9, color=INK_MUTED)
    ax_sweep.set_ylabel('% of evaluated candidates admitted', fontsize=9, color=INK_MUTED)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--logs', default=DEFAULT_LOGS, help='glob of slurm .out logs to parse')
    ap.add_argument('--out', default=str(OUT_DIR), help='directory for the CSVs and the figure')
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(Path('/').glob(args.logs.lstrip('/')))
    if not paths:
        raise SystemExit(f'no logs matched {args.logs!r}')

    rows = []
    for path in paths:
        found = parse_log(path)
        print(f'{path.name}: {len(found):,} CCM candidate evaluation(s)', flush=True)
        rows.extend(found)
    if not rows:
        raise SystemExit('no CCM slope lines found -- are these pipeline run logs?')

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / 'ccm_slope_observations.csv', index=False)

    thresholds = np.round(np.arange(0.0, 0.0505, 0.0005), 5)
    summary_rows = []
    print(f'\n{"":14s} {"n":>8s} {"selected":>9s} {"p5":>9s} {"p25":>9s} {"median":>9s} '
          f'{"p75":>9s} {"p95":>9s} {"%>=0.01":>9s}')
    for kind in SERIES:
        part = df[df['kind'] == kind]
        if part.empty:
            continue
        chosen = part[part['selected']]
        q = np.percentile(part['slope'], [5, 25, 50, 75, 95])
        admitted = float((part['slope'] >= CURRENT_SLOPE).mean()) * 100
        print(f'{kind:14s} {len(part):8,d} {len(chosen):9,d} '
              f'{q[0]:9.5f} {q[1]:9.5f} {q[2]:9.5f} {q[3]:9.5f} {q[4]:9.5f} {admitted:8.1f}%')
        for t, frac in zip(thresholds, sweep(part['slope'], thresholds)):
            summary_rows.append({'kind': kind, 'threshold': t,
                                 'fraction_admitted': frac,
                                 'n_admitted': int(round(frac * len(part))),
                                 'n_evaluated': len(part)})
    pd.DataFrame(summary_rows).to_csv(out_dir / 'ccm_slope_threshold_sweep.csv', index=False)

    print('\nSlopes of candidates that were actually SELECTED -- CENSORED at CCM_SLOPE'
          f'={CURRENT_SLOPE} by construction, so these percentiles cannot locate the bar:')
    for kind in SERIES:
        chosen = df[(df['kind'] == kind) & df['selected']]
        if chosen.empty:
            print(f'  {kind:14s} none selected in these logs')
            continue
        q = np.percentile(chosen['slope'], [5, 50, 95])
        print(f'  {kind:14s} n={len(chosen):4d}   p5={q[0]:.5f}  median={q[1]:.5f}  p95={q[2]:.5f}')
    print('  (p5 landing just above the gate is the censoring, not evidence about the gate.)')

    print('\nWhat a lower bar would admit, priced in candidates:')
    for kind in SERIES:
        part = df[df['kind'] == kind]
        if part.empty:
            continue
        base = float((part['slope'] >= CURRENT_SLOPE).sum())
        for t in (0.005, 0.0025, 0.001):
            extra = float(((part['slope'] >= t) & (part['slope'] < CURRENT_SLOPE)).sum())
            print(f'  {kind:14s} threshold {t:<7.4f} admits {extra:7,.0f} more candidate(s) '
                  f'({extra / base * 100 if base else float("nan"):6.1f}% on top of the '
                  f'{base:,.0f} that clear {CURRENT_SLOPE})')

    plot(df, thresholds, out_dir / 'ccm_slope_distribution.png')
    print(f'\nWrote {out_dir / "ccm_slope_observations.csv"}, '
          f'{out_dir / "ccm_slope_threshold_sweep.csv"} and '
          f'{out_dir / "ccm_slope_distribution.png"}')


if __name__ == '__main__':
    main()
