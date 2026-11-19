"""Render REPORT.md's result tables from aggregate_report.py's CSVs.

  python3 report_tables.py AGG_DIR > tables.md

Every number in the report's result tables comes from here, so re-running
the aggregation and this script regenerates them exactly.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grid import GRID, STAGE2_BASES  # noqa: E402

D = Path(sys.argv[1])
FAMILY = {g['label']: g['family'] for g in GRID}
ORDER = [g['label'] for g in GRID if g['stage'] == 'I']


def md(df):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join(str(r[c]) for c in cols) + ' |')
    return '\n'.join(out)


def f2(v):
    return '' if pd.isna(v) else f'{v:.2f}'


def pm(m, s):
    return f'{m:.2f} ± {s:.2f}'


r = pd.read_csv(D / 'rollout_summary.csv')
mm = pd.read_csv(D / 'xy_mismatched_null.csv')
st = pd.read_csv(D / 'stall_summary.csv')
one = pd.read_csv(D / 'onestep_summary.csv')
nn = pd.read_csv(D / 'neural_null_summary.csv')
gr = pd.read_csv(D / 'graph_summary.csv')


def R(lab, h, c):
    v = r[(r.label == lab) & (r.horizon_s == h)]
    return v[c].iloc[0] if len(v) else np.nan


# ---- Table 1: free-run behaviour, Stage I --------------------------------
rows = []
for lab in ORDER:
    if lab not in set(r.label):
        continue
    m5 = mm[(mm.label == lab) & (mm.horizon_s == 5.0)]['corr_xy_mismatched']
    s = st[st.label == lab]
    rows.append({
        'label': f'`{lab}`',
        'r(x,y) 2 s': pm(R(lab, 2.0, 'corr_xy_mean'), R(lab, 2.0, 'corr_xy_se')),
        'r(x,y) 5 s': pm(R(lab, 5.0, 'corr_xy_mean'), R(lab, 5.0, 'corr_xy_se')),
        'r(x,y) 10 s': pm(R(lab, 10.0, 'corr_xy_mean'), R(lab, 10.0, 'corr_xy_se')),
        'null 5 s': f2(m5.iloc[0]) if len(m5) else '',
        'path err 5 s (median)': f'{R(lab, 5.0, "path_err_mean_median"):.1f}',
        'path err 10 s (median)': f'{R(lab, 10.0, "path_err_mean_median"):.1f}',
        'blew up 10 s': f'{100 * R(lab, 10.0, "blew_up_mean"):.0f}%',
        'stall (median s)': f'{s.stall_s_median.iloc[0]:.1f}' if len(s) else '',
    })
hold5 = R('tsls', 5.0, 'path_err_mean_hold_median')
hold10 = R('tsls', 10.0, 'path_err_mean_hold_median')
print('### Table 1 -- free-run behaviour (Stage I, tau = 4)\n')
print(md(pd.DataFrame(rows)))
print(f'\nHold-the-seed median path error: {hold5:.1f} at 5 s, {hold10:.1f} at 10 s '
      '(identical for every label: same start rows).\n')

# ---- Table 2: one-step ----------------------------------------------------
rows = []
for lab in ORDER:
    t = one[(one.label == lab) & (one.role == 'target')]
    u = one[(one.label == lab) & (one.role == 'unit')]
    if not len(t):
        continue
    rows.append({'label': f'`{lab}`',
                 'x/y r': f'{t.corr_median.iloc[0]:.3f}',
                 'x/y NRMSE': f'{t.nrmse_median.iloc[0]:.3f}',
                 'persistence NRMSE': f'{t.nrmse_persistence_median.iloc[0]:.3f}',
                 'x/y skill': f2(t.skill_vs_persistence_median.iloc[0]),
                 'unit r (median)': f'{u.corr_median.iloc[0]:.3f}' if len(u) else '',
                 'unit NRMSE': f'{u.nrmse_median.iloc[0]:.3f}' if len(u) else '',
                 'unit skill': f2(u.skill_vs_persistence_median.iloc[0]) if len(u) else '',
                 'unit nodes': int(u.n.iloc[0]) if len(u) else 0})
print('### Table 2 -- one-step forecast over the held-out segment\n')
print(md(pd.DataFrame(rows)))
print()

# ---- Table 3: neural traces vs the no-new-spikes null ---------------------
rows = []
for lab in ORDER:
    sub = nn[nn.label == lab].set_index('horizon_s')
    if not len(sub):
        continue
    row = {'label': f'`{lab}`'}
    for h in (2.0, 5.0, 10.0):
        row[f'NRMSE {h:g} s (net / null)'] = (f'{sub.loc[h, "unit_nrmse_median"]:.2f} / '
                                             f'{sub.loc[h, "unit_nrmse_median_null"]:.2f}')
    for h in (2.0, 10.0):
        row[f'r {h:g} s (net / null)'] = (f'{sub.loc[h, "unit_corr_mean"]:.2f} / '
                                          f'{sub.loc[h, "unit_corr_mean_null"]:.2f}')
    row['units beating null, 10 s'] = f'{100 * sub.loc[10.0, "frac_units_beating_null"]:.0f}%'
    rows.append(row)
print('### Table 3 -- generated unit traces vs the no-new-spikes null\n')
print(md(pd.DataFrame(rows)))
print()

# ---- Table 4: tau sweep ----------------------------------------------------
rows = []
for base in STAGE2_BASES:
    for tau in (2, 4, 6, 8):
        lab = base if tau == 4 else f'{base}_tau{tau}'
        if lab not in set(r.label):
            rows.append({'base': f'`{base}`', 'tau (rows)': tau, 'r(x,y) 5 s': 'not run / not finished'})
            continue
        g = gr[gr.label == lab]
        rows.append({'base': f'`{base}`', 'tau (rows)': tau,
                     'r(x,y) 2 s': pm(R(lab, 2.0, 'corr_xy_mean'), R(lab, 2.0, 'corr_xy_se')),
                     'r(x,y) 5 s': pm(R(lab, 5.0, 'corr_xy_mean'), R(lab, 5.0, 'corr_xy_se')),
                     'path err 5 s (median)': f'{R(lab, 5.0, "path_err_mean_median"):.1f}',
                     'blew up 10 s': f'{100 * R(lab, 10.0, "blew_up_mean"):.0f}%',
                     'unit nodes': str(int(g.n_unit_nodes.iloc[0])) if len(g) else ''})
print('### Table 4 -- Takens lag sweep (Stage II; tau = 4 is the Stage I run)\n')
print(md(pd.DataFrame(rows).fillna('')))
print()

# ---- Table 5: graphs -------------------------------------------------------
rows = []
for lab in ORDER + [l for l in gr.label if l not in ORDER]:
    g = gr[gr.label == lab]
    if not len(g):
        continue
    g = g.iloc[0]
    pairs = json.loads(g.region_pair_edges)
    unit_edges = {k: v for k, v in pairs.items() if not k.endswith('->behaviour')}
    within = sum(v for k, v in unit_edges.items() if k.split('->')[0] == k.split('->')[1])
    beh = {k.split('->')[0]: v for k, v in pairs.items() if k.endswith('->behaviour')}
    rows.append({'label': f'`{lab}`', 'unit nodes': int(g.n_unit_nodes),
                 'lagged leaves': int(g.n_lagged_leaves), 'edges': int(g.n_edges),
                 'density': f'{g.density_real:.3f}',
                 'within-region share': f'{within / max(1, sum(unit_edges.values())):.2f}',
                 'x/y edges CA1/OFC/mPFC': '/'.join(str(beh.get(k, 0)) for k in ('CA1', 'OFC', 'mPFC')),
                 'driver Jaccard vs tsls': f2(g.driver_jaccard_vs_tsls)})
print('### Table 5 -- inferred causal graphs\n')
print(md(pd.DataFrame(rows)))
print()
# Units that drive x/y in most representations.
cnt = {}
for u in gr.driver_units:
    for x in str(u).split():
        cnt[x] = cnt.get(x, 0) + 1
top = sorted(cnt.items(), key=lambda kv: -kv[1])[:10]
print('Most frequent direct x/y drivers (number of labels, out of '
      f'{len(gr)}): ' + ', '.join(f'{k} ({v})' for k, v in top) + '\n')

# Is the driver set stable within a family (only the representation's
# hyperparameters change) and between families? Mean pairwise Jaccard.
sets = {l: set(str(u).split()) for l, u in zip(gr.label, gr.driver_units)}
fam = {l: FAMILY.get(l, '?') for l in sets}


def jac(a, b):
    return len(a & b) / len(a | b) if a | b else np.nan


fams = sorted(set(fam.values()))
rows = []
for f1 in fams:
    row = {'family': f1}
    for f2_ in fams:
        vals = [jac(sets[a], sets[b]) for a in sets for b in sets
                if a < b and {fam[a], fam[b]} == {f1, f2_} and (f1 != f2_ or fam[a] == f1)]
        row[f2_] = f2(np.mean(vals)) if vals else ''
    rows.append(row)
print('Mean pairwise Jaccard overlap of the x/y driver-unit sets, within and between families '
      '(all labels, Stage I and II):\n')
print(md(pd.DataFrame(rows)))
print()
