"""Stage II aggregation for exp/spike-conversion-general.

Reads docs/experiments/spike-conversion-general/jobs.tsv, finds each job's
archived outputs on /bucket, and writes into the report directory:
  stage2_summary.csv        one row per label (rollout medians per horizon,
                            one-step, topology, graph)
  stage2_rollouts_all.csv   every (label, start, horizon) row
  stage2_tables.md          the Markdown tables the report embeds
  figures/stage2_*.png      dot plots by horizon; copies of each job's
                            example-rollout and persistence figures

Usage (compute node or login node -- it only reads small files):
  python aggregate.py [report_dir]
"""
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
P = HERE.parents[3]
REPORT = Path(sys.argv[1]) if len(sys.argv) > 1 else P / 'docs/experiments/spike-conversion-general'
OUTPUTS = Path('/bucket/FukaiU/tominaga/pao_rotation/tsls_mde_gmn_outputs')
NET = 'head_position_x_head_position_y'
METHOD_COLOR = {'sawtooth': '#2a78d6', 'gamma': '#eb6834', 'lmu': '#1baf7a', 'logsig': '#eda100'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def job_dir(job):
    for suffix in ('', '_FAIL'):
        d = OUTPUTS / f'job_{job}{suffix}'
        if d.exists():
            return d, ('COMPLETED' if not suffix else 'FAIL')
    return None, 'missing'


def _q(s, q):
    s = pd.to_numeric(s, errors='coerce').dropna()
    return float(s.quantile(q)) if len(s) else np.nan


def collect():
    jobs = pd.read_csv(REPORT / 'jobs.tsv', sep='\t', dtype=str)
    jobs = jobs[~jobs['label'].str.startswith('smoke')].drop_duplicates('label', keep='last')
    rows, roll_all = [], []
    for _, j in jobs.iterrows():
        d, status = job_dir(j['job_id'])
        rec = {'label': j['label'], 'job_id': j['job_id'], 'commit': j['commit'], 'status': status}
        m = re.search(r'SPIKE_TRANSFORM=(\w+)', j.get('settings', '') or '')
        rec['method'] = m.group(1) if m else 'sawtooth'
        if d is None:
            rows.append(rec)
            continue
        st = (d / 'STATUS.txt')
        if st.exists():
            rec['status'] = st.read_text().splitlines()[0].replace('STATUS: ', '')
        a = d / 'results' if (d / 'results' / 'autonomous').exists() else d
        auto = a / 'autonomous'
        logs = list(a.glob('spk_gen-*.out'))
        if logs:
            txt = logs[0].read_text(errors='replace')
            mt = re.search(r'EMBED_TAU=auto -> (\d+) rows', txt)
            rec['embed_tau_rows'] = int(mt.group(1)) if mt else np.nan
            mk = re.search(r'Observability filter: kept (\d+)/(\d+)', txt)
            if mk:
                rec['channels_kept'] = int(mk.group(1))
            mc = re.search(r'commit=(\w+)', txt)
            rec['staged_commit'] = mc.group(1) if mc else ''
        g = auto / f'graph_{NET}.json'
        if g.exists():
            gj = json.loads(g.read_text())
            for k in ('n_real_nodes', 'n_lagged_leaves', 'n_edges', 'n_edges_from_lagged',
                      'density_real', 'edges_per_node', 'in_degree_max', 'zero_driver_nodes',
                      'external_input_mode'):
                rec[k] = gj.get(k)
            rec['target_drivers'] = '; '.join(f"{t.split('_')[-1]}: {', '.join(v)}"
                                              for t, v in gj.get('target_drivers', {}).items())
            rp = gj.get('region_pairs', {})
            within = sum(v for k, v in rp.items() if k.split('->')[0] == k.split('->')[1])
            rec['within_region_edge_frac'] = within / max(1, sum(rp.values()))
            rec['region_pairs'] = json.dumps(rp)
        # Jobs staged before 773e99d mislabelled Forecast() rows for nodes with
        # a lagged in-edge; rescore_onestep.py reran their one-step only.
        o = REPORT / 'rescored' / f'onestep_{j["job_id"]}.csv'
        rec['onestep_rescored'] = o.exists()
        o = o if o.exists() else auto / f'onestep_{NET}.csv'
        if o.exists():
            od = pd.read_csv(o)
            if 'n_drivers' not in od:     # pre-773e99d file: not trustworthy for driven nodes
                od['n_drivers'] = np.nan
            for role in ('target', 'channel'):
                sub = od[od['role'] == role]
                rec[f'onestep_r_{role}'] = _q(sub['r'], 0.5)
                rec[f'onestep_skill_{role}'] = _q(sub['skill_vs_persistence'], 0.5)
            dr = od[(od['role'] == 'channel') & (od['n_drivers'] > 0)]
            rec['onestep_r_driven'] = _q(dr['r'], 0.5)
            rec['onestep_skill_driven'] = _q(dr['skill_vs_persistence'], 0.5)
            rec['n_driven_units'] = len(dr)
        r = auto / f'rollouts_{NET}.csv'
        if r.exists():
            rd = pd.read_csv(r)
            rd.insert(0, 'label', j['label'])
            rd.insert(1, 'method', rec['method'])
            roll_all.append(rd)
            for H, sub in rd.groupby('horizon_s'):
                h = f'{int(H)}s'
                rec[f'r_xy_{h}'] = _q(sub['r_xy'], 0.5)
                rec[f'r_xy_{h}_q25'] = _q(sub['r_xy'], 0.25)
                rec[f'r_xy_{h}_q75'] = _q(sub['r_xy'], 0.75)
                rec[f'vwr_xy_{h}'] = _q(sub['vwr_xy'], 0.5)
                rec[f'nrmse_xy_{h}'] = _q(sub['nrmse_xy'], 0.5)
                rec[f'path_err_{h}'] = _q(sub['path_err_mean'], 0.5)
                rec[f'path_err_hold_{h}'] = _q(sub['path_err_hold_mean'], 0.5)
                rec[f'path_ratio_{h}'] = _q(sub['path_err_mean'] / sub['path_err_hold_mean'], 0.5)
                rec[f'manifold_ratio_{h}'] = _q(sub['manifold_ratio'], 0.5)
                rec[f'motion_ratio_{h}'] = _q(sub['motion_ratio'], 0.5)
                rec[f'blew_up_{h}'] = float(sub['blew_up'].mean())
                rec[f'collapsed_{h}'] = float(sub['collapsed'].mean())
                if 'r_channels_mean' in sub:
                    rec[f'r_channels_{h}'] = _q(sub['r_channels_mean'], 0.5)
            rec['n_starts'] = rd['start'].nunique()
        t = auto / f'topology_{NET}.json'
        if t.exists():
            tj = json.loads(t.read_text())
            for space in ('xy', 'population'):
                for k, v in (tj.get(space) or {}).items():
                    if k.startswith(('W1_', 'H1_', 'H0_real_n', 'H0_generated_n', 'betti')):
                        rec[f'{space}_{k}'] = v
        figs = REPORT / 'figures' / 'stage2'
        figs.mkdir(parents=True, exist_ok=True)
        for src, dst in ((auto / f'rollout_examples_{NET}.png', f'{j["label"]}_rollouts.png'),
                         (auto / f'persistence_xy_{NET}.png', f'{j["label"]}_persistence_xy.png'),
                         (auto / f'persistence_population_{NET}.png',
                          f'{j["label"]}_persistence_population.png')):
            if src.exists():
                shutil.copy(src, figs / dst)
        rows.append(rec)
    summ = pd.DataFrame(rows)
    allr = pd.concat(roll_all, ignore_index=True) if roll_all else pd.DataFrame()
    return summ, allr


def _style(ax):
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)
    for sp in ('left', 'bottom'):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, axis='x', color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def dotplot(allr, metric, xlabel, path, title, ref=None, ref_label=None, logx=False, order=None):
    hs = sorted(allr['horizon_s'].unique())
    labels = order
    fig, axes = plt.subplots(1, len(hs), figsize=(4.0 * len(hs), 0.32 * len(labels) + 1.4),
                             sharey=True, squeeze=False)
    y = {l: i for i, l in enumerate(labels[::-1])}
    for ax, H in zip(axes[0], hs):
        sub = allr[allr['horizon_s'] == H]
        for lab in labels:
            s = pd.to_numeric(sub.loc[sub['label'] == lab, metric], errors='coerce').dropna()
            if not len(s):
                continue
            col = METHOD_COLOR.get(sub.loc[sub['label'] == lab, 'method'].iloc[0], INK)
            ax.plot([s.quantile(0.25), s.quantile(0.75)], [y[lab]] * 2, color=col, lw=2,
                    solid_capstyle='round')
            ax.scatter([s.median()], [y[lab]], s=48, c=col, edgecolors='#fcfcfb', linewidths=1.5,
                       zorder=3)
        if ref is not None:
            ax.axvline(ref, color=INK2, lw=0.8, ls='--')
        if logx:
            ax.set_xscale('log')
        ax.set_title(f'{H:g} s free run', fontsize=9, color=INK, loc='left')
        ax.set_xlabel(xlabel, fontsize=8, color=INK)
        _style(ax)
    axes[0][0].set_yticks(list(y.values()))
    axes[0][0].set_yticklabels(list(y.keys()), fontsize=8)
    note = f'  (dashed: {ref_label})' if ref_label else ''
    fig.suptitle(title + '  -- dot = median over starts, bar = IQR' + note, fontsize=9.5,
                 x=0.01, ha='left', color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def md_table(df, cols, names, fmt=None):
    fmt = fmt or {}
    out = '| ' + ' | '.join(names) + ' |\n|' + '---|' * len(cols) + '\n'
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            if isinstance(v, (float, np.floating)):
                cells.append('' if not np.isfinite(v) else fmt.get(c, '{:.3f}').format(v))
            else:
                cells.append('' if v is None else str(v))
        out += '| ' + ' | '.join(cells) + ' |\n'
    return out


def paired_vs_baseline(allr, labels, base='sawtooth'):
    '''Each setting against the baseline, start by start. Starts are evenly
    spaced over the same held-out segment in every run (they differ only by
    each graph's warm-up of max-delay + 1 rows), so the i-th start of one run
    is paired with the i-th start of the other. Wilcoxon signed-rank, two-sided.'''
    from scipy.stats import wilcoxon
    allr = allr.copy()
    allr['log_path_ratio'] = np.log(allr['path_err_mean'] / allr['path_err_hold_mean'])
    allr['k'] = allr.groupby(['label', 'horizon_s'])['start'].rank(method='first').astype(int)
    rows = []
    for lab in labels:
        if lab == base:
            continue
        rec = {'label': lab}
        for H in sorted(allr['horizon_s'].unique()):
            a = allr[(allr.label == lab) & (allr.horizon_s == H)].set_index('k')
            b = allr[(allr.label == base) & (allr.horizon_s == H)].set_index('k')
            k = a.index.intersection(b.index)
            for m, short in (('r_xy', 'dr'), ('log_path_ratio', 'dlogpath')):
                d = (a.loc[k, m] - b.loc[k, m]).dropna()
                rec[f'{short}_{int(H)}s'] = float(d.median()) if len(d) else np.nan
                rec[f'p_{short}_{int(H)}s'] = (float(wilcoxon(d).pvalue)
                                              if len(d) >= 6 and (d != 0).any() else np.nan)
            rec['n_pairs'] = len(k)
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(REPORT / 'stage2_paired_vs_sawtooth.csv', index=False)
    cols = ['label', 'n_pairs'] + [f'{p}{m}_{h}s' for h in (2, 5, 10)
                                   for m in ('dr', 'dlogpath') for p in ('', 'p_')]
    names = ['setting', 'pairs'] + [f'{n} {h} s' for h in (2, 5, 10)
                                    for n in ('Δr', 'p', 'Δlog path', 'p')]
    return ('### Paired against the sawtooth, start by start (median difference; '
            'Wilcoxon p)\n\nΔr > 0: better (x, y) correlation than the sawtooth. '
            'Δlog path < 0: smaller path error than the sawtooth.\n\n'
            + md_table(df, cols, names, {'n_pairs': '{:.0f}'}))


def main():
    summ, allr = collect()
    summ.to_csv(REPORT / 'stage2_summary.csv', index=False)
    if not len(allr):
        print('no rollout outputs yet')
        return
    allr.to_csv(REPORT / 'stage2_rollouts_all.csv', index=False)
    figs = REPORT / 'figures'
    method_rank = {'sawtooth': 0, 'gamma': 1, 'lmu': 2, 'logsig': 3}
    order = (summ.dropna(subset=['n_starts'])
             .assign(k=lambda d: d['method'].map(method_rank))
             .sort_values(['k', 'label'])['label'].tolist())
    dotplot(allr, 'r_xy', 'Pearson r, generated vs recorded (mean of x, y)',
            figs / 'stage2_rollout_r_xy.png', 'Free-run (x, y) correlation', ref=0, order=order)
    allr['path_ratio'] = allr['path_err_mean'] / allr['path_err_hold_mean']
    dotplot(allr, 'path_ratio', 'mean path error / hold-the-seed path error (log)',
            figs / 'stage2_rollout_path_ratio.png', 'Free-run path error vs holding still',
            ref=1, ref_label='= holding the seed position', logx=True, order=order)
    dotplot(allr, 'manifold_ratio', 'distance to library manifold, generated / recorded (log)',
            figs / 'stage2_rollout_manifold.png', 'Does the free run stay on the empirical manifold?',
            ref=1, ref_label='= as close as the recording', logx=True, order=order)
    dotplot(allr, 'motion_ratio', 'sd over the last 2 s, generated / recorded (log)',
            figs / 'stage2_rollout_motion.png', 'Collapse (<<1) or over-activity (>>1)',
            ref=1, logx=True, order=order)

    s = summ.dropna(subset=['n_starts']).copy()
    s['k'] = s['method'].map(method_rank)
    s = s.sort_values(['k', 'label'])
    tabs = []
    tabs.append('### Free-run accuracy (median over starts)\n\n' + md_table(
        s, ['label', 'r_xy_2s', 'r_xy_5s', 'r_xy_10s', 'vwr_xy_10s', 'nrmse_xy_10s',
            'path_err_2s', 'path_err_hold_2s', 'path_err_10s', 'path_err_hold_10s'],
        ['setting', 'r 2 s', 'r 5 s', 'r 10 s', 'speed-wtd r 10 s', 'NRMSE 10 s',
         'path err 2 s', 'hold 2 s', 'path err 10 s', 'hold 10 s'],
        {c: '{:.2f}' for c in ('path_err_2s', 'path_err_hold_2s', 'path_err_10s', 'path_err_hold_10s')}))
    tabs.append('### Attractor stability (median over starts; fractions over starts)\n\n' + md_table(
        s, ['label', 'manifold_ratio_2s', 'manifold_ratio_10s', 'motion_ratio_2s', 'motion_ratio_10s',
            'collapsed_10s', 'blew_up_10s', 'r_channels_2s'],
        ['setting', 'manifold 2 s', 'manifold 10 s', 'motion 2 s', 'motion 10 s',
         'collapsed 10 s', 'blew up 10 s', 'unit r 2 s']))
    tabs.append('### One-step accuracy over the whole held-out segment (median over nodes)\n\n' + md_table(
        s, ['label', 'onestep_r_target', 'onestep_skill_target', 'n_driven_units', 'onestep_r_driven',
            'onestep_skill_driven', 'onestep_rescored'],
        ['setting', 'r (x, y)', 'skill vs persistence (x, y)', 'driven units', 'r (driven units)',
         'skill vs persistence (driven units)', 'rescored'],
        {'n_driven_units': '{:.0f}'}))
    tabs.append('### Topology of the (x, y) delay embedding: Wasserstein-1 distances\n\n' + md_table(
        s, ['label', 'xy_W1_H0_generated_vs_real', 'xy_W1_H1_generated_vs_real',
            'xy_W1_H0_library_vs_real', 'xy_W1_H1_library_vs_real', 'xy_H1_real_n_bars',
            'xy_H1_generated_n_bars', 'population_W1_H1_generated_vs_real',
            'population_W1_H1_library_vs_real'],
        ['setting', 'H0 gen-real', 'H1 gen-real', 'H0 lib-real (floor)', 'H1 lib-real (floor)',
         'H1 bars real', 'H1 bars gen', 'pop H1 gen-real', 'pop H1 lib-real (floor)']))
    tabs.append('### Graph (one combined network per run)\n\n' + md_table(
        s, ['label', 'embed_tau_rows', 'channels_kept', 'n_real_nodes', 'n_lagged_leaves', 'n_edges',
            'density_real', 'edges_per_node', 'zero_driver_nodes', 'within_region_edge_frac',
            'target_drivers'],
        ['setting', 'Takens lag (rows)', 'channels kept', 'real nodes', 'lagged leaves', 'edges',
         'density', 'edges/node', 'zero-driver', 'within-region', 'target drivers'],
        {'embed_tau_rows': '{:.0f}', 'channels_kept': '{:.0f}', 'n_real_nodes': '{:.0f}',
         'n_lagged_leaves': '{:.0f}', 'n_edges': '{:.0f}', 'zero_driver_nodes': '{:.0f}'}))
    tabs.append(paired_vs_baseline(allr, s['label'].tolist()))
    (REPORT / 'stage2_tables.md').write_text('\n\n'.join(tabs) + '\n')
    print(f'wrote {REPORT / "stage2_tables.md"}; {len(s)} settings with rollouts')


if __name__ == '__main__':
    main()
