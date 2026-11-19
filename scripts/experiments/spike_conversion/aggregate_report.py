"""Collect every ablation job's outputs into tables and figures.

Reads the manifest (label -> job id) and each job's archived results under
/bucket/.../tsls_mde_gmn_outputs/job_<id>[_FAIL]/, and writes into --out:

  rollout_summary.csv     label x horizon: mean (and sd) over rollout starts
  onestep_summary.csv     label: one-step corr/NRMSE, targets and units
  graph_summary.csv       label: network size, density, region-pair edges,
                          x/y driver units and their Jaccard overlap with tsls
  job_status.csv          label: job id, STATUS, commit staged, units kept
  rollout_xy_corr_vs_horizon.png, rollout_path_error_vs_time.png,
  rollout_manifold.png, rollout_example_xy.png, onestep_vs_rollout.png,
  rollout_corr_xy_with_se.png, neural_vs_null.png

Run on a compute node (reads ~30 small CSVs per job):
  srun -p intel -t 0:30:00 -c 2 --mem=8G .venv/bin/python3 \
      scripts/experiments/spike_conversion/aggregate_report.py \
      --manifest docs/experiments/spike-conversion-neuro/jobs.tsv \
      --outputs /bucket/FukaiU/tominaga/pao_rotation/tsls_mde_gmn_outputs \
      --out docs/experiments/spike-conversion-neuro
"""

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
from grid import GRID, spike_params  # noqa: E402
from nulls import label_nulls  # noqa: E402

FAMILY = {g['label']: g['family'] for g in GRID}
STALL_FRAC = 0.1   # 'stalled' = generated (x, y) speed below 10% of the real median speed
N_EFF = 16         # 32 starts ~5 s apart, 10-s runs overlap: ~16 independent runs per label
COLORS = {'sawtooth': '#555555', 'A_calcium': '#1f77b4', 'B_adapt_nr': '#d62728',
          'C_theta_osc': '#2ca02c'}
METRICS = ['corr_head_position_x', 'corr_head_position_y', 'vwcorr_head_position_x',
           'vwcorr_head_position_y', 'nrmse_head_position_x', 'nrmse_head_position_y',
           'path_err_mean', 'path_err_final', 'unit_corr_mean', 'unit_nrmse_median',
           'manifold_ratio', 'motion_ratio', 'blew_up',
           'path_err_mean_hold', 'path_err_final_hold', 'unit_nrmse_median_hold']


def job_dir(root, job):
    for suffix in ('', '_FAIL'):
        d = Path(root) / f'job_{job}{suffix}'
        if d.is_dir():
            return d
    return None


def read_status(d):
    s = {}
    p = d / 'STATUS.txt'
    if p.exists():
        for line in p.read_text().splitlines():
            if ':' in line:
                k, v = line.split(':', 1)
                s[k.strip()] = v.strip()
    commit = branch = None
    for out in glob.glob(str(d / '*.out')):
        txt = Path(out).read_text(errors='replace')
        m = re.search(r'^commit=(\S+)', txt, re.M)
        commit = m.group(1) if m else commit
        m = re.search(r'^branch=(\S+)', txt, re.M)
        branch = m.group(1) if m else branch
    return s, commit, branch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--outputs', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--data', default='/bucket/FukaiU/tominaga/pao_rotation/data/'
                                       'J1620210605_epoch_8_freq_40_Hz.csv')
    args = ap.parse_args()
    out = Path(args.out)
    fig_dir = out / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(args.manifest, sep='\t', dtype=str)
    man = man[~man['label'].str.startswith('smoke')]
    status_rows, roll, one, graphs, examples = [], [], [], [], {}
    for _, r in man.iterrows():
        d = job_dir(args.outputs, r['job_id'])
        row = {'label': r['label'], 'family': FAMILY.get(r['label'], '?'), 'job_id': r['job_id']}
        if d is None:
            status_rows.append({**row, 'status': 'not archived (running or lost)'})
            continue
        st, commit, branch = read_status(d)
        units = d / 'retained_units.csv'
        kept = int(pd.read_csv(units)['kept'].sum()) if units.exists() else np.nan
        status_rows.append({**row, 'status': st.get('STATUS', '?'), 'reason': st.get('REASON', ''),
                            'commit_staged': commit, 'branch': branch, 'units_kept': kept})
        for f in glob.glob(str(d / 'rollouts' / 'rollout_metrics_*.csv')):
            df = pd.read_csv(f)
            if len(df):
                roll.append(df.assign(label=r['label'], family=row['family']))
        for f in glob.glob(str(d / 'rollouts' / 'onestep_metrics_*.csv')):
            df = pd.read_csv(f)
            if len(df):
                one.append(df.assign(label=r['label'], family=row['family']))
        for f in glob.glob(str(d / 'rollouts' / 'graph_summary_*.json')):
            g = json.loads(Path(f).read_text())
            graphs.append({'label': r['label'], 'family': row['family'],
                           **{k: g[k] for k in ('n_nodes', 'n_unit_nodes', 'n_lagged_leaves',
                                                'n_edges', 'density_real', 'mean_in_degree',
                                                'max_in_degree')},
                           'region_pair_edges': json.dumps(g['region_pair_edges'], sort_keys=True),
                           'target_drivers': json.dumps(g['target_drivers'])})
        examples[r['label']] = sorted(glob.glob(str(d / 'rollouts' / 'rollout_*_start*.csv')))

    status = pd.DataFrame(status_rows)
    status.to_csv(out / 'job_status.csv', index=False)
    print(status.to_string())
    if not roll:
        print('no rollout metrics yet')
        return
    roll = pd.concat(roll, ignore_index=True)
    roll['blew_up'] = roll['blew_up'].astype(float)
    have = [m for m in METRICS if m in roll.columns]
    g = roll.groupby(['label', 'family', 'horizon_s'])[have]
    summ = g.mean().add_suffix('_mean').join(g.std().add_suffix('_sd')).join(
        g.size().rename('n_starts')).reset_index()
    summ['corr_xy_mean'] = summ[['corr_head_position_x_mean', 'corr_head_position_y_mean']].mean(axis=1)
    # Standard error of the per-start x/y-averaged correlation, using the
    # effective number of independent runs (not 32: neighbouring runs overlap).
    roll['corr_xy'] = roll[['corr_head_position_x', 'corr_head_position_y']].mean(axis=1)
    gx = roll.groupby(['label', 'family', 'horizon_s'])
    summ = summ.merge((gx['corr_xy'].std() / np.sqrt(N_EFF)).rename('corr_xy_se').reset_index(),
                      on=['label', 'family', 'horizon_s'])
    # Medians: a few divergent runs dominate the mean path error.
    summ = summ.merge(gx[['path_err_mean', 'path_err_mean_hold']].median().add_suffix('_median')
                      .reset_index(), on=['label', 'family', 'horizon_s'])
    summ.to_csv(out / 'rollout_summary.csv', index=False)

    if one:
        one = pd.concat(one, ignore_index=True)
        os_ = one.groupby(['label', 'family', 'role']).agg(
            corr_median=('corr', 'median'), corr_mean=('corr', 'mean'),
            nrmse_median=('nrmse', 'median'), nrmse_persistence_median=('nrmse_persistence', 'median'),
            skill_vs_persistence_median=('skill_vs_persistence', 'median'), n=('node', 'size')).reset_index()
        os_.to_csv(out / 'onestep_summary.csv', index=False)
    if graphs:
        gdf = pd.DataFrame(graphs)
        # Which units drive x/y directly (either lag), and how far that set
        # moves from the sawtooth's: Jaccard of the unit sets vs label 'tsls'.
        def driver_units(js):
            return sorted({n.split('::')[0].removesuffix('_tsls')
                           for drv in json.loads(js).values() for n, _ in drv})
        gdf['driver_units'] = gdf['target_drivers'].map(lambda js: ' '.join(driver_units(js)))
        ref = set(gdf.loc[gdf['label'] == 'tsls', 'driver_units'].iloc[0].split()) \
            if (gdf['label'] == 'tsls').any() else set()
        gdf['driver_jaccard_vs_tsls'] = [
            len(ref & set(u.split())) / len(ref | set(u.split())) if ref else np.nan
            for u in gdf['driver_units']]
        gdf.to_csv(out / 'graph_summary.csv', index=False)

    # ---- figures ---------------------------------------------------------
    labels = list(dict.fromkeys(summ['label']))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, t in zip(axes, ('x', 'y')):
        for lab in labels:
            s = summ[summ['label'] == lab]
            fam = s['family'].iloc[0]
            ax.plot(s['horizon_s'], s[f'corr_head_position_{t}_mean'], marker='o', lw=1,
                    color=COLORS.get(fam, 'k'), alpha=0.9 if fam == 'sawtooth' else 0.55,
                    ls='--' if lab == 'tsls_events' else '-')
        ax.axhline(0, color='gray', lw=0.6)
        ax.set_title(f'free-run correlation, head position {t}')
        ax.set_xlabel('rollout horizon (s)')
    axes[0].set_ylabel('Pearson r, mean over starts')
    handles = [plt.Line2D([], [], color=c, label=f) for f, c in COLORS.items()]
    axes[1].legend(handles=handles, fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'rollout_xy_corr_vs_horizon.png', dpi=200)
    plt.close(fig)

    # Euclidean path error at every step, averaged over starts, per label.
    # The real positions come from the job's own trimmed data, row-aligned
    # with the rollout's 'row' column.
    fig, ax = plt.subplots(figsize=(8, 4))
    curves, curves_med, hold_med, stalls = {}, {}, None, []
    for lab, files in examples.items():
        d = job_dir(args.outputs, man.loc[man['label'] == lab, 'job_id'].iloc[0])
        real_path = d / 'tsls_data_trimmed.csv' if d is not None else None
        if not files or real_path is None or not real_path.exists():
            continue
        real = pd.read_csv(real_path, usecols=['head_position_x', 'head_position_y'])
        errs, holds = [], []
        for f in files:
            gen = pd.read_csv(f)
            if not {'head_position_x', 'head_position_y', 'row'} <= set(gen.columns):
                continue
            rows = gen['row'].to_numpy()
            # hold-the-seed, as rollout_eval scores it: the real position at
            # the run's first row held fixed.
            seed = real[['head_position_x', 'head_position_y']].to_numpy()[rows[0]]
            holds.append(np.hypot(real['head_position_x'].to_numpy()[rows] - seed[0],
                                  real['head_position_y'].to_numpy()[rows] - seed[1]))
            errs.append(np.hypot(gen['head_position_x'].to_numpy() - real['head_position_x'].to_numpy()[rows],
                                 gen['head_position_y'].to_numpy() - real['head_position_y'].to_numpy()[rows]))
            # Time to stall: the first time after which the generated (x, y)
            # speed stays below STALL_FRAC of the real median speed for the
            # rest of the run. Equal to the run length if it never stalls.
            gsp = np.hypot(np.diff(gen['head_position_x'].to_numpy()), np.diff(gen['head_position_y'].to_numpy()))
            rsp = np.hypot(np.diff(real['head_position_x'].to_numpy()[rows[0]:]),
                           np.diff(real['head_position_y'].to_numpy()[rows[0]:]))
            moving = np.nonzero(gsp >= STALL_FRAC * np.median(rsp))[0]
            stall = (moving[-1] + 1 if len(moving) else 0) * 0.025
            stalls.append({'label': lab, 'family': FAMILY.get(lab, '?'), 'start': int(rows[0]),
                           'stall_s': stall, 'run_s': len(gsp) * 0.025,
                           'never_stalls': bool(len(moving) and moving[-1] == len(gsp) - 1)})
        if errs:
            n = min(len(e) for e in errs)
            curves[lab] = np.nanmean([e[:n] for e in errs], axis=0)
            curves_med[lab] = np.nanmedian([e[:n] for e in errs], axis=0)
            if hold_med is None:   # every label uses the same start rows
                hold_med = np.nanmedian([h[:n] for h in holds], axis=0)
    # Median over starts on a log axis: the mean is dominated by the few runs
    # that diverge (blew_up), which hides every other run.
    for lab, c in curves_med.items():
        fam = FAMILY.get(lab, '?')
        ax.plot(np.arange(len(c)) * 0.025, c, lw=1, color=COLORS.get(fam, 'k'),
                alpha=0.9 if fam == 'sawtooth' else 0.5, ls='--' if lab == 'tsls_events' else '-')
    if hold_med is not None:
        ax.plot(np.arange(len(hold_med)) * 0.025, hold_med, color='k', lw=2, ls=':', label='hold the seed')
    ax.set_yscale('log')
    if stalls:
        st = pd.DataFrame(stalls)
        st.to_csv(out / 'stall_per_rollout.csv', index=False)
        st.groupby(['label', 'family']).agg(
            stall_s_median=('stall_s', 'median'), stall_s_q25=('stall_s', lambda v: v.quantile(.25)),
            stall_s_q75=('stall_s', lambda v: v.quantile(.75)), frac_never_stalls=('never_stalls', 'mean'),
            n=('stall_s', 'size')).reset_index().to_csv(out / 'stall_summary.csv', index=False)
    if curves:
        pd.DataFrame({k: pd.Series(v) for k, v in curves.items()}).to_csv(
            out / 'path_error_per_step.csv', index_label='step')
        pd.DataFrame({**{k: pd.Series(v) for k, v in curves_med.items()},
                      'hold_the_seed': pd.Series(hold_med)}).to_csv(
            out / 'path_error_per_step_median.csv', index_label='step')
    ax.set_xlabel('time since rollout start (s)')
    ax.set_ylabel('Euclidean (x, y) error, median over starts')
    ax.legend(handles=handles + [plt.Line2D([], [], color='k', lw=2, ls=':', label='hold the seed')],
              fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / 'rollout_path_error_vs_time.png', dpi=200)
    plt.close(fig)

    if 'manifold_ratio_mean' in summ:
        s10 = summ[summ['horizon_s'] == summ['horizon_s'].max()].sort_values('label')
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        cols = [COLORS.get(f, 'k') for f in s10['family']]
        axes[0].bar(s10['label'], s10['manifold_ratio_mean'], color=cols)
        axes[0].axhline(1, color='gray', lw=0.8)
        axes[0].set_yscale('log')
        axes[0].set_title('distance to empirical manifold, generated / real (10 s)')
        axes[1].bar(s10['label'], s10['motion_ratio_mean'], color=cols)
        axes[1].axhline(1, color='gray', lw=0.8)
        axes[1].set_title('motion in the last 1 s, generated / real (0 = fixed point)')
        for ax in axes:
            ax.tick_params(axis='x', rotation=90, labelsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / 'rollout_manifold.png', dpi=200)
        plt.close(fig)
    # Example free runs: per family, the label best at 5 s, and its start
    # closest to that label's median 5-s x/y correlation (a typical run, not
    # the best one), real vs generated (x, y) over the 10-s rollout.
    s5 = summ[summ['horizon_s'] == 5.0]
    picks = []
    for fam, grp in s5.groupby('family'):
        lab = grp.sort_values('corr_xy_mean', ascending=False)['label'].iloc[0]
        r5 = roll[(roll['label'] == lab) & (roll['horizon_s'] == 5.0)].copy()
        r5['cxy'] = r5[['corr_head_position_x', 'corr_head_position_y']].mean(axis=1)
        start = int(r5.iloc[(r5['cxy'] - r5['cxy'].median()).abs().argsort().iloc[0]]['start'])
        picks.append((fam, lab, start, float(r5['cxy'].median())))
    if picks:
        fig, axes = plt.subplots(len(picks), 3, figsize=(13, 2.6 * len(picks)), squeeze=False)
        for row, (fam, lab, start, med) in zip(axes, picks):
            d = job_dir(args.outputs, man.loc[man['label'] == lab, 'job_id'].iloc[0])
            gen = pd.read_csv(next(f for f in examples[lab] if f.endswith(f'_start{start}.csv')))
            real = pd.read_csv(d / 'tsls_data_trimmed.csv', usecols=['head_position_x', 'head_position_y'])
            rows = gen['row'].to_numpy()
            tt = (rows - rows[0]) * 0.025
            for ax, t in zip(row[:2], ('x', 'y')):
                ax.plot(tt, real[f'head_position_{t}'].to_numpy()[rows], color='k', lw=1, label='real')
                ax.plot(tt, gen[f'head_position_{t}'], color=COLORS.get(fam, 'C0'), lw=1, label='free run')
                ax.set_title(f'{lab}: head position {t} (start row {start})', fontsize=8)
            row[2].plot(real['head_position_x'].to_numpy()[rows], real['head_position_y'].to_numpy()[rows],
                        color='k', lw=1)
            row[2].plot(gen['head_position_x'], gen['head_position_y'], color=COLORS.get(fam, 'C0'), lw=1)
            row[2].set_title(f'path; {lab} median 5-s x/y r = {med:.2f}', fontsize=8)
            row[2].set_aspect('equal', 'datalim')
        axes[0][0].legend(fontsize=7)
        axes[-1][0].set_xlabel('time since rollout start (s)')
        axes[-1][1].set_xlabel('time since rollout start (s)')
        fig.tight_layout()
        fig.savefig(fig_dir / 'rollout_example_xy.png', dpi=200)
        plt.close(fig)

    # Does one-step skill predict free-run quality? One point per label.
    if isinstance(one, pd.DataFrame):
        t1 = os_[os_['role'] == 'target'].set_index('label')
        u1 = os_[os_['role'] == 'unit'].set_index('label')
        r5 = s5.set_index('label')
        common = [l for l in r5.index if l in t1.index]
        if common:
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            for ax, src, name in ((axes[0], t1, 'targets (x, y)'), (axes[1], u1, 'unit nodes')):
                ok = [l for l in common if l in src.index]
                ax.scatter(src.loc[ok, 'skill_vs_persistence_median'], r5.loc[ok, 'corr_xy_mean'],
                           c=[COLORS.get(FAMILY.get(l, '?'), 'k') for l in ok], s=25)
                ax.set_xlabel(f'one-step skill vs persistence, median over {name}')
                ax.axvline(0, color='gray', lw=0.6)
            axes[0].set_ylabel('5-s free-run x/y correlation (mean over starts)')
            axes[1].legend(handles=handles, fontsize=7)
            fig.tight_layout()
            fig.savefig(fig_dir / 'onestep_vs_rollout.png', dpi=200)
            plt.close(fig)
    # Null baselines (nulls.py): the representation's own no-new-spikes
    # evolution for the unit traces, and mismatched starts for x/y.
    raw = pd.read_csv(args.data)
    byname = {g['label']: g for g in GRID}
    nrows, mrows = [], []
    for lab, files in examples.items():
        if lab not in byname or not files:
            continue
        g = byname[lab]
        d = job_dir(args.outputs, man.loc[man['label'] == lab, 'job_id'].iloc[0])
        r_, m_ = label_nulls(lab, g['env']['SPIKE_TRANSFORM'], spike_params(g), d, raw, files,
                             [80, 200, 400])
        nrows += r_
        mrows += m_
        print(f'nulls: {lab} done', flush=True)
    if nrows:
        nd = pd.DataFrame(nrows)
        nd['family'] = nd['label'].map(FAMILY)
        nd.to_csv(out / 'neural_null_per_rollout.csv', index=False)
        ng = nd.groupby(['label', 'family', 'horizon_s'])
        ns = ng[['unit_corr_mean', 'unit_corr_mean_null', 'unit_nrmse_median', 'unit_nrmse_median_null',
                 'frac_units_beating_null']].mean().reset_index()
        ns.to_csv(out / 'neural_null_summary.csv', index=False)
        md = pd.DataFrame(mrows)
        md = md.merge(summ[['label', 'horizon_s', 'corr_xy_mean']], on=['label', 'horizon_s'], how='left')
        md['corr_xy_excess'] = md['corr_xy_mean'] - md['corr_xy_mismatched']
        md.to_csv(out / 'xy_mismatched_null.csv', index=False)
        # The headline comparison: 5-s x/y correlation per label with its
        # standard error, against what a run from a different start scores.
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
        order = summ[summ['horizon_s'] == 5.0].sort_values(['family', 'label'])['label'].tolist()
        for ax, h in zip(axes, (2.0, 5.0, 10.0)):
            sh = summ[summ['horizon_s'] == h].set_index('label').reindex(order)
            mh = md[md['horizon_s'] == h].set_index('label').reindex(order)
            y = np.arange(len(order))[::-1]
            ax.errorbar(sh['corr_xy_mean'], y, xerr=sh['corr_xy_se'], fmt='none', ecolor='gray', lw=1)
            ax.scatter(sh['corr_xy_mean'], y, c=[COLORS.get(f, 'k') for f in sh['family']], s=30, zorder=3)
            ax.scatter(mh['corr_xy_mismatched'], y, marker='x', color='k', s=20, zorder=3)
            ax.axvline(0, color='gray', lw=0.6)
            ref = sh.loc['tsls', 'corr_xy_mean'] if 'tsls' in sh.index else np.nan
            ax.axvline(ref, color=COLORS['sawtooth'], lw=0.8, ls='--')
            ax.set_yticks(y, order, fontsize=7)
            ax.set_title(f'x/y correlation at {h:g} s (mean over starts, \u00b11 SE)', fontsize=9)
            ax.set_xlabel('Pearson r (dot)   x = mismatched-start null')
        fig.tight_layout()
        fig.savefig(fig_dir / 'rollout_corr_xy_with_se.png', dpi=200)
        plt.close(fig)
        print(ns.round(3).to_string())
        fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        for ax, h in zip(axes, (2.0, 5.0, 10.0)):
            sub = ns[ns['horizon_s'] == h].sort_values('label')
            x = np.arange(len(sub))
            ax.bar(x - 0.2, sub['unit_nrmse_median'], 0.4,
                   color=[COLORS.get(f, 'k') for f in sub['family']], label='network')
            ax.bar(x + 0.2, sub['unit_nrmse_median_null'], 0.4, color='lightgray', label='no-new-spikes null')
            ax.set_xticks(x, sub['label'], rotation=90, fontsize=7)
            ax.set_title(f'unit traces, median NRMSE at {h:g} s (lower is better)')
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(fig_dir / 'neural_vs_null.png', dpi=200)
        plt.close(fig)
        print(md.round(3).to_string())
    print(summ.to_string())


if __name__ == '__main__':
    main()
