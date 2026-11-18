"""Collect every ablation job's outputs into tables and figures.

Reads the manifest (label -> job id) and each job's archived results under
/bucket/.../tsls_mde_gmn_outputs/job_<id>[_FAIL]/, and writes into --out:

  rollout_summary.csv     label x horizon: mean (and sd) over rollout starts
  onestep_summary.csv     label: one-step corr/NRMSE, targets and units
  graph_summary.csv       label: network size, density, region-pair edges
  job_status.csv          label: job id, STATUS, commit staged, units kept
  rollout_xy_corr_vs_horizon.png, rollout_path_error_vs_time.png,
  rollout_manifold.png, rollout_example_xy.png, onestep_vs_rollout.png

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
from grid import GRID  # noqa: E402

FAMILY = {g['label']: g['family'] for g in GRID}
COLORS = {'sawtooth': '#555555', 'A_calcium': '#1f77b4', 'B_adapt_nr': '#d62728',
          'C_theta_osc': '#2ca02c'}
METRICS = ['corr_head_position_x', 'corr_head_position_y', 'vwcorr_head_position_x',
           'vwcorr_head_position_y', 'nrmse_head_position_x', 'nrmse_head_position_y',
           'path_err_mean', 'path_err_final', 'unit_corr_mean', 'unit_nrmse_median',
           'manifold_ratio', 'motion_ratio', 'blew_up']


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
    summ['corr_xy_mean'] = summ[['corr_head_position_x_mean', 'corr_head_position_y_mean']].mean(1)
    summ.to_csv(out / 'rollout_summary.csv', index=False)

    if one:
        one = pd.concat(one, ignore_index=True)
        os_ = one.groupby(['label', 'family', 'role']).agg(
            corr_median=('corr', 'median'), corr_mean=('corr', 'mean'),
            nrmse_median=('nrmse', 'median'), n=('node', 'size')).reset_index()
        os_.to_csv(out / 'onestep_summary.csv', index=False)
    if graphs:
        pd.DataFrame(graphs).to_csv(out / 'graph_summary.csv', index=False)

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
    curves = {}
    for lab, files in examples.items():
        d = job_dir(args.outputs, man.loc[man['label'] == lab, 'job_id'].iloc[0])
        real_path = d / 'tsls_data_trimmed.csv' if d is not None else None
        if not files or real_path is None or not real_path.exists():
            continue
        real = pd.read_csv(real_path, usecols=['head_position_x', 'head_position_y'])
        errs = []
        for f in files:
            gen = pd.read_csv(f)
            if not {'head_position_x', 'head_position_y', 'row'} <= set(gen.columns):
                continue
            rows = gen['row'].to_numpy()
            errs.append(np.hypot(gen['head_position_x'].to_numpy() - real['head_position_x'].to_numpy()[rows],
                                 gen['head_position_y'].to_numpy() - real['head_position_y'].to_numpy()[rows]))
        if errs:
            n = min(len(e) for e in errs)
            curves[lab] = np.nanmean([e[:n] for e in errs], axis=0)
    for lab, c in curves.items():
        fam = FAMILY.get(lab, '?')
        ax.plot(np.arange(len(c)) * 0.025, c, lw=1, color=COLORS.get(fam, 'k'),
                alpha=0.9 if fam == 'sawtooth' else 0.5, ls='--' if lab == 'tsls_events' else '-')
    if curves:
        pd.DataFrame({k: pd.Series(v) for k, v in curves.items()}).to_csv(
            out / 'path_error_per_step.csv', index_label='step')
    ax.set_xlabel('time since rollout start (s)')
    ax.set_ylabel('Euclidean (x, y) error, mean over starts')
    ax.legend(handles=handles, fontsize=8)
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
    print(summ.to_string())


if __name__ == '__main__':
    main()
