'''Batch/cluster port of notebooks/tsls_example.ipynb, executed by
scripts/batch/tsls_example.sbatch. Every parameter is read from a
TSLS_EXAMPLE_* environment variable (set by the sbatch script's
MODIFIABLE PARAMETERS block) rather than edited here directly.

Mirrors the notebook cell-for-cell (see that file for the narrative
explanation of *why* each step works the way it does):

  1. Spike counts -> TSLS for an occupancy-filtered "sweet spot" scan
     pool, then a data-driven pick of this run's demo SOURCE/TARGET pair
     via a pairwise proper_simplex correlation scan (rather than a
     hand-picked pair -- this project's established preference for
     data-driven parameter choices).
  2. proper_simplex on that pair.
  3. proper_ccm on that pair.
  4. A second, independently data-driven target/candidate pool -- picked
     by directly probing CCM t2c convergence rather than correlation, so
     it can show a *causally validated* proper_mde(noCCM=False) result
     (the Section-1 pair is deliberately picked by correlation alone, so
     it is expected to fail that causal gate) -- then proper_mde run both
     ways (noCCM=True ranking-only vs. noCCM=False CCM-gated).

Differences from the notebook, all a consequence of running headless
rather than in a live Jupyter kernel:
  - Every plt.show() is replaced with plt.close(fig) after the
    corresponding savefig.
  - All output (CSV/JSON/figures) goes under one flat RESULTS_DIR rather
    than the notebook's separate results/tsls/ vs. figures/ split --
    simpler for the sbatch script's `cp -av` to copy back verbatim.
  - cleaned_helper_functions.py is imported from the same directory as
    this script (the sbatch script copies it there alongside the data
    file) instead of the notebook's sys.path insert against
    ../external/TSLS.

COST NOTE: the pairwise correlation scan (Section 1c) is
O(SCAN_POOL_SIZE^2) proper_simplex calls -- cleaned_helper_functions.py
has no multiprocessing of its own (confirmed by grepping it for
"multiprocessing"/"Pool("), so this and every other proper_* call here
runs single-threaded regardless of how many cores this job requests.
Widening WINDOW_ROWS or SCAN_POOL_SIZE past the notebook's demo-sized
defaults costs roughly linearly (WINDOW_ROWS) or quadratically
(SCAN_POOL_SIZE) more wall time.
'''
import itertools
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# cleaned_helper_functions.py isn't installed as a package -- the sbatch
# script copies it into this script's own directory, so a plain import
# (relying on Python's default "script's own directory is on sys.path")
# already finds it; sys.path is extended explicitly anyway so this also
# works if this file is ever run from a different working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cleaned_helper_functions as tsls  # noqa: E402 -- import after sys.path edit above


# =============================================================================
# Configuration -- every value overridable via TSLS_EXAMPLE_* environment
# variables; see scripts/batch/tsls_example.sbatch's MODIFIABLE PARAMETERS
# block for the canonical list with descriptions.
# =============================================================================

def _env_float_list(name, default):
    val = os.environ.get(name)
    if val is None:
        return list(default)
    return [float(x.strip()) for x in val.split(',') if x.strip()]


def load_config():
    cfg = {}

    # ---- Data -----------------------------------------------------------
    cfg['DATA_DIR'] = Path(os.environ.get('TSLS_EXAMPLE_DATA_DIR', '../../data'))
    cfg['DATA_FILE'] = os.environ.get('TSLS_EXAMPLE_DATA_FILE', 'J1620210605_epoch_8_freq_40_Hz.csv')
    cfg['RESULTS_DIR'] = Path(os.environ.get('TSLS_EXAMPLE_RESULTS_DIR', '../../results'))
    cfg['RESULTS_DIR'].mkdir(parents=True, exist_ok=True)

    # Analysis window: the first WINDOW_ROWS rows of the recording. See
    # module docstring's COST NOTE -- every proper_* call below scales
    # with this (linearly) and SCAN_POOL_SIZE (quadratically, via the
    # pairwise scan).
    cfg['WINDOW_ROWS'] = int(os.environ.get('TSLS_EXAMPLE_WINDOW_ROWS', 6000))

    # ---- Candidate neuron pool (Section 1) -------------------------------
    cfg['OCC_MIN'] = float(os.environ.get('TSLS_EXAMPLE_OCC_MIN', 0.015))
    cfg['OCC_MAX'] = float(os.environ.get('TSLS_EXAMPLE_OCC_MAX', 0.10))
    cfg['SCAN_POOL_SIZE'] = int(os.environ.get('TSLS_EXAMPLE_SCAN_POOL_SIZE', 28))
    cfg['NUM_WORKING_NEURONS'] = int(os.environ.get('TSLS_EXAMPLE_NUM_WORKING_NEURONS', 6))

    # ---- Simplex/CCM demo pair (Sections 2-3) ----------------------------
    cfg['DEMO_E'] = int(os.environ.get('TSLS_EXAMPLE_DEMO_E', 3))
    cfg['DEMO_TAU'] = int(os.environ.get('TSLS_EXAMPLE_DEMO_TAU', -1))
    cfg['DEMO_TP'] = int(os.environ.get('TSLS_EXAMPLE_DEMO_TP', 1))
    cfg['CCM_SAMPLE'] = int(os.environ.get('TSLS_EXAMPLE_CCM_SAMPLE', 20))
    cfg['CCM_LIB_SIZE_PCTS'] = _env_float_list('TSLS_EXAMPLE_CCM_LIB_SIZE_PCTS', (10, 30, 50, 70, 90))
    cfg['CCM_SEED'] = int(os.environ.get('TSLS_EXAMPLE_CCM_SEED', 0))

    # ---- MDE target/candidate discovery (Section 4) ----------------------
    cfg['MDE_TARGET_POOL_SIZE'] = int(os.environ.get('TSLS_EXAMPLE_MDE_TARGET_POOL_SIZE', 8))
    cfg['MDE_DRIVERS_PER_TARGET'] = int(os.environ.get('TSLS_EXAMPLE_MDE_DRIVERS_PER_TARGET', 5))
    cfg['PROBE_LIB_SIZE_PCTS'] = _env_float_list('TSLS_EXAMPLE_PROBE_LIB_SIZE_PCTS', (20, 50, 80))
    cfg['PROBE_SAMPLE'] = int(os.environ.get('TSLS_EXAMPLE_PROBE_SAMPLE', 8))

    # ---- proper_mde runs (Section 4) -------------------------------------
    cfg['MDE_D'] = int(os.environ.get('TSLS_EXAMPLE_MDE_D', 3))
    cfg['MDE_RANK_ONLY_CROSS_MAP_RHO_MIN'] = float(
        os.environ.get('TSLS_EXAMPLE_MDE_RANK_ONLY_CROSS_MAP_RHO_MIN', 0.3))
    cfg['MDE_FULL_SAMPLE'] = int(os.environ.get('TSLS_EXAMPLE_MDE_FULL_SAMPLE', 15))
    cfg['MDE_FULL_P_LIB_SIZES'] = tuple(
        int(x) for x in _env_float_list('TSLS_EXAMPLE_MDE_FULL_P_LIB_SIZES', (20, 40, 60, 80))
    )
    cfg['MDE_FULL_CROSS_MAP_RHO_MIN'] = float(os.environ.get('TSLS_EXAMPLE_MDE_FULL_CROSS_MAP_RHO_MIN', 0.3))
    cfg['MDE_FULL_EMBED_DIM_RHO_MIN'] = float(os.environ.get('TSLS_EXAMPLE_MDE_FULL_EMBED_DIM_RHO_MIN', 0.3))

    return cfg


# =============================================================================
# Section 1: spike counts -> TSLS, scan pool, data-driven demo pair
# =============================================================================

REGION_PREFIXES = ('CA1_', 'mPFC_', 'OFC_')


def load_and_window_data(cfg):
    df_full = pd.read_csv(cfg['DATA_DIR'] / cfg['DATA_FILE'])
    df = df_full.iloc[:cfg['WINDOW_ROWS']].reset_index(drop=True)
    window_seconds = df['time_bin_center'].iloc[-1] - df['time_bin_center'].iloc[0]
    print(f"{cfg['DATA_FILE']}: {df_full.shape[0]} rows total, using first {df.shape[0]} rows "
          f"({window_seconds:.1f} s)", flush=True)
    return df


def select_scan_pool(df, cfg):
    all_neuron_columns = [c for c in df.columns if c.startswith(REGION_PREFIXES)]
    occupancy = {c: (df[c] > 0).mean() for c in all_neuron_columns}
    sweet_spot_neurons = [
        c for c in all_neuron_columns if cfg['OCC_MIN'] <= occupancy[c] <= cfg['OCC_MAX']
    ]
    print(f"{len(all_neuron_columns)} total neuron columns, {len(sweet_spot_neurons)} in the "
          f"[{cfg['OCC_MIN']:.1%}, {cfg['OCC_MAX']:.1%}] occupancy sweet spot", flush=True)

    scan_pool = sweet_spot_neurons[:cfg['SCAN_POOL_SIZE']]
    print(f'scan pool ({len(scan_pool)} neurons): {scan_pool}', flush=True)
    return scan_pool


def convert_scan_pool_to_tsls(df, scan_pool, results_dir):
    time_grid = df['time_bin_center'].to_numpy()

    scan_tsls_df = pd.DataFrame({'time_bin_center': time_grid})
    for neuron in scan_pool:
        spike_times = df.loc[df[neuron] > 0, 'time_bin_center'].to_numpy()
        scan_tsls_df[f'{neuron}_tsls'] = tsls.compute_time_since_previous_spike(spike_times, time_grid)

    scan_tsls_df.to_csv(results_dir / 'tsls_scan_pool.csv', index=False)
    print(f'Converted {len(scan_pool)} neurons to TSLS over {len(time_grid)} bins '
          f'-> tsls_scan_pool.csv', flush=True)
    return scan_tsls_df, time_grid


def pairwise_correlation_scan(scan_tsls_df, scan_pool, cfg, results_dir):
    pair_scores = []
    for a, b in itertools.combinations(scan_pool, 2):
        result = tsls.proper_simplex(
            scan_tsls_df, columns=f'{a}_tsls', target=f'{b}_tsls',
            E=cfg['DEMO_E'], tau=cfg['DEMO_TAU'], Tp=cfg['DEMO_TP'], metric='correlation',
        )
        r = result['metrics']['correlation']
        if np.isfinite(r):
            pair_scores.append((r, a, b))
    pair_scores.sort(reverse=True)

    pair_scan_df = pd.DataFrame(pair_scores, columns=['correlation', 'source', 'target'])
    pair_scan_df.to_csv(results_dir / 'pairwise_correlation_scan.csv', index=False)

    best_r, best_source, best_target = pair_scores[0]
    print(f'Scanned {len(pair_scores)} pairs -> pairwise_correlation_scan.csv', flush=True)
    print(f'Best pair: {best_source} -> {best_target}  '
          f"(E={cfg['DEMO_E']}, tau={cfg['DEMO_TAU']}, Tp={cfg['DEMO_TP']}, correlation={best_r:.3f})",
          flush=True)
    return pair_scores, pair_scan_df, best_source, best_target


def select_working_set(pair_scores, scan_tsls_df, best_source, best_target, cfg, results_dir):
    working_neurons = []
    for _, a, b in pair_scores:
        for x in (a, b):
            if x not in working_neurons:
                working_neurons.append(x)
        if len(working_neurons) >= cfg['NUM_WORKING_NEURONS']:
            break
    working_neurons = working_neurons[:cfg['NUM_WORKING_NEURONS']]

    target = f'{best_target}_tsls'
    source = f'{best_source}_tsls'
    other_columns = [f'{n}_tsls' for n in working_neurons if n != best_target]

    tsls_df = scan_tsls_df[['time_bin_center'] + [f'{n}_tsls' for n in working_neurons]].copy()
    tsls_df.to_csv(results_dir / 'tsls_data.csv', index=False)

    print(f'Working neuron set: {working_neurons}', flush=True)
    print(f'TARGET={target!r}  SOURCE={source!r}', flush=True)
    print(f'OTHER_COLUMNS (proper_mde candidate pool) = {other_columns}', flush=True)
    print('-> tsls_data.csv', flush=True)
    return tsls_df, target, source, other_columns


def sanity_check_and_plot(tsls_df, target, df, best_target, time_grid, results_dir):
    target_tsls_vals = tsls_df[target].to_numpy()
    self_metrics = tsls.adaptive_event_metrics_from_tsls(target_tsls_vals, target_tsls_vals)
    print(f'{target} self-comparison (sanity check, should be precision=recall=f1=1.0): {self_metrics}',
          flush=True)

    near_zero_thr = tsls.infer_near_zero_threshold_from_true_tsls(target_tsls_vals)
    detected_events = tsls.detect_reset_events_from_tsls(target_tsls_vals, near_zero_threshold=near_zero_thr)
    true_spike_bins = np.where(df[best_target].to_numpy() > 0)[0]
    print(f'{target}: {len(true_spike_bins)} true spike bins vs. {len(detected_events)} reset events '
          f'decoded from TSLS (near-zero threshold={near_zero_thr:.3f} s).', flush=True)

    plot_rows = min(1500, len(time_grid))
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(time_grid[:plot_rows] - time_grid[0], target_tsls_vals[:plot_rows],
            color='tab:blue', lw=1, label=f'{target} (TSLS)')
    shown_events = detected_events[detected_events < plot_rows]
    ax.scatter(time_grid[shown_events] - time_grid[0], target_tsls_vals[shown_events],
               color='tab:red', s=18, zorder=3, label='detected reset event')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Time since last spike (s)')
    ax.set_title(f'{target}: TSLS sawtooth with decoded reset events')
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(results_dir / f'tsls_example_sawtooth_{target}.png', dpi=300)
    plt.close(fig)


# =============================================================================
# Section 2: proper_simplex on the demo pair
# =============================================================================

def run_proper_simplex(tsls_df, source, target, cfg, results_dir):
    simplex_out = tsls.proper_simplex(
        tsls_df, columns=source, target=target,
        E=cfg['DEMO_E'], tau=cfg['DEMO_TAU'], Tp=cfg['DEMO_TP'],
        metric=['correlation', 'mae', 'f1', 'gamma'],
    )

    print(f"proper_simplex({source} -> {target}, E={cfg['DEMO_E']}, tau={cfg['DEMO_TAU']}, "
          f"Tp={cfg['DEMO_TP']}):", flush=True)
    for name, value in simplex_out['metrics'].items():
        line = f'  {name:>12s}: {value:.4f}' if np.isfinite(value) else f'  {name:>12s}: {value}'
        print(line, flush=True)

    with open(results_dir / 'proper_simplex_metrics.json', 'w') as f:
        json.dump(
            {k: (float(v) if np.isfinite(v) else None) for k, v in simplex_out['metrics'].items()},
            f, indent=2,
        )
    simplex_out['predictions'].to_csv(results_dir / 'proper_simplex_predictions.csv')
    print('-> proper_simplex_metrics.json, proper_simplex_predictions.csv', flush=True)

    pred_df = simplex_out['predictions']
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(pred_df.index, pred_df['Observations'], color='black', lw=1, alpha=0.8, label='Observed')
    ax.plot(pred_df.index, pred_df['Predictions'], color='tab:orange', lw=1, alpha=0.8, label='Predicted')
    ax.set_xlabel('Row index (prediction range)')
    ax.set_ylabel('TSLS (s)')
    ax.set_title(f'proper_simplex: {source} -> {target}  '
                 f"(correlation={simplex_out['metrics']['correlation']:.3f})")
    ax.legend(loc='upper right')
    fig.tight_layout()
    fig.savefig(results_dir / f'tsls_example_simplex_{source}_to_{target}.png', dpi=300)
    plt.close(fig)
    return simplex_out


# =============================================================================
# Section 3: proper_ccm on the demo pair
# =============================================================================

def run_proper_ccm(tsls_df, source, target, cfg, results_dir):
    n = tsls_df.shape[0]
    ccm_lib_sizes = [int(n * p / 100) for p in cfg['CCM_LIB_SIZE_PCTS']]

    c2t, t2c, lib_sizes_used = tsls.proper_ccm(
        tsls_df, libSizes=ccm_lib_sizes, sample=cfg['CCM_SAMPLE'],
        tau=cfg['DEMO_TAU'], Tp=cfg['DEMO_TP'], E=cfg['DEMO_E'],
        target=target, columns=source,
        target_measurement='correlation', columns_measurement='correlation',
        seed=cfg['CCM_SEED'],
    )
    mean_c2t, mean_t2c = np.nanmean(c2t, axis=0), np.nanmean(t2c, axis=0)
    se_c2t = np.nanstd(c2t, axis=0) / np.sqrt(c2t.shape[0])
    se_t2c = np.nanstd(t2c, axis=0) / np.sqrt(t2c.shape[0])

    print(f'proper_ccm({source} <-> {target}), library sizes {lib_sizes_used}:', flush=True)
    print(f'  c2t ({source} predicts {target}): {np.round(mean_c2t, 3)}', flush=True)
    print(f'  t2c ({target} predicts {source}): {np.round(mean_t2c, 3)}', flush=True)

    ccm_out_df = pd.DataFrame({
        'libSize': lib_sizes_used,
        'c2t_mean': mean_c2t, 'c2t_se': se_c2t,
        't2c_mean': mean_t2c, 't2c_se': se_t2c,
    })
    ccm_csv_path = results_dir / f'proper_ccm_{source}_{target}.csv'
    ccm_out_df.to_csv(ccm_csv_path, index=False)
    print(f'-> {ccm_csv_path.name}', flush=True)

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.errorbar(lib_sizes_used, mean_c2t, yerr=se_c2t, color='black', marker='o',
                 label=f'{source} : {target}')
    ax1.set_ylabel(f'{source} predicts {target} (correlation)', color='black')
    ax2 = ax1.twinx()
    ax2.errorbar(lib_sizes_used, mean_t2c, yerr=se_t2c, color='tab:blue', marker='o',
                 label=f'{target} : {source}')
    ax2.set_ylabel(f'{target} predicts {source} (correlation)', color='tab:blue')
    ax1.set_xlabel('Library size (rows)')
    ax1.set_title('CCM convergence: skill should rise with library size if a real dynamical link exists')
    fig.tight_layout()
    fig.savefig(results_dir / f'tsls_example_ccm_{source}_{target}.png', dpi=300)
    plt.close(fig)


# =============================================================================
# Section 4: proper_mde -- data-driven target/candidate pool, then two runs
# =============================================================================

def select_mde_target_and_candidates(pair_scan_df, scan_tsls_df, cfg):
    mde_target_pool = []
    for _, row in pair_scan_df.iterrows():
        for x in (row['source'], row['target']):
            if x not in mde_target_pool:
                mde_target_pool.append(x)
        if len(mde_target_pool) >= cfg['MDE_TARGET_POOL_SIZE']:
            break
    print(f'candidate MDE targets: {mde_target_pool}', flush=True)

    n_probe = scan_tsls_df.shape[0]
    probe_lib_sizes = [int(n_probe * p / 100) for p in cfg['PROBE_LIB_SIZE_PCTS']]

    target_convergence = []
    for tgt in mde_target_pool:
        partner_rows = pair_scan_df[(pair_scan_df.source == tgt) | (pair_scan_df.target == tgt)]
        drivers = []
        for _, row in partner_rows.iterrows():
            other = row['target'] if row['source'] == tgt else row['source']
            if other not in drivers:
                drivers.append(other)
            if len(drivers) >= cfg['MDE_DRIVERS_PER_TARGET']:
                break

        best_slope = -np.inf
        for drv in drivers:
            _, t2c, _ = tsls.proper_ccm(
                scan_tsls_df, libSizes=probe_lib_sizes, sample=cfg['PROBE_SAMPLE'],
                tau=cfg['DEMO_TAU'], Tp=cfg['DEMO_TP'], E=cfg['DEMO_E'],
                target=f'{tgt}_tsls', columns=f'{drv}_tsls',
                target_measurement='correlation', columns_measurement='correlation',
                seed=cfg['CCM_SEED'],
            )
            mean_t2c = np.nanmean(t2c, axis=0)
            if np.all(np.isnan(mean_t2c)):
                continue
            best_slope = max(best_slope, mean_t2c[-1] - mean_t2c[0])
        target_convergence.append((best_slope, tgt, drivers))

    target_convergence.sort(reverse=True)
    print('Candidate MDE targets ranked by best driver t2c convergence (probe):', flush=True)
    for s, tgt, drivers in target_convergence:
        print(f'  {tgt:8s}  best_slope_proxy={s:+.4f}  drivers={drivers}', flush=True)

    _, mde_target_name, mde_driver_names = target_convergence[0]
    mde_target = f'{mde_target_name}_tsls'
    mde_candidates = [f'{d}_tsls' for d in mde_driver_names]
    mde_tsls_df = scan_tsls_df[['time_bin_center', mde_target] + mde_candidates].copy()

    print(f'MDE_TARGET={mde_target!r}', flush=True)
    print(f'MDE_CANDIDATES={mde_candidates}', flush=True)
    return mde_tsls_df, mde_target, mde_candidates


def _serialize_mde(result):
    return {
        'MDEcolumns': result['MDEcolumns'],
        'MDErho': [float(x) for x in result['MDErho']],
        'EDim': result['EDim'],
        'libSizes': result['libSizes'],
        'rhoD': {
            str(d): [{'score': float(s), 'columns': cl} for s, cl in entries]
            for d, entries in result['rhoD'].items()
        },
    }


def run_proper_mde_pair(mde_tsls_df, mde_target, mde_candidates, cfg, results_dir):
    mde_rank_only = tsls.proper_mde(
        mde_tsls_df, target=mde_target, columns=mde_candidates,
        D=cfg['MDE_D'], tau=cfg['DEMO_TAU'], Tp=cfg['DEMO_TP'], E=cfg['DEMO_E'],
        metric='correlation', noCCM=True, crossMapRhoMin=cfg['MDE_RANK_ONLY_CROSS_MAP_RHO_MIN'],
        verbose=True,
    )
    print('Ranking-only (noCCM=True) selection:', mde_rank_only['MDEcolumns'], flush=True)
    print('scores:', np.round(mde_rank_only['MDErho'], 4), flush=True)

    mde_full = tsls.proper_mde(
        mde_tsls_df, target=mde_target, columns=mde_candidates,
        D=cfg['MDE_D'], tau=cfg['DEMO_TAU'], Tp=cfg['DEMO_TP'], E=cfg['DEMO_E'],
        metric='correlation', ccmMetric='correlation',
        sample=cfg['MDE_FULL_SAMPLE'], pLibSizes=cfg['MDE_FULL_P_LIB_SIZES'],
        crossMapRhoMin=cfg['MDE_FULL_CROSS_MAP_RHO_MIN'], embedDimRhoMin=cfg['MDE_FULL_EMBED_DIM_RHO_MIN'],
        verbose=True,
    )
    print('Full CCM-gated selection:', mde_full['MDEcolumns'], flush=True)
    print('scores:', np.round(mde_full['MDErho'], 4), flush=True)

    with open(results_dir / 'proper_mde_rank_only.json', 'w') as f:
        json.dump(_serialize_mde(mde_rank_only), f, indent=2)
    with open(results_dir / 'proper_mde_full_ccm_gated.json', 'w') as f:
        json.dump(_serialize_mde(mde_full), f, indent=2)
    print('-> proper_mde_rank_only.json, proper_mde_full_ccm_gated.json', flush=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if mde_rank_only['MDEcolumns']:
        dims = np.arange(1, len(mde_rank_only['MDEcolumns']) + 1)
        ax.bar(dims, mde_rank_only['MDErho'], color='tab:blue', alpha=0.6,
               label='noCCM=True (ranking only)')
        for x, y, name in zip(dims, mde_rank_only['MDErho'], mde_rank_only['MDEcolumns']):
            ax.annotate(name, (x, y), xytext=(0, 4), textcoords='offset points',
                        ha='center', fontsize=8, color='tab:blue')
    if mde_full['MDEcolumns']:
        dims_full = np.arange(1, len(mde_full['MDEcolumns']) + 1)
        ax.plot(dims_full, mde_full['MDErho'], color='tab:red', marker='o', lw=2,
                label='noCCM=False (CCM-gated)')
        for x, y, name in zip(dims_full, mde_full['MDErho'], mde_full['MDEcolumns']):
            ax.annotate(name, (x, y), xytext=(0, -12), textcoords='offset points',
                        ha='center', fontsize=8, color='tab:red')
    if not mde_rank_only['MDEcolumns'] and not mde_full['MDEcolumns']:
        ax.text(0.5, 0.5, 'no columns selected', ha='center', va='center', transform=ax.transAxes)
    max_dim = max(len(mde_rank_only['MDEcolumns']), len(mde_full['MDEcolumns']), 1)
    ax.set_xticks(np.arange(1, max_dim + 1))
    ax.set_xlabel('MDE dimension')
    ax.set_ylabel('Ranking-stage correlation')
    ax.set_title(f'proper_mde selection for {mde_target}: ranking-only vs. CCM-gated')
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(results_dir / f'tsls_example_mde_ranking_{mde_target}.png', dpi=300)
    plt.close(fig)

    return mde_rank_only, mde_full


# =============================================================================
# Orchestration
# =============================================================================

def main():
    cfg = load_config()
    print('Config:', cfg, flush=True)
    results_dir = cfg['RESULTS_DIR']

    df = load_and_window_data(cfg)
    scan_pool = select_scan_pool(df, cfg)
    scan_tsls_df, time_grid = convert_scan_pool_to_tsls(df, scan_pool, results_dir)

    pair_scores, pair_scan_df, best_source, best_target = pairwise_correlation_scan(
        scan_tsls_df, scan_pool, cfg, results_dir,
    )
    tsls_df, target, source, _other_columns = select_working_set(
        pair_scores, scan_tsls_df, best_source, best_target, cfg, results_dir,
    )
    sanity_check_and_plot(tsls_df, target, df, best_target, time_grid, results_dir)

    run_proper_simplex(tsls_df, source, target, cfg, results_dir)
    run_proper_ccm(tsls_df, source, target, cfg, results_dir)

    mde_tsls_df, mde_target, mde_candidates = select_mde_target_and_candidates(
        pair_scan_df, scan_tsls_df, cfg,
    )
    run_proper_mde_pair(mde_tsls_df, mde_target, mde_candidates, cfg, results_dir)

    print('Pipeline complete.', flush=True)


if __name__ == "__main__":
    main()
