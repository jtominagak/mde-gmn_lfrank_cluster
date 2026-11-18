"""Autonomous-rollout ensemble, one-step accuracy and graph summary
(exp/spike-conversion-neuro).

Runs after the parent pipeline's own final Generate()/Forecast(). Three outputs
per network, all under results/rollouts/:

  rollout_metrics_<id>.csv   one row per (start, horizon): x/y correlation,
                             NRMSE, speed-weighted correlation, Euclidean path
                             error; unit-trace correlation/NRMSE; and three
                             population-state checks (below).
  rollout_<id>_start<row>.csv  the generated series, for plotting.
  onestep_metrics_<id>.csv   Forecast() over the whole held-out segment
                             [B, end): per-node correlation and NRMSE.
  graph_summary_<id>.json    edge counts, density, region-pair edge counts,
                             the targets' direct drivers.

THE ROLLOUTS ARE INPUT-FREE only when EXTERNAL_INPUT_MODE='self_simulate'
(the ablation sbatch sets it). Under 'real_data' zero-driver leaves replay the
recording, and this module says so in its log line and in every metrics row
(`external_input_mode` column), so a 'real_data' row can never be mistaken
for an autonomous one.

POPULATION-STATE CHECKS. The generated unit vector z(t) (only units that are
network nodes, each standardised by its library mean/sd) is compared with the
recording over the same rows:
  manifold_ratio   mean distance from each generated state to its nearest
                   LIBRARY state, divided by the same for the real states.
                   About 1 = the run stays on the empirical manifold; >> 1 =
                   it left it.
  motion_ratio     sd of z over the last 1 s of the horizon, generated / real.
                   Near 0 = collapsed to a fixed point.
  blew_up          any non-finite value, or |z| above 10x the library's max.

BASELINES. One-step rows carry a persistence forecast (x[t-Tp] predicts
x[t]) and skill_vs_persistence = 1 - NRMSE_model / NRMSE_persistence; rollout
rows carry a hold-the-seed baseline (`*_hold`). At 40 Hz a smooth series is
almost persistent over one step, so raw one-step correlations near 1 say
little on their own.

NRMSE is RMSE divided by the sd of the real series over the whole held-out
segment, a fixed normaliser, so short horizons are not judged against the
small variance of a short window.

Depends on: gmn_build, network, columns; gmn, numpy, pandas, scipy.
Imported by: pipeline.
"""

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import gmn
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from mde_gmn.columns import DELAY_MARKER
from mde_gmn.gmn_build import build_gmn_config
from mde_gmn.gmn_embedding import network_max_embedding_offset
from mde_gmn.progress import _log_progress

BLOW_UP_FACTOR = 10.0
LIB_TREE_MAX_ROWS = 20000


def rollout_starts(validation_start, total_rows, length, n_starts):
    '''Evenly spaced start rows covering [B, end - length]. Fewer when the
    segment is too short to hold n distinct starts.'''
    last = total_rows - length
    if n_starts <= 0 or last < validation_start:
        return []
    return sorted(set(np.linspace(validation_start, last, n_starts).round().astype(int).tolist()))


def _flat(x):
    return x.std() <= 1e-9 * (np.abs(x).max() + 1e-12)


def _corr(a, b, w=None):
    m = np.isfinite(a) & np.isfinite(b)
    if w is not None:
        m &= np.isfinite(w)
    if m.sum() < 3:
        return np.nan
    a, b = a[m], b[m]
    # A series that is constant up to float noise (a run collapsed to a fixed
    # point) has no defined correlation; without the tolerance it would
    # report the correlation of rounding error.
    if _flat(a) or _flat(b):
        return np.nan
    if w is None:
        return float(np.corrcoef(a, b)[0, 1])
    w = w[m]
    if w.sum() <= 0:
        return np.nan
    w = w / w.sum()
    am, bm = (w * a).sum(), (w * b).sum()
    cov = (w * (a - am) * (b - bm)).sum()
    va, vb = (w * (a - am) ** 2).sum(), (w * (b - bm) ** 2).sum()
    return float(cov / np.sqrt(va * vb)) if va > 0 and vb > 0 else np.nan


def _nrmse(a, b, scale):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 1 or not scale or not np.isfinite(scale):
        return np.nan
    return float(np.sqrt(np.mean((a[m] - b[m]) ** 2)) / scale)


def unit_nodes(columns, targets):
    return [c for c in columns if c not in targets and c != 'Time' and DELAY_MARKER not in c]


class Reference:
    '''Everything a score needs from the real data, computed once per run.'''

    def __init__(self, df, targets, prediction_start, validation_start, dt):
        self.df = df
        self.targets = list(targets)
        self.dt = dt
        held = df.iloc[validation_start:]
        self.scale = {c: float(held[c].std()) for c in df.columns if c != 'time_bin_center'}
        lib = df.iloc[:prediction_start]
        self.lib_mean = lib.mean(numeric_only=True)
        self.lib_sd = lib.std(numeric_only=True).replace(0, 1.0)
        self.lib = lib
        self._trees = {}
        if len(self.targets) >= 2:
            x, y = df[self.targets[0]].to_numpy(), df[self.targets[1]].to_numpy()
            self.speed = np.hypot(np.gradient(x), np.gradient(y)) / dt
        else:
            self.speed = None

    def tree(self, units):
        key = tuple(units)
        if key not in self._trees:
            z = ((self.lib[units] - self.lib_mean[units]) / self.lib_sd[units]).to_numpy()
            z = z[np.isfinite(z).all(axis=1)]
            if len(z) > LIB_TREE_MAX_ROWS:
                z = z[np.linspace(0, len(z) - 1, LIB_TREE_MAX_ROWS).astype(int)]
            self._trees[key] = (cKDTree(z), float(np.abs(z).max()) if len(z) else 1.0)
        return self._trees[key]

    def z(self, frame, units):
        return ((frame[units] - self.lib_mean[units]) / self.lib_sd[units]).to_numpy(dtype=float)


def score_rollout(generated, start, ref, horizon_rows):
    '''Metrics for one free run. `generated`: DataFrame of node columns,
    row i is recording row start + i. Returns one dict per horizon.'''
    rows = []
    units = unit_nodes(generated.columns, ref.targets)
    for h in horizon_rows:
        h_eff = min(h, len(generated))
        gen = generated.iloc[:h_eff].reset_index(drop=True)
        act = ref.df.iloc[start:start + h_eff].reset_index(drop=True)
        row = {'start': start, 'horizon_rows': h, 'horizon_s': round(h * ref.dt, 3),
               'rows_scored': h_eff}
        for t in ref.targets:
            if t in gen.columns:
                g, a = gen[t].to_numpy(float), act[t].to_numpy(float)
                row[f'corr_{t}'] = _corr(a, g)
                row[f'nrmse_{t}'] = _nrmse(a, g, ref.scale[t])
                if ref.speed is not None:
                    row[f'vwcorr_{t}'] = _corr(a, g, ref.speed[start:start + h_eff])
        if len(ref.targets) >= 2 and all(t in gen.columns for t in ref.targets[:2]):
            tx, ty = ref.targets[:2]
            err = np.hypot(gen[tx].to_numpy(float) - act[tx].to_numpy(float),
                           gen[ty].to_numpy(float) - act[ty].to_numpy(float))
            row['path_err_mean'] = float(np.nanmean(err))
            row['path_err_final'] = float(err[-1])
            # Naive baseline: stay where the seed left you.
            hold = np.hypot(act[tx].to_numpy(float) - act[tx].iloc[0],
                            act[ty].to_numpy(float) - act[ty].iloc[0])
            row['path_err_mean_hold'] = float(np.nanmean(hold))
            row['path_err_final_hold'] = float(hold[-1])
        row['n_units'] = len(units)
        if units:
            uc = [_corr(act[u].to_numpy(float), gen[u].to_numpy(float)) for u in units]
            un = [_nrmse(act[u].to_numpy(float), gen[u].to_numpy(float), ref.scale[u]) for u in units]
            row['unit_corr_mean'] = float(np.nanmean(uc)) if np.isfinite(uc).any() else np.nan
            row['unit_corr_median'] = float(np.nanmedian(uc)) if np.isfinite(uc).any() else np.nan
            row['unit_nrmse_median'] = float(np.nanmedian(un)) if np.isfinite(un).any() else np.nan
            uh = [_nrmse(act[u].to_numpy(float), np.full(h_eff, act[u].iloc[0]), ref.scale[u])
                  for u in units]
            row['unit_nrmse_median_hold'] = float(np.nanmedian(uh)) if np.isfinite(uh).any() else np.nan
            tree, lib_max = ref.tree(units)
            zg, za = ref.z(gen, units), ref.z(act, units)
            row['blew_up'] = bool((~np.isfinite(zg)).any() or np.abs(zg[np.isfinite(zg)]).max(initial=0)
                                  > BLOW_UP_FACTOR * lib_max)
            okg = np.isfinite(zg).all(axis=1)
            dg = tree.query(zg[okg])[0].mean() if okg.any() else np.nan
            da = tree.query(za)[0].mean()
            row['manifold_ratio'] = float(dg / da) if da > 0 else np.nan
            tail = max(2, int(round(1.0 / ref.dt)))
            sg = np.nanstd(zg[-tail:], axis=0).mean()
            sa = np.nanstd(za[-tail:], axis=0).mean()
            row['motion_ratio'] = float(sg / sa) if sa > 0 else np.nan
        rows.append(row)
    return rows


def graph_summary(graph, targets):
    def region(n):
        if n in targets:
            return 'behaviour'
        return n.split('_', 1)[0]
    nodes = list(graph.nodes)
    units = [n for n in nodes if n not in targets and 'lagged_of' not in graph.nodes[n]]
    lagged = [n for n in nodes if 'lagged_of' in graph.nodes[n]]
    pairs = {}
    for u, v in graph.edges():
        src = graph.nodes[u].get('lagged_of', u)
        key = f'{region(src)}->{region(v)}'
        pairs[key] = pairs.get(key, 0) + 1
    n_real = len(units) + len([t for t in targets if t in graph])
    indeg = [graph.in_degree(n) for n in nodes if n not in lagged]
    return {
        'n_nodes': len(nodes), 'n_unit_nodes': len(units), 'n_lagged_leaves': len(lagged),
        'n_edges': graph.number_of_edges(),
        'density_real': graph.number_of_edges() / max(1, n_real * (n_real - 1)),
        'mean_in_degree': float(np.mean(indeg)) if indeg else 0.0,
        'max_in_degree': int(max(indeg)) if indeg else 0,
        'region_pair_edges': pairs,
        'target_drivers': {t: [(u, d.get('delay')) for u, _, d in graph.in_edges(t, data=True)]
                           for t in targets if t in graph},
        'unit_nodes': sorted(units),
        'external_input_mode': graph.graph.get('external_input_mode'),
    }


def _generate(cfg_file, cores):
    '''One free run from a written config; module-level so a spawned worker
    can unpickle it (importing this module applies the gmn patches).'''
    G = gmn.GMN(configFile=cfg_file, cores=cores)
    G.Generate()
    return G.DataOut


def _onestep_forecast(network_id, net_result, theta, graph, cfg, df, ref, tag, out_dir):
    # One-step accuracy: Forecast() over the whole held-out segment.
    try:
        B = cfg['GENERATE_VALIDATION_START']
        cfg_file = build_gmn_config(
            net_result, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'], cfg['RESULTS_DIR'],
            mode='Forecast', node_theta_overrides=theta, variant='onestep',
            generate_length=len(df) - B, generate_start=B,
            max_offset=network_max_embedding_offset(graph),
            exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'])
        Gf = gmn.GMN(configFile=str(cfg_file), cores=cfg['CORES'])
        Gf.Forecast()
        fc = Gf.DataOut.dropna().reset_index(drop=True)
        fc.to_csv(out_dir / f'onestep_forecast_{network_id}.csv', index=False)
        idx = fc['Time'].to_numpy().astype(int)
        one = []
        for node in [c for c in fc.columns if c != 'Time' and DELAY_MARKER not in c]:
            a, p = df[node].to_numpy(float)[idx], fc[node].to_numpy(float)
            # Persistence baseline: predict the last observed value. At
            # 25 ms a smooth trace is nearly persistent, so one-step skill
            # only means something relative to this.
            persist = df[node].to_numpy(float)[idx - int(net_result.get('gmn_tp', 1))]
            n_model = _nrmse(a, p, ref.scale[node])
            n_pers = _nrmse(a, persist, ref.scale[node])
            one.append({**tag, 'network_id': network_id, 'node': node,
                        'role': 'target' if node in net_result['targets'] else 'unit',
                        'corr': _corr(a, p), 'nrmse': n_model,
                        'corr_persistence': _corr(a, persist), 'nrmse_persistence': n_pers,
                        'skill_vs_persistence': (1 - n_model / n_pers) if n_pers else np.nan,
                        'rows': len(idx)})
        pd.DataFrame(one).to_csv(out_dir / f'onestep_metrics_{network_id}.csv', index=False)
    except Exception as exc:
        print(f'{network_id}: one-step Forecast() over [B, end) FAILED: {exc!r}', flush=True)


def run_rollout_evaluation(gmn_runs, cfg, df, progress_log_path=None):
    '''The whole step, for every network that has a final Generate().'''
    out_dir = cfg['RESULTS_DIR'] / 'rollouts'
    out_dir.mkdir(exist_ok=True)
    dt = float(np.median(np.diff(df['time_bin_center'].to_numpy())))
    ref = Reference(df, cfg['TARGETS'], cfg['PREDICTION_START'], cfg['GENERATE_VALIDATION_START'], dt)
    horizon_rows = [int(round(s / dt)) for s in cfg['ROLLOUT_HORIZONS_S']]
    length = max([cfg['ROLLOUT_LENGTH']] + horizon_rows)
    starts = rollout_starts(cfg['GENERATE_VALIDATION_START'], len(df), length, cfg['ROLLOUT_N_STARTS'])
    tag = {'spike_transform': cfg['SPIKE_TRANSFORM'], 'external_input_mode': cfg['EXTERNAL_INPUT_MODE'],
           'embed_tau': cfg['EMBED_TAU']}
    if cfg['EXTERNAL_INPUT_MODE'] != 'self_simulate':
        print("ROLLOUTS ARE NOT INPUT-FREE: EXTERNAL_INPUT_MODE='real_data' replays the recording "
              'for zero-driver leaves', flush=True)

    for network_id, run in gmn_runs.items():
        net_result = run['net_result']
        graph = net_result['network']['Graph']
        summary = graph_summary(graph, net_result['targets'])
        with open(out_dir / f'graph_summary_{network_id}.json', 'w') as f:
            json.dump({**summary, **tag}, f, indent=1, default=str)
        if 'G' not in run:
            continue
        theta = run['best_node_theta']

        # Build every rollout's config in the parent (cheap, writes files),
        # then run the Generate()s. gmn's Generate() is serial whatever
        # `cores` says, so with ROLLOUT_WORKERS > 1 the starts run in a
        # spawned process pool (spawn: the parent may hold CUDA state) while
        # the parent runs the one-step Forecast(). Results are identical to
        # the serial path -- GENERATE_NOISE is off, every run is
        # deterministic and reads only its own config files.
        jobs = []
        for k, s in enumerate(starts):
            try:
                cfg_file = build_gmn_config(
                    net_result, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'], cfg['RESULTS_DIR'],
                    mode='Generate', node_theta_overrides=theta, variant=f'rollout{k}',
                    generate_length=length, generate_start=s,
                    exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'])
                jobs.append((k, s, str(cfg_file)))
            except Exception as exc:
                print(f'{network_id}: rollout config for row {s} FAILED: {exc!r}', flush=True)
        workers = min(int(cfg.get('ROLLOUT_WORKERS', 1)), len(jobs))
        onestep_args = (network_id, net_result, theta, graph, cfg, df, ref, tag, out_dir)
        t_pool = time.perf_counter()
        if workers > 1:
            _log_progress(progress_log_path, f'{network_id}: {len(jobs)} rollouts on {workers} '
                                             f'spawned workers; one-step Forecast() in the parent')
            with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
                futures = [(k, s, pool.submit(_generate, f, 1)) for k, s, f in jobs]
                _onestep_forecast(*onestep_args)
                outs = []
                for k, s, fut in futures:
                    try:
                        outs.append((k, s, fut.result()))
                    except Exception as exc:
                        print(f'{network_id}: rollout from row {s} FAILED: {exc!r}', flush=True)
                        continue
                    _log_progress(progress_log_path, f'{network_id}: rollout {k + 1}/{len(starts)} '
                                                     f'(row {s}) done at {time.perf_counter() - t_pool:.1f}s')
        else:
            outs = []
            for k, s, f in jobs:
                t0 = time.perf_counter()
                try:
                    outs.append((k, s, _generate(f, cfg['CORES'])))
                except Exception as exc:
                    print(f'{network_id}: rollout from row {s} FAILED: {exc!r}', flush=True)
                    continue
                _log_progress(progress_log_path, f'{network_id}: rollout {k + 1}/{len(starts)} '
                                                 f'(row {s}) {time.perf_counter() - t0:.1f}s')

        rows = []
        for k, s, data_out in outs:
            gen = data_out.reset_index(drop=True).drop(columns=['Time'], errors='ignore')
            gen.insert(0, 'row', np.arange(s, s + len(gen)))
            gen.to_csv(out_dir / f'rollout_{network_id}_start{s}.csv', index=False)
            for r in score_rollout(gen.drop(columns=['row']), s, ref, horizon_rows):
                rows.append({**tag, 'network_id': network_id, **r})
        pd.DataFrame(rows).to_csv(out_dir / f'rollout_metrics_{network_id}.csv', index=False)
        if workers <= 1:
            _onestep_forecast(*onestep_args)
        print(f'{network_id}: rollout evaluation written to {out_dir}', flush=True)
