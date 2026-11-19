"""Autonomous (input-free) evaluation of a built network
(exp/spike-conversion-general).

Runs after the parent pipeline's own final Generate()/Forecast(). Per network,
under RESULTS_DIR/autonomous/:

  onestep_<id>.csv     Forecast() over the WHOLE held-out segment [B, end):
                       per node, Pearson r and NRMSE, next to a persistence
                       forecast (x[t-1] predicts x[t]) and the skill
                       1 - NRMSE / NRMSE_persistence. At 40 Hz a smooth
                       signal is almost persistent over one step, so a raw
                       one-step r near 1 means little without that column.
  rollouts_<id>.csv    one row per (start, horizon): x/y correlation,
                       NRMSE, speed-weighted correlation, Euclidean path
                       error (mean and at the horizon) with a hold-the-seed
                       baseline, mean unit correlation / NRMSE, and the
                       attractor checks below.
  rollouts_<id>.npz    the generated and real series for every start.
  topology_<id>.json   persistence (H0, H1) of delay-embedded trajectories:
                       generated windows vs the recorded windows at the same
                       rows; equal windows from the training rows give the
                       noise floor (two independent sets of real windows).
  graph_<id>.json      edge counts, density, degree, region pairs, the
                       targets' direct drivers.
  figures              example rollouts and persistence diagrams.

INPUT-FREE. Rollouts are autonomous only when the graph was partitioned with
EXTERNAL_INPUT_MODE='self_simulate' (every node, zero-driver leaves included,
is simulated from its own history; nothing is read from the recording after
the seed). Under 'real_data' zero-driver nodes replay the recording; the mode
is written into every row so such rows cannot be mistaken for autonomous ones.

SEEDING. Each rollout is gmn Generate() started at row s: every node's delay
vector is seeded from the recorded rows before s and the library is every row
before s (the parent pipeline's Generate() semantics), then the network runs
ROLLOUT_LENGTH steps on its own output. Starts are evenly spaced over
[B + warm-up, end - ROLLOUT_LENGTH]. Horizons (2, 5, 10 s) are prefixes of
the same rollout.

ATTRACTOR CHECKS, on the real-observable nodes (targets included), each
standardised by its library mean/sd:
  manifold_ratio  mean distance from each generated state to its nearest
                  library state over the same quantity for the real states
                  at those rows. ~1: stays on the empirical manifold; >>1:
                  left it.
  motion_ratio    sd over the last min(h, 2 s) rows, generated / real,
                  median over the nodes whose real sd there exceeds 0.01
                  library sd. ~0: collapsed to a fixed point; >>1: the run
                  oscillates far more than the recording.
  blew_up         any non-finite value or |z| > 10 x the library's max |z|.

Depends on: gmn_build, gmn_embedding, network, persistent_homology,
universal_point_process_embeddings; gmn, numpy, pandas, scipy, matplotlib.
Imported by: pipeline.
"""

import json
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use('Agg')
import gmn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from mde_gmn import persistent_homology as PH
from mde_gmn import universal_point_process_embeddings as U
from mde_gmn.columns import DELAY_MARKER
from mde_gmn.gmn_build import build_gmn_config
from mde_gmn.gmn_embedding import network_max_embedding_offset
from mde_gmn.progress import _log_progress

BLOW_UP_FACTOR = 10.0
MOTION_FLOOR_Z = 0.01      # library-sd units; below this a node is treated as static
LIB_TREE_MAX_ROWS = 20000


def rollout_starts(first, total_rows, length, n_starts):
    last = total_rows - length
    if n_starts <= 0 or last < first:
        return []
    return sorted(set(np.linspace(first, last, n_starts).round().astype(int).tolist()))


def _corr(a, b, w=None):
    ok = np.isfinite(a) & np.isfinite(b)
    if w is not None:
        ok &= np.isfinite(w)
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return np.nan
    if w is None:
        if a.std() <= 1e-12 or b.std() <= 1e-12:
            return np.nan
        return float(np.corrcoef(a, b)[0, 1])
    w = w[ok]
    if w.sum() <= 0:
        return np.nan
    w = w / w.sum()
    ma, mb = (w * a).sum(), (w * b).sum()
    va, vb = (w * (a - ma) ** 2).sum(), (w * (b - mb) ** 2).sum()
    if va <= 1e-24 or vb <= 1e-24:
        return np.nan
    return float((w * (a - ma) * (b - mb)).sum() / np.sqrt(va * vb))


def _nrmse(real, pred, scale):
    ok = np.isfinite(real) & np.isfinite(pred)
    if ok.sum() < 1 or not scale > 0:
        return np.nan
    return float(np.sqrt(np.mean((real[ok] - pred[ok]) ** 2)) / scale)


def _generate_worker(config_file):
    '''One Generate() in a forked worker; returns DataOut as a plain array.'''
    G = gmn.GMN(configFile=str(config_file), cores=1)
    G.Generate()
    out = G.DataOut.reset_index(drop=True)
    return list(out.columns), out.to_numpy(dtype=float)


def _real_nodes(nodes):
    return [n for n in nodes if DELAY_MARKER not in n]


def graph_summary(graph, targets):
    nodes = list(graph.nodes)
    real = _real_nodes(nodes)
    lagged = [n for n in nodes if DELAY_MARKER in n]

    def region(n):
        base = graph.nodes[n].get('lagged_of', n) if n in graph.nodes else n
        base = base.split(DELAY_MARKER)[0]
        return 'target' if base in targets else base.split('_')[0]

    edges = list(graph.edges(data=True))
    pairs = {}
    for u, v, _ in edges:
        k = f'{region(u)}->{region(v)}'
        pairs[k] = pairs.get(k, 0) + 1
    indeg = np.array([graph.in_degree(n) for n in nodes]) if nodes else np.array([0])
    n_real = len(real)
    return {
        'n_nodes': len(nodes), 'n_real_nodes': n_real, 'n_lagged_leaves': len(lagged),
        'n_edges': len(edges),
        'n_edges_from_lagged': sum(1 for u, _, _ in edges if DELAY_MARKER in u),
        'density_real': (sum(1 for u, v, _ in edges if DELAY_MARKER not in u)
                         / max(1, n_real * (n_real - 1))),
        'edges_per_node': len(edges) / max(1, len(nodes)),
        'in_degree_mean': float(indeg.mean()), 'in_degree_max': int(indeg.max()),
        'zero_driver_nodes': int((indeg == 0).sum()),
        'region_pairs': pairs,
        'target_drivers': {t: sorted(graph.predecessors(t)) for t in targets if t in graph},
        'real_channels': sorted(n for n in real if n not in targets),
    }


def _delay_cloud(X, lag):
    '''Rows (x(t), x(t-lag)) for t >= lag, X (T, k) -> (T-lag, 2k).'''
    if lag <= 0 or len(X) <= lag:
        return X
    return np.hstack([X[lag:], X[:-lag]])


def _topology(real_clouds, gen_clouds, lib_clouds, n_landmarks, scale_cloud):
    '''Diagrams and distances for three pooled clouds: generated windows, the
    recorded windows at the same rows, and the same number of equal-length
    windows from the training rows ('library', the noise floor). Every cloud
    is standardised by the whole training library's mean/sd (`scale_cloud`),
    so distances are in the same units for every method.'''
    R = np.vstack(real_clouds)
    Gc = np.vstack(gen_clouds)
    lib_cloud = np.vstack(lib_clouds)
    mu, sd = scale_cloud.mean(0), scale_cloud.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    out = {}
    clouds = {'real': (R - mu) / sd, 'generated': (Gc - mu) / sd, 'library': (lib_cloud - mu) / sd}
    diag = {}
    for name, C in clouds.items():
        C = C[np.all(np.isfinite(C), axis=1)]
        if len(C) < 10:
            diag[name] = (np.empty((0, 2)), np.empty((0, 2)))
            continue
        L = C[PH.maxmin_landmarks(C, n_landmarks)]
        diag[name] = PH.rips_diagrams(L)
    ref_scale = float(np.max(diag['library'][0][:, 1])) if len(diag['library'][0]) else 1.0
    thr = 0.1 * ref_scale
    scales = np.linspace(0, 2 * ref_scale, 60)
    for a, b in (('generated', 'real'), ('library', 'real'), ('generated', 'library')):
        for k, dim in ((0, 'H0'), (1, 'H1')):
            out[f'W1_{dim}_{a}_vs_{b}'] = PH.wasserstein(diag[a][k], diag[b][k])
            out[f'betti_L1_{dim}_{a}_vs_{b}'] = float(np.abs(
                PH.betti_curve(diag[a][k], scales) - PH.betti_curve(diag[b][k], scales)).mean())
    for name in clouds:
        for k, dim in ((0, 'H0'), (1, 'H1')):
            sm = PH.diagram_summary(diag[name][k], thr)
            out[f'{dim}_{name}_n_bars'] = sm['n_bars']
            out[f'{dim}_{name}_total_persistence'] = sm['total_persistence']
            out[f'{dim}_{name}_max_persistence'] = sm['max_persistence']
    out['persistence_threshold'] = thr
    return out, diag


def _plot_diagrams(diag, title, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    styles = {'real': ('k', 'o'), 'generated': ('tab:red', 'x'), 'library': ('tab:blue', '.')}
    for k, ax in enumerate(axes):
        hi = 0
        for name, (col, mk) in styles.items():
            d = diag[name][k]
            if len(d):
                ax.scatter(d[:, 0], d[:, 1], c=col, marker=mk, s=18, label=name, alpha=0.8)
                hi = max(hi, d.max())
        ax.plot([0, hi * 1.05], [0, hi * 1.05], color='gray', lw=0.8)
        ax.set_title(f'H{k}')
        ax.set_xlabel('birth'); ax.set_ylabel('death')
        ax.legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_rollouts(real, gen, targets, starts, dt, title, path, n_show=4):
    idx = np.linspace(0, len(starts) - 1, min(n_show, len(starts))).round().astype(int)
    fig, axes = plt.subplots(len(idx), 3, figsize=(13, 2.6 * len(idx)), squeeze=False)
    t = np.arange(real.shape[1]) * dt
    for r, i in enumerate(idx):
        for c, tgt in enumerate(targets[:2]):
            ax = axes[r, c]
            ax.plot(t, real[i, :, c], color='k', lw=1, label='recorded')
            ax.plot(t, gen[i, :, c], color='tab:red', lw=1, label='free run')
            ax.set_title(f'start row {starts[i]}: {tgt}', fontsize=9)
            if r == len(idx) - 1:
                ax.set_xlabel('s since seed')
        ax = axes[r, 2]
        ax.plot(real[i, :, 0], real[i, :, 1], color='k', lw=1)
        ax.plot(gen[i, :, 0], gen[i, :, 1], color='tab:red', lw=1)
        ax.scatter([real[i, 0, 0]], [real[i, 0, 1]], c='g', s=20, zorder=3)
        ax.set_title('x-y path (green = seed)', fontsize=9)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def run_one_step(net_result, theta, cfg, B, actual, graph, out_dir, cores):
    end = len(actual)
    max_offset = network_max_embedding_offset(graph)
    tp = int(net_result.get('gmn_tp', 1))
    cfg_file = build_gmn_config(net_result, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                                out_dir, mode='Forecast', node_theta_overrides=theta,
                                variant='onestep', generate_length=end - B, generate_start=B,
                                max_offset=max_offset, exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'])
    Gf = gmn.GMN(configFile=str(cfg_file), cores=cores)
    Gf.Forecast()
    fo = Gf.DataOut.dropna()
    # build_gmn_config() starts the window max_offset rows before B; score
    # only the held-out rows from B on.
    fo = fo[fo['Time'].to_numpy().astype(int) >= B].reset_index(drop=True)
    rows = fo['Time'].to_numpy().astype(int)
    targets = net_result['targets']
    recs = []
    for node in [c for c in fo.columns if c != 'Time']:
        a = actual[node].to_numpy()
        real = a[rows]
        pred = fo[node].to_numpy()
        persist = a[rows - tp]
        scale = float(np.nanstd(a[B:]))
        n_m, n_p = _nrmse(real, pred, scale), _nrmse(real, persist, scale)
        recs.append({
            'node': node, 'role': 'target' if node in targets else
                    ('lagged' if DELAY_MARKER in node else 'channel'),
            'r': _corr(real, pred), 'nrmse': n_m,
            'r_persistence': _corr(real, persist), 'nrmse_persistence': n_p,
            'skill_vs_persistence': (1 - n_m / n_p) if n_p and n_p > 0 else np.nan,
            'n_rows': len(rows)})
    return pd.DataFrame(recs)


def run_autonomous_evaluation(gmn_runs, cfg, df, obs_cols, progress_log_path=None):
    out_root = cfg['RESULTS_DIR'] / 'autonomous'
    out_root.mkdir(parents=True, exist_ok=True)
    dt = float(np.median(np.diff(df['time_bin_center'].to_numpy())))
    for network_id, run in gmn_runs.items():
        if 'G' not in run:
            print(f'{network_id}: no final Generate() -- skipping autonomous evaluation', flush=True)
            continue
        t0 = time.perf_counter()
        net_result = run['net_result']
        theta = run['best_node_theta']
        graph = net_result['network']['Graph']
        targets = net_result['targets']
        mode = graph.graph.get('external_input_mode', 'real_data')
        actual = pd.read_csv(net_result['DATA_AFTER_MDE'])
        nodes = [c for c in actual.columns if c != 'Time']
        real_nodes = _real_nodes(nodes)
        channels = [n for n in real_nodes if n not in targets]
        B = cfg['GENERATE_VALIDATION_START']
        end = len(actual)
        L = cfg['ROLLOUT_LENGTH']
        cfg_dir = out_root / f'cfg_{network_id}'
        cfg_dir.mkdir(exist_ok=True)

        summary = graph_summary(graph, targets)
        summary.update(external_input_mode=mode, conversion=cfg.get('CONVERSION_INFO'),
                       embed_tau_rows=cfg.get('EMBED_TAU_RESOLVED'),
                       n_candidate_channels=len(obs_cols))
        with open(out_root / f'graph_{network_id}.json', 'w') as f:
            json.dump(summary, f, indent=1, default=str)

        # ---- one-step over the whole held-out segment --------------------
        try:
            one = run_one_step(net_result, theta, cfg, B, actual, graph, cfg_dir, cfg['CORES'])
            one['external_input_mode'] = mode
            one.to_csv(out_root / f'onestep_{network_id}.csv', index=False)
            print(f'{network_id}: one-step over [{B}, {end}): '
                  + one.groupby('role')[['r', 'nrmse', 'skill_vs_persistence']].median()
                  .round(3).to_string().replace('\n', ' | '), flush=True)
        except Exception as exc:
            print(f'{network_id}: one-step Forecast FAILED: {exc!r}', flush=True)

        # ---- rollout ensemble -------------------------------------------
        warm = network_max_embedding_offset(graph) + 1
        starts = rollout_starts(B + warm, end, L, cfg['ROLLOUT_N_STARTS'])
        cfg_files = []
        for s in starts:
            cfg_files.append(build_gmn_config(
                net_result, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'], cfg_dir,
                mode='Generate', node_theta_overrides=theta, variant=f'roll{s}',
                generate_length=L, generate_start=s, exclusion_radius=cfg['GMN_EXCLUSION_RADIUS']))
        _log_progress(progress_log_path, f'{network_id}: {len(starts)} rollouts START '
                      f'({cfg["ROLLOUT_WORKERS"]} workers, mode={mode})')
        outs = {}
        ctx = mp.get_context('spawn')   # no fork of a process that has run pyEDM threads
        with ProcessPoolExecutor(max_workers=max(1, cfg['ROLLOUT_WORKERS']), mp_context=ctx) as ex:
            futs = {ex.submit(_generate_worker, cf): s for cf, s in zip(cfg_files, starts)}
            for fut in futs:
                s = futs[fut]
                try:
                    outs[s] = fut.result()
                except Exception as exc:
                    print(f'{network_id}: rollout from row {s} FAILED: {exc!r}', flush=True)
        _log_progress(progress_log_path, f'{network_id}: rollouts END '
                      f'{time.perf_counter() - t0:.0f}s, {len(outs)}/{len(starts)} ok')
        starts = [s for s in starts if s in outs]
        if not starts:
            continue

        lib_end = cfg['PREDICTION_START']
        lib = actual[real_nodes].to_numpy()[:lib_end]
        lib_mu, lib_sd = lib.mean(0), lib.std(0)
        lib_sd = np.where(lib_sd > 0, lib_sd, 1.0)
        zlib = (lib - lib_mu) / lib_sd
        zmax = np.abs(zlib).max(0)
        sub = zlib[np.linspace(0, len(zlib) - 1, min(len(zlib), LIB_TREE_MAX_ROWS)).astype(int)]
        tree = cKDTree(sub)
        test_sd = {n: float(actual[n].to_numpy()[B:].std()) for n in real_nodes}

        gen_all = np.full((len(starts), L, len(real_nodes)), np.nan)
        real_all = np.full_like(gen_all, np.nan)
        rows = []
        for i, s in enumerate(starts):
            cols, arr = outs[s]
            gdf = pd.DataFrame(arr, columns=cols)
            n = min(L, len(gdf))
            gen_all[i, :n] = gdf[real_nodes].to_numpy()[:n]
            real_all[i, :n] = actual[real_nodes].to_numpy()[s:s + n]
        ti = [real_nodes.index(t) for t in targets]
        ci = [real_nodes.index(c) for c in channels]
        for i, s in enumerate(starts):
            seed_xy = actual[targets].to_numpy()[s - 1]
            for H in cfg['ROLLOUT_HORIZONS_S']:
                h = int(round(H / dt))
                if h > L:
                    continue
                g = gen_all[i, :h]
                r = real_all[i, :h]
                rec = {'start': s, 'horizon_s': H, 'horizon_rows': h, 'external_input_mode': mode}
                vel = np.r_[0, np.linalg.norm(np.diff(r[:, ti], axis=0), axis=1)]
                for k, t in zip(ti, targets):
                    rec[f'r_{t}'] = _corr(r[:, k], g[:, k])
                    rec[f'nrmse_{t}'] = _nrmse(r[:, k], g[:, k], test_sd[t])
                    rec[f'vwr_{t}'] = _corr(r[:, k], g[:, k], w=vel)
                rec['r_xy'] = np.nanmean([rec[f'r_{t}'] for t in targets])
                rec['vwr_xy'] = np.nanmean([rec[f'vwr_{t}'] for t in targets])
                rec['nrmse_xy'] = np.nanmean([rec[f'nrmse_{t}'] for t in targets])
                d = np.linalg.norm(g[:, ti] - r[:, ti], axis=1)
                rec['path_err_mean'] = float(np.nanmean(d))
                rec['path_err_end'] = float(d[-1])
                rec['path_err_hold_mean'] = float(np.mean(np.linalg.norm(r[:, ti] - seed_xy, axis=1)))
                if ci:
                    rec['r_channels_mean'] = np.nanmean([_corr(r[:, k], g[:, k]) for k in ci])
                    rec['nrmse_channels_mean'] = np.nanmean(
                        [_nrmse(r[:, k], g[:, k], test_sd[real_nodes[k]]) for k in ci])
                zg = (g - lib_mu) / lib_sd
                zr = (r - lib_mu) / lib_sd
                rec['blew_up'] = bool((~np.isfinite(zg)).any()
                                      or (np.abs(zg) > BLOW_UP_FACTOR * np.maximum(zmax, 1)).any())
                if np.isfinite(zg).all():
                    dg, _ = tree.query(zg)
                    dr, _ = tree.query(zr)
                    rec['manifold_ratio'] = float(dg.mean() / max(dr.mean(), 1e-12))
                tail = min(h, int(round(2.0 / dt)))
                sg, sr = np.nanstd(zg[-tail:], axis=0), np.nanstd(zr[-tail:], axis=0)
                ok = sr > MOTION_FLOOR_Z          # nodes that really move in that window
                rec['motion_ratio'] = float(np.nanmedian(sg[ok] / sr[ok])) if ok.any() else np.nan
                rec['collapsed'] = bool(rec['motion_ratio'] < 0.05) if np.isfinite(rec['motion_ratio']) else False
                rows.append(rec)
        roll = pd.DataFrame(rows)
        roll.to_csv(out_root / f'rollouts_{network_id}.csv', index=False)
        np.savez_compressed(out_root / f'rollouts_{network_id}.npz', starts=np.array(starts),
                            nodes=np.array(real_nodes), targets=np.array(targets),
                            generated=gen_all, real=real_all)
        agg = roll.groupby('horizon_s')[['r_xy', 'vwr_xy', 'nrmse_xy', 'path_err_mean',
                                          'path_err_hold_mean', 'manifold_ratio',
                                          'motion_ratio']].median()
        print(f'{network_id}: rollout medians by horizon (mode={mode}):\n{agg.round(3)}', flush=True)

        # ---- topology ----------------------------------------------------
        topo = {}
        try:
            act_xy = actual[targets].to_numpy()
            xy_lag, _, _ = U.dataset_embedding_lag(act_xy, max_lag=40, method='ami', lib_end=lib_end)
            real_c = [_delay_cloud(real_all[i][:, ti], xy_lag) for i in range(len(starts))]
            gen_c = [_delay_cloud(gen_all[i][:, ti], xy_lag) for i in range(len(starts))]
            # Reference: as many windows of the same length from the TRAINING
            # rows, so "library vs real" is the distance between two
            # independent sets of real windows -- the noise floor for
            # "generated vs real". Standardisation uses the whole library.
            lib_starts = np.linspace(xy_lag, lib_end - L, len(starts)).round().astype(int)
            lib_c = [_delay_cloud(act_xy[a:a + L], xy_lag) for a in lib_starts]
            t_xy, d_xy = _topology(real_c, gen_c, lib_c, cfg['PH_LANDMARKS'],
                                   scale_cloud=_delay_cloud(act_xy[:lib_end], xy_lag))
            topo['xy'] = {'delay_lag_rows': xy_lag, **t_xy}
            _plot_diagrams(d_xy, f'{network_id}: (x, y) delay embedding, lag {xy_lag} rows',
                           out_root / f'persistence_xy_{network_id}.png')
            if ci:
                t_pop, d_pop = _topology([real_all[i] for i in range(len(starts))],
                                         [gen_all[i] for i in range(len(starts))],
                                         [lib[a:a + L] for a in lib_starts], cfg['PH_LANDMARKS'],
                                         scale_cloud=lib)
                topo['population'] = {'n_dims': len(real_nodes), **t_pop}
                _plot_diagrams(d_pop, f'{network_id}: population state ({len(real_nodes)} nodes)',
                               out_root / f'persistence_population_{network_id}.png')
            np.savez_compressed(out_root / f'diagrams_{network_id}.npz',
                                **{f'xy_{k}_H{h}': d_xy[k][h] for k in d_xy for h in (0, 1)})
        except Exception as exc:
            print(f'{network_id}: topology FAILED: {exc!r}', flush=True)
        topo['external_input_mode'] = mode
        with open(out_root / f'topology_{network_id}.json', 'w') as f:
            json.dump(topo, f, indent=1, default=float)
        _plot_rollouts(real_all[:, :, ti], gen_all[:, :, ti], targets, starts, dt,
                       f"{network_id}: {cfg.get('SPIKE_TRANSFORM')} -- free runs (mode={mode})",
                       out_root / f'rollout_examples_{network_id}.png')
        _log_progress(progress_log_path, f'{network_id}: autonomous evaluation done '
                      f'{time.perf_counter() - t0:.0f}s')
