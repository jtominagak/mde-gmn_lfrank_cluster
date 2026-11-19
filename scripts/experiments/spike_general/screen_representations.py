"""Stage I of exp/spike-conversion-general: screen every event->continuous
setting on the representation alone, before any MDE/GMN run.

For each setting in GRID (all of Methods 1-3 across their timescale grids,
plus the sawtooth baseline), on the same channels and rows the pipeline uses
(first CAP channels per region, observability filter, common trim, the same
A/B split), it measures:

  smoothness        roughness = sd(2nd diff)/sd(1st diff); max_step =
                    q99.9 |1st diff| / sd (O(1) for jumps, ~1/tau_c smooth)
  collapse          fraction of rows on the channel's floor (within 1e-3 of
                    its range); fraction of library population states whose
                    nearest non-adjacent neighbour is < 1% of the median
                    nearest-neighbour distance (degenerate lookups)
  Takens lag        the pipeline's EMBED_TAU=auto rule (AMI first minimum,
                    ACF 1/e fallback, cap 40 rows)
  decodability      (a) per channel, simplex cross-map of each target one
                    step ahead from the channel's own delay embedding
                    (E=3, its Takens lag): best and top-5 mean r;
                    (b) greedy forward selection of up to 6 channels'
                    current values for a joint simplex readout of (x, y) --
                    a cheap stand-in for MDE's depth-0 selection.
                    Both are scored on rows [A, B), library [0, A).
  autonomy proxy    per channel, self-simplex r at 1 s ahead minus the
                    persistence r at 1 s: how much of the channel's own
                    future its own history predicts beyond "stays the same".
  vector content    for Methods 2-3, the cross-map r of the full per-channel
                    vector (current coordinates, no delays) next to the
                    scalar reduction's, to see what the reduction loses.

Usage (compute node only):
  python screen_representations.py <out_dir> [--workers N] [--only name,name]
Writes <out_dir>/screen.csv, per-setting example traces, and summary figures.
"""

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2] / 'src'))
from mde_gmn import universal_point_process_embeddings as U  # noqa: E402
from mde_gmn.tsls_data import raw_neuron_columns, transform_events  # noqa: E402

DATA = Path('/bucket/FukaiU/tominaga/pao_rotation/data/J1620210605_epoch_8_freq_40_Hz.csv')
TARGETS = ['head_position_x', 'head_position_y']
CAP, N_MIN, WARMUP, LIB_RATIO, VAL_RATIO = 30, 100, 400, 0.7, 0.5
TAU_MAX = 40
AHEAD = 40          # 1 s, for the autonomy proxy

BASE = dict(GAMMA_ORDER=3, GAMMA_TAUS_MS=[250.0], GAMMA_LOG_KAPPA=0.0,
            LMU_THETA_MS=1000.0, LMU_D=3, LMU_DISCRETIZATION='impulse', LMU_PRESMOOTH_ORDER=0,
            LMU_REDUCTION='lmu_norm', LOGSIG_W_MS=1000.0, LOGSIG_DEPTH=2,
            LOGSIG_PRESMOOTH_ORDER=0, LOGSIG_REDUCTION='rms_norm')


def grid():
    g = [('sawtooth', dict(SPIKE_TRANSFORM='sawtooth'))]
    for m in (2, 3):
        for tau in (100, 250, 500):
            for kap in (0, 100):
                g.append((f'gamma_m{m}_t{tau}' + (f'_log{kap}' if kap else ''),
                          dict(SPIKE_TRANSFORM='gamma', GAMMA_ORDER=m, GAMMA_TAUS_MS=[float(tau)],
                               GAMMA_LOG_KAPPA=float(kap))))
    for kap in (0, 100):
        g.append(('gamma_m3_t100+500' + (f'_log{kap}' if kap else ''),
                  dict(SPIKE_TRANSFORM='gamma', GAMMA_ORDER=3, GAMMA_TAUS_MS=[100.0, 500.0],
                       GAMMA_LOG_KAPPA=float(kap))))
    g.append(('gamma_m3_auto0.25', dict(SPIKE_TRANSFORM='gamma', GAMMA_ORDER=3,
                                         GAMMA_TAUS_MS=['auto:0.25'])))
    for th in (500, 1000, 2000):
        for D in (2, 3, 4):
            for pre in (0, 2):
                g.append((f'lmu_th{th}_D{D}' + ('_pre2' if pre else ''),
                          dict(SPIKE_TRANSFORM='lmu', LMU_THETA_MS=float(th), LMU_D=D,
                               LMU_PRESMOOTH_ORDER=pre)))
    for W in (500, 1000, 2000):
        for K in (2, 3):
            for pre in (0, 3):
                g.append((f'logsig_W{W}_K{K}' + ('_pre3' if pre else ''),
                          dict(SPIKE_TRANSFORM='logsig', LOGSIG_W_MS=float(W), LOGSIG_DEPTH=K,
                               LOGSIG_PRESMOOTH_ORDER=pre)))
    return g


def load():
    df = pd.read_csv(DATA)
    cols = raw_neuron_columns(df, CAP)
    counts = df[cols].to_numpy(dtype=float)
    lib_end_raw = int(round(len(df) * LIB_RATIO))
    keep, _ = U.observable_channels(counts, lib_end_raw, N_MIN)
    cols = [cols[i] for i in keep]
    counts = counts[:, keep]
    first = np.array([np.argmax(counts[:, j] > 0) for j in range(counts.shape[1])])
    trim = int(max(WARMUP, first.max() + 1))
    return df, cols, counts, lib_end_raw, trim


def _embed(x, E, lag):
    T = len(x)
    out = np.full((T, E), np.nan)
    for k in range(E):
        out[k * lag:, k] = x[:T - k * lag] if k else x
    return out


def _simplex(Xlib, ylib, Xpred, k=None):
    ok = np.all(np.isfinite(Xlib), 1) & np.isfinite(ylib)
    Xlib, ylib = Xlib[ok], ylib[ok]
    k = k or Xlib.shape[1] + 1
    tree = cKDTree(Xlib)
    good = np.all(np.isfinite(Xpred), 1)
    pred = np.full(len(Xpred), np.nan)
    d, i = tree.query(Xpred[good], k=k)
    d = np.atleast_2d(d)
    i = np.atleast_2d(i)
    d0 = np.maximum(d[:, :1], 1e-12)
    w = np.exp(-d / d0)
    pred[good] = (w * ylib[i]).sum(1) / w.sum(1)
    return pred


def _r(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10 or a[ok].std() == 0 or b[ok].std() == 0:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _degenerate_fraction(Z, theiler=20, n=15000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(Z), size=min(n, len(Z)), replace=False))
    P = Z[idx]
    tree = cKDTree(P)
    d, j = tree.query(P, k=min(40, len(P)))
    nn = np.full(len(P), np.nan)
    for r in range(len(P)):
        far = np.abs(idx[j[r]] - idx[r]) > theiler
        if far.any():
            nn[r] = d[r][far][0]
    med = np.nanmedian(nn)
    return float(np.nanmean(nn < 0.01 * med)) if med > 0 else 1.0


def evaluate(name, spec, df, cols, counts, lib_end_raw, trim, out_dir):
    t0 = time.time()
    s = dict(BASE)
    s.update(spec)
    tg = df['time_bin_center'].to_numpy()
    X, info = transform_events(counts, tg, s, lib_end_raw)
    X = X[trim:]
    Y = df[TARGETS].to_numpy()[trim:]
    n = len(X)
    A = int(round(n * LIB_RATIO))
    B = A + int(round((n - A) * VAL_RATIO))
    rec = {'name': name, 'method': s['SPIKE_TRANSFORM'], **{k: (json.dumps(v) if isinstance(v, list) else v)
                                                              for k, v in spec.items() if k != 'SPIKE_TRANSFORM'}}
    rec.update({k: v for k, v in info.items() if k in ('taus_s', 'theta_s', 'W_s')})
    rec['roughness'] = float(np.nanmedian(U.roughness(X)))
    rec['max_step'] = float(np.nanmedian(U.max_step_ratio(X)))
    rec['collapse_frac'] = float(np.nanmedian(U.collapse_fraction(X)))
    mu, sd = X[:A].mean(0), X[:A].std(0)
    sd = np.where(sd > 0, sd, 1)
    Z = (X - mu) / sd
    rec['degenerate_nn_frac'] = _degenerate_fraction(Z[:A])
    lag, lags, rules = U.dataset_embedding_lag(X, TAU_MAX, 'ami', lib_end=A)
    rec['embed_tau_rows'] = lag
    rec['embed_tau_rules'] = json.dumps(rules)

    # (a) per-channel cross-map of each target, one step ahead
    xm = np.full((X.shape[1], 2), np.nan)
    for j in range(X.shape[1]):
        Ej = _embed(Z[:, j], 3, lag)
        for t in range(2):
            y_next = np.r_[Y[1:, t], np.nan]
            p = _simplex(Ej[:A - 1], y_next[:A - 1], Ej[A:B - 1])
            xm[j, t] = _r(y_next[A:B - 1], p)
    xm_mean = np.nanmean(xm, 1)
    srt = np.sort(xm_mean[np.isfinite(xm_mean)])[::-1]
    rec['xmap_best'] = float(srt[0]) if len(srt) else np.nan
    rec['xmap_top5'] = float(srt[:5].mean()) if len(srt) else np.nan
    rec['xmap_median'] = float(np.median(srt)) if len(srt) else np.nan

    # (b) greedy joint readout of (x, y) from channels' current values
    chosen, best_curve = [], []
    ynext = np.vstack([Y[1:], [np.nan, np.nan]])
    for _ in range(6):
        best = (-np.inf, None)
        for j in range(X.shape[1]):
            if j in chosen:
                continue
            F = Z[:, chosen + [j]]
            rs = [_r(ynext[A:B - 1, t], _simplex(F[:A - 1], ynext[:A - 1, t], F[A:B - 1]))
                  for t in range(2)]
            v = np.nanmean(rs)
            if v > best[0]:
                best = (v, j)
        chosen.append(best[1])
        best_curve.append(best[0])
    rec['joint6_r'] = float(np.nanmax(best_curve))
    rec['joint_curve'] = json.dumps([round(v, 4) for v in best_curve])
    rec['joint_channels'] = json.dumps([cols[j] for j in chosen])

    # autonomy proxy: self-prediction 1 s ahead vs persistence
    gain, selfr = [], []
    for j in range(X.shape[1]):
        Ej = _embed(Z[:, j], 3, lag)
        fut = np.r_[Z[AHEAD:, j], np.full(AHEAD, np.nan)]
        p = _simplex(Ej[:A - AHEAD], fut[:A - AHEAD], Ej[A:B - AHEAD])
        r_s = _r(fut[A:B - AHEAD], p)
        r_p = _r(fut[A:B - AHEAD], Z[A:B - AHEAD, j])
        selfr.append(r_s)
        gain.append(r_s - r_p)
    rec['self_r_1s'] = float(np.nanmedian(selfr))
    rec['self_gain_vs_persist_1s'] = float(np.nanmedian(gain))

    # vector content (Methods 2-3): current coordinates only
    if s['SPIKE_TRANSFORM'] in ('lmu', 'logsig'):
        dt = info['dt']
        if s['SPIKE_TRANSFORM'] == 'lmu':
            st = U.legendre_memory(counts, dt, info['theta_s'], s['LMU_D'],
                                   presmooth_order=s['LMU_PRESMOOTH_ORDER'])[trim:]
        else:
            st = U.fading_log_signature(counts, dt, info['W_s'], s['LOGSIG_DEPTH'],
                                        presmooth_order=s['LOGSIG_PRESMOOTH_ORDER'])[trim:]
        vm = []
        for j in range(st.shape[1]):
            V = st[:, j, :]
            V = (V - V[:A].mean(0)) / np.where(V[:A].std(0) > 0, V[:A].std(0), 1)
            rs = []
            for t in range(2):
                y_next = np.r_[Y[1:, t], np.nan]
                rs.append(_r(y_next[A:B - 1], _simplex(V[:A - 1], y_next[:A - 1], V[A:B - 1])))
            vm.append(np.nanmean(rs))
        # the scalar's cross-map at the SAME (no-delay) footing, for a like-for-like loss
        sc = []
        for j in range(X.shape[1]):
            rs = []
            for t in range(2):
                y_next = np.r_[Y[1:, t], np.nan]
                rs.append(_r(y_next[A:B - 1], _simplex(Z[:A - 1, j:j + 1], y_next[:A - 1],
                                                       Z[A:B - 1, j:j + 1])))
            sc.append(np.nanmean(rs))
        vm, sc = np.sort(np.array(vm))[::-1], np.sort(np.array(sc))[::-1]
        rec['vector_xmap_top5'] = float(np.nanmean(vm[:5]))
        rec['scalar_nodelay_xmap_top5'] = float(np.nanmean(sc[:5]))

    np.save(out_dir / 'traces' / f'{name}.npy', X[A:A + 1600, :8].astype(np.float32))
    rec['seconds'] = round(time.time() - t0, 1)
    print(f"{name}: tau={lag} xmap_top5={rec['xmap_top5']:.3f} joint6={rec['joint6_r']:.3f} "
          f"rough={rec['roughness']:.3f} step={rec['max_step']:.3f} collapse={rec['collapse_frac']:.3f} "
          f"selfgain={rec['self_gain_vs_persist_1s']:.3f} ({rec['seconds']}s)", flush=True)
    return rec


_G = {}


def _run(item):
    name, spec = item
    df, cols, counts, lib_end_raw, trim = _G['data']
    try:
        return evaluate(name, spec, df, cols, counts, lib_end_raw, trim, _G['out'])
    except Exception as exc:  # keep the sweep going; the row records why
        print(f'{name}: FAILED {exc!r}', flush=True)
        return {'name': name, 'error': repr(exc)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out_dir')
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--only', default='')
    a = ap.parse_args()
    out = Path(a.out_dir)
    (out / 'traces').mkdir(parents=True, exist_ok=True)
    items = grid()
    if a.only:
        keep = set(a.only.split(','))
        items = [it for it in items if it[0] in keep]
    df, cols, counts, lib_end_raw, trim = load()
    json.dump({'channels': cols, 'trim': trim, 'n_rows': len(df) - trim,
               'raw_spikes_first8': cols[:8]}, open(out / 'meta.json', 'w'), indent=1)
    np.save(out / 'traces' / '_raw_counts.npy',
            counts[trim:][int(round((len(df) - trim) * LIB_RATIO)):][:1600, :8].astype(np.float32))
    _G['out'] = out
    _G['data'] = (df, cols, counts, lib_end_raw, trim)   # inherited by the forked workers
    ctx = mp.get_context('fork')
    with ctx.Pool(a.workers) as pool:
        rows = pool.map(_run, items, chunksize=1)
    pd.DataFrame(rows).to_csv(out / 'screen.csv', index=False)
    print(f'wrote {out / "screen.csv"} ({len(rows)} settings)')


if __name__ == '__main__':
    main()
