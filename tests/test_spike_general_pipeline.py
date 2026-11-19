'''exp/spike-conversion-general: the pipeline-level changes around the new
operators.

  * transform_events(): 'sawtooth' is exactly the parent's TSLS converter;
    'gamma' is exactly gamma_filter (+ the multi-scale sum); 'auto:<k>'
    timescales resolve from the training rows only.
  * convert_events(): every method gets the SAME rows (common trim), so the
    A/B/end split is identical across methods; no NaN survives the trim.
  * select_observable(): the N_MIN_EVENTS filter counts training rows only.
  * run_mde_for_target(): a neuron node is sized by reset events only under
    an event metric (f1/gamma); under 'correlation' it gets the fixed
    contiguous window a behavioural target gets.
  * mde.embed_tau(): -1 (the parent value) when nothing is exported, the
    resolved lag otherwise; resolve_embed_tau() sets the Stage 2 lag step
    under DELAY_SOURCE='tau_count'.
  * preprocessing_digest(): differs between methods, so no two methods share
    a cache entry.
  * autonomous_eval helpers: start spacing, weighted correlation, graph
    summary of a small nested-MDE-shaped graph.

Run: python3 tests/test_spike_general_pipeline.py
'''
import os
import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'src'))
import cleaned_helper_functions as tsls  # noqa: E402
from mde_gmn import mde  # noqa: E402
from mde_gmn import universal_point_process_embeddings as U  # noqa: E402
from mde_gmn.tsls_data import convert_events, select_observable, transform_events  # noqa: E402

DT = 0.025
BASE = dict(GAMMA_ORDER=3, GAMMA_TAUS_MS=[250.0], GAMMA_LOG_KAPPA=0.0, LMU_THETA_MS=1000.0, LMU_D=3,
            LMU_DISCRETIZATION='impulse', LMU_PRESMOOTH_ORDER=0, LMU_REDUCTION='lmu_norm',
            LOGSIG_W_MS=1000.0, LOGSIG_DEPTH=2, LOGSIG_PRESMOOTH_ORDER=0, LOGSIG_REDUCTION='rms_norm',
            WARMUP_ROWS=50, LIB_RATIO=0.7)


def _raw(T=3000, C=4, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(T) * DT + 0.0125
    counts = (rng.random((T, C)) < 0.04).astype(int)
    counts[:120, 2] = 0                      # a late first event: extends the common trim
    df = pd.DataFrame(counts, columns=[f'CA1_{i}' for i in range(C)])
    df.insert(0, 'time_bin_center', t)
    df['head_position_x'] = np.sin(t)
    df['head_position_y'] = np.cos(t)
    return df


def test_transform_sawtooth_and_gamma_exact():
    df = _raw()
    cols = [c for c in df.columns if c.startswith('CA1_')]
    counts = df[cols].to_numpy(float)
    tg = df['time_bin_center'].to_numpy()
    X, _ = transform_events(counts, tg, {**BASE, 'SPIKE_TRANSFORM': 'sawtooth'}, 2000)
    for j in range(len(cols)):
        ref = tsls.compute_time_since_previous_spike(tg[counts[:, j] > 0], tg)
        assert np.array_equal(np.isnan(X[:, j]), np.isnan(ref))
        assert np.allclose(X[:, j][~np.isnan(ref)], ref[~np.isnan(ref)])
    dt = float(np.median(np.diff(tg)))
    X, info = transform_events(counts, tg, {**BASE, 'SPIKE_TRANSFORM': 'gamma',
                                            'GAMMA_TAUS_MS': [100.0, 500.0]}, 2000)
    assert np.allclose(X, U.gamma_filter(counts, dt, 0.1, 3) + U.gamma_filter(counts, dt, 0.5, 3))
    # auto timescale reads the training rows only: changing later rows does not move it
    c2 = counts.copy()
    c2[2500:] = 1
    _, i1 = transform_events(counts, tg, {**BASE, 'SPIKE_TRANSFORM': 'gamma', 'GAMMA_TAUS_MS': ['auto:0.5']}, 2000)
    _, i2 = transform_events(c2, tg, {**BASE, 'SPIKE_TRANSFORM': 'gamma', 'GAMMA_TAUS_MS': ['auto:0.5']}, 2000)
    assert i1['taus_s'] == i2['taus_s']


def test_common_trim_identical_across_methods():
    df = _raw()
    cols = [c for c in df.columns if c.startswith('CA1_')]
    out = {}
    with tempfile.TemporaryDirectory() as d:
        for m in ('sawtooth', 'gamma', 'lmu', 'logsig'):
            conv, obs, info = convert_events(df, cols, ['head_position_x', 'head_position_y'],
                                             {**BASE, 'SPIKE_TRANSFORM': m}, Path(d))
            assert not conv[obs].isna().any().any(), m
            out[m] = conv
    t0 = out['sawtooth']['time_bin_center'].to_numpy()
    first = int(np.argmax(df['CA1_2'].to_numpy() > 0))
    assert len(df) - len(t0) == max(50, first + 1)
    for m, conv in out.items():
        assert np.array_equal(conv['time_bin_center'].to_numpy(), t0), m
        assert np.array_equal(conv['head_position_x'].to_numpy(),
                              out['sawtooth']['head_position_x'].to_numpy()), m


def test_observability_filter_training_rows_only():
    df = _raw(C=3)
    df['CA1_1'] = 0
    df.loc[2500:, 'CA1_1'] = 1                # busy, but only after the training split
    df['CA1_2'] = 0
    df.loc[:2000:40, 'CA1_2'] = 1             # 51 events, all in training rows
    with tempfile.TemporaryDirectory() as d:
        kept = select_observable(df, ['CA1_0', 'CA1_1', 'CA1_2'], 0.7, 50, Path(d))
        assert kept == ['CA1_0', 'CA1_2'], kept
        assert (Path(d) / 'observability_filter.csv').exists()


def test_event_window_only_for_event_metrics():
    seen = []

    def fake_resolve(target, full_df, usable_df, lib_ratio, continuous, *rest):
        seen.append(continuous)
        return None, 0                         # makes run_mde_for_target stop right here

    real = mde._resolve_target_ranking_window
    mde._resolve_target_ranking_window = fake_resolve
    try:
        df = pd.DataFrame({'n': np.arange(500.0), 'c': np.arange(500.0)})
        with tempfile.TemporaryDirectory() as d:
            for tm in ('f1', 'gamma', 'correlation', 'mae'):
                mde.run_mde_for_target('n', df, ['c'], 300, 100, 1, 0.2, 0.2, 0, Path(d), [],
                                       target_metric=tm, is_behavioral_target=False)
            mde.run_mde_for_target('n', df, ['c'], 300, 100, 1, 0.2, 0.2, 0, Path(d), [],
                                   target_metric='f1', is_behavioral_target=True)
    finally:
        mde._resolve_target_ranking_window = real
    assert seen == [False, False, True, True, True], seen


def test_embed_tau_export_and_lag_step():
    os.environ.pop('TSLS_MDE_GMN_EMBED_TAU_RESOLVED', None)
    assert mde.embed_tau() == -1
    from mde_gmn.pipeline import preprocessing_digest, resolve_embed_tau
    rng = np.random.default_rng(3)
    T = 6000
    phi = np.exp(-1 / 12.0)
    X = np.zeros((T, 3))
    for i in range(1, T):
        X[i] = phi * X[i - 1] + rng.normal(size=3)
    df = pd.DataFrame(X, columns=['a', 'b', 'c'])
    with tempfile.TemporaryDirectory() as d:
        cfg = {'EMBED_TAU': 'auto', 'EMBED_TAU_MAX': 40, 'EMBED_TAU_METHOD': 'acf',
               'PREDICTION_START': 4000, 'RESULTS_DIR': Path(d), 'DELAY_SOURCE': 'tau_count',
               'LAG_STEP': 1}
        lag = resolve_embed_tau(cfg, df, ['a', 'b', 'c'])
        assert 10 <= lag <= 14, lag
        assert cfg['LAG_STEP'] == lag and mde.embed_tau() == -lag
        cfg2 = dict(cfg, EMBED_TAU=7, DELAY_SOURCE='explicit', LAG_STEP=1)
        assert resolve_embed_tau(cfg2, df, ['a']) == 7 and cfg2['LAG_STEP'] == 1
    os.environ.pop('TSLS_MDE_GMN_EMBED_TAU_RESOLVED', None)
    a = preprocessing_digest({'SPIKE_TRANSFORM': 'gamma', 'GAMMA_TAUS_MS': [250.0]})
    b = preprocessing_digest({'SPIKE_TRANSFORM': 'gamma', 'GAMMA_TAUS_MS': [500.0]})
    c = preprocessing_digest({'SPIKE_TRANSFORM': 'sawtooth', 'GAMMA_TAUS_MS': [250.0]})
    assert len({a, b, c}) == 3


def test_autonomous_eval_helpers():
    from mde_gmn.autonomous_eval import _corr, graph_summary, rollout_starts
    s = rollout_starts(1000, 3000, 400, 5)
    assert s[0] == 1000 and s[-1] == 2600 and len(s) == 5
    assert rollout_starts(1000, 1200, 400, 5) == []
    a = np.array([0, 1, 2, 3, 10.0])
    b = np.array([0, 1, 2, 3, -10.0])
    assert _corr(a, b) < 0 < _corr(a, b, w=np.array([1, 1, 1, 1, 0.0]))
    g = nx.MultiDiGraph()
    g.add_edge('CA1_1_gam', 'head_position_x', delay=0)
    g.add_edge('OFC_2_gam', 'CA1_1_gam', delay=0)
    g.add_node('OFC_2_gam::delay-40', lagged_of='OFC_2_gam', lag=-40)
    g.add_edge('OFC_2_gam::delay-40', 'head_position_y', delay=-40)
    gs = graph_summary(g, ['head_position_x', 'head_position_y'])
    assert gs['n_real_nodes'] == 4 and gs['n_lagged_leaves'] == 1 and gs['n_edges'] == 3
    assert gs['region_pairs'] == {'CA1->target': 1, 'OFC->CA1': 1, 'OFC->target': 1}
    assert gs['target_drivers']['head_position_y'] == ['OFC_2_gam::delay-40']


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_') and callable(fn):
            fn()
            print(f'PASS {name}')
    print('ALL PASSED')
