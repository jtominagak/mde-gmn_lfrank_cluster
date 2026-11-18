'''exp/spike-conversion-neuro: the spike -> continuous transforms and the
evaluation changes around them.

    python3 tests/test_spike_filters.py      (on a compute node)

Pins:
  * every method is strictly causal (changing future spikes never moves a past
    value), including Model B's data-fitted sigma
  * Models A and C match their closed-form impulse responses exactly
  * Models A and C are continuous: a bin's own spikes do not move that bin's
    output (the jump lands in g / dx/dt, one integration upstream)
  * the 'tsls' method is the parent branch's sawtooth, value for value
  * convert_units() trims every method at the same row, and select_units()
    prunes by library-side firing rate
  * neuron ranking windows are fixed (no event counting) when
    min_pred_events <= 0
  * config: full_trajectory evaluation selects correlation; 'events' with a
    non-sawtooth method is refused; EMBED_TAU reaches mde.embed_tau()
'''
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'src'))

import cleaned_helper_functions as tsls   # noqa: E402
from mde_gmn import spike_filters as sf    # noqa: E402

DT = 0.025
PARAMS = {'TAU_SYN': 0.05, 'TAU_CA': 0.5, 'TAU_W': 2.5, 'ADAPT_A': 1.0, 'ADAPT_B': 0.025,
          'NR_LAMBDA': 1.0, 'NR_N': 2.0, 'NR_SIGMA': 'median', 'OSC_F0': 8.0, 'OSC_ZETA': 0.7}


def _spikes(T=3000, N=6, seed=0):
    rng = np.random.default_rng(seed)
    rates = np.linspace(0.3, 8.0, N)
    return rng.poisson(rates[None, :] * DT, size=(T, N)).astype(float)


def _grid(T):
    return np.arange(T) * DT + DT / 2


def test_causality_every_method():
    S = _spikes()
    T = len(S)
    n0 = 1700
    S2 = S.copy()
    rng = np.random.default_rng(1)
    S2[n0 + 1:] = rng.poisson(5 * DT, size=S2[n0 + 1:].shape)
    for m in sf.METHODS:
        fit = slice(0, n0)     # sigma (Model B) fitted on the past only
        a, _ = sf.transform_spikes(m, S, _grid(T), PARAMS, fit)
        b, _ = sf.transform_spikes(m, S2, _grid(T), PARAMS, fit)
        assert np.array_equal(a[:n0 + 1], b[:n0 + 1], equal_nan=True), m
        assert not np.allclose(a[n0 + 50:], b[n0 + 50:], equal_nan=True), f'{m}: future spikes ignored?'


def test_calcium_impulse_response_exact():
    T = 400
    S = np.zeros((T, 1))
    S[10, 0] = 1
    c = sf.calcium(S, DT, PARAMS['TAU_SYN'], PARAMS['TAU_CA'])[:, 0]
    ts, tc = PARAMS['TAU_SYN'], PARAMS['TAU_CA']
    t = (np.arange(T) - 10) * DT
    want = np.where(t >= 0, ts * tc / (tc - ts) * (np.exp(-t / tc) - np.exp(-t / ts)), 0.0)
    assert np.allclose(c, want, atol=1e-12), np.abs(c - want).max()
    # linear in the spike count
    S[10, 0] = 3
    assert np.allclose(sf.calcium(S, DT, ts, tc)[:, 0], 3 * want, atol=1e-12)


def test_oscillator_impulse_response_exact():
    T = 200
    S = np.zeros((T, 1))
    S[5, 0] = 1
    x = sf.theta_oscillator(S, DT, PARAMS['OSC_F0'], PARAMS['OSC_ZETA'])[:, 0]
    w0 = 2 * np.pi * PARAMS['OSC_F0']
    z = PARAMS['OSC_ZETA']
    wd = w0 * np.sqrt(1 - z * z)
    t = (np.arange(T) - 5) * DT
    want = np.where(t >= 0, np.exp(-z * w0 * t) * np.sin(wd * t) / wd, 0.0)
    assert np.allclose(x, want, atol=1e-12), np.abs(x - want).max()


def test_outputs_are_continuous_at_spikes():
    '''For A and C, the output at bin n must not depend on bin n's spikes: the
    impulse moves g (or dx/dt), and the output is one integration further.'''
    S = _spikes(T=800)
    for n in (100, 400, 700):
        S2 = S.copy()
        S2[n] += 4
        for f, args in ((sf.calcium, (PARAMS['TAU_SYN'], PARAMS['TAU_CA'])),
                        (sf.theta_oscillator, (PARAMS['OSC_F0'], PARAMS['OSC_ZETA']))):
            a, b = f(S, DT, *args), f(S2, DT, *args)
            assert np.array_equal(a[:n + 1], b[:n + 1]), (f.__name__, n)
    # and the sawtooth is not continuous: it drops to 0 at a spike
    grid = _grid(800)
    saw, _ = sf.transform_spikes('tsls', S, grid, PARAMS, slice(0, 400))
    spike_rows = np.nonzero(S[:, -1] > 0)[0]
    spike_rows = spike_rows[spike_rows > spike_rows[0]]
    assert np.all(saw[spike_rows, -1] == 0)
    assert (saw[spike_rows - 1, -1] >= 0.2).any()     # a drop of >= 8 bins' worth


def test_tsls_method_is_parent_sawtooth():
    S = _spikes()
    grid = _grid(len(S))
    got, _ = sf.transform_spikes('tsls', S, grid, PARAMS, slice(0, 10))
    for j in range(S.shape[1]):
        want = tsls.compute_time_since_previous_spike(grid[S[:, j] > 0], grid)
        assert np.array_equal(got[:, j], want, equal_nan=True)


def test_adapt_nr_range_and_sigma_from_fit_rows_only():
    S = _spikes(T=4000)
    grid = _grid(len(S))
    fit = slice(0, 2000)
    r, info = sf.transform_spikes('adapt_nr', S, grid, PARAMS, fit)
    assert np.all((r >= 0) & (r < 1))
    S2 = S.copy()
    S2[2500:] *= 3
    _, info2 = sf.transform_spikes('adapt_nr', S2, grid, PARAMS, fit)
    assert np.array_equal(info['sigma'], info2['sigma'])
    # p75 sigma >= median sigma, per unit
    _, info3 = sf.transform_spikes('adapt_nr', S, grid, {**PARAMS, 'NR_SIGMA': 'p75'}, fit)
    assert np.all(info3['sigma'] >= info['sigma'])
    # w keeps the drive away from zero during a long silence: after a burst,
    # 2 s of silence leave y = c + lambda*w well above c alone
    S3 = np.zeros((400, 1))
    S3[10:30, 0] = 1
    st = sf.adaptation_states(S3, DT, 0.05, 0.5, 2.5, 1.0, 0.025)
    n = 30 + 80
    assert st[n, 0, 2] > 0.3 * st[n, 0, 1]


def _raw_frame(T=3000, N=6):
    S = _spikes(T, N)
    S[:200, 0] = 0      # unit 0 fires late -> sets the trim
    S[200, 0] = 1
    cols = [f'CA1_{j}' for j in range(N)]
    df = pd.DataFrame(S, columns=cols)
    df.insert(0, 'time_bin_center', _grid(T))
    df['head_position_x'] = np.sin(np.arange(T) / 50.0)
    df['head_position_y'] = np.cos(np.arange(T) / 70.0)
    return df, cols


def test_convert_units_common_trim_and_pruning():
    from mde_gmn.tsls_data import convert_units, select_units
    df_raw, cols = _raw_frame()
    df_raw['CA1_99'] = 0.0
    df_raw.loc[5, 'CA1_99'] = 1.0       # one spike in 75 s: ~0.013 Hz
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        kept = select_units(df_raw, cols + ['CA1_99'], 0.1, 0.7, d)
        assert kept == cols, kept
        table = pd.read_csv(d / 'retained_units.csv')
        assert list(table.loc[~table.kept, 'unit']) == ['CA1_99']
        assert select_units(df_raw, cols + ['CA1_99'], 0.0, 0.7, d) == cols + ['CA1_99']
        shapes = {}
        for m in sf.METHODS:
            cfg = {'SPIKE_TRANSFORM': m, 'SPIKE_PARAMS': PARAMS, 'LIB_RATIO': 0.7}
            df, obs = convert_units(df_raw, cols, ['head_position_x', 'head_position_y'], cfg, d)
            shapes[m] = df.shape
            assert obs == [f'{c}_tsls' for c in cols]
            assert not df[obs].isna().any().any()
            assert df['time_bin_center'].iloc[0] == df_raw['time_bin_center'].iloc[200], m
        assert len(set(shapes.values())) == 1, shapes


def test_fixed_neuron_window_when_no_event_counting():
    from mde_gmn.tsls_data import _resolve_target_ranking_window
    usable = pd.DataFrame({'n': np.linspace(0, 1, 9000)})
    w, ev = _resolve_target_ranking_window('n', usable, usable, 0.8, False, 1000, 0, 4000)
    assert (w, ev) == (4000, None)
    w, ev = _resolve_target_ranking_window('n', usable, usable.iloc[:3000], 0.8, False, 1000, 0, 4000)
    assert (w, ev) == (3000, None)


def test_config_evaluation_and_tau():
    from mde_gmn import mde
    from mde_gmn.config import load_config
    saved = dict(os.environ)
    try:
        with tempfile.TemporaryDirectory() as d:
            for k in list(os.environ):
                if k.startswith('TSLS_MDE_GMN_'):
                    del os.environ[k]
            os.environ['TSLS_MDE_GMN_RESULTS_DIR'] = d
            os.environ['TSLS_MDE_GMN_RUN_CACHE_DIR'] = d
            cfg = load_config()
            assert cfg['METRIC'] == 'correlation' and cfg['MIN_PRED_EVENTS'] == 0
            assert cfg['MAX_WINDOW_ROWS'] == cfg['NEURON_WINDOW_ROWS']
            assert cfg['SPIKE_TRANSFORM'] == 'tsls' and mde.embed_tau() == -1
            os.environ['TSLS_MDE_GMN_EMBED_TAU'] = '6'
            assert load_config()['EMBED_TAU'] == 6 and mde.embed_tau() == -6
            os.environ['TSLS_MDE_GMN_NEURON_EVAL'] = 'events'
            os.environ['TSLS_MDE_GMN_SPIKE_TRANSFORM'] = 'tsls'
            cfg = load_config()
            assert cfg['METRIC'] == 'f1' and cfg['MIN_PRED_EVENTS'] == 20
            os.environ['TSLS_MDE_GMN_SPIKE_TRANSFORM'] = 'calcium'
            try:
                load_config()
            except ValueError:
                pass
            else:
                raise AssertionError("NEURON_EVAL='events' with calcium must be refused")
            os.environ['TSLS_MDE_GMN_NEURON_EVAL'] = 'full_trajectory'
            os.environ['TSLS_MDE_GMN_TAU_SYN_MS'] = '75'
            p = load_config()['SPIKE_PARAMS']
            assert p['TAU_SYN'] == 0.075 and np.isclose(p['ADAPT_B'], 0.075 * 0.5)
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_rollout_scoring_on_a_perfect_and_a_frozen_run():
    from mde_gmn.rollout_eval import Reference, rollout_starts, score_rollout
    T = 3000
    t = np.arange(T)
    df = pd.DataFrame({'time_bin_center': t * DT,
                       'head_position_x': np.sin(t / 40.0), 'head_position_y': np.cos(t / 40.0),
                       'CA1_0_tsls': np.sin(t / 13.0), 'CA1_1_tsls': np.cos(t / 17.0)})
    ref = Reference(df, ['head_position_x', 'head_position_y'], 2000, 2500, DT)
    s = 2550
    perfect = df.iloc[s:s + 400].drop(columns=['time_bin_center']).reset_index(drop=True)
    rows = score_rollout(perfect, s, ref, [80, 400])
    r = rows[-1]
    assert np.isclose(r['corr_head_position_x'], 1) and r['path_err_mean'] == 0
    assert np.isclose(r['manifold_ratio'], 1) and np.isclose(r['motion_ratio'], 1)
    assert not r['blew_up']
    frozen = pd.DataFrame(np.tile(perfect.iloc[0].to_numpy(float), (len(perfect), 1)),
                          columns=perfect.columns)
    r = score_rollout(frozen, s, ref, [400])[0]
    assert r['motion_ratio'] < 1e-9 and r['path_err_mean'] > 0
    assert np.isnan(r['corr_head_position_x']) and np.isnan(r['unit_corr_mean'])
    assert rollout_starts(2500, 3000, 400, 8) == sorted(set(np.linspace(2500, 2600, 8).round().astype(int)))
    assert rollout_starts(2500, 2800, 400, 8) == []


if __name__ == '__main__':
    tests = [v for k, v in sorted(globals().items()) if k.startswith('test_')]
    for fn in tests:
        fn()
        print(f'ok  {fn.__name__}')
    print(f'{len(tests)} passed')
