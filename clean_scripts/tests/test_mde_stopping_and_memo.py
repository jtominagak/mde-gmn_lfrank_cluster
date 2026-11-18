'''proper_mde / proper_ccm: the 2026-11-18 stopping rule and redundancy fixes.

What this pins:
  1. proper_ccm(directions=...) computes only the direction asked for, and
     that direction's values are bit-identical to directions='both' for the
     same seed (the library draws are unchanged).
  2. When nothing passes crossMapRhoMin, proper_mde stops after ONE ranking
     pass, rather than re-ranking the unchanged candidates for every
     remaining dimension to reach the same empty result.
  3. A candidate's E-sweep and CCM verdict are computed at most once per
     proper_mde call (they do not depend on the columns already selected),
     and with a seed the selection is unchanged by that.
  4. minGain=0 never accepts a column that fails to improve the score, so
     MDErho is strictly increasing; minGain=None keeps Run.py's behaviour.
  5. effective_mde_dimension() is MDE_D capped at the pool size, with no
     per-delay multiplier.

No pytest in this project's venv -- run it directly:
    python3 tests/test_mde_stopping_and_memo.py
'''
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parents[1] / 'src'
sys.path.insert(0, str(SRC))
import edm                                       # noqa: E402
from mde_gmn.columns import effective_mde_dimension  # noqa: E402


def _toy_df(n=700, seed=0):
    '''Three coupled logistic maps (A, B drive T) plus pure-noise columns.'''
    rng = np.random.default_rng(seed)
    x = np.zeros((n, 3))
    x[0] = [0.4, 0.3, 0.2]
    for t in range(n - 1):
        a, b, c = x[t]
        x[t + 1, 0] = 3.8 * a * (1 - a)
        x[t + 1, 1] = 3.6 * b * (1 - b) - 0.05 * a
        x[t + 1, 2] = 3.7 * c * (1 - c) - 0.3 * a - 0.25 * b
    x = np.clip(x, 0, 1)
    df = pd.DataFrame({'A': x[:, 0], 'B': x[:, 1], 'T': x[:, 2]})
    for i in range(6):
        df[f'N{i}'] = rng.random(n)
    return df


KW = dict(lib=[1, 350], pred=[351, 700], Tp=1, E=0, maxE=4, libSizes=[20, 60, 150, 300],
          sample=4, ccmSeed=7, embedDimRhoMin=0.05, metric='correlation',
          ccmMetric='correlation')


def test_ccm_directions():
    df = _toy_df()
    kw = dict(libSizes=[20, 60, 150], sample=4, tau=-1, Tp=1, E=3, target='T', columns='A',
              target_measurement='correlation', columns_measurement='correlation', seed=3)
    both = edm.proper_ccm(df, **kw)
    c2t = edm.proper_ccm(df, directions='c2t', **kw)
    t2c = edm.proper_ccm(df, directions='t2c', **kw)
    assert np.array_equal(c2t[0], both[0]) and np.isnan(c2t[1]).all(), 'c2t-only is wrong'
    assert np.array_equal(t2c[1], both[1]) and np.isnan(t2c[0]).all(), 't2c-only is wrong'
    try:
        edm.proper_ccm(df, directions='sideways', **kw)
    except ValueError:
        pass
    else:
        raise AssertionError('an unknown direction must raise')
    print('PASS: proper_ccm computes only the requested direction, values unchanged')


def _counting(name):
    '''Wrap edm.<name> to count calls by the candidate column it was asked about.'''
    original = getattr(edm, name)
    counts = Counter()

    def wrapper(*args, **kwargs):
        counts[kwargs.get('columns')] += 1
        return original(*args, **kwargs)
    setattr(edm, name, wrapper)
    return counts, lambda: setattr(edm, name, original)


def test_no_rerank_after_crossmap_failure():
    df = _toy_df()
    cols = ['A', 'B', 'N0', 'N1']
    original = edm._simplex_metric
    calls = []
    edm._simplex_metric = lambda *a, **k: calls.append(1) or original(*a, **k)
    try:
        out = edm.proper_mde(df, target='T', columns=cols, D=5, crossMapRhoMin=0.999, **KW)
    finally:
        edm._simplex_metric = original
    assert out['MDEcolumns'] == []
    assert len(calls) == len(cols), (
        f'{len(calls)} ranking calls for {len(cols)} candidates -- an unchanged candidate set '
        f'must not be re-ranked once nothing passes crossMapRhoMin')
    print('PASS: a dimension where nothing passes crossMapRhoMin ends the search')


def test_gate_memo():
    df = _toy_df()
    cols = [c for c in df.columns if c != 'T']
    edim, restore_edim = _counting('proper_embed_dimension')
    ccm, restore_ccm = _counting('proper_ccm')
    try:
        first = edm.proper_mde(df, target='T', columns=cols, D=6, crossMapRhoMin=0.05,
                                ccmSlope=0.15, **KW)
    finally:
        restore_edim()
        restore_ccm()
    assert max(edim.values(), default=0) <= 1, f'E-sweep repeated for a candidate: {edim}'
    assert max(ccm.values(), default=0) <= 1, f'CCM repeated for a candidate: {ccm}'
    second = edm.proper_mde(df, target='T', columns=cols, D=6, crossMapRhoMin=0.05,
                             ccmSlope=0.15, **KW)
    assert first['MDEcolumns'] == second['MDEcolumns']
    assert np.array_equal(first['MDErho'], second['MDErho'])
    print(f'PASS: each candidate gated at most once per call '
          f'({sum(edim.values())} E-sweeps, {sum(ccm.values())} CCM runs)')


def test_min_gain():
    # Short-lag copies, as in a lagged run: near-duplicates of columns the
    # selection already has, which is where Run.py's rule over-selects.
    df = _toy_df()
    for col in ['A', 'B', 'N0']:
        for d in (-1, -2):
            df[f'{col}::delay{d}'] = df[col].shift(-d)
    df = df.iloc[2:].reset_index(drop=True)
    cols = [c for c in df.columns if c != 'T']
    kw = {**KW, 'lib': [1, 345], 'pred': [346, len(df)]}
    legacy = edm.proper_mde(df, target='T', columns=cols, D=8, crossMapRhoMin=0.05,
                             ccmSlope=0.01, **kw)
    stopped = edm.proper_mde(df, target='T', columns=cols, D=8, crossMapRhoMin=0.05,
                              ccmSlope=0.01, minGain=0.0, **kw)
    gains = np.diff(stopped['MDErho'])
    assert (gains > 0).all(), f'minGain=0 accepted a non-improving column: {stopped["MDErho"]}'
    assert stopped['MDEcolumns'] == legacy['MDEcolumns'][:len(stopped['MDEcolumns'])], (
        'the stopping rule may only END the selection early, never change its order')
    assert len(legacy['MDEcolumns']) > len(stopped['MDEcolumns']), (
        f'toy data should make legacy over-select: {legacy["MDErho"]}')
    print(f'PASS: minGain=0 stops at {len(stopped["MDEcolumns"])} columns '
          f'(scores {np.round(stopped["MDErho"], 3)}); legacy took {len(legacy["MDEcolumns"])} '
          f'(scores {np.round(legacy["MDErho"], 3)})')


def _stochastic_df(n=900, seed=3):
    """Two AR(1) drivers and noise columns; T depends on both drivers' last
    value. Nothing periodic, so a circular shift genuinely destroys the
    relationship -- unlike the logistic maps above, whose banded (near
    period-2) structure survives an even shift and makes a 'surrogate' of a
    real driver still informative. That is a real limitation of circular-shift
    surrogates on periodic signals; this toy is chosen not to have it."""
    rng = np.random.default_rng(seed)
    a, b = np.zeros(n), np.zeros(n)
    for t in range(1, n):
        a[t] = 0.8 * a[t - 1] + rng.normal()
        b[t] = 0.6 * b[t - 1] + rng.normal()
    target = np.zeros(n)
    target[1:] = np.tanh(a[:-1]) + 0.5 * b[:-1] ** 2 + 0.1 * rng.normal(size=n - 1)
    df = pd.DataFrame({'A': a, 'B': b, 'T': target})
    for i in range(10):
        df[f'N{i}'] = rng.normal(size=n)
    return df


def test_surrogate_threshold():
    '''gainSurrogates: the noise-calibrated stopping rule. It can only END the
    legacy selection earlier, never reorder it; it keeps both real drivers; it
    stops no later than minGain=0 (which it also applies); it is reproducible;
    and every accepted round beats the threshold it recorded.'''
    df = _stochastic_df()
    cols = [c for c in df.columns if c != 'T']
    kw = {**KW, 'lib': [1, 450], 'pred': [451, len(df)]}
    common = dict(target='T', columns=cols, D=8, crossMapRhoMin=0.05, ccmSlope=-1.0, **kw)
    legacy = edm.proper_mde(df, **common)
    plain = edm.proper_mde(df, minGain=0.0, **common)
    runs = [edm.proper_mde(df, minGain=0.0, gainSurrogates=19, **common) for _ in range(2)]
    sel = runs[0]['MDEcolumns']
    assert sel == runs[1]['MDEcolumns'], 'surrogate rule is not reproducible'
    assert sel == legacy['MDEcolumns'][:len(sel)], 'surrogate rule reordered the selection'
    assert len(sel) <= len(plain['MDEcolumns'])
    assert {'A', 'B'} <= set(sel), f'a real driver was rejected: {sel}'
    assert not [c for c in sel if c.startswith('N')], f'a noise column passed: {sel}'
    for d, thr in runs[0]['gainThresholds'].items():
        if d <= len(sel):
            assert runs[0]['MDErho'][d - 1] > thr, f'round {d} accepted a score below {thr}'
    print(f'PASS: surrogate rule keeps {sel} (minGain=0 kept {plain["MDEcolumns"]}, legacy '
          f'{legacy["MDEcolumns"]}); thresholds '
          f'{ {d: round(t, 3) for d, t in runs[0]["gainThresholds"].items()} }')


def test_stopping_rule_config():
    '''MDE_MIN_GAIN's three forms map to the right proper_mde arguments.'''
    import os
    import tempfile
    from mde_gmn import mde
    from mde_gmn.config import load_config
    os.environ['TSLS_MDE_GMN_RESULTS_DIR'] = tempfile.mkdtemp()
    assert mde.stopping_rule_kwargs(None) == {'minGain': None}
    assert mde.stopping_rule_kwargs(0.0) == {'minGain': 0.0}
    sur = mde.stopping_rule_kwargs('surrogate')
    assert sur['minGain'] == 0.0 and sur['gainSurrogates'] >= 2
    for raw, want in [('none', None), ('0', 0.0), ('0.02', 0.02), ('surrogate', 'surrogate')]:
        os.environ['TSLS_MDE_GMN_MDE_MIN_GAIN'] = raw
        assert load_config()['MDE_MIN_GAIN'] == want, raw
    os.environ.pop('TSLS_MDE_GMN_MDE_MIN_GAIN')
    for raw, tp, want in [('match', '20', 20), ('1', '20', 1), ('5', '1', 5)]:
        os.environ['TSLS_MDE_GMN_GMN_TP'], os.environ['TSLS_MDE_GMN_TP'] = raw, tp
        assert load_config()['GMN_TP'] == want, (raw, tp)
    os.environ.pop('TSLS_MDE_GMN_GMN_TP'); os.environ.pop('TSLS_MDE_GMN_TP')
    print('PASS: MDE_MIN_GAIN and GMN_TP parse as documented')


def test_effective_dimension():
    assert effective_mde_dimension(10, [f'c{i}' for i in range(980)]) == 10
    assert effective_mde_dimension(10, ['a', 'b', 'c']) == 3
    assert effective_mde_dimension(10, []) == 0
    print('PASS: effective_mde_dimension is MDE_D capped at the pool size')


if __name__ == '__main__':
    test_ccm_directions()
    test_no_rerank_after_crossmap_failure()
    test_gate_memo()
    test_min_gain()
    test_surrogate_threshold()
    test_stopping_rule_config()
    test_effective_dimension()
    print('ALL PASS')
