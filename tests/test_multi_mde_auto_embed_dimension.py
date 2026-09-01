'''Acceptance test for multi_proper_mde's new per-candidate auto-select
embedding dimension (multi_proper_embed_dimension()), closing the "no
per-candidate EmbedDimension auto-search here, unlike single-target
proper_mde with E=0" gap multi_proper_mde's own docstring used to
describe as a hard limitation.

Two things checked, on real (chaotic, genuinely E-dependent) synthetic
data rather than assumed from reading the code alone:

1. multi_proper_embed_dimension() itself: its own reported per-E scores
   and argmax-E match an independent, direct recomputation via
   _simplex_multi_metric() (the same per-target-then-aggregate primitive
   multi_proper_mde's own ranking stage already relies on) -- not just
   "the function runs without crashing."
2. multi_proper_mde(): E>0 (static) is completely unaffected -- every
   accepted candidate's own EDim entry is exactly E, matching pre-existing
   behavior bit for bit. E<=0 (new) genuinely invokes the auto-select path
   -- EDim gets populated with a real, in-range E, and it matches the same
   independent brute-force computation as (1), not just "some E".

No pytest in this project's venv -- run directly:
    python3 tests/test_multi_mde_auto_embed_dimension.py
'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'src'))
import cleaned_helper_functions as tsls  # noqa: E402

MAX_E = 8


def _logistic_map_series(n, r=3.95, x0=0.4):
    '''Same construction as tests/test_r20_predict_nonlinear.py's own
    helper -- deterministic chaos with genuine curvature, so an E-sweep
    produces an actually-peaked score curve, not flat noise.'''
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x


def _synthetic_df(n=600):
    x = _logistic_map_series(n)
    return pd.DataFrame({
        'time': np.arange(n),
        'cand1': x,
        'cand2': np.roll(x, 2),
        'target_a': np.roll(x, 5),
        'target_b': np.roll(x, 8),
    })


def _brute_force_best_e(df, candidate, targets, lib, pred, tau, Tp, metric, aggregate, max_e=MAX_E):
    '''Independent recomputation: score every E in [1, max_e] directly via
    _simplex_multi_metric (bypassing multi_proper_embed_dimension
    entirely), return (scores_by_E, best_E) for comparison.'''
    scores = {}
    for E in range(1, max_e + 1):
        scores[E] = tsls._simplex_multi_metric(
            df, candidate, targets, lib, pred, tau=tau, Tp=Tp, E=E,
            embedded=False, metric=metric, aggregate=aggregate, noTime=True,
        )
    best_e = max(scores, key=lambda e: scores[e])
    return scores, best_e


def test_multi_proper_embed_dimension_matches_brute_force():
    df = _synthetic_df()
    targets = ['target_a', 'target_b']
    N = len(df)
    lib, pred = [1, N // 2], [N // 2 + 1, N]

    edf = tsls.multi_proper_embed_dimension(
        df, columns='cand1', targets=targets, maxE=MAX_E,
        lib=lib, pred=pred, Tp=1, tau=-1,
        metric='correlation', target_aggregate='mean',
    )
    scores_expected, best_e_expected = _brute_force_best_e(
        df, 'cand1', targets, lib, pred, tau=-1, Tp=1,
        metric='correlation', aggregate='mean',
    )

    reported = dict(zip(edf['E'], edf['correlation']))
    assert set(reported) == set(scores_expected), \
        f'E range mismatch: {sorted(reported)} vs {sorted(scores_expected)}'
    for E in scores_expected:
        assert np.isclose(reported[E], scores_expected[E], equal_nan=True), \
            f'E={E}: multi_proper_embed_dimension={reported[E]} vs brute force={scores_expected[E]}'

    best_e_reported = int(edf.loc[edf['correlation'].idxmax(), 'E'])
    assert best_e_reported == best_e_expected, \
        f'argmax E mismatch: reported={best_e_reported} vs brute force={best_e_expected}'
    print(f'test_multi_proper_embed_dimension_matches_brute_force: PASS -- '
          f'best_E={best_e_reported}, scores match brute force at every E in [1,{MAX_E}]')


def test_static_e_unaffected_by_new_parameter():
    df = _synthetic_df()
    targets = ['target_a', 'target_b']

    result = tsls.multi_proper_mde(
        df, targets=targets, columns=['cand1', 'cand2'], D=2,
        Tp=1, tau=-1, E=3, noCCM=False,
        crossMapRhoMin=-1.0, embedDimRhoMin=-1.0, ccmSlope=-1.0,
        metric='correlation', ccmMetric='correlation',
        libSizes=(16, 24, 40), sample=5, target_aggregate='mean',
        verbose=False,
    )
    assert len(result['MDEcolumns']) > 0, 'nothing selected -- test thresholds too strict'
    assert all(E == 3 for E in result['EDim'].values()), \
        f'static E=3 should be used unconditionally, got EDim={result["EDim"]}'
    print(f'test_static_e_unaffected_by_new_parameter: PASS -- '
          f'MDEcolumns={result["MDEcolumns"]}, EDim={result["EDim"]}')


def test_auto_select_matches_brute_force():
    df = _synthetic_df()
    targets = ['target_a', 'target_b']
    N = len(df)
    lib, pred = [1, N // 2], [N // 2 + 1, N]

    result = tsls.multi_proper_mde(
        df, targets=targets, columns=['cand1', 'cand2'], D=2,
        Tp=1, tau=-1, E=0, maxE=MAX_E, noCCM=False,
        crossMapRhoMin=-1.0, embedDimRhoMin=-1.0, ccmSlope=-1.0,
        metric='correlation', ccmMetric='correlation',
        libSizes=(16, 24, 40), sample=5, target_aggregate='mean',
        verbose=False,
    )
    assert len(result['MDEcolumns']) > 0, 'nothing selected -- test thresholds too strict'
    assert len(result['EDim']) == len(result['MDEcolumns']), \
        'EDim should have one entry per accepted column'

    for newColumn, e_used in result['EDim'].items():
        candidate = newColumn.split(':')[0]
        assert 1 <= e_used <= MAX_E, f'{newColumn}: E={e_used} outside [1,{MAX_E}]'
        _, best_e_expected = _brute_force_best_e(
            df, candidate, targets, lib, pred, tau=-1, Tp=1,
            metric='correlation', aggregate='mean',
        )
        assert e_used == best_e_expected, (
            f'{newColumn}: multi_proper_mde auto-selected E={e_used}, '
            f'brute force says best E={best_e_expected}'
        )
    print(f'test_auto_select_matches_brute_force: PASS -- '
          f'MDEcolumns={result["MDEcolumns"]}, EDim={result["EDim"]}')


if __name__ == '__main__':
    test_multi_proper_embed_dimension_matches_brute_force()
    test_static_e_unaffected_by_new_parameter()
    test_auto_select_matches_brute_force()
    print('ALL PASS')
