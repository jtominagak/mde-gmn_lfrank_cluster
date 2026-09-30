'''R20 acceptance test (docs/review-actions-v1.md): tsls-mde-gmn.py's new
proper_predict_nonlinear() -- a thin pyEDM.SMap-based scoring wrapper that
replaces pyEDM.PredictNonlinear in retune_theta_for_all_nodes(), so theta
retuning routes through _metric_for()/_score() like every other scoring
call site in this pipeline instead of pyEDM's own internal Pearson rho
regardless of METRIC.

R20's own text requires this NOT be a new S-map engine (R10 already found
none exists in cleaned_helper_functions.py, and R20 forbids writing one) --
this wrapper calls pyEDM.SMap directly, the same compiled S-map
PredictNonlinear itself calls internally per theta (confirmed by reading
pyEDM/API.py's PredictNonlinear and pyEDM/PoolFunc.py's PredictNLSMapFunc).
So the acceptance test here is an equivalence check against pyEDM's own
PredictNonlinear at metric='correlation' (Pearson-rho settings) -- both
should select the same best theta and report matching rho, since
ComputeError's own rho and _score(kind='correlation') are the identical
np.corrcoef formula.

No pytest in this project's venv -- run directly:
    python3 tests/test_r20_predict_nonlinear.py
'''
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from pyEDM import PredictNonlinear

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402

THETA_RANGE = [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8]


def _logistic_map_series(n, r=3.95, x0=0.4):
    '''A short logistic-map trajectory -- deterministic chaos with genuine
    curvature, so the theta sweep produces an actually-peaked rho curve
    (not flat/noise), the standard toy system EDM's own S-map literature
    validates against.'''
    x = np.empty(n)
    x[0] = x0
    for i in range(1, n):
        x[i] = r * x[i - 1] * (1 - x[i - 1])
    return x


def _embed_columns(series, e, tau=1):
    '''Builds an (E, tau) delay embedding dataframe matching this
    pipeline's own _generalized_embed()/pyEDM 'col(t-k)' naming, plus the
    plain target column -- the same shape retune_theta_for_all_nodes()
    builds via _generalized_embed() before calling
    proper_predict_nonlinear()/PredictNonlinear.'''
    n = len(series)
    offset = e * tau
    cols = {}
    for k in range(e):
        delay = k * tau
        cols[f'x(t-{delay})'] = series[offset - delay: n - delay]
    df = pd.DataFrame(cols)
    df.insert(0, 'Time', np.arange(len(df)))
    df['x'] = series[offset:]
    return df


def test_equivalence_against_pyedm_predictnonlinear():
    module = load_pipeline_module('tsls-mde-gmn.py', '_test_tsls_mde_gmn_r20')

    series = _logistic_map_series(600)
    E = 3
    embedded = _embed_columns(series, E)
    columns = list(embedded.columns[1:-1])  # exclude Time and the target column

    n = len(embedded)
    split = int(n * 0.7)
    lib_range = f'1 {split}'
    pred_range = f'{split + 1} {n}'

    pn = PredictNonlinear(
        dataFrame=embedded, columns=columns, target='x',
        theta=list(THETA_RANGE), lib=lib_range, pred=pred_range,
        Tp=1, embedded=True, showPlot=False, numProcess=1,
    )
    ours = module.proper_predict_nonlinear(
        embedded, columns=columns, target='x',
        theta=list(THETA_RANGE), lib=lib_range, pred=pred_range,
        Tp=1, embedded=True, metric='correlation', gamma_delta=2,
    )

    pn_best_theta = float(pn.loc[pn['rho'].idxmax(), 'theta'])
    our_best_theta = float(ours.loc[ours['score'].idxmax(), 'theta'])
    assert pn_best_theta == our_best_theta, (
        f'best theta mismatch: pyEDM PredictNonlinear picked {pn_best_theta}, '
        f'proper_predict_nonlinear picked {our_best_theta}\n{pn}\n{ours}'
    )

    # Every theta's own rho must match too (not just the argmax), confirming
    # this is a faithful per-candidate reproduction, not a coincidental match
    # at one point on the curve.
    merged = pn.merge(ours, on='theta', suffixes=('_pyedm', '_ours'))
    assert len(merged) == len(THETA_RANGE)
    max_diff = (merged['rho'] - merged['score']).abs().max()
    assert max_diff < 1e-6, f'rho/score mismatch across theta grid (max diff {max_diff}):\n{merged}'

    print(f'test_equivalence_against_pyedm_predictnonlinear: PASS -- '
          f'best_theta={our_best_theta}, max |rho - score| across grid = {max_diff:.2e}')


def _toy_graph_and_data():
    '''T (target) <- A (driver, real observable, delay 0) -- same tiny
    synthetic pattern as tests/test_r11_per_node_table.py/
    test_r14_external_inputs.py.'''
    rng = np.random.default_rng(3)
    n = 300
    t = np.arange(n)
    a_series = np.sin(t / 6.0) + rng.normal(scale=0.02, size=n)
    t_series = 0.7 * a_series + rng.normal(scale=0.02, size=n)
    data = pd.DataFrame({'Time': t, 'T': t_series, 'A': a_series})

    graph = nx.MultiDiGraph()
    graph.add_node('T')
    graph.add_node('A')
    graph.add_edge('A', 'T', delay=0, weight=0.5)
    return graph, data


def test_retune_theta_for_all_nodes_resolves_metric_per_node():
    '''Direct check that retune_theta_for_all_nodes() resolves each node's
    own metric via _metric_for() (node 'A' is not a target -> 'f1'; 'T' is
    -> target_metric='correlation') rather than a single hardcoded metric
    for every node, the way pyEDM.PredictNonlinear's own internal rho did.
    Deliberately checked at the _metric_for() call itself (deterministic,
    metric-agnostic) rather than by asserting on 'f1''s own TSLS-specific
    numeric behavior against a plain (non-TSLS) synthetic series, which
    this test's toy data was never built to exercise meaningfully.'''
    module = load_pipeline_module('tsls-mde-gmn.py', '_test_tsls_mde_gmn_r20')
    targets = ['T']
    assert module._metric_for('A', targets, 'f1', 'correlation') == 'f1'
    assert module._metric_for('T', targets, 'f1', 'correlation') == 'correlation'

    # Integration smoke check: the end-to-end call (real _embedding_spec_for +
    # _generalized_embed + proper_predict_nonlinear per node) runs to
    # completion and returns a finite theta for every node, for both the
    # target (scored via 'correlation') and the driver (scored via 'f1').
    graph, data = _toy_graph_and_data()
    best_theta = module.retune_theta_for_all_nodes(
        data, graph, ['T', 'A'], targets=targets, theta_range=THETA_RANGE, cores=1,
        metric='f1', target_metric='correlation', gamma_delta=2,
    )
    assert set(best_theta.keys()) == {'T', 'A'}
    for node, theta in best_theta.items():
        assert np.isfinite(theta), f'{node}: non-finite theta {theta}'
    print(f'test_retune_theta_for_all_nodes_resolves_metric_per_node: PASS -- '
          f'best_theta={best_theta}')


if __name__ == '__main__':
    test_equivalence_against_pyedm_predictnonlinear()
    test_retune_theta_for_all_nodes_resolves_metric_per_node()
    print('ALL PASS')
