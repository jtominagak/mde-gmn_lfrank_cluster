'''The skill gates are resolved per node, against the metric that scores it.

EMBED_DIM_RHO_MIN / CROSS_MAP_RHO_MIN used to be one number applied to two
metrics with different ceilings: `correlation` (behavioural targets) tops out
at 1.0, while a NEAR-PERFECT predictor of a real TSLS channel scores `f1`
~0.49-0.54 (scripts/calibration/r9_min_pred_events_curve.csv). A single 0.2
bar therefore demanded ~38% of achievable skill from a neuron and 20% from a
behavioural target -- part of why 101 nodes in job 4774136 had no candidate
clear the gate at all.

The gates are now a {metric: value} mapping, resolved inside
run_mde_for_target() where the node's own metric is already known. This pins:

  1. _rho_min_for_metric()'s rules, including the plain-float path that keeps
     every older caller working unchanged.
  2. That the resolution actually reaches proper_mde -- a neuron node's call
     is made with the f1 gate, a behavioural target's with the correlation
     gate. This is the part that would otherwise fail silently: a mis-wired
     mapping does not raise, it just quietly changes which links exist.
  3. That the RESOLVED float, not the mapping, is what the per-node cache
     fingerprint sees, so a node cached under one gate is never reused under
     the other.

No pytest in this project's venv -- run directly:
    python3 tests/test_per_metric_rho_gates.py
'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402

PARENT_FILE = 'run_serial_mde.py'
GATES = {'default': 0.2, 'f1': 0.1}


class _Captured(Exception):
    '''Carries proper_mde's kwargs out without needing a valid return value.'''
    def __init__(self, kwargs):
        super().__init__('captured')
        self.kwargs = kwargs


def test_helper_rules(ns):
    f = ns._rho_min_for_metric
    assert f(GATES, 'f1') == 0.1, 'f1 must take the f1 entry'
    assert f(GATES, 'correlation') == 0.2, "correlation must fall through to 'default'"
    assert f(GATES, 'anything_else') == 0.2, "an unlisted metric must take 'default'"
    assert f(0.25, 'f1') == 0.25, 'a plain float must still work for every metric'
    assert f(0.25, 'correlation') == 0.25
    print('PASS: _rho_min_for_metric resolves by metric, and a float still passes through')


def _toy_df(n=400):
    rng = np.random.default_rng(0)
    t = np.arange(n)
    a = np.abs(np.sin(t / 7.0)) * 5
    return pd.DataFrame({'T': np.sin(t / 5.0) + rng.normal(scale=0.01, size=n),
                         'A_tsls': a, 'B_tsls': np.roll(a, 3)})


def _gates_used_for(ns, node):
    '''The gates proper_mde is actually called with for `node`.

    Calls the real run_mde_for_target() with proper_mde replaced, so the
    resolution under test runs for real and nothing else does.'''
    captured = {}

    def fake_proper_mde(*args, **kwargs):
        raise _Captured(kwargs)

    real = ns.edm.proper_mde
    ns.edm.proper_mde = fake_proper_mde
    try:
        ns.run_mde_for_target(
            node, _toy_df(), ['B_tsls'], 300, 50,
            2, GATES, GATES, 0,
            Path('.'), [],
            tp=1, metric='f1',
            # exactly what every real submit site passes
            target_metric=ns._metric_for(node, ['T'], 'f1', 'correlation'),
            gamma_delta=2, ccm_sample=2, ccm_slope=0.01,
            is_behavioral_target=(node == 'T'),
            target_ranking_window_rows=200, min_pred_events=5,
            max_window_rows=30000, ccm_lib_sizes_rows=(10, 50),
        )
    except _Captured as exc:
        captured = exc.kwargs
    finally:
        ns.edm.proper_mde = real
    assert captured, f'proper_mde was never reached for {node}'
    return captured['embedDimRhoMin'], captured['crossMapRhoMin']


def test_resolution_reaches_proper_mde(ns):
    embed, cross = _gates_used_for(ns, 'A_tsls')
    assert (embed, cross) == (0.1, 0.1), (
        f'a TSLS neuron node is scored by f1, so it must use the f1 gate 0.1, got {embed}/{cross}')
    print(f'PASS: neuron node A_tsls -> embedDimRhoMin={embed}, crossMapRhoMin={cross} (f1 gate)')

    embed, cross = _gates_used_for(ns, 'T')
    assert (embed, cross) == (0.2, 0.2), (
        f'a behavioural target is scored by correlation, so it keeps 0.2, got {embed}/{cross}')
    print(f'PASS: behavioural target T -> embedDimRhoMin={embed}, crossMapRhoMin={cross} '
          f'(correlation gate)')


def test_fingerprint_sees_the_resolved_value(ns):
    '''Two nodes under different gates must not share a cache entry.'''
    kw = dict(candidate_pool=['B_tsls'], mde_d=2, tp=1, exclusion_radius=0,
              delay_set=[], prediction_start=300, prediction_length=50,
              metric='f1', target_metric='correlation', gamma_delta=2,
              ccm_sample=2, ccm_slope=0.01, data_file='d.csv',
              data_file_sha256='x', code_fingerprint={'source': 'y'},
              is_behavioral_target=False, target_ranking_window_rows=200,
              min_pred_events=5, max_window_rows=30000, window_rows=200,
              pred_event_count=25, ccm_lib_sizes_rows=(10, 50))
    neuron = ns.build_mde_selection_fingerprint(
        node='A_tsls', embed_dim_rho_min=0.1, cross_map_rho_min=0.1, **kw)
    behav = ns.build_mde_selection_fingerprint(
        node='A_tsls', embed_dim_rho_min=0.2, cross_map_rho_min=0.2, **kw)
    assert neuron != behav, (
        'the resolved gate must be part of the per-node fingerprint, or a node computed '
        'under the f1 gate would be reused under the correlation gate')
    print('PASS: the resolved gate is part of the per-node cache fingerprint')


def test_config_builds_the_mapping(ns):
    import os
    for var in ('TSLS_MDE_GMN_EMBED_DIM_RHO_MIN_F1', 'TSLS_MDE_GMN_CROSS_MAP_RHO_MIN_F1'):
        os.environ.pop(var, None)
    # load_config() creates RESULTS_DIR, which defaults to a path relative to
    # the entry script. Point it at this test's own scratch dir so the check
    # does not depend on where it was run from.
    scratch = Path(__file__).resolve().parent / '_scratch_rho_gates'
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ['TSLS_MDE_GMN_RESULTS_DIR'] = str(scratch)
    cfg = ns.load_config()
    # 0.15, not 0.1: raised 2026-11-06 once a circular-shift surrogate null
    # showed surrogate drivers reaching f1=0.097, so a 0.1 gate sat on the
    # noise floor. See config.py's EMBED_DIM_RHO_MIN_F1 comment.
    assert cfg['EMBED_DIM_RHO_MIN_BY_METRIC'] == {'default': 0.2, 'f1': 0.15}, \
        cfg['EMBED_DIM_RHO_MIN_BY_METRIC']
    assert cfg['CROSS_MAP_RHO_MIN_BY_METRIC'] == {'default': 0.2, 'f1': 0.15}, \
        cfg['CROSS_MAP_RHO_MIN_BY_METRIC']
    os.environ['TSLS_MDE_GMN_CROSS_MAP_RHO_MIN_F1'] = '0.05'
    try:
        cfg = ns.load_config()
        assert cfg['CROSS_MAP_RHO_MIN_BY_METRIC']['f1'] == 0.05, 'the f1 gate must be overridable'
    finally:
        os.environ.pop('TSLS_MDE_GMN_CROSS_MAP_RHO_MIN_F1', None)
        os.environ.pop('TSLS_MDE_GMN_RESULTS_DIR', None)
    print('PASS: load_config builds the mapping and the f1 gate is overridable')


def main():
    ns = load_pipeline_module(PARENT_FILE, '_test_per_metric_rho_gates')
    test_helper_rules(ns)
    test_resolution_reaches_proper_mde(ns)
    test_fingerprint_sees_the_resolved_value(ns)
    test_config_builds_the_mapping(ns)
    print('ALL PASS')


if __name__ == '__main__':
    main()
