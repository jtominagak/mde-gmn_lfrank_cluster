'''Regression test for a metric-routing bug found while verifying R22
(docs/review-actions-v1.md) on real data, not one of the 22 numbered tasks
itself -- see docs/review-findings.md's own section for this fix.

build_nested_mde_network()'s Stage 1 and Stage 2 used to pass the closed-
over `target_metric` (cfg['TARGET_METRIC'], 'correlation') unconditionally
into every run_mde_for_target() call, for every node -- correct only at
depth 0, where the node is always a real behavioral TARGETS column, but
wrong for every discovered driver node at depth >= 1, which is virtually
always a neuron and should score its own ranking stage / E-auto-select via
_metric_for()'s own rule (`metric`, 'f1' by default), not `target_metric`.

This test asserts the per-node resolution directly: a target node gets
`target_metric` unchanged; a non-target (driver) node gets `metric`
resolved in its place -- captured directly from the arguments
run_mde_for_target() actually receives, not inferred from a downstream
side effect.

No pytest in this project's venv -- run directly:
    python3 tests/test_metric_routing_depth_ge1.py
'''
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402

CFG_METRIC = 'f1'
CFG_TARGET_METRIC = 'correlation'


class _FakeExecutor:
    '''Runs every submit() synchronously, same convention as
    _pipeline_test_utils.FakeProcessPoolExecutor -- but this test needs its
    own fake run_mde_for_target() (capturing the metric/target_metric
    arguments actually passed, not just the call order), so it doesn't
    reuse that module's shared fake.'''
    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, fn, *args, **kwargs):
        import concurrent.futures
        fut = concurrent.futures.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover
            fut.set_exception(exc)
        return fut


def _make_fake_run_mde_for_target(plan, calls):
    '''Records (node, metric, target_metric) for every call, in addition to
    returning a plan-driven fake result -- same shape run_mde_for_target()
    itself returns, matching _pipeline_test_utils._fake_result().'''
    def fake(target, df, observable_columns, prediction_start, prediction_length,
             mde_d, embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
             results_dir, delay_set, tp=1,
             metric='f1', target_metric='correlation', gamma_delta=2,
             ccm_sample=20, ccm_slope=0.01,
             is_behavioral_target=True, target_ranking_window_rows=1000,
             min_pred_events=20, max_window_rows=30000,
             ccm_lib_sizes_rows=(16, 24, 40, 64, 160, 400, 800),
             reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
             data_file=None, data_file_sha256=None, code_fingerprint=None,
             cache_stage='stage1', max_e=None, inner_workers=1, raw_columns=None,
             use_gpu=False):
        calls.append((cache_stage, target, metric, target_metric))
        selected = plan.get((cache_stage, target), [])
        gains = [0.5] * len(selected)
        return {'SELECTED_COLUMNS': list(selected), 'm': SimpleNamespace(MDErho=np.array(gains))}
    return fake


def test_target_metric_resolved_per_node():
    module = load_pipeline_module('tsls-mde-gmn.py', '_test_tsls_mde_gmn_metric_routing')

    # T (target) -> A (real observable, Stage 1) -> B (real observable,
    # Stage 2's own recursion, STAGE2_ALLOW_REAL_OBSERVABLES=True) --
    # neither A nor B is ever in `targets`, so both must resolve to
    # CFG_METRIC ('f1'), never CFG_TARGET_METRIC ('correlation').
    plan = {
        ('stage1', 'T'): ['A'],
        ('stage1', 'A'): [],
        ('stage2', 'T'): [],
        ('stage2', 'A'): ['B'],
        ('stage2', 'B'): [],
    }
    calls = []
    fake = _make_fake_run_mde_for_target(plan, calls)

    module.ProcessPoolExecutor = _FakeExecutor
    module.run_mde_for_target = fake

    targets = ['T']
    df = pd.DataFrame({'Time': range(10), 'T': range(10), 'A': range(10), 'B': range(10)})

    graph, edges_df, nested_mde_results = module.build_nested_mde_network(
        targets, df, raw_columns=['A', 'B'], expanded_columns=['A', 'B'],
        prediction_start=5, prediction_length=5, nested_n=2, nested_m=3,
        embed_dim_rho_min=0.2, cross_map_rho_min=0.2, exclusion_radius=0,
        results_dir=Path('/tmp'),
        mde_results={'T': {'SELECTED_COLUMNS': ['A'], 'm': SimpleNamespace(MDErho=np.array([0.5]))}},
        delay_set=[], tp=1,
        metric=CFG_METRIC, target_metric=CFG_TARGET_METRIC, gamma_delta=2,
        target_ranking_window_rows=1000, min_pred_events=20, max_window_rows=30000,
        ccm_lib_sizes_rows=(16, 24, 40, 64, 160, 400, 800),
        ccm_sample=10, ccm_slope=0.01, cores=2,
        lag_set=[-1], stage2_allow_real_observables=True,
    )

    assert calls, 'no run_mde_for_target() calls captured at all'
    for cache_stage, node, metric, target_metric in calls:
        if node in targets:
            assert target_metric == CFG_TARGET_METRIC, (
                f'{cache_stage}/{node}: target node got target_metric={target_metric!r}, '
                f'expected {CFG_TARGET_METRIC!r}'
            )
        else:
            assert target_metric == CFG_METRIC, (
                f'{cache_stage}/{node}: non-target node got target_metric={target_metric!r} '
                f'(should be _metric_for()-resolved to {CFG_METRIC!r}, the bug this test guards against)'
            )
        # The CCM-gate metric is untouched by this fix -- always CFG_METRIC,
        # for every node, target or not (it only ever scores a candidate).
        assert metric == CFG_METRIC, f'{cache_stage}/{node}: CCM-gate metric changed unexpectedly: {metric!r}'

    print(f'test_target_metric_resolved_per_node: PASS -- calls={calls}')


if __name__ == '__main__':
    test_target_metric_resolved_per_node()
    print('ALL PASS')
