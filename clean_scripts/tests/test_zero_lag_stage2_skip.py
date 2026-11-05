'''Regression test: an EMPTY lag set skips Stage 2 ENTIRELY, in both builders.

DELAY_SOURCE='explicit' with DELAY_SET=0 resolves to an empty lag set --
derive_lag_set() strips every 0 entry on the explicit path -- which leaves
Stage 2 with no candidate columns at all. Before 2026-11-05 both builders ran
Stage 2 anyway and reached

    effective_mde_dimension(mde_d, candidate_columns, num_delays)
        num_real_observables_available = len(candidate_columns) // num_delays

with num_delays == len(lag_set) == 0. Saion job 4774133 died there with
ZeroDivisionError after Stage 1 had completed (14 min in, 202 Stage 1
selections already cached). Its union twin job_4774136 survived only because
union_num_delays is len(lag_set) + 1, which is never 0.

What this pins:
  1. With an empty lag set, NO 'stage2' MDE call is made -- by either builder,
     under scope='staged' -- and the build completes.
  2. The resulting network is Stage 1's: every edge delay 0, no lagged leaves.
  3. scope='union' still works with an empty lag set (it always did; this
     guards the path job_4774136 proved).
  4. The skip is not over-broad: with a NON-empty lag set, Stage 2 still runs
     in both builders.

No pytest in this project's venv -- run it directly:
    python3 tests/test_zero_lag_stage2_skip.py

Uses a fake run_mde_for_target(), so only the BFS/edge-acceptance control flow
runs -- no real MDE/CCM computation and no data file.
'''
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import (  # noqa: E402
    install_fake_mde,
    load_pipeline_module,
)

# Loaded only to set up sys.path and the gmn patches; the builders themselves
# come from mde_gmn.network / mde_gmn.network_parallel (see main()).
PARENT_FILE = 'run_serial_mde.py'

TARGETS = ['T']
RAW_COLUMNS = ['A', 'B', 'C']

#   T <- A      backbone root
#   A <- B      one level down
# Stage 2, when it runs at all, would attach one lagged leaf per node.
PLAN_REAL = {'T': ['A'], 'A': ['B'], 'B': [], 'C': []}
PLAN_LAGGED = {'T': ['A::delay-1'], 'A': ['B::delay-1'], 'B': [], 'C': []}

COMMON_KWARGS = dict(
    prediction_start=10,
    prediction_length=5,
    nested_n=3,
    nested_m=0,            # uncapped, so the backbone is driver-limited
    embed_dim_rho_min=0.2,
    cross_map_rho_min=0.2,
    exclusion_radius=0,
    results_dir=Path('.'),
    mde_results={},
    delay_set=[],
    tp=1,
    metric='correlation',
    target_metric='correlation',
    gamma_delta=2,
    target_ranking_window_rows=200,
    min_pred_events=5,
    max_window_rows=30000,
    ccm_lib_sizes_rows=(10, 50),
    ccm_sample=2,
    ccm_slope=0.01,
    cores=4,
)


def _toy_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.normal(size=40) for c in TARGETS + RAW_COLUMNS})


def _pool_aware_fake(call_log):
    '''Same contract as test_parallel_mde_equivalence.py's own fake: picks come
    from PLAN_REAL/PLAN_LAGGED by stage, filtered to what the candidate pool
    actually offered, so an empty pool genuinely selects nothing.'''
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
             **kwargs):
        call_log.append((cache_stage, target))
        wanted = {
            'stage1': PLAN_REAL.get(target, []),
            'stage2': PLAN_LAGGED.get(target, []),
            'union': PLAN_REAL.get(target, []) + PLAN_LAGGED.get(target, []),
        }[cache_stage]
        offered = set(observable_columns)
        selected = [c for c in wanted if c in offered]
        return {'SELECTED_COLUMNS': selected,
                'm': SimpleNamespace(MDErho=np.array([0.5] * len(selected)))}
    return fake


def _run(builder_module, lag_set, extra_kwargs=None):
    call_log = []
    install_fake_mde(builder_module, _pool_aware_fake(call_log))
    graph, edges_df, _results = builder_module.build_nested_mde_network(
        targets=list(TARGETS),
        expanded_df=_toy_df(),
        raw_columns=list(RAW_COLUMNS),
        expanded_columns=list(RAW_COLUMNS),
        lag_set=list(lag_set),
        **COMMON_KWARGS,
        **(extra_kwargs or {}),
    )
    return graph, edges_df, call_log


def _edge_rows(edges_df):
    return [(r.driver, r.node, r.delay, round(float(r.weight), 12))
            for r in edges_df.itertuples()]


def _assert_stage1_only(name, graph, edges_df, call_log):
    stage2_calls = [c for c in call_log if c[0] == 'stage2']
    assert not stage2_calls, (
        f'{name}: Stage 2 ran with an empty lag set -- {stage2_calls}. There are no lagged '
        f'candidates to select, so the stage must be skipped outright, not run over an '
        f'empty pool (that is the ZeroDivisionError job 4774133 hit).')
    for driver, node, delay, _w in _edge_rows(edges_df):
        assert delay == 0, (
            f'{name}: edge {driver}->{node} has delay={delay}, but the lag set is empty')
    lagged = [n for n, a in graph.nodes(data=True) if 'lagged_of' in a]
    assert not lagged, f'{name}: lagged leaf nodes {lagged} exist with an empty lag set'
    # The Stage 1 backbone itself must still be built: skipping Stage 2 must
    # not be mistaken for skipping the build.
    assert _edge_rows(edges_df) == [('A', 'T', 0, 0.5), ('B', 'A', 0, 0.5)], (
        f'{name}: Stage 1 backbone is wrong: {_edge_rows(edges_df)}')


def test_serial_builder_skips_stage2(parent):
    graph, edges_df, call_log = _run(parent, lag_set=[])
    _assert_stage1_only('serial', graph, edges_df, call_log)
    print('PASS: serial builder skips Stage 2 on an empty lag set')
    return _edge_rows(edges_df)


def test_parallel_staged_skips_stage2(fork, serial_edges):
    graph, edges_df, call_log = _run(fork, lag_set=[],
                                     extra_kwargs={'parallel_mde_scope': 'staged'})
    _assert_stage1_only('parallel/staged', graph, edges_df, call_log)
    # Degenerate-case equivalence, the question job 4774133/4774136 were a
    # pair to answer: with no lags, staged and the serial builder agree.
    assert _edge_rows(edges_df) == serial_edges, (
        f'parallel/staged disagrees with the serial builder on an empty lag set: '
        f'{_edge_rows(edges_df)} vs {serial_edges}')
    print('PASS: parallel/staged skips Stage 2 on an empty lag set, and matches serial')


def test_parallel_union_still_works(fork, serial_edges):
    # union_num_delays is len(lag_set) + 1, so this path never divided by zero
    # -- job_4774136 completed where its staged twin crashed. Guard it.
    graph, edges_df, call_log = _run(fork, lag_set=[],
                                     extra_kwargs={'parallel_mde_scope': 'union'})
    assert not [c for c in call_log if c[0] == 'stage2'], (
        f'parallel/union: scope=union must make no stage2 calls at all: {call_log}')
    assert all(c[0] == 'union' for c in call_log), (
        f'parallel/union: every call must be a union call: {call_log}')
    for driver, node, delay, _w in _edge_rows(edges_df):
        assert delay == 0, f'parallel/union: edge {driver}->{node} has delay={delay}'
    assert _edge_rows(edges_df) == serial_edges, (
        f'parallel/union disagrees with the serial builder on an empty lag set: '
        f'{_edge_rows(edges_df)} vs {serial_edges}')
    print('PASS: parallel/union builds the same Stage-1-only network on an empty lag set')


def test_skip_is_not_over_broad(parent, fork):
    '''A NON-empty lag set must still run Stage 2 in both builders -- otherwise
    the fix above would have silently deleted the stage.'''
    for name, module, extra in (('serial', parent, None),
                                ('parallel/staged', fork, {'parallel_mde_scope': 'staged'})):
        _graph, edges_df, call_log = _run(module, lag_set=[-1], extra_kwargs=extra)
        stage2_calls = [c for c in call_log if c[0] == 'stage2']
        assert stage2_calls, (
            f'{name}: lag_set=[-1] is NOT empty, so Stage 2 must still run: {call_log}')
        lagged_edges = [e for e in _edge_rows(edges_df) if e[2] != 0]
        assert lagged_edges, (
            f'{name}: lag_set=[-1] produced no lagged edges: {_edge_rows(edges_df)}')
    print('PASS: a non-empty lag set still runs Stage 2 in both builders')


def main():
    # One entry script puts scripts/src on sys.path and applies the gmn
    # patches; the two builders are then taken from the package directly, the
    # same way test_parallel_mde_equivalence.py does it.
    load_pipeline_module(PARENT_FILE, '_test_zero_lag_tsls_mde_gmn')
    from mde_gmn import network as parent           # noqa: E402 -- needs the sys.path above
    from mde_gmn import network_parallel as fork     # noqa: E402

    serial_edges = test_serial_builder_skips_stage2(parent)
    test_parallel_staged_skips_stage2(fork, serial_edges)
    test_parallel_union_still_works(fork, serial_edges)
    test_skip_is_not_over_broad(parent, fork)
    print('ALL PASS')


if __name__ == '__main__':
    main()
