'''R13 acceptance test (docs/review-actions-v1.md): NESTED_M is a
WHOLE-NETWORK depth budget shared jointly by Stage 1 and Stage 2 -- a
node's depth is its distance from the target regardless of which stage
discovered it, and Stage 2 may not extend a path beyond what Stage 1 has
already consumed on a given branch.

R13 made no arithmetic/logic change (the joint-budget behavior already
held, both before and after R12 -- see docs/review-findings.md's R13
section for the source-reading that confirmed this). This test exists so
that fact stays true: it fails if a future change makes the depth cap
per-stage instead of joint, and it exercises the exact interaction A2/R15
describe -- Stage 1 running a branch out to the depth cap, then a Stage 2
real-observable pick on that same branch being accepted as an edge but
never recursed into (budget-bound, not driver-exhausted).

No pytest in this project's venv -- run directly:
    python3 tests/test_r13_shared_depth_budget.py
'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import (  # noqa: E402
    FakeProcessPoolExecutor,
    load_pipeline_module,
    make_fake_run_mde_for_target_simple,
    make_fake_run_mde_for_target_tsls,
)

NESTED_M = 1

# T (depth 0) -- Stage 1 selects real observable A. A (depth 1, exactly at
# the nested_m=1 cap) -- Stage 1 selects nothing further, so Stage 1's own
# backbone stops at depth 1 (2 rounds total: T's call, A's call -- "1 +
# NESTED_M" rounds, matching the existing convention).
#
# Stage 2 then runs on {T, A}. A's own Stage 2 call (still at depth 1, the
# cap Stage 1 already reached on this branch) selects a BRAND NEW real
# observable D -- this must be accepted as a graph edge (A -> D) but must
# NOT recurse into its own further Stage 2 round, because depth 1 is
# already at nested_m=1. D therefore never gets any MDE call in either
# stage -- exactly the "budget-bound, not driver-exhausted" case A2/R15
# describe.
PLAN = {
    ('stage1', 'T'): ['A'],
    ('stage1', 'A'): [],
    ('stage2', 'T'): [],
    ('stage2', 'A'): ['D'],
}


def _toy_df():
    rng = np.random.default_rng(1)
    return pd.DataFrame({'A': rng.normal(size=20), 'B': rng.normal(size=20),
                          'C': rng.normal(size=20), 'D': rng.normal(size=20)})


def _run(module, fake_run_mde_for_target, extra_kwargs):
    call_log = []
    module.run_mde_for_target = fake_run_mde_for_target(PLAN, call_log)
    module.ProcessPoolExecutor = FakeProcessPoolExecutor

    graph, edges_df, _ = module.build_nested_mde_network(
        targets=['T'],
        expanded_df=_toy_df(),
        raw_columns=['A', 'B', 'C', 'D'],
        expanded_columns=['A', 'B', 'C', 'D'],
        prediction_start=10,
        prediction_length=5,
        nested_n=2,
        nested_m=NESTED_M,
        embed_dim_rho_min=0.2,
        cross_map_rho_min=0.2,
        exclusion_radius=0,
        results_dir=Path('.'),
        mde_results={},
        delay_set=[],
        tp=1,
        lag_set=[-1],
        stage2_allow_real_observables=True,
        **extra_kwargs,
    )
    return graph, edges_df, call_log


def _check(name, graph, edges_df, call_log):
    # ---- Stage 1's own backbone stopped exactly at the cap (1 + NESTED_M
    # rounds: T's call, A's call -- no third round) ----
    assert ('stage1', 'T') in call_log and ('stage1', 'A') in call_log, (
        f'{name}: expected Stage 1 to explore both T and A -- call_log={call_log}'
    )
    stage1_calls = [t for stage, t in call_log if stage == 'stage1']
    assert len(stage1_calls) == 2, (
        f'{name}: Stage 1 should make exactly 1+NESTED_M={1 + NESTED_M} calls, got '
        f'{stage1_calls}'
    )

    # ---- D (Stage 2's real-observable pick from A, already at the cap)
    # got an edge but NEVER its own MDE call in either stage -- budget-
    # bound termination, joint across stages ----
    assert ('stage2', 'D') not in call_log and ('stage1', 'D') not in call_log, (
        f'{name}: D should never get its own MDE call (budget-bound on a branch Stage 1 '
        f'already maxed out) -- call_log={call_log}'
    )
    edges = {(row.driver, row.node, row.delay) for row in edges_df.itertuples()}
    assert ('D', 'A', 0) in edges, f'{name}: missing the A<-D edge itself -- edges={edges}'

    # ---- longest root-to-leaf path, counted in nodes: NESTED_M + 2 --
    # (NESTED_M + 1) nodes actually explored (T at depth 0, A at depth
    # NESTED_M) plus the one further leaf (D) that A's own call discovered
    # but the joint budget stopped from being explored itself. Graph edges
    # point driver -> node (see _accept_mde_result_edges()), i.e. FROM the
    # thing that drives TO the target it drives, so the path runs D -> A
    # -> T, not T -> D.
    import networkx as nx
    longest = max(
        len(path) for path in nx.all_simple_paths(graph, source='D', target='T')
    )
    assert longest == NESTED_M + 2, (
        f'{name}: longest D->T path has {longest} nodes, expected {NESTED_M + 2} '
        f'(NESTED_M={NESTED_M})'
    )

    print(f'{name}: PASS (Stage 1 rounds={len(stage1_calls)}, D->T path={longest} nodes, '
          f'call_log={call_log})')


def test_tsls():
    module = load_pipeline_module('tsls-mde-gmn.py', '_test_tsls_mde_gmn_r13')
    graph, edges_df, call_log = _run(
        module, make_fake_run_mde_for_target_tsls,
        {'metric': 'correlation', 'target_metric': 'correlation', 'gamma_delta': 2,
         'target_ranking_window_rows': 200, 'min_pred_events': 5, 'max_window_rows': 30000,
         'ccm_lib_sizes_rows': (10, 50), 'ccm_sample': 2,
         'ccm_slope': 0.01, 'cores': 2},
    )
    _check('tsls-mde-gmn.py', graph, edges_df, call_log)


def test_simple():
    module = load_pipeline_module('simple-mde-gmn.py', '_test_simple_mde_gmn_r13')
    graph, edges_df, call_log = _run(
        module, make_fake_run_mde_for_target_simple, {'cores': 2},
    )
    _check('simple-mde-gmn.py', graph, edges_df, call_log)


if __name__ == '__main__':
    test_tsls()
    test_simple()
    print('ALL PASS')
