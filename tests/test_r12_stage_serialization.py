'''R12 acceptance test (docs/review-actions-v1.md): build_nested_mde_network()
must issue every Stage 2 MDE call only after every Stage 1 MDE call has
returned.

The two pipelines then diverge deliberately (2026-09-30): simple-mde-gmn.py
still lets a real-observable driver Stage 2 selects recurse into its own
further Stage 2 round, while tsls-mde-gmn.py's Stage 2 is a single
non-recursive lagged-driver pass and must never recurse.

No pytest in this project's venv (see docs/review-findings.md, R12 section)
-- run directly:
    python3 tests/test_r12_stage_serialization.py

Uses fake run_mde_for_target()/ProcessPoolExecutor (see
_pipeline_test_utils.py) so this exercises only the BFS/recursion control
flow, not real MDE/CCM computation -- no real data file needed.
'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import (  # noqa: E402
    install_fake_mde,
    load_pipeline_module,
    make_fake_run_mde_for_target_simple,
    make_fake_run_mde_for_target_tsls,
)

# Scenario shared by both pipelines' build_nested_mde_network():
#   Target T -- Stage 1 selects real observable A (delay 0). A's own
#   Stage 1 call selects nothing (dead end).
#   Stage 2 (stage2_allow_real_observables=True): T's Stage 2 call selects
#   real observable B -- one Stage 1 never found, so B only enters the
#   graph via Stage 2's own real-observable pick, and per R12/R15 must
#   recurse into its OWN further Stage 2 round (not a Stage 1 round). A's
#   and B's own Stage 2 calls both select nothing (dead ends).
PLAN = {
    ('stage1', 'T'): ['A'],
    ('stage1', 'A'): [],
    ('stage2', 'T'): ['B'],
    ('stage2', 'A'): [],
    ('stage2', 'B'): [],
}


def _toy_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({'A': rng.normal(size=20), 'B': rng.normal(size=20)})


def _run(module, fake_run_mde_for_target, extra_kwargs, stage2_kwargs):
    call_log = []
    install_fake_mde(module, fake_run_mde_for_target(PLAN, call_log))

    graph, edges_df, nested_mde_results = module.build_nested_mde_network(
        targets=['T'],
        expanded_df=_toy_df(),
        raw_columns=['A', 'B'],
        expanded_columns=['A', 'B'],
        prediction_start=10,
        prediction_length=5,
        nested_n=2,
        nested_m=3,
        embed_dim_rho_min=0.2,
        cross_map_rho_min=0.2,
        exclusion_radius=0,
        results_dir=Path('.'),
        mde_results={},
        delay_set=[],
        tp=1,
        lag_set=[-1],
        **stage2_kwargs,
        **extra_kwargs,
    )
    return graph, edges_df, nested_mde_results, call_log


def _check(name, graph, edges_df, call_log, stage2_recurses):
    # ---- (1) serialization: every stage1 call before every stage2 call ----
    stage1_positions = [i for i, (stage, _) in enumerate(call_log) if stage == 'stage1']
    stage2_positions = [i for i, (stage, _) in enumerate(call_log) if stage == 'stage2']
    assert stage1_positions and stage2_positions, f'{name}: expected both stages to run'
    assert max(stage1_positions) < min(stage2_positions), (
        f'{name}: a Stage 2 MDE call was issued before every Stage 1 call returned '
        f'-- call_log={call_log}'
    )

    # ---- (2) what Stage 2 does with a real-observable pick ----
    if stage2_recurses:
        # simple-mde-gmn.py: the pick recurses into its OWN further Stage 2
        # round, and never gets a Stage-1-style call.
        assert ('stage2', 'B') in call_log, (
            f'{name}: B (Stage 2\'s real-observable pick from T) never got its own Stage 2 '
            f'call -- recursion is broken'
        )
        assert ('stage1', 'B') not in call_log, (
            f'{name}: B got a Stage 1 call -- Stage 2 recursion must never trigger a fresh '
            f'Stage-1-style call'
        )
    else:
        # tsls-mde-gmn.py (2026-09-30): Stage 2 is a single non-recursive
        # pass over Stage 1's nodes, so nothing it selects gets an MDE call
        # of its own in either stage.
        assert ('stage2', 'B') not in call_log and ('stage1', 'B') not in call_log, (
            f'{name}: Stage 2 recursed into B -- Stage 2 must not recurse '
            f'(call_log={call_log})'
        )
    # Every Stage 1 node (T, A) also got its own Stage 2 call (lagged/real
    # embedding attachment, unconditionally -- no depth-0 shortcut).
    assert ('stage2', 'T') in call_log and ('stage2', 'A') in call_log, (
        f'{name}: a Stage 1 backbone node never got its Stage 2 call -- call_log={call_log}'
    )

    # ---- resulting graph: T->A (delay 0), T->B (delay 0), B present ----
    assert set(graph.nodes) >= {'T', 'A', 'B'}, f'{name}: graph.nodes={set(graph.nodes)}'
    edges = {(row.driver, row.node, row.delay) for row in edges_df.itertuples()}
    assert ('A', 'T', 0) in edges, f'{name}: missing T<-A stage1 edge -- edges={edges}'
    assert ('B', 'T', 0) in edges, f'{name}: missing T<-B stage2 edge -- edges={edges}'

    print(f'{name}: PASS ({len(call_log)} MDE calls, order={call_log})')


def test_tsls():
    module = load_pipeline_module('tsls-mde-gmn.py', '_test_tsls_mde_gmn')
    graph, edges_df, _, call_log = _run(
        module, make_fake_run_mde_for_target_tsls,
        {'metric': 'correlation', 'target_metric': 'correlation', 'gamma_delta': 2,
         'target_ranking_window_rows': 200, 'min_pred_events': 5, 'max_window_rows': 30000,
         'ccm_lib_sizes_rows': (10, 50), 'ccm_sample': 2,
         'ccm_slope': 0.01, 'cores': 2},
        {},
    )
    _check('tsls-mde-gmn.py', graph, edges_df, call_log, stage2_recurses=False)


def test_simple():
    module = load_pipeline_module('simple-mde-gmn.py', '_test_simple_mde_gmn')
    graph, edges_df, _, call_log = _run(
        module, make_fake_run_mde_for_target_simple, {'cores': 2},
        {'stage2_allow_real_observables': True},
    )
    _check('simple-mde-gmn.py', graph, edges_df, call_log, stage2_recurses=True)


if __name__ == '__main__':
    test_tsls()
    test_simple()
    print('ALL PASS')
