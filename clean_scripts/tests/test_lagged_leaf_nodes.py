'''Acceptance test for the 2026-09-30 algorithm change in tsls-mde-gmn.py:

  1. A lagged pick becomes ITS OWN LEAF NODE ('A::delay-1'), carrying
     lagged_of/lag attributes, with no in-edges -- never an edge out of the
     raw observable's own node.
  2. A lagged copy of an observable that is ALREADY in the network upstream
     of the node is therefore a legal driver. Before this change it was
     rejected as a cycle, because the edge came from the real node -- that
     was the bug this test pins down.
  3. A node's own lag is legal too: the self-loop guard applies to delay=0
     only.
  4. Stage 1 excludes observables that would close a cycle from the
     candidate pool BEFORE the MDE call -- and that set is the node's
     DESCENDANTS in edge direction (edges run driver -> node), i.e. the
     nodes between it and the target.
  5. embedding_spec_for() resolves a leaf in-edge back to (raw observable,
     delay), since an embedding coordinate has to name a real data column.

No pytest in this project's venv (see docs/review-findings.md) -- run it
directly:
    python3 clean_scripts/tests/test_lagged_leaf_nodes.py

Uses a fake run_mde_for_target()/ProcessPoolExecutor, so only the
BFS/edge-acceptance control flow runs -- no real MDE/CCM computation and no
data file.
'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import (  # noqa: E402
    install_fake_mde,
    load_pipeline_module,
    _fake_result,
)

# Backbone: T <- A <- B (edges point driver -> node), so for node B the
# cycle-closing set is {A, T} (B's descendants), and for node A it is {T}.
#
# Stage 2 then attaches lagged drivers:
#   T <- A::delay-1   A is upstream of T -- the case that used to be
#                     rejected as a cycle (check 2)
#   A <- A::delay-2   a node's own lag (check 3)
#   B <- C::delay-1   a lag of an observable Stage 1 never selected
PLAN = {
    ('stage1', 'T'): ['A'],
    ('stage1', 'A'): ['B'],
    ('stage1', 'B'): [],
    ('stage2', 'T'): ['A::delay-1'],
    ('stage2', 'A'): ['A::delay-2'],
    ('stage2', 'B'): ['C::delay-1'],
}
RAW_COLUMNS = ['A', 'B', 'C']
LAG_SET = [-1, -2]


def _toy_df():
    rng = np.random.default_rng(3)
    return pd.DataFrame({c: rng.normal(size=40) for c in ['T'] + RAW_COLUMNS})


def _fake_with_pools(plan, call_log, pool_log):
    '''Same contract as _pipeline_test_utils.make_fake_run_mde_for_target_tsls,
    but also records each call's candidate pool so check 4 can inspect it.'''
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
        pool_log[(cache_stage, target)] = list(observable_columns)
        return _fake_result(plan, call_log, cache_stage, target)
    return fake


def run_network():
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_lagged_leaves')
    call_log, pool_log = [], {}
    install_fake_mde(module, _fake_with_pools(PLAN, call_log, pool_log))

    graph, edges_df, _ = module.build_nested_mde_network(
        targets=['T'],
        expanded_df=_toy_df(),
        raw_columns=RAW_COLUMNS,
        expanded_columns=list(RAW_COLUMNS),
        prediction_start=20,
        prediction_length=10,
        nested_n=3,
        nested_m=0,            # uncapped: let Stage 1 reach B
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
        cores=2,
        lag_set=LAG_SET,
    )
    return module, graph, edges_df, call_log, pool_log


def test_lagged_picks_are_leaf_nodes(graph, edges_df):
    edges = {(row.driver, row.node, row.delay) for row in edges_df.itertuples()}

    for leaf, driven, delay in [('A::delay-1', 'T', -1),
                                ('A::delay-2', 'A', -2),
                                ('C::delay-1', 'B', -1)]:
        assert leaf in graph, f'{leaf} is not a graph node -- nodes={sorted(graph.nodes)}'
        attrs = graph.nodes[leaf]
        observable = leaf.split('::')[0]
        assert attrs.get('lagged_of') == observable and attrs.get('lag') == delay, (
            f'{leaf}: expected lagged_of={observable!r}/lag={delay}, got {attrs}')
        assert graph.in_degree(leaf) == 0, (
            f'{leaf} has in-edges, so it is not a leaf: '
            f'{list(graph.in_edges(leaf, data=True))}')
        assert (leaf, driven, delay) in edges, (
            f'missing {leaf} -> {driven} (delay {delay}) edge -- edges={edges}')

    # check 2, stated as its own assertion: the lagged edge into T comes from
    # the LEAF, and no delay!=0 edge ever leaves the raw observable's node.
    assert ('A', 'T', -1) not in edges, (
        f'lagged driver was attached to the raw observable node instead of a leaf -- '
        f'edges={edges}')
    print('lagged picks are leaf nodes (incl. upstream A::delay-1 -> T '
          'and self-lag A::delay-2 -> A): PASS')


def test_backbone_and_self_loop_guard(graph, edges_df):
    edges = {(row.driver, row.node, row.delay) for row in edges_df.itertuples()}
    assert ('A', 'T', 0) in edges and ('B', 'A', 0) in edges, (
        f'Stage 1 backbone T <- A <- B missing -- edges={edges}')
    assert not [e for e in edges if e[0] == e[1]], f'self-loop edge present -- edges={edges}'
    print('Stage 1 backbone intact, no delay=0 self-loops: PASS')


def test_upstream_excluded_from_candidates(pool_log):
    # Stage 1 for A runs when T <- A exists, so T is A's descendant.
    # Stage 1 for B runs when T <- A <- B exists, so A and T are B's.
    for node, must_exclude in [('A', {'T'}), ('B', {'A', 'T'})]:
        pool = pool_log[('stage1', node)]
        assert node not in pool, f"stage1 {node}: own column in its candidate pool ({pool})"
        leaked = must_exclude & set(pool)
        assert not leaked, (
            f'stage1 {node}: cycle-closing candidates {sorted(leaked)} were offered to MDE '
            f'-- pool={pool}')
    # Nothing is over-excluded: C is unrelated and must stay available.
    assert 'C' in pool_log[('stage1', 'B')], (
        f"stage1 B: unrelated observable C was excluded -- pool={pool_log[('stage1', 'B')]}")
    # Stage 2 pools are lagged-only.
    for (stage, node), pool in pool_log.items():
        if stage == 'stage2':
            assert all('::delay' in c for c in pool), (
                f'stage2 {node}: real observables in a Stage 2 candidate pool -- pool={pool}')
    print('upstream (descendant) observables excluded before MDE, Stage 2 pools '
          'lagged-only: PASS')


def test_embedding_spec_resolves_leaves(module, graph):
    spec = module.embedding_spec_for(graph, 'T')
    assert ('T', 0) in spec, f'T lost its own anchor coordinate -- spec={spec}'
    assert ('A', -1) in spec, (
        f"T's leaf in-edge did not resolve to the raw observable A at delay -1 -- spec={spec}")
    assert not any(name == 'A::delay-1' for name, _ in spec), (
        f'embedding spec names a leaf node instead of a data column -- spec={spec}')
    assert ('A', 0) in spec, f"T's real driver A is missing from its embedding -- spec={spec}"
    print('embedding spec resolves leaf in-edges to (observable, delay): PASS')


if __name__ == '__main__':
    module, graph, edges_df, call_log, pool_log = run_network()
    test_lagged_picks_are_leaf_nodes(graph, edges_df)
    test_backbone_and_self_loop_guard(graph, edges_df)
    test_upstream_excluded_from_candidates(pool_log)
    test_embedding_spec_resolves_leaves(module, graph)
    print('ALL PASS')
