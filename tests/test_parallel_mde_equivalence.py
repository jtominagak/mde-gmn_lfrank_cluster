'''Equivalence test for the no-kedm-optimization-parallelized-mde fork.

scripts/src/no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py moves
every MDE call out of the nested-MDE BFS into one up-front barrier-free
phase, then replays the BFS serially against the precomputed table. Its
whole contract is that this changes ONLY the schedule -- the network it
builds must be indistinguishable from scripts/src/tsls-mde-gmn.py's for the
same inputs.

This asserts exactly that, for both PARALLEL_MDE_SCOPE values:
  1. identical graph nodes,
  2. identical edges -- driver, node, delay, weight -- IN ORDER (order
     matters: _accept_mde_result_edges()'s cycle guard rejects against the
     graph as built so far, so a reordered build is a different build),
  3. identical nested_mde_results,
  4. the fork issues a superset of the parent's MDE calls (it computes the
     whole node universe), and every node the parent computed is in it,
  5. nodes the BFS never reaches are computed by the fork and still absent
     from the graph -- i.e. "pruning" is "never consulted", not "scored and
     dropped".

The scenario deliberately exercises the three things that could break under
reordering: an order-dependent cycle rejection (A and B select each other),
a NESTED_M budget cut (C, D reached as edges but never recursed into), and
Stage 2's own real-observable recursion (E).

No pytest in this project's venv (see docs/review-findings.md) -- run directly:
    python3 tests/test_parallel_mde_equivalence.py
'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import (  # noqa: E402
    FakeProcessPoolExecutor,
    load_pipeline_module,
    make_fake_run_mde_for_target_tsls,
)

PARENT_FILE = 'tsls-mde-gmn.py'
FORK_FILE = 'no-kedm-optimization-parallelized-mde--tsls-mde-gmn.py'

TARGETS = ['T']
RAW_COLUMNS = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
LAG_SET = [-1]
NESTED_M = 1

# Stage 1 (NESTED_M=1: T recurses into its drivers; those drivers do not
# recurse further):
#   T  -> A, B                  both enqueued at depth 1
#   A  -> B, C                  B->A and C->A edges; neither recursed (depth 1)
#   B  -> A, D                  A->B WOULD close the B->A cycle => rejected
#   C/D/E/F/G                   never reached by Stage 1's BFS
# Stage 2 (seeded with visited1 = {T, A, B}, real-observable picks recurse):
#   T  -> E (real)              E enqueued for its OWN Stage 2 round
#   A  -> B::delay-1            lagged leaf
#   B  -> (nothing)             driver-exhausted
#   E  -> F (real)              E is at depth 1 => budget-bound, not enqueued
PLAN = {
    ('stage1', 'T'): ['A', 'B'],
    ('stage1', 'A'): ['B', 'C'],
    ('stage1', 'B'): ['A', 'D'],
    ('stage1', 'C'): ['E'],          # never consulted -- C is never visited
    ('stage1', 'E'): ['F', 'G'],     # never consulted
    ('stage2', 'T'): ['E'],
    ('stage2', 'A'): ['B::delay-1'],
    ('stage2', 'B'): [],
    ('stage2', 'E'): ['F'],
    ('stage2', 'C'): ['G'],          # never consulted
}

UNIVERSE = TARGETS + RAW_COLUMNS
# What the parent actually computes -- the BFS-reachable subset.
PARENT_EXPECTED_CALLS = {
    ('stage1', 'T'), ('stage1', 'A'), ('stage1', 'B'),
    ('stage2', 'T'), ('stage2', 'A'), ('stage2', 'B'), ('stage2', 'E'),
}
# What the fork computes -- everything, both stages.
FORK_EXPECTED_CALLS = {(stage, node) for stage in ('stage1', 'stage2') for node in UNIVERSE}

COMMON_KWARGS = dict(
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
    lag_set=LAG_SET,
    stage2_allow_real_observables=True,
)


def _toy_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.normal(size=40) for c in TARGETS + RAW_COLUMNS})


def _run(module, extra_kwargs=None):
    call_log = []
    module.run_mde_for_target = make_fake_run_mde_for_target_tsls(PLAN, call_log)
    module.ProcessPoolExecutor = FakeProcessPoolExecutor

    graph, edges_df, nested_mde_results = module.build_nested_mde_network(
        targets=list(TARGETS),
        expanded_df=_toy_df(),
        raw_columns=list(RAW_COLUMNS),
        expanded_columns=list(RAW_COLUMNS),
        **COMMON_KWARGS,
        **(extra_kwargs or {}),
    )
    return graph, edges_df, nested_mde_results, call_log


def _edge_rows(edges_df):
    '''Edges as an ORDERED list -- see this module's docstring for why order
    is part of the contract, not an artifact.'''
    return [(r.driver, r.node, r.delay, round(float(r.weight), 12))
            for r in edges_df.itertuples()]


def _assert_same_network(name, ref, got):
    ref_graph, ref_edges, ref_results = ref
    got_graph, got_edges, got_results = got

    assert sorted(ref_graph.nodes) == sorted(got_graph.nodes), (
        f'{name}: node sets differ -- parent={sorted(ref_graph.nodes)} '
        f'fork={sorted(got_graph.nodes)}')
    assert _edge_rows(ref_edges) == _edge_rows(got_edges), (
        f'{name}: edge lists differ\n  parent={_edge_rows(ref_edges)}\n'
        f'  fork  ={_edge_rows(got_edges)}')
    assert sorted(ref_results) == sorted(got_results), (
        f'{name}: nested_mde_results keys differ -- parent={sorted(ref_results)} '
        f'fork={sorted(got_results)}')
    for node in ref_results:
        assert (ref_results[node]['SELECTED_COLUMNS']
                == got_results[node]['SELECTED_COLUMNS']), (
            f'{name}: nested_mde_results[{node!r}] differs')


def main():
    parent = load_pipeline_module(PARENT_FILE, '_test_parent_tsls_mde_gmn')
    fork = load_pipeline_module(FORK_FILE, '_test_fork_parallel_mde')

    ref_graph, ref_edges, ref_results, ref_calls = _run(parent)

    # ---- the parent really did run wave-by-wave, on the reachable subset --
    assert set(ref_calls) == PARENT_EXPECTED_CALLS, (
        f'parent call set changed -- expected {sorted(PARENT_EXPECTED_CALLS)}, '
        f'got {sorted(set(ref_calls))}. The scenario, not the fork, needs updating.')

    for scope in ('union', 'staged'):
        got_graph, got_edges, got_results, got_calls = _run(
            fork, {'parallel_mde_scope': scope})

        _assert_same_network(f'scope={scope}', (ref_graph, ref_edges, ref_results),
                             (got_graph, got_edges, got_results))

        # ---- (4) the fork computed the whole universe, both stages --------
        assert set(got_calls) == FORK_EXPECTED_CALLS, (
            f'scope={scope}: fork call set is not the full universe -- '
            f'missing={sorted(FORK_EXPECTED_CALLS - set(got_calls))} '
            f'extra={sorted(set(got_calls) - FORK_EXPECTED_CALLS)}')
        assert PARENT_EXPECTED_CALLS <= set(got_calls), (
            f'scope={scope}: fork skipped an MDE call the parent made')
        assert len(got_calls) == len(set(got_calls)), (
            f'scope={scope}: a node was computed twice -- call_log={got_calls}')

        # ---- (1) exactly one barrier: for scope='union', every MDE call is
        # issued before any BFS bookkeeping, so Stage 1 and Stage 2 calls are
        # free to interleave. The parent's R12 invariant (all stage1 before
        # all stage2) is deliberately NOT preserved here -- that ordering was
        # a consequence of the barriers this fork removes. It is preserved
        # for scope='staged', which keeps one barrier between the stages.
        stage1_pos = [i for i, (s, _) in enumerate(got_calls) if s == 'stage1']
        stage2_pos = [i for i, (s, _) in enumerate(got_calls) if s == 'stage2']
        if scope == 'staged':
            assert max(stage1_pos) < min(stage2_pos), (
                f'scope=staged: Stage 2 precompute started before Stage 1 finished')

        # ---- (5) computed-but-never-reached nodes stay out of the graph ---
        assert ('stage1', 'G') in set(got_calls), 'G should have been precomputed'
        assert 'G' not in set(got_graph.nodes), (
            f'scope={scope}: G was computed but never selected by anything, so it must '
            f'not be in the graph -- nodes={sorted(got_graph.nodes)}')

        print(f'scope={scope}: PASS -- identical network '
              f'({len(got_graph.nodes)} nodes, {len(_edge_rows(got_edges))} edges); '
              f'{len(got_calls)} MDE calls vs the parent\'s {len(ref_calls)}')

    # ---- the order-dependent cycle rejection really was exercised ---------
    edges = {(d, n, dl) for d, n, dl, _ in _edge_rows(ref_edges)}
    assert ('B', 'A', 0) in edges, f'expected B->A to win the cycle race -- edges={edges}'
    assert ('A', 'B', 0) not in edges, f'expected A->B to be cycle-rejected -- edges={edges}'
    assert 'G' not in set(ref_graph.nodes)

    # ---- _stage2_precompute_nodes() shrinks to exactly visited1 when Stage
    # 2 provably cannot recurse ---------------------------------------------
    universe = fork.mde_node_universe(TARGETS, RAW_COLUMNS)
    assert universe == UNIVERSE, f'universe={universe}'
    visited1 = {'T', 'A', 'B'}
    depths = {'T': 0, 'A': 1, 'B': 1}
    assert fork._stage2_precompute_nodes(universe, visited1, depths, NESTED_M, False) == \
        ['T', 'A', 'B'], 'no real-observable picks possible => exactly visited1'
    assert fork._stage2_precompute_nodes(universe, visited1, depths, 1, True) == universe, \
        'T is still under budget => recursion possible => whole universe'
    assert fork._stage2_precompute_nodes(universe, visited1, {'T': 1, 'A': 1, 'B': 1},
                                          1, True) == ['T', 'A', 'B'], \
        'every node already at the depth cap => recursion impossible => visited1'
    print('_stage2_precompute_nodes: PASS')

    print('ALL PASS')


if __name__ == '__main__':
    main()
