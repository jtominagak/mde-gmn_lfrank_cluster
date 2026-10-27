'''Contract test for the no-kedm-optimization-parallelized-mde fork.

The fork moves every MDE call out of the nested-MDE BFS into an up-front
barrier-free phase, then replays the BFS against the precomputed table. Its
two scopes now promise DIFFERENT things (2026-09-30):

  PARALLEL_MDE_SCOPE='staged'  the parent pipeline's algorithm exactly --
      Stage 1 over real observables (recursive), then Stage 2 attaching
      lagged leaves -- minus the per-wave barriers. Must build the network
      scripts/src/tsls-mde-gmn.py builds: same nodes, same edges IN ORDER
      (order matters -- when two nodes select each other, whichever the loop
      reaches first keeps its edge), same nested_mde_results.

  PARALLEL_MDE_SCOPE='union'   ONE MDE per node over real AND lagged
      candidates together. A different algorithm on purpose, so this is
      checked against the CONSTRAINTS instead: acyclic, lagged picks are
      leaves, one call per node, nothing unreached in the graph.

Both scopes must also honour the acyclicity exclusion. Phase 0 cannot know
which observables will end up upstream, so when a precomputed result selects
one that turns out to be cycle-closing, that node has to be RECOMPUTED with
the offending candidates removed -- asserted here, not assumed.

The fake MDE is POOL-AWARE: it only ever "selects" columns that were actually
offered in the candidate pool. That is what makes exclusion observable at
all -- with a pool-blind fake, a recompute would return the same forbidden
pick and the test would prove nothing.

No pytest in this project's venv (see docs/review-findings.md) -- run it
directly:
    python3 clean_scripts/tests/test_parallel_mde_equivalence.py
'''
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import (  # noqa: E402
    install_fake_mde,
    load_pipeline_module,
)
from types import SimpleNamespace  # noqa: E402

# Loaded only to set up sys.path and the gmn patches; the builders themselves
# come from mde_gmn.network / mde_gmn.network_parallel (see main()).
PARENT_FILE = 'run_serial_mde.py'

TARGETS = ['T']
RAW_COLUMNS = ['A', 'B', 'C', 'D', 'E']
LAG_SET = [-1]
NESTED_M = 0          # uncapped, so the backbone is driver-limited, not depth-limited

# What each node would pick if everything were on offer. Real picks first,
# then lagged, so selection order is deterministic.
#   T <- A, B            (backbone roots)
#   A <- B, C
#   B <- A, D            A is by then BETWEEN B and T, so A is cycle-closing
#                        for B and must be excluded -> B keeps only D
#   E                    never reached by anything
PLAN_REAL = {'T': ['A', 'B'], 'A': ['B', 'C'], 'B': ['A', 'D'],
             'C': [], 'D': [], 'E': ['A']}
PLAN_LAGGED = {'T': ['A::delay-1'], 'A': ['A::delay-1'], 'B': ['C::delay-1'],
               'C': [], 'D': [], 'E': []}
UNIVERSE = TARGETS + RAW_COLUMNS

COMMON_KWARGS = dict(
    prediction_start=10,
    prediction_length=5,
    nested_n=3,
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
)


def _toy_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.normal(size=40) for c in TARGETS + RAW_COLUMNS})


def _pool_aware_fake(call_log):
    '''run_mde_for_target() stand-in. Picks come from PLAN_REAL/PLAN_LAGGED
    depending on the stage, filtered to what the candidate pool actually
    offered -- so excluding a candidate genuinely prevents its selection.'''
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


def _run(module, extra_kwargs=None):
    call_log = []
    # install_fake_mde() follows the builder to wherever it is defined: the
    # parent's now lives in mde_gmn/network.py and calls MDE through the mde
    # module, the fork's is still in its own entry script.
    install_fake_mde(module, _pool_aware_fake(call_log))
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
    '''Edges as an ORDERED list -- order is part of the contract.'''
    return [(r.driver, r.node, r.delay, round(float(r.weight), 12))
            for r in edges_df.itertuples()]


def _assert_constraints(name, graph, edges_df):
    assert nx.is_directed_acyclic_graph(graph), f'{name}: network is not acyclic'
    for node, attrs in graph.nodes(data=True):
        if 'lagged_of' in attrs:
            assert graph.in_degree(node) == 0, (
                f'{name}: lagged node {node} has drivers, so it is not a leaf')
    for driver, target_node, delay, _weight in _edge_rows(edges_df):
        if delay != 0:
            assert 'lagged_of' in graph.nodes[driver], (
                f'{name}: delay={delay} edge {driver}->{target_node} does not come from a '
                f'lagged leaf')
    assert 'E' not in graph.nodes, (
        f'{name}: E was never selected by anything, so it must not be in the graph')


def test_staged_matches_the_parent(parent, fork):
    ref_graph, ref_edges, ref_results, ref_calls = _run(parent)
    got_graph, got_edges, got_results, got_calls = _run(fork, {'parallel_mde_scope': 'staged'})

    assert sorted(ref_graph.nodes) == sorted(got_graph.nodes), (
        f'node sets differ -- parent={sorted(ref_graph.nodes)} fork={sorted(got_graph.nodes)}')
    assert _edge_rows(ref_edges) == _edge_rows(got_edges), (
        f'edge lists differ\n  parent={_edge_rows(ref_edges)}\n  fork  ={_edge_rows(got_edges)}')
    assert sorted(ref_results) == sorted(got_results), (
        f'nested_mde_results keys differ -- parent={sorted(ref_results)} '
        f'fork={sorted(got_results)}')
    for node in ref_results:
        assert (ref_results[node]['SELECTED_COLUMNS']
                == got_results[node]['SELECTED_COLUMNS']), f'results[{node!r}] differs'

    _assert_constraints('parent', ref_graph, ref_edges)
    _assert_constraints('scope=staged', got_graph, got_edges)

    # The fork computed the whole universe for Stage 1; the parent only the
    # nodes its waves reached.
    assert {n for stage, n in got_calls if stage == 'stage1'} >= set(UNIVERSE) - set(TARGETS), (
        f'fork did not precompute Stage 1 for the whole universe -- calls={got_calls}')
    assert set(ref_calls) <= set(got_calls), 'fork skipped an MDE call the parent made'
    # Stage 2 is precomputed only for Stage 1's visited set -- no overshoot.
    stage2_nodes = {n for stage, n in got_calls if stage == 'stage2'}
    assert stage2_nodes == set(ref_graph.nodes) - {n for n, a in ref_graph.nodes(data=True)
                                                    if 'lagged_of' in a}, (
        f'Stage 2 node set should be exactly Stage 1 visited -- got {sorted(stage2_nodes)}')
    print(f'scope=staged: PASS -- identical network to the parent '
          f'({len(got_graph.nodes)} nodes, {len(_edge_rows(got_edges))} edges); '
          f'{len(got_calls)} MDE calls vs the parent\'s {len(ref_calls)}')
    return ref_graph, ref_edges, got_calls


def test_union_is_one_combined_call_per_node(fork):
    graph, edges_df, _results, calls = _run(fork, {'parallel_mde_scope': 'union'})
    stages = {stage for stage, _ in calls}
    assert stages == {'union'}, f'union scope must issue only union calls -- got {stages}'
    for node in UNIVERSE:
        assert ('union', node) in calls, f'union: no MDE call for {node}'
    _assert_constraints('scope=union', graph, edges_df)
    # Real and lagged candidates really did compete in one call: T ends up
    # with both kinds of driver from a single MDE result.
    t_edges = {(d, dl) for d, n, dl, _ in _edge_rows(edges_df) if n == 'T'}
    assert ('A', 0) in t_edges and ('A::delay-1', -1) in t_edges, (
        f'union: expected T to take both a real and a lagged driver -- {t_edges}')
    print(f'scope=union: PASS -- one combined MDE per node, constraints hold '
          f'({len(graph.nodes)} nodes, {len(_edge_rows(edges_df))} edges)')


def test_cycle_closing_pick_forces_a_recompute(fork, staged_calls):
    # B's precomputed pick list is ['A', 'D'], but by the time the replay
    # reaches B, A sits between B and T -- so B must be recomputed with A
    # excluded, and must end up with D only.
    assert staged_calls.count(('stage1', 'B')) == 2, (
        f"B should be computed once in phase 0 and recomputed once during the replay -- "
        f"calls for B: {staged_calls.count(('stage1', 'B'))}")
    graph, edges_df, _results, _calls = _run(fork, {'parallel_mde_scope': 'staged'})
    edges = {(d, n, dl) for d, n, dl, _ in _edge_rows(edges_df)}
    assert ('D', 'B', 0) in edges, f'B lost its legal driver D -- edges={edges}'
    assert ('A', 'B', 0) not in edges, f'cycle-closing A->B was kept -- edges={edges}'
    assert nx.is_directed_acyclic_graph(graph)
    print('recompute on cycle-closing precomputed pick: PASS')


def main():
    # Both pipelines are the same package now (Phase 2); the entry scripts
    # differ only in which builder they pass. So load one entry script -- that
    # puts scripts/src on sys.path and applies the gmn patches -- and then
    # compare the two builder MODULES directly, which is what "the parent" and
    # "the fork" actually mean.
    load_pipeline_module(PARENT_FILE, '_test_parent_tsls_mde_gmn')
    from mde_gmn import network as parent           # noqa: E402 -- needs the sys.path above
    from mde_gmn import network_parallel as fork    # noqa: E402

    _ref_graph, _ref_edges, staged_calls = test_staged_matches_the_parent(parent, fork)
    test_union_is_one_combined_call_per_node(fork)
    test_cycle_closing_pick_forces_a_recompute(fork, staged_calls)
    print('ALL PASS')


if __name__ == '__main__':
    main()
