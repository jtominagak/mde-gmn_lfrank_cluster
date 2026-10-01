'''Regression test: the exported data frame must carry one column per lagged leaf.

THE BUG THIS PINS. A lagged pick becomes a leaf node in its own right
('A::delay-3', attributes lagged_of='A', lag=-3). partition_network() classifies
leaves as EXTERNAL INPUTS, and gmn feeds an external input from the exported
frame BY NODE NAME -- so if the frame has no 'A::delay-3' column, the node has
nothing to read.

build_network() therefore writes one column per leaf, holding that observable
shifted by its lag. That was added to the serial pipeline on 2026-09-30
(fa804e8) and NOT to the parallel-MDE fork, which kept its own copy of
build_network(). The fork consequently built networks with lagged leaves and
handed gmn a frame missing their columns -- broken for any network containing a
lagged pick, which is nearly all of them, and invisible until GMN ran.

Nothing asserted the invariant, which is why the two copies could drift. Now
something does. Both pipelines share one build_network(), so this test covers
both.

No pytest in this venv -- run it directly:
    python3 tests/test_lagged_leaf_data_export.py
'''
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402

TARGETS = ['T']
RAW_COLUMNS = ['A', 'B']
ROWS = 60

# The network the fake builder hands back: T is driven by the real observable A
# and by two lagged leaves -- one of A (an observable that IS in the backbone)
# and one of B (an observable that is NOT). Both cases have to appear in the
# frame, since both are external inputs gmn will look up by name.
LEAVES = {'A::delay-3': ('A', -3), 'B::delay-1': ('B', -1)}


def _toy_df():
    rng = np.random.default_rng(0)
    data = {c: rng.normal(size=ROWS) for c in TARGETS + RAW_COLUMNS}
    return pd.DataFrame(data)


def _fake_builder(*args, **kwargs):
    '''Stands in for build_nested_mde_network(). Returns a graph with lagged
    leaves already attached, so this test exercises what build_network() does
    with such a graph rather than how the graph came to exist.'''
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(TARGETS)
    graph.add_node('A')
    graph.add_edge('A', 'T', delay=0, weight=0.5)
    for leaf, (observable, lag) in LEAVES.items():
        graph.add_node(leaf, lagged_of=observable, lag=lag)
        graph.add_edge(leaf, 'T', delay=lag, weight=0.4)
    return graph, pd.DataFrame(), {}


def _build(module, results_dir):
    df = _toy_df()
    return module.build_network(
        list(TARGETS), {t: {'SELECTED_COLUMNS': ['A']} for t in TARGETS}, df,
        prediction_start=40, prediction_length=10,
        mode='MDE-GMN', e_range=range(3, 5), tau_range=range(-1, 0),
        results_dir=results_dir, cores=1,
        metric='correlation', target_metric='correlation', gamma_delta=2,
        ccm_window_rows=40, ccm_lib_size_pcts=(20, 50), ccm_sample=2, ccm_slope=0.01,
        observable_columns=list(RAW_COLUMNS),
        nested_n=2, nested_m=1, embed_dim_rho_min=0.2, cross_map_rho_min=0.2,
        exclusion_radius=0, lag_set=[-1, -3], tp=1,
        network_builder=_fake_builder,
    ), df


def main():
    module = load_pipeline_module('tsls-mde-gmn.py', '_test_tsls_leaf_export')
    results_dir = Path(__file__).resolve().parent / '_scratch_leaf_export'
    results_dir.mkdir(exist_ok=True)

    net_result, df = _build(module, results_dir)
    exported = pd.read_csv(net_result['DATA_AFTER_MDE'])
    graph = net_result['network']['Graph']

    # 1. every lagged leaf in the graph has a column in the frame
    for leaf in LEAVES:
        assert leaf in exported.columns, (
            f'lagged leaf {leaf} has no column in DATA_AFTER_MDE -- gmn feeds external '
            f'inputs by node name, so this node would have nothing to read. '
            f'Columns present: {list(exported.columns)}')

    # 2. more generally: EVERY node gmn will look up must be a column
    for node in graph.nodes:
        assert node in exported.columns, f'graph node {node} missing from DATA_AFTER_MDE'

    # 3. the leaf column actually holds the shifted observable, not a copy
    for leaf, (observable, lag) in LEAVES.items():
        expected = df[observable].shift(-lag).reset_index(drop=True)
        got = exported[leaf]
        # Compare only where the shift leaves a real value; the warm-up rows at
        # the tail are NaN by construction.
        valid = expected.notna() & got.notna()
        assert valid.sum() > 0, f'{leaf}: no overlapping non-NaN rows to compare'
        assert np.allclose(got[valid], expected[valid]), (
            f'{leaf} is not {observable} shifted by {lag} -- a wrong shift would feed gmn '
            f'the right column name with the wrong history')
        assert not np.allclose(got[valid], df[observable][valid]), (
            f'{leaf} is identical to unshifted {observable}, so the lag was not applied')

    # 4. the leaves really are external inputs, which is what makes 1-3 matter
    external = [n for n, a in graph.nodes(data=True) if 'lagged_of' in a]
    assert sorted(external) == sorted(LEAVES), f'unexpected leaf set: {external}'
    for leaf in external:
        assert graph.in_degree(leaf) == 0, f'{leaf} has drivers, so it is not a leaf'

    print(f'lagged leaf columns present and correctly shifted '
          f'({", ".join(sorted(LEAVES))}): PASS')
    print('ALL PASS')


if __name__ == '__main__':
    main()
