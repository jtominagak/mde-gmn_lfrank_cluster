'''R11 acceptance test (docs/review-actions-v1.md): rebuild_at_best_params()
must write the same per-node score data it already prints to a CSV in
results_dir (purely additive -- one row per node, tagged TARGET/driver,
each row labeled with its own metric, per-node scores not summed).

Uses a real (tiny, synthetic) gmn.GMN.Generate() run, built the same way
the pipeline builds one for real (same pattern as
tests/test_r14_external_inputs.py).

No pytest in this project's venv -- run directly:
    python3 clean_scripts/tests/test_r11_per_node_table.py
'''
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402

SCRATCH = Path(__file__).resolve().parent / '_scratch_r11'


def _toy_graph():
    '''T (target, interior) <- A (delay 0, external input, real_data mode).'''
    graph = nx.MultiDiGraph()
    graph.add_node('T')
    graph.add_node('A')
    graph.add_edge('A', 'T', delay=0, weight=0.5)
    return graph


def _run_for_module(filename, module_name, name, extra_kwargs):
    module = load_pipeline_module(filename, module_name)
    results_dir = SCRATCH / module_name
    results_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(4)
    n_rows = 200
    t = np.arange(n_rows)
    a_series = np.sin(t / 5.0) + rng.normal(scale=0.01, size=n_rows)
    t_series = 0.5 * a_series + rng.normal(scale=0.01, size=n_rows)
    data_after_mde = pd.DataFrame({'Time': t, 'T': t_series, 'A': a_series})
    data_file = results_dir / 'data_after_mde_toy.csv'
    data_after_mde.to_csv(data_file, index=False)

    graph = _toy_graph()
    module.partition_network(graph, ['T'], 'real_data')
    network_file = results_dir / 'network_toy.pkl'
    with open(network_file, 'wb') as f:
        pickle.dump({'Graph': graph, 'Map': {n: list(graph.predecessors(n)) for n in graph.nodes}}, f)

    network_id = 'toy_r11'
    net_result = {
        'targets': ['T'],
        'network_id': network_id,
        'DATA_AFTER_MDE': data_file,
        'NETWORK_FILE': network_file,
    }
    gmn_runs = {network_id: {'net_result': net_result}}
    theta_search_results = {network_id: {'best_node_theta': {}}}

    generate_start = 150
    generate_length = 10
    module.rebuild_at_best_params(
        gmn_runs, theta_search_results, generate_start, generate_length,
        results_dir, 1, generate_length=generate_length,
        generate_starts={network_id: generate_start},
        generate_validation_start=generate_start,
        **extra_kwargs,
    )

    csv_path = results_dir / f'per_node_scores_{network_id}.csv'
    assert csv_path.exists(), f'{name}: rebuild_at_best_params() did not write {csv_path}'
    table = pd.read_csv(csv_path)
    assert set(table.columns) == {'node', 'role', 'metric', 'score'}, (
        f'{name}: unexpected columns -- {table.columns.tolist()}'
    )
    assert set(table['node']) == {'T', 'A'}, f'{name}: expected rows for both T and A -- {table}'
    t_row = table[table['node'] == 'T'].iloc[0]
    a_row = table[table['node'] == 'A'].iloc[0]
    assert t_row['role'] == 'TARGET', f'{name}: T should be tagged TARGET -- {t_row}'
    assert a_row['role'] == 'driver', f'{name}: A should be tagged driver -- {a_row}'
    assert np.isfinite(t_row['score']) and np.isfinite(a_row['score']), (
        f'{name}: scores should be finite -- {table}'
    )
    print(f'{name}: PASS\n{table}')


def test_tsls():
    _run_for_module(
        'run_serial_mde.py', '_test_tsls_mde_gmn_r11', 'run_serial_mde.py',
        {'metric': 'f1', 'target_metric': 'correlation', 'gamma_delta': 2},
    )


# The 1 Hz sibling pipeline is not part of clean_scripts, so this file no
# longer has a second half exercising it -- see clean_scripts/README.md on
# scope.

if __name__ == '__main__':
    test_tsls()
    print('ALL PASS')
