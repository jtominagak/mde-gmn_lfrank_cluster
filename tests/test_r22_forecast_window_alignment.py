'''R22 acceptance test (docs/review-findings.md): the real-data external-
input leaf shortcut in _generalized_node_forecast() used to return
self.data in full (Network.py sets dataLib_i = range(self.data.shape[0])
in Forecast mode, so self.data is always the ENTIRE dataset there) --
thousands of rows longer than every SMap node's own short pred-range
Predictions Series. Since gmn.GMN.Forecast() assembles DataOut via plain
per-node column assignment and takes the shared 'Time' column only from
Network.TopologicalSorted[0] (a leaf, almost always first since it has no
in-edges), that full-length passthrough silently became DataOut's own row
index -- every SMap node's short Series then pandas-aligned onto only the
first few rows of it, mislabeling genuinely-near-generate_start
predictions as though they occurred at the very start of the recording.

This test builds a real (tiny) T <- A network and runs a genuine
gmn.GMN(...).Forecast() call -- the same way build_network()/
run_forecast_diagnostic() build one for real -- and checks the ACTUAL
DataOut this produces, not just that the leaf shortcut's own return value
looks right in isolation.

No pytest in this project's venv -- run directly:
    python3 tests/test_r22_forecast_window_alignment.py
'''
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402

SCRATCH = Path(__file__).resolve().parent / '_scratch_r22'


def _toy_graph():
    '''T (target, interior) <- A (delay 0) -- same toy shape as R14's own
    _toy_graph(): A is the external input (leaf) under test.'''
    import networkx as nx
    graph = nx.MultiDiGraph()
    graph.add_node('T')
    graph.add_node('A')
    graph.add_edge('A', 'T', delay=0, weight=0.5)
    return graph


def _run_forecast(module, results_dir):
    rng = np.random.default_rng(0)
    n_rows = 200
    t = np.arange(n_rows)
    a_series = np.sin(t / 5.0) + rng.normal(scale=0.01, size=n_rows)
    t_series = 0.5 * np.roll(a_series, 1) + rng.normal(scale=0.01, size=n_rows)
    data_after_mde = pd.DataFrame({'Time': t, 'T': t_series, 'A': a_series})

    data_file = results_dir / 'data_after_mde_toy_r22.csv'
    data_after_mde.to_csv(data_file, index=False)

    graph = _toy_graph()
    module.partition_network(graph, ['T'], 'real_data')
    network_file = results_dir / 'network_toy_r22.pkl'
    with open(network_file, 'wb') as f:
        pickle.dump({'Graph': graph, 'Map': {n: list(graph.predecessors(n)) for n in graph.nodes}}, f)

    net_result = {
        'targets': ['T'],
        'network_id': 'toy_r22',
        'DATA_AFTER_MDE': data_file,
        'NETWORK_FILE': network_file,
    }

    generate_start = 150
    generate_length = 10
    max_offset = module.network_max_embedding_offset(graph)
    config_file = module.build_gmn_config(
        net_result, generate_start, generate_length, results_dir,
        mode='Forecast', theta=3, generate_length=generate_length,
        generate_start=generate_start, max_offset=max_offset,
    )
    Gf = module.gmn.GMN(configFile=str(config_file), cores=1)
    Gf.Forecast()
    return Gf, data_after_mde, generate_start, generate_length, max_offset


def _check_forecast_window_alignment(module, name, results_dir):
    Gf, data_after_mde, generate_start, generate_length, max_offset = _run_forecast(module, results_dir)
    n_rows = len(data_after_mde)

    # The pre-fix bug: DataOut's own row count silently became n_rows (200,
    # the leaf's full untouched dataset), not a short pred-range window.
    assert len(Gf.DataOut) < n_rows - 50, (
        f'{name}: Gf.DataOut has {len(Gf.DataOut)} rows out of {n_rows} total -- '
        f'looks like the leaf passthrough is still returning the full dataset, '
        f'not a pred-range slice (expected a short window near generate_start={generate_start})'
    )

    # Every surviving Time value should sit near generate_start, not near 0
    # (the pre-fix symptom: the leaf's full-dataset Time column mislabeled
    # every other node's short prediction window as rows 0..N).
    time_vals = Gf.DataOut['Time'].dropna().to_numpy()
    assert len(time_vals) > 0, f'{name}: Gf.DataOut has no Time values at all'
    assert time_vals.min() >= generate_start - 2 * generate_length - max_offset - 2, (
        f'{name}: Gf.DataOut Time values start at {time_vals.min()}, far earlier than '
        f'generate_start={generate_start} -- looks like the near-zero mislabeling bug'
    )
    assert time_vals.max() <= generate_start + generate_length + 2, (
        f'{name}: Gf.DataOut Time values reach {time_vals.max()}, past the intended '
        f'generate_start={generate_start}+generate_length={generate_length} window'
    )

    # The leaf's own column ('A') should be real, unmodified data from the
    # CORRECT absolute rows (indexed by the now-correct Time column), not
    # from the start of the recording.
    dataout = Gf.DataOut.dropna().reset_index(drop=True)
    time_idx = dataout['Time'].to_numpy().astype(int)
    real_a_at_those_rows = data_after_mde['A'].to_numpy()[time_idx]
    assert np.allclose(dataout['A'].to_numpy(), real_a_at_those_rows), (
        f'{name}: leaf column A does not match the real data at the rows its own '
        f'(now-correct) Time column claims'
    )

    print(f'{name}: R22 forecast-window-alignment PASS '
          f'(DataOut rows={len(Gf.DataOut)}/{n_rows}, '
          f'Time range=[{time_vals.min()}, {time_vals.max()}], generate_start={generate_start})')


def _run_for_module(filename, module_name, name):
    module = load_pipeline_module(filename, module_name)
    results_dir = SCRATCH / module_name
    results_dir.mkdir(parents=True, exist_ok=True)
    _check_forecast_window_alignment(module, name, results_dir)


def test_tsls():
    _run_for_module('tsls-mde-gmn.py', '_test_tsls_mde_gmn_r22', 'tsls-mde-gmn.py')


if __name__ == '__main__':
    SCRATCH.mkdir(parents=True, exist_ok=True)
    test_tsls()
    print('ALL PASS')
