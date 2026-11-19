'''exp/spike-conversion-general: Forecast() row alignment for a node whose
embedding has a lagged in-edge.

_generalized_embed() drops a node's first `offset` rows (offset = its deepest
edge delay), so row i of the frame SMap sees is data row i + offset.
gmn.GMN.Forecast() takes DataOut's shared Time column from an offset-0
source node and lines every node's predictions up by position. Before the
fix, a configured pred range was used unchanged on every node's own frame,
so a node with offset k predicted rows k later than its Time label: its
Forecast() looked like the data shifted k rows early, and its one-step
score collapsed (job 4832849: r ~ 0 for every driven node, at delays
-40/-80). The fix moves the configured lib/pred range back by the node's
own offset.

Toy network: T <- 'A::delay-6' (a lagged leaf), T <- S (delay 0). T(t) is
A(t-7) plus noise, A a fast sine, so T(t+1) is the
embedded coordinate A(t-6). Real gmn, so this runs on the cluster:
    python3 tests/test_forecast_offset_alignment.py
'''
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402

SCRATCH = Path(__file__).resolve().parent / '_scratch_forecast_offset'
LAG = 6


def _run(module, results_dir):
    rng = np.random.default_rng(1)
    n = 320
    t = np.arange(n)
    a = np.sin(t / 2.5) + rng.normal(scale=0.01, size=n)
    s = np.cos(t / 9.0) + rng.normal(scale=0.01, size=n)
    # T(t+1) = A(t-6): exactly the lagged coordinate T embeds, so the
    # one-step map is informative (T's own present value is not enough)
    T = np.roll(a, LAG + 1) + 0.05 * s + rng.normal(scale=0.01, size=n)
    T[:LAG + 1] = T[LAG + 1]
    leaf = f'A::delay-{LAG}'
    df = pd.DataFrame({'Time': t, 'T': T, 'S': s,
                       leaf: pd.Series(a).shift(LAG).bfill().to_numpy()})
    data_file = results_dir / 'data_after_mde_toy_off.csv'
    df.to_csv(data_file, index=False)

    g = nx.MultiDiGraph()
    g.add_node('T')
    g.add_node('S')
    g.add_node(leaf, lagged_of='A', lag=-LAG)
    g.add_edge(leaf, 'T', delay=-LAG, weight=0.9)
    g.add_edge('S', 'T', delay=0, weight=0.1)
    module.partition_network(g, ['T'], 'real_data')
    net_file = results_dir / 'network_toy_off.pkl'
    with open(net_file, 'wb') as f:
        pickle.dump({'Graph': g, 'Map': {k: list(g.predecessors(k)) for k in g.nodes}}, f)
    net_result = {'targets': ['T'], 'network_id': 'toy_off',
                  'DATA_AFTER_MDE': data_file, 'NETWORK_FILE': net_file}
    start, length = 250, 50
    M = module.network_max_embedding_offset(g)
    assert M == LAG
    cfg = module.build_gmn_config(net_result, start, length, results_dir, mode='Forecast',
                                  theta=1, generate_length=length, generate_start=start,
                                  max_offset=M)
    Gf = module.gmn.GMN(configFile=str(cfg), cores=1)
    Gf.Forecast()
    return Gf, df


def test_lagged_node_forecast_is_labelled_at_its_own_rows():
    module = load_pipeline_module('tsls-mde-gmn.py', '_test_forecast_offset')
    d = SCRATCH
    d.mkdir(parents=True, exist_ok=True)
    Gf, df = _run(module, d)
    out = Gf.DataOut.dropna().reset_index(drop=True)
    rows = out['Time'].to_numpy().astype(int)
    pred = out['T'].to_numpy()
    real = df['T'].to_numpy()

    def r_at(shift):
        return float(np.corrcoef(pred, real[rows + shift])[0, 1])
    best = max(range(-2 * LAG, 2 * LAG + 1), key=r_at)
    print(f'r at label rows = {r_at(0):.4f}; r at rows+{LAG} = {r_at(LAG):.4f}; best shift = {best}')
    assert best == 0, f'T forecast best matches the data {best} rows away from its Time label'
    assert r_at(0) > 0.98, r_at(0)
    # the leaf passthrough is still the recording at its own label rows
    assert np.allclose(out[f'A::delay-{LAG}'].to_numpy(), df[f'A::delay-{LAG}'].to_numpy()[rows])


def test_pred_range_before_first_embedded_row_raises():
    from mde_gmn.gmn_embedding import _data_rows_to_embedded
    assert _data_rows_to_embedded('10 50', 0, 'n', clamp=False) == '10 50'
    assert _data_rows_to_embedded('10 50', 6, 'n', clamp=False) == '4 44'
    assert _data_rows_to_embedded('1 50', 6, 'n', clamp=True) == '1 44'
    try:
        _data_rows_to_embedded('3 50', 6, 'n', clamp=False)
    except ValueError:
        pass
    else:
        raise AssertionError('a pred range before the first embedded row must raise')


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'src'))
    test_pred_range_before_first_embedded_row_raises()
    print('PASS test_pred_range_before_first_embedded_row_raises')
    test_lagged_node_forecast_is_labelled_at_its_own_rows()
    print('PASS test_lagged_node_forecast_is_labelled_at_its_own_rows')
    print('ALL PASSED')
