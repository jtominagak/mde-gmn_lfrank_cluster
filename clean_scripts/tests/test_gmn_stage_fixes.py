'''GMN-stage fixes of 2026-11-18, on a real (tiny) gmn.GMN.Generate() run.

What this pins:
  1. NARROW FRAMES. After Generate(), every node's own frame holds only its
     own columns (time, predecessors, itself). gmn appends the whole network
     row each step; that made every frame network-wide, which is memory and
     copying in nodes^2 and what ran job 4799407 out of memory.
  2. NO REPEATED Generate(). The start sweep and the final rebuild share one
     memo: the rebuild IS the sweep's winning free run.
  3. STOCHASTIC RUN. GENERATE_NOISE='local_residual' is reproducible for a
     seed, differs between seeds, and on a toy TSLS network driven by a
     spiking input it produces resets where the deterministic run's
     conditional mean does not.
  4. PARALLEL THETA. retune_theta_for_all_nodes() with a process pool returns
     exactly what the serial loop returns.

Builds the network the way tests/test_r11_per_node_table.py does.

No pytest in this project's venv -- run it directly:
    python3 tests/test_gmn_stage_fixes.py
'''
import pickle
import shutil
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402

SCRATCH = Path(__file__).resolve().parent / '_scratch_gmn_stage_fixes'
DT = 0.025
GEN_START, GEN_LEN, N_ROWS = 500, 120, 700


def _tsls(spikes):
    out, last = np.zeros(len(spikes)), None
    for i, s in enumerate(spikes):
        if s:
            last = i
        out[i] = (i - last) * DT if last is not None else i * DT
    return out


def _setup(module, tag):
    '''A (spiking input, external, real data) -> T (fires one row after A
    with probability 0.6, plus a low spontaneous rate).'''
    results_dir = SCRATCH / tag
    shutil.rmtree(results_dir, ignore_errors=True)
    results_dir.mkdir(parents=True)
    rng = np.random.default_rng(11)
    a_spikes = rng.random(N_ROWS) < 0.06
    t_spikes = np.zeros(N_ROWS, bool)
    t_spikes[1:] = a_spikes[:-1] & (rng.random(N_ROWS - 1) < 0.6)
    t_spikes |= rng.random(N_ROWS) < 0.01
    data = pd.DataFrame({'Time': np.arange(N_ROWS), 'T': _tsls(t_spikes), 'A': _tsls(a_spikes)})
    data_file = results_dir / 'data_after_mde_toy.csv'
    data.to_csv(data_file, index=False)

    graph = nx.MultiDiGraph()
    graph.add_node('T')
    graph.add_node('A')
    graph.add_edge('A', 'T', delay=0, weight=0.5)
    module.partition_network(graph, ['T'], 'real_data')
    network_file = results_dir / 'network_toy.pkl'
    with open(network_file, 'wb') as f:
        pickle.dump({'Graph': graph, 'Map': {n: list(graph.predecessors(n)) for n in graph.nodes}}, f)
    # 'T' is the behavioural target here; score its free run with f1 below by
    # treating it as a neuron (targets=['A'] would make A unpartitionable).
    net_result = {'targets': ['T'], 'network_id': f'toy_{tag}',
                  'DATA_AFTER_MDE': data_file, 'NETWORK_FILE': network_file}
    return results_dir, net_result, data


def _count_generate(gmn):
    original = gmn.GMN.Generate
    calls = []

    def counted(self):
        calls.append(1)
        return original(self)
    gmn.GMN.Generate = counted
    return calls, lambda: setattr(gmn.GMN, 'Generate', original)


def test_narrow_frames_and_memo(module):
    import gmn
    results_dir, net_result, _ = _setup(module, 'memo')
    nid = net_result['network_id']
    gmn_runs = {nid: {'net_result': net_result}}
    theta = {nid: {'best_node_theta': {'T': 1.0}}}
    calls, restore = _count_generate(gmn)
    try:
        starts = module.sweep_generate_start(
            gmn_runs, theta, GEN_START, GEN_LEN, results_dir, 1, GEN_LEN, [0, 1, 2],
            'correlation', 2, generate_validation_start=GEN_START)
        module.rebuild_at_best_params(
            gmn_runs, theta, GEN_START, GEN_LEN, results_dir, 1, 'f1', 'correlation', 2,
            generate_length=GEN_LEN, generate_starts=starts, generate_validation_start=GEN_START)
    finally:
        restore()
    assert len(calls) == 3, (
        f'{len(calls)} Generate() calls for 3 distinct start rows -- the rebuild repeats the '
        f'sweep\'s winner and must come from the memo')
    assert (results_dir / f'start_sweep_summary_{nid}.csv').exists()

    G = gmn_runs[nid]['G']
    graph = G.Network.Graph
    for name in graph.nodes:
        node = graph.nodes[name]['Node']
        allowed = {G.Network.timeColumnName, name, *graph.predecessors(name)}
        extra = set(node.data.columns) - allowed
        assert not extra, f'node {name} frame grew network-wide columns {extra}'
    print('PASS: 3 Generate() calls for 3 start rows (memo shared), node frames stay narrow')
    return gmn_runs, theta, results_dir, net_result


def test_stochastic(module, gmn_runs, results_dir, net_result):
    nid = net_result['network_id']

    def run(seed):
        module.run_stochastic_generate(
            gmn_runs, GEN_START, GEN_LEN, results_dir, 1, 'f1', 'f1', 2, GEN_LEN, 0,
            'local_residual', seed)
        out = pd.read_csv(results_dir / f'gmn_generate_output_{nid}_stochastic.csv')
        table = pd.read_csv(results_dir / f'generate_noise_comparison_{nid}.csv')
        return out, table

    out1, table = run(1)
    out1b, _ = run(1)
    out2, _ = run(2)
    assert out1['T'].equals(out1b['T']), 'same seed must reproduce the same stochastic run'
    assert not out1['T'].equals(out2['T']), 'different seeds must give different runs'
    det = gmn_runs[nid]['G'].DataOut.reset_index(drop=True)['T'].to_numpy()
    assert not np.allclose(out1['T'].to_numpy(), det), 'stochastic run equals deterministic'
    assert out1['A'].equals(pd.Series(gmn_runs[nid]['G'].DataOut['A'].to_numpy(), name='A')), (
        'an external input is replayed from data, so noise must not touch it')

    row = table.set_index('node').loc['T']
    real, d_ev, s_ev = row['events_real'], row['events_deterministic'], row['events_stochastic']
    assert abs(s_ev - real) < abs(d_ev - real), (
        f'stochastic run should reproduce the reset count better: real={real} '
        f'deterministic={d_ev} stochastic={s_ev}')
    assert row['sd_stochastic'] > row['sd_deterministic'] or abs(s_ev - real) < abs(d_ev - real)
    print(f'PASS: stochastic run reproducible per seed; resets real={real} '
          f'deterministic={d_ev} stochastic={s_ev}; sd real={row["sd_real"]:.3f} '
          f'deterministic={row["sd_deterministic"]:.3f} stochastic={row["sd_stochastic"]:.3f}')


def test_parallel_theta(module):
    from mde_gmn.theta import theta_sweep_workers
    _, _, data = _setup(module, 'theta')
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(['T', 'A', 'B'])
    graph.add_edge('A', 'T', delay=0, weight=0.5)
    graph.add_edge('T', 'B', delay=0, weight=0.5)
    data = data.assign(B=np.roll(data['T'].to_numpy(), 1))
    thetas = [0.1, 1, 3, 6]
    kwargs = dict(targets=['T'], theta_range=thetas, metric='f1',
                  target_metric='correlation', gamma_delta=2)
    serial = module.retune_theta_for_all_nodes(data, graph, ['T', 'B'], cores=1, **kwargs)
    parallel = module.retune_theta_for_all_nodes(data, graph, ['T', 'B'], cores=2, **kwargs)
    assert serial == parallel, f'parallel {parallel} != serial {serial}'
    assert list(parallel) == ['T', 'B'], 'node order must be the caller\'s'
    # memory bound: ~33k rows is ~9 GB per sweep, so 240 GB allows far fewer than 18
    assert theta_sweep_workers(18, 131, 33000, memory_bytes=240 * 2**30) < 18
    assert theta_sweep_workers(18, 131, 2000, memory_bytes=240 * 2**30) == 18
    assert theta_sweep_workers(18, 3, 2000, memory_bytes=240 * 2**30) == 3
    assert theta_sweep_workers(4, 10, 10**6, memory_bytes=2**30) == 1
    print(f'PASS: parallel theta search == serial ({parallel}); worker count memory-bounded')


def test_generate_tp_greater_than_one(module):
    '''GMN_TP > 1: a Tp-step model must generate on the right time axis.

    A (external, real data) is a sine; T is A delayed by 5 rows, so from
    (T(t), A(t)) the value T(t+Tp) is determined for any Tp. A correct free
    run reproduces T with no time shift. gmn's own scheme predicted T(t+Tp)
    from the LAST row and appended it as the next row, which would show up
    here as a run shifted Tp-1 rows ahead of the recording.'''
    tag = 'tp'
    results_dir = SCRATCH / tag
    shutil.rmtree(results_dir, ignore_errors=True)
    results_dir.mkdir(parents=True)
    t = np.arange(N_ROWS)
    a = np.sin(0.07 * t)
    data = pd.DataFrame({'Time': t, 'T': np.sin(0.07 * (t - 5)), 'A': a})
    data_file = results_dir / 'data_after_mde_toy.csv'
    data.to_csv(data_file, index=False)
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(['T', 'A'])
    graph.add_edge('A', 'T', delay=0, weight=0.5)
    module.partition_network(graph, ['T'], 'real_data')
    network_file = results_dir / 'network_toy.pkl'
    with open(network_file, 'wb') as f:
        pickle.dump({'Graph': graph, 'Map': {n: list(graph.predecessors(n)) for n in graph.nodes}}, f)
    real = data['T'].to_numpy()[GEN_START:GEN_START + GEN_LEN]
    for tp in (1, 3):
        nid = f'toy_tp{tp}'
        net_result = {'targets': ['T'], 'network_id': nid, 'DATA_AFTER_MDE': data_file,
                      'NETWORK_FILE': network_file, 'gmn_tp': tp}
        runs = {nid: {'net_result': net_result}}
        module.rebuild_at_best_params(
            runs, {nid: {'best_node_theta': {'T': 2.0}}}, GEN_START, GEN_LEN, results_dir, 1,
            'correlation', 'correlation', 2, generate_length=GEN_LEN,
            generate_starts={nid: GEN_START}, generate_validation_start=GEN_START)
        gen = runs[nid]['G'].DataOut.reset_index(drop=True)['T'].to_numpy()
        err = np.max(np.abs(gen - real))
        shifted = np.max(np.abs(gen[:-2] - real[2:])) if tp > 1 else np.inf
        assert err < 0.05, f'Tp={tp}: free run drifts from the recording (max err {err:.3f})'
        assert err < shifted, f'Tp={tp}: run matches a time-shifted recording better'
        print(f'PASS: Tp={tp} free run tracks the recording, max error {err:.4f}')


if __name__ == '__main__':
    module = load_pipeline_module('run_serial_mde.py', '_test_gmn_stage_fixes')
    gmn_runs, theta, results_dir, net_result = test_narrow_frames_and_memo(module)
    test_stochastic(module, gmn_runs, results_dir, net_result)
    test_parallel_theta(module)
    test_generate_tp_greater_than_one(module)
    print('ALL PASS')
