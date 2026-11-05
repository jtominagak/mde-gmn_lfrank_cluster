'''R14 acceptance tests (docs/review-actions-v1.md): leaf nodes (zero
discovered drivers) become external inputs, not simulated network members.

Two kinds of check:
  1. Unit-level: partition_network()'s classification and its explicit
     target-zero-drivers exemption (raises, per the user's own correction
     during this review -- a target can never become an external input).
  2. Integration-level: a real (tiny, synthetic) gmn.GMN.Generate() run,
     built the same way build_network()/build_gmn_config() build one for
     real, proving EXTERNAL_INPUT_MODE='real_data' actually feeds the real
     recorded trajectory for a zero-driver node (not a self-prediction),
     while 'self_simulate' reproduces the pre-R14 unconditional
     self-history embedding without crashing.

No pytest in this project's venv -- run directly:
    python3 clean_scripts/tests/test_r14_external_inputs.py
'''
import pickle
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402

SCRATCH = Path(__file__).resolve().parent / '_scratch_r14'


def _toy_graph():
    '''T (target, interior) <- A (delay 0) -- driven by A, so T has 1
    discovered driver and A has 0 -- A is the external input under test.'''
    graph = nx.MultiDiGraph()
    graph.add_node('T')
    graph.add_node('A')
    graph.add_edge('A', 'T', delay=0, weight=0.5)
    return graph


def _check_partition(module, name):
    graph = _toy_graph()
    interior, external = module.partition_network(graph, ['T'], 'real_data')
    assert interior == {'T'}, f'{name}: expected interior={{"T"}}, got {interior}'
    assert external == {'A'}, f'{name}: expected external={{"A"}}, got {external}'
    assert graph.graph['interior_nodes'] == {'T'}
    assert graph.graph['external_inputs'] == {'A'}
    assert graph.graph['external_input_mode'] == 'real_data'

    # A target with zero drivers must raise, never become an external input
    # (T2 is isolated -- zero in-edges of its own -- exactly Run A's real
    # head_position_y case: a target that found no drivers at all).
    zero_driver_target_graph = nx.MultiDiGraph()
    zero_driver_target_graph.add_node('T')
    zero_driver_target_graph.add_node('T2')
    raised = False
    try:
        module.partition_network(zero_driver_target_graph, ['T2'], 'real_data')
    except ValueError as exc:
        raised = True
        assert 'T2' in str(exc) and 'failed network' in str(exc), (
            f'{name}: error message should name the target and explain why -- {exc}'
        )
    assert raised, f'{name}: a zero-driver TARGET must raise, not silently become external'

    # R14's acceptance test, corrected: "no node's embedding contains a
    # self-lag that nested-MDE did not select" applies to NONZERO self-lags
    # only -- the delay=0 anchor is expected on every node (see
    # embedding_spec_for()'s own docstring for why: gmn's SMap call needs
    # it as the target column, and it's standard generalized-embedding
    # practice besides). T's spec must be exactly its own anchor plus its
    # one discovered driver (A, delay 0) -- no deeper self-lag (T, -1),
    # (T, -3), etc. anywhere, since none was ever offered as a candidate
    # nested-MDE could select in the first place (self-loops are excluded).
    graph2 = _toy_graph()
    module.partition_network(graph2, ['T'], 'real_data')
    spec_t = module.embedding_spec_for(graph2, 'T')
    assert ('T', 0) in spec_t, (
        f'{name}: T\'s spec is missing its own unconditional (T, 0) anchor -- spec={spec_t}'
    )
    assert ('A', 0) in spec_t, f"{name}: T's spec is missing its discovered driver (A, 0) -- spec={spec_t}"
    nonzero_self_lags = [(col, d) for col, d in spec_t if col == 'T' and d != 0]
    assert not nonzero_self_lags, (
        f"{name}: T's spec contains a nonzero self-lag nested-MDE never selected -- {nonzero_self_lags}"
    )
    assert len(spec_t) == 2, f"{name}: T's spec should be exactly [(T,0), (A,0)] -- got {spec_t}"

    print(f'{name}: partition_network PASS (spec_t={spec_t})')


def _generate_run(module, external_input_mode, results_dir):
    '''Builds and runs a real (tiny) gmn.GMN.Generate() call the same way
    build_network()/build_gmn_config() build one for a genuine pipeline
    run, for the T<-A toy graph above, under the given
    EXTERNAL_INPUT_MODE. Returns (G, data_after_mde) -- the live gmn.GMN
    object post-Generate() and the full real dataframe it was built from.'''
    rng = np.random.default_rng(0)
    n_rows = 200
    t = np.arange(n_rows)
    a_series = np.sin(t / 5.0) + rng.normal(scale=0.01, size=n_rows)
    t_series = 0.5 * np.roll(a_series, 1) + rng.normal(scale=0.01, size=n_rows)
    data_after_mde = pd.DataFrame({'Time': t, 'T': t_series, 'A': a_series})

    data_file = results_dir / 'data_after_mde_toy.csv'
    data_after_mde.to_csv(data_file, index=False)

    graph = _toy_graph()
    module.partition_network(graph, ['T'], external_input_mode)
    network_file = results_dir / f'network_toy_{external_input_mode}.pkl'
    with open(network_file, 'wb') as f:
        pickle.dump({'Graph': graph, 'Map': {n: list(graph.predecessors(n)) for n in graph.nodes}}, f)

    net_result = {
        'targets': ['T'],
        'network_id': f'toy_{external_input_mode}',
        'DATA_AFTER_MDE': data_file,
        'NETWORK_FILE': network_file,
    }

    generate_start = 150
    generate_length = 10
    config_file = module.build_gmn_config(
        net_result, generate_start, generate_length, results_dir,
        mode='Generate', theta=3, generate_length=generate_length,
        generate_start=generate_start,
    )
    G = module.gmn.GMN(configFile=str(config_file), cores=1)
    G.Generate()
    return G, data_after_mde, generate_start, generate_length


def _check_real_data_mode(module, name, results_dir):
    G, data_after_mde, generate_start, generate_length = _generate_run(
        module, 'real_data', results_dir)
    generated_a = G.DataOut['A'].to_numpy()
    real_a = data_after_mde['A'].to_numpy()[generate_start:generate_start + generate_length]
    assert np.allclose(generated_a, real_a), (
        f"{name}: EXTERNAL_INPUT_MODE='real_data' should feed A's real recorded values "
        f"verbatim -- generated={generated_a}, real={real_a}"
    )
    generated_t = G.DataOut['T'].to_numpy()
    assert np.all(np.isfinite(generated_t)), (
        f"{name}: T (interior, driven by A) produced non-finite output: {generated_t}"
    )
    print(f"{name}: EXTERNAL_INPUT_MODE='real_data' PASS (A matches real data exactly, "
          f"T finite: {generated_t})")


def _check_self_simulate_mode(module, name, results_dir):
    G, data_after_mde, generate_start, generate_length = _generate_run(
        module, 'self_simulate', results_dir)
    generated_a = G.DataOut['A'].to_numpy()
    generated_t = G.DataOut['T'].to_numpy()
    assert np.all(np.isfinite(generated_a)), (
        f"{name}: EXTERNAL_INPUT_MODE='self_simulate' should still self-predict A via its "
        f"old unconditional self-history embedding without crashing -- got {generated_a}"
    )
    assert np.all(np.isfinite(generated_t)), f'{name}: T produced non-finite output: {generated_t}'
    print(f"{name}: EXTERNAL_INPUT_MODE='self_simulate' PASS (A self-simulated: {generated_a})")


def _toy_graph_with_lag_and_ar():
    '''T <- A (delay 0, external leaf) and T <- T (delay -5, AR self-loop --
    exercises the delay_copy code path for an interior node's own lagged
    copy) and T <- A (delay -3, exercises the delay_copy path for an
    external input's OWN lagged edge, a distinct entity from A's plain
    delay=0 leaf). Only used against tsls-mde-gmn.py's interactive 3D
    redesign -- _toy_graph() alone never creates a delay_copy entity.'''
    graph = nx.MultiDiGraph()
    graph.add_node('T')
    graph.add_node('A')
    graph.add_edge('A', 'T', delay=0, weight=0.5)
    graph.add_edge('T', 'T', delay=-5, weight=0.3)
    graph.add_edge('A', 'T', delay=-3, weight=0.2)
    return graph


def _check_partition_report_and_3d_figure(module, name, results_dir,
                                           expected_suffix='.png', check_elevated_entities=False):
    '''R14's other two explicit deliverables: the partition report (run log
    + CSV in results_dir) and the 3D network figure, both required "on a
    normal run" -- exercised here directly against the toy graph, real
    matplotlib/networkx (or plotly, for tsls-mde-gmn.py's interactive HTML
    redesign) calls, no mocking. `check_elevated_entities` additionally
    exercises the delay_copy/leaf entity code path introduced by that
    redesign -- only meaningful (and only passed) for tsls-mde-gmn.py,
    since simple-mde-gmn.py keeps the old static-PNG version.'''
    graph = _toy_graph()
    module.partition_network(graph, ['T'], 'real_data')
    module.report_partition(graph, 'toy_report', results_dir)
    partition_csv = results_dir / 'partition_toy_report.csv'
    assert partition_csv.exists(), f'{name}: report_partition() did not write {partition_csv}'
    partition_df = pd.read_csv(partition_csv)
    assert set(partition_df['node']) == {'T', 'A'}, f'{name}: partition CSV missing a node -- {partition_df}'
    a_row = partition_df[partition_df['node'] == 'A'].iloc[0]
    assert a_row['role'] == 'external_input' and a_row['feeds'] == 'T' and int(a_row['delay']) == 0, (
        f'{name}: partition CSV row for A is wrong -- {a_row.to_dict()}'
    )

    net_result = {'network': {'Graph': graph}, 'targets': ['T'], 'network_id': 'toy_3d'}
    figure_path = results_dir / f'network_3d_toy{expected_suffix}'
    module.plot_3d_network(net_result, {'T': 'tab:red'}, figure_path)
    assert figure_path.exists() and figure_path.stat().st_size > 0, (
        f'{name}: plot_3d_network() did not write a non-empty figure to {figure_path}'
    )
    print(f'{name}: report_partition + plot_3d_network PASS ({figure_path})')

    if not check_elevated_entities:
        return

    lag_graph = _toy_graph_with_lag_and_ar()
    module.partition_network(lag_graph, ['T'], 'real_data')
    lag_net_result = {'network': {'Graph': lag_graph}, 'targets': ['T'], 'network_id': 'toy_3d_lag'}
    lag_figure_path = results_dir / 'network_3d_toy_lag.html'
    module.plot_3d_network(lag_net_result, {'T': 'tab:red'}, lag_figure_path)
    html = lag_figure_path.read_text()
    assert 'A' in html, f'{name}: leaf entity "A" missing from {lag_figure_path}'
    assert 'T_delay_-5' in html, f'{name}: AR self-loop entity "T_delay_-5" missing from {lag_figure_path}'
    assert 'A_delay_-3' in html, f'{name}: lagged-leaf entity "A_delay_-3" missing from {lag_figure_path}'
    assert '"cone"' in html, f'{name}: no Cone (arrowhead) trace found in {lag_figure_path}'
    print(f'{name}: plot_3d_network elevated-entity check PASS ({lag_figure_path})')


def _check_fidelity_skips_passthrough(module, name, results_dir):
    '''gmn_fidelity_rho() must NOT score a real-data passthrough node.

    Under EXTERNAL_INPUT_MODE='real_data', A has zero drivers, so Generate()
    replays A's recording verbatim -- _check_real_data_mode() above asserts
    exactly that. Scoring A would then cross-map a channel onto an identical
    copy of itself and report it as generative fidelity: circular in the same
    way partition_network() already refuses for a target. Job 4774136
    (2026-10-27) made that concrete -- 23 of its 29 nodes were passthroughs
    and supplied every fidelity rho above 0.44, two of them 1.0000, while all
    six simulated nodes scored <= 0.054.

    Under 'self_simulate' the same node IS simulated, so it must be scored --
    which is why this asserts the mode is respected rather than that
    zero-driver nodes are always excluded.'''
    for mode, expect_scored, expect_passthrough in (('real_data', {'T'}, ['A']),
                                                    ('self_simulate', {'T', 'A'}, [])):
        G, _data, generate_start, _length = _generate_run(module, mode, results_dir)
        with open(results_dir / f'network_toy_{mode}.pkl', 'rb') as f:
            network = pickle.load(f)
        net_result = {
            'targets': ['T'],
            'network': network,
            'DATA_AFTER_MDE': results_dir / 'data_after_mde_toy.csv',
            'node_dimensions': {},
        }
        rho, passthrough = module.gmn_fidelity_rho(
            net_result, G, generate_start, 'correlation', 'correlation', 2)
        assert passthrough == expect_passthrough, (
            f'{name}: EXTERNAL_INPUT_MODE={mode!r} should report passthrough='
            f'{expect_passthrough}, got {passthrough}')
        assert set(rho) == expect_scored, (
            f'{name}: EXTERNAL_INPUT_MODE={mode!r} should score {expect_scored}, '
            f'got {set(rho)}')
        # The node is excluded from SCORING only -- it is still a node of the
        # network, and both network figures still draw it.
        assert set(network['Graph'].nodes) == {'T', 'A'}, (
            f'{name}: the graph itself must be unchanged: {set(network["Graph"].nodes)}')
    print(f'{name}: fidelity scores only simulated nodes, passthrough reported separately PASS')


def _check_forecast_skips_passthrough(module, name, results_dir):
    '''run_forecast_diagnostic() must not score a passthrough node either --
    but must still PLOT it.

    Forecast() replays the recording for a zero-driver node just as Generate()
    does, so tsls._score(actual, forecast_pred) compares the recording with
    itself. Unlike the fidelity score, the node keeps its panel in the
    figure: seeing the real trajectory beside the Generate() curve is the
    point of that diagnostic. This also exercises the real call path, where
    the returned tuple is unpacked -- the kind of slip that otherwise only
    shows up hours into a job.'''
    G, data_after_mde, generate_start, generate_length = _generate_run(
        module, 'real_data', results_dir)
    with open(results_dir / 'network_toy_real_data.pkl', 'rb') as f:
        network = pickle.load(f)
    net_result = {
        'targets': ['T'],
        'network_id': 'toy_forecast',
        'network': network,
        'DATA_AFTER_MDE': results_dir / 'data_after_mde_toy.csv',
        'NETWORK_FILE': results_dir / 'network_toy_real_data.pkl',
    }
    save_path = results_dir / 'forecast_toy.png'
    Gf, rhos, passthrough = module.run_forecast_diagnostic(
        net_result, G, generate_start, generate_length,
        {'T': 3, 'A': 3}, results_dir, 1, save_path,
        'correlation', 'correlation', 2,
        generate_length=generate_length, generate_start=generate_start,
        total_rows=len(data_after_mde),
    )
    assert Gf is not None, (
        f'{name}: the toy Forecast() run failed, so this check proved nothing '
        f'(rhos={rhos}, passthrough={passthrough})')
    assert passthrough == ['A'], (
        f"{name}: A is a real-data passthrough and must be reported as one, got {passthrough}")
    assert 'A' not in rhos, f'{name}: A must not be scored by Forecast(), got {rhos}'
    assert 'T' in rhos, f'{name}: T is simulated and must be scored, got {rhos}'
    # Plotted, not dropped: the figure is written and covers both nodes.
    parts = sorted(results_dir.glob('forecast_toy_part*.png'))
    assert parts, f'{name}: run_forecast_diagnostic() wrote no figure'
    print(f'{name}: Forecast() scores only simulated nodes, passthrough still plotted PASS '
          f'({[p.name for p in parts]})')


def _run_for_module(filename, module_name, name, expected_suffix='.png', check_elevated_entities=False):
    module = load_pipeline_module(filename, module_name)
    results_dir = SCRATCH / module_name
    results_dir.mkdir(parents=True, exist_ok=True)

    _check_partition(module, name)
    _check_real_data_mode(module, name, results_dir)
    _check_self_simulate_mode(module, name, results_dir)
    _check_partition_report_and_3d_figure(
        module, name, results_dir,
        expected_suffix=expected_suffix, check_elevated_entities=check_elevated_entities,
    )
    # Unconditional here: the 1 Hz pipeline, which keeps its own older
    # one-file versions of these two functions, is not part of this tree.
    _check_fidelity_skips_passthrough(module, name, results_dir)
    _check_forecast_skips_passthrough(module, name, results_dir)


def test_tsls():
    _run_for_module('run_serial_mde.py', '_test_tsls_mde_gmn_r14', 'run_serial_mde.py',
                     expected_suffix='.html', check_elevated_entities=True)


# The 1 Hz sibling pipeline is not part of clean_scripts, so this file no
# longer has a second half exercising it -- see clean_scripts/README.md on
# scope. Prose below that still mentions it is describing the contrast, not
# running anything.

if __name__ == '__main__':
    SCRATCH.mkdir(parents=True, exist_ok=True)
    test_tsls()
    print('ALL PASS')
