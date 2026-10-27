'''R19 acceptance test (docs/review-actions-v1.md): tsls-mde-gmn.py's new
run_mde_evaluate() -- TSLS-side counterpart of simple-mde-gmn.py's existing
dx.Evaluate()-based run_mde_evaluate(), which this pipeline never had (see
that function's own docstring). Not a sibling-parity port: simple-mde-gmn.py
keeps its own existing dx.Evaluate() call unchanged -- this task is scoped
to tsls-mde-gmn.py only (see docs/review-findings.md's R19 section).

Covers both GMN_MODE shapes on a tiny synthetic network (same pattern as
tests/test_r14_external_inputs.py/test_r11_per_node_table.py -- a real,
tiny, synthetic graph + CSV, no real data needed):

- separate (1 target, 2 direct drivers): standard time-series panel 1,
  CAE recovered matches a hand-computed sum(abs(actual - predicted)).
- combined (2 targets, asymmetric driver sets -- one target has both
  drivers, the other only one): spatial (x, y) panel 1, D = the UNION of
  both targets' own direct-driver sets, PCA/diffusion-map predict both
  target columns jointly (2-column output), and panel 2 lists each driver
  exactly once even though one driver feeds both targets (regression
  guard for a duplicate-label bug found and fixed during this task's own
  testing).

No pytest in this project's venv -- run directly:
    python3 clean_scripts/tests/test_r19_mde_evaluate.py
'''
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402

SCRATCH = Path(__file__).resolve().parent / '_scratch_r19'

RNG = np.random.default_rng(7)
N = 400
T_IDX = np.arange(N)
A = np.sin(T_IDX / 9.0) + RNG.normal(scale=0.05, size=N)
B = np.cos(T_IDX / 7.0) + RNG.normal(scale=0.05, size=N)
TX = 0.6 * A + 0.3 * B + RNG.normal(scale=0.05, size=N)
TY = 0.4 * A - 0.5 * B + RNG.normal(scale=0.05, size=N)

PREDICTION_START = 250
PREDICTION_LENGTH = 100


def _make_net_result(module, out_dir, targets, edges, network_id):
    graph = nx.MultiDiGraph()
    for t in targets:
        graph.add_node(t)
    graph.add_node('A')
    graph.add_node('B')
    for driver, target in edges:
        graph.add_edge(driver, target, delay=0, weight=1.0)

    df = pd.DataFrame({
        'Time': T_IDX, 'head_position_x': TX, 'head_position_y': TY, 'A': A, 'B': B,
    })[['Time'] + list(targets) + ['A', 'B']]
    data_file = out_dir / f'data_after_mde_{network_id}.csv'
    df.to_csv(data_file, index=False)

    return {
        'targets': list(targets), 'network_id': network_id,
        'DATA_AFTER_MDE': data_file, 'network': {'Graph': graph},
    }


def test_separate_mode():
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_mde_gmn_r19')
    out_dir = SCRATCH / 'separate'
    out_dir.mkdir(parents=True, exist_ok=True)

    net_result = _make_net_result(
        module, out_dir, ['head_position_x'],
        [('A', 'head_position_x'), ('B', 'head_position_x')], 'separate_toy',
    )
    result = module.run_mde_evaluate(
        net_result, PREDICTION_START, PREDICTION_LENGTH, out_dir,
        'correlation', gamma_delta=2,
    )

    assert result['D'] == 2, f'expected D=2 (A, B), got {result["D"]}'
    assert set(result['direct_drivers']['head_position_x']) == {'A', 'B'}
    png = out_dir / 'mde_evaluate_separate_toy.png'
    assert png.exists() and png.stat().st_size > 0, f'{png} was not written'

    # Recompute MDE's own CAE by hand from the actual/predicted arrays
    # proper_simplex returns, and confirm it matches _cae()'s reported value
    # exactly -- the whole point of R19 is reproducing dimx's own
    # sum(abs(x - y)) definition, not an approximation of it.
    data_after_mde = pd.read_csv(net_result['DATA_AFTER_MDE'])
    out = module.edm.proper_simplex(
        data_after_mde, columns=['A', 'B'], target='head_position_x',
        lib=[1, PREDICTION_START], pred=[PREDICTION_START + 1, PREDICTION_START + PREDICTION_LENGTH],
        Tp=0, embedded=True, exclusionRadius=0, metric='correlation', gammaDelta=2,
    )
    hand_cae = float(np.sum(np.abs(
        out['predictions']['Observations'].to_numpy() - out['predictions']['Predictions'].to_numpy()
    )))
    assert abs(result['mde_cae']['head_position_x'] - hand_cae) < 1e-9, (
        f"CAE mismatch: {result['mde_cae']['head_position_x']} vs hand-computed {hand_cae}"
    )
    assert np.isfinite(result['mde_score']['head_position_x'])
    assert np.isfinite(result['pca_score']) and np.isfinite(result['dmap_score'])
    print(f'test_separate_mode: PASS -- D={result["D"]}, '
          f'mde_cae={result["mde_cae"]}, pca_cae={result["pca_cae"]:.2f}, '
          f'dmap_cae={result["dmap_cae"]:.2f}')


def test_combined_mode_asymmetric_drivers():
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_mde_gmn_r19')
    out_dir = SCRATCH / 'combined'
    out_dir.mkdir(parents=True, exist_ok=True)

    net_result = _make_net_result(
        module, out_dir, ['head_position_x', 'head_position_y'],
        [('A', 'head_position_x'), ('B', 'head_position_x'), ('A', 'head_position_y')],
        'combined_toy',
    )
    result = module.run_mde_evaluate(
        net_result, PREDICTION_START, PREDICTION_LENGTH, out_dir,
        'correlation', gamma_delta=2,
    )

    # D is the UNION of both targets' own direct-driver sets: x has {A, B},
    # y has {A} -- union is {A, B}, not 3 (this pipeline's own reading of a
    # task requirement that only defines D for a single target -- see
    # docs/review-findings.md's R19 section).
    assert result['D'] == 2, f'expected union D=2 (A, B), got {result["D"]}'
    assert set(result['direct_drivers']['head_position_x']) == {'A', 'B'}
    assert set(result['direct_drivers']['head_position_y']) == {'A'}

    png = out_dir / 'mde_evaluate_combined_toy.png'
    assert png.exists() and png.stat().st_size > 0, f'{png} was not written'

    # PCA/diffusion-map jointly regress onto BOTH target columns -- a
    # 2-column prediction, not two independent single-target runs.
    assert result['pca_cae'] > 0 and np.isfinite(result['pca_score'])
    assert result['dmap_cae'] > 0 and np.isfinite(result['dmap_score'])
    assert set(result['mde_score'].keys()) == {'head_position_x', 'head_position_y'}
    for t in ('head_position_x', 'head_position_y'):
        assert np.isfinite(result['mde_score'][t]), f'{t}: non-finite MDE score'

    print(f'test_combined_mode_asymmetric_drivers: PASS -- D={result["D"]}, '
          f'mde_cae={result["mde_cae"]}, mde_score={result["mde_score"]}')


if __name__ == '__main__':
    test_separate_mode()
    test_combined_mode_asymmetric_drivers()
    print('ALL PASS')
