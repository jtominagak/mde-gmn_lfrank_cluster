'''R7 acceptance test (docs/review-actions-v1.md): run_tp_vs_rho_sweep()
and run_univariate_e_tau_grid() must not leave a half-written/empty figure
on the per-target skip path, and must not crash the whole diagnostics step
when one target's depth-0 MDE call failed (mde_df=None, distinct from a
target that legitimately selected zero drivers -- see
failed_mde_result()'s own docstring).

No pytest in this project's venv -- run directly:
    python3 clean_scripts/tests/test_r7_diagnostics_figures.py
'''
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402


def _toy_mde_df(target_name):
    rng = np.random.default_rng(3)
    n = 60
    a = np.sin(np.arange(n) / 4.0) + rng.normal(scale=0.01, size=n)
    t = 0.6 * a + rng.normal(scale=0.01, size=n)
    return pd.DataFrame({target_name: t, 'A': a})


def _valid_result(target_name='T'):
    df = _toy_mde_df(target_name)
    return {
        'target': target_name,
        'mde_df': df,
        'm': SimpleNamespace(args=SimpleNamespace(lib=[1, 40], pred=[41, 60], tau=-1, Tp=1)),
        'SELECTED_COLUMNS': ['A'],
    }


def _zero_driver_result(target_name='T'):
    r = _valid_result(target_name)
    r['SELECTED_COLUMNS'] = []
    return r


def _failed_result(target_name='T'):
    return {
        'target': target_name,
        'mde_df': None,
        'm': SimpleNamespace(args=SimpleNamespace(lib=None, pred=None, tau=None, Tp=None)),
        'SELECTED_COLUMNS': [],
    }


def _check_tp_vs_rho_all_skip(name, module, results_dir, extra_kwargs):
    figure_path = results_dir / 'mde_tp_vs_rho_multi_target.png'
    if figure_path.exists():
        figure_path.unlink()
    mde_results = {'T1': _zero_driver_result('T1'), 'T2': _zero_driver_result('T2')}
    module.run_tp_vs_rho_sweep(mde_results, ['T1', 'T2'], 5, {'T1': 'red', 'T2': 'blue'},
                                results_dir, **extra_kwargs)
    assert not figure_path.exists(), (
        f'{name}: run_tp_vs_rho_sweep() wrote a figure even though every target had zero '
        f'MDE-selected columns -- {figure_path}'
    )
    print(f'{name}: run_tp_vs_rho_sweep all-skip -> no figure written, PASS')


def _check_e_tau_grid_survives_one_failure(name, module, results_dir, extra_kwargs):
    e_range, tau_range = [2, 3], [-1, -2]
    good_path = results_dir / 'mde_E_tau_grid_good.png'
    bad_path = results_dir / 'mde_E_tau_grid_bad.png'
    for p in (good_path, bad_path):
        if p.exists():
            p.unlink()

    mde_results = {'good': _valid_result('good'), 'bad': _failed_result('bad')}
    # Must not raise -- a failed target's mde_df=None must not crash the
    # whole function (and take 'good''s own otherwise-computable figure
    # down with it, since the exception would propagate mid-loop).
    module.run_univariate_e_tau_grid(mde_results, ['bad', 'good'], e_range, tau_range, 0.9,
                                      results_dir, **extra_kwargs)
    assert not bad_path.exists(), f"{name}: 'bad' (mde_df=None) should not have produced a figure"
    assert good_path.exists() and good_path.stat().st_size > 0, (
        f"{name}: 'good' target's figure should still be produced even though 'bad' failed -- "
        f"{good_path}"
    )
    print(f'{name}: run_univariate_e_tau_grid survives one failed target, PASS')


def _run_for_module(filename, module_name, name, extra_tp_kwargs, extra_grid_kwargs):
    module = load_pipeline_module(filename, module_name)
    results_dir = Path(__file__).resolve().parent / '_scratch_r7' / module_name
    results_dir.mkdir(parents=True, exist_ok=True)

    _check_tp_vs_rho_all_skip(name, module, results_dir, extra_tp_kwargs)
    _check_e_tau_grid_survives_one_failure(name, module, results_dir, extra_grid_kwargs)


def test_tsls():
    _run_for_module(
        'run_serial_mde.py', '_test_tsls_mde_gmn_r7', 'run_serial_mde.py',
        {'target_metric': 'correlation', 'gamma_delta': 2},
        {'target_metric': 'correlation', 'gamma_delta': 2},
    )


# The 1 Hz sibling pipeline is not part of clean_scripts, so this file no
# longer has a second half exercising it -- see clean_scripts/README.md on
# scope. Prose below that still mentions it is describing the contrast, not
# running anything.

if __name__ == '__main__':
    test_tsls()
    print('ALL PASS')
