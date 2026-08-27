'''R15 acceptance test (docs/review-actions-v1.md): Stage 2's real-
observable recursion terminations must be logged distinguishably --
budget-bound (an accepted delay=0 pick that would have recursed except the
shared nested_m budget already stopped at this depth) versus driver-
exhausted (a node still within budget whose Stage 2 call simply selected
no real-observable driver at all).

Reuses R13's exact scenario (nested_m=1): T (depth 0) -> Stage 1 -> A
(depth 1, exactly at the cap). Stage 2: T's own call selects nothing (T is
under budget with zero real picks -> driver-exhausted); A's call selects a
brand-new real observable D, but A is already at the depth cap -> D's pick
is budget-bound. Expected: exactly 1 driver-exhausted (T), 1 budget-bound
(D) -- captured from build_nested_mde_network()'s own log line.

No pytest in this project's venv -- run directly:
    python3 tests/test_r15_termination_causes.py
'''
import contextlib
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import (  # noqa: E402
    FakeProcessPoolExecutor,
    load_pipeline_module,
    make_fake_run_mde_for_target_simple,
    make_fake_run_mde_for_target_tsls,
)

NESTED_M = 1

PLAN = {
    ('stage1', 'T'): ['A'],
    ('stage1', 'A'): [],
    ('stage2', 'T'): [],      # T: under budget (depth 0 < 1), zero real picks -> driver-exhausted
    ('stage2', 'A'): ['D'],   # A: at the cap (depth 1), one real pick -> budget-bound
}


def _toy_df():
    rng = np.random.default_rng(2)
    return pd.DataFrame({'A': rng.normal(size=20), 'B': rng.normal(size=20),
                          'C': rng.normal(size=20), 'D': rng.normal(size=20)})


def _run(module, fake_run_mde_for_target, extra_kwargs):
    call_log = []
    module.run_mde_for_target = fake_run_mde_for_target(PLAN, call_log)
    module.ProcessPoolExecutor = FakeProcessPoolExecutor

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        module.build_nested_mde_network(
            targets=['T'],
            expanded_df=_toy_df(),
            raw_columns=['A', 'B', 'C', 'D'],
            expanded_columns=['A', 'B', 'C', 'D'],
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
            lag_set=[-1],
            stage2_allow_real_observables=True,
            **extra_kwargs,
        )
    return buf.getvalue()


def _check(name, log_text):
    m = re.search(
        r'real-observable recursion terminations -- (\d+) budget-bound, (\d+) driver-exhausted',
        log_text,
    )
    assert m, f'{name}: expected termination-cause log line not found -- log:\n{log_text}'
    budget_bound, driver_exhausted = int(m.group(1)), int(m.group(2))
    assert budget_bound == 1, f'{name}: expected 1 budget-bound termination (D), got {budget_bound}'
    assert driver_exhausted == 1, (
        f'{name}: expected 1 driver-exhausted termination (T), got {driver_exhausted}'
    )
    print(f'{name}: PASS (budget_bound={budget_bound}, driver_exhausted={driver_exhausted})')


def test_tsls():
    log_text = _run(
        load_pipeline_module('tsls-mde-gmn.py', '_test_tsls_mde_gmn_r15'),
        make_fake_run_mde_for_target_tsls,
        {'metric': 'correlation', 'target_metric': 'correlation', 'gamma_delta': 2,
         'ccm_window_rows': 200, 'ccm_lib_size_pcts': (10, 50), 'ccm_sample': 2,
         'ccm_slope': 0.01, 'cores': 2},
    )
    _check('tsls-mde-gmn.py', log_text)


def test_simple():
    log_text = _run(
        load_pipeline_module('simple-mde-gmn.py', '_test_simple_mde_gmn_r15'),
        make_fake_run_mde_for_target_simple, {'cores': 2},
    )
    _check('simple-mde-gmn.py', log_text)


if __name__ == '__main__':
    test_tsls()
    test_simple()
    print('ALL PASS')
