'''R2 acceptance test (docs/review-actions-v1.md): ccm_prefilter_observables()
raises if handed any DELAY_MARKER-suffixed column, confirming the guard
added by the R2 audit is live -- audited and confirmed the pre-filter's
candidate list can never actually contain such a column in a real run (see
docs/review-findings.md's R2 section), so this test is verifying the
defensive assertion itself, not working around a live bug.

No pytest in this project's venv -- run directly:
    python3 clean_scripts/tests/test_r2_prefilter_guard.py
'''
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402


def _call_tsls(module, bad_column):
    df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'T': [1.0, 2.0, 3.0]})
    module.ccm_prefilter_observables(
        df, [bad_column], ['T'], 1,
        0.2, 3, 1, 'correlation', 'correlation', 2, 0, 200, 2, [0.5], 1,
    )


def _call_simple(module, bad_column):
    df = pd.DataFrame({'A': [1.0, 2.0, 3.0], 'T': [1.0, 2.0, 3.0]})
    module.ccm_prefilter_observables(
        df, [bad_column], ['T'], 1, 1, 0.2, 3, 1, 0, [0.5], 1,
    )


def _check(name, module, call):
    bad_column = f'A{module.DELAY_MARKER}-1'
    raised = False
    try:
        call(module, bad_column)
    except AssertionError as exc:
        raised = True
        assert 'DELAY_MARKER' in str(exc), f'{name}: assertion message should name the guard -- {exc}'
    assert raised, f'{name}: expected an AssertionError for a DELAY_MARKER-suffixed candidate column'
    print(f'{name}: PASS')


def test_tsls():
    _check('run_serial_mde.py', load_pipeline_module('run_serial_mde.py', '_test_tsls_mde_gmn_r2'), _call_tsls)


# The 1 Hz sibling pipeline is not part of clean_scripts, so this file no
# longer has a second half exercising it -- see clean_scripts/README.md on
# scope. Prose below that still mentions it is describing the contrast, not
# running anything.

if __name__ == '__main__':
    test_tsls()
    print('ALL PASS')
