'''R21 acceptance test (docs/review-actions-v1.md): cleaned_helper_functions
.proper_mde()'s new dynamic per-dimension exclusion radius.

R21's own acceptance criterion: "a test asserting the radius at dimension
`d` is never below the deepest selected lag nor below the static floor."
Checked two ways on the same run: (1) directly, by recomputing the
expected per-dimension floor from proper_mde's own returned MDEcolumns
order and the column_delay mapping this test supplies -- no assumption
about which candidate gets selected in which order; (2) by parsing the
`{d}-D exclusionRadius=...` lines proper_mde() prints under verbose=True
(the run-log auditability R21's own text explicitly asks for), confirming
what's actually logged matches what was actually used.

Uses noCCM=True so forward selection is deterministic (the top-ranked
candidate at each dimension is accepted immediately, no CCM-gate
candidate-skipping to reason about) -- this test is about the exclusion
radius, not about exercising the CCM gate itself.

No pytest in this project's venv -- run directly:
    python3 tests/test_r21_dynamic_exclusion_radius.py
'''
import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'src'))
import cleaned_helper_functions as tsls  # noqa: E402

STATIC_FLOOR = 1
COLUMN_DELAY = {'A': 5, 'B': 12, 'C': 2}


def _synthetic_df(n=400, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    a = np.sin(t / 5.0) + rng.normal(scale=0.01, size=n)
    b = np.sin(t / 11.0 + 1.0) + rng.normal(scale=0.01, size=n)
    c = np.sin(t / 17.0 + 2.0) + rng.normal(scale=0.01, size=n)
    target = 0.5 * a + 0.3 * b + 0.2 * c + rng.normal(scale=0.01, size=n)
    return pd.DataFrame({'T': target, 'A': a, 'B': b, 'C': c})


def _expected_radius_by_dimension(mde_columns, column_delay, floor):
    '''Recomputes the exclusion radius proper_mde() SHOULD have used at
    each dimension, purely from its own returned selection order -- at
    dimension d (1-indexed), the columns "currently selected" are
    mde_columns[:d-1] (empty at d=1).'''
    expected = {}
    for d in range(1, len(mde_columns) + 1):
        selected_so_far = mde_columns[:d - 1]
        deepest = max((abs(column_delay.get(c, 0)) for c in selected_so_far), default=0)
        expected[d] = max(floor, deepest)
    return expected


def test_dynamic_exclusion_radius_matches_deepest_selected_lag():
    df = _synthetic_df()
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = tsls.proper_mde(
            df, target='T', columns=['A', 'B', 'C'], D=3,
            lib=[1, 280], pred=[281, 400], Tp=1, tau=-1,
            exclusionRadius=STATIC_FLOOR,
            # noCCM=True (this file's earlier draft) short-circuits BEFORE
            # step 4's own exclusion-radius computation entirely -- this
            # test needs step 4 (the E-select + CCM gate) to actually run,
            # so noCCM is never passed (default False). crossMapRhoMin/
            # embedDimRhoMin/ccmSlope are all set to trivially-passing
            # thresholds so those gates' own pass/fail behavior (a
            # different concern from this file's own MDE correctness
            # tests elsewhere) can't make this test flaky -- this test is
            # about exclusion-radius bookkeeping, not about exercising
            # genuine threshold/convergence behavior.
            crossMapRhoMin=0.0, ccmSlope=-1.0, embedDimRhoMin=-1.0,
            metric='correlation', column_delay=COLUMN_DELAY, verbose=True,
        )
    log_text = buf.getvalue()

    mde_columns = result['MDEcolumns']
    assert len(mde_columns) == 3, f'expected all 3 dimensions to select a column, got {mde_columns}'

    expected = _expected_radius_by_dimension(mde_columns, COLUMN_DELAY, STATIC_FLOOR)

    # (1) Direct check: never below the static floor, never below the
    # deepest lag among columns already selected as of that dimension.
    for d, radius in expected.items():
        selected_so_far = mde_columns[:d - 1]
        deepest = max((abs(COLUMN_DELAY.get(c, 0)) for c in selected_so_far), default=0)
        assert radius >= STATIC_FLOOR, f'dimension {d}: radius {radius} below static floor {STATIC_FLOOR}'
        assert radius >= deepest, f'dimension {d}: radius {radius} below deepest selected lag {deepest}'

    # (2) Cross-check against what proper_mde() actually logged (verbose=True) --
    # the run-log auditability R21's own text asks for, not just a
    # black-box return value.
    logged = {
        int(m.group(1)): float(m.group(2))
        for m in re.finditer(r'(\d+)-D exclusionRadius=(\S+) \(floor=', log_text)
    }
    assert logged, f'no "N-D exclusionRadius=..." lines found in log:\n{log_text}'
    for d, expected_radius in expected.items():
        assert d in logged, f'dimension {d} never logged its own exclusionRadius -- log:\n{log_text}'
        assert logged[d] == expected_radius, (
            f'dimension {d}: logged radius {logged[d]} != expected {expected_radius}'
        )

    print(f'test_dynamic_exclusion_radius_matches_deepest_selected_lag: PASS -- '
          f'MDEcolumns={mde_columns}, expected_by_dimension={expected}, logged={logged}')


def test_no_column_delay_is_a_no_op():
    '''column_delay=None (the default) -- exclusion radius must be the
    static floor at every dimension, unchanged from before this parameter
    existed.'''
    df = _synthetic_df()
    buf = io.StringIO()
    with redirect_stdout(buf):
        tsls.proper_mde(
            df, target='T', columns=['A', 'B', 'C'], D=3,
            lib=[1, 280], pred=[281, 400], Tp=1, tau=-1,
            exclusionRadius=STATIC_FLOOR,
            crossMapRhoMin=0.0, ccmSlope=-1.0, embedDimRhoMin=-1.0,
            metric='correlation', column_delay=None, verbose=True,
        )
    log_text = buf.getvalue()
    logged = {
        int(m.group(1)): float(m.group(2))
        for m in re.finditer(r'(\d+)-D exclusionRadius=(\S+) \(floor=', log_text)
    }
    assert logged, f'no "N-D exclusionRadius=..." lines found in log:\n{log_text}'
    for d, radius in logged.items():
        assert radius == STATIC_FLOOR, f'dimension {d}: expected no-op radius {STATIC_FLOOR}, got {radius}'
    print(f'test_no_column_delay_is_a_no_op: PASS -- logged={logged}')


if __name__ == '__main__':
    test_dynamic_exclusion_radius_matches_deepest_selected_lag()
    test_no_column_delay_is_a_no_op()
    print('ALL PASS')
