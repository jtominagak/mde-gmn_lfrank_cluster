'''Test for the progress/timing monitoring log added to tsls-mde-gmn.py
(not one of the 22 numbered review tasks -- a follow-up requested after a
real /bucket outage, to diagnose this pipeline's own history of runs that
never finish -- see the plan this was implemented from for the full
design). Covers `_log_progress()`/`_format_eta()` directly (pure,
side-effect-isolated helpers) and `_preview_window_rows()`'s own
behavioral-target short-circuit (the exact bug this test would have
caught: an earlier version read the target's own column unconditionally,
which crashed test_r12/r13/r15's own toy DataFrames that never include
the target's column since those tests fake out run_mde_for_target()
entirely and never actually read it).

No pytest in this project's venv -- run directly:
    python3 clean_scripts/tests/test_progress_monitoring.py
'''
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402


def test_log_progress_writes_timestamped_lines():
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_mde_gmn_progress_log')
    with tempfile.TemporaryDirectory() as tmp:
        log_path = Path(tmp) / 'progress.log'
        module._log_progress(log_path, 'first event')
        module._log_progress(log_path, 'second event')
        lines = log_path.read_text().splitlines()
        assert len(lines) == 2, f'expected 2 lines, got {len(lines)}: {lines}'
        for line in lines:
            # ISO timestamp, then a space, then the message -- split(' ', 1)
            # isolates the timestamp field without assuming anything about
            # the message's own content.
            ts_field = line.split(' ', 1)[0]
            assert 'T' in ts_field, f'expected an ISO timestamp (with a T) at the start of: {line!r}'
        assert lines[0].endswith('first event')
        assert lines[1].endswith('second event')
    print('test_log_progress_writes_timestamped_lines: PASS')


def test_log_progress_none_path_is_a_noop():
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_mde_gmn_progress_log_noop')
    # Must not raise -- every real call site passes a real path, but tests
    # (and any future call site that doesn't have one yet) rely on `None`
    # being silently accepted rather than needing an `if` guard everywhere.
    module._log_progress(None, 'should be a no-op, not an exception')
    print('test_log_progress_none_path_is_a_noop: PASS')


def test_format_eta_empty_inputs_return_empty_string():
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_mde_gmn_progress_eta_empty')
    assert module._format_eta([], 5) == ''
    assert module._format_eta([1.0, 2.0], 0) == ''
    assert module._format_eta([], 3, per_unit=[], unit_remaining=100) == ''
    print('test_format_eta_empty_inputs_return_empty_string: PASS')


def test_format_eta_flat_average():
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_mde_gmn_progress_eta_flat')
    eta = module._format_eta([10.0, 20.0, 30.0], 4)
    # mean=20, 4 remaining -> ~80s
    assert '~80s' in eta, f'expected ~80s in {eta!r}'
    assert '3 completed' in eta
    print(f'test_format_eta_flat_average: PASS -- {eta!r}')


def test_format_eta_per_window_row_rate_preferred_over_flat():
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_mde_gmn_progress_eta_rate')
    # per_unit (seconds per window-row) present and unit_remaining > 0 ->
    # the rate-based estimate must be used, not the flat one, even though
    # both `durations`/`n_remaining` are also supplied.
    eta = module._format_eta(durations=[999.0], n_remaining=999,
                              per_unit=[0.01, 0.02], unit_remaining=1000)
    # mean rate = 0.015 s/row * 1000 rows = 15s
    assert '~15s' in eta, f'expected the rate-based ~15s estimate, got {eta!r}'
    assert '999' not in eta, f'flat estimate leaked through instead of the rate-based one: {eta!r}'
    print(f'test_format_eta_per_window_row_rate_preferred_over_flat: PASS -- {eta!r}')


def _periodic_tsls(n, period):
    vals = np.arange(n, dtype=float) % period
    return vals


def test_preview_window_rows_behavioral_target_never_reads_column():
    '''The exact bug found via the full test suite: an earlier version of
    _preview_window_rows() read full_df[[node]] unconditionally, before
    checking is_behavioral_target -- crashing on any real caller (e.g.
    test_r12/r13/r15's own toy DataFrames) whose frame doesn't happen to
    include the target's own column, even though the behavioral branch
    never needed to read it in the first place. Passing a DataFrame that
    genuinely lacks the target's column is the direct regression test for
    that fix, not just a convenience.'''
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_mde_gmn_progress_preview_behavioral')
    df_without_target_column = pd.DataFrame({'A': np.arange(2000, dtype=float),
                                              'B': np.arange(2000, dtype=float)})
    window_rows = module._preview_window_rows(
        'T', df_without_target_column, delay_set=[], prediction_start=1000,
        prediction_length=500, tp=1,
        is_behavioral_target=True, target_ranking_window_rows=300,
        min_pred_events=20, max_window_rows=30000,
    )
    assert window_rows == 300, f'expected the fixed target_ranking_window_rows (300), got {window_rows}'
    print('test_preview_window_rows_behavioral_target_never_reads_column: PASS')


def test_preview_window_rows_neuron_target_matches_real_resolve():
    '''Cross-check against _resolve_target_ranking_window() itself (the
    function _preview_window_rows() is meant to preview) on a real
    synthetic periodic TSLS series -- the preview must agree with an
    independent direct call, not just avoid crashing.'''
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_mde_gmn_progress_preview_neuron')

    n = 6000
    period = 40
    vals = _periodic_tsls(n, period)
    df = pd.DataFrame({'CA1_1_tsls': vals})

    prediction_start, prediction_length, tp = 4000, 1000, 1
    warmup = 0
    usable_end = prediction_start + prediction_length - tp
    usable_df = df.iloc[warmup:usable_end].reset_index(drop=True)
    lib_ratio = (prediction_start - warmup) / max(1, usable_end - warmup)
    expected_window_rows, _ = module._resolve_target_ranking_window(
        'CA1_1_tsls', df, usable_df, lib_ratio,
        is_behavioral_target=False, target_ranking_window_rows=1000,
        min_pred_events=5, max_window_rows=30000,
    )

    preview_window_rows = module._preview_window_rows(
        'CA1_1_tsls', df, delay_set=[], prediction_start=prediction_start,
        prediction_length=prediction_length, tp=tp,
        is_behavioral_target=False, target_ranking_window_rows=1000,
        min_pred_events=5, max_window_rows=30000,
    )
    assert preview_window_rows == expected_window_rows, (
        f'preview ({preview_window_rows}) disagrees with the real resolve ({expected_window_rows})'
    )
    print(f'test_preview_window_rows_neuron_target_matches_real_resolve: PASS -- '
          f'window_rows={preview_window_rows}')


if __name__ == '__main__':
    test_log_progress_writes_timestamped_lines()
    test_log_progress_none_path_is_a_noop()
    test_format_eta_empty_inputs_return_empty_string()
    test_format_eta_flat_average()
    test_format_eta_per_window_row_rate_preferred_over_flat()
    test_preview_window_rows_behavioral_target_never_reads_column()
    test_preview_window_rows_neuron_target_matches_real_resolve()
    print('ALL PASS')
