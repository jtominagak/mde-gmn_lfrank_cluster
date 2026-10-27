'''Test for the window-sizing fix found and implemented while verifying
R22 (docs/review-actions-v1.md) -- not one of the 22 numbered tasks itself,
see docs/review-findings.md's own window-sizing section for the full
real-data investigation and design (including the two abandoned designs
this file's own history reflects: a first pred-portion-only check against
`min_spike_events_fraction` found mathematically unreachable, then a
whole-window check against the same fraction found on real data to pass
while the pred portion itself still lacked enough events -- both replaced
by the `MIN_PRED_EVENTS`/`MAX_WINDOW_ROWS` design tested here).

Covers `_resolve_target_ranking_window()` directly (tsls-mde-gmn.py):
- behavioral targets: fixed `target_ranking_window_rows`, capped by
  available data, no event-count concept.
- neuron targets: extended until the window's own PRED portion (the last
  `1 - lib_ratio` fraction of ROWS -- the part `f1`-scoring actually
  touches) contains `min_pred_events` real reset events -- checked against
  a synthetic, perfectly periodic TSLS-like series where the ground truth
  is computable by construction (a value that resets to 0 every `period`
  rows is a real TSLS reset event at every reset row, confirmed by
  cross-checking against `detect_reset_events_from_tsls()` itself via an
  independent, non-vectorized per-window count rather than assumed or
  re-run through the same vectorized machinery the implementation uses).
- the "cannot be met" case, two distinct reasons per the redesign:
  (a) a neuron whose only real firing activity sits entirely outside the
      leakage-safe range (usable_df) -- unreachable regardless of
      `max_window_rows`.
  (b) a neuron that WOULD reach `min_pred_events` given enough rows, but
      `max_window_rows` (the compute ceiling) is set below what's needed --
      unreachable because of the cap, not the data.
  Both must return `(None, pred_event_count)` -- never silently resolve
  some window anyway -- with `pred_event_count` the best count achieved at
  ANY window size tried within the cap (not necessarily at the cap itself,
  since the pred portion's own event count is not guaranteed monotonic in
  window size -- see `_resolve_target_ranking_window()`'s own docstring).

No pytest in this project's venv -- run directly:
    python3 clean_scripts/tests/test_window_sizing.py
'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402


def _periodic_tsls(n, period, start=0):
    '''A perfectly periodic TSLS-like sawtooth: value = (row - start) % period
    for row >= start (a real reset -- drop to 0 -- every `period` rows),
    held at a large constant (`period`, never near zero) for row < start,
    so the series has NO detectable reset events before `start`.'''
    vals = np.full(n, float(period), dtype=float)
    idx = np.arange(start, n)
    vals[start:] = (idx - start) % period
    return vals


def _independent_pred_event_count(tsls, vals, w, lib_ratio):
    '''Ground truth for one window size w, via a plain per-call
    detect_reset_events_from_tsls + boolean filter -- deliberately NOT
    reusing the implementation's own up-front-detect-then-searchsorted
    approach, so this is a genuinely independent check, not a restatement
    of the same code path.'''
    thr = tsls.infer_near_zero_threshold_from_true_tsls(vals)
    events = tsls.detect_reset_events_from_tsls(
        vals[:w], near_zero_threshold=thr, drop_threshold=0.0)
    pred_start = round(w * lib_ratio)
    return int(np.sum((events >= pred_start) & (events < w)))


def test_behavioral_target_is_fixed_window():
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_mde_gmn_window_behavioral')
    df = pd.DataFrame({'head_position_x': np.arange(5000, dtype=float)})
    usable_df = df.iloc[:3000].reset_index(drop=True)

    window_rows, pred_event_count = module._resolve_target_ranking_window(
        'head_position_x', df, usable_df, lib_ratio=0.7,
        is_behavioral_target=True, target_ranking_window_rows=1000,
        min_pred_events=20, max_window_rows=30000,
    )
    assert pred_event_count is None, 'behavioral target should have no event-count concept'
    assert window_rows == 1000, f'expected the fixed constant 1000, got {window_rows}'

    # Capped by available data when the constant exceeds it.
    small_usable_df = df.iloc[:200].reset_index(drop=True)
    window_rows2, _ = module._resolve_target_ranking_window(
        'head_position_x', df, small_usable_df, lib_ratio=0.7,
        is_behavioral_target=True, target_ranking_window_rows=1000,
        min_pred_events=20, max_window_rows=30000,
    )
    assert window_rows2 == 200, f'expected capped at available data (200), got {window_rows2}'
    print('test_behavioral_target_is_fixed_window: PASS')


def test_neuron_target_extends_to_min_pred_events():
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_mde_gmn_window_neuron')
    tsls = module.edm

    n = 6000
    period = 40  # a real reset every 40 rows
    vals = _periodic_tsls(n, period)
    df = pd.DataFrame({'CA1_1_tsls': vals})
    usable_df = df.iloc[:n].reset_index(drop=True)
    lib_ratio = 0.7
    min_pred_events = 3

    window_rows, pred_event_count = module._resolve_target_ranking_window(
        'CA1_1_tsls', df, usable_df, lib_ratio,
        is_behavioral_target=False, target_ranking_window_rows=1000,
        min_pred_events=min_pred_events, max_window_rows=n,
    )
    assert window_rows is not None, 'expected a resolvable window for a steadily-firing neuron'

    # Cross-check against an independent, non-vectorized count -- both the
    # returned pred_event_count and the window's own claimed sufficiency.
    achieved = _independent_pred_event_count(tsls, vals, window_rows, lib_ratio)
    assert achieved == pred_event_count, (
        f'implementation reported pred_event_count={pred_event_count}, '
        f'independent count says {achieved}'
    )
    assert achieved >= min_pred_events, (
        f'resolved window claims to satisfy min_pred_events={min_pred_events} but the pred '
        f'portion only has {achieved} events'
    )

    # Minimality: one row less should NOT satisfy it (confirms the search
    # finds the SMALLEST passing window, not an arbitrarily larger one).
    achieved_smaller = _independent_pred_event_count(tsls, vals, window_rows - 1, lib_ratio)
    assert achieved_smaller < min_pred_events, (
        f'window_rows={window_rows} is not minimal -- window_rows-1 already satisfies '
        f'min_pred_events too ({achieved_smaller} >= {min_pred_events})'
    )
    print(f'test_neuron_target_extends_to_min_pred_events: PASS -- '
          f'min_pred_events={min_pred_events}, window_rows={window_rows}, '
          f'pred_event_count={pred_event_count}')


def test_neuron_target_unreachable_data():
    '''A neuron whose only real firing activity sits entirely outside the
    leakage-safe range (usable_df) -- unreachable regardless of
    max_window_rows, must return (None, pred_event_count), not silently
    resolve some window anyway.'''
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_mde_gmn_window_unreachable_data')

    n = 3000
    fire_start = 2000  # all real firing activity is in [2000, 3000) --
    period = 10         # entirely past where usable_df below ends.
    vals = _periodic_tsls(n, period, start=fire_start)
    df = pd.DataFrame({'CA1_dead_tsls': vals})
    usable_df = df.iloc[:1500].reset_index(drop=True)  # ends before any firing starts

    window_rows, pred_event_count = module._resolve_target_ranking_window(
        'CA1_dead_tsls', df, usable_df, lib_ratio=0.7,
        is_behavioral_target=False, target_ranking_window_rows=1000,
        min_pred_events=5, max_window_rows=30000,
    )
    assert window_rows is None, (
        f'expected None (unreachable within the leakage-safe range), got window_rows={window_rows}'
    )
    assert pred_event_count == 0, f'expected 0 events ever seen, got {pred_event_count}'
    print(f'test_neuron_target_unreachable_data: PASS -- pred_event_count={pred_event_count}')


def test_neuron_target_unreachable_compute_cap():
    '''A neuron that WOULD reach min_pred_events given enough rows -- the
    data itself is not the obstacle -- but max_window_rows (the compute
    ceiling) is set below what's needed. Must still return (None, ...),
    distinguishing "capped by compute budget" from the data-unreachable
    case above only in the caller's own log message, not in this
    function's return contract.'''
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_mde_gmn_window_unreachable_cap')
    tsls = module.edm

    n = 20000
    period = 200  # slow-firing but steady -- plenty of events over the full recording
    vals = _periodic_tsls(n, period)
    df = pd.DataFrame({'CA1_slow_tsls': vals})
    usable_df = df.iloc[:n].reset_index(drop=True)
    lib_ratio = 0.7
    min_pred_events = 5
    max_window_rows = 500  # deliberately too small to accumulate 5 pred-portion events

    # Confirm the premise: min_pred_events IS reachable given the full
    # recording (no cap), so the cap below is what causes the failure, not
    # the underlying data.
    window_rows_uncapped, _ = module._resolve_target_ranking_window(
        'CA1_slow_tsls', df, usable_df, lib_ratio,
        is_behavioral_target=False, target_ranking_window_rows=1000,
        min_pred_events=min_pred_events, max_window_rows=n,
    )
    assert window_rows_uncapped is not None, 'test premise failed: should be reachable uncapped'
    assert window_rows_uncapped > max_window_rows, (
        f'test premise failed: uncapped resolution ({window_rows_uncapped} rows) should '
        f'exceed the deliberately-too-small cap ({max_window_rows}) for this test to be meaningful'
    )

    window_rows, pred_event_count = module._resolve_target_ranking_window(
        'CA1_slow_tsls', df, usable_df, lib_ratio,
        is_behavioral_target=False, target_ranking_window_rows=1000,
        min_pred_events=min_pred_events, max_window_rows=max_window_rows,
    )
    assert window_rows is None, (
        f'expected None (capped by max_window_rows={max_window_rows}), got window_rows={window_rows}'
    )
    assert 0 <= pred_event_count < min_pred_events

    # Cross-check pred_event_count is the best seen across [1, cap], not
    # just at the cap itself (the two can differ -- see the implementation's
    # own docstring on non-monotonicity).
    best_independent = max(
        _independent_pred_event_count(tsls, vals, w, lib_ratio)
        for w in range(1, max_window_rows + 1)
    )
    assert pred_event_count == best_independent, (
        f'implementation reported best pred_event_count={pred_event_count}, '
        f'independent search over every window size up to the cap says {best_independent}'
    )
    print(f'test_neuron_target_unreachable_compute_cap: PASS -- '
          f'uncapped would need {window_rows_uncapped} rows, capped at {max_window_rows}, '
          f'best pred_event_count={pred_event_count}')


if __name__ == '__main__':
    test_behavioral_target_is_fixed_window()
    test_neuron_target_extends_to_min_pred_events()
    test_neuron_target_unreachable_data()
    test_neuron_target_unreachable_compute_cap()
    print('ALL PASS')
