'''The TSLS event metric must be scorable against a CONTINUOUS prediction.

An observed TSLS series resets to exactly 0.0, so
infer_near_zero_threshold_from_true_tsls() returns exactly 0.0 for every
channel of this dataset. The old scorer applied that same threshold to the
PREDICTED series, requiring `pred <= 0.0` -- a measure-zero condition for a
continuous regressor. Consequences, all measured on job 4799018:

  * 38 of 39 simulated nodes produced ZERO predicted events and so scored
    f1 = 0 regardless of prediction quality. The one exception scored
    nonzero only because its trajectory dipped to -0.017.
  * On the selection side the test fires only where a Simplex neighbourhood
    is all-zero; just 5 of 22 such hits on CA1_44_tsls coincided with a real
    reset, and against circular-shift surrogates the real driver set beat
    all of its own surrogates in only 2 of 6 cases -- i.e. chance.
  * f1 returned a hard 0 whenever precision or recall was undefined, so 9
    real-data passthrough nodes scored f1 = 0.000 against a BIT-IDENTICAL
    copy of themselves, indistinguishable in the log from a wrong answer.

This pins the three fixes: a relative prediction-side tolerance, the
NaN/0 split, and that a perfect prediction now actually scores 1.0.

No pytest in this project's venv -- run directly:
    python3 tests/test_tsls_event_metric.py
'''
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'src'))
import cleaned_helper_functions as tsls  # noqa: E402


def _sawtooth(n=600, period=20, amp=None):
    '''A TSLS series: ramps up by one bin per step, resets to exactly 0.'''
    x = np.array([(i % period) * 0.025 for i in range(n)], dtype=float)
    return x


def test_observed_detector_unchanged():
    '''The ground-truth side must keep its exact-zero semantics.'''
    x = _sawtooth()
    thr = tsls.infer_near_zero_threshold_from_true_tsls(x)
    assert thr == 0.0, f'a series resetting to 0 must infer threshold 0.0, got {thr}'
    ev = tsls.detect_reset_events_from_tsls(x, near_zero_threshold=thr)
    # 30 reset bins, less the one at index 0 -- np.diff(prepend=x[0]) makes the
    # first sample a non-drop by construction.
    assert len(ev) == 29, f'expected 29 detected resets, got {len(ev)}'
    assert all(x[i] == 0.0 for i in ev), 'every detected true event must sit at exactly 0'
    print(f'PASS: observed-side detector unchanged ({len(ev)} resets at exactly 0.0)')


def test_continuous_prediction_is_scorable():
    '''The core fix: a near-perfect but continuous prediction must not score 0.'''
    x = _sawtooth()
    # A good prediction: the same shape, but it bottoms out slightly ABOVE 0
    # and is smoothed -- exactly what an S-map/Simplex output looks like.
    # Bottoms out at 0.008 -- inside the relative tolerance (10% of this
    # series' median, ~0.024) but never at or below 0.0.
    pred = np.clip(x, 0.005, None) + 0.003
    old_events = tsls.detect_reset_events_from_tsls(pred, near_zero_threshold=0.0)
    assert len(old_events) == 0, (
        'precondition: the exact-zero test must find NOTHING in this prediction '
        f'(that is the bug), got {len(old_events)}')
    new_events = tsls.detect_reset_events_from_prediction(pred, x, near_zero_threshold=0.0)
    assert len(new_events) > 0, (
        'the prediction-side detector must find the resets the exact-zero test missed')
    f1 = tsls._score(x, pred, 'f1')
    assert f1 > 0.5, f'a near-perfect continuous prediction must score well, got f1={f1}'
    print(f'PASS: continuous prediction scores f1={f1:.3f} '
          f'(exact-zero test found 0 events, new detector found {len(new_events)})')


def test_tolerance_is_relative_to_scale():
    '''A fixed absolute tolerance cannot serve a 4 Hz and a 0.05 Hz cell.'''
    fast = _sawtooth(period=4)      # median TSLS ~0.04
    slow = _sawtooth(period=200)    # median TSLS ~2.5
    f_ev = tsls.detect_reset_events_from_prediction(
        np.clip(fast, 0.002, None), fast, near_zero_threshold=0.0)
    s_ev = tsls.detect_reset_events_from_prediction(
        np.clip(slow, 0.12, None), slow, near_zero_threshold=0.0)
    assert len(f_ev) > 0, 'a fast cell must still be scorable at its own small scale'
    assert len(s_ev) > 0, 'a slow cell must be scorable at its own much larger scale'
    # The slow cell's tolerance must be the larger one, in absolute terms.
    assert np.median(slow) > np.median(fast) * 10
    print(f'PASS: tolerance scales with the series ({len(f_ev)} fast events, '
          f'{len(s_ev)} slow events)')


def test_identical_series_never_scores_zero():
    '''Comparing a series to itself must never look like a wrong answer.'''
    x = _sawtooth()
    f1 = tsls._score(x, x.copy(), 'f1')
    assert f1 == 1.0, f'a bit-identical copy must score f1=1.0, got {f1}'
    # The job-4799018 case: an identical copy of a window holding NO event.
    flat = np.linspace(1.0, 2.0, 400)       # rises throughout, never resets
    thr = tsls.infer_near_zero_threshold_from_true_tsls(flat)
    assert len(tsls.detect_reset_events_from_tsls(flat, near_zero_threshold=thr)) == 0
    f1_flat = tsls._score(flat, flat.copy(), 'f1')
    assert np.isnan(f1_flat), (
        'an eventless window has UNDEFINED event-timing skill, not zero -- this is the '
        f'passthrough-scores-0.000 bug, got {f1_flat}')
    print('PASS: identical series -> f1=1.0, and eventless window -> NaN (not 0)')


def test_genuine_miss_still_scores_zero():
    '''NaN must not become a way to excuse a predictor that produced nothing.'''
    x = _sawtooth()
    nothing = np.linspace(5.0, 6.0, len(x))     # no resets at all, but x HAS them
    f1 = tsls._score(x, nothing, 'f1')
    assert f1 == 0.0, (
        f'real events existed and none were predicted: that is a miss, scoring 0, got {f1}')
    assert not np.isnan(f1)
    print('PASS: real events + no predicted events -> 0.0, not NaN')


def test_default_tolerance_is_the_calibrated_one():
    assert tsls.PRED_EVENT_TOL_FRAC == 0.1, (
        'the default prediction tolerance is the surrogate-calibrated 0.1; changing it '
        'changes every f1 in the pipeline, so change it deliberately and re-run the '
        'calibration in detect_reset_events_from_prediction()')
    print(f'PASS: default prediction tolerance is {tsls.PRED_EVENT_TOL_FRAC}')


def main():
    test_observed_detector_unchanged()
    test_continuous_prediction_is_scorable()
    test_tolerance_is_relative_to_scale()
    test_identical_series_never_scores_zero()
    test_genuine_miss_still_scores_zero()
    test_default_tolerance_is_the_calibrated_one()
    print('ALL PASS')


if __name__ == '__main__':
    main()
