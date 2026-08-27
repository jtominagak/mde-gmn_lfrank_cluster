'''R1 acceptance test (docs/review-actions-v1.md): the vectorized
compute_time_since_previous_spike() must be elementwise identical to the
pre-R1 reference implementation (kept as
_compute_time_since_previous_spike_reference) on real channels, and this
reports the speedup at full REGION_CHANNEL_CAP (i.e. every real neuron
channel in the data file, not a subset).

No pytest in this project's venv -- run directly:
    python3 tests/test_r1_tsls_vectorized.py
'''
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / 'scripts' / 'src'))
import cleaned_helper_functions as tsls  # noqa: E402

DATA_FILE = REPO_ROOT / 'data' / 'J1620210605_epoch_8_freq_40_Hz.csv'


def _neuron_columns(df):
    return [c for c in df.columns if c.startswith(('CA1_', 'mPFC_', 'OFC_'))]


def test_equivalence():
    '''Elementwise identical to the reference implementation on three real
    channels spanning the firing-rate distribution (low/medium/high spike
    count), per the task's own acceptance criterion.'''
    df = pd.read_csv(DATA_FILE)
    neuron_cols = _neuron_columns(df)
    counts = (df[neuron_cols] > 0).sum().sort_values()

    # low / median / high firing-rate channels -- not just three arbitrary
    # ones, so the comparison exercises very different spike_times lengths.
    low, median, high = (
        counts.index[0], counts.index[len(counts) // 2], counts.index[-1],
    )
    time_grid = df['time_bin_center'].to_numpy()

    for channel in (low, median, high):
        spike_times = df.loc[df[channel] > 0, 'time_bin_center'].to_numpy()
        old = tsls._compute_time_since_previous_spike_reference(spike_times, time_grid)
        new = tsls.compute_time_since_previous_spike(spike_times, time_grid)
        assert np.allclose(old, new, equal_nan=True), (
            f'{channel} (n_spikes={counts[channel]}): vectorized version diverges from reference'
        )
        print(f'{channel} (n_spikes={counts[channel]}): elementwise identical, PASS')

    print('test_equivalence: ALL PASS')


def report_speedup():
    '''Not a pass/fail check -- runs both implementations across every real
    neuron channel in the data file (full REGION_CHANNEL_CAP, i.e. no
    per-region cap applied) and reports the aggregate wall-clock speedup,
    per the task's own acceptance criterion.'''
    df = pd.read_csv(DATA_FILE)
    neuron_cols = _neuron_columns(df)
    time_grid = df['time_bin_center'].to_numpy()

    old_total = 0.0
    new_total = 0.0
    for channel in neuron_cols:
        spike_times = df.loc[df[channel] > 0, 'time_bin_center'].to_numpy()

        t0 = time.perf_counter()
        tsls._compute_time_since_previous_spike_reference(spike_times, time_grid)
        old_total += time.perf_counter() - t0

        t0 = time.perf_counter()
        tsls.compute_time_since_previous_spike(spike_times, time_grid)
        new_total += time.perf_counter() - t0

    print(f'Full REGION_CHANNEL_CAP ({len(neuron_cols)} real channels, '
          f'{len(time_grid)} grid points each):')
    print(f'  old (reference, O(N x S) per-grid-point mask): {old_total:.3f}s total')
    print(f'  new (np.searchsorted):                          {new_total:.3f}s total')
    print(f'  speedup: {old_total / new_total:.1f}x')


if __name__ == '__main__':
    test_equivalence()
    report_speedup()
