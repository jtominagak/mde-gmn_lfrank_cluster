'''GPU CCM-gate tests (docs/no-kEDM-gpu-optimization-plan.md section 8.5).

Covers all four items from that section, scoped to proper_ccm's own
use_gpu=... keyword (cleaned_helper_functions.py) and the new
scripts/src/gpu_edm.py primitive it calls:

  1. Equivalence -- proper_ccm run twice with the SAME seed, once
     use_gpu=False and once use_gpu=True, must produce matching c2t/t2c
     arrays. THE important test (see the task instructions this was
     written against): this is what would catch a silent divergence in
     which columns MDE ends up selecting, not just a crash. Skips
     cleanly (not fail) when no CUDA device is visible, so it's runnable
     on Deigo -- it can only be MEANINGFULLY exercised on real GPU
     hardware (Saion).
  2. Fallback -- use_gpu=True on a machine with NO GPU must produce
     results IDENTICAL (not just close) to use_gpu=False, no exception.
     This is the inverse precondition of test 1: it only means something
     on a machine that genuinely has no GPU, so it skips there instead.
  3. Reproducibility -- proper_ccm draws every rng sample up front,
     before any compute (see its own docstring), specifically so a given
     seed reproduces identical library draws regardless of `pool`/
     `use_gpu`. Checked here by literally recording every
     numpy.random.Generator.choice() call proper_ccm makes, not just by
     comparing final scores -- this is a stronger check and runs on any
     machine (it only requires that use_gpu doesn't change which numbers
     get drawn, not that a GPU is present).
  4. The existing suite staying green under GPU_ENABLED=False is
     verified separately (tests/test_gpu_fork_safety.py plus this
     project's other tests) -- nothing GPU-specific to add here.

No pytest in this project's venv -- run directly:
    .venv/bin/python3 tests/test_gpu_ccm_equivalence.py

Note: cleaned_helper_functions.py imports pandas/numpy/pyEDM at module
scope, so importing it below takes a few seconds -- expected, not a hang.
'''
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent.parent / 'scripts' / 'src'
sys.path.insert(0, str(SRC_DIR))
import cleaned_helper_functions as tsls  # noqa: E402
import gpu_edm  # noqa: E402


def _make_ccm_dataframe(M=800, seed=0):
    '''Synthetic two-column dataFrame for proper_ccm, same production
    scale (M=800, matching CCM_WINDOW_ROWS) as
    scripts/gpu_prototypes/validate_batched_ccm.py's own benchmark data.
    `y` is built from a lagged, noisy copy of `x` so the CCM gate has a
    genuine (if weak) causal link to detect -- pure independent noise
    would make every cross-map prediction equally bad and wouldn't
    exercise the weighting logic meaningfully. Deterministic given
    `seed`, independent of proper_ccm's own `seed` argument (that one
    controls the library draws; this one only controls the input data).'''
    rng = np.random.default_rng(seed)
    t = np.arange(M)
    x = np.sin(np.linspace(0, 40, M)) + rng.normal(0, 0.1, M)
    y = np.roll(x, 3) + rng.normal(0, 0.1, M)
    return pd.DataFrame({'Time': t, 'x_tsls': x, 'y_tsls': y})


# 'correlation' (Pearson rho, cleaned_helper_functions._score) deliberately
# chosen over 'f1'/'gamma' for these tests: those event-timing metrics
# threshold a TSLS-shaped (sawtooth) signal, which this synthetic sine-wave
# data isn't, and thresholding is exactly the kind of step where a tiny
# float32-vs-float64 prediction difference could flip a boolean outcome for
# reasons unrelated to whether the GPU port is correct. 'correlation' is a
# smooth function of the raw predictions, so it isolates the thing these
# tests actually care about: do the predictions themselves match.
CCM_KWARGS = dict(
    libSizes=[16, 24, 40, 64, 160, 400, 800],
    sample=20, tau=-1, Tp=1, E=3,
    target='x_tsls', columns='y_tsls',
    target_measurement='correlation', columns_measurement='correlation',
    exclusionRadius=2, gammaDelta=2,
)


def test_equivalence_gpu_vs_cpu():
    if not gpu_edm.is_available():
        print('  SKIP (no CUDA device visible): equivalence can only be '
              'meaningfully checked against a real GPU run')
        return
    df = _make_ccm_dataframe()
    c2t_cpu, t2c_cpu, libSizes_cpu = tsls.proper_ccm(df, seed=42, use_gpu=False, **CCM_KWARGS)
    c2t_gpu, t2c_gpu, libSizes_gpu = tsls.proper_ccm(df, seed=42, use_gpu=True, **CCM_KWARGS)

    assert libSizes_cpu == libSizes_gpu, (
        f'libSizes must be identical regardless of use_gpu: {libSizes_cpu} vs {libSizes_gpu}')
    np.testing.assert_allclose(
        c2t_cpu, c2t_gpu, atol=2e-3, equal_nan=True,
        err_msg='c2t diverged between use_gpu=False and use_gpu=True')
    np.testing.assert_allclose(
        t2c_cpu, t2c_gpu, atol=2e-3, equal_nan=True,
        err_msg='t2c diverged between use_gpu=False and use_gpu=True')
    print('  equivalence (GPU vs CPU, same seed): c2t/t2c match within tolerance  OK')


def test_fallback_use_gpu_true_no_gpu_matches_cpu():
    if gpu_edm.is_available():
        print('  SKIP (a GPU IS visible here): "fallback when absent" is not '
              'the scenario this machine can test -- see test_equivalence_gpu_vs_cpu')
        return
    df = _make_ccm_dataframe()
    c2t_a, t2c_a, libSizes_a = tsls.proper_ccm(df, seed=7, use_gpu=False, **CCM_KWARGS)
    c2t_b, t2c_b, libSizes_b = tsls.proper_ccm(df, seed=7, use_gpu=True, **CCM_KWARGS)

    assert libSizes_a == libSizes_b
    np.testing.assert_array_equal(c2t_a, c2t_b)
    np.testing.assert_array_equal(t2c_a, t2c_b)
    print('  fallback (use_gpu=True, no GPU present): byte-identical to use_gpu=False  OK')


def _recorded_choice_draws(fn, *args, **kwargs):
    '''Runs fn(*args, **kwargs) while recording every
    numpy.random.Generator.choice() call made anywhere during it, in call
    order. numpy.random.Generator is a Cython extension type -- its
    methods can't be monkeypatched directly ("cannot set 'choice'
    attribute of immutable type") -- so this patches
    numpy.random.default_rng (an ordinary Python function) to hand back a
    thin recording wrapper instead of the real Generator, delegating
    every other attribute straight through.'''
    draws = []
    orig_default_rng = np.random.default_rng

    class _RecordingRNG:
        def __init__(self, real):
            object.__setattr__(self, '_real', real)

        def choice(self, *a, **k):
            result = self._real.choice(*a, **k)
            draws.append(np.array(result, copy=True))
            return result

        def __getattr__(self, name):
            return getattr(self._real, name)

    def patched_default_rng(seed=None):
        return _RecordingRNG(orig_default_rng(seed))

    np.random.default_rng = patched_default_rng
    try:
        fn(*args, **kwargs)
    finally:
        np.random.default_rng = orig_default_rng
    return draws


def test_reproducibility_draws_independent_of_use_gpu():
    '''Stronger than comparing final scores: proves use_gpu doesn't change
    WHICH library indices get drawn, regardless of whether a GPU is
    actually present (the draw loop runs unconditionally, before the
    use_gpu branch -- this test would catch a regression that moved a
    draw inside that branch even on a machine with no GPU to exercise the
    branch's own compute path).'''
    df = _make_ccm_dataframe()
    draws_a = _recorded_choice_draws(tsls.proper_ccm, df, seed=99, use_gpu=False, **CCM_KWARGS)
    draws_b = _recorded_choice_draws(tsls.proper_ccm, df, seed=99, use_gpu=True, **CCM_KWARGS)

    assert len(draws_a) > 0, 'expected at least one rng.choice() draw -- test setup is broken'
    assert len(draws_a) == len(draws_b), (
        f'use_gpu changed the NUMBER of rng.choice() calls: {len(draws_a)} vs {len(draws_b)}')
    for i, (a, b) in enumerate(zip(draws_a, draws_b)):
        np.testing.assert_array_equal(a, b, err_msg=f'draw #{i} differs between use_gpu=False/True')
    print(f'  reproducibility: all {len(draws_a)} rng.choice() draws identical '
          f'regardless of use_gpu  OK')


if __name__ == '__main__':
    test_equivalence_gpu_vs_cpu()
    test_fallback_use_gpu_true_no_gpu_matches_cpu()
    test_reproducibility_draws_independent_of_use_gpu()
    print('ALL PASS')
