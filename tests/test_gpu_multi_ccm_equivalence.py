'''GPU multi-target CCM-gate tests (docs/no-kEDM-gpu-optimization-plan.md
section 9), the GMN_MODE='combined' counterpart to
tests/test_gpu_ccm_equivalence.py's proper_ccm tests -- same three checks,
now against multi_proper_ccm(use_gpu=...):

  1. Equivalence -- multi_proper_ccm run twice with the SAME seed, once
     use_gpu=False and once use_gpu=True, must produce matching c2t/t2c
     arrays. Run against BOTH target_select_E=None (use_full_pool=True --
     both the fwd AND rev directions get GPU-batched) and an int
     target_select_E (use_full_pool=False -- fwd batched, rev stays on
     CPU per plan section 9.2 item 3), since those are genuinely different
     code paths inside multi_proper_ccm and both need real-hardware
     coverage. Skips cleanly (not fail) when no CUDA device is visible.
  2. Fallback -- use_gpu=True on a machine with NO GPU must produce
     results IDENTICAL to use_gpu=False, no exception. Skips instead on a
     machine that DOES have a GPU (see test_gpu_ccm_equivalence.py's own
     identical reasoning for why these two tests are complementary).
  3. Reproducibility -- multi_proper_ccm's rng draws are NOT collected up
     front the way proper_ccm's are (plan section 9.2 item 1); the actual
     restructuring work here was making sure that doesn't change draw
     order/count. Checked by recording every numpy.random.Generator
     .choice() call (both the fwd rng_fwd and rev rng_rev streams, across
     every libSize's freshly-created Generators) and asserting the
     recorded sequence is identical between use_gpu=False and True. Runs
     on any machine -- the draw loop is unconditional, before the
     use_gpu branch.

No pytest in this project's venv -- run directly:
    .venv/bin/python3 tests/test_gpu_multi_ccm_equivalence.py

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


def _make_multi_ccm_dataframe(M=800, n_targets=2, seed=0):
    '''Synthetic (source, targets...) dataFrame, production scale (M=800,
    matching CCM_WINDOW_ROWS) same spirit as test_gpu_ccm_equivalence.py's
    own _make_ccm_dataframe -- each target is a distinctly-lagged, noisy
    copy of the source so multi_proper_ccm's fwd direction (source
    predicting EACH target separately) has a genuine, but per-target
    distinguishable, signal.'''
    rng = np.random.default_rng(seed)
    t = np.arange(M)
    src = np.sin(np.linspace(0, 40, M)) + rng.normal(0, 0.1, M)
    data = {'Time': t, 'src_tsls': src}
    for j in range(n_targets):
        data[f'tgt{j}_tsls'] = np.roll(src, 3 + j) + rng.normal(0, 0.1, M)
    return pd.DataFrame(data)


# 'correlation' chosen deliberately over 'f1'/'gamma' -- see
# test_gpu_ccm_equivalence.py's own CCM_KWARGS comment for the full
# reasoning (event-timing metrics threshold a TSLS-shaped signal this
# synthetic sine-wave data isn't, which could flip a boolean outcome on a
# tiny float32-vs-float64 difference for reasons unrelated to GPU-port
# correctness).
BASE_KWARGS = dict(
    libSizes=[16, 24, 40, 64, 160, 400, 800],
    sample=20, tau=-1, Tp=1, E=3,
    columns='src_tsls',
    target_measurement='correlation', columns_measurement='correlation',
    exclusionRadius=2, gammaDelta=2,
)


def _run_pair(df, targets, seed, target_aggregate, target_select_E):
    kwargs = dict(
        BASE_KWARGS, targets=targets, seed=seed,
        target_aggregate=target_aggregate, target_select_E=target_select_E,
    )
    cpu = tsls.multi_proper_ccm(df, use_gpu=False, **kwargs)
    gpu = tsls.multi_proper_ccm(df, use_gpu=True, **kwargs)
    return cpu, gpu


def test_equivalence_gpu_vs_cpu():
    if not gpu_edm.is_available():
        print('  SKIP (no CUDA device visible): equivalence can only be '
              'meaningfully checked against a real GPU run')
        return

    df = _make_multi_ccm_dataframe(n_targets=2)
    targets = ['tgt0_tsls', 'tgt1_tsls']

    # use_full_pool=True (target_select_E=None): both fwd AND rev get
    # GPU-batched -- the fully-accelerated path.
    (c2t_a, t2c_a, ls_a), (c2t_b, t2c_b, ls_b) = _run_pair(
        df, targets, seed=42, target_aggregate='mean', target_select_E=None)
    assert ls_a == ls_b
    np.testing.assert_allclose(c2t_a, c2t_b, atol=2e-3, equal_nan=True,
                                err_msg='c2t diverged (use_full_pool=True)')
    np.testing.assert_allclose(t2c_a, t2c_b, atol=2e-3, equal_nan=True,
                                err_msg='t2c diverged (use_full_pool=True)')
    print('  equivalence (use_full_pool=True, fwd+rev both GPU-batched): OK')

    # use_full_pool=False (int target_select_E): fwd GPU-batched, rev
    # stays on CPU (plan section 9.2 item 3) -- a genuinely different code
    # path, needs its own hardware coverage.
    (c2t_c, t2c_c, ls_c), (c2t_d, t2c_d, ls_d) = _run_pair(
        df, targets, seed=42, target_aggregate='min', target_select_E=3)
    assert ls_c == ls_d
    np.testing.assert_allclose(c2t_c, c2t_d, atol=2e-3, equal_nan=True,
                                err_msg='c2t diverged (use_full_pool=False)')
    # t2c must be BYTE-IDENTICAL here, not just close -- the rev direction
    # never touches the GPU path when use_full_pool=False, regardless of
    # use_gpu, so both runs computed it via the exact same CPU code.
    np.testing.assert_array_equal(
        t2c_c, t2c_d, err_msg='t2c changed even though rev never leaves the CPU path')
    print('  equivalence (use_full_pool=False, fwd GPU-batched / rev CPU-only): OK')


def test_fallback_use_gpu_true_no_gpu_matches_cpu():
    if gpu_edm.is_available():
        print('  SKIP (a GPU IS visible here): "fallback when absent" is not '
              'the scenario this machine can test -- see test_equivalence_gpu_vs_cpu')
        return
    df = _make_multi_ccm_dataframe(n_targets=2)
    (c2t_a, t2c_a, ls_a), (c2t_b, t2c_b, ls_b) = _run_pair(
        df, ['tgt0_tsls', 'tgt1_tsls'], seed=7, target_aggregate='mean', target_select_E=None)

    assert ls_a == ls_b
    np.testing.assert_array_equal(c2t_a, c2t_b)
    np.testing.assert_array_equal(t2c_a, t2c_b)
    print('  fallback (use_gpu=True, no GPU present): byte-identical to use_gpu=False  OK')


def _recorded_choice_draws(fn, *args, **kwargs):
    '''Same technique as test_gpu_ccm_equivalence.py's own helper --
    numpy.random.Generator.choice can't be monkeypatched directly (it's a
    Cython extension type method), so this patches numpy.random.default_rng
    (an ordinary Python function) to hand back a thin recording wrapper
    instead of the real Generator. multi_proper_ccm creates TWO Generators
    per libSize (rng_fwd, rng_rev, both via default_rng) -- patching the
    factory itself means both get wrapped automatically, in true creation/
    usage order, with no need to know that structure from the caller side.'''
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
    '''Stronger than comparing final scores (plan section 9.3): proves
    use_gpu doesn't change WHICH library indices get drawn -- for BOTH the
    fwd (rng_fwd) and rev (rng_rev) streams, across every libSize -- since
    section 9.2 item 1 (the draws are not collected up front the way
    proper_ccm's are, and had to be carefully preserved during the
    restructuring this work required) is the likeliest way a refactor like
    this silently breaks reproducibility. Covers both use_full_pool True
    and False, since the rev direction's draw shape differs between them
    (an extra rng_rev.choice() for the random col_idx draw when False and
    n_extra>0).'''
    df = _make_multi_ccm_dataframe(n_targets=2)
    targets = ['tgt0_tsls', 'tgt1_tsls']

    for label, target_select_E in [('use_full_pool=True', None), ('use_full_pool=False', 3)]:
        kwargs = dict(BASE_KWARGS, targets=targets, seed=99,
                       target_aggregate='mean', target_select_E=target_select_E)
        draws_a = _recorded_choice_draws(tsls.multi_proper_ccm, df, use_gpu=False, **kwargs)
        draws_b = _recorded_choice_draws(tsls.multi_proper_ccm, df, use_gpu=True, **kwargs)

        assert len(draws_a) > 0, f'[{label}] expected at least one rng.choice() draw'
        assert len(draws_a) == len(draws_b), (
            f'[{label}] use_gpu changed the NUMBER of rng.choice() calls: '
            f'{len(draws_a)} vs {len(draws_b)}')
        for i, (a, b) in enumerate(zip(draws_a, draws_b)):
            np.testing.assert_array_equal(
                a, b, err_msg=f'[{label}] draw #{i} differs between use_gpu=False/True')
        print(f'  reproducibility [{label}]: all {len(draws_a)} rng.choice() draws '
              f'identical regardless of use_gpu  OK')


if __name__ == '__main__':
    test_equivalence_gpu_vs_cpu()
    test_fallback_use_gpu_true_no_gpu_matches_cpu()
    test_reproducibility_draws_independent_of_use_gpu()
    print('ALL PASS')
