'''Fork-safety acceptance test (docs/no-kEDM-gpu-optimization-plan.md section 7a).

CUDA does not support fork(): forking a process that holds a CUDA context is
undefined behaviour, typically surfacing as "RuntimeError: Cannot re-initialize
CUDA in forked subprocess". PyTorch's own multiprocessing guidance requires
'spawn' (or 'forkserver') for CUDA in subprocesses. tsls-mde-gmn.py's two outer
ProcessPoolExecutor pools (Stage 1 / Stage 2 BFS waves) used a hardcoded
mp_context=fork, so any future torch/CUDA code reaching a worker would break.

The fix is deliberately CONDITIONAL rather than a blanket switch to spawn:
spawn re-imports this module per worker (measured ~18s -- pandas/numpy/
matplotlib/pyEDM/gmn), and the pools are recreated once per BFS wave, so
CPU-only runs would pay that repeatedly for a GPU path they never use.
GPU_ENABLED=False (the default) therefore keeps fork and today's exact
behaviour; GPU_ENABLED=True switches to spawn.

What this test checks:
  1. The default really is unchanged (GPU_ENABLED False -> 'fork').
  2. The flag really does flip it ('spawn').
  3. mp_start_method actually threads through build_network() ->
     build_nested_mde_network() rather than being accepted and ignored.
  4. A spawn-context ProcessPoolExecutor can genuinely run work that imports
     this module and round-trips the ARGUMENT TYPES the real submit() call
     passes (DataFrame, Path, dict, list, ndarray) -- i.e. spawn is actually
     viable here, not just configurable. This is the part that would catch an
     unpicklable-argument regression, which fork would silently tolerate.

No pytest in this project's venv -- run directly:
    python3 clean_scripts/tests/test_gpu_fork_safety.py
'''
import importlib
import inspect
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402

SCRATCH = Path(__file__).resolve().parent / '_scratch_gpu_fork_safety'


def _spawn_probe(payload):
    '''Runs INSIDE a spawn worker. Returns a summary of what survived the
    pickle round-trip. Must be module-level (spawn pickles by qualified name).'''
    df, path, mapping, seq, arr = payload
    return {
        'df_shape': tuple(df.shape),
        'path_ok': isinstance(path, Path),
        'mapping_ok': mapping == {'a': 1},
        'seq_ok': list(seq) == [1, 2, 3],
        'arr_sum': float(np.asarray(arr).sum()),
    }


def _check_config_flag(module):
    '''Default must stay fork; the flag must flip it to spawn.'''
    prev = os.environ.get('TSLS_MDE_GMN_GPU_ENABLED')
    results_dir = SCRATCH / 'cfg'
    results_dir.mkdir(parents=True, exist_ok=True)
    os.environ['TSLS_MDE_GMN_RESULTS_DIR'] = str(results_dir)
    os.environ['TSLS_MDE_GMN_PREV_RUNS_DIR'] = str(results_dir)
    try:
        os.environ.pop('TSLS_MDE_GMN_GPU_ENABLED', None)
        cfg = module.load_config()
        assert cfg['GPU_ENABLED'] is False, f"default GPU_ENABLED should be False, got {cfg['GPU_ENABLED']}"
        assert cfg['MP_START_METHOD'] == 'fork', (
            f"default start method must stay 'fork' (unchanged CPU behaviour), "
            f"got {cfg['MP_START_METHOD']!r}")

        os.environ['TSLS_MDE_GMN_GPU_ENABLED'] = '1'
        cfg = module.load_config()
        assert cfg['GPU_ENABLED'] is True, 'GPU_ENABLED=1 should enable the flag'
        assert cfg['MP_START_METHOD'] == 'spawn', (
            f"GPU_ENABLED=1 must select 'spawn' (CUDA cannot fork), "
            f"got {cfg['MP_START_METHOD']!r}")
    finally:
        if prev is None:
            os.environ.pop('TSLS_MDE_GMN_GPU_ENABLED', None)
        else:
            os.environ['TSLS_MDE_GMN_GPU_ENABLED'] = prev
    print('  config flag: default=fork, GPU_ENABLED=1 -> spawn  OK')


def _check_threading_through(module):
    '''mp_start_method must actually reach the pools, not be silently dropped
    somewhere in the call chain.'''
    for fname in ('build_network', 'build_nested_mde_network'):
        sig = inspect.signature(getattr(module, fname))
        assert 'mp_start_method' in sig.parameters, (
            f'{fname}() must accept mp_start_method')
        assert sig.parameters['mp_start_method'].default == 'fork', (
            f"{fname}()'s mp_start_method must default to 'fork' so existing "
            f"callers are unaffected, got {sig.parameters['mp_start_method'].default!r}")

    # Every worker pool is now opened in ONE place, waves.run_mde_batch(), so
    # the contract is checked there: the builder has to hand the method down,
    # and run_mde_batch() has to be the thing that applies it. (This used to
    # count two literal mp.get_context(mp_start_method) calls inside the
    # builder itself, back when Stage 1 and Stage 2 each opened their own.)
    waves = importlib.import_module('mde_gmn.waves')

    builder_src = inspect.getsource(module.build_nested_mde_network)
    assert "mp.get_context('fork')" not in builder_src, (
        'build_nested_mde_network() hardcodes fork somewhere -- the start '
        'method must come from mp_start_method')
    assert 'mp_start_method' in builder_src, (
        'build_nested_mde_network() never mentions mp_start_method -- it must '
        'pass it down to waves.run_mde_batch()')

    waves_src = inspect.getsource(waves.run_mde_batch)
    assert "mp.get_context('fork')" not in waves_src, (
        "waves.run_mde_batch() hardcodes fork -- it must use the "
        "mp_start_method it was given")
    assert 'mp.get_context(mp_start_method)' in waves_src, (
        'waves.run_mde_batch() must build its pool context from the passed-in '
        'mp_start_method')

    sig = inspect.signature(waves.run_mde_batch)
    assert 'mp_start_method' in sig.parameters, (
        'waves.run_mde_batch() must accept mp_start_method')

    # And the one pool that exists is the one it opens.
    assert waves_src.count('ProcessPoolExecutor(') == 1, (
        'waves.run_mde_batch() should open exactly one pool; every other MDE '
        'pool in the pipeline was consolidated into it')
    print('  threading-through: builder -> waves.run_mde_batch -> '
          'mp.get_context(mp_start_method), defaults are fork  OK')


def _check_spawn_actually_works():
    '''The part fork would silently let us get away with: prove a spawn worker
    can run and that the argument TYPES the real submit() passes survive the
    pickle round-trip spawn requires.'''
    df = pd.DataFrame({'Time': np.arange(1, 6), 'x': np.linspace(0, 1, 5)})
    payload = (df, Path('/tmp/example'), {'a': 1}, [1, 2, 3], np.array([1.0, 2.0, 3.5]))

    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
        got = ex.submit(_spawn_probe, payload).result(timeout=300)

    assert got['df_shape'] == (5, 2), f"DataFrame did not survive spawn: {got['df_shape']}"
    assert got['path_ok'], 'Path did not survive spawn'
    assert got['mapping_ok'], 'dict did not survive spawn'
    assert got['seq_ok'], 'list did not survive spawn'
    assert abs(got['arr_sum'] - 6.5) < 1e-9, f"ndarray did not survive spawn: {got['arr_sum']}"
    print('  spawn worker: ran, and DataFrame/Path/dict/list/ndarray round-tripped  OK')


def test_tsls():
    module = load_pipeline_module('run_serial_mde.py', '_test_tsls_gpu_fork_safety')
    _check_config_flag(module)
    _check_threading_through(module)
    _check_spawn_actually_works()


if __name__ == '__main__':
    SCRATCH.mkdir(parents=True, exist_ok=True)
    test_tsls()
    print('ALL PASS')
