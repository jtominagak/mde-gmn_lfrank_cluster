'''Shared helpers for loading scripts/src/*.py (hyphenated filenames, not
importable via a plain `import` statement) as modules for testing, and for
faking run_mde_for_target() so build_nested_mde_network()'s BFS/recursion
logic can be exercised without real MDE/CCM computation or real data.'''
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / 'scripts' / 'src'


def load_pipeline_module(filename, module_name):
    '''Loads scripts/src/<filename> as a fresh module named `module_name`
    (must be unique per test process -- re-loading under the same name
    would reuse the cached sys.modules entry). Safe to call for
    tsls-mde-gmn.py / simple-mde-gmn.py: both only monkeypatch gmn.Node
    and define names at import time, no env-var-dependent top-level code
    (that's all inside load_config()/main(), guarded by
    `if __name__ == "__main__"`).'''
    spec = importlib.util.spec_from_file_location(module_name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _fake_result(plan, call_log, cache_stage, target):
    call_log.append((cache_stage, target))
    selected = plan.get((cache_stage, target), [])
    gains = [0.5] * len(selected)
    return {'SELECTED_COLUMNS': list(selected), 'm': SimpleNamespace(MDErho=np.array(gains))}


def make_fake_run_mde_for_target_tsls(plan, call_log):
    '''Stand-in for tsls-mde-gmn.py's run_mde_for_target(), same positional
    signature as the real one (`cache_stage` is a real positional-or-
    keyword parameter there, defaulting to \'stage1\' -- real Stage 1
    calls omit it, real Stage 2 calls pass \'stage2\' positionally right
    after code_fingerprint, so this must sit at the identical position to
    bind correctly either way) -- driven entirely by `plan`:
    {(cache_stage, target): [selected_column, ...]}, any (cache_stage,
    target) pair absent from `plan` selects nothing (a dead end). Every
    call is appended to `call_log` as (cache_stage, target) in exact
    invocation order -- meaningful because FakeProcessPoolExecutor below
    runs calls synchronously at submit() time, so submission order IS
    invocation order, not just scheduling order.'''
    def fake(target, df, observable_columns, prediction_start, prediction_length,
             mde_d, embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
             results_dir, delay_set, tp=1,
             metric='f1', target_metric='correlation', gamma_delta=2,
             ccm_sample=20, ccm_slope=0.01,
             is_behavioral_target=True, target_ranking_window_rows=1000,
             min_pred_events=20, max_window_rows=30000,
             ccm_lib_sizes_rows=(16, 24, 40, 64, 160, 400, 800),
             reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
             data_file=None, data_file_sha256=None, code_fingerprint=None,
             cache_stage='stage1', max_e=None, inner_workers=1, raw_columns=None):
        return _fake_result(plan, call_log, cache_stage, target)
    return fake


def make_fake_run_mde_for_target_simple(plan, call_log):
    '''Same as make_fake_run_mde_for_target_tsls() above, but matching
    simple-mde-gmn.py's own run_mde_for_target() signature (no metric/
    target_metric/gamma_delta/ccm_* params -- it scores via dimx.MDE
    directly, not cleaned_helper_functions.py's TSLS metrics -- and
    `cores` sits where tsls's version has no equivalent positional slot).
    '''
    def fake(target, df, observable_columns, prediction_start, prediction_length,
             mde_d, embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
             cores, results_dir, delay_set, tp=1,
             reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
             data_file=None, data_file_sha256=None, code_fingerprint=None,
             cache_stage='stage1'):
        return _fake_result(plan, call_log, cache_stage, target)
    return fake


class FakeProcessPoolExecutor:
    '''Drop-in for concurrent.futures.ProcessPoolExecutor that runs every
    submitted call synchronously, in the calling process, at submit()
    time (not at result() time) -- no forking, no pickling, and (this is
    the property the R12 test relies on) every call's side effect (its
    call_log append) happens in exact submission order. Ordering
    assertions on call_log therefore directly reflect the order
    build_nested_mde_network() issues MDE calls in, which is exactly what
    "no Stage 2 call before every Stage 1 call has returned" needs to
    check -- a real ProcessPoolExecutor would only add nondeterministic
    interleaving noise here, not a more faithful test.'''
    def __init__(self, max_workers=None, mp_context=None, initializer=None):
        # `initializer` (simple-mde-gmn.py's _mde_wave_worker_init, which
        # reapplies dimx.Run monkeypatches inside a real worker process) is
        # irrelevant here -- run_mde_for_target itself is faked below, so
        # no real dimx.MDE/CCM call ever happens for this to matter to.
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, fn, *args, **kwargs):
        import concurrent.futures
        fut = concurrent.futures.Future()
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as exc:  # pragma: no cover - exercised only if a plan is wrong
            fut.set_exception(exc)
        return fut
