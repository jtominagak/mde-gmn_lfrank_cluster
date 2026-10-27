'''Shared helpers for loading scripts/src/*.py (hyphenated filenames, not
importable via a plain `import` statement) as modules for testing, and for
faking run_mde_for_target() so build_nested_mde_network()'s BFS/recursion
logic can be exercised without real MDE/CCM computation or real data.'''
import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / 'src'


# The order the mde_gmn package itself documents (see its __init__.py). Used
# only to assemble the namespace below; a later module wins a name clash, of
# which there are none today. network_parallel is deliberately ABSENT: it
# defines build_nested_mde_network too, and adding it would make the namespace
# hand back the parallel builder to every test that wants the serial one.
# tests/test_parallel_mde_equivalence.py imports both modules by name instead.
_PACKAGE_MODULES = (
    'columns', 'progress', 'tsls_data', 'caching', 'mde', 'waves', 'network',
    'gmn_embedding', 'gmn_build', 'theta', 'generate', 'plots', 'diagnostics',
    'config', 'pipeline',
)


def load_pipeline_module(filename, module_name):
    '''Loads scripts/src/<filename> so a test can reach the pipeline's
    functions as attributes of one object.

    For simple-mde-gmn.py that object is the module itself: that pipeline is
    still one file, and every name it defines is on it.

    For tsls-mde-gmn.py it is a namespace assembled from the mde_gmn package.
    Phase 2 moved that pipeline's code into one module per stage and left the
    entry script as four lines, so there is no longer a single module holding
    build_nested_mde_network, _metric_for and run_mde_evaluate at once -- but
    the tests are written against both pipelines with the same attribute
    access, and the point of the tests is the behaviour, not which file a
    function sits in. The namespace holds the real objects from the real
    modules, so `ns.build_nested_mde_network.__module__` still names
    mde_gmn.network, which is how install_fake_mde() finds where to patch.

    It is a snapshot taken at load time, so patch the module (see
    install_fake_mde) rather than the namespace. `module_name` must be unique
    per test process: re-loading under a name already in sys.modules would
    hand back the cached module instead of a fresh one.
    '''
    spec = importlib.util.spec_from_file_location(module_name, SRC_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    if hasattr(module, 'build_nested_mde_network'):
        return module                       # one-file pipeline, nothing to assemble
    return _package_namespace()


def _package_namespace():
    '''Every top-level name in the mde_gmn package, flattened onto one object.

    Includes the modules' own imports (`gmn`, `tsls`, `np`), which is
    deliberate: tests read those through the pipeline module too, and they are
    the same objects the pipeline itself uses.
    '''
    namespace = SimpleNamespace()
    for name in _PACKAGE_MODULES:
        module = importlib.import_module(f'mde_gmn.{name}')
        for attr, value in vars(module).items():
            if not attr.startswith('__'):
                setattr(namespace, attr, value)
    return namespace


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
             cache_stage='stage1', max_e=None, inner_workers=1, raw_columns=None,
             use_gpu=False):
        return _fake_result(plan, call_log, cache_stage, target)
    return fake


def make_fake_run_mde_for_target_simple(plan, call_log):
    '''Same as make_fake_run_mde_for_target_tsls() above, but matching
    simple-mde-gmn.py's own run_mde_for_target() signature (no metric/
    target_metric/gamma_delta/ccm_* params -- it scores via dimx.MDE
    directly, not edm.py's TSLS metrics -- and
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


def install_fake_mde(module, fake, executor=None):
    '''Install `fake` (and a fake process pool) where `module`'s network
    builder will actually look them up, and return the module patched.

    This is not the same place for both pipelines. tsls-mde-gmn.py's
    build_nested_mde_network() now lives in mde_gmn/network.py and calls MDE
    as `mde.run_mde_for_target(...)` through the module object, so the fake
    belongs on mde_gmn.mde -- assigning to the loaded entry script would
    rebind a name nothing reads, and the test would silently run real MDE
    against real pyEDM. simple-mde-gmn.py still defines its builder in the
    entry script itself, so there the entry module IS the right target.

    Rather than hard-code either case, follow the builder to wherever it is
    defined and ask whether it reaches MDE through a module: patch
    <builder module>.mde if so, the builder module itself otherwise. That
    keeps working if more code moves into the package later.
    '''
    builder_module = sys.modules[module.build_nested_mde_network.__module__]
    mde_home = getattr(builder_module, 'mde', builder_module)
    if mde_home is not builder_module and 'run_mde_for_target' in vars(builder_module):
        raise AssertionError(
            f'{builder_module.__name__} imports run_mde_for_target directly as well as '
            'through the mde module; the direct binding would shadow this fake and the '
            'test would run real MDE. Call it as mde.run_mde_for_target(...) only.')
    mde_home.run_mde_for_target = fake
    # The pool lives in mde_gmn.waves now -- both builders call
    # waves.run_mde_batch() rather than each opening their own executor --
    # so that is where the synchronous double has to be installed. Patched on
    # the builder module too, harmlessly, for any test that asserts on it.
    waves_module = importlib.import_module('mde_gmn.waves')
    waves_module.ProcessPoolExecutor = executor or FakeProcessPoolExecutor
    builder_module.ProcessPoolExecutor = executor or FakeProcessPoolExecutor
    return builder_module


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
    def __init__(self, max_workers=None, mp_context=None, initializer=None,
                 initargs=()):
        # `initializer` is run ONCE here, in the calling process, which is
        # the closest this synchronous double can get to a real pool's
        # per-worker init. Two callers rely on it:
        #   * simple-mde-gmn.py's _mde_wave_worker_init, which reapplies
        #     dimx.Run monkeypatches -- irrelevant to these tests, since
        #     run_mde_for_target itself is faked below and no real
        #     dimx.MDE/CCM call ever happens;
        #   * network_parallel's gpu_budget.init_worker, which is a no-op
        #     on a GPU-less host (empty budget list) and must therefore
        #     stay harmless when it IS called.
        # It is called rather than ignored so the double keeps matching
        # ProcessPoolExecutor's signature -- it silently diverging from the
        # real class is what let an initargs= change reach the suite as a
        # TypeError instead of a behavioural assertion.
        if initializer is not None:
            initializer(*initargs)

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
