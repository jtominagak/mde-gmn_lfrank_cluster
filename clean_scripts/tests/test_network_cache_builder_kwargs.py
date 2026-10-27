'''Regression test: the network cache must distinguish builders' own options.

THE BUG THIS PINS. build_network() takes `network_builder_kwargs` -- options
only the chosen builder understands. Today that is the parallel-MDE builder's
`parallel_mde_scope`, which selects between two genuinely different algorithms:

    'staged'  Stage 1 (real observables) then Stage 2 (lagged leaves), the
              serial pipeline's own network.
    'union'   ONE MDE per node over real AND lagged candidates together -- a
              different network, with a (1 + len(LAG_SET))x wider pool.

Those kwargs were deliberately kept out of build_network_fingerprint's
signature, so they were not in the cache key. Everything else about the two
runs is identical -- same targets, same observables, same entry script and
therefore the same code fingerprint -- so a 'union' run submitted after a
'staged' run would match the staged network in the shared cache, announce
"matched cached network", skip the entire MDE phase, and report the other
algorithm's network as its own. Invisible: the config dump still says 'union'.

This matters precisely when the two are run as a pair, which is the only way
to compare them.

No pytest in this venv -- run it directly:
    python3 clean_scripts/tests/test_network_cache_builder_kwargs.py
'''
import sys
import tempfile
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _pipeline_test_utils import load_pipeline_module  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from mde_gmn.caching import build_network_fingerprint  # noqa: E402

TARGETS = ['T']
RAW_COLUMNS = ['A', 'B']
ROWS = 60


def _toy_df():
    rng = np.random.default_rng(0)
    return pd.DataFrame({c: rng.normal(size=ROWS) for c in TARGETS + RAW_COLUMNS})


def _builder_returning(driver):
    '''A fake builder whose network says which scope produced it, so a cache
    mix-up is visible in the returned graph rather than only in a log line.'''
    def build(*args, **kwargs):
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(TARGETS)
        graph.add_node(driver)
        graph.add_edge(driver, 'T', delay=0, weight=0.5)
        return graph, pd.DataFrame(), {}
    return build


def test_a_fingerprint_separates_the_scopes():
    def fingerprint(builder_kwargs):
        return build_network_fingerprint(
            'MDE-GMN', ['T'], ['A', 'B'], 2, 1, 1, 0.2, 0.2, 0, [-1, -3],
            range(3, 5), range(-1, 0), 1200, 100, 'f1', 'correlation', 2,
            800, (20, 50), 2, 0.01, 'toy.csv', 'deadbeef', 'codefp',
            builder_kwargs=builder_kwargs)

    assert fingerprint({'parallel_mde_scope': 'staged'}) != fingerprint(
        {'parallel_mde_scope': 'union'}), (
        "'staged' and 'union' share a cache key, so whichever runs second would "
        'reuse the first one\'s network and never run its own algorithm')
    print('A. staged and union get different cache keys: PASS')


def test_b_absent_kwargs_leave_the_key_untouched():
    '''The serial pipeline passes no builder kwargs. Adding the field
    unconditionally would have invalidated every existing cache entry on the
    cluster, so it is omitted when empty -- that is the point of this case.'''
    def fingerprint(**kw):
        return build_network_fingerprint(
            'MDE-GMN', ['T'], ['A', 'B'], 2, 1, 1, 0.2, 0.2, 0, [-1, -3],
            range(3, 5), range(-1, 0), 1200, 100, 'f1', 'correlation', 2,
            800, (20, 50), 2, 0.01, 'toy.csv', 'deadbeef', 'codefp', **kw)

    assert 'builder_kwargs' not in fingerprint(), (
        'the serial pipeline now carries a builder_kwargs field, which '
        'invalidates every cache entry written before this change')
    assert fingerprint() == fingerprint(builder_kwargs={}), (
        'an empty kwargs dict changed the key')
    assert fingerprint() == fingerprint(builder_kwargs=None)
    print('B. no builder kwargs leaves the serial key byte-identical: PASS')


def test_c_a_union_run_does_not_inherit_a_staged_network():
    '''End to end through build_network(), which is where the mix-up would
    actually have happened.

    The directory layout mirrors the cluster's, and has to: every cached
    network is filed under the same name (network_cache_<targets>.pkl), so what
    keeps separate jobs' entries apart is that each writes into its OWN
    RUN_CACHE_DIR ($CACHE_ROOT/job_<id>), while lookups glob the shared
    CACHE_ROOT recursively. Pointing all three runs at one directory would just
    have them overwrite each other, testing nothing.'''
    module = load_pipeline_module('run_serial_mde.py', '_test_builder_kwargs_cache')
    df = _toy_df()
    with tempfile.TemporaryDirectory() as tmp:
        cache_root = Path(tmp) / 'cache_root'
        cache_root.mkdir()

        def run(job, driver, scope):
            results_dir = Path(tmp) / f'results_{job}'
            run_cache_dir = cache_root / f'job_{job}'
            results_dir.mkdir()
            run_cache_dir.mkdir()
            result = module.build_network(
                list(TARGETS), {t: {'SELECTED_COLUMNS': ['A']} for t in TARGETS}, df,
                prediction_start=40, prediction_length=10,
                mode='MDE-GMN', e_range=range(3, 5), tau_range=range(-1, 0),
                results_dir=results_dir, cores=1,
                metric='correlation', target_metric='correlation', gamma_delta=2,
                ccm_window_rows=40, ccm_lib_size_pcts=(20, 50), ccm_sample=2,
                ccm_slope=0.01, observable_columns=list(RAW_COLUMNS),
                nested_n=2, nested_m=1, embed_dim_rho_min=0.2, cross_map_rho_min=0.2,
                exclusion_radius=0, lag_set=[-1, -3], tp=1,
                reuse_cache=True, prev_runs_dir=cache_root, run_cache_dir=run_cache_dir,
                data_file='toy.csv', data_file_sha256='deadbeef',
                code_fingerprint='codefp',
                network_builder=_builder_returning(driver),
                network_builder_kwargs={'parallel_mde_scope': scope},
            )
            return set(result['network']['Graph'].nodes)

        first = run(1, 'A', 'staged')
        assert 'A' in first, f'fixture broken: first run did not build ({sorted(first)})'

        # Same everything except the scope. Its own builder picks 'B'.
        second = run(2, 'B', 'union')
        assert 'B' in second, (
            f"the union run returned the staged run's cached network ({sorted(second)}) "
            f'instead of building its own -- it would have skipped the whole MDE '
            f'phase and reported the wrong algorithm')
        assert 'A' not in second, f'union network contains the staged driver: {sorted(second)}'

        # The same scope twice DOES still reuse, so the fix did not simply
        # disable caching for the fork. This run's own builder would pick 'B',
        # so finding 'A' proves the graph came from run one's cache.
        third = run(3, 'B', 'staged')
        assert 'A' in third and 'B' not in third, (
            f'a repeat staged run rebuilt instead of reusing run 1\'s cache '
            f'({sorted(third)}) -- caching is now broken for the fork')
    print('C. union builds its own network; a repeat staged run still reuses: PASS')


def main():
    test_a_fingerprint_separates_the_scopes()
    test_b_absent_kwargs_leave_the_key_untouched()
    test_c_a_union_run_does_not_inherit_a_staged_network()
    print('ALL PASS')


if __name__ == '__main__':
    main()
