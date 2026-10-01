'''Pins the three ways the Stage 2 lag set can be defined (derive_lag_set).

WHY THIS EXISTS. Stage 2 offers MDE one candidate column per
(real observable x lag), and Stage 2 is where essentially all of a run's time
goes -- so the SIZE of the lag set is a compute budget. There are three ways
to state it, differing only in which quantity you pin:

    DELAY_SOURCE='lag_mode'   pins the DEPTH, the count follows (auto-derived)
    DELAY_SOURCE='tau_count'  pins the COUNT, the depth follows (LAG_STEP=tau)
    DELAY_SOURCE='explicit'   pins EVERY VALUE (DELAY_SET)

'tau_count' is the one added 2026-10-01. Before it, the only way to cap the
count was to back-solve GENERATE_LENGTH -- a knob that means "how many steps
Generate() free-runs for" and has no reason to agree with how many lags are
worth searching. Test C below pins the auto-derived path against the values
the real sbatch scripts resolve to, so that coupling stays visible rather than
being rediscovered the next time a job runs long.

No pytest in this venv -- run it directly:
    python3 tests/test_lag_set_sources.py
'''
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'src'))
from mde_gmn.caching import build_network_fingerprint  # noqa: E402
from mde_gmn.columns import derive_lag_set  # noqa: E402

# derive_lag_set's positional tail, so each case below varies only what it means to.
DEFAULTS = dict(explicit_delay_set=[0, -1, -3, -5, -10], lag_mode='generation_length_adjusted',
                lag_density='strided', lag_step=4, generate_length=400, mde_lib_size=1200,
                lag_count=10)


def resolve(**overrides):
    kwargs = {**DEFAULTS, **overrides}
    return derive_lag_set(kwargs.pop('delay_source'), kwargs.pop('explicit_delay_set'),
                          kwargs.pop('lag_mode'), kwargs.pop('lag_density'),
                          kwargs.pop('lag_step'), kwargs.pop('generate_length'),
                          kwargs.pop('mde_lib_size'), kwargs.pop('lag_count'))


def test_a_each_source_produces_its_documented_set():
    cases = [
        # (what it pins, overrides, expected lag set)
        ('explicit: every value',
         dict(delay_source='explicit', explicit_delay_set=[0, -2, -7, -13]), [-2, -7, -13]),
        ('explicit: a 0 entry is dropped, not kept',
         dict(delay_source='explicit', explicit_delay_set=[0, -1]), [-1]),
        ('tau_count: tau=5, 4 delays',
         dict(delay_source='tau_count', lag_step=5, lag_count=4), [-5, -10, -15, -20]),
        ('tau_count: tau=1 gives consecutive lags',
         dict(delay_source='tau_count', lag_step=1, lag_count=3), [-1, -2, -3]),
        ('tau_count: a single lag is legal',
         dict(delay_source='tau_count', lag_step=7, lag_count=1), [-7]),
    ]
    for label, overrides, expected in cases:
        got = resolve(**overrides)
        assert got == expected, f'{label}: expected {expected}, got {got}'
    print(f'A. all {len(cases)} documented source cases: PASS')


def test_b_tau_count_pins_the_count_exactly():
    '''The whole point of the mode: the count is what you asked for, for every
    (tau, count) -- not an artifact of some other knob.'''
    for tau in (1, 2, 5, 13, 40):
        for count in (1, 2, 10, 37):
            got = resolve(delay_source='tau_count', lag_step=tau, lag_count=count)
            assert len(got) == count, f'tau={tau} count={count}: got {len(got)} lags'
            assert min(got) == -tau * count, f'tau={tau} count={count}: deepest {min(got)}'
            strides = {got[i] - got[i + 1] for i in range(len(got) - 1)}
            assert strides <= {tau}, f'tau={tau}: non-uniform stride {strides}'
            # The conventions every path shares, restated here because Stage 2
            # and self_delay_set both rely on them.
            assert all(d < 0 for d in got), f'non-negative lag in {got}'
            assert got == sorted(got, reverse=True), f'not descending: {got}'
    print('B. tau_count pins the count for 20 (tau, count) pairs: PASS')


def test_c_auto_derived_path_is_unchanged():
    '''Regression guard. These are the values the shipped sbatch scripts
    resolve to; the 100 is why tau_count was added.'''
    cases = [
        # (label, overrides, expected count, expected deepest)
        ("tsls-mde-gmn.sbatch today: GENERATE_LENGTH=400, LAG_STEP=4", dict(), 100, -397),
        ('dense ignores LAG_STEP', dict(lag_density='dense'), 400, -400),
        ('every_lag_possible reaches into the library segment',
         dict(lag_mode='every_lag_possible'), 288, -1149),
        ('GENERATE_LENGTH=0 still yields one lag, never an empty set',
         dict(generate_length=0), 1, -1),
    ]
    for label, overrides, count, deepest in cases:
        got = resolve(delay_source='lag_mode', **overrides)
        assert len(got) == count, f'{label}: expected {count} lags, got {len(got)}'
        assert min(got) == deepest, f'{label}: expected deepest {deepest}, got {min(got)}'
    print(f'C. all {len(cases)} auto-derived fixtures unchanged: PASS')


def test_d_bad_input_raises_instead_of_falling_through():
    '''An unknown delay_source used to be treated as 'lag_mode' (the function
    only special-cased 'explicit' and let everything else fall through), so a
    typo silently ran the auto path. It raises now.'''
    cases = [
        ('unknown delay_source', dict(delay_source='lagmode'), 'delay_source'),
        ('unknown lag_mode', dict(delay_source='lag_mode', lag_mode='nope'), 'lag_mode'),
        ('unknown lag_density', dict(delay_source='lag_mode', lag_density='nope'), 'lag_density'),
        ('tau_count without a count', dict(delay_source='tau_count', lag_count=None), 'lag_count'),
        ('tau_count with count 0', dict(delay_source='tau_count', lag_count=0), 'lag_count'),
    ]
    for label, overrides, expected_in_message in cases:
        try:
            got = resolve(**overrides)
        except ValueError as exc:
            assert expected_in_message in str(exc), (
                f'{label}: message does not name {expected_in_message}: {exc}')
        else:
            raise AssertionError(f'{label}: no ValueError, returned {len(got)} lags')
    print(f'D. all {len(cases)} invalid inputs raise: PASS')


def test_e_config_plumbs_the_knobs_through():
    from mde_gmn.config import load_config
    with tempfile.TemporaryDirectory() as tmp:
        def load(**env):
            for key in [k for k in os.environ if k.startswith('TSLS_MDE_GMN_')]:
                del os.environ[key]
            os.environ['TSLS_MDE_GMN_RESULTS_DIR'] = tmp
            os.environ.update({f'TSLS_MDE_GMN_{k}': v for k, v in env.items()})
            return load_config()

        cfg = load(DELAY_SOURCE='tau_count', LAG_STEP='5', LAG_COUNT='8')
        assert (cfg['DELAY_SOURCE'], cfg['LAG_STEP'], cfg['LAG_COUNT']) == ('tau_count', 5, 8)
        assert resolve(delay_source=cfg['DELAY_SOURCE'], lag_step=cfg['LAG_STEP'],
                       lag_count=cfg['LAG_COUNT']) == [-5, -10, -15, -20, -25, -30, -35, -40]

        assert load()['DELAY_SOURCE'] == 'lag_mode', 'the default source changed'

        for label, env in [('a typo in DELAY_SOURCE', dict(DELAY_SOURCE='tau-count')),
                           ('LAG_COUNT=0', dict(LAG_COUNT='0'))]:
            try:
                load(**env)
            except ValueError:
                pass
            else:
                raise AssertionError(f'{label} was accepted')
    print('E. config accepts and validates DELAY_SOURCE/LAG_STEP/LAG_COUNT: PASS')


def test_f_cache_key_follows_the_resolved_list_only():
    '''build_network_fingerprint stores the resolved lag list, so the SOURCE is
    not part of the cache key -- switching from lag_mode to an equivalent
    tau_count reuses cached work, while changing the count does not.'''
    def fingerprint(delay_set):
        return build_network_fingerprint(
            'MDE-GMN', ['T'], ['A', 'B'], 2, 1, 1, 0.2, 0.2, 0, delay_set,
            range(3, 5), range(-1, 0), 1200, 100, 'f1', 'correlation', 2,
            800, (20, 50), 2, 0.01, 'data.csv', 'deadbeef', 'codefp')

    # Two sources, same resolved list: lag_step=5/lag_count=4 and the hand-picked
    # equivalent both give [-5, -10, -15, -20].
    by_tau_count = resolve(delay_source='tau_count', lag_step=5, lag_count=4)
    by_explicit = resolve(delay_source='explicit', explicit_delay_set=[-5, -10, -15, -20])
    assert by_tau_count == by_explicit, 'fixture broken: the two sources disagree'
    assert fingerprint(by_tau_count) == fingerprint(by_explicit), (
        'equivalent lag sets from different sources produced different cache keys, '
        'so switching source would needlessly discard cached MDE work')

    assert fingerprint(by_tau_count) != fingerprint(
        resolve(delay_source='tau_count', lag_step=5, lag_count=5)), (
        'changing LAG_COUNT did not change the cache key, so a wider Stage 2 '
        'search would reuse the narrower one\'s cached result')
    print('F. cache key follows the resolved list, not the source: PASS')


def test_g_every_source_has_lag_depth_remedy_advice():
    '''main() refuses a lag set deeper than the MDE library segment and tells
    the operator what to change -- advice that is only correct for the path that
    produced the set. The mapping is keyed by DELAY_SOURCE, so a source without
    an entry turns that clear startup error into a KeyError.'''
    from mde_gmn.config import DELAY_SOURCES
    from mde_gmn.pipeline import _LAG_DEPTH_REMEDY

    missing = set(DELAY_SOURCES) - set(_LAG_DEPTH_REMEDY)
    assert not missing, (
        f'DELAY_SOURCE value(s) {sorted(missing)} have no entry in '
        f"pipeline._LAG_DEPTH_REMEDY, so main()'s lag-depth guard would raise "
        f'KeyError instead of its intended message. A new source also needs: '
        f"derive_lag_set()'s own branch, the sbatch DELAY_SOURCE comment blocks, "
        f'and Section sec:lag-set of docs/tsls-mde-gmn-pipeline.tex.')
    stale = set(_LAG_DEPTH_REMEDY) - set(DELAY_SOURCES)
    assert not stale, f'remedy advice for removed source(s): {sorted(stale)}'
    print(f'G. all {len(DELAY_SOURCES)} sources have lag-depth remedy advice: PASS')


def main():
    test_a_each_source_produces_its_documented_set()
    test_b_tau_count_pins_the_count_exactly()
    test_c_auto_derived_path_is_unchanged()
    test_d_bad_input_raises_instead_of_falling_through()
    test_e_config_plumbs_the_knobs_through()
    test_f_cache_key_follows_the_resolved_list_only()
    test_g_every_source_has_lag_depth_remedy_advice()
    print('ALL PASS')


if __name__ == '__main__':
    main()
