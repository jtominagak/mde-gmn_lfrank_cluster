"""Multiview Delay Embedding (MDE) for one node: which signals predict it?

Step 2 of the pipeline. Given one target observable and a pool of candidate
drivers (other observables, and delayed copies of observables), it answers:
which small set of candidates best predicts the target?

The method is forward selection. Start with nothing chosen. Score every
candidate by how well it predicts the target, keep the best one, then score
every remaining candidate again *alongside* the one already kept, and so on.
Each pass is one "round"; MDE_D caps the number of rounds, so a node ends up
with at most MDE_D drivers. Every round still ranks all remaining candidates --
the cap limits how many are chosen, never how many are considered.

A candidate that predicts the target well is not necessarily driving it, so
each kept candidate also has to pass a causality gate: convergent cross
mapping (CCM), which checks that prediction improves as more data is used --
the signature of a real causal link rather than a coincidence.

    run_mde_for_target        one target: forward selection + CCM gate
    run_multi_mde_for_targets the same for several targets at once, so one
                              shared embedding is forward-selected rather
                              than one per target (GMN_MODE='combined')
    ccm_prefilter_observables a cheap CCM screen run BEFORE selection, to
                              drop candidates that cannot be causal at all
    failed_mde_result         the empty result shape used when a node has
                              too little usable data, so callers need no
                              special case
    plot_mde_*                diagnostic figures for a finished selection

Results are cached (see caching.py): an MDE call is the expensive part of the
pipeline, so a completed selection is reused across jobs.

THE RESULT SHAPE. Every consumer -- the edge-acceptance rule, the diagnostics
figures, the sweeps -- reads the same dict: 'SELECTED_COLUMNS', 'mde_df', and
'm', a namespace mirroring the reference MDE library's own result object
(MDEcolumns/MDErho/MDEOut/args). _mde_namespace() builds it, and the failure
shapes below build the same thing empty, which is what lets a dead-end node
flow through untouched instead of being special-cased at every reader.

Depends on: edm (the EDM engine), columns, tsls_data, caching; numpy, pandas,
matplotlib.
Imported by: network, network_parallel, waves, pipeline.

NOTE for tests: tests replace run_mde_for_target with a fake by assigning to
this module (mde_gmn.mde.run_mde_for_target = fake). Callers must therefore
reach it through the module object -- `from mde_gmn import mde` then
`mde.run_mde_for_target(...)` -- not `from mde_gmn.mde import run_mde_for_target`,
which would bind the real function before the fake is installed.
"""

import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext

import matplotlib
matplotlib.use('Agg')          # cluster jobs have no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import edm
from mde_gmn.caching import (
    _load_pickle_cache_entry, build_mde_selection_fingerprint,
    build_multi_mde_selection_fingerprint, find_cached, save_pickle_to_cache)
from mde_gmn.columns import DELAY_MARKER, observable_and_delay
from mde_gmn.tsls_data import (
    _ccm_window, _metric_for, _resolve_target_ranking_window, _rho_min_for_metric)


# =====================================================================
# The result shape
# =====================================================================

def _mde_namespace(selected_columns, rho, mde_out, lib, pred, tp, tau=-1):
    """The 'm' namespace every consumer of an MDE result reads."""
    return types.SimpleNamespace(
        MDEcolumns=list(selected_columns),
        MDErho=rho,
        MDEOut=mde_out,
        args=types.SimpleNamespace(lib=lib, pred=pred, tau=tau, Tp=tp),
    )


def _empty_result(key, value):
    """A result indistinguishable from a node that selected zero drivers.

    Used when an MDE call RAISED rather than returning (the EDM engine has
    internal failure modes against degenerate candidate data that this
    pipeline's own gates neither catch nor control), and when a node cannot
    reach MIN_PRED_EVENTS. Selecting nothing is already an ordinary
    termination condition for the network builders' breadth-first walk, so
    shaping a failure the same way lets one degenerate node dead-end
    cleanly instead of the exception propagating out of a worker pool and
    destroying everything computed alongside it.

    `key` is 'target' or 'targets' -- the two entry points below return the
    same dict under different names for the thing that was predicted.
    """
    return {
        key: value,
        'mde_df': None,
        'm': _mde_namespace([], np.array([]), None, None, None, None, tau=None),
        'SELECTED_COLUMNS': [],
    }


def failed_mde_result(target):
    """Zero-driver result for a single target whose MDE call failed."""
    return _empty_result('target', target)


def failed_multi_mde_result(targets):
    """Zero-driver result for a target GROUP whose joint MDE call failed."""
    return _empty_result('targets', list(targets))


# =====================================================================
# Cache
# =====================================================================

_CACHE_FIELDS = ('MDEcolumns', 'MDErho', 'MDEOut', 'lib', 'pred', 'tau', 'Tp')


def _load_cached_selection(prev_runs_dir, filename, fingerprint, csv_path, label):
    """A previously computed selection, rebuilt into the 'm' namespace.

    Returns (m, selected_columns) on a hit, None on a miss. Writes the
    cached MDEOut back out to `csv_path` so a reused run leaves the same
    files behind as a computed one.
    """
    hit = find_cached(prev_runs_dir, filename, fingerprint,
                      load_one=_load_pickle_cache_entry)
    if hit is None:
        print(f'{label}: REUSE_BEST_PARAMS set but no matching cached MDE selection '
              f'found under {prev_runs_dir} -- running forward-selection.', flush=True)
        return None
    entry, source = hit
    print(f'{label}: REUSE_BEST_PARAMS matched cached MDE selection {source} -- '
          f'skipping forward-selection entirely', flush=True)
    m = _mde_namespace(entry['MDEcolumns'], entry['MDErho'], entry['MDEOut'],
                       entry['lib'], entry['pred'], entry['Tp'], tau=entry['tau'])
    m.MDEOut.to_csv(csv_path, index=False)
    return m, list(m.MDEcolumns)


def _save_selection(filename, fingerprint, m, results_dir, run_cache_dir):
    save_pickle_to_cache(
        filename,
        {'fingerprint': fingerprint,
         'MDEcolumns': m.MDEcolumns, 'MDErho': m.MDErho, 'MDEOut': m.MDEOut,
         'lib': m.args.lib, 'pred': m.args.pred, 'tau': m.args.tau, 'Tp': m.args.Tp},
        results_dir, run_cache_dir,
    )


def _leakage_safe_window(df, columns, prediction_start, prediction_length, delay_set, tp):
    """The rows an MDE call may use, and the library fraction within them.

    See config.py's three-way A/B/end split. `prediction_start` is A and
    `prediction_length` is B-A. Two trims happen here, BEFORE the per-node
    ranking window is resolved from what is left:

      * `warmup` rows off the front -- delay expansion leaves that many
        leading NaN rows in every lagged candidate column;
      * `tp` rows off the back (B - tp, not B) -- so the last validated
        row's own t+tp label can never land at or past B.

    Doing both first is what guarantees no window resolved downstream can
    reach into the Generate/Forecast validation segment [B, end), however
    large it grows.

    Returns (frame_up_to_B, usable_df, lib_ratio).
    """
    frame = df[columns].iloc[:prediction_start + prediction_length]
    warmup = max((abs(d) for d in delay_set), default=0)
    usable_end = prediction_start + prediction_length - tp
    usable_df = frame.iloc[warmup:usable_end].reset_index(drop=True)
    lib_ratio = (prediction_start - warmup) / max(1, usable_end - warmup)
    return frame, usable_df, lib_ratio


# =====================================================================
# Forward selection, one target
# =====================================================================

def run_mde_for_target(target, df, observable_columns, prediction_start, prediction_length,
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
    '''Forward selection for one target: which candidates best predict it?

    Runs edm.proper_mde and returns its result in the shape this module's
    docstring describes.

    `mde_d` is already the *effective* dimension count (see
    effective_mde_dimension()), which Stage 2 inflates by the number of lags
    for an unrelated reason -- candidate-pool breadth, not embedding depth.
    `max_e` is the RAW, un-inflated MDE_D, and bounds something different:
    the per-candidate embedding-dimension auto-select sweep. Passing `mde_d`
    for both would silently deepen every Stage 2 embedding.

    E AND TAU ARE NOT FIXED HERE. E=0 asks proper_mde to auto-select the
    embedding dimension per candidate, capped at `max_e`. tau is simply not
    passed -- proper_mde's own default (-1) applies.

    TWO METRICS, AND THEY ARE NOT INTERCHANGEABLE. `target_metric` scores
    everything that predicts `target` (the ranking stage and the E-select
    sweep); `metric` scores the CCM causality gate, which predicts a
    CANDIDATE column -- always a TSLS-converted neuron series. Callers
    resolve `target_metric` for whatever `target` actually is, via
    _metric_for(): 'correlation' for a behavioural target, 'f1' for a
    neuron. See config.py's TARGET_METRIC entry.

    WINDOW SIZING is per node, not one global window -- see
    _resolve_target_ranking_window(). A behavioural target gets a fixed
    `target_ranking_window_rows`; a neuron's window grows until its own
    pred portion holds `min_pred_events` real reset events, capped at
    `max_window_rows`. The CCM gate's library sweep is separate and
    absolute (`ccm_lib_sizes_rows`), never a percentage of that window:
    the convergence curve's shape depends on the absolute library size, not
    its ratio to the surrounding frame.

    INNER POOL (`inner_workers`, default 1 = no pool). proper_mde has no
    parallelism of its own, so this opens ONE ThreadPoolExecutor for this
    node's whole call and hands it down -- threads, because every hot loop
    it reaches is a compiled pyEDM/scipy call that releases the GIL.
    Callers size it via waves.split_cores() so outer x inner never exceeds
    the core budget.

    `raw_columns`: when given, maps each candidate column back to its
    (observable, delay) pair so proper_mde can grow its Theiler exclusion
    radius to cover the deepest lag already selected. None (the depth-0
    seed's always-real pool) means there are no delays to look up.

    `use_gpu`: passed through to proper_mde. True changes anything only on a
    machine with a visible CUDA device, and falls back to the CPU silently
    otherwise.
    '''
    # Both skill gates are resolved HERE, before the cache fingerprint is
    # built and before proper_mde sees them, so a node cached under the f1
    # gate can never be reused for one scored by correlation -- the resolved
    # float is what goes into the fingerprint. `target_metric` is already the
    # node's own metric: every call site passes _metric_for(node, ...).
    embed_dim_rho_min = _rho_min_for_metric(embed_dim_rho_min, target_metric)
    cross_map_rho_min = _rho_min_for_metric(cross_map_rho_min, target_metric)
    if max_e is None:
        # Fallback only; every real call site passes the raw MDE_D.
        max_e = mde_d

    mde_df, usable_df, lib_ratio = _leakage_safe_window(
        df, [target] + observable_columns, prediction_start, prediction_length, delay_set, tp)

    # Resolved BEFORE the fingerprint below: the window size and (for a
    # neuron) its event count are real, data-dependent inputs to this
    # call's outcome, so they belong in its cache key.
    window_rows, pred_event_count = _resolve_target_ranking_window(
        target, df, usable_df, lib_ratio,
        is_behavioral_target, target_ranking_window_rows, min_pred_events, max_window_rows,
    )
    if window_rows is None:
        cap = min(max_window_rows, len(usable_df))
        print(f'{target}: cannot reach MIN_PRED_EVENTS={min_pred_events} in the pred portion '
              f'of any window up to min(MAX_WINDOW_ROWS, leakage-safe range)={cap} rows -- '
              f'best achievable was {pred_event_count} events (short by '
              f'{min_pred_events - pred_event_count}). Treating as a dead-end (zero drivers '
              f'selected) rather than scoring it on too few events.', flush=True)
        return failed_mde_result(target)
    if not is_behavioral_target:
        print(f'{target}: resolved ranking window={window_rows} rows, '
              f'pred_event_count={pred_event_count} (MIN_PRED_EVENTS={min_pred_events})',
              flush=True)

    csv_path = results_dir / f'mde_output_{cache_stage}_{target}.csv'
    cache_filename = f'mde_selection_{cache_stage}_{target}.pkl'
    fingerprint = None
    if reuse_cache:
        fingerprint = build_mde_selection_fingerprint(
            target, observable_columns, mde_d, tp, embed_dim_rho_min, cross_map_rho_min,
            exclusion_radius, delay_set, prediction_start, prediction_length,
            metric, target_metric, gamma_delta,
            is_behavioral_target, target_ranking_window_rows, min_pred_events, max_window_rows,
            window_rows, pred_event_count, ccm_lib_sizes_rows, ccm_sample, ccm_slope,
            data_file, data_file_sha256, code_fingerprint, cache_stage, max_e,
        )
        cached = _load_cached_selection(prev_runs_dir, cache_filename, fingerprint,
                                        csv_path, target)
        if cached is not None:
            m, selected_columns = cached
            print(f'MDE selected columns for {target}:', selected_columns, flush=True)
            return {'target': target, 'mde_df': mde_df, 'm': m,
                    'SELECTED_COLUMNS': selected_columns}

    ccm_df, ccm_lib_end = _ccm_window(usable_df, lib_ratio, window_rows)

    # One (observable, delay) lookup per candidate, reused for both maps
    # proper_mde wants. column_delay drives its dynamic per-dimension
    # exclusion radius; column_observable only feeds its verbose log (how
    # many DISTINCT real observables a long run has found). raw_columns=None
    # -> both None -> proper_mde's own no-op defaults.
    column_delay = column_observable = None
    if raw_columns is not None:
        pairs = {c: observable_and_delay(c, raw_columns) for c in observable_columns}
        column_delay = {c: delay for c, (_obs, delay) in pairs.items()}
        column_observable = {c: obs for c, (obs, _delay) in pairs.items()}

    # One inner pool for this node's whole proper_mde call -- opened here,
    # reused across every dimension's ranking loop and every CCM-gate
    # candidate, closed on return. nullcontext() when not pooling, so a
    # caller that passes nothing gets exactly the unpooled behaviour.
    pool_ctx = (ThreadPoolExecutor(max_workers=inner_workers) if inner_workers > 1
                else nullcontext())
    with pool_ctx as inner_pool:
        result = edm.proper_mde(
            ccm_df, target=target, columns=observable_columns, D=mde_d,
            lib=[1, ccm_lib_end], pred=[ccm_lib_end + 1, len(ccm_df)],
            Tp=tp, E=0, maxE=max_e, exclusionRadius=exclusion_radius,
            libSizes=ccm_lib_sizes_rows, sample=ccm_sample, ccmSlope=ccm_slope,
            crossMapRhoMin=cross_map_rho_min, embedDimRhoMin=embed_dim_rho_min,
            metric=target_metric, ccmMetric=metric, gammaDelta=gamma_delta,
            verbose=True, pool=inner_pool, column_delay=column_delay,
            column_observable=column_observable, use_gpu=use_gpu,
        )

    mde_out = pd.DataFrame({'rho': result['MDErho']})
    mde_out.to_csv(csv_path, index=False)
    selected_columns = list(result['MDEcolumns'])
    print(f'MDE selected columns for {target}:', selected_columns, flush=True)

    m = _mde_namespace(selected_columns, np.asarray(result['MDErho']), mde_out,
                       [1, ccm_lib_end], [ccm_lib_end + 1, len(ccm_df)], tp)
    if fingerprint is not None:
        _save_selection(cache_filename, fingerprint, m, results_dir, run_cache_dir)

    return {'target': target, 'mde_df': ccm_df, 'm': m,
            'SELECTED_COLUMNS': selected_columns}


# =====================================================================
# Forward selection, a whole target group at once
# =====================================================================

def run_multi_mde_for_targets(targets, df, observable_columns, prediction_start,
                              prediction_length, mde_d, embed_dim_rho_min, cross_map_rho_min,
                              exclusion_radius, results_dir, delay_set, tp=1,
                              metric='f1', target_metric='correlation', gamma_delta=2,
                              ccm_sample=20, ccm_slope=0.01,
                              target_ranking_window_rows=1000,
                              ccm_lib_sizes_rows=(16, 24, 40, 64, 160, 400, 800),
                              reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                              data_file=None, data_file_sha256=None, code_fingerprint=None,
                              cache_stage='stage1', E=3, max_e=10, target_aggregate='mean',
                              use_gpu=False):
    '''Forward selection once for a whole target group -- GMN_MODE='combined'.

    Picks drivers that JOINTLY predict every target at once, instead of
    running one independent selection per target. Every argument means what
    it does in run_mde_for_target(); the genuinely new ones are `E`/`max_e`
    (E<=0 auto-selects per candidate, bounded by max_e) and
    `target_aggregate` ('mean' does not penalise a driver that is strong for
    one target and weak for another; 'min' accepts only drivers good for
    every target simultaneously).

    Returns the same dict shape, keyed 'targets' instead of 'target'. The
    joint selection produces no per-target breakdown by design -- these are
    the same drivers, accepted because they predict the GROUP -- so the
    caller fans this one result out into a per-target view (see pipeline.py)
    and every downstream reader that indexes by target keeps working.

    Every target here is behavioural (this is only ever the depth-0 seed),
    so there is no per-node event-count extension: one fixed window for the
    whole joint call.
    '''
    # Both skill gates are resolved HERE, before the cache fingerprint is
    # built and before proper_mde sees them, so a node cached under the f1
    # gate can never be reused for one scored by correlation -- the resolved
    # float is what goes into the fingerprint. `target_metric` is already the
    # node's own metric: every call site passes _metric_for(node, ...).
    embed_dim_rho_min = _rho_min_for_metric(embed_dim_rho_min, target_metric)
    cross_map_rho_min = _rho_min_for_metric(cross_map_rho_min, target_metric)
    mde_df, usable_df, lib_ratio = _leakage_safe_window(
        df, list(targets) + list(observable_columns),
        prediction_start, prediction_length, delay_set, tp)
    window_rows = min(target_ranking_window_rows, len(usable_df))

    targets_tag = '_'.join(targets)
    csv_path = results_dir / f'mde_output_{cache_stage}_multi_{targets_tag}.csv'
    cache_filename = f'mde_selection_{cache_stage}_multi_{targets_tag}.pkl'
    fingerprint = None
    if reuse_cache:
        fingerprint = build_multi_mde_selection_fingerprint(
            targets, observable_columns, mde_d, tp, embed_dim_rho_min, cross_map_rho_min,
            exclusion_radius, delay_set, prediction_start, prediction_length,
            metric, target_metric, gamma_delta,
            target_ranking_window_rows, window_rows, ccm_lib_sizes_rows,
            ccm_sample, ccm_slope,
            data_file, data_file_sha256, code_fingerprint, cache_stage, E, target_aggregate,
        )
        cached = _load_cached_selection(prev_runs_dir, cache_filename, fingerprint,
                                        csv_path, targets_tag)
        if cached is not None:
            m, selected_columns = cached
            print(f'Joint MDE selected columns for {list(targets)}:', selected_columns, flush=True)
            return {'targets': list(targets), 'mde_df': mde_df, 'm': m,
                    'SELECTED_COLUMNS': selected_columns}

    ccm_df, ccm_lib_end = _ccm_window(usable_df, lib_ratio, window_rows)

    result = edm.multi_proper_mde(
        ccm_df, targets=list(targets), columns=observable_columns, D=mde_d,
        lib=[1, ccm_lib_end], pred=[ccm_lib_end + 1, len(ccm_df)],
        Tp=tp, E=E, maxE=max_e, exclusionRadius=exclusion_radius,
        libSizes=ccm_lib_sizes_rows, sample=ccm_sample, ccmSlope=ccm_slope,
        crossMapRhoMin=cross_map_rho_min, embedDimRhoMin=embed_dim_rho_min,
        metric=target_metric, ccmMetric=metric, gammaDelta=gamma_delta,
        target_aggregate=target_aggregate,
        verbose=True, use_gpu=use_gpu,
    )

    mde_out = pd.DataFrame({'rho': result['MDErho']})
    mde_out.to_csv(csv_path, index=False)
    selected_columns = list(result['MDEcolumns'])
    print(f'Joint MDE selected columns for {list(targets)}:', selected_columns, flush=True)

    m = _mde_namespace(selected_columns, np.asarray(result['MDErho']), mde_out,
                       [1, ccm_lib_end], [ccm_lib_end + 1, len(ccm_df)], tp)
    if fingerprint is not None:
        _save_selection(cache_filename, fingerprint, m, results_dir, run_cache_dir)

    return {'targets': list(targets), 'mde_df': ccm_df, 'm': m,
            'SELECTED_COLUMNS': selected_columns}


# =====================================================================
# GPU memory failures
# =====================================================================

GPU_MEMORY_STOP_ADVICE = (
    'This candidate was NOT evaluated, and recording it as a dead end would make it '
    'indistinguishable from a measured absence. Raise --gres=gpu:N, lower '
    '--cpus-per-task, or reduce MAX_WINDOW_ROWS/CCM_SAMPLE, then resubmit -- every '
    'node finished so far is cached.'
)


def is_gpu_memory_failure(exc):
    """True when `exc` reports GPU memory exhaustion rather than a finding.

    EVERY caller that answers a failed MDE call with a dead-end result must
    ask this first, and re-raise when it says yes. A candidate dropped for
    want of memory is a MISSING EDGE, and downstream it is indistinguishable
    from a candidate that was measured and found absent -- so swallowing one
    does not lose a data point, it fabricates one. Job 4741044 lost 51 that
    way; job 4744295 recorded 20 more as dead ends even after gpu_budget had
    been added precisely to stop that.

    A genuine finding ("Maximum libSize 800 exceeds data size 703", no
    convergence) is NOT a memory failure and still falls through to the
    caller's dead-end handling. An OOM is not a finding about anything.

    gpu_budget sits beside the entry script rather than in this package and
    does not exist at all on a CPU-only deployment, so the import is lazy
    and a missing module simply means there is no GPU path to re-raise for.
    """
    try:
        from gpu_budget import GpuBudgetError, is_gpu_memory_error
    except ImportError:
        return False
    return isinstance(exc, GpuBudgetError) or is_gpu_memory_error(exc)


# =====================================================================
# CCM pre-filter
# =====================================================================

def _ccm_prefilter_pair(candidate, target, targets, windowed, e, tp, metric, target_metric,
                        gamma_delta, exclusion_radius, ccm_sample, lib_sizes):
    '''One (candidate, target) pair's screen. Runs in a thread (proper_ccm's
    compiled pyEDM/KDTree calls release the GIL). Returns (candidate,
    target, rho); rho is NaN on any failure, the same spirit as a dead-end
    MDE result elsewhere in this file.'''
    try:
        c2t, _, _ = edm.proper_ccm(
            windowed, libSizes=lib_sizes, sample=ccm_sample, tau=-1, Tp=tp, E=e,
            target=target, columns=candidate,
            target_measurement=_metric_for(target, targets, metric, target_metric),
            columns_measurement=_metric_for(candidate, targets, metric, target_metric),
            exclusionRadius=exclusion_radius, gammaDelta=gamma_delta,
            # Only the candidate -> target direction is a reason to keep a
            # candidate, and only c2t is read below.
            directions='c2t',
        )
        return candidate, target, float(np.nanmean(c2t, axis=0)[-1])
    except Exception:
        return candidate, target, float('nan')


def ccm_prefilter_observables(df, candidate_columns, targets, prediction_start,
                              rho_min, e, tp, metric, target_metric, gamma_delta,
                              exclusion_radius, ccm_window_rows, ccm_sample,
                              libsize_ratios, cores):
    '''Drop candidates that cannot be causal, before selection starts.

    A cheap CCM(observable -> target) screen, run once per target group
    rather than once per node. It covers both nested-MDE stages in one pass,
    because Stage 2's lagged candidates are drawn from this same filtered
    list of observables. Anything that fails the screen cannot be driving
    the target, so paying for it in every later forward selection is waste.

    Parameters are reused from elsewhere in the pipeline wherever one fits
    (Tp, the metrics, exclusionRadius, CCM_SAMPLE) or match proper_mde's own
    default (tau=-1). The one genuinely independent knob is `e`, from
    CCM_PREFILTER_E: at 40 Hz, one embedding step covers 1/40th the real
    time it covers in the 1 Hz sibling pipeline, so that pipeline's value
    would not transfer.

    The window matches every other CCM call in the pipeline:
    `df.iloc[:prediction_start]` (the leakage-safe library slice) narrowed
    by `_ccm_window(..., 0.5, ccm_window_rows)`. `libsize_ratios` are
    fractions of that narrowed window, converted here to row counts,
    deduped and floored at 10.

    Returns (kept_columns, scores) -- `kept_columns` is `candidate_columns`
    minus `targets`, filtered to those scoring >= `rho_min`, in original
    order; `scores` is {candidate: mean_rho} for everything considered,
    including what was dropped, for diagnostics.

    The assertion guards an invariant rather than a live bug: this function
    must only ever see real (delay-0) observables. Every column it is handed
    today traces back to convert_to_tsls()'s own naming, and delay expansion
    happens strictly later in the run -- the guard is here so a future change
    that reorders those cannot quietly feed lagged columns to a screen whose
    results are then reused for every lag.
    '''
    lagged = [c for c in candidate_columns if DELAY_MARKER in c]
    assert not lagged, (
        f'ccm_prefilter_observables() received DELAY_MARKER-suffixed column(s): '
        f'{lagged} -- candidate_columns must be real (delay-0) observables only')

    candidates = [c for c in candidate_columns if c not in targets]
    if not candidates:
        return [], {}

    windowed, _ = _ccm_window(df.iloc[:prediction_start].reset_index(drop=True),
                              0.5, ccm_window_rows)
    lib_sizes = sorted({max(10, round(r * len(windowed))) for r in libsize_ratios})

    per_candidate = {c: [] for c in candidates}
    with ThreadPoolExecutor(max_workers=max(1, cores)) as executor:
        futures = [
            executor.submit(_ccm_prefilter_pair, c, t, targets, windowed, e, tp, metric,
                            target_metric, gamma_delta, exclusion_radius, ccm_sample, lib_sizes)
            for c in candidates for t in targets
        ]
        for future in as_completed(futures):
            candidate, _target, rho = future.result()
            per_candidate[candidate].append(rho)

    scores = {c: float(np.nanmean(v)) if v else float('nan')
              for c, v in per_candidate.items()}
    kept = [c for c in candidates if scores.get(c, float('nan')) >= rho_min]
    return kept, scores


# =====================================================================
# Diagnostic figures
# =====================================================================

def plot_mde_rho_vs_dimension(mde_results, targets, target_colors, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for target in targets:
        mde_out = mde_results[target]['m'].MDEOut
        ax.plot(np.arange(1, len(mde_out) + 1), mde_out['rho'],
                marker='o', label=target, color=target_colors[target])
    ax.set_xlabel('Embedding dimension')
    ax.set_ylabel('Cross-map rho')
    ax.set_title('MDE rho vs. dimension, per target')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_3d_mde_projection(target, second_target, mde_results, df, raw_columns,
                           prediction_start, prediction_length, save_path):
    '''Trajectory of `target` against its top-2 MDE-selected observables.

    SELECTED_COLUMNS can hold delay-suffixed names that do not exist in the
    raw frame this indexes, so each is mapped back to its underlying real
    observable -- the delay is not meaningful for a visual trajectory of the
    driver's own values.
    '''
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

    obs = list(dict.fromkeys(
        observable_and_delay(c, raw_columns)[0]
        for c in mde_results[target]['SELECTED_COLUMNS']
    ))[:2]
    if len(obs) < 2:
        print(f'{target}: fewer than 2 MDE-selected columns, skipping 3D projection.',
              flush=True)
        return

    window = df.iloc[:prediction_start + prediction_length]
    if second_target and second_target in window.columns:
        color_vals, color_label = window[second_target].to_numpy(), second_target
    else:
        color_vals, color_label = np.arange(len(window)), 'time index'

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    x, y, z = (window[target].to_numpy(),
               window[obs[0]].to_numpy(), window[obs[1]].to_numpy())
    ax.plot(x, y, z, color='gray', linewidth=0.6, alpha=0.4)
    scatter = ax.scatter(x, y, z, c=color_vals, cmap='viridis', s=10)
    ax.set_xlabel(target)
    ax.set_ylabel(obs[0])
    ax.set_zlabel(obs[1])
    ax.set_title(f'MDE projection: {target} vs. its top-2 observables')
    fig.colorbar(scatter, ax=ax, shrink=0.7, label=color_label)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
