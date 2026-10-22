"""Multiview Delay Embedding (MDE) for one node: which signals predict it?

This is step 2 of the pipeline. Given one target observable and a pool of
candidate drivers (other observables, and delayed copies of observables), it
answers: which small set of candidates best predicts the target?

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

The functions here:
    run_mde_for_target        one target, forward selection + CCM gate
    run_multi_mde_for_targets the same for several targets in one call, so a
                              shared embedding is computed once
    ccm_prefilter_observables a cheap CCM screen run BEFORE selection, to drop
                              candidates that cannot be causal at all
    plot_mde_*                diagnostic figures for a finished selection
    _failed_*_result          the empty result shape used when a target has too
                              little usable data, so callers need no special case

Results are cached (see caching.py): an MDE call is the expensive part of the
pipeline, so a completed selection is reused across jobs.

Depends on: cleaned_helper_functions (the EDM engine), columns, tsls_data,
caching; numpy, pandas, matplotlib.
Imported by: network.py, network_parallel.py, and the entry scripts.

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

import cleaned_helper_functions as tsls
from mde_gmn.caching import (
    _load_pickle_cache_entry, build_mde_selection_fingerprint,
    build_multi_mde_selection_fingerprint, find_cached, save_pickle_to_cache)
from mde_gmn.columns import DELAY_MARKER, observable_and_delay
from mde_gmn.tsls_data import _ccm_window, _metric_for, _resolve_target_ranking_window


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

    Runs cleaned_helper_functions.proper_mde and returns its result wrapped in
    an 'm' namespace (MDEcolumns/MDErho/MDEOut/args.lib/args.pred/args.tau/
    args.Tp). Every consumer -- _accept_mde_result_edges(),
    _failed_mde_result(), plot_mde_rho_vs_dimension(), run_tp_vs_rho_sweep(),
    run_univariate_e_tau_grid() -- expects that exact shape, which is why
    _failed_mde_result() has to produce it too.

    `mde_d` here is already the *effective* dimension count (see
    effective_mde_dimension()) -- NOT what `max_e` should be capped at (see
    below), since Stage 2 inflates `mde_d` by num_delays for a completely
    unrelated reason (candidate-pool breadth, not per-candidate embedding
    depth).

    E AND TAU ARE NOT FIXED HERE. E=0 means proper_mde auto-selects the
    embedding dimension per candidate, scored with whichever metric that call
    uses, capped at maxE=`max_e`. `max_e` is the RAW (un-scaled-by-num_delays)
    MDE_D value: the per-node driver budget the pipeline already exposes,
    reused rather than inventing a second ceiling.
    tau is simply no longer passed -- proper_mde's own default (-1) applies,
    same numeric value as before, just not redundantly re-pinned here.

    metric vs. ccmMetric passed to proper_mde: `target_metric` scores the
    RANKING stage (predictions of `target` -- a continuous TARGETS column
    at the depth-0 seed, a neuron at every Stage 1/Stage 2 node -- never
    `metric`) AND the E-auto-select sweep above (same reasoning: it also
    predicts `target`); `metric` scores the CCM causality gate (predictions
    of each CANDIDATE column, always a TSLS-converted neuron series) -- see
    load_config()'s TARGET_METRIC docstring and this file's _metric_for().
    Callers are responsible for resolving `target_metric` correctly for
    whatever `target` actually is here (`_metric_for(target, targets,
    metric, target_metric)` at the depth>=1 call sites -- see the
    _metric_for() resolves it correctly for a neuron node, which needs 'f1'
    rather than the targets' 'correlation').

    WINDOW SIZING. The ranking-stage/E-auto-select window is NOT
    CCM_WINDOW_ROWS; it is resolved per node by
    `_resolve_target_ranking_window()` (see that function's own docstring),
    decoupled entirely from the CCM gate's own convergence-curve library
    sizing (`ccm_lib_sizes_rows`, absolute row counts, never a percentage
    of this window -- see that parameter's own docstring below for why).
    `is_behavioral_target=True` (the depth-0 seed's own case): a fixed
    `target_ranking_window_rows` (default 1000), no event-count concept.
    `is_behavioral_target=False` (virtually every Stage 1/Stage 2 node):
    extended until `target`'s own PRED portion of the window (the part
    `f1`-scoring actually touches, not the window as a whole) contains at
    least `min_pred_events` (default 20, empirically calibrated -- see
    load_config()'s own docstring) real reset events, capped at
    `max_window_rows` (default 30000, a plain compute ceiling, not
    calibrated) -- see `_resolve_target_ranking_window()`'s own docstring.

    LEAKAGE (see config.py's three-way A/B/end window entry):
    `prediction_start`/`prediction_length` here are always A and B-A. The
    row range handed to proper_mde is trimmed twice before being resolved
    to this node's own ranking window (see _resolve_target_ranking_window()
    and _ccm_window()): `warmup` rows are skipped at the front (delay-
    expansion NaN warm-up) and `tp` rows are
    trimmed from the back (B - tp, not B, so the last validated row's own
    t+tp target label can never land at or past B) -- both applied BEFORE
    the window is resolved, so neither the fixed depth-0 window nor the
    extended per-node neuron window can ever reach into the Generate/
    Forecast validation segment [B, end), regardless of how large either
    grows.

    CACHE (reuse_cache, gated by TSLS_MDE_GMN_REUSE_BEST_PARAMS): a cache hit
    reconstructs the same 'm' namespace from a pickled dict instead of calling
    proper_mde at all. `cache_stage` ('stage1' by default, 'stage2' from the
    builder's Stage 2 pass) keeps the two stages' cache files distinct.

    INNER POOL (inner_workers, default 1 -- no pool at all): this call's
    own proper_mde() forward-selection is otherwise entirely single-core
    (see this module's own docstring -- proper_mde has no multiprocessing
    of its own). `inner_workers` opens exactly ONE ThreadPoolExecutor here,
    for this node's whole proper_mde() call -- reused across every
    dimension's ranking loop and every CCM-gate candidate inside that one
    call, then closed when this function returns -- and passes it down as
    proper_mde's own `pool` argument (see that function's docstring for
    why threads, not another process pool: every hot loop it reaches is a
    compiled pyEDM/scipy call that releases the GIL). Callers
    (build_nested_mde_network()'s own Stage 1 and Stage 2 passes) size
    this as `max(1, cores // outer_workers)`, recomputed fresh per wave
    from that wave's own actual size -- see that function's own docstring
    -- which is what keeps outer_workers * inner_workers <= cores always. That
    invariant matters: nested pools that each claim all the cores multiply
    rather than share, and this pipeline has been brought down that way before.
    No TSLS-side
    equivalent of that bug ever existed here (proper_mde never had any
    multiprocessing at all until this parameter) -- this inner pool is
    new, sized the same careful way from the start.

    `raw_columns`: when given, builds a
    `{column: delay}` lookup via `observable_and_delay()` for every column
    in `observable_columns` and passes it to `proper_mde` as
    `column_delay`, enabling that function's own dynamic per-dimension
    exclusion radius (see its docstring). `None` (default, and what
    main()'s own depth-0 seed call passes -- that pool is always
    real-only, so there is no delay to look up) -> `column_delay=None` ->
    proper_mde's own no-op default. `build_nested_mde_network()`'s Stage 1
    and Stage 2 calls both pass `raw_columns=raw_columns` (their
    `observable_columns`/`candidate_pool` genuinely can contain delay-suffixed
    synthetic names).

    `use_gpu`: passed straight through to `proper_mde`. False (the default)
    behaves exactly as before the GPU path existed; True changes anything only
    on a machine with a visible CUDA device, and falls back to the CPU
    transparently otherwise. It arrives via cfg['GPU_ENABLED'], down the same
    chain that carries `mp_start_method` (main -> build_network -> the builder
    -> here) rather than a flag of its own.

    `ccm_lib_sizes_rows`: explicit, absolute row counts for the CCM gate's own convergence-curve
    library sweep, passed to `proper_mde` as `libSizes` (bypassing its own
    `pLibSizes`-as-percentage-of-window computation entirely). Deliberately
    NOT a percentage of this node's own (now node-specific, potentially
    much larger) ranking window. The CCM convergence curve's characteristic
    "rising, then plateauing" shape turns out to be a property of the ABSOLUTE
    library size, not of its ratio to the surrounding frame: calibrating at an
    800-row frame and at an 8000-row frame gave the same qualitative shape for
    the same absolute values. See config.py's CCM_LIB_SIZES_ROWS entry for the
    calibrated default.'''
    # Fallback only -- every real call site (main()'s depth-0 seed, and
    # both of build_nested_mde_network()'s per-wave passes) passes this
    # explicitly as the raw TSLS_MDE_GMN_MDE_D value. Falls back to mde_d
    # (the effective, possibly num_delays-scaled dimension) only if a
    # caller genuinely omits it, matching this function's own pre-max_e
    # behavior for that edge case rather than raising.
    if max_e is None:
        max_e = mde_d

    mde_df = df[[target] + observable_columns].iloc[:prediction_start + prediction_length]

    # See this function's own LEAKAGE docstring above for why both trims
    # happen before windowing, not after.
    warmup = max((abs(d) for d in delay_set), default=0)
    usable_end = prediction_start + prediction_length - tp

    usable_df = mde_df.iloc[warmup:usable_end].reset_index(drop=True)
    lib_ratio = (prediction_start - warmup) / max(1, usable_end - warmup)

    # Resolved BEFORE the cache fingerprint below: both the window size and
    # (for a neuron target) pred_event_count are real, data-dependent inputs to
    # this call's outcome, so they have to be part of its cache key.
    window_rows, pred_event_count = _resolve_target_ranking_window(
        target, df, usable_df, lib_ratio,
        is_behavioral_target, target_ranking_window_rows, min_pred_events, max_window_rows,
    )
    if window_rows is None:
        shortfall = min_pred_events - pred_event_count
        cap = min(max_window_rows, len(usable_df))
        print(f'{target}: cannot reach MIN_PRED_EVENTS={min_pred_events} in the pred portion '
              f'of any window up to min(MAX_WINDOW_ROWS, leakage-safe range)={cap} rows -- '
              f'best achievable was {pred_event_count} events (short by {shortfall}). '
              f'Treating as a dead-end (zero drivers selected, no further nesting from this '
              f'node) rather than scoring it on too few events.', flush=True)
        return _failed_mde_result(target)
    if not is_behavioral_target:
        print(f'{target}: resolved ranking window={window_rows} rows, '
              f'pred_event_count={pred_event_count} (MIN_PRED_EVENTS={min_pred_events})',
              flush=True)

    fingerprint = None
    cache_filename = f'mde_selection_{cache_stage}_{target}.pkl'
    if reuse_cache:
        fingerprint = build_mde_selection_fingerprint(
            target, observable_columns, mde_d, tp, embed_dim_rho_min, cross_map_rho_min,
            exclusion_radius, delay_set, prediction_start, prediction_length,
            metric, target_metric, gamma_delta,
            is_behavioral_target, target_ranking_window_rows, min_pred_events, max_window_rows,
            window_rows, pred_event_count, ccm_lib_sizes_rows,
            ccm_sample, ccm_slope,
            data_file, data_file_sha256, code_fingerprint, cache_stage, max_e,
        )
        hit = find_cached(prev_runs_dir, cache_filename, fingerprint,
                           load_one=_load_pickle_cache_entry)
        if hit is not None:
            entry, source = hit
            print(f'{target}: TSLS_MDE_GMN_REUSE_BEST_PARAMS matched cached MDE selection '
                  f'{source} -- skipping forward-selection entirely', flush=True)
            m = types.SimpleNamespace(
                MDEcolumns=entry['MDEcolumns'], MDErho=entry['MDErho'], MDEOut=entry['MDEOut'],
                args=types.SimpleNamespace(lib=entry['lib'], pred=entry['pred'],
                                            tau=entry['tau'], Tp=entry['Tp']),
            )
            m.MDEOut.to_csv(results_dir / f'mde_output_{cache_stage}_{target}.csv', index=False)
            selected_columns = list(m.MDEcolumns)
            print(f'MDE selected columns for {target}:', selected_columns, flush=True)
            return {
                'target': target,
                'mde_df': mde_df,
                'm': m,
                'SELECTED_COLUMNS': selected_columns,
            }
        print(f'{target}: TSLS_MDE_GMN_REUSE_BEST_PARAMS set but no matching cached MDE selection '
              f'found under {prev_runs_dir} -- running forward-selection.', flush=True)

    ccm_df, ccm_lib_end = _ccm_window(usable_df, lib_ratio, window_rows)

    # One inner pool for this node's WHOLE proper_mde() call -- opened
    # once here, reused across every dimension's ranking loop and every
    # CCM-gate candidate inside that one call, closed on exit. See this
    # function's own INNER POOL docstring above for the outer_workers *
    # inner_workers <= cores invariant this relies on, and why threads
    # (nullcontext()/None when inner_workers <= 1, so this is a no-op
    # for any caller that doesn't pass it -- identical to before this
    # parameter existed).
    inner_pool_ctx = (
        ThreadPoolExecutor(max_workers=inner_workers) if inner_workers > 1
        else nullcontext()
    )
    # column_delay lookup for proper_mde's
    # own dynamic per-dimension exclusion radius -- only meaningful when
    # `observable_columns` can contain delay-suffixed synthetic names
    # (build_nested_mde_network()'s Stage 1/Stage 2 candidate pools; see
    # that function's own call sites, which pass raw_columns=raw_columns).
    # `raw_columns=None` (main()'s own depth-0 seed call, always a
    # real-only pool) -> column_delay=None -> proper_mde's own no-op
    # default, identical to before this parameter existed.
    column_delay = (
        {c: observable_and_delay(c, raw_columns)[1] for c in observable_columns}
        if raw_columns is not None else None
    )
    # Diagnostic only, added while investigating a very slow Stage 2 call --
    # same raw_columns-gated pattern as
    # column_delay above, just the observable half of observable_and_delay()'s
    # own return value instead of the delay half. Lets proper_mde's own
    # verbose log show how many *distinct* real observables a long run has
    # actually found, alongside how many dimensions it took to find them.
    column_observable = (
        {c: observable_and_delay(c, raw_columns)[0] for c in observable_columns}
        if raw_columns is not None else None
    )

    with inner_pool_ctx as inner_pool:
        result = tsls.proper_mde(
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
    mde_out.to_csv(results_dir / f'mde_output_{cache_stage}_{target}.csv', index=False)

    selected_columns = list(result['MDEcolumns'])
    print(f'MDE selected columns for {target}:', selected_columns, flush=True)

    m = types.SimpleNamespace(
        MDEcolumns=selected_columns,
        MDErho=np.asarray(result['MDErho']),
        MDEOut=mde_out,
        args=types.SimpleNamespace(
            lib=[1, ccm_lib_end], pred=[ccm_lib_end + 1, len(ccm_df)], tau=-1, Tp=tp,
        ),
    )

    if fingerprint is not None:
        save_pickle_to_cache(
            cache_filename,
            {
                'fingerprint': fingerprint,
                'MDEcolumns': m.MDEcolumns, 'MDErho': m.MDErho, 'MDEOut': m.MDEOut,
                'lib': m.args.lib, 'pred': m.args.pred, 'tau': m.args.tau, 'Tp': m.args.Tp,
            },
            results_dir, run_cache_dir,
        )

    return {
        'target': target,
        'mde_df': ccm_df,
        'm': m,
        'SELECTED_COLUMNS': selected_columns,
    }

def run_multi_mde_for_targets(targets, df, observable_columns, prediction_start, prediction_length,
                               mde_d, embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                               results_dir, delay_set, tp=1,
                               metric='f1', target_metric='correlation', gamma_delta=2,
                               ccm_sample=20, ccm_slope=0.01,
                               target_ranking_window_rows=1000,
                               ccm_lib_sizes_rows=(16, 24, 40, 64, 160, 400, 800),
                               reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                               data_file=None, data_file_sha256=None, code_fingerprint=None,
                               cache_stage='stage1', E=3, max_e=10, target_aggregate='mean',
                               use_gpu=False):
    '''Multi-target counterpart of run_mde_for_target() -- GMN_MODE='combined'
    only (see load_config()'s MULTI_MDE_E docstring and main()'s own MDE
    block). Runs cleaned_helper_functions.multi_proper_mde's forward
    selection ONCE for the whole `targets` group, picking drivers that
    jointly predict every target at once, instead of one independent
    proper_mde call per target. Every other argument has the identical
    meaning to run_mde_for_target()'s own; `E`/`max_e` (E<=0 auto-selects
    per candidate via multi_proper_embed_dimension(), bounded by max_e --
    see multi_proper_mde's own docstring; E>0 is still the static,
    every-candidate-identical original behavior) and `target_aggregate`
    are the genuinely new knobs.

    Returns the same dict shape as run_mde_for_target() except keyed by
    `'targets'` (list) instead of a single `'target'` -- main() builds one
    per-target *view* of this single shared result (same 'm'/'mde_df'/
    'SELECTED_COLUMNS', just re-keyed with a 'target' entry) so every
    downstream consumer that indexes mde_results[target] (network
    construction's depth-0 seed, the MDE diagnostics plots/sweeps) keeps
    working completely unchanged -- see main()'s own MDE block. This is
    also why `mde_df`/`SELECTED_COLUMNS`/edge weights end up identical
    (shared) across every target in the group: the joint selection doesn't
    produce a per-target breakdown, by design (it's the same drivers,
    accepted because they predict the *group*, not any one target alone).

    `use_gpu` (docs/no-kEDM-gpu-optimization-plan.md section 9): threaded
    straight through to `multi_proper_mde`'s own `use_gpu` -- see that
    function's docstring, and run_mde_for_target()'s own identical
    `use_gpu` docstring for the general contract (transparent CPU
    fallback, byte-for-byte unchanged when False). This is the call this
    section 9 follow-up exists for: GMN_MODE='combined' (the pipeline
    default) reaches its own CCM gate through multi_proper_ccm via this
    function, not through run_mde_for_target()'s proper_ccm -- section 8's
    work alone left this path unaccelerated. Callers pass
    `cfg['GPU_ENABLED']`, same source as every other use_gpu in this file.
    '''
    mde_df = df[list(targets) + list(observable_columns)].iloc[:prediction_start + prediction_length]

    # See run_mde_for_target()'s own LEAKAGE docstring -- identical windowing
    # rationale, just against a dataFrame carrying every target column
    # instead of one.
    warmup = max((abs(d) for d in delay_set), default=0)
    usable_end = prediction_start + prediction_length - tp
    usable_df = mde_df.iloc[warmup:usable_end].reset_index(drop=True)
    lib_ratio = (prediction_start - warmup) / max(1, usable_end - warmup)

    # Every target here is always behavioural (GMN_MODE='combined''s depth-0 seed only, per this
    # function's own docstring) -- no per-node event-count extension, just
    # the same fixed constant _resolve_target_ranking_window() uses for a
    # single behavioral target, applied once for the whole joint call.
    window_rows = min(target_ranking_window_rows, len(usable_df))

    targets_tag = '_'.join(targets)
    fingerprint = None
    cache_filename = f'mde_selection_{cache_stage}_multi_{targets_tag}.pkl'
    if reuse_cache:
        fingerprint = build_multi_mde_selection_fingerprint(
            targets, observable_columns, mde_d, tp, embed_dim_rho_min, cross_map_rho_min,
            exclusion_radius, delay_set, prediction_start, prediction_length,
            metric, target_metric, gamma_delta,
            target_ranking_window_rows, window_rows, ccm_lib_sizes_rows,
            ccm_sample, ccm_slope,
            data_file, data_file_sha256, code_fingerprint, cache_stage, E, target_aggregate,
        )
        hit = find_cached(prev_runs_dir, cache_filename, fingerprint,
                           load_one=_load_pickle_cache_entry)
        if hit is not None:
            entry, source = hit
            print(f'{targets_tag}: TSLS_MDE_GMN_REUSE_BEST_PARAMS matched cached joint MDE '
                  f'selection {source} -- skipping forward-selection entirely', flush=True)
            m = types.SimpleNamespace(
                MDEcolumns=entry['MDEcolumns'], MDErho=entry['MDErho'], MDEOut=entry['MDEOut'],
                args=types.SimpleNamespace(lib=entry['lib'], pred=entry['pred'],
                                            tau=entry['tau'], Tp=entry['Tp']),
            )
            m.MDEOut.to_csv(results_dir / f'mde_output_{cache_stage}_multi_{targets_tag}.csv', index=False)
            selected_columns = list(m.MDEcolumns)
            print(f'Joint MDE selected columns for {list(targets)}:', selected_columns, flush=True)
            return {
                'targets': list(targets),
                'mde_df': mde_df,
                'm': m,
                'SELECTED_COLUMNS': selected_columns,
            }
        print(f'{targets_tag}: TSLS_MDE_GMN_REUSE_BEST_PARAMS set but no matching cached joint '
              f'MDE selection found under {prev_runs_dir} -- running forward-selection.', flush=True)

    ccm_df, ccm_lib_end = _ccm_window(usable_df, lib_ratio, window_rows)

    result = tsls.multi_proper_mde(
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
    mde_out.to_csv(results_dir / f'mde_output_{cache_stage}_multi_{targets_tag}.csv', index=False)

    selected_columns = list(result['MDEcolumns'])
    print(f'Joint MDE selected columns for {list(targets)}:', selected_columns, flush=True)

    m = types.SimpleNamespace(
        MDEcolumns=selected_columns,
        MDErho=np.asarray(result['MDErho']),
        MDEOut=mde_out,
        args=types.SimpleNamespace(
            lib=[1, ccm_lib_end], pred=[ccm_lib_end + 1, len(ccm_df)], tau=-1, Tp=tp,
        ),
    )

    if fingerprint is not None:
        save_pickle_to_cache(
            cache_filename,
            {
                'fingerprint': fingerprint,
                'MDEcolumns': m.MDEcolumns, 'MDErho': m.MDErho, 'MDEOut': m.MDEOut,
                'lib': m.args.lib, 'pred': m.args.pred, 'tau': m.args.tau, 'Tp': m.args.Tp,
            },
            results_dir, run_cache_dir,
        )

    return {
        'targets': list(targets),
        'mde_df': ccm_df,
        'm': m,
        'SELECTED_COLUMNS': selected_columns,
    }

def plot_mde_rho_vs_dimension(mde_results, targets, target_colors, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for target in targets:
        mde_out = mde_results[target]['m'].MDEOut
        dims = np.arange(1, len(mde_out) + 1)
        ax.plot(dims, mde_out['rho'], marker='o', label=target, color=target_colors[target])
    ax.set_xlabel('Embedding dimension')
    ax.set_ylabel('Cross-map rho')
    ax.set_title('MDE rho vs. dimension, per target')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

def plot_3d_mde_projection(target, second_target, mde_results, df, raw_columns,
                            prediction_start, prediction_length, save_path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

    # SELECTED_COLUMNS can now be delay-suffixed names (see
    # expand_delay_columns()'s docstring) that don't exist in the raw `df`
    # this function indexes into -- mapped back to their underlying raw
    # observable (the delay itself isn't meaningful for this plot, a visual
    # trajectory of the real, undelayed driver values).
    obs = list(dict.fromkeys(
        observable_and_delay(c, raw_columns)[0] for c in mde_results[target]['SELECTED_COLUMNS']
    ))[:2]
    if len(obs) < 2:
        print(f'{target}: fewer than 2 MDE-selected columns, skipping 3D projection.', flush=True)
        return

    window = df.iloc[:prediction_start + prediction_length]
    x = window[target].to_numpy()
    y = window[obs[0]].to_numpy()
    z = window[obs[1]].to_numpy()

    if second_target and second_target in window.columns:
        color_vals = window[second_target].to_numpy()
        color_label = second_target
    else:
        color_vals = np.arange(len(window))
        color_label = 'time index'

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
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

def is_gpu_memory_failure(exc):
    """True when `exc` reports GPU memory exhaustion rather than a finding.

    EVERY caller that answers a failed MDE call with _failed_mde_result()
    must ask this first, and re-raise when it says yes. A candidate dropped
    for want of memory is a MISSING EDGE, and downstream it is
    indistinguishable from a candidate that was measured and found absent --
    so swallowing one does not lose a data point, it fabricates one. Job
    4741044 lost 51 that way; job 4744295 recorded 20 more as dead ends
    even though gpu_budget had already raised precisely to stop that.

    A genuine finding ("Maximum libSize 800 exceeds data size 703", no
    convergence) is NOT a memory failure and still falls through to the
    caller's own dead-end handling. An OOM is not a finding about anything.

    gpu_budget lives beside the entry script rather than in this package
    and does not exist at all on the CPU-only branch, so the import is
    lazy and a missing module simply means there is no GPU path here and
    nothing to re-raise.
    """
    try:
        from gpu_budget import GpuBudgetError, is_gpu_memory_error
    except ImportError:
        return False
    return isinstance(exc, GpuBudgetError) or is_gpu_memory_error(exc)


_GPU_MEMORY_STOP_ADVICE = (
    'This candidate was NOT evaluated, and recording it as a dead end would make it '
    'indistinguishable from a measured absence. Raise --gres=gpu:N, lower '
    '--cpus-per-task, or reduce MAX_WINDOW_ROWS/CCM_SAMPLE, then resubmit -- every '
    'node finished so far is cached.'
)


def _failed_mde_result(target):
    '''Fallback run_mde_for_target()-shaped result for a node/target whose
    MDE call raised, instead of returning normally (job 37946756: dimx's own internal EmbedDimension hit
    an all-NaN rho column; proper_mde has its own analogous internal
    failure modes against degenerate candidate data, not something this
    pipeline's own gates catch or control). Shaped to be indistinguishable,
    to every downstream consumer, from a node that legitimately selected
    zero drivers -- already one of build_nested_mde_network()'s documented
    ordinary BFS termination conditions -- so a single degenerate node
    dead-ends cleanly instead of the exception propagating out of a
    ProcessPoolExecutor wave and crashing everything already computed
    alongside it.'''
    return {
        'target': target,
        'mde_df': None,
        'm': types.SimpleNamespace(
            MDEcolumns=[], MDErho=np.array([]), MDEOut=None,
            args=types.SimpleNamespace(lib=None, pred=None, tau=None, Tp=None),
        ),
        'SELECTED_COLUMNS': [],
    }

def _failed_multi_mde_result(targets):
    '''Multi-target counterpart of _failed_mde_result() -- same shape as
    run_multi_mde_for_targets()'s own return (``'targets'`` list instead of
    a single ``'target'``), used when GMN_MODE='combined''s joint
    multi_proper_mde call raises instead of returning normally.'''
    return {
        'targets': list(targets),
        'mde_df': None,
        'm': types.SimpleNamespace(
            MDEcolumns=[], MDErho=np.array([]), MDEOut=None,
            args=types.SimpleNamespace(lib=None, pred=None, tau=None, Tp=None),
        ),
        'SELECTED_COLUMNS': [],
    }

def _ccm_prefilter_pair(candidate, target, targets, windowed, e, tp, metric, target_metric,
                         gamma_delta, exclusion_radius, ccm_sample, lib_sizes):
    '''One (candidate, target) pair's worth of ccm_prefilter_observables()
    work -- run in an outer ThreadPoolExecutor worker (see that function's
    docstring for why threads: proper_ccm's compiled pyEDM/KDTree calls
    release the GIL, already established for run_mde_for_target()'s own
    inner_workers pool). `targets` (the full group, not just `target`) is
    passed straight through to _metric_for() so it correctly classifies any
    column that happens to be one of the group's own targets, matching
    build_interaction_matrix()'s own usage. Returns (candidate, target,
    rho) -- rho is NaN on any failure (degenerate column, etc.), same
    spirit as _failed_mde_result() elsewhere in this file.'''
    try:
        c2t, _, _ = tsls.proper_ccm(
            windowed, libSizes=lib_sizes, sample=ccm_sample, tau=-1, Tp=tp, E=e,
            target=target, columns=candidate,
            target_measurement=_metric_for(target, targets, metric, target_metric),
            columns_measurement=_metric_for(candidate, targets, metric, target_metric),
            exclusionRadius=exclusion_radius, gammaDelta=gamma_delta,
        )
        rho = float(np.nanmean(c2t, axis=0)[-1])
    except Exception:
        rho = float('nan')
    return candidate, target, rho

def ccm_prefilter_observables(df, candidate_columns, targets, prediction_start,
                               rho_min, e, tp, metric, target_metric, gamma_delta,
                               exclusion_radius, ccm_window_rows, ccm_sample,
                               libsize_ratios, cores):
    '''Drop candidates that cannot be causal, before selection starts.

    A cheap CCM(observable -> target) screen, run once per target group rather
    than once per node, right before either nested-MDE stage. It covers Stage 1
    and Stage 2 in one pass, because Stage 2's lagged candidates are drawn from
    this same filtered list of observables. Anything that fails the screen
    cannot be driving the target, so paying for it in every later forward
    selection is wasted.

    Every CCM-call parameter here is either reused from an existing config
    value used elsewhere in this pipeline (Tp, metric/target_metric/
    gamma_delta via the existing _metric_for() convention, exclusionRadius,
    sample via CCM_SAMPLE) or, where nothing existing fits, either matches
    proper_mde's own unoverridden default (`tau`: confirmed "tau is simply
    no longer passed -- proper_mde's own default (-1) applies",
    run_mde_for_target()'s own docstring) or is its own new, deliberately
    pipeline-specific parameter (`e`, from CCM_PREFILTER_E -- NOT a reuse of
    MULTI_MDE_E, because 40 Hz data needs a different static E from 1 Hz data
    to cover a comparable span of real time at the same tau=-1 -- see config.py's
    CCM_PREFILTER_E entry).

    `windowed` mirrors build_interaction_matrix()'s own windowing exactly:
    `df.iloc[:prediction_start]` (the same leakage-safe MDE library slice
    build_network()'s own `data_after_mde_lib` uses, just computed directly
    from the raw candidate pool here since `selected_union` doesn't exist
    yet at this point in build_network()) through `_ccm_window(..., 0.5,
    ccm_window_rows)` -- the *same* CCM_WINDOW_ROWS-bounded sub-window every
    other CCM call in this pipeline already goes through. `libsize_ratios`
    (fractions of `len(windowed)`, i.e. of this same bounded window) are
    converted to actual row-count libSizes here, deduped and floored at 10
    rows -- same floor convention build_interaction_matrix()'s own
    CCM_LIB_SIZE_PCTS conversion uses.

    Parallelism: one outer ThreadPoolExecutor (sized by `cores`) over
    (candidate, target) pairs -- see _ccm_prefilter_pair()'s own docstring
    for why threads are safe/effective here.

    Returns (kept_columns, scores) -- `kept_columns` is `candidate_columns`
    (with `targets` excluded) filtered down to those scoring >= `rho_min`,
    in original order; `scores` is {candidate: mean_rho} for every
    candidate considered (including dropped ones), for diagnostics.

    audited main()'s call site and traced
    every column this function ever sees back to convert_to_tsls()'s own
    `f'{n}_tsls'` construction -- real channel names, never containing
    DELAY_MARKER, and expand_delay_columns() (the only place a
    DELAY_MARKER-suffixed name is ever created) is never called before
    this point in main() (it only runs inside build_nested_mde_network(),
    itself called from build_network(), which runs strictly after every
    call to this function). No cached mde_results entry feeds this
    function's candidate list either -- it always comes straight from
    main()'s own `obs_cols`. Confirmed clean; this assertion is kept as a
    cheap, permanent guard against a future change reopening that path,
    not a fix for a live issue.'''
    bad = [c for c in candidate_columns if DELAY_MARKER in c]
    assert not bad, (
        f'ccm_prefilter_observables() received DELAY_MARKER-suffixed column(s): {bad} -- '
        f'candidate_columns must be real (delay-0) observables only'
    )
    candidates = [c for c in candidate_columns if c not in targets]
    if not candidates:
        return [], {}

    usable_df = df.iloc[:prediction_start].reset_index(drop=True)
    windowed, _ = _ccm_window(usable_df, 0.5, ccm_window_rows)
    lib_sizes = sorted({max(10, round(r * len(windowed))) for r in libsize_ratios})

    per_candidate_scores = {c: [] for c in candidates}
    with ThreadPoolExecutor(max_workers=max(1, cores)) as executor:
        futures = [
            executor.submit(_ccm_prefilter_pair, c, t, targets, windowed, e, tp, metric,
                             target_metric, gamma_delta, exclusion_radius, ccm_sample, lib_sizes)
            for c in candidates for t in targets
        ]
        for fut in as_completed(futures):
            candidate, _target, rho = fut.result()
            per_candidate_scores[candidate].append(rho)

    scores = {c: float(np.nanmean(v)) if v else float('nan') for c, v in per_candidate_scores.items()}
    kept = [c for c in candidates if scores.get(c, float('nan')) >= rho_min]
    return kept, scores
