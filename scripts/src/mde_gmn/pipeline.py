"""Orchestration: main() -- the run, start to finish.

Read this file to find out what actually happens when a job starts. It calls
the other modules in order and owns nothing else: every step's real work lives
in the module named for it, and main() is the list of steps with the
bookkeeping between them (caching, logging, writing results out).

WHAT main() DOES
    1. load the configuration (config.load_config) and fingerprint the code and
       data, so a cache entry can only be reused by an identical run
    2. read the recording and convert it (tsls_data)
    3. screen the candidate pool with a cheap CCM pre-filter (mde), once per
       target group rather than once per node
    4. build the network (gmn_build.build_network, which runs the MDE stages
       through `network_builder`)
    5. tune theta per node (theta.run_theta_search)
    6. choose where the free run starts, rerun at the winning parameters, and
       score the result (generate)
    7. optional figures and baseline comparisons (diagnostics, plots)

    target_groups_for_mode() decides step 4's grouping: one network spanning
    all targets, or one network per target, per GMN_MODE.

THE ONE ARGUMENT
    main(entry_path, network_builder=None). `network_builder` is the only thing
    the two entry scripts disagree about: the serial pipeline leaves it None
    (network.build_nested_mde_network), and the GPU branch's parallel-MDE
    pipeline passes network_parallel.build_nested_mde_network. Everything else
    -- config, data, caching, GMN, scoring, output -- is shared, which is why
    both entry scripts are only a few lines long. `entry_path` is the entry
    script's own __file__, which the code fingerprint needs; see main().

Depends on: everything above it in mde_gmn/__init__.py's dependency list.
Imported by: the entry scripts only. Nothing in the package imports this.
"""

import time

import numpy as np
import pandas as pd

from mde_gmn import gmn_embedding  # noqa: F401 -- patches gmn on import; see its docstring
from mde_gmn import mde            # module object, so a fake can be installed on it
from mde_gmn.caching import _hash_file, build_code_fingerprint, build_search_fingerprint
from mde_gmn.columns import derive_lag_set, effective_mde_dimension
from mde_gmn.config import load_config
from mde_gmn.diagnostics import (
    run_mde_evaluate, run_tp_vs_rho_sweep, run_univariate_e_tau_grid)
from mde_gmn.generate import (
    gmn_fidelity_rho, plot_fidelity, rebuild_at_best_params, run_forecast_diagnostic,
    sweep_generate_start)
from mde_gmn.gmn_build import build_network
from mde_gmn.plots import draw_colored_network, plot_3d_network
from mde_gmn.progress import _log_progress
from mde_gmn.theta import run_theta_search
from mde_gmn.tsls_data import burn_in_trim, convert_to_tsls, raw_neuron_columns


def target_groups_for_mode(targets, gmn_mode):
    if gmn_mode == 'combined':
        return [list(targets)]
    elif gmn_mode == 'separate':
        return [[t] for t in targets]
    else:
        raise ValueError(f"Unknown GMN_MODE: {gmn_mode!r}")

def main(entry_path, network_builder=None):
    """Run the whole pipeline once.

    entry_path: the calling entry script's __file__. It goes into the code
    fingerprint, which is what keeps each pipeline's cached results in its own
    namespace -- passing pipeline.py's own __file__ instead would make the
    serial and parallel pipelines share one namespace, because every other
    input to the fingerprint is identical. Required, not defaulted, so it
    cannot be forgotten silently.

    network_builder: the nested-MDE implementation to use. None means the
    serial one (network.build_nested_mde_network); the GPU branch's
    parallel-MDE entry script passes network_parallel's instead. See this
    module's docstring.
    """
    cfg = load_config()

    # Progress-monitoring log (see HEARTBEAT_SECONDS/_log_progress()'s own
    # docstrings): cfg['RESULTS_DIR'] is already on /flash for the whole
    # run's duration (TSLS_MDE_GMN_RESULTS_DIR -> $OUT_DIR in the sbatch
    # script, archived to bucket only at job end), so this needs no new
    # path plumbing. Opened in truncate mode -- each job already gets its
    # own fresh RESULTS_DIR, so there is no earlier content to preserve.
    progress_log_path = cfg['RESULTS_DIR'] / 'progress.log'
    open(progress_log_path, 'w').close()
    pipeline_start = time.perf_counter()
    _log_progress(progress_log_path, 'PIPELINE START')

    df_raw = pd.read_csv(cfg['DATA_DIR'] / cfg['DATA_FILE'])
    print('Raw data shape:', df_raw.shape, flush=True)

    neuron_columns = raw_neuron_columns(df_raw, cfg['REGION_CHANNEL_CAP'])
    print(f'{len(neuron_columns)} raw spike-count channels to TSLS-convert', flush=True)
    tsls_df_full, obs_cols = convert_to_tsls(df_raw, neuron_columns, cfg['TARGETS'], cfg['RESULTS_DIR'])
    df = burn_in_trim(tsls_df_full, obs_cols, cfg['RESULTS_DIR'])

    # Three-way A/B/end split -- see load_config()'s docstring for the full
    # rationale (identical to simple-mde-gmn.py, ported here), computed
    # against the TSLS-converted, burn-in-trimmed row count (len(df)), not
    # the raw file's row count.
    cfg['PREDICTION_START'] = int(round(len(df) * cfg['LIB_RATIO']))                          # A
    post_lib_remainder = len(df) - cfg['PREDICTION_START']
    cfg['PREDICTION_LENGTH'] = int(round(post_lib_remainder * cfg['MDE_VALIDATION_RATIO']))   # B - A
    cfg['GENERATE_VALIDATION_START'] = cfg['PREDICTION_START'] + cfg['PREDICTION_LENGTH']     # B
    cfg['GENERATE_VALIDATION_LENGTH'] = len(df) - cfg['GENERATE_VALIDATION_START']            # end - B

    if cfg['GENERATE_LENGTH'] <= 0:
        cfg['GENERATE_LENGTH'] = cfg['GENERATE_VALIDATION_LENGTH']
    elif cfg['GENERATE_LENGTH'] > cfg['GENERATE_VALIDATION_LENGTH']:
        print(f"GENERATE_LENGTH={cfg['GENERATE_LENGTH']} exceeds the Generate/Forecast "
              f"validation segment ({cfg['GENERATE_VALIDATION_LENGTH']} rows); clamping "
              f"to it.", flush=True)
        cfg['GENERATE_LENGTH'] = cfg['GENERATE_VALIDATION_LENGTH']

    max_feasible_offset = cfg['GENERATE_VALIDATION_LENGTH'] - cfg['GENERATE_LENGTH']
    start_offsets = list(range(cfg['GENERATE_START_MIN_OFFSET'], cfg['GENERATE_START_MAX_OFFSET'],
                                cfg['GENERATE_START_STEP']))
    feasible_offsets = [o for o in start_offsets if 0 <= o <= max_feasible_offset]
    if feasible_offsets != start_offsets:
        print(f'GENERATE_START offsets {start_offsets} include values outside the feasible '
              f'range [0, {max_feasible_offset}]; using {feasible_offsets} instead.', flush=True)
    cfg['GENERATE_START_OFFSETS'] = feasible_offsets or [0]

    cfg['LAG_SET'] = derive_lag_set(
        cfg['DELAY_SOURCE'], cfg['DELAY_SET'], cfg['LAG_MODE'], cfg['LAG_DENSITY'], cfg['LAG_STEP'],
        cfg['GENERATE_LENGTH'], cfg['PREDICTION_START'],
    )

    # VERIFY (see this pipeline's own design-decision record for why this
    # guard exists here and not in simple-mde-gmn.py): Stage 2's lag-
    # embedding depth must fit within PREDICTION_START (A), the MDE library
    # segment -- an implicit assumption in simple-mde-gmn.py too, but
    # TSLS's own PREDICTION_START is data-driven (burn_in_trim() shrinks
    # the usable row count depending on which channel fires slowest, unlike
    # simple-mde-gmn.py's fixed-length 1 Hz input), so this fails fast with
    # a clear message instead of a confusing downstream all-NaN/shape
    # error deep inside proper_mde or gmn's own SMap calls.
    max_lag_depth = max((abs(d) for d in cfg['LAG_SET']), default=0)
    if max_lag_depth >= cfg['PREDICTION_START']:
        raise ValueError(
            f"Stage 2's lag-embedding depth ({max_lag_depth} rows, from LAG_SET= "
            f"{cfg['LAG_SET']}) does not fit within PREDICTION_START "
            f"({cfg['PREDICTION_START']} rows, the MDE library segment) -- every "
            f"lagged candidate column would be all-NaN across the entire library. "
            f"This is data-driven for TSLS (burn-in trim shrinks the usable row "
            f"count depending on which channel fires slowest), unlike "
            f"simple-mde-gmn.py's fixed-length input. Narrow "
            f"TSLS_MDE_GMN_LAG_MODE/LAG_STEP, shrink TSLS_MDE_GMN_GENERATE_LENGTH, "
            f"or raise TSLS_MDE_GMN_LIB_RATIO to fix."
        )

    print('Config:', {k: v for k, v in cfg.items() if k != 'TARGET_COLORS'}, flush=True)

    data_file_sha256 = _hash_file(cfg['DATA_DIR'] / cfg['DATA_FILE']) if cfg['REUSE_BEST_PARAMS'] else None
    code_fingerprint = build_code_fingerprint(entry_path) if cfg['REUSE_BEST_PARAMS'] else None

    # ---- MDE (or its full-channel-pool stand-in for MODE='GMN') -------
    mde_results = {}
    filtered_pools = {}
    if cfg['MODE'] == 'MDE-GMN':
        # CCM pre-filter runs once per target_groups_for_mode() group, here,
        # BEFORE depth-0 -- so depth-0's own candidate pool is already the
        # filtered one, same as every later nested-MDE round. See this
        # pipeline's own build_network() docstring: that function no longer
        # runs the pre-filter itself, precisely because main() now always
        # does it first.
        for group in target_groups_for_mode(cfg['TARGETS'], cfg['GMN_MODE']):
            if cfg['CCM_PREFILTER_RHO_MIN'] > 0:
                filtered_columns, _ = mde.ccm_prefilter_observables(
                    df, obs_cols, group, cfg['PREDICTION_START'],
                    cfg['CCM_PREFILTER_RHO_MIN'], cfg['CCM_PREFILTER_E'], cfg['TP'],
                    cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
                    cfg['EXCLUSION_RADIUS_VALUE'], cfg['CCM_WINDOW_ROWS'], cfg['CCM_SAMPLE'],
                    cfg['LIBSIZE_RATIOS'], cfg['CORES'],
                )
                print(f'CCM pre-filter: kept {len(filtered_columns)}/{len(obs_cols)} '
                      f'observables (rho >= {cfg["CCM_PREFILTER_RHO_MIN"]}) for {group}', flush=True)
            else:
                filtered_columns = obs_cols
            filtered_pools[tuple(group)] = filtered_columns

        # Stage 1 seed: real-observable-only MDE, no delay expansion --
        # matches build_nested_mde_network()'s own Stage-1 candidate pool
        # exactly, so its depth-0 reuse of this result stays valid.
        if cfg['GMN_MODE'] == 'combined':
            # One joint multi_proper_mde call across every target, instead
            # of GMN_MODE='separate's per-target proper_mde loop below --
            # see mde.run_multi_mde_for_targets()'s and load_config()'s own
            # MULTI_MDE_E docstrings. The single shared result is fanned out
            # into one per-target *view* below (same 'm'/'mde_df'/
            # 'SELECTED_COLUMNS' object, just re-keyed with a 'target' entry
            # each) so every downstream consumer that indexes
            # mde_results[target] -- build_nested_mde_network()'s depth-0
            # reuse (adds the SAME joint drivers as edges into every target
            # node in the group, which is the intended "combined" MDE
            # behaviour), and the diagnostics plots/sweeps below -- keeps
            # working completely unchanged.
            candidate_pool = [c for c in filtered_pools[tuple(cfg['TARGETS'])] if c not in cfg['TARGETS']]
            effective_d = effective_mde_dimension(cfg['MDE_D'], candidate_pool, 1)
            depth0_start = time.perf_counter()
            _log_progress(progress_log_path, f"depth0 joint MDE START -- targets={cfg['TARGETS']}")
            try:
                shared_result = mde.run_multi_mde_for_targets(
                    cfg['TARGETS'], df, candidate_pool, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                    effective_d, cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
                    cfg['EXCLUSION_RADIUS_VALUE'], cfg['RESULTS_DIR'], [],
                    tp=cfg['TP'],
                    metric=cfg['METRIC'], target_metric=cfg['TARGET_METRIC'],
                    gamma_delta=cfg['GAMMA_DELTA'],
                    target_ranking_window_rows=cfg['TARGET_RANKING_WINDOW_ROWS'],
                    ccm_lib_sizes_rows=cfg['CCM_LIB_SIZES_ROWS'],
                    ccm_sample=cfg['CCM_SAMPLE'], ccm_slope=cfg['CCM_SLOPE'],
                    reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
                    run_cache_dir=cfg['RUN_CACHE_DIR'],
                    data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
                    code_fingerprint=code_fingerprint,
                    E=0, max_e=cfg['MULTI_MDE_E'], target_aggregate=cfg['MULTI_MDE_TARGET_AGGREGATE'],
                    use_gpu=cfg['GPU_ENABLED'],
                )
            except Exception as exc:
                # Same failure mode mde._failed_mde_result() guards against below
                # -- this is the depth-0 seed for the Stage-1 BFS (shared
                # across every target in the group), so it needs the
                # identical guard or the exact crash just relocates one
                # level up.
                print(f"{cfg['TARGETS']}: joint (multi-target) MDE call FAILED: {exc!r} -- "
                      f'treating as a dead-end (zero drivers selected for every target)', flush=True)
                shared_result = mde._failed_multi_mde_result(cfg['TARGETS'])
            _log_progress(progress_log_path,
                          f'depth0 joint MDE END -- {time.perf_counter() - depth0_start:.1f}s '
                          f"selected={shared_result['SELECTED_COLUMNS']}")
            for target in cfg['TARGETS']:
                mde_results[target] = {**shared_result, 'target': target}
        else:
            for target in cfg['TARGETS']:
                candidate_pool = [c for c in filtered_pools[(target,)] if c != target]
                effective_d = effective_mde_dimension(cfg['MDE_D'], candidate_pool, 1)
                depth0_start = time.perf_counter()
                _log_progress(progress_log_path, f'depth0 MDE START -- {target}')
                try:
                    mde_results[target] = mde.run_mde_for_target(
                        target, df, candidate_pool, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                        effective_d, cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
                        cfg['EXCLUSION_RADIUS_VALUE'], cfg['RESULTS_DIR'], [],
                        tp=cfg['TP'],
                        metric=cfg['METRIC'], target_metric=cfg['TARGET_METRIC'],
                        gamma_delta=cfg['GAMMA_DELTA'],
                        is_behavioral_target=True,
                        target_ranking_window_rows=cfg['TARGET_RANKING_WINDOW_ROWS'],
                        min_pred_events=cfg['MIN_PRED_EVENTS'],
                        max_window_rows=cfg['MAX_WINDOW_ROWS'],
                        ccm_lib_sizes_rows=cfg['CCM_LIB_SIZES_ROWS'],
                        ccm_sample=cfg['CCM_SAMPLE'], ccm_slope=cfg['CCM_SLOPE'],
                        reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
                        run_cache_dir=cfg['RUN_CACHE_DIR'],
                        data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
                        code_fingerprint=code_fingerprint,
                        max_e=cfg['MDE_D'],
                        # This loop is sequential over cfg['TARGETS'] (one
                        # target computed at a time, no outer pool sharing
                        # cores with it) -- so unlike build_nested_mde_
                        # network()'s own per-wave outer_workers-based
                        # split, this depth-0 seed call gets the FULL core
                        # budget for its own inner pool.
                        inner_workers=max(1, cfg['CORES']),
                        use_gpu=cfg['GPU_ENABLED'],
                    )
                except Exception as exc:
                    # Same failure mode mde._failed_mde_result() guards against
                    # in build_nested_mde_network() below -- this is the
                    # depth-0 seed for that same Stage-1 BFS, so it needs
                    # the identical guard or the exact crash just
                    # relocates one level up.
                    print(f'{target}: top-level MDE call FAILED: {exc!r} -- treating as a dead-end '
                          f'(zero drivers selected for this target)', flush=True)
                    mde_results[target] = mde._failed_mde_result(target)
                _log_progress(progress_log_path,
                              f'depth0 MDE END -- {target} {time.perf_counter() - depth0_start:.1f}s '
                              f"selected={mde_results[target]['SELECTED_COLUMNS']}")

        if cfg['RUN_MDE_DIAGNOSTICS']:
            mde.plot_mde_rho_vs_dimension(
                mde_results, cfg['TARGETS'], cfg['TARGET_COLORS'],
                cfg['RESULTS_DIR'] / 'mde_rho_vs_dimension_multi_target.png',
            )
            mde.plot_3d_mde_projection(
                cfg['TARGETS'][0], cfg['SECOND_TARGET'], mde_results, df, obs_cols,
                cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                cfg['RESULTS_DIR'] / f"mde_3d_projection_{cfg['TARGETS'][0]}.png",
            )
    else:  # MODE == 'GMN'
        for target in cfg['TARGETS']:
            mde_results[target] = {'target': target, 'SELECTED_COLUMNS': list(obs_cols)}
        print(f"MODE='GMN': skipping MDE, using all {len(obs_cols)} TSLS-converted observable "
              f"columns as the candidate driver pool for {cfg['TARGETS']}.", flush=True)

    # ---- GMN network construction + node generation --------------------
    gmn_runs = {}
    for group in target_groups_for_mode(cfg['TARGETS'], cfg['GMN_MODE']):
        net_result = build_network(
            group, mde_results, df, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
            cfg['MODE'], cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['RESULTS_DIR'], cfg['CORES'],
            cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'], cfg['CCM_WINDOW_ROWS'],
            cfg['CCM_LIB_SIZE_PCTS'], cfg['CCM_SAMPLE'], cfg['CCM_SLOPE'],
            target_ranking_window_rows=cfg['TARGET_RANKING_WINDOW_ROWS'],
            min_pred_events=cfg['MIN_PRED_EVENTS'],
            max_window_rows=cfg['MAX_WINDOW_ROWS'],
            ccm_lib_sizes_rows=cfg['CCM_LIB_SIZES_ROWS'],
            observable_columns=filtered_pools.get(tuple(group), obs_cols),
            nested_n=cfg['MDE_D'], nested_m=cfg['NESTED_M'],
            embed_dim_rho_min=cfg['EMBED_DIM_RHO_MIN'], cross_map_rho_min=cfg['CROSS_MAP_RHO_MIN'],
            exclusion_radius=cfg['EXCLUSION_RADIUS_VALUE'],
            lag_set=cfg['LAG_SET'], tp=cfg['TP'],
            reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
            run_cache_dir=cfg['RUN_CACHE_DIR'],
            data_file=cfg['DATA_FILE'], data_file_sha256=data_file_sha256,
            code_fingerprint=code_fingerprint,
            gmn_mode=cfg['GMN_MODE'], multi_mde_e=cfg['MULTI_MDE_E'],
            multi_mde_target_aggregate=cfg['MULTI_MDE_TARGET_AGGREGATE'],
            external_input_mode=cfg['EXTERNAL_INPUT_MODE'],
            progress_log_path=progress_log_path,
            mp_start_method=cfg['MP_START_METHOD'],
            use_gpu=cfg['GPU_ENABLED'],
            network_builder=network_builder,
        )
        draw_colored_network(
            net_result, cfg['TARGET_COLORS'],
            cfg['RESULTS_DIR'] / f"gmn_network_{net_result['network_id']}.png",
        )
        if cfg['MODE'] == 'MDE-GMN':
            plot_3d_network(
                net_result, cfg['TARGET_COLORS'],
                cfg['RESULTS_DIR'] / f"network_3d_{net_result['network_id']}.html",
            )
        gmn_runs[net_result['network_id']] = {'group': group, 'net_result': net_result}

    fingerprints = None
    if cfg['REUSE_BEST_PARAMS']:
        fingerprints = {
            network_id: build_search_fingerprint(cfg, run['net_result'], data_file_sha256, code_fingerprint)
            for network_id, run in gmn_runs.items()
        }

    theta_search_results = run_theta_search(
        gmn_runs, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'], cfg['THETA_RANGE'],
        cfg['RESULTS_DIR'], cfg['CORES'], cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
        generate_length=cfg['GENERATE_LENGTH'],
        reuse_best_params=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
        run_cache_dir=cfg['RUN_CACHE_DIR'],
        fingerprints=fingerprints, exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
        generate_validation_start=cfg['GENERATE_VALIDATION_START'],
        progress_log_path=progress_log_path,
    )

    generate_starts = sweep_generate_start(
        gmn_runs, theta_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['RESULTS_DIR'], cfg['CORES'], cfg['GENERATE_LENGTH'], cfg['GENERATE_START_OFFSETS'],
        cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
        exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
        reuse_cache=cfg['REUSE_BEST_PARAMS'], prev_runs_dir=cfg['PREV_RUNS_DIR'],
        run_cache_dir=cfg['RUN_CACHE_DIR'], fingerprints=fingerprints,
        generate_validation_start=cfg['GENERATE_VALIDATION_START'],
        progress_log_path=progress_log_path,
    )

    rebuild_at_best_params(
        gmn_runs, theta_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['RESULTS_DIR'], cfg['CORES'], cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
        generate_length=cfg['GENERATE_LENGTH'], generate_starts=generate_starts,
        exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
        generate_validation_start=cfg['GENERATE_VALIDATION_START'],
        progress_log_path=progress_log_path,
    )

    for network_id, run in gmn_runs.items():
        if 'G' not in run:
            # rebuild_at_best_params() already logged why and skipped this
            # network -- nothing downstream has anything to work with
            # without G.
            print(f'{network_id}: no successful Generate() rebuild -- skipping '
                  f'Forecast()/fidelity/output for this network', flush=True)
            continue

        forecast_start = time.perf_counter()
        _log_progress(progress_log_path, f'{network_id}: Forecast() diagnostic START')
        Gf, forecast_rho = run_forecast_diagnostic(
            run['net_result'], run['G'], cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
            run['best_node_theta'],
            cfg['RESULTS_DIR'], cfg['CORES'],
            cfg['RESULTS_DIR'] / f'gmn_generate_forecast_{network_id}.png',
            cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
            generate_length=cfg['GENERATE_LENGTH'], generate_start=run['generate_start'],
            exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'],
            plot_range_mode=cfg['FORECAST_PLOT_RANGE'],
            generate_validation_start=cfg['GENERATE_VALIDATION_START'], total_rows=len(df),
        )
        _log_progress(progress_log_path,
                      f'{network_id}: Forecast() diagnostic END -- '
                      f'{time.perf_counter() - forecast_start:.1f}s')
        run['Gf'] = Gf
        run['forecast_rho'] = forecast_rho
        if Gf is None:
            print(f'{network_id}: no Forecast() rho to report for this network', flush=True)
        else:
            print(f"{network_id} Forecast() rho per node (weakest first):", flush=True)
            for node, rho in sorted(forecast_rho.items(), key=lambda kv: kv[1]):
                print(f"  {node:20s} {rho:.4f}", flush=True)

        rho = gmn_fidelity_rho(run['net_result'], run['G'], run['generate_start'],
                                cfg['METRIC'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
        plot_fidelity(rho, network_id, cfg['RESULTS_DIR'] / f'gmn_fidelity_{network_id}.png')
        print(f'{network_id} fidelity rho (weakest first):', flush=True)
        for node in sorted(rho, key=lambda n: (np.isnan(rho[n]), -(rho[n] if not np.isnan(rho[n]) else 0))):
            print(f'  {node:20s} {rho[node]:.4f}', flush=True)

        generated = run['G'].DataOut
        generated.to_csv(cfg['RESULTS_DIR'] / f'gmn_generate_output_{network_id}.csv', index=False)

    # ---- MDE-only diagnostics -------------------------------------------
    if cfg['MODE'] == 'MDE-GMN' and cfg['RUN_MDE_DIAGNOSTICS']:
        _log_progress(progress_log_path, 'R7 diagnostics START (tp-vs-rho sweep, E/tau grid)')
        r7_start = time.perf_counter()
        run_tp_vs_rho_sweep(mde_results, cfg['TARGETS'], cfg['MAX_TP'], cfg['TARGET_COLORS'],
                             cfg['RESULTS_DIR'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
        run_univariate_e_tau_grid(mde_results, cfg['TARGETS'], cfg['E_RANGE'], cfg['TAU_RANGE'],
                                   cfg['GRID_RHO_THRESHOLD'], cfg['RESULTS_DIR'],
                                   cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'])
        _log_progress(progress_log_path,
                      f'R7 diagnostics END -- {time.perf_counter() - r7_start:.1f}s')
        # R19 (docs/review-actions-v1.md): per constructed network -- needs
        # net_result['network']['Graph'] for D (direct-driver count), which
        # mde_results (depth-0-only) doesn't carry. Gated on its own
        # RUN_MDE_EVALUATE, deliberately separate from RUN_MDE_DIAGNOSTICS
        # above (see that config's own docstring) -- R7's two diagnostics
        # just above are unaffected and always run when RUN_MDE_DIAGNOSTICS
        # is set, regardless of this toggle.
        if cfg['RUN_MDE_EVALUATE']:
            for network_id, run in gmn_runs.items():
                r19_start = time.perf_counter()
                _log_progress(progress_log_path, f'{network_id}: R19 run_mde_evaluate START')
                run_mde_evaluate(run['net_result'], cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                                  cfg['RESULTS_DIR'], cfg['TARGET_METRIC'], cfg['GAMMA_DELTA'],
                                  exclusion_radius=cfg['GMN_EXCLUSION_RADIUS'])
                _log_progress(progress_log_path,
                              f'{network_id}: R19 run_mde_evaluate END -- '
                              f'{time.perf_counter() - r19_start:.1f}s')
        else:
            # Logged once per run, unconditionally -- a skipped diagnostic
            # that never appears anywhere is exactly how §18.6's swallowed
            # exception went unnoticed for so long (docs/review-findings.md).
            msg = ("R19 run_mde_evaluate SKIPPED by configuration "
                   "(RUN_MDE_EVALUATE=false) -- see docs/review-findings.md's "
                   "'diffusion-map crash and hang' finding for why this defaults off")
            print(msg, flush=True)
            _log_progress(progress_log_path, msg)
    else:
        print(f"MODE={cfg['MODE']!r} / RUN_MDE_DIAGNOSTICS={cfg['RUN_MDE_DIAGNOSTICS']}: "
              f"skipping MDE-only diagnostics.", flush=True)

    _log_progress(progress_log_path,
                  f'PIPELINE COMPLETE -- {time.perf_counter() - pipeline_start:.1f}s total')
    print('Pipeline complete.', flush=True)
