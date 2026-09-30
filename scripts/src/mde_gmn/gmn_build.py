"""Handing the finished network to GMN: node dimensions and the .cfg file.

This is step 4 of the pipeline. network.py produced a graph; GMN needs that
graph plus a configuration file describing every node. This module is the
bridge. It mirrors cells 7-8 of notebooks/tsls-mde-gmn.ipynb.

    build_network()         the entry point. Two unrelated constructions,
                            chosen by `mode`:
                              MODE='MDE-GMN' delegates to
                                build_nested_mde_network() (network.py) --
                                the nested-MDE algorithm this project is about;
                              MODE='GMN' instead cross-maps every pair and
                                thresholds the result into an interaction
                                matrix (build_interaction_matrix), the stock
                                GMN construction, kept for comparison.
                            Returns the graph, the data GMN will read, which
                            nodes it simulates and which are external inputs.
    estimate_node_dimension()
                            how many embedding coordinates a node needs, when
                            the network itself does not already say. Used by
                            MODE='GMN' and as a fallback.
    build_gmn_config()      writes the .cfg file gmn.GMN's constructor reads:
                            which nodes exist, which function each uses
                            (SMap/Simplex), theta, the prediction window, the
                            Theiler exclusion radius.

E and tau are NOT configuration here for MODE='MDE-GMN'. Each node's embedding
comes from its own in-edges (gmn_embedding.py), so the .cfg carries per-node
embeddings rather than one global E and tau. MODE='GMN' still needs a grid
search for them (target_E_tau_rho_grid, _ccm_converged).

Network construction is cached: it is expensive, and a finished network is
reused across jobs (caching.py).

Depends on: network, gmn_embedding, caching, tsls_data, progress, columns;
cleaned_helper_functions, apps.CreateNetwork, numpy, pandas.
Imported by: theta, generate, pipeline, the entry scripts.
"""

import pickle
import time
from configparser import ConfigParser

import cleaned_helper_functions as tsls
import numpy as np
import pandas as pd
from apps.CreateNetwork import CreateNetwork

from mde_gmn.caching import (
    _load_pickle_cache_entry, build_network_fingerprint, find_cached,
    save_pickle_to_cache)
from mde_gmn.network import build_nested_mde_network, partition_network, report_partition
from mde_gmn.progress import _log_progress
from mde_gmn.tsls_data import _ccm_window, _metric_for


def _lib_train_test_ranges(n_rows, train_frac=0.7):
    # [start, stop] list form (1-offset, inclusive) -- proper_simplex/
    # proper_embed_dimension expect this form, not pyEDM's own "start stop"
    # string form.
    split = int(n_rows * train_frac)
    return [1, split], [split + 1, n_rows]

def target_E_tau_rho_grid(lib_df, target, e_vals, tau_vals, metric, gamma_delta):
    lib_range, pred_range = _lib_train_test_ranges(len(lib_df))
    rho_grid = np.full((len(e_vals), len(tau_vals)), np.nan)
    for i, E in enumerate(e_vals):
        for j, tau in enumerate(tau_vals):
            try:
                out = tsls.proper_simplex(
                    lib_df, columns=target, target=target,
                    E=E, tau=tau, lib=lib_range, pred=pred_range,
                    embedded=False, metric=metric, gammaDelta=gamma_delta,
                )
                rho_grid[i, j] = out['metrics'][metric]
            except Exception:
                pass
    return rho_grid

def estimate_node_dimension(lib_df, node, metric, gamma_delta, max_e=15, fallback=4):
    '''EmbedDimension-equivalent for TSLS-aware metrics via
    proper_embed_dimension (pyEDM's own EmbedDimension is Pearson-rho only,
    so it can't score 'f1'/'gamma').'''
    lib_range, pred_range = _lib_train_test_ranges(len(lib_df))
    try:
        ed = tsls.proper_embed_dimension(
            lib_df, columns=node, target=node,
            maxE=max_e, lib=lib_range, pred=pred_range,
            metric=metric, gammaDelta=gamma_delta,
        )
        return int(ed.loc[ed[metric].idxmax(), 'E'])
    except Exception:
        return fallback

def _ccm_converged(mean_vals, direction, ccm_slope):
    '''Two-part convergence test mirroring proper_mde's own CCM gate
    (cleaned_helper_functions.py: endpoint improved AND fitted slope vs.
    normalized libSize position exceeds ccm_slope). Returns (passed, slope).'''
    if np.all(np.isnan(mean_vals)):
        return False, 0.0
    x = np.arange(len(mean_vals), dtype=float)
    x /= x[-1] if x[-1] > 0 else 1.0
    slope = float(np.polyfit(x, mean_vals, 1)[0]) if len(mean_vals) > 1 else 0.0
    endpoint_improved = (direction * (mean_vals[-1] - mean_vals[0])) > 0
    return (endpoint_improved and (direction * slope) > ccm_slope), slope

def build_interaction_matrix(data_lib, nodes, targets, ccm_E, ccm_tau, metric, target_metric,
                              gamma_delta, ccm_window_rows, ccm_lib_size_pcts, ccm_sample,
                              ccm_slope):
    '''NxN interaction matrix via cleaned_helper_functions.proper_ccm,
    replacing apps.InteractionMatrix's CLI subprocess -- MODE='GMN' only.
    No multiprocessing-pool-nesting concern at all here (unlike
    simple-mde-gmn.py's own MODE='GMN' branch): proper_ccm has no pool of
    its own to nest inside an outer one.

    Per-pair metric selection via _metric_for(): `nodes` mixes TSLS-
    converted neuron columns with the (continuous, non-TSLS) TARGETS
    columns, so each proper_ccm call's target_measurement/
    columns_measurement must independently match whichever side (a or b) is
    actually being predicted in that direction.

    Matches apps.CreateNetwork()'s documented convention:
    `iMatrix.loc[driven_node, driver_node]` = skill of predicting
    driven_node FROM driver_node's history. proper_ccm's c2t direction
    (`target` predicted FROM `columns`) maps directly onto that; calling it
    once per *unordered* pair yields both c2t and t2c. Non-converged
    directions are left at 0 (the pre-filled default), which is what
    build_network()'s downstream threshold + per-node top-N-drivers
    selection already expects.'''
    windowed, _ = _ccm_window(data_lib, 0.5, ccm_window_rows)
    libSizes = sorted({max(10, int(len(windowed) * p / 100.0)) for p in ccm_lib_size_pcts})

    iMatrix = pd.DataFrame(0.0, index=nodes, columns=nodes)
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            a_metric = _metric_for(a, targets, metric, target_metric)
            b_metric = _metric_for(b, targets, metric, target_metric)
            try:
                c2t, t2c, _ = tsls.proper_ccm(
                    windowed, libSizes=libSizes, sample=ccm_sample,
                    tau=ccm_tau, Tp=1, E=ccm_E,
                    target=a, columns=b,
                    target_measurement=a_metric, columns_measurement=b_metric,
                    gammaDelta=gamma_delta,
                )
            except Exception:
                continue

            for series, driven, driver, node_metric in (
                (c2t, a, b, a_metric), (t2c, b, a, b_metric),
            ):
                mean_vals = np.nanmean(series, axis=0)
                direction = 1 if node_metric in ('correlation', 'f1', 'gamma') else -1
                passed, slope = _ccm_converged(mean_vals, direction, ccm_slope)
                if passed:
                    iMatrix.loc[driven, driver] = float(mean_vals[-1])

    for n in nodes:
        iMatrix.loc[n, n] = np.nan
    return iMatrix

def build_network(targets, mde_results, df, prediction_start, prediction_length,
                   mode, e_range, tau_range, results_dir, cores,
                   metric, target_metric, gamma_delta, ccm_window_rows, ccm_lib_size_pcts,
                   ccm_sample, ccm_slope,
                   target_ranking_window_rows=1000, min_pred_events=20, max_window_rows=30000,
                   ccm_lib_sizes_rows=(16, 24, 40, 64, 160, 400, 800),
                   observable_columns=None, nested_n=None, nested_m=None,
                   embed_dim_rho_min=None, cross_map_rho_min=None, exclusion_radius=None,
                   lag_set=None, tp=1,
                   threshold=0, fallback_node_dimension=4, max_node_dimension=15,
                   reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                   data_file=None, data_file_sha256=None, code_fingerprint=None,
                   gmn_mode=None, multi_mde_e=None, multi_mde_target_aggregate=None,
                   external_input_mode='real_data', progress_log_path=None,
                   mp_start_method='fork', use_gpu=False,
                   network_builder=None):
    '''Build one network spanning `targets` and their driver candidates, via
    one of two unrelated constructions depending on `mode`.

    network_builder: which nested-MDE implementation to use for
    mode == 'MDE-GMN'. Defaults to network.build_nested_mde_network (one MDE
    call at a time, in breadth-first waves). The GPU branch's parallel-MDE
    pipeline passes network_parallel.build_nested_mde_network instead, which
    computes the MDE results up front and then replays them. Both produce the
    same network; this argument is the only difference between the two
    pipelines at this level.

    The two constructions, same mode-branch restructuring as
    simple-mde-gmn.py's own build_network():
      - mode == 'MDE-GMN': two-stage nested-MDE, both stages unified into
        build_nested_mde_network()'s own BFS, no apps.CreateNetwork() call
        -- see simple-mde-gmn.py's own build_network() docstring for the
        full rationale (identical here). `observable_columns` is already
        the CCM-pre-filtered pool -- main() runs the pre-filter once per
        target_groups_for_mode() group before this function is ever
        called, so this function no longer runs it itself.
        observable_columns/nested_n/nested_m/embed_dim_rho_min/
        cross_map_rho_min/lag_set are required in this mode.
      - mode == 'GMN': the interaction matrix via
        cleaned_helper_functions.proper_ccm (build_interaction_matrix()) --
        TSLS's own existing replacement for apps.InteractionMatrix's CLI
        subprocess, unaffected by any of the MDE-GMN-side two-stage rework
        -- then a node-dimension estimate and apps.CreateNetwork with its
        driver-ranking-bug workaround, same as before.
    '''
    network_id = '_'.join(targets)
    if mode == 'GMN':
        network_id = f'gmn_{network_id}'

    ccm_E = ccm_tau = None
    node_dimensions = {}

    fingerprint = None
    cache_filename = f'network_cache_{network_id}.pkl'
    cache_hit = None
    if reuse_cache:
        fingerprint = build_network_fingerprint(
            mode, targets, observable_columns, nested_n, nested_m, tp,
            embed_dim_rho_min, cross_map_rho_min, exclusion_radius, lag_set,
            e_range, tau_range, prediction_start, prediction_length,
            metric, target_metric, gamma_delta, ccm_window_rows, ccm_lib_size_pcts,
            ccm_sample, ccm_slope,
            data_file, data_file_sha256, code_fingerprint,
            gmn_mode=gmn_mode, multi_mde_e=multi_mde_e,
            multi_mde_target_aggregate=multi_mde_target_aggregate,
            target_ranking_window_rows=target_ranking_window_rows,
            min_pred_events=min_pred_events, max_window_rows=max_window_rows,
            ccm_lib_sizes_rows=ccm_lib_sizes_rows,
        )
        hit = find_cached(prev_runs_dir, cache_filename, fingerprint, load_one=_load_pickle_cache_entry)
        if hit is not None:
            cache_hit, source = hit
            print(f'{network_id}: TSLS_MDE_GMN_REUSE_BEST_PARAMS matched cached network {source} '
                  f'-- skipping network construction entirely', flush=True)
        else:
            print(f'{network_id}: TSLS_MDE_GMN_REUSE_BEST_PARAMS set but no matching cached network '
                  f'found under {prev_runs_dir} -- building from scratch.', flush=True)

    if mode == 'MDE-GMN':
        if cache_hit is not None:
            graph = cache_hit['graph']
        else:
            # `observable_columns` is already the CCM-pre-filtered pool --
            # main() runs ccm_prefilter_observables() once per
            # target_groups_for_mode() group before this function is ever
            # called, so depth-0 itself only ever searches already-
            # filtered observables too (see main()'s own depth-0
            # docstring for why the pre-filter had to move there rather
            # than staying here). Both stages, fully serial (Stage 1 runs
            # to completion, then Stage 2), in one call -- see
            # build_nested_mde_network()'s own docstring (revised
            # 2026-08-27, R12: Stage 2's own real-observable selections
            # still recurse exactly like Stage 1's, just after Stage 1
            # completes rather than interleaved with it wave-by-wave).
            graph, _, _ = (network_builder or build_nested_mde_network)(
                targets, df, observable_columns, observable_columns,
                prediction_start, prediction_length,
                nested_n, nested_m, embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                results_dir, mde_results, [], tp,
                metric, target_metric, gamma_delta,
                target_ranking_window_rows, min_pred_events, max_window_rows,
                ccm_lib_sizes_rows, ccm_sample, ccm_slope, cores,
                lag_set,
                reuse_cache, prev_runs_dir, run_cache_dir, data_file, data_file_sha256,
                code_fingerprint,
                progress_log_path=progress_log_path,
                mp_start_method=mp_start_method,
                use_gpu=use_gpu,
            )
        graph.graph['self_delay_set'] = sorted(set(lag_set) | {0}, reverse=True)
        partition_network(graph, targets, external_input_mode)
        report_partition(graph, network_id, results_dir)
        selected_union = [n for n in graph.nodes if n not in targets]
        # {leaf node name: (raw observable, delay)} for every lagged leaf in
        # the network -- see the data_after_mde block below for why.
        lagged_leaves = {n: (graph.nodes[n]['lagged_of'], graph.nodes[n]['lag'])
                          for n in selected_union if 'lagged_of' in graph.nodes[n]}
        edges_file = results_dir / f'edges_nestedMDE_N{nested_n}_M{nested_m}_{network_id}.csv'
        edges_rows = [{'driver': u, 'node': v, 'delay': d['delay'], 'weight': d['weight']}
                      for u, v, d in graph.edges(data=True)]
        pd.DataFrame(edges_rows, columns=['driver', 'node', 'delay', 'weight']).to_csv(
            edges_file, index=False)
    else:  # mode == 'GMN'
        if cache_hit is not None:
            network = cache_hit['network']
            ccm_E, ccm_tau, node_dimensions = (
                cache_hit['ccm_E'], cache_hit['ccm_tau'], cache_hit['node_dimensions'])
        selected_union = []
        for t in targets:
            for c in mde_results[t]['SELECTED_COLUMNS']:
                if c not in selected_union:
                    selected_union.append(c)
        lagged_leaves = {}   # MODE='GMN' has no nested-MDE graph, so no leaves

    # Lagged leaf nodes are graph nodes in their own right, and
    # partition_network() classifies them as external inputs -- which gmn
    # feeds from real data BY NODE NAME (see _generalized_node_generate()).
    # So the exported frame needs one column per leaf, holding that
    # observable's value abs(delay) rows in the past, exactly as
    # expand_delay_columns() builds it for the MDE candidate pool.
    node_columns = targets + selected_union
    data_after_mde = (
        df[[n for n in node_columns if n not in lagged_leaves]]
        .iloc[:len(df)]
        .reset_index(drop=True)
    )
    for leaf, (observable, lag) in lagged_leaves.items():
        data_after_mde[leaf] = (
            df[observable].shift(-lag).iloc[:len(df)].reset_index(drop=True))
    data_after_mde = data_after_mde[node_columns]   # restore graph-node order
    data_after_mde.insert(0, 'Time', np.arange(len(data_after_mde)))
    data_after_mde_file = results_dir / f'data_after_mde_{network_id}.csv'
    data_after_mde.to_csv(data_after_mde_file, index=False)

    data_after_mde_lib = data_after_mde.iloc[:prediction_start]
    data_after_mde_lib_file = results_dir / f'data_after_mde_lib_{network_id}.csv'
    data_after_mde_lib.to_csv(data_after_mde_lib_file, index=False)

    if mode == 'MDE-GMN':
        network_dict = {node: list(graph.predecessors(node)) for node in graph.nodes}
        network = {'Graph': graph, 'Map': network_dict}
        network_file = results_dir / f'Network_nestedMDE_N{nested_n}_M{nested_m}_{network_id}.pkl'
        with open(network_file, 'wb') as f:
            pickle.dump(network, f)

        if fingerprint is not None and cache_hit is None:
            save_pickle_to_cache(cache_filename, {'fingerprint': fingerprint, 'graph': graph},
                                  results_dir, run_cache_dir)
    else:  # mode == 'GMN'
        if cache_hit is None:
            gmn_construction_start = time.perf_counter()
            _log_progress(progress_log_path, f'{network_id}: GMN network construction START')
            e_vals, tau_vals = list(e_range), list(tau_range)
            # target_E_tau_rho_grid always scores a TARGETS column's own
            # self-predictability -- always target_metric (continuous),
            # never metric.
            target_grids = [
                target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals, target_metric, gamma_delta)
                for t in targets
            ]
            mean_grid = np.nanmean(np.stack(target_grids), axis=0)
            best_i, best_j = np.unravel_index(np.nanargmax(mean_grid), mean_grid.shape)
            ccm_E, ccm_tau = e_vals[best_i], tau_vals[best_j]
            print(f'{network_id}: CCM E={ccm_E}, tau={ccm_tau} '
                  f'(grid-search {target_metric}={mean_grid[best_i, best_j]:.3f})', flush=True)

            nodes = [c for c in data_after_mde_lib.columns if c != 'Time']
            iMatrix = build_interaction_matrix(
                data_after_mde_lib, nodes, targets, ccm_E, ccm_tau, metric, target_metric,
                gamma_delta, ccm_window_rows, ccm_lib_size_pcts, ccm_sample, ccm_slope,
            )
            interaction_csv = results_dir / f'iMatrix_E{ccm_E}_tau{ccm_tau}_{network_id}_CCM.csv'
            iMatrix.to_csv(interaction_csv)

            node_dimensions = {
                node: estimate_node_dimension(
                    data_after_mde_lib, node, _metric_for(node, targets, metric, target_metric),
                    gamma_delta, max_e=max_node_dimension, fallback=fallback_node_dimension,
                )
                for node in iMatrix.index
            }
            print(f'{network_id}: node N_D =', node_dimensions, flush=True)

            # apps.CreateNetwork() keeps each node's *weakest* above-
            # threshold drivers instead of the strongest -- worked around
            # by pre-selecting the strongest N_D drivers ourselves, then
            # calling CreateNetwork with numDrivers set to the full column
            # count so its buggy truncation never has anything left to
            # remove.
            strongest_iMatrix = iMatrix.copy()
            for node in strongest_iMatrix.index:
                row = strongest_iMatrix.loc[node].drop(node)
                n_drivers = max(1, min(node_dimensions[node], len(row)))
                keep = row[row > threshold].sort_values(ascending=False).index[:n_drivers]
                drop = row.index.difference(keep)
                strongest_iMatrix.loc[node, drop] = threshold

            network_file = results_dir / f'Network_E{ccm_E}_tau{ccm_tau}_CCM_{network_id}.pkl'
            network = CreateNetwork(
                interactionMatrix = strongest_iMatrix,
                targetCols        = list(targets),
                threshold         = threshold,
                numDrivers        = len(strongest_iMatrix.columns),
                outputFile        = str(network_file),
                plotNetwork       = False,
            )
            _log_progress(
                progress_log_path,
                f'{network_id}: GMN network construction END -- '
                f'{time.perf_counter() - gmn_construction_start:.1f}s total')
        else:
            network_file = results_dir / f'Network_E{ccm_E}_tau{ccm_tau}_CCM_{network_id}.pkl'

        network['Graph'].graph['fallback_E'] = ccm_E
        network['Graph'].graph['fallback_tau'] = ccm_tau
        with open(network_file, 'wb') as f:
            pickle.dump(network, f)

        if fingerprint is not None and cache_hit is None:
            save_pickle_to_cache(cache_filename, {
                'fingerprint': fingerprint, 'network': network,
                'ccm_E': ccm_E, 'ccm_tau': ccm_tau, 'node_dimensions': node_dimensions,
            }, results_dir, run_cache_dir)

    for t in targets:
        print(f'{t} parents:', network['Map'][t], flush=True)

    return {
        'targets': list(targets),
        'network_id': network_id,
        'DATA_AFTER_MDE': data_after_mde_file,
        'NETWORK_FILE': network_file,
        'network': network,
        'ccm_E': ccm_E,
        'ccm_tau': ccm_tau,
        'node_dimensions': node_dimensions,
    }

def build_gmn_config(net_result, prediction_start, prediction_length, results_dir,
                      mode='Generate', function='SMap', theta=3,
                      node_theta_overrides=None, variant='', generate_length=None,
                      generate_start=None, max_offset=None, exclusion_radius=0):
    '''Stage the .cfg file gmn.GMN's configFile= constructor expects.

    exclusion_radius: written into the [EDM] section below and read
    straight through by gmn's own (monkeypatched) Node.Forecast/Generate as
    `Parameters.exclusionRadius`, passed to every pyEDM Simplex/SMap call
    gmn itself makes -- the same Theiler-window neighbor-exclusion
    mechanism as TSLS_MDE_GMN_EXCLUSION_RADIUS (this pipeline's MDE-side
    parameter), just applied to GMN's own recursive Generate()/Forecast()
    nearest-neighbor search rather than MDE's forward selection -- two
    independent EDM contexts with their own separate embeddings, so there's
    no single shared value that would make sense for both. This one is a
    plain static knob, not dynamically recomputed the way MDE's 'auto' mode
    would be (that mode has no equivalent here regardless -- see this
    module's own docstring).

    E/tau are no longer parameters here -- every node's real embedding now
    comes from _node_embedding_spec()/_embedding_spec_for() (per-driver
    delays off the network graph, plus self_delay_set or
    fallback_E/fallback_tau), not a shared network-wide (E, tau) written
    into this .cfg. Placeholder 'E'/'tau' values are still written into the
    [EDM] section purely because gmn.Node.__init__'s own (unpatched)
    arithmetic reads them before _generalized_node_init immediately
    corrects self.libEnd_i; their actual values are otherwise inert.

    generate_start (both modes): row index where Generate() starts
    free-running / where Forecast()'s pred window starts, independent of
    prediction_start -- the latter is the fixed MDE lib/pred boundary that
    MDE variable selection and the CCM/embedDimRhoMin/crossMapRhoMin gates
    validate against, and stays fixed regardless of what row a diagnostic
    Generate()/Forecast() call happens to start at. Defaults to
    prediction_start when None.

    generate_length (both modes): number of steps simulated/predicted from
    generate_start, independent of prediction_length. A full Generate()
    call over the whole held-out window can be the single most expensive
    step in the search loop, so decoupling this lets you shrink it without
    also shrinking the held-out window MDE/CCM validate against. Defaults
    to prediction_length when None. mode='Forecast' honors this same cap so
    its window matches Generate()'s -- run_forecast_diagnostic() plots both
    on the same axes.

    function is fixed to 'SMap' for every node in this pipeline -- gmn's
    Simplex branch never calls pyEDM.Embed() before predicting, so it can't
    do real Takens delay embedding. gmn's SMap branch does call
    Embed(E, tau, columns) first (now replaced by this pipeline's own
    generalized embedding).

    node_theta_overrides: optional {nodeName: theta} dict -- written as
    per-node .cfg files (gmn/Node.py's nodeConfigPath + f'{name}.cfg'
    lookup); every other node uses the network-wide `theta`. The trailing
    '/' on configPath below works around gmn/Node.py:78's naive string
    concatenation path lookup.

    max_offset (mode='Forecast' only): the network's largest per-node
    embedding offset (network_max_embedding_offset()), used to size a
    single, network-wide lib/pred window valid for every node's own
    embedding.'''
    if mode not in ('Generate', 'Forecast'):
        raise ValueError(f"Unknown mode: {mode!r}")
    Tp = 1
    generate_start = generate_start if generate_start is not None else prediction_start

    if mode == 'Generate':
        gmn_prediction_start = str(generate_start)
        gmn_prediction_length = str(generate_length if generate_length is not None else prediction_length)
        lib_range = ''
        pred_range = ''
    else:
        gmn_prediction_start = '1'
        gmn_prediction_length = '1'
        offset = max_offset if max_offset is not None else 0
        forecast_length = generate_length if generate_length is not None else prediction_length
        if offset >= generate_start or offset >= forecast_length - Tp:
            raise ValueError(
                f'build_gmn_config(mode=Forecast): embedding offset {offset} '
                f'leaves too few rows in generate_start={generate_start}/'
                f'forecast_length={forecast_length} for a valid lib/pred split '
                f'-- shrink TSLS_MDE_GMN_DELAY_SET or grow the prediction window.'
            )
        lib_range = f'1 {generate_start - offset}'
        pred_range = f'{generate_start - offset} {generate_start + forecast_length - Tp - offset}'

    node_cfg_dir = results_dir / f"node_cfgs_{net_result['network_id']}_{variant or mode.lower()}"

    def make_sections(node_theta):
        return {
            'GMN': {
                'mode': mode,
                'predictionStart': gmn_prediction_start,
                'predictionLength': gmn_prediction_length,
                'outPath': str(results_dir),
                'dataOutFile': '',
                'showPlot': 'False',
                'plotType': 'state',
                'plotColumns': '',
                'plotFile': '',
            },
            'Network': {
                'name': 'GMN',
                'targetNode': ','.join(net_result['targets']),
                'file': str(net_result['NETWORK_FILE']),
                'data': str(net_result['DATA_AFTER_MDE']),
            },
            'Node': {
                'info': '',
                'function': function,
                'configPath': str(node_cfg_dir) + '/' if node_theta_overrides else '',
                'data': '',
            },
            'EDM': {
                'lib': lib_range,
                'pred': pred_range,
                'E': '2',    # placeholder, inert -- see this function's docstring
                'Tp': str(Tp),
                'knn': '0',
                'tau': '-1',  # placeholder, inert -- see this function's docstring
                'theta': str(node_theta),
                'exclusionRadius': str(exclusion_radius),
                'columns': '',
                'target': '',
                'solver': '',
                'embedded': 'True',
                'validLib': '',
                'generateSteps': '0',
                'libSizes': '',
                'sample': '0',
                'random': 'False',
                'includeData': 'False',
                'seed': '0',
            },
            'Scale': {'factor': '1', 'offset': '0'},
        }

    def write_config(sections, path):
        config = ConfigParser()
        for section, values in sections.items():
            config[section] = values
        with open(path, 'w') as f:
            config.write(f)

    suffix = '' if mode == 'Generate' else f'_{mode.lower()}'
    if variant:
        suffix += f'_{variant}'
    config_file = results_dir / f"gmn_network_{net_result['network_id']}{suffix}.cfg"
    write_config(make_sections(theta), config_file)

    if node_theta_overrides:
        node_cfg_dir.mkdir(parents=True, exist_ok=True)
        for node_name, node_theta in node_theta_overrides.items():
            write_config(make_sections(node_theta), node_cfg_dir / f'{node_name}.cfg')

    return config_file
