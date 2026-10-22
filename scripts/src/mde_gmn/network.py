"""Building the network: turn per-node MDE results into one acyclic graph.

This is step 3 of the pipeline. mde.py answers "what predicts node X?" for a
single node; this module decides which nodes to ask about, in what order, and
how to wire the answers together into a graph GMN can simulate.

THE GRAPH
    Nodes are observables. An edge runs DRIVER -> NODE, i.e. in the direction
    the influence flows, so a node's *upstream* influences are its graph
    descendants. (That is easy to read backwards; _real_candidate_pool()
    depends on getting it right.)

    Two kinds of node:
      * real      an observable at delay 0, e.g. 'CA1_3_tsls'. Has drivers of
                  its own, so it can be expanded further.
      * lagged    a delayed copy, e.g. 'CA1_3_tsls::delay-9', carrying the
                  attributes lagged_of='CA1_3_tsls' and lag=9. These are LEAF
                  nodes: no in-edges, never expanded. They stand for input the
                  network does not explain, so the simulation feeds them in
                  from the recorded data.

TWO STAGES
    Stage 1 (real observables) starts at the behavioural targets and works
    outwards, breadth-first: run MDE on a node, add its selected drivers as
    edges, then run MDE on each of those drivers in turn. NESTED_M caps how
    many rounds of that expansion happen, so the backbone has bounded depth.

    Stage 2 (lagged observables) runs once over every node Stage 1 found, with
    delayed copies as the candidates -- of the node itself, of observables
    Stage 1 selected, and of ones it did not. Each pick becomes a leaf. Stage 2
    adds no real nodes and does not recurse, so NESTED_M governs Stage 1 alone.

STAYING ACYCLIC
    A cycle would make the simulation ill-defined, so a node's candidate pool
    excludes the node itself and everything upstream of it BEFORE the MDE call
    (_real_candidate_pool). Excluding beforehand rather than rejecting the
    pick afterwards matters: forward selection is greedy, so a pick that got
    thrown out would already have consumed one of the node's driver slots.
    Lagged candidates need no such exclusion -- a leaf has no in-edges and so
    cannot be part of any cycle, which is why a lagged copy of an upstream
    observable is a legal driver.

    Nodes in one breadth-first wave are evaluated in parallel, so two of them
    can pick each other. Whoever is added second is recomputed serially with
    the other's edges in place. tests/test_lagged_leaf_nodes.py is the
    executable spec for all of this.

AFTER THE GRAPH IS BUILT
    partition_network() splits the finished graph into the nodes GMN simulates
    and the leaves it reads from data; report_partition() prints that split.

Depends on: mde (reached through the module object -- see below), columns,
tsls_data, progress; networkx, numpy, pandas.
Imported by: network_parallel.py (GPU branch), pipeline, the entry scripts.

NOTE for tests: MDE is faked by assigning to the mde module
(mde_gmn.mde.run_mde_for_target = fake), so this module calls it as
`mde.run_mde_for_target(...)` through the module object. A
`from mde_gmn.mde import run_mde_for_target` here would bind the real function
at import time and the fake would never be seen -- the tests would silently
run real MDE. tests/_pipeline_test_utils.install_fake_mde() handles this.
ProcessPoolExecutor is replaced the same way, on this module.
"""

import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import networkx as nx
import numpy as np
import pandas as pd

from mde_gmn import mde          # module object, not its names -- see above
from mde_gmn.columns import (
    _delayed_column_name, effective_mde_dimension, expand_delay_columns,
    observable_and_delay)
from mde_gmn.progress import HEARTBEAT_SECONDS, _format_eta, _log_progress
from mde_gmn.tsls_data import _metric_for, _preview_window_rows


def _accept_mde_result_edges(graph, edges, node, result, raw_columns):
    '''Shared MDE-edge-acceptance rules -- self-loop guard, NaN
    marginal-gain rejection, eager cycle rejection (see that file's build_nested_mde_network() docstring
    for the full rationale on each). Applied to every column
    mde.run_mde_for_target() selected for `node`. Mutates `graph` in place and
    appends accepted rows to `edges`.

    Returns the list of (driver_node, delay) pairs actually accepted as
    edges into `node` (in selection order). `driver_node` is the graph node
    name, which for a lagged pick is the delayed column name, not the raw
    observable -- callers deciding what to recurse into should test
    `delay == 0`, never the name.'''
    accepted = []
    cols = result['SELECTED_COLUMNS']
    gains = np.diff(np.concatenate(([0.0], result['m'].MDErho)))

    for col, gain in zip(cols, gains):
        observable, delay = observable_and_delay(col, raw_columns)

        if observable == node and delay == 0:
            continue  # explicit self-loop guard (belt-and-suspenders)

        if np.isnan(gain):
            print(f'MDE edge accept: rejected {observable} (delay {delay}) -> {node} '
                  f'(marginal MDE rho gain is NaN -- treating as a dead-end edge)', flush=True)
            continue

        if delay == 0:
            # Real observable: one graph node per observable, which may
            # itself acquire drivers when a later round recurses into it.
            driver_node = observable
            graph.add_node(driver_node)
        else:
            # Lagged observable: ITS OWN LEAF NODE, never an edge out of
            # the raw observable's node. Two consequences, both intended:
            #   * a leaf never has in-edges, so it can never take part in
            #     a cycle -- which is what makes a lagged copy of an
            #     UPSTREAM observable a legal driver (the value it carries
            #     is that observable's past, so there is no circularity);
            #   * it has zero drivers by construction, so
            #     partition_network() classifies it as an external input,
            #     i.e. a purely exogenous driver of the network.
            # `lagged_of`/`lag` let consumers recover (observable, delay)
            # without re-parsing the name -- see _embedding_spec_for().
            driver_node = _delayed_column_name(observable, delay)
            graph.add_node(driver_node, lagged_of=observable, lag=delay)

        key = graph.add_edge(driver_node, node, delay=delay, weight=gain)

        if delay == 0 and not nx.is_directed_acyclic_graph(graph):
            # Should be unreachable: every stage excludes observables
            # upstream of `node` from its candidate pool BEFORE the MDE
            # call (see _real_candidate_pool()), so no accepted real
            # driver can close a cycle. Kept as a loud guard rather than
            # deleted -- if it ever fires, the exclusion is wrong.
            graph.remove_edge(driver_node, node, key=key)
            if graph.degree(driver_node) == 0:
                graph.remove_node(driver_node)
            print(f'MDE edge accept: rejected {driver_node} -> {node} (would create a cycle -- '
                  f'UNEXPECTED, upstream observables are excluded before MDE runs)', flush=True)
            continue

        edges.append({'driver': driver_node, 'node': node, 'delay': delay, 'weight': gain})
        accepted.append((driver_node, delay))

    return accepted

def _real_candidate_pool(graph, node, real_columns, raw_columns):
    '''Stage 1's candidate pool for `node`: every real (delay=0) observable
    except `node` itself and except any observable UPSTREAM of `node` in the
    network built so far.

    Upstream observables are excluded here, before the MDE call, rather than
    having their edges rejected after selection. Selecting one would close a
    cycle, so the edge could never be kept -- and forward selection is
    greedy, so a candidate that gets picked and then thrown away has still
    consumed one of the node's driver slots and skewed every later round it
    conditioned.

    The blocked set is `nx.descendants(graph, node)`, NOT `nx.ancestors`:
    edges run driver -> node (from a driver to the thing it drives), so
    accepting candidate C adds C -> node, which closes a cycle exactly when
    node already reaches C. Those are node's descendants in edge direction
    -- which are the nodes between it and the behavioural targets, i.e.
    "upstream of this node" in the causal sense the network describes.

    Lagged candidates need no such filter (see _accept_mde_result_edges():
    they become leaves and cannot cycle), which is why Stage 2 does not use
    this function. A lagged leaf can never appear in this blocked set
    either, since a leaf has no in-edges and so is reachable from nothing.'''
    blocked = {node} | (nx.descendants(graph, node) if node in graph else set())
    return [c for c in real_columns
            if observable_and_delay(c, raw_columns)[0] not in blocked]

def build_nested_mde_network(targets, expanded_df, raw_columns, expanded_columns,
                              prediction_start, prediction_length, nested_n, nested_m,
                              embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                              results_dir, mde_results, delay_set, tp,
                              metric, target_metric, gamma_delta,
                              target_ranking_window_rows, min_pred_events, max_window_rows,
                              ccm_lib_sizes_rows, ccm_sample, ccm_slope, cores,
                              lag_set,
                              reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                              data_file=None, data_file_sha256=None, code_fingerprint=None,
                              progress_log_path=None, mp_start_method='fork', use_gpu=False):
    '''Two-stage nested-MDE network construction, MODE='MDE-GMN' only.

    STAGE 1 builds the backbone out of REAL (delay=0) observables. It starts
    at the behavioural targets, runs one MDE call per node, turns that
    node's selected drivers into edges, and recurses into each selected
    driver -- up to `nested_m` further rounds. A node's candidate pool
    excludes itself and every observable upstream of it, so no selected
    driver can close a cycle (see _real_candidate_pool()).

    STAGE 2 then runs ONCE over every node Stage 1 discovered, with the
    LAGGED columns as candidates (every observable's lag embeddings,
    whether or not Stage 1 selected that observable). Each lagged pick
    becomes its own LEAF node -- zero drivers, therefore a purely external
    input to the network, and structurally unable to create a cycle (see
    _accept_mde_result_edges()). Stage 2 cannot extend the backbone and
    does not recurse, so `nested_m` applies to Stage 1 alone.

    Stage 1 runs to completion before Stage 2 starts. That ordering matters:
    Stage 2 attaches leaves to the nodes Stage 1 found, so it cannot run until
    the backbone is final.

    DEPTH. A node's depth is its distance from the nearest target, whichever
    stage discovered it. `nested_m` caps it: a node at depth d expands further
    only while `nested_m == 0 or d < nested_m` (0 means no cap). Because Stage 1
    runs to its full depth first, a branch already carried to the cap simply
    stops -- that is the budget working, not a broken recursion.

    WAVES, AND WHY DEPTH IS TRACKED AS A MINIMUM. Nodes are processed in waves,
    all nodes of one wave in parallel. A wave can contain nodes at different
    depths, and two parents in one wave can both propose the same new driver at
    different depths. Such a driver is recorded at the SHALLOWEST proposed
    depth (the dict-of-minimums below), so its depth does not depend on
    iteration order.

    WITHIN-WAVE RACE. A wave's candidate pools are built before any of that
    wave's results come back, so two nodes in one wave can select each other and
    neither pool could have known. Whichever is accepted second is recomputed
    serially with the other's edges in place -- see _recomputed_stage1_result().

    HOW THE CORES ARE SPLIT. Per wave: outer_workers = min(nodes in the wave,
    cores), inner_workers = cores // outer_workers. The outer pool runs one
    process per node; the inner count becomes a thread pool opened inside
    mde.run_mde_for_target() (see its INNER POOL docstring), because proper_mde
    has no parallelism of its own to hand a core count to. The product never
    exceeds `cores`, and is logged so that can be checked.'''
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(targets)

    stage1_num_delays = len(expanded_columns) // len(raw_columns)

    def _recomputed_stage1_result(node, result):
        """`result`, or a fresh serial Stage 1 MDE when `result` selected a
        real observable that is cycle-closing for `node`.

        Needed because a wave's candidate pools are built before any of that
        wave's results come back: two nodes in the same wave can select each
        other, and the second one's pool could not have known. Rejecting the
        offending edge afterwards is not equivalent -- forward selection is
        greedy, so the discarded pick has already consumed a driver slot and
        conditioned every later round. See _real_candidate_pool()."""
        blocked = {node} | (nx.descendants(graph, node) if node in graph else set())
        conflicts = sorted({
            observable for observable, delay in
            (observable_and_delay(c, raw_columns) for c in result['SELECTED_COLUMNS'])
            if delay == 0 and observable in blocked})
        if not conflicts:
            return result
        print(f'nested MDE (Stage 1): recomputing {node} -- its selection includes '
              f'{conflicts}, which would close a cycle now that the rest of this wave has '
              f'been accepted', flush=True)
        _log_progress(progress_log_path, f'stage1 {node} RECOMPUTE (cycle-closing {conflicts})')
        pool = _real_candidate_pool(graph, node, expanded_columns, raw_columns)
        try:
            return mde.run_mde_for_target(
                node, expanded_df, pool, prediction_start, prediction_length,
                effective_mde_dimension(nested_n, pool, stage1_num_delays),
                embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                results_dir, delay_set,
                tp=tp, metric=metric,
                target_metric=_metric_for(node, targets, metric, target_metric),
                gamma_delta=gamma_delta, ccm_sample=ccm_sample, ccm_slope=ccm_slope,
                is_behavioral_target=(node in targets),
                target_ranking_window_rows=target_ranking_window_rows,
                min_pred_events=min_pred_events, max_window_rows=max_window_rows,
                ccm_lib_sizes_rows=ccm_lib_sizes_rows,
                reuse_cache=reuse_cache, prev_runs_dir=prev_runs_dir,
                run_cache_dir=run_cache_dir, data_file=data_file,
                data_file_sha256=data_file_sha256, code_fingerprint=code_fingerprint,
                max_e=nested_n, inner_workers=1, raw_columns=raw_columns,
                use_gpu=use_gpu,
            )
        except Exception as exc:
            if mde.is_gpu_memory_failure(exc):
                print(f'nested MDE (Stage 1): {node} recompute hit a GPU MEMORY failure: '
                      f'{exc!r} -- STOPPING the run. {mde._GPU_MEMORY_STOP_ADVICE}', flush=True)
                raise
            print(f'nested MDE (Stage 1): {node} recompute FAILED: {exc!r} -- treating as a '
                  f'dead end', flush=True)
            return mde._failed_mde_result(node)

    # Stage 2's candidates are the LAGGED columns only -- every observable's
    # lag embeddings (its own, other backbone nodes', and observables Stage 1
    # never selected). Real (delay=0) observables are Stage 1's business, so
    # they are dropped from this pool: expand_delay_columns() returns the raw
    # columns first, then one column per nonzero delay.
    stage2_df, stage2_expanded_columns = expand_delay_columns(expanded_df, raw_columns, lag_set)
    stage2_candidate_universe = stage2_expanded_columns[len(raw_columns):]
    stage2_num_delays = len(lag_set)

    edges = []
    nested_mde_results = {}
    node_depth = {t: 0 for t in targets}

    # =====================================================================
    # PHASE 1 -- Stage 1: real-observable-only BFS, run to full completion
    # (every depth the nested_m budget allows) before Stage 2 starts at all.
    # =====================================================================
    visited1 = set()
    current_wave = [(t, 0) for t in targets]
    stage1_start = time.perf_counter()
    stage1_durations = []   # seconds per completed node, this run only
    stage1_rates = []       # seconds per resolved window-row, this run only -- see _format_eta()
    _log_progress(progress_log_path, 'STAGE 1 START (real-observable BFS)')

    while current_wave:
        wave_nodes, seen = [], set()
        for node, depth in current_wave:
            if node in visited1 or node in seen:
                continue
            seen.add(node)
            wave_nodes.append((node, depth))

        if not wave_nodes:
            break

        to_compute = [(node, depth) for node, depth in wave_nodes
                      if not (depth == 0 and node in mde_results)]
        stage1_results = {
            node: mde_results[node] for node, depth in wave_nodes
            if depth == 0 and node in mde_results
        }

        if to_compute:
            outer_workers = min(len(to_compute), max(1, cores))
            inner_workers = max(1, cores // outer_workers)
            # product logged explicitly so
            # a run's own log can be checked for outer*inner <= cores
            # without re-deriving it by hand.
            print(f'nested MDE (Stage 1): wave of {len(to_compute)} node(s) at depth '
                  f'{to_compute[0][1]} -- {outer_workers} parallel worker(s) '
                  f'x {inner_workers} inner thread(s) each '
                  f'(product={outer_workers * inner_workers}, cores={cores})', flush=True)
            wave_start = time.perf_counter()
            _log_progress(progress_log_path,
                          f'stage1 wave START -- {len(to_compute)} node(s) at depth '
                          f'{to_compute[0][1]}: {[n for n, _ in to_compute]}')

            # mp_start_method: 'fork' (default, CPU-only runs -- unchanged
            # behaviour) or 'spawn' when the GPU/torch path is enabled, since
            # CUDA does not support fork(). See load_config()'s GPU_ENABLED
            # docstring and docs/no-kEDM-gpu-optimization-plan.md section 7a.
            with ProcessPoolExecutor(max_workers=outer_workers,
                                      mp_context=mp.get_context(mp_start_method)) as executor:
                futures = {}
                node_window_estimate = {}
                for node, depth in to_compute:
                    # Upstream observables excluded BEFORE the MDE call --
                    # see _real_candidate_pool()'s docstring for why this
                    # is not equivalent to rejecting their edges afterwards.
                    candidate_pool = _real_candidate_pool(graph, node, expanded_columns, raw_columns)
                    effective_d = effective_mde_dimension(nested_n, candidate_pool, stage1_num_delays)
                    is_behav = (node in targets)
                    # Monitoring-log ETA input only -- see _preview_window_rows()'s
                    # own docstring for why this speculative resolve is safe and
                    # independent of the real one mde.run_mde_for_target() performs.
                    node_window_estimate[node] = _preview_window_rows(
                        node, expanded_df, delay_set, prediction_start, prediction_length, tp,
                        is_behav, target_ranking_window_rows, min_pred_events, max_window_rows,
                    )
                    # `node` here is virtually never
                    # one of `targets` (Stage 1 discovers real-observable
                    # DRIVERS) -- passing the closed-over `target_metric`
                    # unconditionally would score this node's own ranking
                    # stage/E-auto-select via TARGET_METRIC ('correlation')
                    # instead of _metric_for()'s own rule ('f1' for any
                    # non-TARGETS column). See mde.run_mde_for_target()'s own
                    # docstring for why target_metric drives that stage
                    # (predictions of `target` == `node` here) -- the CCM
                    # gate's own `metric` argument is untouched, already
                    # correct (it always scores a candidate, never `node`
                    # itself, and no candidate here is ever a target).
                    # All arguments from `tp` onward passed by keyword --
                    # this call site inserts new parameters (window-sizing
                    # fix) ahead of several trailing ones (reuse_cache
                    # onward); relying on positional order here would be a
                    # silent-miswiring trap the moment either signature
                    # changes again.
                    fut = executor.submit(
                        mde.run_mde_for_target, node, expanded_df, candidate_pool,
                        prediction_start, prediction_length, effective_d,
                        embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                        results_dir, delay_set,
                        tp=tp, metric=metric,
                        target_metric=_metric_for(node, targets, metric, target_metric),
                        gamma_delta=gamma_delta, ccm_sample=ccm_sample, ccm_slope=ccm_slope,
                        is_behavioral_target=is_behav,
                        target_ranking_window_rows=target_ranking_window_rows,
                        min_pred_events=min_pred_events, max_window_rows=max_window_rows,
                        ccm_lib_sizes_rows=ccm_lib_sizes_rows,
                        reuse_cache=reuse_cache, prev_runs_dir=prev_runs_dir,
                        run_cache_dir=run_cache_dir, data_file=data_file,
                        data_file_sha256=data_file_sha256, code_fingerprint=code_fingerprint,
                        max_e=nested_n, inner_workers=inner_workers, raw_columns=raw_columns,
                        use_gpu=use_gpu,
                    )
                    futures[fut] = node

                # Heartbeat-polling replacement for a plain `for fut in
                # as_completed(futures):` -- see this module's own
                # HEARTBEAT_SECONDS docstring for why: a wave with no
                # heartbeat can go silent for hours (job 38976891) with no
                # way to tell "still working" from "stuck". Each timeout
                # with nothing newly done logs one heartbeat line; each
                # newly completed future is logged (and timed) immediately.
                pending = set(futures)
                while pending:
                    done, pending = wait(pending, timeout=HEARTBEAT_SECONDS,
                                          return_when=FIRST_COMPLETED)
                    if not done:
                        elapsed = time.perf_counter() - wave_start
                        still_waiting = sorted(futures[f] for f in pending)
                        remaining_rows = sum(node_window_estimate.get(futures[f], 0)
                                              for f in pending)
                        eta = _format_eta(stage1_durations, len(pending),
                                          per_unit=stage1_rates, unit_remaining=remaining_rows)
                        _log_progress(
                            progress_log_path,
                            f'stage1 wave HEARTBEAT -- {len(futures) - len(pending)}/{len(futures)} '
                            f'done, elapsed={elapsed:.0f}s, still waiting on: {still_waiting}{eta}')
                        continue
                    for fut in done:
                        node = futures[fut]
                        node_duration = time.perf_counter() - wave_start
                        try:
                            stage1_results[node] = fut.result()
                            _log_progress(
                                progress_log_path,
                                f'stage1 node END -- {node} duration={node_duration:.1f}s '
                                f'selected={stage1_results[node]["SELECTED_COLUMNS"]}')
                        except Exception as exc:
                            # See mde._failed_mde_result()'s own docstring -- a
                            # single node's MDE call failing must not crash the
                            # rest of this wave or every already-completed node
                            # in prior waves.
                            if mde.is_gpu_memory_failure(exc):
                                print(f'nested MDE (Stage 1): {node} MDE call hit a GPU MEMORY '
                                      f'failure: {exc!r} -- STOPPING the run. '
                                      f'{mde._GPU_MEMORY_STOP_ADVICE}', flush=True)
                                raise
                            print(f'nested MDE (Stage 1): {node} MDE call FAILED: {exc!r} -- treating '
                                  f'as a dead-end (zero drivers selected, no further nesting from this '
                                  f'node)', flush=True)
                            _log_progress(
                                progress_log_path,
                                f'stage1 node END -- {node} duration={node_duration:.1f}s FAILED: {exc!r}')
                            stage1_results[node] = mde._failed_mde_result(node)
                        stage1_durations.append(node_duration)
                        window_est = node_window_estimate.get(node, 0)
                        if window_est > 0:
                            stage1_rates.append(node_duration / window_est)

            _log_progress(progress_log_path,
                          f'stage1 wave END -- {len(to_compute)} node(s), '
                          f'{time.perf_counter() - wave_start:.1f}s total')

        next_wave = {}
        for node, depth in wave_nodes:
            visited1.add(node)
            node_depth[node] = depth
            result1 = _recomputed_stage1_result(node, stage1_results[node])
            nested_mde_results[node] = result1

            accepted1 = _accept_mde_result_edges(graph, edges, node, result1, raw_columns)
            for driver, delay in accepted1:
                if delay == 0 and (nested_m == 0 or depth < nested_m) and driver not in visited1:
                    next_wave[driver] = min(next_wave.get(driver, depth + 1), depth + 1)

        current_wave = list(next_wave.items())

    _log_progress(progress_log_path,
                  f'STAGE 1 END -- {len(visited1)} node(s), '
                  f'{time.perf_counter() - stage1_start:.1f}s total')
    print(f'nested MDE (Stage 1): completed -- {len(visited1)} node(s) in the real-observable '
          f'backbone, Stage 2 starts now', flush=True)

    # =====================================================================
    # PHASE 2 -- Stage 2: attach lagged drivers. ONE non-recursive pass over
    # every node Stage 1 discovered, so each backbone node gets its lagged
    # embedding coordinates. Stage 2 cannot grow the backbone: its
    # candidates are lagged columns only, and every lagged pick becomes a
    # LEAF node (see _accept_mde_result_edges()), which by definition has no
    # drivers of its own to recurse into. There is therefore no depth budget
    # to apply here -- nested_m governs Stage 1's recursion alone.
    # =====================================================================
    visited2 = set()
    current_wave = [(node, node_depth[node]) for node in visited1]
    stage2_start = time.perf_counter()
    stage2_durations = []
    stage2_rates = []
    _log_progress(progress_log_path, 'STAGE 2 START (lagged-driver attachment)')

    while current_wave:
        wave_nodes, seen = [], set()
        for node, depth in current_wave:
            if node in visited2 or node in seen:
                continue
            seen.add(node)
            wave_nodes.append((node, depth))

        if not wave_nodes:
            break

        wave_depths = sorted({d for _, d in wave_nodes})
        outer_workers2 = min(len(wave_nodes), max(1, cores))
        inner_workers2 = max(1, cores // outer_workers2)
        # Product logged explicitly so the core budget can be checked from
        # the log, same as Stage 1's print above.
        print(f'nested MDE (Stage 2): wave of {len(wave_nodes)} node(s) at depth(s) '
              f'{wave_depths} -- {outer_workers2} parallel worker(s) '
              f'x {inner_workers2} inner thread(s) each '
              f'(product={outer_workers2 * inner_workers2}, cores={cores}), candidate pool size '
              f'{len(stage2_candidate_universe)} (lagged columns only, lag_set={lag_set})', flush=True)
        wave_start = time.perf_counter()
        _log_progress(progress_log_path,
                      f'stage2 wave START -- {len(wave_nodes)} node(s) at depth(s) '
                      f'{wave_depths}: {[n for n, _ in wave_nodes]}')

        stage2_results = {}
        with ProcessPoolExecutor(max_workers=outer_workers2,
                                  mp_context=mp.get_context(mp_start_method)) as executor:
            futures = {}
            node_window_estimate = {}
            for node, depth in wave_nodes:
                candidate_pool = [c for c in stage2_candidate_universe
                                   if observable_and_delay(c, raw_columns)[0] != node]
                effective_d = effective_mde_dimension(nested_n, candidate_pool, stage2_num_delays)
                is_behav = (node in targets)
                # Monitoring-log ETA input only -- see Stage 1's own
                # identical comment above / _preview_window_rows()'s
                # docstring. Uses stage2_df/lag_set (Stage 2's own frame
                # and delay set), matching the real call just below.
                node_window_estimate[node] = _preview_window_rows(
                    node, stage2_df, lag_set, prediction_start, prediction_length, tp,
                    is_behav, target_ranking_window_rows, min_pred_events, max_window_rows,
                )
                # Metric routing -- see Stage 1's identical comment above.
                # `node` here is always a Stage 1 backbone node, since Stage 2
                # only attaches leaves and never discovers new real nodes.
                # Arguments from `tp` onward are passed by keyword for the
                # reason Stage 1's comment gives.
                fut = executor.submit(
                    mde.run_mde_for_target, node, stage2_df, candidate_pool,
                    prediction_start, prediction_length, effective_d,
                    embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                    results_dir, lag_set,
                    tp=tp, metric=metric,
                    target_metric=_metric_for(node, targets, metric, target_metric),
                    gamma_delta=gamma_delta, ccm_sample=ccm_sample, ccm_slope=ccm_slope,
                    is_behavioral_target=is_behav,
                    target_ranking_window_rows=target_ranking_window_rows,
                    min_pred_events=min_pred_events, max_window_rows=max_window_rows,
                    ccm_lib_sizes_rows=ccm_lib_sizes_rows,
                    reuse_cache=reuse_cache, prev_runs_dir=prev_runs_dir,
                    run_cache_dir=run_cache_dir, data_file=data_file,
                    data_file_sha256=data_file_sha256, code_fingerprint=code_fingerprint,
                    cache_stage='stage2',
                    max_e=nested_n, inner_workers=inner_workers2, raw_columns=raw_columns,
                    use_gpu=use_gpu,
                )
                futures[fut] = node

            # Heartbeat-polling -- see Stage 1's own identical comment above.
            pending = set(futures)
            while pending:
                done, pending = wait(pending, timeout=HEARTBEAT_SECONDS,
                                      return_when=FIRST_COMPLETED)
                if not done:
                    elapsed = time.perf_counter() - wave_start
                    still_waiting = sorted(futures[f] for f in pending)
                    remaining_rows = sum(node_window_estimate.get(futures[f], 0) for f in pending)
                    eta = _format_eta(stage2_durations, len(pending),
                                      per_unit=stage2_rates, unit_remaining=remaining_rows)
                    _log_progress(
                        progress_log_path,
                        f'stage2 wave HEARTBEAT -- {len(futures) - len(pending)}/{len(futures)} '
                        f'done, elapsed={elapsed:.0f}s, still waiting on: {still_waiting}{eta}')
                    continue
                for fut in done:
                    node = futures[fut]
                    node_duration = time.perf_counter() - wave_start
                    try:
                        stage2_results[node] = fut.result()
                        _log_progress(
                            progress_log_path,
                            f'stage2 node END -- {node} duration={node_duration:.1f}s '
                            f'selected={stage2_results[node]["SELECTED_COLUMNS"]}')
                    except Exception as exc:
                        if mde.is_gpu_memory_failure(exc):
                            print(f'nested MDE (Stage 2): {node} MDE call hit a GPU MEMORY '
                                  f'failure: {exc!r} -- STOPPING the run. '
                                  f'{mde._GPU_MEMORY_STOP_ADVICE}', flush=True)
                            raise
                        print(f'nested MDE (Stage 2): {node} MDE call FAILED: {exc!r} -- treating as '
                              f'a dead-end (zero lagged drivers attached for this node)', flush=True)
                        _log_progress(
                            progress_log_path,
                            f'stage2 node END -- {node} duration={node_duration:.1f}s FAILED: {exc!r}')
                        stage2_results[node] = mde._failed_mde_result(node)
                    stage2_durations.append(node_duration)
                    window_est = node_window_estimate.get(node, 0)
                    if window_est > 0:
                        stage2_rates.append(node_duration / window_est)

        _log_progress(progress_log_path,
                      f'stage2 wave END -- {len(wave_nodes)} node(s), '
                      f'{time.perf_counter() - wave_start:.1f}s total')

        for node, depth in wave_nodes:
            visited2.add(node)
            node_depth.setdefault(node, depth)
            result2 = stage2_results[node]
            # Every accepted pick here is a lagged leaf, so nothing is
            # enqueued: Stage 2 is a single pass by construction.
            _accept_mde_result_edges(graph, edges, node, result2, raw_columns)

        current_wave = []

    _log_progress(progress_log_path,
                  f'STAGE 2 END -- {len(visited2)} node(s), '
                  f'{time.perf_counter() - stage2_start:.1f}s total')

    edges_df = pd.DataFrame(edges, columns=['driver', 'node', 'delay', 'weight'])

    # nested_m is a whole-network depth budget shared by both stages
    # (node_depth is one dict populated by whichever stage discovers a
    # node first, checked against the same nested_m in both phases' own
    # recursion above) -- realized_max_depth reports how deep that joint
    # budget actually got used this run, distinct from the configured cap
    # itself (nested_m_display below).
    realized_max_depth = max(node_depth.values()) if node_depth else 0
    nested_m_display = 'unlimited' if nested_m == 0 else nested_m
    print(f'nested MDE: {len(graph.nodes)} nodes, {len(edges_df)} edges '
          f'(N={nested_n}, M={nested_m_display}, realized max depth={realized_max_depth})',
          flush=True)
    _log_progress(progress_log_path,
                  f'NESTED MDE NETWORK COMPLETE -- {len(graph.nodes)} nodes, {len(edges_df)} edges')

    return graph, edges_df, nested_mde_results

def partition_network(graph, targets, external_input_mode):
    '''partitions a MultiDiGraph built by
    build_nested_mde_network() into interior nodes (>=1 discovered driver
    -- genuinely simulated, their embedding built from those in-edges only,
    see _embedding_spec_for()'s docstring) and external inputs (0 discovered
    drivers -- what EXTERNAL_INPUT_MODE does with them is that function's
    own concern, not this one's). Stored explicitly on the graph object
    (`graph.graph['interior_nodes']`/`['external_inputs']`/
    `['external_input_mode']`) rather than left implicit in edge counts, so
    every consumer (embedding-spec construction, theta search, the 3D
    figure, run-log reporting) reads the same authoritative classification
    instead of re-deriving it.

    A TARGET with zero discovered drivers is NEVER classified as an
    external input, regardless of `external_input_mode` -- raises instead.
    This is a failed network, not a valid one: feeding a target's own real
    values back in as its "simulation" (EXTERNAL_INPUT_MODE='real_data')
    and then scoring Generate() fidelity against those same real values
    would be circular, producing a spuriously perfect score for exactly the
    case that most needs to be flagged as a failure (confirmed against Run
    A: head_position_y selected zero real drivers at depth 0 in that run --
    this exemption exists because that is a real, not hypothetical, case).

    Mutates `graph` in place; returns (interior_nodes, external_inputs) as
    plain sets for convenience.'''
    interior_nodes = set()
    external_inputs = set()
    for node in graph.nodes:
        if graph.in_degree(node) == 0:
            if node in targets:
                raise ValueError(
                    f"{node}: target selected zero drivers after nested-MDE (both stages) -- "
                    f"this is a failed network, not a valid external input. A target can never "
                    f"be exogenous: EXTERNAL_INPUT_MODE='real_data' would feed its own real "
                    f"values back in as its \"simulation\" and then score fidelity against those "
                    f"same real values, which is circular. Investigate why {node} found no "
                    f"drivers (EMBED_DIM_RHO_MIN/CROSS_MAP_RHO_MIN/CCM_PREFILTER_RHO_MIN too "
                    f"strict? too few candidate observables?) rather than working around it here."
                )
            external_inputs.add(node)
        else:
            interior_nodes.add(node)

    graph.graph['interior_nodes'] = interior_nodes
    graph.graph['external_inputs'] = external_inputs
    graph.graph['external_input_mode'] = external_input_mode
    return interior_nodes, external_inputs

def _simulated_nodes(graph):
    '''Nodes that actually go through gmn's own SMap/embedding machinery --
    interior nodes always, plus external inputs too under
    EXTERNAL_INPUT_MODE='self_simulate' (where they keep the old
    unconditional self_delay_set embedding, see _embedding_spec_for()'s
    docstring). Excludes real-data-mode external inputs: their Generate()/
    Forecast() short-circuit to a real-data passthrough with no embedding
    at all, so retuning theta for them (retune_theta_for_all_nodes(), which
    calls _embedding_spec_for() directly) would either waste effort on an
    inert result or crash on an empty spec.'''
    external_inputs = graph.graph.get('external_inputs', set())
    mode = graph.graph.get('external_input_mode', 'real_data')
    if mode == 'self_simulate':
        return list(graph.nodes)
    return [n for n in graph.nodes if n not in external_inputs]

def report_partition(graph, network_id, results_dir):
    '''Report the partition: how many external inputs there are, which
    observables and delays they stand for, and which interior nodes each feeds --
    printed to the run log and written to a per-network CSV in
    `results_dir` (partition_{network_id}.csv). Reads the classification
    partition_network() already stored on `graph.graph` rather than
    recomputing it.'''
    external_inputs = graph.graph.get('external_inputs', set())
    interior_nodes = graph.graph.get('interior_nodes', set())
    mode = graph.graph.get('external_input_mode', 'real_data')

    print(f'{network_id}: partition -- {len(interior_nodes)} interior node(s), '
          f'{len(external_inputs)} external input(s) (EXTERNAL_INPUT_MODE={mode!r})', flush=True)

    rows = []
    for node in sorted(external_inputs):
        out_edges = list(graph.out_edges(node, data=True))
        if not out_edges:
            print(f'  external input {node}: feeds nothing (dead-end leaf)', flush=True)
            rows.append({'node': node, 'role': 'external_input', 'feeds': '', 'delay': ''})
            continue
        for _, fed_node, data in out_edges:
            print(f'  external input {node} -> {fed_node} (delay {data["delay"]})', flush=True)
            rows.append({'node': node, 'role': 'external_input', 'feeds': fed_node,
                         'delay': data['delay']})
    for node in sorted(interior_nodes):
        rows.append({'node': node, 'role': 'interior', 'feeds': '', 'delay': ''})

    partition_file = results_dir / f'partition_{network_id}.csv'
    pd.DataFrame(rows, columns=['node', 'role', 'feeds', 'delay']).to_csv(partition_file, index=False)
