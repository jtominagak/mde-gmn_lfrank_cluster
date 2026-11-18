"""Building the network: turn per-node MDE results into one acyclic graph.

Step 3 of the pipeline. mde.py answers "what predicts node X?" for a single
node; this module decides which nodes to ask about, in what order, and how to
wire the answers together into a graph GMN can simulate.

THE GRAPH
    Nodes are observables. An edge runs DRIVER -> NODE, i.e. in the direction
    the influence flows, so a node's *upstream* influences are its graph
    descendants. (That is easy to read backwards, and _real_candidate_pool()
    depends on getting it right.)

    Two kinds of node:
      * real      an observable at delay 0, e.g. 'CA1_3_tsls'. Has drivers of
                  its own, so it can be expanded further.
      * lagged    a delayed copy, e.g. 'CA1_3_tsls::delay-9', carrying the
                  attributes lagged_of='CA1_3_tsls' and lag=-9. These are LEAF
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

    Stage 1 runs to completion before Stage 2 starts, because Stage 2 attaches
    leaves to the nodes Stage 1 found.

STAYING ACYCLIC
    A cycle would make the simulation ill-defined, so a node's candidate pool
    excludes the node itself and everything upstream of it BEFORE the MDE call
    (_real_candidate_pool). Excluding beforehand rather than rejecting the
    pick afterwards matters: forward selection is greedy, so a pick that got
    thrown out would already have consumed one of the node's driver slots and
    conditioned every later round.

    Lagged candidates need no such exclusion -- a leaf has no in-edges and so
    cannot be part of any cycle, which is why a lagged copy of an upstream
    observable is a legal driver.

    Nodes in one breadth-first wave are evaluated in parallel, so two of them
    can pick each other and neither pool could have known. Whoever is added
    second is recomputed serially with the other's edges in place.
    tests/test_lagged_leaf_nodes.py is the executable spec for all of this.

AFTER THE GRAPH IS BUILT
    partition_network() splits the finished graph into the nodes GMN simulates
    and the leaves it reads from data; report_partition() prints that split.

WHERE THE PARALLELISM WENT. Both stages' pool handling, heartbeat logging and
failure policy live in waves.py, which is why the two stage functions below
read as graph walks rather than as executor plumbing.

Depends on: mde (through the module object -- see below), waves, columns,
tsls_data, progress; networkx, numpy, pandas.
Imported by: network_parallel, gmn_build, pipeline.

NOTE for tests: MDE is faked by assigning to the mde module
(mde_gmn.mde.run_mde_for_target = fake), so this module calls it as
`mde.run_mde_for_target(...)` through the module object. A
`from mde_gmn.mde import run_mde_for_target` here would bind the real function
at import time and the fake would never be seen -- the tests would silently
run real MDE. tests/_pipeline_test_utils.install_fake_mde() handles this.
ProcessPoolExecutor is replaced the same way, on waves.
"""

import time

import networkx as nx
import numpy as np
import pandas as pd

from mde_gmn import mde          # module object, not its names -- see above
from mde_gmn import waves
from mde_gmn.columns import (
    _delayed_column_name, effective_mde_dimension, expand_delay_columns,
    observable_and_delay)
from mde_gmn.progress import _log_progress
from mde_gmn.tsls_data import _metric_for, _preview_window_rows


# =====================================================================
# Accepting what MDE picked
# =====================================================================

def _accept_mde_result_edges(graph, edges, node, result, raw_columns):
    '''Turn one node's MDE selection into edges, applying every rejection rule.

    Three rules, all of which have to be here rather than inside MDE: the
    self-loop guard, NaN marginal gain (a candidate whose contribution could
    not be measured is a dead-end edge, not a weak one), and the
    cycle-rejection backstop.

    Mutates `graph` and appends accepted rows to `edges`. Returns the
    (driver_node, delay) pairs accepted, in selection order. For a lagged
    pick `driver_node` is the DELAYED column name, not the raw observable,
    so a caller deciding what to recurse into must test `delay == 0` and
    never the name.
    '''
    accepted = []
    columns = result['SELECTED_COLUMNS']
    # MDErho is cumulative, so the marginal gain of each pick is the
    # difference from the previous one; the first is measured from 0.
    gains = np.diff(np.concatenate(([0.0], result['m'].MDErho)))

    for column, gain in zip(columns, gains):
        observable, delay = observable_and_delay(column, raw_columns)

        if observable == node and delay == 0:
            continue                      # self-loop guard (belt and braces)

        if np.isnan(gain):
            print(f'MDE edge accept: rejected {observable} (delay {delay}) -> {node} '
                  f'(marginal MDE rho gain is NaN -- treating as a dead-end edge)',
                  flush=True)
            continue

        if delay == 0:
            # Real observable: one graph node per observable, which may
            # itself acquire drivers when a later round recurses into it.
            driver_node = observable
            graph.add_node(driver_node)
        else:
            # Lagged observable: ITS OWN LEAF NODE, never an edge out of the
            # raw observable's node. Two consequences, both intended: a leaf
            # never has in-edges, so it can never take part in a cycle
            # (which is what makes a lagged copy of an UPSTREAM observable a
            # legal driver -- the value it carries is that observable's
            # past, so there is no circularity); and it has zero drivers by
            # construction, so partition_network() classifies it as an
            # external input. lagged_of/lag let consumers recover
            # (observable, delay) without re-parsing the name.
            driver_node = _delayed_column_name(observable, delay)
            graph.add_node(driver_node, lagged_of=observable, lag=delay)

        key = graph.add_edge(driver_node, node, delay=delay, weight=gain)

        if delay == 0 and not nx.is_directed_acyclic_graph(graph):
            # Should be unreachable: every stage excludes observables
            # upstream of `node` from its candidate pool before the MDE call
            # (_real_candidate_pool). Kept as a loud guard rather than
            # deleted -- if it ever fires, that exclusion is wrong.
            graph.remove_edge(driver_node, node, key=key)
            if graph.degree(driver_node) == 0:
                graph.remove_node(driver_node)
            print(f'MDE edge accept: rejected {driver_node} -> {node} (would create a '
                  f'cycle -- UNEXPECTED, upstream observables are excluded before MDE '
                  f'runs)', flush=True)
            continue

        edges.append({'driver': driver_node, 'node': node, 'delay': delay, 'weight': gain})
        accepted.append((driver_node, delay))

    return accepted


def blocked_observables(graph, node):
    '''Observables an edge into `node` may not come from: `node` itself plus
    everything reachable FROM it.

    nx.descendants, NOT nx.ancestors. Edges run driver -> node, so accepting
    candidate C adds C -> node, which closes a cycle exactly when `node`
    already reaches C. Those are node's descendants in edge direction --
    which are the nodes between it and the behavioural targets, i.e.
    "upstream of this node" in the causal sense the network describes.

    A lagged leaf can never appear here: a leaf has no in-edges, so nothing
    reaches it.
    '''
    return {node} | (nx.descendants(graph, node) if node in graph else set())


def _real_candidate_pool(graph, node, real_columns, raw_columns):
    '''Stage 1's candidate pool for `node`: every real (delay=0) observable
    except `node` itself and anything upstream of it.

    Excluded here, before the MDE call, rather than by rejecting edges after
    selection. Selecting an upstream observable would close a cycle, so the
    edge could never be kept -- and forward selection is greedy, so a
    candidate that gets picked and then thrown away has still consumed one
    of the node's driver slots and conditioned every later round.

    Stage 2 does not use this: its picks become leaves and cannot cycle.
    '''
    blocked = blocked_observables(graph, node)
    return [c for c in real_columns
            if observable_and_delay(c, raw_columns)[0] not in blocked]


def cycle_closing_picks(result, blocked, raw_columns):
    '''Which real observables in `result` are now cycle-closing for the node.

    A wave's candidate pools are built before any of that wave's results come
    back, so two nodes in one wave can select each other and the second one's
    pool could not have known. Lagged picks never appear here -- they become
    leaves.
    '''
    return sorted({
        observable
        for observable, delay in (observable_and_delay(c, raw_columns)
                                  for c in result['SELECTED_COLUMNS'])
        if delay == 0 and observable in blocked
    })


# =====================================================================
# The builder
# =====================================================================

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
                             progress_log_path=None, mp_start_method='fork', use_gpu=False,
                             min_gain=None):
    '''Two-stage nested-MDE network construction, MODE='MDE-GMN' only.

    See this module's docstring for the algorithm. What this function adds on
    top of it is the schedule:

    WAVES, AND WHY DEPTH IS A MINIMUM. Nodes are processed in waves, all nodes
    of one wave in parallel. A wave can hold nodes at different depths, and two
    parents in one wave can both propose the same new driver at different
    depths. Such a driver is recorded at the SHALLOWEST proposed depth (the
    dict-of-minimums below), so its depth does not depend on iteration order.

    DEPTH is distance from the nearest target, whichever stage discovered the
    node. `nested_m` caps it: a node at depth d expands further only while
    `nested_m == 0 or d < nested_m` (0 means no cap). It is one budget shared
    by both stages, not two per-stage counters.

    WITHIN-WAVE RACE. Two nodes in one wave can select each other. Whichever
    is accepted second is recomputed serially with the other's edges in place
    -- see _recompute_for_cycles().

    Returns (graph, edges_df, per_node_mde_results).
    '''
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(targets)

    # expand_delay_columns() returns the raw columns first, then one column
    # per nonzero delay, so the tail is Stage 2's lagged-only pool.
    stage2_df, stage2_expanded = expand_delay_columns(expanded_df, raw_columns, lag_set)
    stage2_universe = stage2_expanded[len(raw_columns):]

    # One (observable, delay) resolve per Stage 2 candidate column, done once
    # here instead of once per column per node. Stage 2 asks the same
    # question of the same columns for every node it visits.
    stage2_observable = {c: observable_and_delay(c, raw_columns)[0] for c in stage2_universe}

    def mde_call_kwargs(node, inner_workers, cache_stage):
        """The keyword arguments every mde.run_mde_for_target() call shares.

        One place, so the two stages and the serial recompute path cannot
        drift apart. Everything from `tp` on is passed by keyword: this
        signature has had parameters inserted ahead of trailing ones
        before, and relying on positional order through it is a
        silent-miswiring trap.

        `target_metric` is resolved PER NODE via _metric_for(). Passing the
        run-wide TARGET_METRIC instead would score a neuron node's own
        ranking stage with 'correlation' where it needs 'f1'. The CCM
        gate's `metric` is left alone -- it always scores a candidate,
        never the node itself.
        """
        return dict(
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
            cache_stage=cache_stage, max_e=nested_n, inner_workers=inner_workers,
            raw_columns=raw_columns, use_gpu=use_gpu, min_gain=min_gain,
        )

    def mde_call_args(node, frame, candidate_pool, stage_delay_set):
        """The positional arguments, including the effective D for this pool."""
        return (node, frame, candidate_pool, prediction_start, prediction_length,
                effective_mde_dimension(nested_n, candidate_pool),
                embed_dim_rho_min, cross_map_rho_min, exclusion_radius,
                results_dir, stage_delay_set)

    def _recompute_for_cycles(node, result):
        """`result`, or a fresh serial Stage 1 MDE if it picked an observable
        that has become cycle-closing for `node`.

        Rejecting the offending edge afterwards is not equivalent: forward
        selection is greedy, so the discarded pick has already consumed a
        driver slot and conditioned every later round.
        """
        conflicts = cycle_closing_picks(
            result, blocked_observables(graph, node), raw_columns)
        if not conflicts:
            return result
        print(f'nested MDE (Stage 1): recomputing {node} -- its selection includes '
              f'{conflicts}, which would close a cycle now that the rest of this wave '
              f'has been accepted', flush=True)
        _log_progress(progress_log_path,
                      f'stage1 {node} RECOMPUTE (cycle-closing {conflicts})')
        pool = _real_candidate_pool(graph, node, expanded_columns, raw_columns)
        try:
            return mde.run_mde_for_target(
                *mde_call_args(node, expanded_df, pool, delay_set),
                # Serial -- it has to see the graph as accepted so far -- so it
                # gets every core as inner threads.
                **mde_call_kwargs(node, max(1, cores), 'stage1'))
        except Exception as exc:
            if mde.is_gpu_memory_failure(exc):
                print(f'nested MDE (Stage 1): {node} recompute hit a GPU MEMORY failure: '
                      f'{exc!r} -- STOPPING the run. {mde.GPU_MEMORY_STOP_ADVICE}',
                      flush=True)
                raise
            print(f'nested MDE (Stage 1): {node} recompute FAILED: {exc!r} -- treating '
                  f'as a dead end', flush=True)
            return mde.failed_mde_result(node)

    def stage_submitter(frame, pool_for, stage_delay_set, cache_stage, window_estimate):
        """A submit_one() callback for waves.run_mde_batch()."""
        def submit_one(executor, node, inner_workers):
            # Speculative window resolve, for the heartbeat's remaining-time
            # estimate only -- see _preview_window_rows(). Independent of,
            # and without effect on, the real resolve the worker performs.
            window_estimate[node] = _preview_window_rows(
                node, frame, stage_delay_set, prediction_start, prediction_length, tp,
                node in targets, target_ranking_window_rows, min_pred_events,
                max_window_rows)
            return executor.submit(
                mde.run_mde_for_target,
                *mde_call_args(node, frame, pool_for(node), stage_delay_set),
                **mde_call_kwargs(node, inner_workers, cache_stage))
        return submit_one

    edges = []
    nested_mde_results = {}
    node_depth = {t: 0 for t in targets}

    # =================================================================
    # STAGE 1 -- real-observable BFS, run to the full depth nested_m allows
    # before Stage 2 starts at all.
    # =================================================================
    visited1 = set()
    current_wave = [(t, 0) for t in targets]
    stage1_start = time.perf_counter()
    _log_progress(progress_log_path, 'STAGE 1 START (real-observable BFS)')

    while current_wave:
        wave_nodes = _dedupe_wave(current_wave, visited1)
        if not wave_nodes:
            break

        # A depth-0 target whose MDE main() already ran is reused, not
        # recomputed. Targets are never candidates, so a target can never be
        # rediscovered at depth > 0 and this reuse can never shadow a
        # genuinely different call.
        to_compute = [n for n, depth in wave_nodes
                      if not (depth == 0 and n in mde_results)]
        stage1_results = {n: mde_results[n] for n, depth in wave_nodes
                          if depth == 0 and n in mde_results}

        if to_compute:
            window_estimate = {}
            wave_depths = sorted({d for n, d in wave_nodes if n in set(to_compute)})
            stage1_results.update(waves.run_mde_batch(
                to_compute,
                stage_submitter(
                    expanded_df,
                    lambda n: _real_candidate_pool(graph, n, expanded_columns, raw_columns),
                    delay_set, 'stage1', window_estimate),
                cores, mp_start_method, progress_log_path,
                f'nested MDE (Stage 1) wave at depth(s) {wave_depths}',
                window_estimate=window_estimate,
            ))

        next_wave = {}
        for node, depth in wave_nodes:
            visited1.add(node)
            node_depth[node] = depth
            result = _recompute_for_cycles(node, stage1_results[node])
            nested_mde_results[node] = result
            for driver, delay in _accept_mde_result_edges(graph, edges, node, result,
                                                          raw_columns):
                if delay == 0 and (nested_m == 0 or depth < nested_m) and driver not in visited1:
                    next_wave[driver] = min(next_wave.get(driver, depth + 1), depth + 1)
        current_wave = list(next_wave.items())

    _log_progress(progress_log_path,
                  f'STAGE 1 END -- {len(visited1)} node(s), '
                  f'{time.perf_counter() - stage1_start:.1f}s total')
    print(f'nested MDE (Stage 1): completed -- {len(visited1)} node(s) in the '
          f'real-observable backbone, Stage 2 starts now', flush=True)

    # =================================================================
    # STAGE 2 -- attach lagged drivers. ONE pass over every node Stage 1
    # discovered. Every pick is a leaf, so there is nothing to recurse into
    # and no depth budget to apply: nested_m governs Stage 1 alone.
    # =================================================================
    # ZERO LAGS. DELAY_SOURCE='explicit' with DELAY_SET=0 resolves to an EMPTY
    # lag set (derive_lag_set() strips every 0 entry on that path), leaving
    # stage2_universe empty: nothing to select and nothing to attach, so the
    # stage is skipped outright.
    if not lag_set:
        print('nested MDE (Stage 2): SKIPPED ENTIRELY -- the lag set is empty (zero lags), '
              'so there are no lagged candidates to select or attach; this network is '
              'Stage 1 only', flush=True)
        _log_progress(progress_log_path, 'STAGE 2 SKIPPED -- empty lag_set (zero lags)')
    else:
        stage2_start = time.perf_counter()
        _log_progress(progress_log_path, 'STAGE 2 START (lagged-driver attachment)')
        stage2_nodes = sorted(visited1, key=lambda n: (node_depth[n], n))
        print(f'nested MDE (Stage 2): {len(stage2_nodes)} node(s), candidate pool size '
              f'{len(stage2_universe)} (lagged columns only, lag_set={lag_set})', flush=True)

        window_estimate = {}
        stage2_results = waves.run_mde_batch(
            stage2_nodes,
            stage_submitter(
                stage2_df,
                lambda n: [c for c in stage2_universe if stage2_observable[c] != n],
                lag_set, 'stage2', window_estimate),
            cores, mp_start_method, progress_log_path,
            'nested MDE (Stage 2)', window_estimate=window_estimate,
        )

        for node in stage2_nodes:
            # Every accepted pick is a lagged leaf, so nothing is enqueued.
            _accept_mde_result_edges(graph, edges, node, stage2_results[node], raw_columns)

        _log_progress(progress_log_path,
                      f'STAGE 2 END -- {len(stage2_nodes)} node(s), '
                      f'{time.perf_counter() - stage2_start:.1f}s total')

    return graph, _finish(graph, edges, node_depth, nested_n, nested_m,
                          progress_log_path), nested_mde_results


def _dedupe_wave(current_wave, visited):
    '''This wave's (node, depth) pairs, minus anything already visited or
    repeated within the wave itself.'''
    wave_nodes, seen = [], set()
    for node, depth in current_wave:
        if node in visited or node in seen:
            continue
        seen.add(node)
        wave_nodes.append((node, depth))
    return wave_nodes


def _finish(graph, edges, node_depth, nested_n, nested_m, progress_log_path):
    '''The edge table, plus the one-line summary both builders print.

    `realized max depth` is how deep the shared nested_m budget actually got
    used this run, which is not the same thing as the configured cap.
    '''
    edges_df = pd.DataFrame(edges, columns=['driver', 'node', 'delay', 'weight'])
    realized_max_depth = max(node_depth.values()) if node_depth else 0
    print(f'nested MDE: {len(graph.nodes)} nodes, {len(edges_df)} edges '
          f'(N={nested_n}, M={"unlimited" if nested_m == 0 else nested_m}, '
          f'realized max depth={realized_max_depth})', flush=True)
    _log_progress(progress_log_path,
                  f'NESTED MDE NETWORK COMPLETE -- {len(graph.nodes)} nodes, '
                  f'{len(edges_df)} edges')
    return edges_df


# =====================================================================
# Splitting the finished graph
# =====================================================================

def partition_network(graph, targets, external_input_mode):
    '''Split the graph into what GMN simulates and what it reads from data.

    Interior nodes have at least one discovered driver and are genuinely
    simulated, their embedding built from those in-edges. External inputs
    have none; what Generate()/Forecast() then do with them is
    EXTERNAL_INPUT_MODE's business, not this function's.

    The split is stored on the graph object itself
    (graph.graph['interior_nodes'] / ['external_inputs'] /
    ['external_input_mode']) rather than left implicit in edge counts, so
    every consumer -- embedding construction, the theta search, the figures,
    the run log -- reads one authoritative classification instead of
    re-deriving it.

    A TARGET with zero drivers is NEVER an external input, whatever the mode:
    this raises instead. That is a failed network, not a valid one. Feeding a
    target's own real values back in as its "simulation" and then scoring
    fidelity against those same values is circular, and would produce a
    spuriously perfect score for exactly the case that most needs flagging as
    a failure. (A real case, not hypothetical: head_position_y selected zero
    real drivers at depth 0 in one run.)

    Mutates `graph`; returns (interior_nodes, external_inputs) as plain sets.
    '''
    interior_nodes, external_inputs = set(), set()
    for node in graph.nodes:
        if graph.in_degree(node) > 0:
            interior_nodes.add(node)
            continue
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

    graph.graph['interior_nodes'] = interior_nodes
    graph.graph['external_inputs'] = external_inputs
    graph.graph['external_input_mode'] = external_input_mode
    return interior_nodes, external_inputs


def simulated_nodes(graph):
    '''Nodes that actually go through gmn's SMap/embedding machinery.

    Interior nodes always; external inputs too under
    EXTERNAL_INPUT_MODE='self_simulate', where they keep a self-history
    embedding. Real-data-mode external inputs are excluded: their
    Generate()/Forecast() short-circuit to a real-data passthrough with no
    embedding at all, so retuning theta for them would either waste effort
    on an inert result or crash on an empty spec.
    '''
    external_inputs = graph.graph.get('external_inputs', set())
    if graph.graph.get('external_input_mode', 'real_data') == 'self_simulate':
        return list(graph.nodes)
    return [n for n in graph.nodes if n not in external_inputs]


def report_partition(graph, network_id, results_dir):
    '''Print the partition and write it to partition_<network_id>.csv.

    Reads the classification partition_network() already stored rather than
    recomputing it.
    '''
    external_inputs = graph.graph.get('external_inputs', set())
    interior_nodes = graph.graph.get('interior_nodes', set())
    mode = graph.graph.get('external_input_mode', 'real_data')

    print(f'{network_id}: partition -- {len(interior_nodes)} interior node(s), '
          f'{len(external_inputs)} external input(s) (EXTERNAL_INPUT_MODE={mode!r})',
          flush=True)

    rows = []
    for node in sorted(external_inputs):
        out_edges = list(graph.out_edges(node, data=True))
        if not out_edges:
            print(f'  external input {node}: feeds nothing (dead-end leaf)', flush=True)
            rows.append({'node': node, 'role': 'external_input', 'feeds': '', 'delay': ''})
            continue
        for _driver, fed_node, data in out_edges:
            print(f'  external input {node} -> {fed_node} (delay {data["delay"]})', flush=True)
            rows.append({'node': node, 'role': 'external_input', 'feeds': fed_node,
                         'delay': data['delay']})
    for node in sorted(interior_nodes):
        rows.append({'node': node, 'role': 'interior', 'feeds': '', 'delay': ''})

    pd.DataFrame(rows, columns=['node', 'role', 'feeds', 'delay']).to_csv(
        results_dir / f'partition_{network_id}.csv', index=False)
