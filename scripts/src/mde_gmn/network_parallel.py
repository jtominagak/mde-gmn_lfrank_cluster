"""Building the network, but with every MDE call made up front.

The GPU branch's alternative to network.py's step 3. Same result, different
schedule -- and the schedule is the whole point.

WHY THIS EXISTS
    network.py walks the graph breadth-first: run MDE on this wave's nodes,
    look at what they picked, then decide which nodes the next wave contains.
    Each wave therefore ends in a barrier -- every node must finish before the
    next wave can even be named. With a handful of nodes per wave, most of the
    machine sits idle at every barrier, and on a GPU that is worse, because one
    node's work does not fill the device.

    So this module inverts it. Phase 0 computes MDE for EVERY node that could
    possibly be needed, all at once, with no barriers. Phase 1 then walks the
    graph exactly as network.py does, but instead of calling MDE it looks the
    answer up. More MDE calls are computed than the walk ends up consulting --
    that is the trade: wasted compute bought with wall-clock.

TWO SCOPES (config: PARALLEL_MDE_SCOPE)
    'staged'  the serial pipeline's algorithm, minus the barriers: Stage 1
              (real candidates) for the whole node universe, replay, then
              Stage 2 (lagged candidates) for exactly the nodes Stage 1
              visited. Builds the SAME network as network.py, asserted edge for
              edge by tests/test_parallel_mde_equivalence.py.
    'union'   ONE MDE per node, over real AND lagged candidates together. A
              different algorithm on purpose -- real and lagged candidates
              compete inside one forward selection -- so a different network,
              not a wrong one. One barrier for the entire build.

THE CATCH, AND THE FIX
    Phase 0 cannot know which observables will end up upstream of a node,
    because "upstream" only exists once the graph is being built. So a
    precomputed result can select a candidate that turns out to close a cycle.
    When the replay hits that, the node is RECOMPUTED serially with the
    offending candidates excluded (_mde_result_for_node, keyed by stage, node
    and the blocked set, so a recompute is itself cached). Without that, this
    module would quietly build networks the serial one never would.

Depends on: network (the shared edge-acceptance rule), mde (through the module
object), columns, tsls_data, progress; networkx, pandas.
Imported by: the parallel-MDE entry script only.
"""

import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import networkx as nx
import pandas as pd

import gpu_budget                # staged beside the entry script, not in the package
from mde_gmn import mde          # module object, so tests' fake is honoured
from mde_gmn.columns import effective_mde_dimension, expand_delay_columns, observable_and_delay
from mde_gmn.network import _accept_mde_result_edges
from mde_gmn.progress import HEARTBEAT_SECONDS, _format_eta, _log_progress
from mde_gmn.tsls_data import _metric_for, _preview_window_rows


def mde_node_universe(targets, raw_columns):
    '''Every node either stage's BFS could possibly visit, in a
    deterministic order (targets first, then raw observables in their given
    order -- `raw_columns` order is already meaningful to the pipeline, and
    forward selection can tie-break on column order).

    Provably a superset of both stages' visited sets:
      * Stage 1's wave 0 is `targets`, and it only ever enqueues drivers
        with delay == 0, i.e. real observables -- `observable_and_delay(c,
        raw_columns)[0]` is by construction an element of `raw_columns`.
      * Stage 2's wave 0 is Stage 1's visited set (a subset of this), and
        it likewise only enqueues delay == 0 real observables.
    Targets are never themselves candidates (main() passes the CCM-pre-
    filtered observable pool as both `raw_columns` and `expanded_columns`,
    and behavioural targets are not in it), so nothing outside this list can
    enter either BFS. build_nested_mde_network() below raises rather than
    silently degrading if a lookup ever misses, so a wrong universe here
    fails loudly instead of quietly changing the network.'''
    universe, seen = [], set()
    for node in list(targets) + list(raw_columns):
        if node not in seen:
            seen.add(node)
            universe.append(node)
    return universe


def _mde_stage_spec(cache_stage, df, candidate_universe, stage_delay_set, num_delays):
    '''The per-stage half of a precompute task: everything that differs
    between the parent pipeline's Stage 1 and Stage 2 executor.submit()
    sites but not between nodes within a stage.'''
    return {
        'cache_stage': cache_stage,
        'df': df,
        'candidate_universe': candidate_universe,
        'delay_set': stage_delay_set,
        'num_delays': num_delays,
    }


def _mde_call_arguments(spec, node, targets, raw_columns, nested_n, metric, target_metric,
                         blocked=()):
    '''The per-node half of an MDE call: candidate pool, D, and metric
    routing, matching the parent pipeline's own submit sites.

    `blocked` is the set of observables that must not be offered as
    candidates because an edge from them would close a cycle (see
    _real_candidate_pool() in the parent). Phase 0 always passes an EMPTY
    set -- nothing is upstream of anything before the network exists -- and
    the replay passes the real set when it has to recompute a node; see
    _mde_result_for_node().'''
    blocked = set(blocked)
    candidate_pool = [c for c in spec['candidate_universe']
                       if observable_and_delay(c, raw_columns)[0] != node
                       and observable_and_delay(c, raw_columns)[0] not in blocked]
    return {
        'candidate_pool': candidate_pool,
        'effective_d': effective_mde_dimension(nested_n, candidate_pool, spec['num_delays']),
        'is_behavioral_target': node in targets,
        # Metric-routing fix, same as both parent submit sites: a neuron
        # node must be scored by _metric_for()'s rule ('f1'), not by the
        # closed-over TARGET_METRIC.
        'target_metric': _metric_for(node, targets, metric, target_metric),
    }


def _run_one_mde(node, spec, call, common, inner_workers=1):
    """One serial mde.run_mde_for_target() call -- the same arguments
    precompute_all_mde() submits to its pool, in one place so the replay's
    recompute path cannot drift from the precompute path."""
    return mde.run_mde_for_target(
        node, spec['df'], call['candidate_pool'],
        common['prediction_start'], common['prediction_length'], call['effective_d'],
        common['embed_dim_rho_min'], common['cross_map_rho_min'],
        common['exclusion_radius'], common['results_dir'], spec['delay_set'],
        tp=common['tp'], metric=common['metric'],
        target_metric=call['target_metric'],
        gamma_delta=common['gamma_delta'], ccm_sample=common['ccm_sample'],
        ccm_slope=common['ccm_slope'],
        is_behavioral_target=call['is_behavioral_target'],
        target_ranking_window_rows=common['target_ranking_window_rows'],
        min_pred_events=common['min_pred_events'],
        max_window_rows=common['max_window_rows'],
        ccm_lib_sizes_rows=common['ccm_lib_sizes_rows'],
        reuse_cache=common['reuse_cache'], prev_runs_dir=common['prev_runs_dir'],
        run_cache_dir=common['run_cache_dir'], data_file=common['data_file'],
        data_file_sha256=common['data_file_sha256'],
        code_fingerprint=common['code_fingerprint'],
        cache_stage=spec['cache_stage'],
        max_e=common['nested_n'], inner_workers=inner_workers,
        raw_columns=common['raw_columns'], use_gpu=common['use_gpu'],
    )


def _mde_result_for_node(node, spec, common, blocked, precomputed, recompute_cache,
                          progress_log_path):
    """`node`'s MDE result, honouring the acyclicity exclusion.

    THE ONE PLACE THIS FORK CANNOT PRECOMPUTE EXACTLY. Phase 0 ran every
    node's MDE over the full candidate pool, because nothing is upstream of
    anything until the network is being built. If the replay then reaches a
    node whose precomputed result selected a real observable that is by now
    cycle-closing, that result is unusable: forward selection is greedy, so
    dropping the offending pick would leave every later pick conditioned on
    a driver the network cannot keep. Such a node is recomputed here,
    serially, with the blocked observables removed from its pool -- which is
    exactly what the serial pipeline would have computed for it.

    Lagged picks never trigger this: they become leaf nodes and cannot
    cycle. Results are cached on (stage, node, blocked), so a node reached
    again under the same exclusions is not recomputed twice."""
    if not blocked:
        return precomputed
    conflicts = sorted({
        observable for observable, delay in
        (observable_and_delay(c, common['raw_columns'])
         for c in precomputed['SELECTED_COLUMNS'])
        if delay == 0 and observable in blocked})
    if not conflicts:
        return precomputed

    key = (spec['cache_stage'], node, frozenset(blocked))
    if key in recompute_cache:
        return recompute_cache[key]

    print(f'parallelized MDE: {node} ({spec["cache_stage"]}) must be RECOMPUTED -- its '
          f'precomputed selection includes {conflicts}, which would now close a cycle; '
          f'rerunning with {len(blocked)} observable(s) excluded', flush=True)
    _log_progress(progress_log_path,
                  f'{spec["cache_stage"]} {node} RECOMPUTE (cycle-closing picks {conflicts})')
    call = _mde_call_arguments(spec, node, common['targets'], common['raw_columns'],
                               common['nested_n'], common['metric'], common['target_metric'],
                               blocked=blocked)
    started = time.perf_counter()
    try:
        result = _run_one_mde(node, spec, call, common)
    except Exception as exc:   # same policy as the precompute pool's own handler
        if mde.is_gpu_memory_failure(exc):
            print(f'parallelized MDE: {node} recompute hit a GPU MEMORY failure: {exc!r} '
                  f'-- STOPPING the run. {mde._GPU_MEMORY_STOP_ADVICE}', flush=True)
            raise
        print(f'parallelized MDE: {node} recompute FAILED: {exc!r} -- treating as a dead end',
              flush=True)
        result = mde._failed_mde_result(node)
    _log_progress(progress_log_path,
                  f'{spec["cache_stage"]} {node} RECOMPUTE END -- '
                  f'{time.perf_counter() - started:.1f}s')
    recompute_cache[key] = result
    return result


def _blocked_observables(graph, node):
    """Observables an edge into `node` may not come from: `node` itself plus
    everything reachable FROM it (edges run driver -> node, so those are the
    nodes between it and the targets). Mirrors the parent pipeline's
    _real_candidate_pool(); lagged leaves can never appear here, having no
    in-edges."""
    return {node} | (nx.descendants(graph, node) if node in graph else set())


def precompute_all_mde(tasks, stage_specs, common, cores, mp_start_method,
                        progress_log_path, phase_label):
    '''Run mde.run_mde_for_target() for every (cache_stage, node) in `tasks` in
    ONE pool with no barrier inside it, and return
    {cache_stage: {node: result}}.

    `tasks` may mix stages freely -- a Stage 1 call and a Stage 2 call for
    the same node are independent, so PARALLEL_MDE_SCOPE='union' hands both
    stages' entire task lists to a single pool and the whole network build
    contains exactly one barrier.

    Submission is bounded to `max_inflight` outstanding futures rather than
    submitting all of `tasks` at once: ProcessPoolExecutor pickles a call's
    arguments at submit() time and holds them queued, and every task here
    carries its stage's whole DataFrame, so an unbounded submit of a
    production-sized universe would queue |tasks| copies of it. A new task
    is submitted the instant one completes, so this bounds memory without
    reintroducing a barrier (it is a sliding window, not a wave).

    Worker sizing follows the parent pipeline's own convention exactly --
    outer_workers = min(task count, cores), inner_workers = cores //
    outer_workers, and the product is logged so the "never oversubscribe the
    cores we were given" invariant can be checked from the log. Because this
    phase's task count is the whole node universe rather than one BFS
    wave, outer_workers is normally pinned at `cores` and inner_workers at
    1: the parallelism has moved from inside each node (the inner
    ThreadPoolExecutor) to across nodes, which is the point.

    A failing node is recorded as mde._failed_mde_result() exactly as the
    parent's per-wave handlers do -- one node's MDE blowing up must not
    take the other several thousand queued nodes with it.'''
    tables = {name: {} for name in stage_specs}
    if not tasks:
        return tables

    outer_workers = min(len(tasks), max(1, cores))
    inner_workers = max(1, cores // outer_workers)
    max_inflight = min(len(tasks), max(outer_workers * 2, outer_workers))

    print(f'parallel MDE precompute [{phase_label}]: {len(tasks)} MDE call(s) '
          f'-- {outer_workers} parallel worker(s) x {inner_workers} inner thread(s) each '
          f'(product={outer_workers * inner_workers}, cores={cores}), '
          f'max {max_inflight} in flight, NO barrier inside this phase', flush=True)
    _log_progress(progress_log_path,
                  f'{phase_label} START -- {len(tasks)} MDE call(s), {outer_workers}x{inner_workers}')

    phase_start = time.perf_counter()
    durations = []   # seconds per completed task, this run only
    rates = []       # seconds per resolved window-row -- see _format_eta()
    window_estimate = {}
    completed = 0

    # Same fork-vs-spawn rationale as the parent pipeline's per-wave pools:
    # 'spawn' whenever the torch/CUDA path is on, since CUDA cannot fork.
    # A further (fork-only) win of collapsing every wave into one pool: the
    # 10-28s spawn startup docs/no-kEDM-gpu-optimization-progress.md
    # measured is now paid once per network build instead of once per wave.
    #
    # GPU admission control. One budget per visible device, sized from
    # what the device reports rather than a hardcoded card size (the job
    # goes to whichever of gpu-p100/gpu-v100 starts sooner, and those need
    # not match). The budgets and the round-robin device counter are built
    # HERE, in the parent, because multiprocessing primitives can only
    # reach a spawned child by being passed at process construction --
    # which is exactly what ProcessPoolExecutor's initargs are. Without
    # this every worker would launch its GPU work blind, which is how job
    # 4741044 lost 51 candidates. Empty list on a GPU-less host, and then
    # gpu_budget.reserve() is a transparent no-op.
    ctx = mp.get_context(mp_start_method)
    gpu_budgets = gpu_budget.create_budgets(ctx, workers=outer_workers)
    device_counter = ctx.Value('i', 0)
    if gpu_budgets:
        print(gpu_budget.env_summary(), flush=True)

    with ProcessPoolExecutor(max_workers=outer_workers,
                              mp_context=ctx,
                              initializer=gpu_budget.init_worker,
                              initargs=(gpu_budgets, device_counter)) as executor:
        pending = {}
        next_task = 0

        def _submit_next():
            nonlocal next_task
            cache_stage, node = tasks[next_task]
            next_task += 1
            spec = stage_specs[cache_stage]
            call = _mde_call_arguments(spec, node, common['targets'], common['raw_columns'],
                                        common['nested_n'], common['metric'],
                                        common['target_metric'])
            # Monitoring-log ETA input only -- see _preview_window_rows()'s
            # own docstring for why this speculative resolve is safe and
            # independent of the real one mde.run_mde_for_target() performs.
            window_estimate[(cache_stage, node)] = _preview_window_rows(
                node, spec['df'], spec['delay_set'], common['prediction_start'],
                common['prediction_length'], common['tp'], call['is_behavioral_target'],
                common['target_ranking_window_rows'], common['min_pred_events'],
                common['max_window_rows'],
            )
            # All arguments from `tp` onward passed by keyword -- same
            # silent-miswiring trap the parent pipeline's own submit sites
            # document: this signature has parameters inserted ahead of
            # several trailing ones.
            fut = executor.submit(
                mde.run_mde_for_target, node, spec['df'], call['candidate_pool'],
                common['prediction_start'], common['prediction_length'], call['effective_d'],
                common['embed_dim_rho_min'], common['cross_map_rho_min'],
                common['exclusion_radius'], common['results_dir'], spec['delay_set'],
                tp=common['tp'], metric=common['metric'],
                target_metric=call['target_metric'],
                gamma_delta=common['gamma_delta'], ccm_sample=common['ccm_sample'],
                ccm_slope=common['ccm_slope'],
                is_behavioral_target=call['is_behavioral_target'],
                target_ranking_window_rows=common['target_ranking_window_rows'],
                min_pred_events=common['min_pred_events'],
                max_window_rows=common['max_window_rows'],
                ccm_lib_sizes_rows=common['ccm_lib_sizes_rows'],
                reuse_cache=common['reuse_cache'], prev_runs_dir=common['prev_runs_dir'],
                run_cache_dir=common['run_cache_dir'], data_file=common['data_file'],
                data_file_sha256=common['data_file_sha256'],
                code_fingerprint=common['code_fingerprint'],
                cache_stage=cache_stage,
                max_e=common['nested_n'], inner_workers=inner_workers,
                raw_columns=common['raw_columns'], use_gpu=common['use_gpu'],
            )
            pending[fut] = (cache_stage, node)

        while next_task < len(tasks) and len(pending) < max_inflight:
            _submit_next()

        while pending:
            done, _ = wait(set(pending), timeout=HEARTBEAT_SECONDS,
                            return_when=FIRST_COMPLETED)
            if not done:
                # Heartbeat -- see the parent pipeline's own HEARTBEAT_SECONDS
                # docstring: a long phase with no heartbeat can go silent for
                # hours with no way to tell "still working" from "stuck".
                elapsed = time.perf_counter() - phase_start
                still_waiting = sorted(f'{cs}:{n}' for cs, n in pending.values())
                remaining_rows = sum(window_estimate.get(v, 0) for v in pending.values())
                eta = _format_eta(durations, len(tasks) - completed,
                                   per_unit=rates, unit_remaining=remaining_rows)
                _log_progress(
                    progress_log_path,
                    f'{phase_label} HEARTBEAT -- {completed}/{len(tasks)} done, '
                    f'{len(pending)} in flight, {len(tasks) - next_task} not yet submitted, '
                    f'elapsed={elapsed:.0f}s, still waiting on: {still_waiting}{eta}')
                continue

            for fut in done:
                cache_stage, node = pending.pop(fut)
                # Elapsed-since-phase-start, matching the parent's own
                # per-wave duration semantics (ETA input only, not a true
                # per-node runtime).
                node_duration = time.perf_counter() - phase_start
                try:
                    tables[cache_stage][node] = fut.result()
                    _log_progress(
                        progress_log_path,
                        f'{phase_label} node END -- {cache_stage} {node} '
                        f'duration={node_duration:.1f}s '
                        f'selected={tables[cache_stage][node]["SELECTED_COLUMNS"]}')
                except Exception as exc:
                    # A GPU-MEMORY failure is not a finding and must never be
                    # recorded as one. gpu_budget re-raises OOM for exactly that
                    # reason, but this clause used to catch it anyway and write a
                    # dead end regardless -- job 4744295 printed "the run is
                    # stopping rather than recording it as a dead end" 20 times
                    # and then recorded a dead end 20 times. A candidate dropped
                    # for want of memory is indistinguishable from one measured
                    # and found absent, which is the whole failure mode this
                    # machinery exists to remove, so stop the run instead.
                    if mde.is_gpu_memory_failure(exc):
                        print(f'parallel MDE precompute [{phase_label}]: {cache_stage} {node} MDE '
                              f'call hit a GPU MEMORY failure: {exc!r} -- STOPPING the run. '
                              f'{mde._GPU_MEMORY_STOP_ADVICE}', flush=True)
                        _log_progress(
                            progress_log_path,
                            f'{phase_label} node END -- {cache_stage} {node} '
                            f'duration={node_duration:.1f}s GPU MEMORY FAILURE: {exc!r}')
                        raise
                    # See mde._failed_mde_result()'s own docstring -- one node's
                    # MDE call failing must not crash the whole precompute.
                    print(f'parallel MDE precompute [{phase_label}]: {cache_stage} {node} MDE call '
                          f'FAILED: {exc!r} -- treating as a dead-end (zero drivers selected, no '
                          f'further nesting from this node)', flush=True)
                    _log_progress(
                        progress_log_path,
                        f'{phase_label} node END -- {cache_stage} {node} '
                        f'duration={node_duration:.1f}s FAILED: {exc!r}')
                    tables[cache_stage][node] = mde._failed_mde_result(node)
                completed += 1
                window_est = window_estimate.get((cache_stage, node), 0)
                durations.append(node_duration)
                if window_est > 0:
                    rates.append(node_duration / window_est)

            while next_task < len(tasks) and len(pending) < max_inflight:
                _submit_next()

    _log_progress(progress_log_path,
                  f'{phase_label} END -- {len(tasks)} MDE call(s), '
                  f'{time.perf_counter() - phase_start:.1f}s total')
    print(f'parallel MDE precompute [{phase_label}]: completed {len(tasks)} MDE call(s) in '
          f'{time.perf_counter() - phase_start:.1f}s -- network construction is serial from here',
          flush=True)
    return tables


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
                              parallel_mde_scope='union'):
    '''Two-stage nested-MDE network construction, MODE='MDE-GMN' only --
    PARALLELIZED-MDE fork of tsls-mde-gmn.py's function of the same name.
    Same signature plus `parallel_mde_scope`, same return value, and (this
    is the contract, asserted by tests/test_parallel_mde_equivalence.py)
    the SAME NETWORK for the same inputs.

    Structure, versus the parent's alternating compute/assemble waves:

        PHASE 0  precompute_all_mde() -- every MDE call for the whole node
                 universe, both stages, in one pool with no barrier inside
                 it. This is ~all of the runtime.
        PHASE 1  Stage 1's BFS, verbatim from the parent, with its ~90-line
                 executor block replaced by a dict lookup. Milliseconds.
        PHASE 2  Stage 2's BFS, likewise verbatim, likewise a lookup.

    WHY THE BFS BODIES ARE COPIED RATHER THAN REWRITTEN. Everything from
    `next_wave = {}` onward in both phases below is unchanged from the
    parent: same wave construction and de-duplication, same iteration order
    over `wave_nodes`, same _accept_mde_result_edges() call sequence, same
    dict-of-mins shallowest-depth rule, same shared nested_m budget
    arithmetic (`nested_m == 0 or depth < nested_m`, depth = the
    discovering node's OWN depth), same termination counters, same joint
    `node_depth` across both phases. That is load-bearing, not laziness: _accept_mde_result_edges()'s cycle guard rejects an edge
    against the graph AS BUILT SO FAR, so if node C selects F and node F
    selects C, which edge survives depends on which node the loop reaches
    first. Copying the loop preserves that tie-break by construction. The
    same goes for the NaN-marginal-gain rejection and the self-loop guard,
    both of which run in the serial phase here exactly as they did there.

    WHAT "PRUNING" MEANS HERE. Nothing is scored and dropped. A node the
    BFS never reaches simply never has its Phase 0 table entry read, so
    _accept_mde_result_edges() is never called for it and no edge into it
    ever enters the graph; its result is discarded when this function
    returns. Both the budget-bound case (a driver past the nested_m depth
    cap) and the unreachable case (an observable nothing ever selects)
    are handled by the identical BFS code that handled them in the parent.

    STAGE ORDER, AND WHY test_r12_stage_serialization.py IS NOT POINTED HERE.
    The serial pipeline guarantees Stage 1 finishes before Stage 2 begins. That
    still holds: the Stage 1 replay completes before the Stage 2 replay starts,
    so no Stage 2 node can be reached before Stage 1 finished discovering it.

    What scope='union' does not preserve is the order of the MDE CALLS. They
    all go into one pool and interleave freely. That order was a CONSEQUENCE of
    the serial pipeline's per-wave barriers, never an input to the algorithm: no
    MDE call reads the graph, so a Stage 2 call cannot observe anything a Stage
    1 call produced. test_r12_stage_serialization.py asserts the call order as a
    proxy for the stage order -- exact in the serial pipeline, so it stays
    pointed there. tests/test_parallel_mde_equivalence.py asserts the property
    that actually matters here instead: the same network, edges in the same
    order. scope='staged' does preserve the call order as well.

    `parallel_mde_scope`:
      'union'  (default) -- one pool, both stages, whole universe. Exactly
                one barrier in the entire network build; maximum GPU/core
                utilisation; largest overshoot (Stage 2 is computed for
                every observable, including ones Stage 1's backbone never
                reaches).
      'staged' -- two pools: Stage 1 over the universe, then the Stage 1
                replay, then Stage 2 over only _stage2_precompute_nodes().
                One extra barrier, but when Stage 2 cannot recurse (see
                that helper) its task list shrinks to exactly `visited1`,
                which is the parent's own Stage 2 workload with none of the
                parent's per-wave barriers. Use when Stage 2's overshoot
                is what you cannot afford -- its candidate pool is
                len(lag_set)x wider than Stage 1's, so its per-node calls
                are the expensive ones.'''
    if parallel_mde_scope not in ('union', 'staged'):
        raise ValueError(f'parallel_mde_scope must be "union" or "staged", got '
                          f'{parallel_mde_scope!r}')

    graph = nx.MultiDiGraph()
    graph.add_nodes_from(targets)

    stage1_num_delays = len(expanded_columns) // len(raw_columns)

    # expand_delay_columns() returns the raw columns first, then one column
    # per nonzero delay, so the tail is the lagged-only pool Stage 2 uses and
    # the whole thing is the combined pool scope='union' uses.
    stage2_df, stage2_expanded_columns = expand_delay_columns(expanded_df, raw_columns, lag_set)
    stage2_candidate_universe = stage2_expanded_columns[len(raw_columns):]
    stage2_num_delays = len(lag_set)
    union_candidate_universe = stage2_expanded_columns
    union_num_delays = len(lag_set) + 1

    edges = []
    nested_mde_results = {}
    node_depth = {t: 0 for t in targets}

    # =====================================================================
    # PHASE 0 -- every MDE call for the whole node universe, one pool,
    # no barrier. See precompute_all_mde() and this module's docstring.
    # =====================================================================
    stage_specs = {
        'stage1': _mde_stage_spec('stage1', expanded_df, expanded_columns,
                                   delay_set, stage1_num_delays),
        'stage2': _mde_stage_spec('stage2', stage2_df, stage2_candidate_universe,
                                   lag_set, stage2_num_delays),
        # scope='union' only: real and lagged candidates in ONE pool, so each
        # node gets a single MDE call in which they compete directly.
        'union': _mde_stage_spec('union', stage2_df, union_candidate_universe,
                                  lag_set, union_num_delays),
    }
    common = {
        'targets': targets, 'raw_columns': raw_columns, 'nested_n': nested_n,
        'metric': metric, 'target_metric': target_metric, 'gamma_delta': gamma_delta,
        'prediction_start': prediction_start, 'prediction_length': prediction_length,
        'embed_dim_rho_min': embed_dim_rho_min, 'cross_map_rho_min': cross_map_rho_min,
        'exclusion_radius': exclusion_radius, 'results_dir': results_dir, 'tp': tp,
        'ccm_sample': ccm_sample, 'ccm_slope': ccm_slope,
        'target_ranking_window_rows': target_ranking_window_rows,
        'min_pred_events': min_pred_events, 'max_window_rows': max_window_rows,
        'ccm_lib_sizes_rows': ccm_lib_sizes_rows,
        'reuse_cache': reuse_cache, 'prev_runs_dir': prev_runs_dir,
        'run_cache_dir': run_cache_dir, 'data_file': data_file,
        'data_file_sha256': data_file_sha256, 'code_fingerprint': code_fingerprint,
        'use_gpu': use_gpu,
    }

    universe = mde_node_universe(targets, raw_columns)
    # Depth-0 targets whose MDE main() already ran are reused, not recomputed
    # -- the parent's own `to_compute` filter, unchanged. Targets are never
    # candidates, so a target can never be rediscovered at depth > 0 and this
    # reuse can never shadow a genuinely different call.
    if parallel_mde_scope == 'union':
        # ONE combined MDE per node over real + lagged candidates. main()'s
        # depth-0 result is NOT reusable here: it ranked real observables
        # only, against a different (narrower) pool.
        precompute_tasks = [('union', n) for n in universe]
        phase_label = 'phase0 union'
    else:
        # Stage 1 for the whole universe now; Stage 2's task list is exactly
        # Stage 1's visited set, which is only known after the replay below.
        precompute_tasks = [('stage1', n) for n in universe
                            if not (n in targets and n in mde_results)]
        phase_label = 'phase0a stage1'

    print(f'nested MDE (parallelized): node universe = {len(universe)} node(s) '
          f'({len(targets)} target(s) + {len(universe) - len(targets)} observable(s)), '
          f'scope={parallel_mde_scope}', flush=True)
    _log_progress(progress_log_path,
                  f'PHASE 0 START (parallelized MDE precompute, scope={parallel_mde_scope}) -- '
                  f'universe={len(universe)}')

    mde_tables = precompute_all_mde(precompute_tasks, stage_specs, common, cores,
                                     mp_start_method, progress_log_path, phase_label)
    if parallel_mde_scope == 'staged':
        for t in targets:
            if t in mde_results:
                mde_tables['stage1'][t] = mde_results[t]
    # Results recomputed by the replay because a precomputed pick turned out
    # to be cycle-closing -- see _mde_result_for_node().
    recompute_cache = {}

    def _lookup(cache_stage, node):
        '''Phase 0 result for one node. A miss means mde_node_universe() is
        wrong (it is provably a superset of both BFS visited sets) -- fail
        loudly rather than substituting a dead end, which would silently
        change the network.'''
        try:
            return mde_tables[cache_stage][node]
        except KeyError:
            raise RuntimeError(
                f'parallelized MDE: no {cache_stage} result precomputed for node {node!r} -- '
                f'mde_node_universe() missed a node the BFS reached. This is a bug in the '
                f'universe construction, not a recoverable condition.') from None

    # =====================================================================
    # PHASE 1 -- Stage 1: real-observable-only BFS. Body unchanged from
    # tsls-mde-gmn.py except that the per-wave ProcessPoolExecutor block is
    # replaced by _lookup(). Serial and effectively instantaneous.
    # =====================================================================
    visited1 = set()
    current_wave = [(t, 0) for t in targets]
    stage1_start = time.perf_counter()
    _log_progress(progress_log_path, 'STAGE 1 REPLAY START (real-observable BFS, no compute)')

    while current_wave:
        wave_nodes, seen = [], set()
        for node, depth in current_wave:
            if node in visited1 or node in seen:
                continue
            seen.add(node)
            wave_nodes.append((node, depth))

        if not wave_nodes:
            break

        print(f'nested MDE (Stage 1 replay): wave of {len(wave_nodes)} node(s) at depth '
              f'{wave_nodes[0][1]} -- serial, results from phase 0', flush=True)
        _log_progress(progress_log_path,
                      f'stage1 replay wave -- {len(wave_nodes)} node(s) at depth '
                      f'{wave_nodes[0][1]}: {[n for n, _ in wave_nodes]}')

        replay_stage = 'union' if parallel_mde_scope == 'union' else 'stage1'

        next_wave = {}
        for node, depth in wave_nodes:
            visited1.add(node)
            node_depth[node] = depth
            # The graph already holds every edge accepted so far, so the
            # blocked set is known here even though Phase 0 could not know
            # it. A precomputed result that respects it is used as-is; one
            # that does not forces a serial recompute for this node only.
            result1 = _mde_result_for_node(
                node, stage_specs[replay_stage], common,
                _blocked_observables(graph, node), _lookup(replay_stage, node),
                recompute_cache, progress_log_path)
            nested_mde_results[node] = result1

            accepted1 = _accept_mde_result_edges(graph, edges, node, result1, raw_columns)
            for driver, delay in accepted1:
                if delay == 0 and (nested_m == 0 or depth < nested_m) and driver not in visited1:
                    next_wave[driver] = min(next_wave.get(driver, depth + 1), depth + 1)

        current_wave = list(next_wave.items())

    _log_progress(progress_log_path,
                  f'STAGE 1 REPLAY END -- {len(visited1)} node(s), '
                  f'{time.perf_counter() - stage1_start:.1f}s total')
    print(f'nested MDE (Stage 1): completed -- {len(visited1)} node(s) in the real-observable '
          f'backbone, Stage 2 starts now', flush=True)

    # ---- scope='staged': Stage 2's own precompute, now that visited1 is known
    if parallel_mde_scope == 'staged':
        # Stage 2 attaches lagged leaves and never recurses, so its node set
        # is EXACTLY Stage 1's visited set -- no overshoot at all, which is
        # the whole point of this scope.
        stage2_nodes = [n for n in universe if n in visited1]
        print(f'nested MDE (parallelized): scope=staged -- precomputing Stage 2 for '
              f'{len(stage2_nodes)} node(s) of {len(universe)}', flush=True)
        stage2_tables = precompute_all_mde(
            [('stage2', n) for n in stage2_nodes], stage_specs, common, cores,
            mp_start_method, progress_log_path, 'phase0b stage2')
        mde_tables['stage2'].update(stage2_tables['stage2'])

    # =====================================================================
    # PHASE 2 -- Stage 2: attach lagged leaf drivers. ONE pass over the nodes
    # Stage 1 discovered, no recursion (every pick is a leaf), so no depth
    # budget applies. Skipped entirely under scope='union', where each node's
    # single MDE call already ranked lagged candidates alongside real ones.
    # =====================================================================
    visited2 = set()
    if parallel_mde_scope == 'staged':
        stage2_start = time.perf_counter()
        _log_progress(progress_log_path, 'STAGE 2 REPLAY START (lagged drivers, no compute)')
        wave_nodes = [(node, node_depth[node]) for node in visited1]
        print(f'nested MDE (Stage 2 replay): {len(wave_nodes)} node(s) -- serial, results from '
              f'phase 0, candidate pool size {len(stage2_candidate_universe)} '
              f'(lagged columns only, lag_set={lag_set})', flush=True)
        _log_progress(progress_log_path,
                      f'stage2 replay -- {len(wave_nodes)} node(s): {[n for n, _ in wave_nodes]}')

        for node, depth in wave_nodes:
            visited2.add(node)
            node_depth.setdefault(node, depth)
            # No exclusion needed: lagged picks become leaves, which cannot
            # close a cycle, so the precomputed result is always usable.
            _accept_mde_result_edges(graph, edges, node, _lookup('stage2', node), raw_columns)

        _log_progress(progress_log_path,
                      f'STAGE 2 REPLAY END -- {len(visited2)} node(s), '
                      f'{time.perf_counter() - stage2_start:.1f}s total')

    edges_df = pd.DataFrame(edges, columns=['driver', 'node', 'delay', 'weight'])

    # nested_m is a whole-network depth budget shared by both stages
    # (node_depth is one dict populated by whichever stage discovers a node
    # first, checked against the same nested_m in both phases' own recursion
    # above) -- realized_max_depth reports how deep that joint budget
    # actually got used this run, distinct from the configured cap itself.
    realized_max_depth = max(node_depth.values()) if node_depth else 0
    nested_m_display = 'unlimited' if nested_m == 0 else nested_m
    # Overshoot accounting -- the price this fork pays for having no
    # barriers. |computed| is what Phase 0 ran; |consulted| is what the two
    # BFS replays actually read.
    computed = sum(len(t) for t in mde_tables.values())
    consulted = len(visited1) + len(visited2)   # union: visited2 is empty by design
    print(f'nested MDE: {len(graph.nodes)} nodes, {len(edges_df)} edges '
          f'(N={nested_n}, M={nested_m_display}, realized max depth={realized_max_depth})',
          flush=True)
    print(f'nested MDE (parallelized): {computed} MDE result(s) precomputed, {consulted} '
          f'consulted by the BFS ({computed - consulted} computed but never reached -- the '
          f'cost of removing the wave barriers)', flush=True)
    _log_progress(progress_log_path,
                  f'NESTED MDE NETWORK COMPLETE -- {len(graph.nodes)} nodes, {len(edges_df)} edges, '
                  f'precomputed={computed}, consulted={consulted}')

    return graph, edges_df, nested_mde_results
