"""Building the network, but with every MDE call made up front.

The alternative to network.py's step 3. Same result, different schedule --
and the schedule is the whole point.

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
              visited. Builds the SAME network as network.py, asserted edge
              for edge by tests/test_parallel_mde_equivalence.py.
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

WHY THE GRAPH WALK IS NOT SHARED WITH network.py. Everything from the wave
loop onward is deliberately the same shape as the serial builder's: same wave
de-duplication, same iteration order over a wave, same _accept_mde_result_edges()
call sequence, same shallowest-depth rule, same shared nested_m arithmetic.
That is load-bearing rather than lazy. The cycle guard rejects an edge against
the graph AS BUILT SO FAR, so if node C selects F and node F selects C, which
edge survives depends on which node the loop reaches first. Keeping the walks
textually parallel is what makes that tie-break checkable by reading them side
by side; the executor plumbing they used to ALSO duplicate is what moved out,
into waves.py.

Depends on: network (the shared edge-acceptance rule and graph helpers), mde
(through the module object), waves, columns, tsls_data, progress; networkx.
Imported by: the parallel-MDE entry script only.
"""

import multiprocessing as mp
import time

import networkx as nx

import gpu_budget                # staged beside the entry script, not in the package
from mde_gmn import mde          # module object, so tests' fake is honoured
from mde_gmn import waves
from mde_gmn.columns import effective_mde_dimension, expand_delay_columns, observable_and_delay
from mde_gmn.network import (
    _accept_mde_result_edges, _dedupe_wave, _finish, blocked_observables,
    cycle_closing_picks)
from mde_gmn.progress import _log_progress
from mde_gmn.tsls_data import _metric_for, _preview_window_rows


def mde_node_universe(targets, raw_columns):
    '''Every node either stage's BFS could possibly visit, in a deterministic
    order (targets first, then raw observables in their given order --
    `raw_columns` order is already meaningful, and forward selection can
    tie-break on column order).

    Provably a superset of both stages' visited sets:
      * Stage 1's wave 0 is `targets`, and it only ever enqueues drivers with
        delay == 0, i.e. real observables, which are by construction elements
        of `raw_columns`.
      * Stage 2's wave 0 is Stage 1's visited set (a subset of this), and it
        likewise only enqueues real observables.
    Targets are never themselves candidates, so nothing outside this list can
    enter either BFS. The lookup below raises rather than silently degrading
    if it ever misses, so a wrong universe fails loudly instead of quietly
    changing the network.
    '''
    universe, seen = [], set()
    for node in list(targets) + list(raw_columns):
        if node not in seen:
            seen.add(node)
            universe.append(node)
    return universe


def _stage_spec(cache_stage, df, candidate_universe, stage_delay_set, num_delays,
                raw_columns):
    '''Everything that differs between stages but not between nodes.

    `observable_of` resolves each candidate column to its underlying real
    observable once per stage, rather than once per column per node -- every
    node asks the same question of the same columns.
    '''
    return {
        'cache_stage': cache_stage,
        'df': df,
        'candidate_universe': candidate_universe,
        'delay_set': stage_delay_set,
        'num_delays': num_delays,
        'observable_of': {c: observable_and_delay(c, raw_columns)[0]
                          for c in candidate_universe},
    }


def _mde_call(spec, node, common, blocked=(), inner_workers=1):
    '''(positional args, keyword args) for one mde.run_mde_for_target() call.

    `blocked` is the set of observables that must not be offered because an
    edge from them would close a cycle (see the serial builder's
    _real_candidate_pool()). Phase 0 always passes an empty set -- nothing is
    upstream of anything before the network exists -- and the replay passes
    the real set when it has to recompute a node.
    '''
    blocked = set(blocked)
    observable_of = spec['observable_of']
    candidate_pool = [c for c in spec['candidate_universe']
                      if observable_of[c] != node and observable_of[c] not in blocked]
    args = (node, spec['df'], candidate_pool,
            common['prediction_start'], common['prediction_length'],
            effective_mde_dimension(common['nested_n'], candidate_pool, spec['num_delays']),
            common['embed_dim_rho_min'], common['cross_map_rho_min'],
            common['exclusion_radius'], common['results_dir'], spec['delay_set'])
    kwargs = dict(
        tp=common['tp'], metric=common['metric'],
        # Resolved per node -- a neuron node needs 'f1', not the run-wide
        # TARGET_METRIC. Same rule as both serial submit sites.
        target_metric=_metric_for(node, common['targets'], common['metric'],
                                  common['target_metric']),
        gamma_delta=common['gamma_delta'], ccm_sample=common['ccm_sample'],
        ccm_slope=common['ccm_slope'],
        is_behavioral_target=node in common['targets'],
        target_ranking_window_rows=common['target_ranking_window_rows'],
        min_pred_events=common['min_pred_events'],
        max_window_rows=common['max_window_rows'],
        ccm_lib_sizes_rows=common['ccm_lib_sizes_rows'],
        reuse_cache=common['reuse_cache'], prev_runs_dir=common['prev_runs_dir'],
        run_cache_dir=common['run_cache_dir'], data_file=common['data_file'],
        data_file_sha256=common['data_file_sha256'],
        code_fingerprint=common['code_fingerprint'],
        cache_stage=spec['cache_stage'], max_e=common['nested_n'],
        inner_workers=inner_workers, raw_columns=common['raw_columns'],
        use_gpu=common['use_gpu'],
    )
    return args, kwargs


def _mde_result_for_node(node, spec, common, blocked, precomputed, recompute_cache,
                         progress_log_path):
    """`node`'s MDE result, honouring the acyclicity exclusion.

    THE ONE PLACE THIS BUILDER CANNOT PRECOMPUTE EXACTLY. Phase 0 ran every
    node's MDE over the full candidate pool, because nothing is upstream of
    anything until the network is being built. If the replay then reaches a
    node whose precomputed result selected a real observable that is by now
    cycle-closing, that result is unusable: forward selection is greedy, so
    dropping the offending pick would leave every later pick conditioned on a
    driver the network cannot keep. Such a node is recomputed here, serially,
    with the blocked observables removed from its pool -- which is exactly
    what the serial pipeline would have computed for it.

    Lagged picks never trigger this: they become leaf nodes and cannot cycle.
    Results are cached on (stage, node, blocked), so a node reached again
    under the same exclusions is not recomputed twice.
    """
    if not blocked:
        return precomputed
    conflicts = cycle_closing_picks(precomputed, blocked, common['raw_columns'])
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
    started = time.perf_counter()
    args, kwargs = _mde_call(spec, node, common, blocked=blocked)
    try:
        result = mde.run_mde_for_target(*args, **kwargs)
    except Exception as exc:          # same policy as the precompute pool's handler
        if mde.is_gpu_memory_failure(exc):
            print(f'parallelized MDE: {node} recompute hit a GPU MEMORY failure: {exc!r} '
                  f'-- STOPPING the run. {mde.GPU_MEMORY_STOP_ADVICE}', flush=True)
            raise
        print(f'parallelized MDE: {node} recompute FAILED: {exc!r} -- treating as a dead end',
              flush=True)
        result = mde.failed_mde_result(node)
    _log_progress(progress_log_path,
                  f'{spec["cache_stage"]} {node} RECOMPUTE END -- '
                  f'{time.perf_counter() - started:.1f}s')
    recompute_cache[key] = result
    return result


def precompute_all_mde(tasks, stage_specs, common, cores, mp_start_method,
                       progress_log_path, phase_label):
    '''Run every (cache_stage, node) in `tasks` in ONE pool with no barrier
    inside it; return {cache_stage: {node: result}}.

    `tasks` may mix stages freely -- a Stage 1 call and a Stage 2 call for the
    same node are independent, which is what lets PARALLEL_MDE_SCOPE='union'
    hand both stages' entire task lists to a single pool and contain exactly
    one barrier in the whole network build.

    The pool, the heartbeat, the bounded submission window and the per-task
    failure policy are all waves.run_mde_batch(); see that module. What this
    function adds is GPU admission control.

    GPU ADMISSION CONTROL. One budget per visible device, sized from what the
    device reports rather than a hardcoded card size (the job goes to
    whichever GPU partition starts sooner, and those need not match). The
    budgets and the round-robin device counter are built HERE, in the parent,
    because multiprocessing primitives can only reach a spawned child by being
    passed at process construction -- which is what ProcessPoolExecutor's
    initargs are. Without this every worker would launch its GPU work blind,
    which is how job 4741044 lost 51 candidates. Empty list on a GPU-less
    host, and then gpu_budget.reserve() is a transparent no-op.
    '''
    tables = {name: {} for name in stage_specs}
    if not tasks:
        return tables

    outer_workers, _inner = waves.split_cores(len(tasks), cores)
    ctx = mp.get_context(mp_start_method)
    gpu_budgets = gpu_budget.create_budgets(ctx, workers=outer_workers)
    device_counter = ctx.Value('i', 0)
    if gpu_budgets:
        print(gpu_budget.env_summary(), flush=True)

    window_estimate = {}

    def submit_one(executor, key, inner_workers):
        cache_stage, node = key
        spec = stage_specs[cache_stage]
        args, kwargs = _mde_call(spec, node, common, inner_workers=inner_workers)
        # Heartbeat ETA input only -- see _preview_window_rows().
        window_estimate[key] = _preview_window_rows(
            node, spec['df'], spec['delay_set'], common['prediction_start'],
            common['prediction_length'], common['tp'],
            node in common['targets'], common['target_ranking_window_rows'],
            common['min_pred_events'], common['max_window_rows'])
        return executor.submit(mde.run_mde_for_target, *args, **kwargs)

    phase_start = time.perf_counter()
    results = waves.run_mde_batch(
        tasks, submit_one, cores, mp_start_method, progress_log_path,
        f'parallel MDE precompute [{phase_label}]',
        window_estimate=window_estimate,
        # A sliding window rather than one unbounded submit: every task
        # carries its stage's whole DataFrame, and ProcessPoolExecutor
        # pickles a call's arguments at submit() time, so submitting a
        # production-sized universe at once would queue that many copies.
        max_inflight=max(outer_workers * 2, outer_workers),
        pool_initializer=gpu_budget.init_worker,
        pool_initargs=(gpu_budgets, device_counter),
    )
    for (cache_stage, node), result in results.items():
        tables[cache_stage][node] = result

    print(f'parallel MDE precompute [{phase_label}]: completed {len(tasks)} MDE call(s) '
          f'in {time.perf_counter() - phase_start:.1f}s -- network construction is '
          f'serial from here', flush=True)
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
    '''Two-stage nested-MDE network construction, every MDE call made up front.

    Same signature as the serial builder plus `parallel_mde_scope`, same
    return value, and -- this is the contract, asserted by
    tests/test_parallel_mde_equivalence.py -- the SAME NETWORK for the same
    inputs under scope='staged'.

        PHASE 0  precompute_all_mde() -- every MDE call for the whole node
                 universe, both stages, in one pool with no barrier inside
                 it. This is ~all of the runtime.
        PHASE 1  Stage 1's BFS, with the MDE call replaced by a dict lookup.
                 Milliseconds.
        PHASE 2  Stage 2, likewise a lookup. Skipped under scope='union',
                 where each node's single MDE call already ranked lagged
                 candidates alongside real ones.

    WHAT "PRUNING" MEANS HERE. Nothing is scored and dropped. A node the BFS
    never reaches simply never has its Phase 0 entry read, so no edge into it
    ever enters the graph and its result is discarded when this returns. Both
    the budget-bound case (a driver past the nested_m cap) and the unreachable
    case (an observable nothing ever selects) are handled by the identical BFS
    code that handled them in the serial builder.

    STAGE ORDER. The serial pipeline guarantees Stage 1 finishes before Stage 2
    begins, and that still holds: the Stage 1 replay completes before the Stage
    2 replay starts. What scope='union' does not preserve is the order of the
    MDE CALLS -- they all go into one pool and interleave freely. That order
    was a CONSEQUENCE of the serial pipeline's per-wave barriers, never an
    input to the algorithm: no MDE call reads the graph, so a Stage 2 call
    cannot observe anything a Stage 1 call produced.

    `parallel_mde_scope`:
      'union'  (default) one pool, both stages, whole universe. Exactly one
               barrier; maximum GPU/core utilisation; largest overshoot
               (Stage 2 is computed for every observable, including ones the
               backbone never reaches), and a much more expensive call per
               node, since every pool is (1 + len(lag_set))x wider from the
               first selection round.
      'staged' two pools: Stage 1 over the universe, the Stage 1 replay, then
               Stage 2 over only the nodes Stage 1 visited. One extra barrier,
               but Stage 2's task list is then exactly the serial builder's
               own Stage 2 workload with none of its per-wave barriers. Use
               when Stage 2's overshoot is what you cannot afford -- its
               candidate pool is len(lag_set)x wider than Stage 1's, so its
               per-node calls are the expensive ones.
    '''
    if parallel_mde_scope not in ('union', 'staged'):
        raise ValueError(f'parallel_mde_scope must be "union" or "staged", got '
                         f'{parallel_mde_scope!r}')

    graph = nx.MultiDiGraph()
    graph.add_nodes_from(targets)

    # expand_delay_columns() returns the raw columns first, then one column
    # per nonzero delay, so the tail is Stage 2's lagged-only pool and the
    # whole thing is the combined pool scope='union' uses.
    stage1_num_delays = len(expanded_columns) // len(raw_columns)
    stage2_df, stage2_expanded = expand_delay_columns(expanded_df, raw_columns, lag_set)
    stage2_universe = stage2_expanded[len(raw_columns):]

    stage_specs = {
        'stage1': _stage_spec('stage1', expanded_df, expanded_columns,
                              delay_set, stage1_num_delays, raw_columns),
        'stage2': _stage_spec('stage2', stage2_df, stage2_universe,
                              lag_set, len(lag_set), raw_columns),
        # scope='union' only: real and lagged candidates in ONE pool, so each
        # node gets a single MDE call in which they compete directly.
        'union': _stage_spec('union', stage2_df, stage2_expanded,
                             lag_set, len(lag_set) + 1, raw_columns),
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

    edges = []
    nested_mde_results = {}
    node_depth = {t: 0 for t in targets}

    # =================================================================
    # PHASE 0 -- every MDE call for the whole node universe, one pool, no
    # barrier.
    # =================================================================
    universe = mde_node_universe(targets, raw_columns)
    if parallel_mde_scope == 'union':
        # main()'s depth-0 result is NOT reusable here: it ranked real
        # observables only, against a different (narrower) pool.
        precompute_tasks = [('union', n) for n in universe]
        phase_label = 'phase0 union'
    else:
        # A depth-0 target whose MDE main() already ran is reused, not
        # recomputed -- the serial builder's own filter. Targets are never
        # candidates, so a target can never be rediscovered at depth > 0.
        precompute_tasks = [('stage1', n) for n in universe
                            if not (n in targets and n in mde_results)]
        phase_label = 'phase0a stage1'

    print(f'nested MDE (parallelized): node universe = {len(universe)} node(s) '
          f'({len(targets)} target(s) + {len(universe) - len(targets)} observable(s)), '
          f'scope={parallel_mde_scope}', flush=True)
    _log_progress(progress_log_path,
                  f'PHASE 0 START (parallelized MDE precompute, '
                  f'scope={parallel_mde_scope}) -- universe={len(universe)}')

    mde_tables = precompute_all_mde(precompute_tasks, stage_specs, common, cores,
                                    mp_start_method, progress_log_path, phase_label)
    if parallel_mde_scope == 'staged':
        for target in targets:
            if target in mde_results:
                mde_tables['stage1'][target] = mde_results[target]

    recompute_cache = {}

    def lookup(cache_stage, node):
        '''Phase 0's result for one node. A miss means mde_node_universe() is
        wrong (it is provably a superset of both BFS visited sets) -- fail
        loudly rather than substituting a dead end, which would silently
        change the network.'''
        try:
            return mde_tables[cache_stage][node]
        except KeyError:
            raise RuntimeError(
                f'parallelized MDE: no {cache_stage} result precomputed for node '
                f'{node!r} -- mde_node_universe() missed a node the BFS reached. This '
                f'is a bug in the universe construction, not a recoverable condition.'
            ) from None

    # =================================================================
    # PHASE 1 -- Stage 1's real-observable BFS. Serial, and effectively
    # instantaneous: the per-wave pool is now a dict lookup.
    # =================================================================
    visited1 = set()
    current_wave = [(t, 0) for t in targets]
    replay_stage = 'union' if parallel_mde_scope == 'union' else 'stage1'
    stage1_start = time.perf_counter()
    _log_progress(progress_log_path, 'STAGE 1 REPLAY START (real-observable BFS, no compute)')

    while current_wave:
        wave_nodes = _dedupe_wave(current_wave, visited1)
        if not wave_nodes:
            break
        print(f'nested MDE (Stage 1 replay): wave of {len(wave_nodes)} node(s) at depth '
              f'{wave_nodes[0][1]} -- serial, results from phase 0', flush=True)
        _log_progress(progress_log_path,
                      f'stage1 replay wave -- {len(wave_nodes)} node(s) at depth '
                      f'{wave_nodes[0][1]}: {[n for n, _ in wave_nodes]}')

        next_wave = {}
        for node, depth in wave_nodes:
            visited1.add(node)
            node_depth[node] = depth
            # The graph already holds every edge accepted so far, so the
            # blocked set is known here even though Phase 0 could not know it.
            # A precomputed result that respects it is used as-is; one that
            # does not forces a serial recompute for this node only.
            result = _mde_result_for_node(
                node, stage_specs[replay_stage], common,
                blocked_observables(graph, node), lookup(replay_stage, node),
                recompute_cache, progress_log_path)
            nested_mde_results[node] = result
            for driver, delay in _accept_mde_result_edges(graph, edges, node, result,
                                                          raw_columns):
                if delay == 0 and (nested_m == 0 or depth < nested_m) and driver not in visited1:
                    next_wave[driver] = min(next_wave.get(driver, depth + 1), depth + 1)
        current_wave = list(next_wave.items())

    _log_progress(progress_log_path,
                  f'STAGE 1 REPLAY END -- {len(visited1)} node(s), '
                  f'{time.perf_counter() - stage1_start:.1f}s total')
    print(f'nested MDE (Stage 1): completed -- {len(visited1)} node(s) in the '
          f'real-observable backbone, Stage 2 starts now', flush=True)

    # =================================================================
    # PHASE 2 -- Stage 2. Only under scope='staged'; 'union' already ranked
    # lagged candidates inside each node's single call.
    # =================================================================
    visited2 = set()
    # ZERO LAGS. DELAY_SOURCE='explicit' with DELAY_SET=0 resolves to an EMPTY
    # lag set (derive_lag_set() strips every 0 entry on that path), leaving
    # Stage 2 with an empty candidate pool: nothing to select, nothing to
    # attach, and effective_mde_dimension() would divide by its num_delays of
    # 0. Job 4774133 died exactly there (2026-10-27), after Stage 1 had
    # completed. Skipped outright rather than guarded at the division: a stage
    # with an empty pool has no work to do. scope='union' needs no guard --
    # its num_delays is len(lag_set) + 1, so never 0.
    if parallel_mde_scope == 'staged' and lag_set:
        # Stage 2 attaches lagged leaves and never recurses, so its node set
        # is EXACTLY Stage 1's visited set -- no overshoot at all, which is
        # the whole point of this scope. Sorted for the same reason the serial
        # builder sorts: a set's iteration order is not reproducible, and
        # these two must agree node for node.
        stage2_nodes = sorted(visited1, key=lambda n: (node_depth[n], n))
        print(f'nested MDE (parallelized): scope=staged -- precomputing Stage 2 for '
              f'{len(stage2_nodes)} node(s) of {len(universe)}', flush=True)
        stage2_tables = precompute_all_mde(
            [('stage2', n) for n in stage2_nodes], stage_specs, common, cores,
            mp_start_method, progress_log_path, 'phase0b stage2')
        mde_tables['stage2'].update(stage2_tables['stage2'])

        stage2_start = time.perf_counter()
        _log_progress(progress_log_path, 'STAGE 2 REPLAY START (lagged drivers, no compute)')
        print(f'nested MDE (Stage 2 replay): {len(stage2_nodes)} node(s) -- serial, '
              f'results from phase 0, candidate pool size {len(stage2_universe)} '
              f'(lagged columns only, lag_set={lag_set})', flush=True)
        for node in stage2_nodes:
            visited2.add(node)
            # No exclusion needed: lagged picks become leaves, which cannot
            # close a cycle, so the precomputed result is always usable.
            _accept_mde_result_edges(graph, edges, node, lookup('stage2', node), raw_columns)
        _log_progress(progress_log_path,
                      f'STAGE 2 REPLAY END -- {len(visited2)} node(s), '
                      f'{time.perf_counter() - stage2_start:.1f}s total')
    elif parallel_mde_scope == 'staged':
        print('nested MDE (parallelized): scope=staged -- Stage 2 SKIPPED ENTIRELY, the lag '
              'set is empty (zero lags), so there are no lagged candidates to select or '
              'attach; this network is Stage 1 only', flush=True)
        _log_progress(progress_log_path, 'STAGE 2 SKIPPED -- empty lag_set (zero lags)')

    edges_df = _finish(graph, edges, node_depth, nested_n, nested_m, progress_log_path)

    # Overshoot accounting -- the price this builder pays for having no
    # barriers. `computed` is what Phase 0 ran; `consulted` is what the two
    # replays actually read.
    computed = sum(len(table) for table in mde_tables.values())
    # union, and staged with an empty lag set: visited2 is empty by design.
    consulted = len(visited1) + len(visited2)
    print(f'nested MDE (parallelized): {computed} MDE result(s) precomputed, {consulted} '
          f'consulted by the BFS ({computed - consulted} computed but never reached -- '
          f'the cost of removing the wave barriers)', flush=True)
    _log_progress(progress_log_path,
                  f'PRECOMPUTE ACCOUNTING -- precomputed={computed}, consulted={consulted}')

    return graph, edges_df, nested_mde_results
