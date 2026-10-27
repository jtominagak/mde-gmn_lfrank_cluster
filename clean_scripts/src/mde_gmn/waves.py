"""Running a batch of MDE calls in parallel, with a heartbeat.

Both network builders have to do the same bookkeeping around a group of
independent MDE calls, and all of it is incidental to the algorithm:

  * decide how to split the core budget between processes and threads,
  * submit the calls,
  * wait for them without going silent (a single call can run for hours),
  * record each result, or convert a failure into a dead end,
  * keep enough timing to estimate what is left.

That bookkeeping used to be written out four times -- Stage 1 and Stage 2 in
network.py, and the precompute phase in network_parallel.py -- in copies that
had drifted apart in wording while staying identical in behaviour. It lives
here once instead, so the builders read as the graph algorithms they are.

    split_cores()    how many processes x how many threads each
    run_mde_batch()  submit a batch, wait with a heartbeat, collect results

THE ONE RULE THAT MATTERS. outer_workers * inner_workers must never exceed
the cores the job was given. Nested pools that each claim every core
multiply instead of sharing, and this pipeline has been brought down that
way before. split_cores() is the only place that arithmetic happens, and
run_mde_batch() logs the product so a run's own log can be checked.

A GPU MEMORY FAILURE IS NOT A RESULT. Every other exception becomes a dead
end -- one degenerate node must not take down the thousands queued beside
it. Running out of GPU memory is different: the candidate was never
evaluated, and downstream a missing edge is indistinguishable from one that
was measured and found absent. So that one failure stops the run. See
mde.is_gpu_memory_failure().

Depends on: mde (through the module object, so tests' fake is honoured),
progress; nothing else in the package.
Imported by: network, network_parallel.
"""

import multiprocessing as mp
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from mde_gmn import mde          # module object, not its names -- see docstring
from mde_gmn.progress import HEARTBEAT_SECONDS, _format_eta, _log_progress


def split_cores(n_tasks, cores):
    """(outer_workers, inner_workers) for `n_tasks` independent MDE calls.

    One process per task up to the core budget, then whatever is left over
    becomes threads inside each process (mde.run_mde_for_target()'s own
    inner pool -- proper_mde has no parallelism of its own to hand a core
    count to). The product never exceeds `cores`.

    With more tasks than cores -- the parallel builder's whole-universe
    precompute -- this pins outer at `cores` and inner at 1, which is the
    point: the parallelism moves from inside each node to across nodes.
    """
    outer = min(max(1, n_tasks), max(1, cores))
    inner = max(1, cores // outer)
    return outer, inner


def run_mde_batch(tasks, submit_one, cores, mp_start_method, progress_log_path,
                  label, window_estimate=None, max_inflight=None,
                  pool_initializer=None, pool_initargs=()):
    """Run every task in `tasks` across a process pool; return {key: result}.

    tasks
        List of opaque keys. Each is passed to `submit_one` and used to key
        the returned dict. A key is whatever the caller finds useful -- a
        node name in network.py, a (stage, node) pair in network_parallel.py.
    submit_one(executor, key) -> Future
        Submits one task. The caller owns every MDE argument; this module
        deliberately knows none of them.
    window_estimate
        Optional {key: rows} of each task's expected scoring-window size,
        used only to weight the heartbeat's remaining-time estimate (node
        cost scales roughly with window size, so treating every pending
        node as equally expensive makes the estimate much worse). Omitted
        -> a flat per-task average.
    max_inflight
        Cap on outstanding futures. ProcessPoolExecutor pickles a call's
        arguments at submit() time and holds them queued, and these calls
        each carry a whole DataFrame, so submitting a production-sized
        universe at once would queue that many copies of it. A new task is
        submitted the instant one completes, so this bounds memory without
        reintroducing a barrier: a sliding window, not a wave. None ->
        submit everything up front (fine for one BFS wave).
    pool_initializer / pool_initargs
        Passed to ProcessPoolExecutor. The parallel builder uses these to
        hand each worker its GPU admission-control budget, which can only
        reach a spawned child through initargs.

    Raises only on a GPU memory failure (see this module's docstring).
    """
    results = {}
    if not tasks:
        return results

    outer_workers, inner_workers = split_cores(len(tasks), cores)
    inflight_cap = min(len(tasks), max_inflight or len(tasks))
    window_estimate = window_estimate or {}

    print(f'{label}: {len(tasks)} MDE call(s) -- {outer_workers} parallel worker(s) '
          f'x {inner_workers} inner thread(s) each '
          f'(product={outer_workers * inner_workers}, cores={cores}), '
          f'max {inflight_cap} in flight', flush=True)
    _log_progress(progress_log_path,
                  f'{label} START -- {len(tasks)} MDE call(s), '
                  f'{outer_workers}x{inner_workers}')

    started = time.perf_counter()
    durations = []     # seconds per completed task, this run only
    rates = []         # seconds per window-row -- see progress._format_eta()

    # 'fork' for CPU-only runs; 'spawn' whenever the torch/CUDA path is on,
    # because CUDA does not support fork(). See config.py's GPU_ENABLED.
    ctx = mp.get_context(mp_start_method)
    with ProcessPoolExecutor(max_workers=outer_workers, mp_context=ctx,
                             initializer=pool_initializer,
                             initargs=pool_initargs) as executor:
        pending = {}
        next_task = 0

        def fill():
            nonlocal next_task
            while next_task < len(tasks) and len(pending) < inflight_cap:
                key = tasks[next_task]
                next_task += 1
                pending[submit_one(executor, key, inner_workers)] = key

        fill()
        while pending:
            done, _ = wait(set(pending), timeout=HEARTBEAT_SECONDS,
                           return_when=FIRST_COMPLETED)
            if not done:
                # Nothing finished within the heartbeat interval. Say so:
                # a batch with no heartbeat can go silent for hours (job
                # 38976891 did, for 4+ hours of real work) with no way to
                # tell "still working" from "stuck".
                waiting = sorted(str(k) for k in pending.values())
                remaining_rows = sum(window_estimate.get(k, 0) for k in pending.values())
                eta = _format_eta(durations, len(tasks) - len(results),
                                  per_unit=rates, unit_remaining=remaining_rows)
                _log_progress(
                    progress_log_path,
                    f'{label} HEARTBEAT -- {len(results)}/{len(tasks)} done, '
                    f'{len(pending)} in flight, {len(tasks) - next_task} not yet '
                    f'submitted, elapsed={time.perf_counter() - started:.0f}s, '
                    f'still waiting on: {waiting}{eta}')
                continue

            for future in done:
                key = pending.pop(future)
                node = _node_of(key)
                # Elapsed since the batch started, not this task's own
                # runtime -- these overlap, so there is no per-task clock to
                # read. It is an ETA input, nothing else.
                elapsed = time.perf_counter() - started
                try:
                    results[key] = future.result()
                    _log_progress(progress_log_path,
                                  f'{label} node END -- {key} elapsed={elapsed:.1f}s '
                                  f'selected={results[key]["SELECTED_COLUMNS"]}')
                except Exception as exc:
                    if mde.is_gpu_memory_failure(exc):
                        print(f'{label}: {key} hit a GPU MEMORY failure: {exc!r} -- '
                              f'STOPPING the run. {mde.GPU_MEMORY_STOP_ADVICE}', flush=True)
                        _log_progress(progress_log_path,
                                      f'{label} node END -- {key} elapsed={elapsed:.1f}s '
                                      f'GPU MEMORY FAILURE: {exc!r}')
                        raise
                    print(f'{label}: {key} MDE call FAILED: {exc!r} -- treating as a '
                          f'dead-end (zero drivers selected, no further nesting from '
                          f'this node)', flush=True)
                    _log_progress(progress_log_path,
                                  f'{label} node END -- {key} elapsed={elapsed:.1f}s '
                                  f'FAILED: {exc!r}')
                    results[key] = mde.failed_mde_result(node)
                durations.append(elapsed)
                rows = window_estimate.get(key, 0)
                if rows > 0:
                    rates.append(elapsed / rows)

            fill()

    _log_progress(progress_log_path,
                  f'{label} END -- {len(tasks)} MDE call(s), '
                  f'{time.perf_counter() - started:.1f}s total')
    return results


def _node_of(key):
    """The node name inside a task key.

    Keys are node names in network.py and (stage, node) pairs in
    network_parallel.py; failed_mde_result() needs the node either way.
    """
    return key[1] if isinstance(key, tuple) else key
