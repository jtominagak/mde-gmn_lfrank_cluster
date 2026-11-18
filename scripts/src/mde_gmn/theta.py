"""Tuning theta: how strongly each node weights its nearest neighbours.

This is step 5 of the pipeline. The network and the embeddings are fixed by
now; one number per node is still free. GMN predicts a node with S-map, a local
linear fit whose neighbours are weighted by distance, and theta sets how
sharply that weighting falls off. theta=0 weights all neighbours equally (one
global linear model); large theta trusts only the closest neighbours (strongly
nonlinear). The best value differs per node, so it is searched per node.

    proper_predict_nonlinear()  sweeps theta for one node and returns the score
                                for each value. Replaces pyEDM's
                                PredictNonlinear, which scores by Pearson
                                correlation only -- this pipeline needs 'f1'
                                for spike-train nodes (see tsls_data._metric_for).
    retune_theta_for_all_nodes()
                                runs that sweep for every node, on a 70/30
                                train/test split of the node's own embedding.
                                Falls back to theta=3 when a node has too
                                little data.
    evaluate_smap_candidate()   scores one complete configuration by actually
                                running GMN's Generate() and comparing it with
                                the held-out recording.
    run_theta_search()          retunes every network once, then scores it with
                                a single Generate() call.

Only theta is searched. E and each driver's delay used to be searched too;
they now come from the network itself (gmn_embedding.py), which is what makes
searching one parameter per node affordable.

Results are cached per network (caching.py).

Depends on: gmn_build, gmn_embedding, caching, tsls_data, progress;
cleaned_helper_functions, gmn, pyEDM.SMap, numpy, pandas.
Imported by: generate, pipeline, the entry scripts.
"""

import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

import cleaned_helper_functions as tsls
import gmn
import numpy as np
import pandas as pd
from pyEDM import SMap

from mde_gmn.caching import find_cached_best_params, save_best_params
from mde_gmn.gmn_build import build_gmn_config
from mde_gmn.gmn_embedding import _embedding_spec_for, _generalized_embed
from mde_gmn.network import _simulated_nodes
from mde_gmn.progress import _format_eta, _log_progress
from mde_gmn.tsls_data import _metric_for


def proper_predict_nonlinear(dataFrame, columns, target, theta, lib, pred,
                              E=1, Tp=1, knn=0, tau=-1, exclusionRadius=0,
                              embedded=False, metric='correlation', gamma_delta=2):
    '''Sweep theta for one node and score each value with a TSLS-aware metric.

    A replacement for pyEDM.PredictNonlinear's theta sweep, NOT a new S-map
    engine -- the S-map itself is still pyEDM's. Reading pyEDM's source
    (API.py's PredictNonlinear; PoolFunc.py's PredictNLSMapFunc worker) shows
    PredictNonlinear
    already does nothing more than call pyEDM.SMap(...) once per
    candidate theta and score it via `ComputeError(obs, pred)['rho']` --
    the SAME compiled S-map gmn's own Forecast()/Generate() already calls
    directly at runtime for every node (unaffected by this function
    either way -- theta is just a number once chosen, regardless of which
    sweep chose it). `ComputeError`'s own `rho = corrcoef(obs, pred)[0,1]`
    is the identical formula `cleaned_helper_functions._score(kind=
    'correlation')` uses, so this function reproduces PredictNonlinear's
    selection exactly at metric='correlation' -- asserted by
    tests/test_r20_predict_nonlinear.py -- while
    also supporting 'f1'/'gamma'/'mae' for non-target (neuron) nodes,
    which PredictNonlinear itself can never score.

    Deliberately sequential -- one pyEDM.SMap call per theta, in-process,
    no multiprocessing.Pool. See sitecustomize.py's own module docstring
    (this venv) for why: pyEDM's own default PredictNonlinear/
    EmbedDimension/PredictInterval each spawn a *fresh* Pool per call,
    which is what turned this pipeline's jobs into a ~2-day-regardless-
    of-config bug before that file patched all three to a sequential
    drop-in. This function must not reintroduce that mistake -- theta
    grids here are ~10-20 values, cheap enough sequentially that adding
    pooling machinery back in isn't warranted.

    Returns a DataFrame with columns 'theta'/'score' (one row per
    candidate) -- callers pick `df.loc[df['score'].idxmax(), 'theta']`,
    mirroring PredictNonlinear's own `df.loc[df['rho'].idxmax(), 'theta']`
    idiom exactly, just against a metric-agnostic column name instead of
    a hardcoded 'rho'.'''
    rows = []
    for th in theta:
        S = SMap(
            dataFrame=dataFrame, columns=columns, target=target,
            lib=lib, pred=pred, E=E, Tp=Tp, knn=knn, tau=tau,
            theta=th, exclusionRadius=exclusionRadius,
            embedded=embedded, showPlot=False,
        )
        predictions = S['predictions']
        obs = predictions['Observations'].to_numpy()
        prd = predictions['Predictions'].to_numpy()
        rows.append({'theta': th, 'score': tsls._score(obs, prd, metric, gammaDelta=gamma_delta)})
    return pd.DataFrame(rows)

def _theta_sweep_one_node(node, embedded, theta_range, node_metric, gamma_delta,
                          min_pred_rows, tp=1):
    """One node's theta sweep -- the body retune_theta_for_all_nodes() runs
    per node, as a top-level function so a worker process can run it.
    Returns (node, best_theta, seconds)."""
    started = time.perf_counter()
    n_emb = len(embedded)
    split = int(n_emb * 0.7)
    if n_emb - split < min_pred_rows:
        return node, 3, time.perf_counter() - started
    try:
        pn = proper_predict_nonlinear(
            embedded, columns=list(embedded.columns[1:-1]), target=node,
            theta=list(theta_range), lib=f'1 {split}', pred=f'{split + 1} {n_emb}',
            Tp=tp, embedded=True, metric=node_metric, gamma_delta=gamma_delta,
        )
        best = float(pn.loc[pn['score'].idxmax(), 'theta'])
    except Exception:
        best = 3
    return node, best, time.perf_counter() - started


def _memory_limit_bytes():
    """The job's memory limit: SLURM's per-node allocation when there is one,
    otherwise the machine's physical memory."""
    mb = os.environ.get('SLURM_MEM_PER_NODE')
    if mb and mb.isdigit():
        return int(mb) * 2**20
    return os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')


def theta_sweep_workers(cores, n_nodes, n_rows, memory_bytes=None):
    """How many nodes' theta sweeps can run at once.

    Bounded by cores, by the node count, and by memory. pyEDM's SMap with
    knn=0 regresses each prediction row on EVERY library row, so one call
    holds knn_distances, knn_neighbors, W, B and W*B, each n_pred x n_lib
    8-byte values: about 5 * 8 * 0.3n * 0.7n bytes for an n-row embedding
    split 70/30. At job 4799404's ~33,000 rows that is ~9 GB per worker, so
    cores alone would ask for more memory than the job has. 60% of the
    limit is handed out, leaving room for the parent's own frames."""
    per_worker = 5 * 8 * (0.3 * n_rows) * (0.7 * n_rows) * 1.2
    limit = memory_bytes if memory_bytes is not None else _memory_limit_bytes()
    by_memory = int(0.6 * limit // max(per_worker, 1))
    return max(1, min(cores, n_nodes, by_memory))


def retune_theta_for_all_nodes(data_lib, graph, nodes, targets, theta_range, cores,
                                metric, target_metric, gamma_delta, min_pred_rows=20,
                                progress_log_path=None, tp=1):
    '''Finds each node's best theta via proper_predict_nonlinear(), on a 70/30
    train/test split of that node's own embedding.

    Scores each node with _metric_for(node, targets, metric, target_metric) --
    'f1' for neuron nodes, 'correlation' (target_metric) for TARGETS, matching
    every other scoring call site in this file. Falls back to theta=3
    whenever there's too little data for the split or the sweep itself fails.

    PARALLEL ACROSS NODES. Nodes are independent, so up to
    theta_sweep_workers() of them run at once in a process pool. This used
    to be a plain loop that ignored `cores`, and it was the single largest
    cost of every run: 54,551s for 116 nodes in job 4799404 and 290,824s
    (81h) for 131 in job 4799407, on 18 allocated cores. Each sweep is still
    sequential inside, one pyEDM.SMap call per theta -- proper_predict_
    nonlinear()'s reason for that (pyEDM's per-call Pool spawn) is
    untouched: this is ONE pool for the whole stage, opened once. Results
    are identical to the loop; only the order nodes finish in changes.
    Embeddings are built in this process (they need the graph and the full
    frame) and handed to workers ready-made, a few MB each. The pool uses
    'spawn': the parent may hold CUDA state, which fork must not copy.'''
    best_theta_by_node = {}
    theta_durations = []
    _log_progress(progress_log_path, f'theta retuning START -- {len(nodes)} node(s)')
    theta_stage_start = time.perf_counter()

    def _task(node):
        spec = _embedding_spec_for(graph, node)
        embedded, offset = _generalized_embed(data_lib, spec)
        embedded.insert(0, 'Time', np.arange(len(embedded)))
        embedded[node] = data_lib[node].to_numpy()[offset:]
        return (node, embedded, list(theta_range),
                _metric_for(node, targets, metric, target_metric), gamma_delta, min_pred_rows,
                tp)

    def _record(node, best, seconds):
        best_theta_by_node[node] = best
        theta_durations.append(seconds)
        eta = _format_eta(theta_durations, len(nodes) - len(theta_durations))
        _log_progress(progress_log_path,
                      f'theta node END -- {node} duration={seconds:.1f}s theta={best}{eta}')

    workers = theta_sweep_workers(cores, len(nodes), len(data_lib)) if nodes else 1
    print(f'theta retuning: {len(nodes)} node(s), {workers} at a time '
          f'(cores={cores}, rows={len(data_lib)})', flush=True)
    if workers <= 1:
        for node in nodes:
            _record(*_theta_sweep_one_node(*_task(node)))
    else:
        queue = list(nodes)
        with ProcessPoolExecutor(max_workers=workers,
                                 mp_context=mp.get_context('spawn')) as executor:
            pending = set()
            while queue or pending:
                while queue and len(pending) < 2 * workers:
                    pending.add(executor.submit(_theta_sweep_one_node, *_task(queue.pop(0))))
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for fut in done:
                    _record(*fut.result())
        # Keep the caller's node order in the returned dict.
        best_theta_by_node = {n: best_theta_by_node[n] for n in nodes}

    _log_progress(progress_log_path,
                  f'theta retuning END -- {len(nodes)} node(s), '
                  f'{time.perf_counter() - theta_stage_start:.1f}s total')
    return best_theta_by_node

def generate_memo_key(generate_start, generate_length, exclusion_radius, node_theta):
    '''What makes two Generate() calls on one network the same computation:
    the start row, the length, the exclusion radius and every node's theta.
    Generate() is deterministic in exactly these (plus the network, which a
    memo is kept per-network for), so equal keys mean identical output.'''
    return (int(generate_start), generate_length, exclusion_radius,
            tuple(sorted((str(n), float(t)) for n, t in (node_theta or {}).items())))


def evaluate_smap_candidate(net_result, prediction_start, prediction_length,
                             node_theta, variant, results_dir, cores,
                             target_metric, gamma_delta,
                             generate_length=None, generate_start=None, exclusion_radius=0,
                             generate_validation_start=None, memo=None):
    '''Runs one Generate() call and scores it as mean rho, across
    `net_result['targets']` only, between the simulated and real held-out
    series. Scored with tsls._score(metric=target_metric) rather than pyEDM's
    plain ComputeError rho; TARGETS are continuous, so target_metric defaults to
    'correlation' (see config.py's TARGET_METRIC entry). This is "generative skill" as this pipeline defines it
    everywhere -- run_theta_search() and sweep_generate_start() both call
    this same function, at the same objective, so a search over either axis
    is comparable to the other.

    generate_start defaults to generate_validation_start (== cfg's B, the
    start of the Generate/Forecast validation segment), NOT prediction_start
    (== A) -- a Generate() call scored here must never launch from inside
    the segment MDE's own driver selection already validated against.

    `memo`: a dict shared by one network's callers. An identical call (see
    generate_memo_key()) is answered from it instead of rerun. The pipeline
    used to run two Generate()s twice over: the theta search's own scoring
    call IS start-sweep offset 0 (both scored 0.7407 in job 4799407), and the
    final rebuild IS the sweep's winning offset. Each one is ~1h at job
    4799404's size and was ~30h at job 4799407's.'''
    generate_start = generate_start if generate_start is not None else generate_validation_start
    key = generate_memo_key(generate_start, generate_length, exclusion_radius, node_theta)
    if memo is not None and key in memo:
        return memo[key]
    config_file = build_gmn_config(
        net_result, prediction_start, prediction_length, results_dir,
        mode='Generate', node_theta_overrides=node_theta,
        variant=variant, generate_length=generate_length,
        generate_start=generate_start, exclusion_radius=exclusion_radius,
    )
    G = gmn.GMN( configFile = str(config_file), cores = cores )
    G.Generate()

    actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
    generated = G.DataOut.reset_index(drop=True)
    n_gen = len(generated)

    target_rhos = {}
    for target in net_result['targets']:
        actual_vals = actual_df[target].to_numpy()[generate_start: generate_start + n_gen]
        gen_vals = generated[target].to_numpy()
        target_rhos[target] = tsls._score(actual_vals, gen_vals, target_metric, gammaDelta=gamma_delta)

    score = float(np.nanmean(list(target_rhos.values())))
    if memo is not None:
        memo[key] = (score, target_rhos, G)
    return score, target_rhos, G

def run_theta_search(gmn_runs, prediction_start, prediction_length, theta_range, results_dir,
                      cores, metric, target_metric, gamma_delta,
                      generate_length=None, reuse_best_params=False, prev_runs_dir=None,
                      run_cache_dir=None, fingerprints=None, exclusion_radius=0,
                      generate_validation_start=None, progress_log_path=None):
    '''Retunes each network's per-node theta once, then scores the result
    with a single Generate() call. E and each driver's delay are not search
    parameters at all: for
    MODE='MDE-GMN' they come directly from nested-MDE's own discovered
    network; for MODE='GMN' they still come from that mode's own CCM E/tau
    grid search (fallback_E/fallback_tau, set in build_network()).'''
    theta_search_results = {}
    for network_id, run in gmn_runs.items():
        net_result = run['net_result']
        graph = net_result['network']['Graph']
        # External inputs fed from real data have no embedding, so there is no
        # theta to retune for them (see _simulated_nodes()'s docstring).
        all_nodes = _simulated_nodes(graph)
        data_after_mde_lib = pd.read_csv(net_result['DATA_AFTER_MDE']).iloc[:prediction_start]

        fingerprint = fingerprints[network_id] if fingerprints else None
        if reuse_best_params and fingerprint is not None:
            cached = find_cached_best_params(prev_runs_dir, network_id, fingerprint)
            if cached is not None:
                best_node_theta, source = cached
                print(f'{network_id}: TSLS_MDE_GMN_REUSE_BEST_PARAMS matched {source} -- '
                      f'skipping theta retuning entirely', flush=True)
                theta_search_results[network_id] = {
                    'best_node_theta': best_node_theta, 'reused_from': str(source),
                }
                save_best_params(results_dir, run_cache_dir, network_id, fingerprint, best_node_theta,
                                  source=f'reused_from:{source}')
                continue
            print(f'{network_id}: TSLS_MDE_GMN_REUSE_BEST_PARAMS set but no matching cached run '
                  f'found under {prev_runs_dir} -- retuning theta.', flush=True)

        print(f'{network_id}: retuning theta for {len(all_nodes)} node(s)', flush=True)
        best_node_theta = retune_theta_for_all_nodes(
            data_after_mde_lib, graph, all_nodes, net_result['targets'], theta_range, cores,
            metric, target_metric, gamma_delta,
            progress_log_path=progress_log_path,
            # theta is tuned at the horizon the model will run at
            tp=net_result.get('gmn_tp', 1),
        )

        try:
            score, _, _ = evaluate_smap_candidate(
                net_result, prediction_start, prediction_length, best_node_theta,
                'theta_search', results_dir, cores, target_metric, gamma_delta,
                generate_length=generate_length,
                exclusion_radius=exclusion_radius,
                generate_validation_start=generate_validation_start,
                memo=run.setdefault('generate_memo', {}),
            )
            print(f'{network_id}: theta search score={score:.4f}', flush=True)
        except Exception as exc:
            # gmn.Generate raises RuntimeError on SMap divergence (blown-up
            # recursive forecast, >1E15) -- possible even at a retuned
            # theta on a large, high-dimensional network, not a bug to
            # crash the whole run over.
            print(f'{network_id}: theta search candidate FAILED: {exc!r}', flush=True)

        theta_search_results[network_id] = {'best_node_theta': best_node_theta}

        if fingerprint is not None:
            save_best_params(results_dir, run_cache_dir, network_id, fingerprint, best_node_theta,
                              source='computed')

    return theta_search_results
