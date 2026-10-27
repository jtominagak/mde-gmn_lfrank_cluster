"""Optional diagnostics: sanity checks and baseline comparisons.

Step 7, and the only optional one. Nothing here feeds back into the model; it
all answers "is the result any good, and better than something simpler?".
Skipped entirely when MODE == 'GMN'.

    run_tp_vs_rho_sweep()   how far ahead each behavioural target can predict
                            itself. Predictability that decays with horizon is
                            the signature of deterministic dynamics; flat or
                            noisy means there is little to model.
    run_univariate_e_tau_grid()
                            single-signal embedding scan over E and tau, as a
                            reference point for what one channel achieves alone.
    run_mde_evaluate()      the real comparison. MDE chose D drivers for a
                            target; PCA and a diffusion map each fit their first
                            D components on the same pool of candidates, so all
                            three are compared at equal dimensionality. If MDE's
                            chosen drivers do not beat generic dimensionality
                            reduction, the selection is not earning its keep.
                            D is the target's direct in-degree in the complete
                            two-stage network.

The baselines run in a separate process under a wall-clock bound
(_run_baseline_with_timeout, MDE_EVALUATE_BASELINE_TIMEOUT_SECONDS). A
diffusion-map fit on real TSLS data at full scale has been seen to run past 74
CPU-minutes without returning, crashing or printing anything -- a hang, which
try/except cannot catch, because nothing is raised until the call returns. A
subprocess with a timeout is the only thing that can bound it.

Depends on: tsls_data, columns; edm, sklearn, pydiffmap,
numpy, matplotlib.
Imported by: pipeline, the entry scripts.
"""

import multiprocessing as mp

import matplotlib
matplotlib.use('Agg')          # cluster jobs have no display
import edm
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from pydiffmap import diffusion_map as dmap
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

from mde_gmn.tsls_data import network_data_frame


def run_tp_vs_rho_sweep(mde_results, targets, max_tp, target_colors, results_dir,
                         target_metric, gamma_delta):
    '''How far ahead can each behavioural target predict itself?

    Always scored with target_metric, never `metric`: TARGETS are continuous
    (see config.py's TARGET_METRIC entry). Reads r['m'].args.lib/pred/tau, the
    namespace shape every run_mde_for_target() result carries.

    If every target hits the `if not cols` skip below -- all zero-driver, or all
    failed -- no figure is written at all. Writing one anyway produced an empty
    axes with an empty legend, saved to disk looking like a real result. The
    figure is closed either way, so nothing leaks.'''
    fig, ax = plt.subplots(figsize=(8, 5))
    annotation_lines = []
    for target in targets:
        r = mde_results[target]
        cols = r['SELECTED_COLUMNS']
        if not cols:
            print(f'{target}: no MDE-selected columns, skipping Tp-vs-rho sweep.', flush=True)
            continue
        pi = edm.proper_predict_interval(
            r['mde_df'], columns=cols, target=target,
            lib=r['m'].args.lib, pred=r['m'].args.pred,
            E=len(cols), tau=r['m'].args.tau,
            maxTp=max_tp, metric=target_metric, gammaDelta=gamma_delta,
        )
        ax.plot(pi['Tp'], pi[target_metric], marker='o', label=target, color=target_colors[target])
        annotation_lines.append(f'{target}: E={len(cols)}, vars={", ".join(cols)}')

    if not annotation_lines:
        print('Tp-vs-rho sweep: every target had zero MDE-selected columns -- '
              'skipping the figure entirely (no data to plot).', flush=True)
        plt.close(fig)
        return

    ax.axvline(1, color='gray', ls=':', lw=1)
    ax.set_xlabel('Forecast interval Tp (steps)')
    ax.set_ylabel(f'Prediction skill ({target_metric})')
    ax.set_title('MDE forecast skill vs. horizon')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.text(0.02, -0.02, '\n'.join(annotation_lines), fontsize=7, va='top')
    fig.savefig(results_dir / 'mde_tp_vs_rho_multi_target.png', dpi=300, bbox_inches='tight')
    plt.close(fig)

def run_univariate_e_tau_grid(mde_results, targets, e_range, tau_range, rho_threshold, results_dir,
                               target_metric, gamma_delta):
    '''How well does each target predict itself from its OWN history alone?

    A univariate grid over E and tau (columns=target, target=target). Unlike
    run_tp_vs_rho_sweep() above it needs no MDE-selected drivers, so it does NOT
    skip a target with an empty SELECTED_COLUMNS -- a genuinely zero-driver
    target still gets a real, informative grid.

    It does need `mde_df` to be a real DataFrame. A target whose MDE call RAISED
    (a different case from "selected nothing" -- see mde.failed_mde_result()) has
    mde_df=None, which would crash proper_simplex() and, because the exception
    would propagate out of this function, take every remaining target's figure
    down with it. Guarded per target instead, following this codebase's rule
    that one degenerate item dead-ends cleanly rather than failing the batch.'''
    e_vals, tau_vals = list(e_range), list(tau_range)

    for target in targets:
        r = mde_results[target]
        if r['mde_df'] is None:
            print(f'{target}: MDE call failed (mde_df is None), skipping E/tau grid.', flush=True)
            continue
        rho_grid = np.full((len(e_vals), len(tau_vals)), np.nan)

        for i, E in enumerate(e_vals):
            for j, tau in enumerate(tau_vals):
                out = edm.proper_simplex(
                    r['mde_df'], columns=target, target=target,
                    E=E, tau=tau, lib=r['m'].args.lib, pred=r['m'].args.pred,
                    embedded=False, metric=target_metric, gammaDelta=gamma_delta,
                )
                rho_grid[i, j] = out['metrics'][target_metric]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(
            rho_grid, aspect='auto', origin='upper', cmap='viridis',
            extent=[-0.5, len(tau_vals) - 0.5, len(e_vals) - 0.5, -0.5],
        )
        ax.set_xticks(range(len(tau_vals))); ax.set_xticklabels(tau_vals)
        ax.set_yticks(range(len(e_vals))); ax.set_yticklabels(e_vals)
        ax.set_xlabel('tau'); ax.set_ylabel('E')
        ax.set_title(f'E, tau grid search: {target}')
        fig.colorbar(im, ax=ax, label=target_metric)

        for i in range(len(e_vals)):
            for j in range(len(tau_vals)):
                if rho_grid[i, j] > rho_threshold:
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                              fill=False, edgecolor='red', linewidth=2.5)
                    ax.add_patch(rect)

        fig.tight_layout()
        fig.savefig(results_dir / f'mde_E_tau_grid_{target}.png', dpi=300)
        plt.close(fig)

def _cae(x, y):
    '''Cumulative absolute error, defined exactly as dimx.Evaluate's own CAE()
    defines it (dimx/Evaluate.py: `sum(abs(x - y))`), so the numbers here are
    comparable with that tool's. Unnormalized -- not a mean, not a fraction -- and, for
    a 2-column (x, y) array (GMN_MODE='combined'), sums the absolute error
    over BOTH columns at once (numpy's own `abs`/`sum` broadcast this way
    with no axis given), which is the natural generalization used for the
    combined-mode spatial panel's own CAE.'''
    return float(np.sum(np.abs(np.asarray(x) - np.asarray(y))))

def _score_baseline(target_pred, predicted, targets, combined, target_metric, gamma_delta):
    '''Shared scoring step for run_mde_evaluate()'s PCA/diffusion-map
    baselines -- factored out so both baselines' own try/except blocks
    (see that function's own docstring on why they're independently
    guarded) call one identical implementation rather than two copies of
    the combined-vs-single-target branch drifting apart.'''
    if combined:
        return float(np.nanmean([
            edm._score(target_pred[:, i], predicted[:, i], target_metric, gammaDelta=gamma_delta)
            for i in range(len(targets))
        ]))
    return edm._score(target_pred, predicted, target_metric, gammaDelta=gamma_delta)

# Wall-clock bound for the PCA/diffusion-map baselines below. A real
# diffusion-map fit on real TSLS data at full (~31,000-row) scale has been
# observed to run past 74 CPU-minutes with no completion, crash or output.
# Not calibrated against a known-good completion time -- no successful fit at
# this scale has ever been timed -- so it is chosen as generous headroom over
# the 1.8s a genuine crash takes to surface, while still bounding the worst
# case to one tolerable, once-per-network cost.
MDE_EVALUATE_BASELINE_TIMEOUT_SECONDS = 300

def _pca_baseline_worker(queue, pool_lib, pool_pred, target_lib, D):
    '''Runs in a genuinely separate process -- see
    _run_baseline_with_timeout()'s own docstring for why. Catches its own
    exception and puts it on the queue rather than letting the child
    process's own traceback (still visible in the job's .err) be the only
    record -- so the parent can log it in run_mde_evaluate()'s own
    existing "FAILED for {targets}: {exc!r}" style regardless of which
    process actually raised it.'''
    try:
        pca = PCA(n_components=D)
        pca.fit(pool_lib)
        pca_lib_proj, pca_pred_proj = pca.transform(pool_lib), pca.transform(pool_pred)
        pca_predicted = LinearRegression().fit(pca_lib_proj, target_lib).predict(pca_pred_proj)
        queue.put(('ok', pca_predicted, pca_pred_proj))
    except Exception as exc:
        queue.put(('error', repr(exc), None))

def _dmap_baseline_worker(queue, pool_lib, pool_pred, target_lib, D, dmap_k, dmap_epsilon):
    '''Same contract as _pca_baseline_worker() above.'''
    try:
        dmap_model = dmap.DiffusionMap.from_sklearn(
            n_evecs=D, k=dmap_k, epsilon=dmap_epsilon,
            neighbor_params={'n_jobs': -1, 'algorithm': 'kd_tree'},
        )
        dmap_model.fit(pool_lib)
        dmap_lib_proj, dmap_pred_proj = dmap_model.transform(pool_lib), dmap_model.transform(pool_pred)
        dmap_predicted = LinearRegression().fit(dmap_lib_proj, target_lib).predict(dmap_pred_proj)
        queue.put(('ok', dmap_predicted, dmap_pred_proj))
    except Exception as exc:
        queue.put(('error', repr(exc), None))

def _run_baseline_with_timeout(worker_fn, args, timeout_seconds=MDE_EVALUATE_BASELINE_TIMEOUT_SECONDS):
    '''Runs worker_fn(queue, *args) in a genuinely separate process,
    force-killed if it exceeds timeout_seconds -- the wall-clock bound
    run_mde_evaluate()'s PCA/diffusion-map guards need underneath their
    exception handling. A try/except plus a signal-based "soft" timeout is not
    enough, because pydiffmap's own
    KD-tree/BLAS calls can run for a long time without ever returning
    control to the Python interpreter -- a signal fires but its handler
    can't actually run until the call returns on its own, which is
    exactly the case that needs covering. Only an OS-level SIGKILL of a
    genuinely separate process reclaims control regardless of what the
    computation is doing internally). Cost: one process fork per baseline
    call (cheap under `fork` -- no re-import, copy-on-write memory, real
    overhead is milliseconds) plus pickling the (modest-sized, D-column)
    result back through a Queue; the real cost is that a hard kill on
    timeout discards whatever progress a slow-but-would-eventually-
    succeed computation had made, converting it into a skip -- the
    deliberate trade for a bounded worst case instead of an unbounded one.

    Returns (status, value): ('ok', predicted_and_proj_tuple) on success;
    ('timeout', None) if force-killed; ('error', repr(exc)) if the worker
    itself raised (only ever read in the success path below -- the
    process has already finished and flushed its queue write by the time
    proc.join() returns non-alive, so there is no kill-mid-write race to
    guard against).'''
    ctx = mp.get_context('fork')
    queue = ctx.Queue()
    proc = ctx.Process(target=worker_fn, args=(queue, *args))
    proc.start()
    proc.join(timeout_seconds)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return 'timeout', None
    try:
        status, a, b = queue.get_nowait()
    except Exception:
        return 'error', f'child process exited without a result (exitcode={proc.exitcode})'
    if status == 'ok':
        return 'ok', (a, b)
    return 'error', a

def run_mde_evaluate(net_result, prediction_start, prediction_length, results_dir,
                      target_metric, gamma_delta, exclusion_radius=0,
                      dmap_k=15, dmap_epsilon='bgh_generous', max_traces_shown=7,
                      figsize=(8, 8)):
    '''Is MDE's driver selection better than generic dimensionality reduction?

    Compares three ways of reconstructing the target from the same pool of
    candidates: MDE's chosen drivers (proper_simplex), the first D principal
    components (sklearn.PCA), and the first D diffusion-map coordinates
    (pydiffmap). Same comparison dimx.Evaluate makes, against the same two
    libraries, but built directly here: dimx.Evaluate has no awareness of this
    pipeline's TSLS metrics and needs a result object this pipeline does not
    construct.

    D = the number of *direct* drivers of the target in the COMPLETE
    nested-MDE network (both stages) -- graph.predecessors(target), read
    from net_result['network']['Graph'] (built by build_network() /
    build_nested_mde_network()), NOT a depth-0-only SELECTED_COLUMNS list.
    PCA and diffusion-map each fit their first D components on the same
    pool, so all three methods are compared at equal dimensionality -- without
    that, MDE could look better simply by using more coordinates. That pool is
    every non-target column in
    net_result['DATA_AFTER_MDE'] -- i.e. every observable nested-MDE kept
    anywhere in the whole network (a superset of the target's own direct
    drivers whenever Stage 2 chains through an intermediate node), the
    analog of dx.Evaluate's FilterData()-selected column subset: every
    candidate column except the target itself, never the full pre-pre-filter
    universe, which is not serialized anywhere after build_network() returns.

    net_result['targets'] has one entry (GMN_MODE='separate') -> a
    standard single-target figure, panel 1 an actual-vs-predicted time
    series, matching docs/reference/mde-evaluate-reference-plot.png. Two
    entries (GMN_MODE='combined') -> ONE joint figure for the pair, whose first
    panel is a spatial (x, y) trajectory instead.

    D for that joint figure is the UNION of each target's direct-driver set:
    "how many distinct driver nodes feed the targets this figure is about".
    PCA and the diffusion map predict both target columns at once in their
    regression step (sklearn's LinearRegression takes a 2-column y), rather than
    running two independent single-target comparisons; MDE still reconstructs each target
    separately (proper_simplex predicts one column at a time) and the two
    reconstructions are combined into one (x, y) trajectory for the
    spatial panel, since there is no multivariate Simplex call to make
    instead.

    Held-out window: [prediction_start, prediction_start+prediction_length)
    -- the same MDE validation segment (B) evaluate_smap_candidate() and
    every other MDE-side diagnostic in this file already uses; fit window
    is [0, prediction_start). Never touches the Generate/Forecast
    validation segment [B, end) -- same leakage boundary as everywhere
    else in this pipeline.

    Every target in net_result['targets'] is guaranteed to have D >= 1 by
    the time this runs: build_network() already calls partition_network()
    on this same graph, which raises if any target has zero drivers
    -- so there is no zero-driver-target case to guard here.

    Scoring: TARGETS are always continuous, so _metric_for() always
    resolves to target_metric here regardless of which node is being
    scored -- reported per method via edm._score(), alongside CAE (which
    stays the reference figure's own display convention, not replaced by
    it). Writes results_dir/mde_evaluate_{network_id}.png; returns a dict
    of the computed CAE/score values for the caller to log.'''
    targets = net_result['targets']
    graph = net_result['network']['Graph']
    combined = len(targets) > 1

    data_after_mde = network_data_frame(net_result)
    pool_columns = [c for c in data_after_mde.columns if c not in ('Time', *targets)]

    lib = [1, prediction_start]
    pred = [prediction_start + 1, prediction_start + prediction_length]
    pred_slice = slice(prediction_start, prediction_start + prediction_length)

    direct_drivers = {t: list(graph.predecessors(t)) for t in targets}
    driver_union = sorted(set().union(*direct_drivers.values()))
    D = len(driver_union)

    # ---- MDE: each target's own reconstruction from its own direct drivers ----
    mde_predicted, mde_cae, mde_score = {}, {}, {}
    for t in targets:
        out = edm.proper_simplex(
            data_after_mde, columns=direct_drivers[t], target=t,
            lib=lib, pred=pred, Tp=0, embedded=True,
            exclusionRadius=exclusion_radius,
            metric=target_metric, gammaDelta=gamma_delta,
        )
        predicted = out['predictions']['Predictions'].to_numpy()
        actual = out['predictions']['Observations'].to_numpy()
        mde_predicted[t] = predicted
        mde_cae[t] = _cae(actual, predicted)
        mde_score[t] = out['metrics'][target_metric]

    # ---- PCA / diffusion-map: fit on the network-wide driver pool, regress
    # onto the target(s) (jointly, in combined mode) ----------------------
    pool_lib = data_after_mde[pool_columns].iloc[:prediction_start].to_numpy()
    pool_pred = data_after_mde[pool_columns].iloc[pred_slice].to_numpy()
    target_lib = data_after_mde[targets].iloc[:prediction_start].to_numpy()
    target_pred = data_after_mde[targets].iloc[pred_slice].to_numpy()
    if not combined:
        target_lib = target_lib.ravel()
        target_pred = target_pred.ravel()

    # PCA and diffusion-map are optional, unsupervised comparison
    # baselines, not part of MDE's own causal claim -- guarded independently
    # per this codebase's own established "one degenerate item dead-ends
    # cleanly, it doesn't crash the batch" convention (already applied to
    # Stage 1/Stage 2's per-node exceptions, sec:stage1-parallelization;
    # rebuild_at_best_params()'s per-network Generate() divergence,
    # sec:validation; run_tp_vs_rho_sweep()'s per-target skip,
    # sec:diagnostics), which run_mde_evaluate() itself had never received.
    # A failure in either is logged with the exception text and the
    # target(s) it occurred on, and that baseline's own numbers/plot panel
    # are skipped -- the run continues to completion and archives per
    # sec:cluster-storage regardless of which (if either) baseline
    # actually produced a result.
    pca_predicted = pca_pred_proj = pca_cae = pca_score = None
    status, value = _run_baseline_with_timeout(_pca_baseline_worker, (pool_lib, pool_pred, target_lib, D))
    if status == 'ok':
        pca_predicted, pca_pred_proj = value
        pca_cae = _cae(target_pred, pca_predicted)
        pca_score = _score_baseline(target_pred, pca_predicted, targets, combined,
                                     target_metric, gamma_delta)
    elif status == 'timeout':
        print(f"{net_result['network_id']}: PCA baseline exceeded "
              f"{MDE_EVALUATE_BASELINE_TIMEOUT_SECONDS}s wall-clock bound for {targets} -- "
              f"terminated and skipped (MDE and any other baseline continue unaffected)",
              flush=True)
    else:
        print(f"{net_result['network_id']}: PCA baseline FAILED for {targets}: {value} -- "
              f"skipping this baseline (MDE and any other baseline continue unaffected)",
              flush=True)

    dmap_predicted = dmap_pred_proj = dmap_cae = dmap_score = None
    status, value = _run_baseline_with_timeout(
        _dmap_baseline_worker, (pool_lib, pool_pred, target_lib, D, dmap_k, dmap_epsilon))
    if status == 'ok':
        dmap_predicted, dmap_pred_proj = value
        dmap_cae = _cae(target_pred, dmap_predicted)
        dmap_score = _score_baseline(target_pred, dmap_predicted, targets, combined,
                                      target_metric, gamma_delta)
    elif status == 'timeout':
        print(f"{net_result['network_id']}: Diffusion-map baseline exceeded "
              f"{MDE_EVALUATE_BASELINE_TIMEOUT_SECONDS}s wall-clock bound for {targets} -- "
              f"terminated and skipped (MDE and any other baseline continue unaffected)",
              flush=True)
    else:
        print(f"{net_result['network_id']}: Diffusion-map baseline FAILED for {targets}: {value} -- "
              f"skipping this baseline (MDE and any other baseline continue unaffected)",
              flush=True)

    pca_summary = (f"PCA CAE={pca_cae:.2f} {target_metric}={pca_score:.3f}"
                   if pca_cae is not None else "PCA FAILED (skipped)")
    dmap_summary = (f"D-Map CAE={dmap_cae:.2f} {target_metric}={dmap_score:.3f}"
                    if dmap_cae is not None else "D-Map FAILED (skipped)")
    print(f"run_mde_evaluate ({net_result['network_id']}, D={D}): "
          f"MDE CAE={ {t: round(mde_cae[t], 2) for t in targets} } "
          f"{target_metric}={ {t: round(mde_score[t], 3) for t in targets} } | "
          f"{pca_summary} | {dmap_summary}", flush=True)

    # ---- Figure: 4 stacked panels, matching the reference -----------------
    maxN = min(max_traces_shown, D)
    x_pred = np.arange(prediction_start, prediction_start + prediction_length)

    fig = plt.figure(figsize=figsize)
    if combined:
        # Panel 1 is spatial (x-y position), not a time series -- cannot
        # share a time x-axis with panels 2-4, unlike the single-target case.
        gs = fig.add_gridspec(4, 1, height_ratios=[2, 1, 1, 1])
        ax0 = fig.add_subplot(gs[0])
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2], sharex=ax1)
        ax3 = fig.add_subplot(gs[3], sharex=ax1)
    else:
        (ax0, ax1, ax2, ax3) = fig.subplots(4, 1, sharex=True)
    axs = [ax0, ax1, ax2, ax3]

    axs[0].set_title(f"{net_result['network_id']}: train {lib}  predict {pred}")
    axs[1].set_ylabel('MDE', fontsize=12)
    axs[2].set_ylabel('Diffusion Map', fontsize=12)
    axs[3].set_ylabel('PCA', fontsize=12)
    lw = 2.0

    if combined:
        tx, ty = targets
        axs[0].plot(target_pred[:, 0], target_pred[:, 1], color='black', lw=lw,
                    label=f'{tx}/{ty}')
        mde_xy = np.column_stack([mde_predicted[tx], mde_predicted[ty]])
        axs[0].plot(mde_xy[:, 0], mde_xy[:, 1], lw=lw,
                     label=f'MDE   {mde_cae[tx] + mde_cae[ty]:.2f}')
        if pca_predicted is not None:
            axs[0].plot(pca_predicted[:, 0], pca_predicted[:, 1], lw=lw, label=f'PCA    {pca_cae:.2f}')
        if dmap_predicted is not None:
            axs[0].plot(dmap_predicted[:, 0], dmap_predicted[:, 1], lw=lw, label=f'D-Map {dmap_cae:.2f}')
        axs[0].set_xlabel(tx)
        axs[0].set_ylabel(ty)
        axs[0].set_ylabel('Predictions', fontsize=12)
    else:
        target = targets[0]
        axs[0].set_ylabel('Predictions', fontsize=12)
        if dmap_predicted is not None:
            axs[0].plot(x_pred, dmap_predicted, lw=lw, label=f'D-Map {dmap_cae:.2f}')
        if pca_predicted is not None:
            axs[0].plot(x_pred, pca_predicted, lw=lw, label=f'PCA    {pca_cae:.2f}')
        axs[0].plot(x_pred, mde_predicted[target], lw=lw, label=f'MDE   {mde_cae[target]:.2f}')
        axs[0].plot(x_pred, target_pred, color='black', lw=lw, label=target)
    axs[0].legend(title='CAE', bbox_to_anchor=(1.0, 1.0), loc='upper left')

    # driver_union, not a per-target loop -- a driver shared by more than
    # one target in combined mode (e.g. a common upstream node) would
    # otherwise get plotted and labeled once per target it feeds, showing
    # the identical trace twice under the same name.
    for driver in driver_union[:maxN]:
        axs[1].plot(x_pred, data_after_mde[driver].iloc[pred_slice], lw=lw, label=driver)
    axs[1].legend(title='MDE', ncol=1, bbox_to_anchor=(1.0, 1.0), loc='upper left')

    # Panels 2/3: same skip-not-crash handling as panel 0 above -- a failed
    # baseline has no projection to plot, so the panel is left empty with
    # an explicit label rather than raising on a None subscript.
    if dmap_pred_proj is not None:
        for col in range(maxN):
            axs[2].plot(x_pred, dmap_pred_proj[:, col], lw=lw, label=str(col))
        axs[2].legend(title='Diffusion Map', ncol=1, bbox_to_anchor=(1.0, 1.0), loc='upper left')
    else:
        axs[2].text(0.5, 0.5, 'Diffusion Map baseline FAILED (see log)', ha='center', va='center',
                    transform=axs[2].transAxes, fontsize=9, color='gray')

    if pca_pred_proj is not None:
        for col in range(maxN):
            axs[3].plot(x_pred, pca_pred_proj[:, col], lw=lw, label=str(col))
        axs[3].legend(title='PCA', ncol=1, bbox_to_anchor=(1.0, 1.0), loc='upper left')
    else:
        axs[3].text(0.5, 0.5, 'PCA baseline FAILED (see log)', ha='center', va='center',
                    transform=axs[3].transAxes, fontsize=9, color='gray')

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.3 if combined else 0.04)
    fig.savefig(results_dir / f"mde_evaluate_{net_result['network_id']}.png",
                dpi=300, bbox_inches='tight')
    plt.close(fig)

    return {
        'D': D, 'direct_drivers': direct_drivers,
        'mde_cae': mde_cae, 'mde_score': mde_score,
        'pca_cae': pca_cae, 'pca_score': pca_score,
        'dmap_cae': dmap_cae, 'dmap_score': dmap_score,
    }
