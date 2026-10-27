"""Free-running the network, and scoring how well it reproduces reality.

This is step 6 of the pipeline -- the payoff. Everything so far fitted a model;
this runs it forward on its own and asks whether the result looks like the
recording it never saw.

    sweep_generate_start()  Generate() has to start somewhere. Which row it
                            starts from changes the result, so this tries
                            several offsets and keeps the best-matching one.
    rebuild_at_best_params()
                            reruns Generate() at the winning per-node theta
                            (theta.py) and the winning start row, and writes
                            the per-node scores to CSV. Guarded per network: a
                            network that fails here is logged and skipped
                            rather than taking the job down.
    run_forecast_diagnostic()
                            the easier comparison. Forecast() predicts one step
                            ahead with the real data available at each step,
                            so it isolates one-step accuracy from the error
                            accumulation a long free run suffers.
    gmn_fidelity_rho()      generative fidelity: can the GENERATED series
                            predict the real one? A fresh embedding of the
                            generated output cross-maps to the real signal on a
                            50/50 split of the generated window. Passing this
                            means the model reproduced the dynamics, not just
                            the numbers.
    plot_fidelity()         that score, per node, as a figure.

Free run vs one-step: Generate() feeds each prediction back in as input, so
errors compound and the run drifts; Forecast() never does. A model can be good
at one and poor at the other, so both are reported.

Depends on: theta, gmn_build, gmn_embedding, caching, tsls_data, progress;
edm, gmn, numpy, pandas, matplotlib.
Imported by: pipeline, the entry scripts.
"""

import pickle
import time

import matplotlib
matplotlib.use('Agg')          # cluster jobs have no display
import edm
import gmn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mde_gmn.caching import build_start_sweep_fingerprint, find_cached, save_json_to_cache
from mde_gmn.gmn_build import build_gmn_config
from mde_gmn.gmn_embedding import network_max_embedding_offset
from mde_gmn.progress import _log_progress
from mde_gmn.theta import evaluate_smap_candidate
from mde_gmn.tsls_data import _metric_for, network_data_frame


def sweep_generate_start(gmn_runs, theta_search_results, prediction_start, prediction_length,
                          results_dir, cores, generate_length, start_offsets,
                          target_metric, gamma_delta, exclusion_radius=0,
                          reuse_cache=False, prev_runs_dir=None, run_cache_dir=None,
                          fingerprints=None, generate_validation_start=None,
                          progress_log_path=None):
    '''Finds, per network, which row Generate() should start free-running from
    to best match the real held-out data. Scored with the same
    evaluate_smap_candidate() the theta search uses, so the two searches
    optimise the same thing. Returns {network_id: best_generate_start_row}.'''
    best_starts = {}
    for network_id, run in gmn_runs.items():
        if len(start_offsets) == 1:
            best_starts[network_id] = generate_validation_start + start_offsets[0]
            continue

        net_result = run['net_result']
        node_theta = theta_search_results[network_id]['best_node_theta']
        network_fingerprint = fingerprints[network_id] if fingerprints else None

        sweep_fingerprint = None
        cache_filename = f'generate_start_sweep_{network_id}.json'
        if reuse_cache and network_fingerprint is not None:
            sweep_fingerprint = build_start_sweep_fingerprint(
                network_fingerprint, node_theta, start_offsets, generate_length, exclusion_radius,
            )
            hit = find_cached(prev_runs_dir, cache_filename, sweep_fingerprint)
            if hit is not None:
                entry, source = hit
                scores = {int(k): v for k, v in entry['scores'].items()}
                best_offset = entry['best_offset']
                best_starts[network_id] = generate_validation_start + best_offset
                print(f'{network_id}: TSLS_MDE_GMN_REUSE_BEST_PARAMS matched cached start sweep '
                      f'{source} -- skipping sweep entirely (best offset {best_offset}, '
                      f'score={scores.get(best_offset, float("nan")):.4f})', flush=True)

                fig, ax = plt.subplots(figsize=(7, 4))
                ax.plot(start_offsets, [scores.get(o, np.nan) for o in start_offsets], marker='o')
                ax.axvline(best_offset, color='gray', ls='--', lw=0.8)
                ax.set_xlabel('Start offset from GENERATE_VALIDATION_START (rows)')
                ax.set_ylabel('Generate() rho (mean over targets)')
                ax.set_title(f'Generate() start-point sweep -- {network_id} (cached)')
                fig.tight_layout()
                fig.savefig(results_dir / f'generate_start_sweep_{network_id}.png', dpi=300)
                plt.close(fig)
                continue
            print(f'{network_id}: TSLS_MDE_GMN_REUSE_BEST_PARAMS set but no matching cached start '
                  f'sweep found under {prev_runs_dir} -- running sweep.', flush=True)

        checkpoint_file = results_dir / f'generate_start_checkpoint_{network_id}.pkl'
        scores = {}
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                ckpt = pickle.load(f)
            if (ckpt.get('start_offsets') == start_offsets
                    and ckpt.get('generate_length') == generate_length):
                scores = ckpt['scores']
                print(f'{network_id}: resumed start-sweep checkpoint, '
                      f'{len(scores)}/{len(start_offsets)} offsets already done', flush=True)

        for offset in start_offsets:
            if offset in scores and not np.isnan(scores[offset]):
                continue

            generate_start = generate_validation_start + offset
            try:
                score, _, _ = evaluate_smap_candidate(
                    net_result, prediction_start, prediction_length,
                    node_theta, f'startsweep_offset{offset}', results_dir, cores,
                    target_metric, gamma_delta,
                    generate_length=generate_length, generate_start=generate_start,
                    exclusion_radius=exclusion_radius,
                )
            except Exception as exc:
                print(f'{network_id}: start offset {offset} FAILED: {exc!r} -- scoring as NaN', flush=True)
                score = np.nan

            scores[offset] = score
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({'start_offsets': start_offsets,
                             'generate_length': generate_length, 'scores': scores}, f)
            print(f'{network_id}: start offset {offset:5d} (row {generate_start}) score={score:.4f}', flush=True)
            _log_progress(progress_log_path,
                          f'{network_id}: generate-start offset {offset} done '
                          f'({len(scores)}/{len(start_offsets)}), score={score:.4f}')

        valid_offsets = [o for o in start_offsets if not np.isnan(scores.get(o, np.nan))]
        best_offset = max(valid_offsets, key=lambda o: scores[o]) if valid_offsets else start_offsets[0]
        best_starts[network_id] = generate_validation_start + best_offset
        print(f'{network_id}: best generate start = row {best_starts[network_id]} '
              f'(offset {best_offset}, score={scores.get(best_offset, float("nan")):.4f})', flush=True)

        if sweep_fingerprint is not None:
            save_json_to_cache(
                cache_filename,
                {
                    'fingerprint': sweep_fingerprint,
                    'scores': {str(k): v for k, v in scores.items()},
                    'best_offset': best_offset,
                },
                results_dir, run_cache_dir,
            )

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(start_offsets, [scores.get(o, np.nan) for o in start_offsets], marker='o')
        ax.axvline(best_offset, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('Start offset from GENERATE_VALIDATION_START (rows)')
        ax.set_ylabel('Generate() rho (mean over targets)')
        ax.set_title(f'Generate() start-point sweep -- {network_id}')
        fig.tight_layout()
        fig.savefig(results_dir / f'generate_start_sweep_{network_id}.png', dpi=300)
        plt.close(fig)

    return best_starts

def rebuild_at_best_params(gmn_runs, theta_search_results, prediction_start, prediction_length,
                            results_dir, cores, metric, target_metric, gamma_delta,
                            generate_length=None, generate_starts=None,
                            exclusion_radius=0, generate_validation_start=None,
                            progress_log_path=None):
    '''Reruns Generate() at the winning per-node theta found by
    run_theta_search(), and the winning start row found by
    sweep_generate_start(). Guarded per network: a failure here is logged and
    that network is skipped (run['G'] stays unset) rather than taking the whole
    job down.

    It also REPORTS every node's score, not just the targets': printed, and
    written to per_node_scores_<network_id>.csv. That is reporting only and
    feeds back into nothing -- what the searches optimise is the mean over the
    targets alone (evaluate_smap_candidate()). Scores in that file are not
    comparable across rows, because each node is scored with its own metric
    (correlation for targets, f1 for neurons), so each row names its metric and
    the score column must not be summed or averaged.'''
    for network_id, run in gmn_runs.items():
        net_result = run['net_result']
        node_theta = theta_search_results[network_id]['best_node_theta']
        generate_start = generate_starts[network_id] if generate_starts else generate_validation_start

        generate_rebuild_start = time.perf_counter()
        _log_progress(progress_log_path, f'{network_id}: FINAL Generate() rebuild START')
        try:
            config_file = build_gmn_config(
                net_result, prediction_start, prediction_length, results_dir,
                mode='Generate', node_theta_overrides=node_theta,
                variant='best', generate_length=generate_length, generate_start=generate_start,
                exclusion_radius=exclusion_radius,
            )
            G = gmn.GMN( configFile = str(config_file), cores = cores )
            G.Generate()
        except Exception as exc:
            print(f'{network_id}: FINAL Generate() rebuild FAILED: {exc!r} -- '
                  f'skipping this network (no plots/fidelity/output for it)', flush=True)
            _log_progress(
                progress_log_path,
                f'{network_id}: FINAL Generate() rebuild END -- FAILED '
                f'{time.perf_counter() - generate_rebuild_start:.1f}s: {exc!r}')
            continue
        _log_progress(progress_log_path,
                      f'{network_id}: FINAL Generate() rebuild END -- '
                      f'{time.perf_counter() - generate_rebuild_start:.1f}s')

        run['G'] = G
        run['best_node_theta'] = node_theta
        run['generate_start'] = generate_start

        actual_df = network_data_frame(net_result)
        generated = G.DataOut.reset_index(drop=True)
        nodes = [c for c in generated.columns if c != 'Time']

        print(f"{network_id} Generate() score, held-out window (start row {generate_start}):", flush=True)
        per_node_rows = []
        for node in nodes:
            node_metric = _metric_for(node, net_result['targets'], metric, target_metric)
            a = actual_df[node].to_numpy()[generate_start:generate_start + len(generated)]
            r = edm._score(a, generated[node].to_numpy(), node_metric, gammaDelta=gamma_delta)
            tag = 'TARGET' if node in net_result['targets'] else 'driver'
            print(f"  {node:20s} {tag:7s} {node_metric}={r:7.3f}", flush=True)
            per_node_rows.append({'node': node, 'role': tag, 'metric': node_metric, 'score': r})

        # The same data the loop above printed, written to a file so it
        # survives past the log. Metrics differ per row (correlation for
        # targets, f1 for neurons) -- do not sum or average the score column,
        # it mixes incompatible scales.
        pd.DataFrame(per_node_rows, columns=['node', 'role', 'metric', 'score']).to_csv(
            results_dir / f'per_node_scores_{network_id}.csv', index=False)

def run_forecast_diagnostic(net_result, G, prediction_start, prediction_length,
                             node_theta, results_dir, cores, save_path,
                             metric, target_metric, gamma_delta,
                             generate_length=None, generate_start=None, nodes_per_figure=10,
                             exclusion_radius=0, plot_range_mode='full',
                             generate_validation_start=None, total_rows=None):
    '''Runs the Forecast() diagnostic and draws the one
    actual-vs-Generate()-vs-Forecast() figure for this network.

    plot_range_mode='full' plots the whole recording; 'windowed' plots only the
    generated window. Returns (None, {}) instead of raising if the Forecast()
    build or call fails, so one bad network does not lose the others.'''
    generate_length = generate_length if generate_length is not None else prediction_length
    generate_start = generate_start if generate_start is not None else generate_validation_start

    if plot_range_mode == 'windowed':
        forecast_compute_length = min(2 * generate_length, max(total_rows - generate_start, 0))
    else:
        forecast_compute_length = generate_length

    try:
        max_offset = network_max_embedding_offset(net_result['network']['Graph'])
        config_file = build_gmn_config(
            net_result, prediction_start, prediction_length, results_dir, mode='Forecast',
            node_theta_overrides=node_theta, generate_length=forecast_compute_length,
            generate_start=generate_start, max_offset=max_offset,
            exclusion_radius=exclusion_radius,
        )
        Gf = gmn.GMN( configFile = str(config_file), cores = cores )
        Gf.Forecast()
    except Exception as exc:
        print(f"{net_result['network_id']}: Forecast() diagnostic FAILED: {exc!r} -- "
              f"skipping its plot/rho for this network (Generate()-based outputs are unaffected)",
              flush=True)
        return None, {}

    actual_df = network_data_frame(net_result)
    forecast_out = Gf.DataOut.dropna().reset_index(drop=True)
    time_idx = forecast_out['Time'].to_numpy().astype(int)
    nodes = [c for c in forecast_out.columns if c != 'Time']

    generate_out = G.DataOut.reset_index(drop=True)
    generate_time = np.arange(generate_start, generate_start + len(generate_out))

    if plot_range_mode == 'windowed':
        plot_start = generate_start - min(2 * generate_length, generate_start)
        plot_end = generate_start + forecast_compute_length
    else:
        plot_start = 0
        plot_end = generate_start + generate_length

    rhos = {}
    n_parts = (len(nodes) + nodes_per_figure - 1) // nodes_per_figure
    for part in range(n_parts):
        part_nodes = nodes[part * nodes_per_figure:(part + 1) * nodes_per_figure]
        fig, axes = plt.subplots(len(part_nodes), 1, figsize=(11, 2.2 * len(part_nodes)), sharex=False)
        if len(part_nodes) == 1:
            axes = [axes]

        for ax, node in zip(axes, part_nodes):
            node_metric = _metric_for(node, net_result['targets'], metric, target_metric)
            actual = actual_df[node].to_numpy()
            forecast_pred = forecast_out[node].to_numpy()
            rhos[node] = edm._score(actual[time_idx], forecast_pred, node_metric, gammaDelta=gamma_delta)

            ax.plot(np.arange(plot_start, plot_end), actual[plot_start:plot_end],
                     color='tab:blue', lw=1, alpha=0.6, label='Lib/actual')
            if node in generate_out.columns:
                ax.plot(generate_time, generate_out[node].to_numpy(),
                        color='tab:red', lw=1, label='Generate()')
            ax.plot(time_idx, forecast_pred, color='tab:purple', lw=1.2, ls=':', label='Forecast()')
            ax.axvline(generate_start, color='gray', ls='--', lw=0.8)
            # Clip the view to the intended window explicitly. A second
            # layer of defence: any stray out-of-window data would otherwise
            # stretch matplotlib's autoscale and hide the mislabelling this
            # section exists to avoid.
            ax.set_xlim(plot_start, plot_end)
            ax.set_title(f"{node}   Forecast {node_metric}={rhos[node]:.3f}", fontsize=10)
            ax.legend(fontsize=7, loc='upper right')

        fig.suptitle(f"Generate() vs Forecast() vs actual -- {net_result['network_id']} "
                     f"(nodes {part * nodes_per_figure + 1}-"
                     f"{part * nodes_per_figure + len(part_nodes)} of {len(nodes)})", fontsize=11)
        fig.tight_layout()
        part_path = save_path.with_name(f'{save_path.stem}_part{part + 1:02d}{save_path.suffix}')
        fig.savefig(part_path, dpi=300)
        plt.close(fig)

    return Gf, rhos

def gmn_fidelity_rho(net_result, G, generate_start, metric, target_metric, gamma_delta,
                      default_E=3, tau=-1):
    '''Fig-4e-style generative fidelity: per node, a fresh proper_simplex
    predicts the real actual series *from* the generated series on a 50/50
    split of the generated window. Scored with _metric_for()/proper_simplex
    rather than pyEDM's plain Simplex/ComputeError (see config.py's METRIC
    entry for why TSLS signals need the TSLS-aware version). E defaults to
    node_dimensions, build_network()'s per-node estimate, when it computed
    one.'''
    actual_df = network_data_frame(net_result)
    generated = G.DataOut.reset_index(drop=True)
    nodes = [c for c in generated.columns if c != 'Time']
    n_gen = len(generated)
    split = n_gen // 2
    node_dimensions = net_result.get('node_dimensions', {})

    fidelity_rho = {}
    for node in nodes:
        node_metric = _metric_for(node, net_result['targets'], metric, target_metric)
        gen_vals = generated[node].to_numpy()
        actual_vals = actual_df[node].to_numpy()[generate_start: generate_start + n_gen]
        pair_df = pd.DataFrame({
            'Time': np.arange(n_gen),
            f'{node}_gen': gen_vals,
            f'{node}_actual': actual_vals,
        })
        E = max(1, min(node_dimensions.get(node, default_E), split - 1))
        try:
            out = edm.proper_simplex(
                pair_df, columns=f'{node}_gen', target=f'{node}_actual',
                E=E, tau=tau, Tp=0,
                lib=[1, split], pred=[split + 1, n_gen],
                embedded=False, metric=node_metric, gammaDelta=gamma_delta,
            )
            fidelity_rho[node] = out['metrics'][node_metric]
        except Exception:
            fidelity_rho[node] = np.nan
    return fidelity_rho

def plot_fidelity(rho, network_id, save_path):
    order = sorted(rho, key=lambda n: (np.isnan(rho[n]), -(rho[n] if not np.isnan(rho[n]) else 0)))
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(range(len(order)), [rho[n] for n in order], color='tab:green')
    ax.axhline(0, color='gray', lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=90, fontsize=7)
    ax.set_ylabel('rho (Simplex prediction of actual from generated embedding)')
    ax.set_title(f'GMN generative fidelity (Fig. 4e-style) -- {network_id}')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
