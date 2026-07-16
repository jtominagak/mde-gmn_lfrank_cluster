'''Batch/cluster port of notebooks/simple-mde-gmn.ipynb, executed by
scripts/batch/simple-mde-gmn.sbatch. Every parameter is read from an
MDE_GMN_* environment variable (set by the sbatch script's MODIFIABLE
PARAMETERS block) rather than edited here directly.

This mirrors the notebook cell-for-cell (see that file for the narrative
explanation of *why* each step works the way it does -- docstrings below
are trimmed to what's specific to running headless/non-interactively):

  1. MDE forward-selection (skipped entirely if MODE='GMN')
  2. GMN network construction: CCM interaction matrix (data-driven E/tau
     grid search) + per-node driver-count estimate (EmbedDimension) +
     apps.CreateNetwork, with a driver-ranking bug workaround
  3. GMN node generation: SMap uniformly for every node (not Simplex, not
     a per-node mix -- see build_gmn_config()'s docstring for why)
  4. Empirical (E, tau, per-node theta) search maximizing free-running
     Generate() fidelity against the real held-out targets
  5. Final GMN rebuild at the winning parameters + Forecast() diagnostic
     + Fig-4e-style generative fidelity metric
  6. MDE-only diagnostics (Evaluate vs PCA/Diffusion Map, Tp-vs-rho,
     univariate E/tau grid) -- skipped if MODE='GMN', same as the notebook

Differences from the notebook, all a consequence of running headless
rather than in a live Jupyter kernel:
  - apps.InteractionMatrix is invoked via subprocess.run instead of
    get_ipython().system() (get_ipython() only exists inside a real
    kernel and raises NameError otherwise).
  - Every plt.show() is replaced with plt.close(fig) after the
    corresponding savefig -- there's no display to show anything to, and
    under MPLBACKEND=Agg plt.show() is already a documented no-op, but
    closing explicitly avoids accumulating open figures across a long run.
  - All output (data CSVs, .cfg files, checkpoints, figures) goes under
    one flat RESULTS_DIR rather than the notebook's separate
    results/vs/figures/ split -- simpler for the sbatch script's
    `cp -av simple_mde_gmn_outputs/. "$OUT_DIR"/` to copy back verbatim.
  - CORES is threaded through everywhere a *_cores/numProcess parameter
    exists (dimx.MDE, apps.InteractionMatrix -cr, pyEDM.EmbedDimension/
    PredictNonlinear's numProcess, gmn.GMN's cores -- the last one is
    currently a no-op inside the installed gmn package, confirmed by
    grepping its source for args.cores; passed anyway in case a future
    gmn version starts using it) instead of relying on each tool's own
    hardcoded default, which doesn't know how many cores SLURM actually
    allocated.

KNOWN CAVEAT (found empirically running the MODE='GMN' interaction-matrix
step against the real dataset): apps.InteractionMatrix's own
ProcessPoolExecutor (sized by -cr/CORES) and pyEDM.CCM()'s own *internal*
multiprocessing nest inside each other -- with CORES=8 this produced
~90 concurrent processes contending for the cores actually available,
not 8. This isn't a hang (confirmed via per-process CPU-time deltas: work
keeps accumulating, just slower than a naive CORES-fold speedup would
predict) but it does mean interaction-matrix wall time won't scale
linearly with -cr, and can badly oversubscribe a shared/cgroup-limited
compute node. If a job runs far longer than expected, check actual CPU
utilization before assuming something is broken.
'''
import os
import pickle
import subprocess
import time
from configparser import ConfigParser
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx
import numpy as np
import pandas as pd

import dimx as dx
import gmn
from gmn.Common import FunctionType
from apps.CreateNetwork import CreateNetwork
from pyEDM import (
    Simplex, SMap, CCM, ComputeError, Embed, EmbedDimension,
    PredictNonlinear, PredictInterval,
)


def _fixed_node_forecast(self):
    '''Corrected copy of gmn.Node.Forecast (gmn/Forecast.py), fixing two
    bugs found empirically in the installed package's SMap branch -- the
    only branch this pipeline exercises, since every node uses SMap
    uniformly (see build_gmn_config's docstring):

    (1) that module calls Embed() without ever importing pyEDM.Embed
    (confirmed: gmn/Forecast.py only does `from .Common import *`, while
    gmn/Generate.py's identical SMap branch correctly imports it) --
    raises NameError on the very first Forecast() call against any SMap
    node.

    (2) `val` is assigned as a 2-tuple (time, predictions) and then
    immediately checked via `val.max() > 1E15` for divergence -- tuples
    have no .max() method, so this raises AttributeError unconditionally,
    not just on genuine divergence (confirmed: this fires as soon as (1)
    is worked around, on ordinary non-diverging output). Fixed to check
    the predictions Series specifically (val[1]).

    Every other branch is copied verbatim from the installed
    gmn/Forecast.py (unused by this pipeline, since every node is SMap,
    but kept intact rather than trimmed so this patch doesn't silently
    change behavior for any future non-SMap use). Monkeypatched onto
    gmn.Node (== gmn.Node.Node -- gmn/__init__.py's `from .Node import
    Node` shadows the submodule name with the class in gmn's own
    namespace) rather than editing the installed package, to keep this
    script self-contained on other machines/the cluster.'''
    if self.args.DEBUG :
        print( '-----> Node:Forecast() : ',self.FunctionType.name, self.name)
        print( self.data.tail( 2 ), flush = True );

    Parameters = self.Parameters
    data       = self.data

    if self.args.DEBUG :
        print( '  data : shape: ', data.shape );
        print( data.tail( 3 ) ); print('', flush = True)

    if self.FunctionType.value == FunctionType.Simplex.value :
        if len( Parameters.lib ) and not Parameters.lib.isspace():
            lib = Parameters.lib
        else:
            lib = "1 %d" % self.libEnd_i

        if len( Parameters.pred ) and not Parameters.pred.isspace():
            pred = Parameters.pred
        else:
            pred = "%d %d" % ( data.shape[0] - 1, data.shape[0] )

        S = self.Function( dataFrame       = data,
                           lib             = lib,
                           pred            = pred,
                           E               = Parameters.E,
                           Tp              = Parameters.Tp,
                           knn             = Parameters.knn,
                           tau             = Parameters.tau,
                           exclusionRadius = Parameters.exclusionRadius,
                           columns         = Parameters.columns,
                           target          = Parameters.target,
                           embedded        = Parameters.embedded,
                           validLib        = Parameters.validLib,
                           generateSteps   = Parameters.generateSteps )

        val = ( S.iloc[:,0], S['Predictions'] )

    elif self.FunctionType.value == FunctionType.SMap.value :
        df = Embed( dataFrame = data, E = Parameters.E,
                    tau = Parameters.tau, columns = Parameters.columns )

        df = df.dropna()

        df.insert( 0, data.columns[0], data.iloc[:,0] )

        if len( Parameters.lib ) and not Parameters.lib.isspace():
            lib = Parameters.lib
        else :
            lib  = "1 %d" % self.libEnd_i

        if len( Parameters.pred ) and not Parameters.pred.isspace():
            pred = Parameters.pred
        else:
            pred = "%d %d" % ( df.shape[0] - 1, df.shape[0] )

        S = self.Function( dataFrame       = df,
                           lib             = lib,
                           pred            = pred,
                           E               = Parameters.E,
                           Tp              = Parameters.Tp,
                           knn             = Parameters.knn,
                           tau             = Parameters.tau,
                           theta           = Parameters.theta,
                           exclusionRadius = Parameters.exclusionRadius,
                           columns         = df.columns[1:-1],
                           target          = Parameters.target + '(t-0)',
                           solver          = None,
                           embedded        = Parameters.embedded,
                           validLib        = Parameters.validLib,
                           generateSteps   = Parameters.generateSteps )

        val = ( S['predictions'].iloc[:,0], S['predictions']['Predictions'] )

        if val[1].max() > 1E15 :  # FIX: was val.max() -- val is the tuple above
            raise RuntimeError( "Forecast:SMap() Node: " + self.name +\
                                "   Divergence detected (>1E15)" )

    elif self.FunctionType.value == FunctionType.kedmSimplex.value :
        raise RuntimeError( "Forecast kedmSimplex function not available" )

    elif self.FunctionType.value == FunctionType.kedmSMap.value :
        raise RuntimeError( "Forecast kedmSMap function not available" )

    elif self.FunctionType.value == FunctionType.Linear.value :
        raise RuntimeError( "Forecast Linear function not available" )

    elif self.FunctionType.value == FunctionType.SVR.value :
        raise RuntimeError( "Forecast SVR function not available" )

    elif self.FunctionType.value == FunctionType.knn.value :
        raise RuntimeError( "Forecast knn function not available" )

    if self.args.DEBUG :
        print( self.name, "val:", val )
        print( '<----- Node:Forecast() : ',self.FunctionType.name,self.name )
        print('', flush = True)

    return val


gmn.Node.Forecast = _fixed_node_forecast


# =============================================================================
# Configuration -- every value overridable via MDE_GMN_* environment
# variables; see scripts/batch/simple-mde-gmn.sbatch's MODIFIABLE
# PARAMETERS block for the canonical list with descriptions.
# =============================================================================

def _env_bool(name, default):
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ('1', 'true', 'yes', 'on')


def _env_list(name, default):
    val = os.environ.get(name)
    if val is None:
        return list(default)
    return [x.strip() for x in val.split(',') if x.strip()]


def _env_float_list(name, default):
    val = os.environ.get(name)
    if val is None:
        return list(default)
    return [float(x.strip()) for x in val.split(',') if x.strip()]


def load_config():
    cfg = {}

    # ---- Data ---------------------------------------------------------
    # NOTE: unlike the notebook (DATA_DIR = '../data/transfer_.../'), the
    # cluster deployment expects data files copied flat into DATA_DIR by
    # the sbatch script -- no nested extraction-folder name.
    cfg['DATA_DIR'] = Path(os.environ.get('MDE_GMN_DATA_DIR', '../../data'))
    cfg['DATA_FILE'] = os.environ.get('MDE_GMN_DATA_FILE', 'J1620210605_epoch_8_freq_4_Hz.csv')
    cfg['RESULTS_DIR'] = Path(os.environ.get('MDE_GMN_RESULTS_DIR', '../../results'))
    cfg['RESULTS_DIR'].mkdir(parents=True, exist_ok=True)

    # ---- Overall pipeline shape ---------------------------------------
    cfg['MODE'] = os.environ.get('MDE_GMN_MODE', 'MDE-GMN')  # 'MDE-GMN' | 'GMN'
    if cfg['MODE'] not in ('MDE-GMN', 'GMN'):
        raise ValueError(f"Unknown MDE_GMN_MODE: {cfg['MODE']!r}")
    cfg['GMN_MODE'] = os.environ.get('MDE_GMN_GMN_MODE', 'combined')  # 'combined' | 'separate'

    cfg['TARGETS'] = _env_list('MDE_GMN_TARGETS', ['head_position_x', 'head_position_y'])
    cfg['SECOND_TARGET'] = cfg['TARGETS'][1] if len(cfg['TARGETS']) > 1 else None

    cfg['PREDICTION_START'] = int(os.environ.get('MDE_GMN_PREDICTION_START', 3500))
    cfg['PREDICTION_LENGTH'] = int(os.environ.get('MDE_GMN_PREDICTION_LENGTH', 300))

    # ---- MDE (variable selection; unused when MODE='GMN') -------------
    cfg['MDE_D'] = int(os.environ.get('MDE_GMN_D', 20))
    cfg['EMBED_DIM_RHO_MIN'] = float(os.environ.get('MDE_GMN_EMBED_DIM_RHO_MIN', 0.2))
    cfg['CROSS_MAP_RHO_MIN'] = float(os.environ.get('MDE_GMN_CROSS_MAP_RHO_MIN', 0.2))
    # Candidate observable pool: first N channels of each region. Default
    # 999 (effectively uncapped -- matches the current notebook's default
    # and Park et al. 2026's own full-channel-set construction); cap this
    # down for a fast test job.
    cfg['REGION_CHANNEL_CAP'] = int(os.environ.get('MDE_GMN_REGION_CHANNEL_CAP', 999))

    # ---- E/tau/theta search ranges -------------------------------------
    # E >= 2 always: every GMN node uses SMap uniformly (see
    # build_gmn_config's docstring), and SMap's predictor-exclusion
    # convention needs E >= 2 even for leaf/undriven nodes.
    cfg['E_RANGE'] = range(
        int(os.environ.get('MDE_GMN_E_MIN', 3)),
        int(os.environ.get('MDE_GMN_E_MAX', 16)),
        int(os.environ.get('MDE_GMN_E_STEP', 2)),
    )
    cfg['TAU_RANGE'] = range(
        int(os.environ.get('MDE_GMN_TAU_MIN', -15)),
        int(os.environ.get('MDE_GMN_TAU_MAX', 0)),
        int(os.environ.get('MDE_GMN_TAU_STEP', 2)),
    )
    cfg['THETA_RANGE'] = _env_float_list(
        'MDE_GMN_THETA_RANGE',
        [0.01, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7, 8, 9],
    )
    cfg['COARSE_THETA_RANGE'] = cfg['THETA_RANGE'][::3]
    cfg['SMAP_SEARCH_MODE'] = os.environ.get('MDE_GMN_SMAP_SEARCH_MODE', 'coarse_retune')  # 'coarse_retune' | 'joint'

    # ---- MDE-only diagnostics (skipped when MODE='GMN') ----------------
    cfg['MAX_TP'] = int(os.environ.get('MDE_GMN_MAX_TP', 20))
    cfg['GRID_RHO_THRESHOLD'] = float(os.environ.get('MDE_GMN_GRID_RHO_THRESHOLD', 0.9))
    cfg['RUN_MDE_DIAGNOSTICS'] = _env_bool('MDE_GMN_RUN_MDE_DIAGNOSTICS', True)

    # ---- Parallelism ----------------------------------------------------
    # On HPC this should be SLURM_CPUS_PER_TASK, not left at each tool's
    # own package default (dimx: 5, apps.InteractionMatrix: 8, pyEDM
    # EmbedDimension/PredictNonlinear: their own internal default) -- see
    # module docstring's KNOWN CAVEAT for why this doesn't scale linearly.
    cfg['CORES'] = int(os.environ.get('MDE_GMN_CORES', os.cpu_count() or 1))
    # Deliberately NOT the same as CORES -- see build_network()'s
    # docstring: apps.InteractionMatrix's -cr and pyEDM.CCM()'s own
    # per-pair internal pool multiply rather than coordinate, so this
    # stays small regardless of how large CORES is.
    cfg['INTERACTION_MATRIX_OUTER_CORES'] = int(os.environ.get('MDE_GMN_INTERACTION_MATRIX_OUTER_CORES', 2))

    cfg['TARGET_COLORS'] = {t: plt.get_cmap('tab10')(i % 10) for i, t in enumerate(cfg['TARGETS'])}

    return cfg


# =============================================================================
# MDE (mirrors notebook cells 2-5)
# =============================================================================

def region_columns(df, prefix, n):
    candidates = [f'{prefix}{i}' for i in range(n)]
    return [c for c in candidates if c in df.columns]


def observable_columns(df, cap):
    return (
        region_columns(df, 'CA1_', cap)
        + region_columns(df, 'mPFC_', cap)
        + region_columns(df, 'OFC_', cap)
    )


def run_mde_for_target(target, df, observable_columns, prediction_start, prediction_length,
                        mde_d, embed_dim_rho_min, cross_map_rho_min, cores, results_dir):
    '''Run dimx.MDE's forward selection for a single target variable.
    See notebooks/simple-mde-gmn.ipynb cell 2's run_mde_for_target for the
    full rationale (identical logic, just cores threaded through).'''
    mde_df = df[[target] + observable_columns].iloc[:prediction_start + prediction_length]

    m = dx.MDE( mde_df, target = target, D = mde_d,
                removeColumns = [target],
                lib  = [1, prediction_start],
                pred = [prediction_start + 1, prediction_start + prediction_length],
                embedDimRhoMin = embed_dim_rho_min,
                crossMapRhoMin = cross_map_rho_min,
                cores = cores )

    m.Run()

    m.MDEOut.to_csv(results_dir / f'mde_output_{target}.csv', index=False)

    selected_columns = list(m.MDEcolumns)
    print(f'MDE selected columns for {target}:', selected_columns, flush=True)

    return {
        'target': target,
        'mde_df': mde_df,
        'm': m,
        'SELECTED_COLUMNS': selected_columns,
    }


def plot_mde_rho_vs_dimension(mde_results, targets, target_colors, save_path):
    fig, ax = plt.subplots(figsize=(9, 5))
    for target in targets:
        mde_out = mde_results[target]['m'].MDEOut
        dims = np.arange(1, len(mde_out) + 1)
        ax.plot(dims, mde_out['rho'], marker='o', label=target, color=target_colors[target])
    ax.set_xlabel('Embedding dimension')
    ax.set_ylabel('Cross-map rho')
    ax.set_title('MDE rho vs. dimension, per target')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


def plot_3d_mde_projection(target, second_target, mde_results, df,
                            prediction_start, prediction_length, save_path):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers the '3d' projection

    obs = mde_results[target]['SELECTED_COLUMNS'][:2]
    if len(obs) < 2:
        print(f'{target}: fewer than 2 MDE-selected columns, skipping 3D projection.', flush=True)
        return

    window = df.iloc[:prediction_start + prediction_length]
    x = window[target].to_numpy()
    y = window[obs[0]].to_numpy()
    z = window[obs[1]].to_numpy()

    if second_target and second_target in window.columns:
        color_vals = window[second_target].to_numpy()
        color_label = second_target
    else:
        color_vals = np.arange(len(window))
        color_label = 'time index'

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z, color='gray', linewidth=0.6, alpha=0.4)
    scatter = ax.scatter(x, y, z, c=color_vals, cmap='viridis', s=10)
    ax.set_xlabel(target)
    ax.set_ylabel(obs[0])
    ax.set_zlabel(obs[1])
    ax.set_title(f'MDE projection: {target} vs. its top-2 observables')
    fig.colorbar(scatter, ax=ax, shrink=0.7, label=color_label)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =============================================================================
# GMN network construction (mirrors notebook cell 7)
# =============================================================================

def _lib_train_test_ranges(n_rows, train_frac=0.7):
    split = int(n_rows * train_frac)
    return f'1 {split}', f'{split + 1} {n_rows}'


def target_E_tau_rho_grid(lib_df, target, e_vals, tau_vals):
    lib_range, pred_range = _lib_train_test_ranges(len(lib_df))
    rho_grid = np.full((len(e_vals), len(tau_vals)), np.nan)
    for i, E in enumerate(e_vals):
        for j, tau in enumerate(tau_vals):
            try:
                S = Simplex(
                    dataFrame = lib_df, columns = target, target = target,
                    E = E, tau = tau, lib = lib_range, pred = pred_range,
                    embedded = False, showPlot = False,
                )
                rho_grid[i, j] = ComputeError(S['Observations'], S['Predictions'])['rho']
            except Exception:
                pass
    return rho_grid


def estimate_node_dimension(lib_df, node, cores, max_e=15, fallback=4):
    lib_range, pred_range = _lib_train_test_ranges(len(lib_df))
    try:
        ed = EmbedDimension(
            dataFrame = lib_df, columns = node, target = node,
            maxE = max_e, lib = lib_range, pred = pred_range, showPlot = False,
            numProcess = cores,
        )
        return int(ed.loc[ed['rho'].idxmax(), 'E'])
    except Exception:
        return fallback


def build_network(targets, mde_results, df, prediction_start, prediction_length,
                   mode, e_range, tau_range, results_dir, cores, interaction_matrix_outer_cores,
                   threshold=0, fallback_node_dimension=4, max_node_dimension=15):
    '''Build one interaction matrix + strongest-driver network spanning
    `targets` and the union of their driver candidate columns
    (mde_results[t]['SELECTED_COLUMNS'] -- MDE-selected when mode ==
    'MDE-GMN', the full observable pool when mode == 'GMN'; this function
    itself has no mode-awareness beyond that one flag used for the
    network_id prefix below).

    Mirrors notebooks/simple-mde-gmn.ipynb cell 7's build_network(). See
    that cell's docstring for the CreateNetwork driver-ranking-bug
    workaround rationale (identical here).

    interaction_matrix_outer_cores is deliberately a *separate* knob from
    `cores`, not just reused: apps.InteractionMatrix's own -cr sizes an
    outer ProcessPoolExecutor across column *pairs*, but its internal
    pyEDM.CCM() call (one per pair) creates its *own* fresh multiprocessing
    pool every time, sized to the full visible cpu_count() by default --
    confirmed by reading pyEDM/CCM.py (`_resolve_workers`) and
    apps/InteractionMatrix.py's InteractFunc (which calls CCM() without
    ever passing `parallel=`, so it can't be overridden via the CLI).
    Neither tool coordinates with the other, so peak concurrent processes
    is -cr * cpu_count(), not -cr -- setting -cr = cores (the seemingly
    natural choice) squares your core request into a process count
    (confirmed empirically: -cr=cores on a 10-core machine produced ~90
    live processes, not ~10). Keeping this outer count low and letting
    `cores` size CCM's own inner pool avoids that multiplication.
    '''
    network_id = '_'.join(targets)
    # Distinguishes MODE='GMN' network artifacts from MODE='MDE-GMN'
    # artifacts for the same targets -- these are genuinely different
    # networks (full raw channel pool vs. MDE-selected columns) and must
    # not share a checkpoint file (see the search function's checkpoint
    # resume logic, which only validates e_vals/tau_vals/theta_vals, not
    # network topology).
    if mode == 'GMN':
        network_id = f'gmn_{network_id}'

    selected_union = []
    for t in targets:
        for c in mde_results[t]['SELECTED_COLUMNS']:
            if c not in selected_union:
                selected_union.append(c)

    data_after_mde = (
        df[targets + selected_union]
        .iloc[:prediction_start + prediction_length]
        .reset_index(drop=True)
    )
    data_after_mde.insert(0, 'Time', np.arange(len(data_after_mde)))
    data_after_mde_file = results_dir / f'data_after_mde_{network_id}.csv'
    data_after_mde.to_csv(data_after_mde_file, index=False)

    data_after_mde_lib = data_after_mde.iloc[:prediction_start]
    data_after_mde_lib_file = results_dir / f'data_after_mde_lib_{network_id}.csv'
    data_after_mde_lib.to_csv(data_after_mde_lib_file, index=False)

    e_vals, tau_vals = list(e_range), list(tau_range)
    target_grids = [target_E_tau_rho_grid(data_after_mde_lib, t, e_vals, tau_vals) for t in targets]
    mean_grid = np.nanmean(np.stack(target_grids), axis=0)
    best_i, best_j = np.unravel_index(np.nanargmax(mean_grid), mean_grid.shape)
    ccm_E, ccm_tau = e_vals[best_i], tau_vals[best_j]
    print(f'{network_id}: CCM E={ccm_E}, tau={ccm_tau} '
          f'(grid-search rho={mean_grid[best_i, best_j]:.3f})', flush=True)

    interaction_prefix = results_dir / f'iMatrix_E{ccm_E}_tau{ccm_tau}_{network_id}'
    # apps.InteractionMatrix is a CLI-only tool (no in-memory API) --
    # invoked via subprocess instead of the notebook's IPython
    # get_ipython().system(), which only exists inside a live kernel.
    # MPLBACKEND=Agg keeps -P from trying to open a GUI window.
    subprocess_env = dict(os.environ, MPLBACKEND='Agg')
    subprocess.run(
        [
            'python', '-m', 'apps.InteractionMatrix',
            '-d', str(data_after_mde_lib_file),
            '-oc', str(interaction_prefix),
            '-ccm', '-E', str(ccm_E), '-t', str(ccm_tau),
            '-l', '1', str(prediction_start), '-p', '1', str(prediction_start),
            '-P', '-cr', str(interaction_matrix_outer_cores),
        ],
        check=True, env=subprocess_env,
    )

    interaction_csv = Path(f'{interaction_prefix}_CCM.csv')
    iMatrix = pd.read_csv(interaction_csv, index_col=0)

    node_dimensions = {
        node: estimate_node_dimension(data_after_mde_lib, node, cores,
                                       max_e=max_node_dimension, fallback=fallback_node_dimension)
        for node in iMatrix.index
    }
    print(f'{network_id}: node N_D =', node_dimensions, flush=True)

    # apps.CreateNetwork() keeps each node's *weakest* above-threshold
    # drivers instead of the strongest (confirmed against the installed
    # apps/CreateNetwork.py:110-116 -- it sorts each row ascending, then
    # truncates to numDrivers). Worked around by pre-selecting the
    # strongest N_D drivers ourselves, then calling CreateNetwork with
    # numDrivers set to the full column count so its buggy truncation
    # never has anything left to remove.
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


def draw_colored_network(net_result, target_colors, save_path):
    graph = net_result['network']['Graph']
    targets = net_result['targets']

    node_colors = [target_colors.get(n, 'lightgray') for n in graph.nodes]
    edge_colors = [target_colors.get(v, 'gray') for _, v in graph.edges]

    pos = nx.arf_layout(graph)
    fig, ax = plt.subplots(figsize=(10, 10))
    nx.draw_networkx(
        graph, pos=pos, ax=ax, with_labels=True, arrowsize=10,
        node_size=300, node_color=node_colors, edge_color=edge_colors,
        width=1.2, font_size=9, font_weight='bold',
    )
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', markersize=12,
                   markerfacecolor=target_colors[t], label=t)
        for t in targets
    ]
    ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(1.0, 1.0))
    ax.set_title(f"GMN network ({net_result['network_id']})")
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)


# =============================================================================
# GMN node generation config (mirrors notebook cell 8)
# =============================================================================

def build_gmn_config(net_result, prediction_start, prediction_length, results_dir,
                      mode='Generate', function='SMap', E=3, tau=-3, theta=3,
                      node_theta_overrides=None, variant=''):
    '''Stage the .cfg file gmn.GMN's configFile= constructor expects.

    function is fixed to 'SMap' for every node in this pipeline -- gmn's
    Simplex branch never calls pyEDM.Embed() before predicting (confirmed
    by reading gmn/Generate.py directly), so it can't do real Takens
    delay embedding: embedded=True forces pyEDM to overwrite E with the
    raw predecessor-column count, silently discarding tau. gmn's SMap
    branch does call Embed(E, tau, columns) first -- a genuine delay
    embedding. Per explicit direction, this pipeline uses SMap uniformly
    for every node rather than mixing generators per node or patching
    Simplex -- see notebooks/simple-mde-gmn.ipynb's build_gmn_config
    docstring for the full evidence trail. Consequence: every node
    (including leaf/undriven ones) needs E >= 2, or SMap's own
    predictor-column-exclusion convention leaves it with zero predictors.

    node_theta_overrides: optional {nodeName: theta} dict -- written as
    per-node .cfg files (gmn/Node.py's nodeConfigPath + f'{name}.cfg'
    lookup); every other node uses the network-wide `theta`. The trailing
    '/' on configPath below works around gmn/Node.py:78's naive string
    concatenation path lookup (confirmed empirically: without it, every
    override is silently ignored).

    Forecast mode's lib/pred offset: gmn/Forecast.py's SMap branch calls
    Embed(E, tau, columns) then dropna() before predicting, which drops
    the leading (E-1)*abs(tau) NaN rows the time-shift introduces --
    shrinking the dataframe SMap actually sees relative to node.data's
    full row count, which is what `prediction_start`/`n_rows` below are
    expressed in. Since every node uses SMap and E >= 2 always (see
    docstring above), this offset is never zero, and an unadjusted
    lib/pred computed in original-row terms overflows the shrunk frame
    (confirmed empirically: pyEDM.SMap's CreateIndices() raises
    "prediction index N exceeds the number of data rows M" with M
    matching n_rows - offset exactly). gmn/Generate.py's own SMap branch
    sidesteps this by building a fresh 0-based time index after its own
    equivalent offset slice; gmn/Forecast.py instead reuses the node's
    original (now index-shifted) time column, so the fix here is to shift
    our own explicit lib/pred bounds back by the same offset instead.'''
    if mode not in ('Generate', 'Forecast'):
        raise ValueError(f"Unknown mode: {mode!r}")

    n_rows = prediction_start + prediction_length
    Tp = 1

    if mode == 'Generate':
        gmn_prediction_start = str(prediction_start)
        gmn_prediction_length = str(prediction_length)
        lib_range = ''
        pred_range = ''
    else:
        gmn_prediction_start = '1'
        gmn_prediction_length = '1'
        offset = (E - 1) * abs(tau)
        if offset >= prediction_start or offset >= prediction_length - Tp:
            raise ValueError(
                f'build_gmn_config(mode=Forecast): embedding offset {offset} '
                f'(E={E}, tau={tau}) leaves too few rows in prediction_start='
                f'{prediction_start}/prediction_length={prediction_length} for '
                f'a valid lib/pred split after SMap\'s Embed()+dropna() shrinks '
                f'the frame -- shrink E/|tau| or grow the prediction window.'
            )
        lib_range = f'1 {prediction_start - offset}'
        pred_range = f'{prediction_start - offset} {n_rows - Tp - offset}'

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
                'E': str(E),
                'Tp': str(Tp),
                'knn': '0',
                'tau': str(tau),
                'theta': str(node_theta),
                'exclusionRadius': '0',
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


# =============================================================================
# E/tau/theta search (mirrors notebook cell 10)
# =============================================================================

def retune_theta_for_candidate(data_lib, graph, nodes, E, tau, theta_range, cores,
                                min_pred_rows=20):
    best_theta_by_node = {}

    for node in nodes:
        columns = list(graph.predecessors(node)) + [node]
        embedded = Embed(dataFrame=data_lib, E=E, tau=tau, columns=columns)
        offset = (E - 1) * abs(tau)
        embedded = embedded.iloc[offset:].reset_index(drop=True)
        embedded.insert(0, 'Time', np.arange(len(embedded)))
        embedded[node] = data_lib[node].to_numpy()[offset:]

        n_emb = len(embedded)
        split = int(n_emb * 0.7)
        n_pred = n_emb - split
        if n_pred < min_pred_rows:
            best_theta_by_node[node] = 3
            continue

        lib_range = f'1 {split}'
        pred_range = f'{split + 1} {n_emb}'

        try:
            pn = PredictNonlinear(
                dataFrame=embedded, columns=list(embedded.columns[1:-1]), target=node,
                theta=list(theta_range), lib=lib_range, pred=pred_range,
                Tp=1, embedded=True, showPlot=False, numProcess=cores,
            )
            best_theta_by_node[node] = float(pn.loc[pn['rho'].idxmax(), 'theta'])
        except Exception:
            best_theta_by_node[node] = 3

    return best_theta_by_node


def evaluate_smap_candidate(net_result, prediction_start, prediction_length,
                             E, tau, node_theta, variant, results_dir, cores):
    config_file = build_gmn_config(
        net_result, prediction_start, prediction_length, results_dir,
        mode='Generate', E=E, tau=tau,
        node_theta_overrides=node_theta,
        variant=variant,
    )
    G = gmn.GMN( configFile = str(config_file), cores = cores )
    G.Generate()

    actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
    generated = G.DataOut.reset_index(drop=True)
    n_gen = len(generated)

    target_rhos = {}
    for target in net_result['targets']:
        actual_vals = actual_df[target].to_numpy()[prediction_start: prediction_start + n_gen]
        gen_vals = generated[target].to_numpy()
        target_rhos[target] = ComputeError(actual_vals, gen_vals)['rho']

    score = float(np.nanmean(list(target_rhos.values())))
    return score, target_rhos, G


def run_smap_search(gmn_runs, prediction_start, prediction_length, e_range, tau_range,
                     theta_range, coarse_theta_range, search_mode, results_dir, cores):
    smap_search_results = {}
    for network_id, run in gmn_runs.items():
        net_result = run['net_result']
        graph = net_result['network']['Graph']
        all_nodes = list(graph.nodes)
        data_after_mde_lib = pd.read_csv(net_result['DATA_AFTER_MDE']).iloc[:prediction_start]

        e_vals, tau_vals = list(e_range), list(tau_range)
        theta_vals = list(theta_range) if search_mode == 'joint' else [None]
        n_candidates = len(e_vals) * len(tau_vals) * len(theta_vals)

        checkpoint_file = results_dir / f'smap_search_checkpoint_{network_id}_{search_mode}.pkl'
        score_grid = np.full((len(e_vals), len(tau_vals), len(theta_vals)), np.nan)
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                ckpt = pickle.load(f)
            if (ckpt.get('e_vals') == e_vals and ckpt.get('tau_vals') == tau_vals
                    and ckpt.get('theta_vals') == theta_vals):
                score_grid = ckpt['score_grid']
                print(f'{network_id}: resumed checkpoint, '
                      f'{int(np.sum(~np.isnan(score_grid)))}/{score_grid.size} candidates already done',
                      flush=True)

        print(f'{network_id}: {search_mode} search over {n_candidates} (E, tau) candidates '
              f'{"x " + str(len(theta_vals)) + " theta " if search_mode == "joint" else ""}'
              f'(mode={search_mode})', flush=True)

        start_t = time.monotonic()
        done_at_start = int(np.sum(~np.isnan(score_grid)))
        done = done_at_start

        for i, E in enumerate(e_vals):
            for j, tau in enumerate(tau_vals):
                if np.all(~np.isnan(score_grid[i, j])):
                    continue

                if search_mode == 'coarse_retune':
                    node_theta_for_candidate = retune_theta_for_candidate(
                        data_after_mde_lib, graph, all_nodes, E, tau, coarse_theta_range, cores,
                    )

                for k, theta in enumerate(theta_vals):
                    if not np.isnan(score_grid[i, j, k]):
                        continue

                    if search_mode == 'joint':
                        node_theta = {n: theta for n in all_nodes}
                    else:
                        node_theta = node_theta_for_candidate

                    variant = (f'smap_search_{search_mode}_E{E}_tau{tau}'
                               + (f'_theta{theta}' if search_mode == 'joint' else ''))
                    score, _, _ = evaluate_smap_candidate(
                        net_result, prediction_start, prediction_length,
                        E, tau, node_theta, variant, results_dir, cores,
                    )
                    score_grid[i, j, k] = score

                    with open(checkpoint_file, 'wb') as f:
                        pickle.dump({'e_vals': e_vals, 'tau_vals': tau_vals,
                                     'theta_vals': theta_vals, 'score_grid': score_grid}, f)

                    done += 1
                    elapsed = time.monotonic() - start_t
                    per_candidate = elapsed / max(1, done - done_at_start)
                    eta = per_candidate * (n_candidates - done)
                    theta_str = f', theta={theta}' if search_mode == 'joint' else ''
                    print(f'[PROGRESS] {network_id} candidate {done}/{n_candidates} '
                          f'(E={E}, tau={tau}{theta_str}) score={score:.4f} '
                          f'(elapsed {elapsed:6.1f}s, ETA {eta:6.1f}s)', flush=True)

        smap_search_results[network_id] = {
            'mode': search_mode,
            'e_vals': e_vals, 'tau_vals': tau_vals, 'theta_vals': theta_vals,
            'score_grid': score_grid,
        }

        flat_grid = score_grid[:, :, 0] if search_mode != 'joint' else np.nanmax(score_grid, axis=2)
        best_i, best_j = np.unravel_index(np.nanargmax(flat_grid), flat_grid.shape)
        best_E, best_tau = e_vals[best_i], tau_vals[best_j]

        if search_mode == 'joint':
            best_k = int(np.nanargmax(score_grid[best_i, best_j]))
            best_theta_global = theta_vals[best_k]
            print(f'{network_id}: best E={best_E}, tau={best_tau}, theta={best_theta_global} '
                  f'(score={score_grid[best_i, best_j, best_k]:.4f})', flush=True)
            smap_search_results[network_id]['best_node_theta'] = {n: best_theta_global for n in all_nodes}
        else:
            best_node_theta = retune_theta_for_candidate(
                data_after_mde_lib, graph, all_nodes, best_E, best_tau, coarse_theta_range, cores,
            )
            print(f'{network_id}: best E={best_E}, tau={best_tau} '
                  f'(theta=coarse-retuned per node, score={flat_grid[best_i, best_j]:.4f})', flush=True)
            smap_search_results[network_id]['best_node_theta'] = best_node_theta

        smap_search_results[network_id]['best_E'] = best_E
        smap_search_results[network_id]['best_tau'] = best_tau

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(flat_grid, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(tau_vals))); ax.set_xticklabels(tau_vals)
        ax.set_yticks(range(len(e_vals))); ax.set_yticklabels(e_vals)
        ax.set_xlabel('tau'); ax.set_ylabel('E')
        ax.set_title(f'GMN E/tau search ({search_mode}) -- {network_id}\nmean Generate() rho across targets')
        fig.colorbar(im, ax=ax, label='mean target rho')
        rect = patches.Rectangle((best_j - 0.5, best_i - 0.5), 1, 1,
                                  fill=False, edgecolor='red', linewidth=2.5)
        ax.add_patch(rect)
        fig.tight_layout()
        fig.savefig(results_dir / f'gmn_E_tau_search_{search_mode}_{network_id}.png', dpi=300)
        plt.close(fig)

    return smap_search_results


# =============================================================================
# Final rebuild + Forecast() diagnostic + fidelity metric
# (mirrors notebook cells 11-13)
# =============================================================================

def rebuild_at_best_params(gmn_runs, smap_search_results, prediction_start, prediction_length,
                            results_dir, cores):
    for network_id, run in gmn_runs.items():
        net_result = run['net_result']
        search = smap_search_results[network_id]
        best_E, best_tau = search['best_E'], search['best_tau']
        node_theta = search['best_node_theta']

        config_file = build_gmn_config(
            net_result, prediction_start, prediction_length, results_dir,
            mode='Generate', E=best_E, tau=best_tau,
            node_theta_overrides=node_theta,
            variant='best',
        )

        G = gmn.GMN( configFile = str(config_file), cores = cores )
        G.Generate()

        run['G'] = G
        run['best_E'] = best_E
        run['best_tau'] = best_tau
        run['best_node_theta'] = node_theta

        actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
        generated = G.DataOut.reset_index(drop=True)
        generate_time = np.arange(prediction_start, prediction_start + len(generated))
        n_rows = prediction_start + prediction_length
        nodes = [c for c in generated.columns if c != 'Time']

        fig, axes = plt.subplots(len(nodes), 1, figsize=(11, 2.2 * len(nodes)), sharex=False)
        if len(nodes) == 1:
            axes = [axes]

        for ax, node in zip(axes, nodes):
            ax.plot(np.arange(n_rows), actual_df[node].to_numpy(),
                    color='tab:blue', lw=1, label='Lib/actual')
            ax.plot(generate_time, generated[node].to_numpy(),
                    color='tab:red', lw=1.2, label=f'Generate() SMap E={best_E}, tau={best_tau}')
            ax.axvline(prediction_start, color='gray', ls='--', lw=0.8)
            ax.set_title(node, fontsize=10)
            ax.legend(fontsize=7, loc='upper right')

        fig.suptitle(f"GMN Generate() (SMap, {search['mode']} search) -- {network_id}", fontsize=11)
        fig.tight_layout()
        fig.savefig(results_dir / f'gmn_generate_{network_id}.png', dpi=300)
        plt.close(fig)

        print(f"{network_id} Generate() rho, held-out window:", flush=True)
        for node in nodes:
            a = actual_df[node].to_numpy()[prediction_start:n_rows]
            r = ComputeError(a, generated[node].to_numpy())['rho']
            tag = 'TARGET' if node in net_result['targets'] else 'driver'
            print(f"  {node:20s} {tag:7s} rho={r:7.3f}", flush=True)


def run_forecast_diagnostic(net_result, G, prediction_start, prediction_length,
                             E, tau, node_theta, results_dir, cores, save_path):
    config_file = build_gmn_config(
        net_result, prediction_start, prediction_length, results_dir, mode='Forecast',
        E=E, tau=tau, node_theta_overrides=node_theta,
    )

    Gf = gmn.GMN( configFile = str(config_file), cores = cores )
    Gf.Forecast()

    actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
    forecast_out = Gf.DataOut.dropna().reset_index(drop=True)
    time_idx = forecast_out['Time'].to_numpy().astype(int)
    nodes = [c for c in forecast_out.columns if c != 'Time']

    generate_out = G.DataOut.reset_index(drop=True)
    generate_time = np.arange(prediction_start, prediction_start + len(generate_out))
    n_rows = prediction_start + prediction_length

    rhos = {}
    fig, axes = plt.subplots(len(nodes), 1, figsize=(11, 2.2 * len(nodes)), sharex=False)
    if len(nodes) == 1:
        axes = [axes]

    for ax, node in zip(axes, nodes):
        actual = actual_df[node].to_numpy()
        forecast_pred = forecast_out[node].to_numpy()
        rhos[node] = ComputeError(actual[time_idx], forecast_pred)['rho']

        ax.plot(np.arange(n_rows), actual, color='tab:blue', lw=1, label='Lib/actual')
        if node in generate_out.columns:
            ax.plot(generate_time, generate_out[node].to_numpy(),
                    color='tab:red', lw=1, label='Generate()')
        ax.plot(time_idx, forecast_pred, color='tab:purple', lw=1.2, ls=':', label='Forecast()')
        ax.axvline(prediction_start, color='gray', ls='--', lw=0.8)
        ax.set_title(f"{node}   Forecast rho={rhos[node]:.3f}", fontsize=10)
        ax.legend(fontsize=7, loc='upper right')

    fig.suptitle(f"Forecast() vs Generate() vs actual -- {net_result['network_id']}", fontsize=11)
    fig.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    return Gf, rhos


def gmn_fidelity_rho(net_result, G, prediction_start, default_E=3, tau=-1):
    actual_df = pd.read_csv(net_result['DATA_AFTER_MDE'])
    generated = G.DataOut.reset_index(drop=True)
    nodes = [c for c in generated.columns if c != 'Time']
    n_gen = len(generated)
    split = n_gen // 2
    node_dimensions = net_result.get('node_dimensions', {})

    fidelity_rho = {}
    for node in nodes:
        gen_vals = generated[node].to_numpy()
        actual_vals = actual_df[node].to_numpy()[prediction_start: prediction_start + n_gen]
        pair_df = pd.DataFrame({
            'Time': np.arange(n_gen),
            f'{node}_gen': gen_vals,
            f'{node}_actual': actual_vals,
        })
        E = max(1, min(node_dimensions.get(node, default_E), split - 1))
        try:
            S = Simplex(
                dataFrame = pair_df, columns = f'{node}_gen', target = f'{node}_actual',
                E = E, tau = tau, Tp = 0,
                lib  = f'1 {split}', pred = f'{split + 1} {n_gen}',
                embedded = False, showPlot = False,
            )
            fidelity_rho[node] = ComputeError(S['Observations'], S['Predictions'])['rho']
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


# =============================================================================
# MDE-only diagnostics (mirrors notebook cells 15, 17, 19 -- skipped when
# MODE == 'GMN', same guard the notebook uses)
# =============================================================================

def run_mde_evaluate(mde_results, targets, results_dir):
    for target in targets:
        r = mde_results[target]
        Eval = dx.Evaluate(
            dataFrame     = r['mde_df'],
            columns_range = [1, r['mde_df'].shape[1]],
            mde_columns   = r['SELECTED_COLUMNS'],
            predictVar    = target,
            library       = r['m'].args.lib,
            prediction    = r['m'].args.pred,
            Tp            = r['m'].args.Tp,
            tau           = r['m'].args.tau,
            components    = 5,
            dmap_k        = 15,
            dmap_epsilon  = 'bgh_generous',
            figsize       = (7, 6),
        )
        Eval.Run()
        Eval.Plot()
        plt.gcf().savefig(results_dir / f'mde_evaluate_{target}.png', dpi=300)
        plt.close('all')


def run_tp_vs_rho_sweep(mde_results, targets, max_tp, target_colors, results_dir):
    fig, ax = plt.subplots(figsize=(8, 5))
    annotation_lines = []
    for target in targets:
        r = mde_results[target]
        cols = r['SELECTED_COLUMNS']
        pi = PredictInterval(
            dataFrame = r['mde_df'],
            columns   = cols,
            target    = target,
            lib       = r['m'].args.lib,
            pred      = r['m'].args.pred,
            E         = len(cols),
            tau       = r['m'].args.tau,
            embedded  = True,
            maxTp     = max_tp,
            showPlot  = False,
        )
        ax.plot(pi['Tp'], pi['rho'], marker='o', label=target, color=target_colors[target])
        annotation_lines.append(f'{target}: E={len(cols)}, vars={", ".join(cols)}')

    ax.axvline(1, color='gray', ls=':', lw=1)
    ax.set_xlabel('Forecast interval Tp (steps)')
    ax.set_ylabel('Prediction skill (rho)')
    ax.set_title('MDE forecast skill vs. horizon')
    ax.legend(loc='best')
    fig.tight_layout()
    fig.text(0.02, -0.02, '\n'.join(annotation_lines), fontsize=7, va='top')
    fig.savefig(results_dir / 'mde_tp_vs_rho_multi_target.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def run_univariate_e_tau_grid(mde_results, targets, e_range, tau_range, rho_threshold, results_dir):
    e_vals, tau_vals = list(e_range), list(tau_range)

    for target in targets:
        r = mde_results[target]
        rho_grid = np.full((len(e_vals), len(tau_vals)), np.nan)

        for i, E in enumerate(e_vals):
            for j, tau in enumerate(tau_vals):
                S = Simplex(
                    dataFrame = r['mde_df'], columns = target, target = target,
                    E = E, tau = tau, lib = r['m'].args.lib, pred = r['m'].args.pred,
                    embedded = False, showPlot = False,
                )
                err = ComputeError(S['Observations'], S['Predictions'])
                rho_grid[i, j] = err['rho']

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(
            rho_grid, aspect='auto', origin='upper', cmap='viridis',
            extent=[-0.5, len(tau_vals) - 0.5, len(e_vals) - 0.5, -0.5],
        )
        ax.set_xticks(range(len(tau_vals))); ax.set_xticklabels(tau_vals)
        ax.set_yticks(range(len(e_vals))); ax.set_yticklabels(e_vals)
        ax.set_xlabel('tau'); ax.set_ylabel('E')
        ax.set_title(f'E, tau grid search: {target}')
        fig.colorbar(im, ax=ax, label='rho')

        for i in range(len(e_vals)):
            for j in range(len(tau_vals)):
                if rho_grid[i, j] > rho_threshold:
                    rect = patches.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                              fill=False, edgecolor='red', linewidth=2.5)
                    ax.add_patch(rect)

        fig.tight_layout()
        fig.savefig(results_dir / f'mde_E_tau_grid_{target}.png', dpi=300)
        plt.close(fig)


# =============================================================================
# Orchestration
# =============================================================================

def target_groups_for_mode(targets, gmn_mode):
    if gmn_mode == 'combined':
        return [list(targets)]
    elif gmn_mode == 'separate':
        return [[t] for t in targets]
    else:
        raise ValueError(f"Unknown GMN_MODE: {gmn_mode!r}")


def main():
    cfg = load_config()
    print('Config:', {k: v for k, v in cfg.items() if k != 'TARGET_COLORS'}, flush=True)

    df = pd.read_csv(cfg['DATA_DIR'] / cfg['DATA_FILE'])
    print('Data shape:', df.shape, flush=True)

    obs_cols = observable_columns(df, cfg['REGION_CHANNEL_CAP'])

    # ---- MDE (or its full-channel-pool stand-in for MODE='GMN') -------
    mde_results = {}
    if cfg['MODE'] == 'MDE-GMN':
        for target in cfg['TARGETS']:
            mde_results[target] = run_mde_for_target(
                target, df, obs_cols, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                cfg['MDE_D'], cfg['EMBED_DIM_RHO_MIN'], cfg['CROSS_MAP_RHO_MIN'],
                cfg['CORES'], cfg['RESULTS_DIR'],
            )

        if cfg['RUN_MDE_DIAGNOSTICS']:
            plot_mde_rho_vs_dimension(
                mde_results, cfg['TARGETS'], cfg['TARGET_COLORS'],
                cfg['RESULTS_DIR'] / 'mde_rho_vs_dimension_multi_target.png',
            )
            plot_3d_mde_projection(
                cfg['TARGETS'][0], cfg['SECOND_TARGET'], mde_results, df,
                cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
                cfg['RESULTS_DIR'] / f"mde_3d_projection_{cfg['TARGETS'][0]}.png",
            )
    else:  # MODE == 'GMN'
        for target in cfg['TARGETS']:
            mde_results[target] = {'target': target, 'SELECTED_COLUMNS': list(obs_cols)}
        print(f"MODE='GMN': skipping MDE, using all {len(obs_cols)} observable columns "
              f"as the candidate driver pool for {cfg['TARGETS']}.", flush=True)

    # ---- GMN network construction + node generation --------------------
    gmn_runs = {}
    for group in target_groups_for_mode(cfg['TARGETS'], cfg['GMN_MODE']):
        net_result = build_network(
            group, mde_results, df, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
            cfg['MODE'], cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['RESULTS_DIR'], cfg['CORES'],
            cfg['INTERACTION_MATRIX_OUTER_CORES'],
        )
        draw_colored_network(
            net_result, cfg['TARGET_COLORS'],
            cfg['RESULTS_DIR'] / f"gmn_network_{net_result['network_id']}.png",
        )
        gmn_runs[net_result['network_id']] = {'group': group, 'net_result': net_result}

    smap_search_results = run_smap_search(
        gmn_runs, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['E_RANGE'], cfg['TAU_RANGE'], cfg['THETA_RANGE'], cfg['COARSE_THETA_RANGE'],
        cfg['SMAP_SEARCH_MODE'], cfg['RESULTS_DIR'], cfg['CORES'],
    )

    rebuild_at_best_params(
        gmn_runs, smap_search_results, cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
        cfg['RESULTS_DIR'], cfg['CORES'],
    )

    for network_id, run in gmn_runs.items():
        Gf, forecast_rho = run_forecast_diagnostic(
            run['net_result'], run['G'], cfg['PREDICTION_START'], cfg['PREDICTION_LENGTH'],
            run['best_E'], run['best_tau'], run['best_node_theta'],
            cfg['RESULTS_DIR'], cfg['CORES'],
            cfg['RESULTS_DIR'] / f'gmn_forecast_vs_generate_{network_id}.png',
        )
        run['Gf'] = Gf
        run['forecast_rho'] = forecast_rho
        print(f"{network_id} Forecast() rho per node (weakest first):", flush=True)
        for node, rho in sorted(forecast_rho.items(), key=lambda kv: kv[1]):
            print(f"  {node:20s} {rho:.4f}", flush=True)

        rho = gmn_fidelity_rho(run['net_result'], run['G'], cfg['PREDICTION_START'])
        plot_fidelity(rho, network_id, cfg['RESULTS_DIR'] / f'gmn_fidelity_{network_id}.png')
        print(f'{network_id} fidelity rho (weakest first):', flush=True)
        for node in sorted(rho, key=lambda n: (np.isnan(rho[n]), -(rho[n] if not np.isnan(rho[n]) else 0))):
            print(f'  {node:20s} {rho[node]:.4f}', flush=True)

        generated = run['G'].DataOut
        generated.to_csv(cfg['RESULTS_DIR'] / f'gmn_generate_output_{network_id}.csv', index=False)

    # ---- MDE-only diagnostics ------------------------------------------
    if cfg['MODE'] == 'MDE-GMN' and cfg['RUN_MDE_DIAGNOSTICS']:
        run_mde_evaluate(mde_results, cfg['TARGETS'], cfg['RESULTS_DIR'])
        run_tp_vs_rho_sweep(mde_results, cfg['TARGETS'], cfg['MAX_TP'], cfg['TARGET_COLORS'], cfg['RESULTS_DIR'])
        run_univariate_e_tau_grid(mde_results, cfg['TARGETS'], cfg['E_RANGE'], cfg['TAU_RANGE'],
                                   cfg['GRID_RHO_THRESHOLD'], cfg['RESULTS_DIR'])
    else:
        print(f"MODE={cfg['MODE']!r} / RUN_MDE_DIAGNOSTICS={cfg['RUN_MDE_DIAGNOSTICS']}: "
              f"skipping MDE-only diagnostics.", flush=True)

    print('Pipeline complete.', flush=True)


if __name__ == "__main__":
    main()
